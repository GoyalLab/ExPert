import os
import sys
import argparse
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

import scanpy as sc
import anndata as ad

import multiprocessing as mp
import threading
import gc


from src.utils.constants import PREDICTION_KEYS
import src.utils.io as io
from src.tune._statics import CONF_KEYS
from src.models._expert import ExPert

from typing import Literal

import logging
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file (train/val split)')
    parser.add_argument('--test', type=str, default=None, help='Path to h5ad test file (test split)')
    parser.add_argument('--config', type=str, required=True, help='config.yaml file that specififes hyperparameters for training')
    parser.add_argument('--fine_tune_config', type=str, default=None, help='config.yaml file that specififes hyperparameters for fine-tune training')
    parser.add_argument('--cls_texts', type=str, default=None, help='Path to class texts dict as json')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory, default is config file directory')
    parser.add_argument('--test_unseen', action='store_true', help='Test model on unseen perturbations')
    parser.add_argument('--no_save', action='store_true', help='Save model output data')
    return parser.parse_args()


def clean_up():
    print("Cleaning up dataloaders, worker processes, CUDA contexts...")

    # --- Terminate PyTorch DataLoader worker processes ---
    try:
        children = mp.active_children()
        for p in children:
            print(f"  -> terminating child process {p.pid}")
            p.terminate()
            p.join(timeout=2)
    except Exception as e:
        print(f"Could not inspect or terminate multiprocessing children: {e}")

    # --- Clear CUDA contexts ---
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("  -> cleared CUDA context")
    except Exception as e:
        print(f"Could not clear CUDA: {e}")

    # --- Inspect daemon threads (cannot force-kill, but useful for debugging) ---
    try:
        for t in threading.enumerate():
            if t.daemon and t.is_alive():
                print(f"  -> daemon thread still alive: {t.name}")
    except Exception as e:
        print(f"Thread inspection error: {e}")

    # --- Force garbage collection (closes mmap and file handles) ---
    gc.collect()

    print("Cleanup finished.")

    
def get_ouptut_dir(config_p: str, output_base_dir: str | None = None) -> str:
    # Create directory based on the input config name
    if output_base_dir is None:
        output_dir = os.path.dirname(config_p)
    else:
        output_dir = os.path.join(output_base_dir, os.path.basename(config_p).replace('.yaml', ''))
    log.info(f'Run output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def _train(
        adata_p: str, 
        step_model_dir: str, 
        config: dict, 
        cls_label: str = 'cls_label',
        batch_key: str = 'context',
        verbose: bool = True,
        context_filter: list[str] | None = None,
        cls_texts_p: str | None = None,
        **train_kwargs
    ) -> ExPert:
    """Train wrapper for ExPert.train()"""
    log.info(f'Reading training data from: {adata_p}')
    model_set = sc.read(adata_p)
    # Check if dataset is compatible
    assert cls_label in model_set.obs.columns and batch_key in model_set.obs.columns
    
    # Subset model to only the include specified datasets
    if context_filter is not None:
        log.info(f'Subsetting to context keys: {context_filter}')
        model_set._inplace_subset_obs(model_set.obs[batch_key].isin(context_filter))
    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Setup model
    log.info('Setting up model.')
    # Setup anndata with model
    setup_kwargs = {'batch_key': batch_key, 'labels_key': cls_label}
    setup_kwargs.update(config.get(CONF_KEYS.MODEL_SETUP, {}))
    # Load class texts if path is given
    cls_texts = io.load_json(cls_texts_p)
    ExPert.setup_anndata(
        model_set,
        **setup_kwargs
    )
    # Initialize main model
    model = ExPert(model_set, cls_text_dict=cls_texts, **config[CONF_KEYS.MODEL].copy())
    if verbose:
        print(model.module)
    # Set training logger
    config[CONF_KEYS.TRAIN]['logger'] = pl.loggers.TensorBoardLogger(step_model_dir)
    # Train the model
    log.info(f'Running at: {step_model_dir}')
    model.train(config, **train_kwargs)
    return model

def train(adata_p: str, config_p: str, out_dir: str, **kwargs) -> dict[str: nn.Module | pd.DataFrame | ad.AnnData | str]:
    """Train wrapper for use with config file."""
    # Load run config
    config = io.read_config(config_p, do_setup=False, check_schema=False)
    # Init run output dir
    step_model_dir = get_ouptut_dir(config_p, output_base_dir=out_dir)
    # Train the model
    model: ExPert = _train(
        adata_p=adata_p, 
        step_model_dir=step_model_dir, 
        config=config,
        **kwargs
    )
    return model

def full_run(
        config_p: str,
        train_p: str,
        model_dir: str,
        test_p: str,
        test_unseen: bool = True,
        load_checkpoint: bool = False,
        cls_label: str = 'perturbation',
        batch_label: str = 'context',
        ctrl_key: str | None = 'control',
        results_mode: Literal['return', 'save'] | None | list[str] = 'save',
        save_anndata: bool = False,
        fine_tune_config_p: str | None = None,
        **kwargs
    ) -> pd.DataFrame:
    """Function to perform a full model training, evaluation, and testing process."""
    # Get config name
    config_name = os.path.basename(config_p).replace('.yaml', '')
    # Set default model output to config directory
    model_dir = model_dir if model_dir is not None else os.path.dirname(config_p)
    # Train model with loaded config file
    base_model = train(
        adata_p=train_p, 
        config_p=config_p, 
        out_dir=model_dir,
        **kwargs
    )
    # Evaluate model
    base_model.evaluate(results_mode=results_mode, save_anndata=save_anndata)
    # Save output dir
    output_dir = base_model.model_log_dir
    # Try to load best checkpoint, otherwise stick to final model
    if load_checkpoint:
        base_model = ExPert.load_checkpoint(
            output_dir,
            adata=base_model.adata
        )
    # Do not test model if no test path is provided
    if test_p is None or not os.path.exists(test_p):
        log.info(f'Skipping model test since no valid test path is provided.')
        return None
    # Run fine-tune stage if extra config is provided
    if fine_tune_config_p:
        # Load fine-tune config params
        fine_tune_config = io.read_config(fine_tune_config_p)
        # Add logger to config
        fine_tune_output_dir = os.path.join(base_model.model_log_dir, 'fine-tune')
        fine_tune_config[CONF_KEYS.TRAIN]['logger'] = pl.loggers.TensorBoardLogger(fine_tune_output_dir)
        # Get model params
        model_kwargs = fine_tune_config[CONF_KEYS.MODEL]
        # Create new model from pre-trained with frozen base encoder and decoder
        model = ExPert.from_base_model(
            base_model, 
            freeze_modules=['z_encoder', 'decoder'], 
            check_model_kwargs=False, 
            **model_kwargs
        )
        # Train fine-tune model
        model.train(fine_tune_config, cache_data_splitter=False)
        # Evaluate fine-tuned model
        model.evaluate(results_mode=results_mode, save_anndata=save_anndata)
    else:
        # Treat single trained model as full model
        model = base_model
    # Create test output directory
    test_out = os.path.join(output_dir, 'test')
    os.makedirs(test_out, exist_ok=True)
    # Test model with specified test set
    test_output = model.test(
        test_adata_p=test_p,
        output_dir=test_out,
        incl_unseen=False,
        plot=True,
        results_mode=results_mode,
        cls_label=cls_label,
        batch_label=batch_label,
        ctrl_key=ctrl_key,
    )
    # Unpack output
    top_n_predictions = test_output.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY]
    top_n_predictions['incl_unseen'] = False
    # Test model with unseen perturbations if option is given
    if test_unseen:
        top_n_predictions_unseen = model.test(
            test_adata_p=test_p,
            output_dir=test_out,
            incl_unseen=True,
            plot=True,
            results_mode=results_mode,
            cls_label=cls_label,
            batch_label=batch_label,
            ctrl_key=ctrl_key,
        ).uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY]
        top_n_predictions_unseen['incl_unseen'] = True
        # Concat both predictions
        top_n_predictions = pd.concat((top_n_predictions, top_n_predictions_unseen), axis=0)
    # Add config name to predictions
    top_n_predictions['config_name'] = config_name
    # Save overall testing performance to file
    top_n_predictions.to_csv(os.path.join(output_dir, 'test_top_n_predictions.csv'))
    return top_n_predictions

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        # Train model
        _ = full_run(
            config_p=args.config,
            train_p=args.input,
            model_dir=args.outdir,
            test_p=args.test,
            test_unseen=args.test_unseen,
            fine_tune_config_p=args.fine_tune_config,
            cls_texts_p=args.cls_texts,
            save_anndata=not args.no_save
        )

    except Exception as e:
        print("Error in full_run:", e)
        raise e

    finally:
        # Close all dataloaders, etc.
        clean_up()

        print("Exiting job.")
        sys.stdout.flush()
        sys.stderr.flush()

        # Force termination so Slurm marks the job as finished
        os._exit(0)


if __name__ == "__main__":
    main()