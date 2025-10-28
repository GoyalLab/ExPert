import os
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

import scanpy as sc
import anndata as ad


from src.utils.constants import PREDICTION_KEYS
from src.utils.io import read_config
from src.tune._statics import CONF_KEYS
from src.models._jedvi import JEDVI


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file (train/val split)')
    parser.add_argument('--test', type=str, default=None, help='Path to h5ad test file (test split)')
    parser.add_argument('--config', type=str, required=True, help='config.yaml file that specififes hyperparameters for training')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory, default is config file directory')
    parser.add_argument('--test_unseen', action='store_true', help='Test model on unseen perturbations')
    return parser.parse_args()

    
def get_ouptut_dir(config_p: str, output_base_dir: str | None = None) -> str:
    # Create directory based on the input config name
    if output_base_dir is None:
        output_dir = os.path.dirname(config_p)
    else:
        output_dir = os.path.join(output_base_dir, os.path.basename(config_p).replace('.yaml', ''))
    logging.info(f'Run output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def _train(
        adata_p: str, 
        step_model_dir: str, 
        config: dict, 
        cls_label: str = 'cls_label',
        batch_key: str = 'dataset',
        verbose: bool = False,
        **train_kwargs
    ) -> JEDVI:
    """Train wrapper for JEDVI.train()"""
    logging.info(f'Reading training data from: {adata_p}')
    model_set = sc.read(adata_p)
    # Check if dataset is compatible
    assert cls_label in model_set.obs.columns and batch_key in model_set.obs.columns
  
    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Setup model
    logging.info('Setting up model.')
    # Setup anndata with model
    setup_kwargs = {'batch_key': batch_key, 'labels_key': cls_label}
    setup_kwargs.update(config.get(CONF_KEYS.MODEL_SETUP, {}))
    JEDVI.setup_anndata(
        model_set,
        **setup_kwargs
    )
    model = JEDVI(model_set, **config[CONF_KEYS.MODEL].copy())
    if verbose:
        print(model.module)
    # Set training logger
    config[CONF_KEYS.TRAIN]['logger'] = pl.loggers.TensorBoardLogger(step_model_dir)
    # Train the model
    logging.info(f'Running at: {step_model_dir}')
    model.train(config, **train_kwargs)
    return model

def train(adata_p: str, config_p: str, out_dir: str, **kwargs) -> dict[str: nn.Module | pd.DataFrame | ad.AnnData | str]:
    """Train wrapper for use with config file."""
    # Load run config
    config = read_config(config_p)
    # Init run output dir
    step_model_dir = get_ouptut_dir(config_p, output_base_dir=out_dir)
    # Train the model
    model: JEDVI = _train(
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
        test_unseen: bool = False,
        load_checkpoint: bool = True,
        cls_label: str = 'cls_label',
        batch_label: str = 'dataset',
        ctrl_key: str | None = 'neg;control',
        **kwargs
    ) -> pd.DataFrame:
    """Function to perform a full model training, evaluation, and testing process."""
    # Get config name
    config_name = os.path.basename(config_p).replace('.yaml', '')
    # Set default model output to config directory
    model_dir = model_dir if model_dir is not None else os.path.dirname(config_p)
    # Train model with loaded config file
    model = train(adata_p=train_p, config_p=config_p, out_dir=model_dir, **kwargs)
    # Evaluate model
    model.evaluate()
    # Save output dir
    output_dir = model.model_log_dir
    # Try to load best checkpoint, otherwise stick to final model
    if load_checkpoint:
        model = JEDVI.load_checkpoint(
            output_dir,
            adata=model.adata
        )
    # Do not test model if no test path is provided
    if test_p is None or not os.path.exists(test_p):
        logging.info(f'Skipping model test since no valid test path is provided.')
        return None
    # Create test output directory
    test_out = os.path.join(output_dir, 'test')
    os.makedirs(test_out, exist_ok=True)
    # Test model with specified test set
    top_n_predictions = model.test(
        test_adata_p=test_p,
        output_dir=test_out,
        incl_unseen=False,
        plot=True,
        results_mode='return',
        cls_label=cls_label,
        batch_label=batch_label,
        ctrl_key=ctrl_key,
    ).uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY]
    top_n_predictions['incl_unseen'] = False
    # Test model with unseen perturbations if option is given
    if test_unseen:
        top_n_predictions_unseen = model.test(
            test_adata_p=test_p,
            output_dir=test_out,
            incl_unseen=True,
            plot=True,
            results_mode='return',
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

if __name__ == '__main__':
    # Parse cmd line args
    args = parse_args()
    # Full model run
    _ = full_run(config_p=args.config, train_p=args.input, model_dir=args.outdir, test_p=args.test, test_unseen=args.test_unseen)
