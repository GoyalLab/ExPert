# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import os
import sys
import signal
import argparse
import pandas as pd

import torch
import torch.nn as nn

import scanpy as sc
import anndata as ad

import multiprocessing as mp
import threading
import gc


import expert.utils.io as io
import expert.utils.legacy as legacy
from expert.tune._statics import CONF_KEYS
from expert.models._expert import ExPert

from typing import Literal

import logging
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file (train/val split)')
    parser.add_argument('--config', type=str, required=True, help='config.yaml file that specififes hyperparameters for training')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory, default is config file directory')
    parser.add_argument('--resume', action='store_true', help='Resume model training from checkpoints.')
    parser.add_argument('--no_save', action='store_true', help='Save model output data')
    parser.add_argument('--eval', dest='run_eval', action='store_true',
                        help='Run the post-training evaluate() pass (minify + full predict). OFF by default: it is the step that OOMs, and the per-epoch val/ood metrics it duplicates are already written to tfevents during training. Predict from checkpoints instead.')
    parser.add_argument('--fold_dir', type=str, default=None, help='Path to a pre-computed fold directory (e.g. benchmark/splits/fold_0) to load train/val/test/ood_test indices from.')
    parser.add_argument('--unseen_pert', type=str, default=None, help='Optional path to a pre-computed merged h5ad of UNSEEN-perturbation cells for zero-shot eval (id_ood_pert / ood_ood_pert splits). Overrides cluster.unseen_pert_adata_path in the config.')
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
        model_dir: str, 
        config: dict, 
        verbose: bool = True,
        blacklist: list[str] | None = None,
        resume: bool = False,
        fold_dir: str | None = None,
        dry: bool = False,
        **train_kwargs
    ) -> ExPert:
    """Train wrapper for ExPert.train()"""
    # Load training dataset
    logging.info(f'Reading training data from: {adata_p}')
    model_set = sc.read(adata_p)

    # Subset model to only the include specified datasets
    if blacklist is not None and len(blacklist) > 0:
        bk = config['model_setup']['batch_key']
        logging.info(f"Removing contexts matching keys: {blacklist}")
        pattern = "|".join(blacklist)
        mask = ~model_set.obs[bk].astype(str).str.contains(pattern, regex=True)
        model_set._inplace_subset_obs(mask)
        sc.pp.filter_genes(model_set, min_counts=1)
    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Setup model
    log.info('Setting up model.')
    # Setup anndata with model
    setup_kwargs = dict(config.get(CONF_KEYS.MODEL_SETUP, {}))
    # Setup adata for model
    ExPert.setup_anndata(
        model_set,
        **setup_kwargs
    )

    # Check if model training can be resumed (from same version)
    model = None
    if resume and io.get_latest_ckpt_dir(model_dir) is not None:
        # Load the latest model checkpoint from latest model version in run directory
        model = ExPert.load(model_dir, adata=model_set, use_latest_checkpoint=True)
    # Create a new model instance (creates new run version) if not resumed
    if model is None:
        # Initialize model instance
        model = ExPert(
            model_set,
            model_log_dir=model_dir,
            **config[CONF_KEYS.MODEL].copy()
        )
    # Overwrite fold dir in data config if specified here
    if fold_dir is not None:
        config[CONF_KEYS.DATA]['fold_dir'] = fold_dir
    # Print module summary
    if verbose:
        logging.info(model.module)
    # Train the model
    if not dry:
        log.info(f'Running at: {model_dir}')
        model.train(config, **train_kwargs)
    return model

def train(adata_p: str, config_p: str, out_dir: str, **kwargs) -> dict[str: nn.Module | pd.DataFrame | ad.AnnData | str]:
    """Train wrapper for use with config file."""
    # Load run config
    config = io.read_config(config_p, do_setup=False, check_schema=False)
    # Pre-release configs: drop knobs of removed mechanisms (only at their off value, else raise)
    config = legacy.translate_config(config)
    # Init run output dir
    model_dir = get_ouptut_dir(config_p, output_base_dir=out_dir)
    # Train the model
    model: ExPert = _train(
        adata_p=adata_p, 
        model_dir=model_dir, 
        config=config,
        **kwargs
    )
    return model

def full_run(
        config_p: str,
        train_p: str,
        model_dir: str,
        results_mode: Literal['return', 'save'] | None | list[str] = 'save',
        save_anndata: bool = True,
        resume: bool = True,
        unseen_pert_p: str | None = None,
        run_eval: bool = False,
        **kwargs
    ) -> None:
    """Function to perform a full model training and (opt-in) evaluation process."""
    # Get config name
    config_name = os.path.basename(config_p).replace('.yaml', '')
    # Set default model output to config directory
    model_dir = model_dir if model_dir is not None else os.path.dirname(config_p)
    # Train model with loaded config file
    base_model = train(
        adata_p=train_p, 
        config_p=config_p, 
        out_dir=model_dir,
        resume=resume,
        **kwargs
    )
    # Evaluate model -- OPT-IN (--eval). This is the minify + full-predict pass that OOMs on the
    # small arms (~67GB); every per-epoch val/ood_test metric it would report is already logged to
    # tfevents during training, and checkpoints are saved, so the default skip costs nothing.
    if run_eval:
        base_model.evaluate(results_mode=results_mode, save_anndata=save_anndata)
    else:
        log.info('[Eval] Skipping post-training evaluate() (pass --eval to enable).')
    # Optional zero-shot eval on a pre-computed UNSEEN-perturbation adata
    # (id_ood_pert / ood_ood_pert splits). Path from the CLI override or the
    # config's cluster.unseen_pert_adata_path; skipped silently when unset.
    _unseen_p = unseen_pert_p
    if _unseen_p is None:
        try:
            _cfg = io.read_config(config_p, do_setup=False, check_schema=False)
            _unseen_p = (_cfg.get('cluster', {}) or {}).get('unseen_pert_adata_path')
        except Exception as e:
            log.warning(f'[ZeroShot] Could not read unseen_pert_adata_path from config: {e}')
            _unseen_p = None
    if run_eval and _unseen_p and os.path.exists(str(_unseen_p)):
        log.info(f'[ZeroShot] Evaluating unseen perturbations from: {_unseen_p}')
        try:
            base_model.evaluate_zero_shot_perturbations(
                _unseen_p,
                save_path=os.path.join(base_model.model_log_dir, f'{config_name}_zero_shot_perturbations.csv'),
            )
        except Exception as e:
            import traceback
            log.error(f'[ZeroShot] Zero-shot perturbation eval failed: {e}\n{traceback.format_exc()}')
    elif run_eval and _unseen_p:
        log.warning(f'[ZeroShot] unseen_pert path set but not found: {_unseen_p}; skipping.')
    return None

def _handle_sigterm(signum, frame):
    """Handle SIGTERM from SLURM (scancel / timeout) by cleaning up and exiting."""
    log.warning(f"Received signal {signum}, cleaning up...")
    clean_up()
    log.info("Cleanup complete, exiting.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def main():
    # Handle SLURM cancel signals correctly
    signal.signal(signal.SIGTERM, _handle_sigterm)
    EXIT_CODE = 0
    # Train model
    try:
        # Parse command line arguments
        args = parse_args()
        for _flag in ('test', 'fine_tune_config', 'test_unseen'):
            if getattr(args, _flag):
                log.warning(f'--{_flag} is no longer supported and is ignored.')
        # Train model
        _ = full_run(
            config_p=args.config,
            train_p=args.input,
            model_dir=args.outdir,
            resume=args.resume,
            save_anndata=not args.no_save,
            fold_dir=args.fold_dir,
            unseen_pert_p=args.unseen_pert,
            run_eval=args.run_eval,
        )

    except Exception as e:
        import traceback
        log.error(f"Error in full_run: {e}")
        log.error("Full traceback:\n" + traceback.format_exc())
        EXIT_CODE = 1
        raise e
    # Clean up data loaders etc.
    finally:
        # Close all dataloaders, etc.
        clean_up()

        log.info("Exiting job.")
        sys.stdout.flush()
        sys.stderr.flush()

        # Force termination so Slurm marks the job as finished
        os._exit(EXIT_CODE)


if __name__ == "__main__":
    main()