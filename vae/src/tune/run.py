import os
import argparse
import logging
import yaml
import pandas as pd
import itertools, random
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

import scanpy as sc

from ._statics import CONF_KEYS
from ..models._jedvi import JEDVI
from ..plotting import get_model_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file')
    parser.add_argument('--config', type=str, required=True, help='config.yaml file that specififes hyperparameters for training')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory, default is config file directory')
    return parser.parse_args()

def generate_random_configs(space: dict, n_samples: int | None = None, fixed_weights: bool = True, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    # Helper to create valid schedule combos
    def valid_schedule_combos(subspace):
        keys = list(subspace.keys())
        vals = [subspace[k] for k in keys]
        for combo in itertools.product(*vals):
            combo_dict = dict(zip(keys, combo))
            # Check schedule weights
            if 'min_weight' in combo_dict and 'max_weight' in combo_dict:
                # Enforce constant weights
                if fixed_weights and combo_dict['max_weight'] != combo_dict['min_weight']:
                    continue
                # Filter for ascending schedules only
                if combo_dict['max_weight'] < combo_dict['min_weight']:
                    continue
            yield combo_dict

    # Build precomputed lists for schedules (valid only)
    schedule_spaces = {name: list(valid_schedule_combos(params)) for name, params in space['schedules'].items()}

    # Flatten other subspaces
    data_items = list(itertools.product(*space['data'].values()))
    encoder_items = list(itertools.product(*space['encoder'].values()))
    decoder_items = list(itertools.product(*space['decoder'].values()))
    plan_items = list(itertools.product(*space['plan'].values()))
    model_items = list(itertools.product(*space['model'].values()))
    schedules_items = list(itertools.product(*schedule_spaces.values()))

    # Keys for nesting
    data_keys = list(space['data'].keys())
    encoder_keys = list(space['encoder'].keys())
    decoder_keys = list(space['decoder'].keys())
    plan_keys = list(space['plan'].keys())
    model_keys = list(space['model'].keys())
    schedule_keys = list(schedule_spaces.keys())

    total_combos = (len(data_items) * len(encoder_items) * len(decoder_items) *
                    len(plan_items) * len(model_items) * len(schedules_items))
    logging.info(f"Total possible configs: {total_combos}")

    def random_config():
        return {
            'data': dict(zip(data_keys, random.choice(data_items))),
            'encoder': dict(zip(encoder_keys, random.choice(encoder_items))),
            'decoder': dict(zip(decoder_keys, random.choice(decoder_items))),
            'plan': dict(zip(plan_keys, random.choice(plan_items))),
            'model': dict(zip(model_keys, random.choice(model_items))),
            'schedules': {
                schedule_keys[i]: random.choice(schedule_spaces[schedule_keys[i]])
                for i in range(len(schedule_keys))
            }
        }

    # Generator: yield n_samples (or infinite if n_samples=None)
    count = 0
    while n_samples is None or count < n_samples:
        yield random_config()
        count += 1

def recursive_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def sample_configs(space: dict, src_config: dict, N: int = 10, base_dir: str | None = None):
    # Sample configs
    space_configs, config_paths = [], []
    for i, sample in enumerate(generate_random_configs(space, n_samples=N, seed=42)):
        # Merge sample into a deep copy of src_config
        merged_config = recursive_update(deepcopy(src_config), sample)
        space_configs.append(merged_config)

        if base_dir is None:
            continue
        
        # Create run directory & save merged config
        run_dir = os.path.join(base_dir, f'run_{str(i)}')
        os.makedirs(run_dir, exist_ok=True)
        conf_out = os.path.join(run_dir, 'config.yaml')
        config_paths.append(conf_out)
        logging.info(f'Saving run config to {conf_out}')
        with open(conf_out, 'w') as f:
            yaml.dump(merged_config, f, sort_keys=False)

    # Return configs or configs + saved paths
    if base_dir is None:
        return space_configs
    else:
        return space_configs, pd.DataFrame({'config_path': config_paths})

# Recursive function to replace "nn.*" strings with actual torch.nn classes
def replace_nn_modules(d):
    if isinstance(d, dict):
        return {k: replace_nn_modules(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nn_modules(v) for v in d]
    elif isinstance(d, str) and d.startswith("nn."):
        attr = d.split("nn.")[-1]
        return getattr(nn, attr)
    elif isinstance(d, str) and hasattr(nn, d):
        return getattr(nn, d)
    else:
        return d

def read_config(config_p: str) -> dict:
    """Read hyperparameter yaml file"""
    logging.info(f'Loading config file: {config_p}')
    with open(config_p, 'r') as f:
        config: dict = yaml.safe_load(f)
    # Check for config keys
    expected_keys = set(CONF_KEYS._asdict().values())
    assert expected_keys.issubset(config.keys()), f"Missing keys: {expected_keys - set(config.keys())}"
    # Convert nn modules to actual classes
    config = replace_nn_modules(config)
    return config

def run(
        adata_p: str, 
        step_model_dir: str, 
        config: dict, 
        cls_label: str = 'cls_label',
        batch_key: str = 'dataset',
        verbose: bool = True,
    ) -> None:
    logging.info(f'Reading training data from: {adata_p}')
    model_set = sc.read(adata_p)
    # Check if dataset is compatible
    assert cls_label in model_set.obs.columns and batch_key in model_set.obs.columns
    
    # Define all labels to classify on
    if 'perturbation_direction' in model_set.obs.columns:
        logging.info('Using perturbation direction to classify')
        cls_labels = ['perturbation_direction', 'perturbation']
    else:
        cls_labels = ['celltype', 'perturbation_type', 'perturbation']
    if verbose:
        # Check number of unique perturbations to classify
        logging.info(f'Initializing dataset with {model_set.obs.cls_label.nunique()} classes')
        logging.info(f'{model_set.obs[cls_labels[1:]].drop_duplicates().shape[0]} unique perturbations')
        logging.info(f'{model_set.obs[cls_labels[-1]].nunique()} unique gene-perturbations')
        logging.info(f'{model_set.obs["celltype"].nunique()} unique cell types')
        logging.info(f'{model_set.obs["dataset"].nunique()} datasets')
        logging.info(f'Mean number of cells / perturbation {model_set.obs.cls_label.value_counts().mean()}')
        logging.info(f'Class embedding shape: {model_set.uns["cls_embedding"].shape}')
        logging.info(f'Adata shape: {model_set.shape}')

    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Add schedule params to plan
    config[CONF_KEYS.PLAN]['anneal_schedules'] = config[CONF_KEYS.SCHEDULES]
    # Add plan to train
    config[CONF_KEYS.TRAIN]['plan_kwargs'] = config[CONF_KEYS.PLAN]
    # Add encoder and decoder args to model
    config[CONF_KEYS.MODEL]['extra_encoder_kwargs'] = config[CONF_KEYS.ENCODER]
    config[CONF_KEYS.MODEL]['extra_decoder_kwargs'] = config[CONF_KEYS.DECODER]
    # Add classifier args to model
    config[CONF_KEYS.MODEL]['classifier_parameters'] = config[CONF_KEYS.CLS]

    # Setup model
    logging.info('Setting up model.')
    # Setup anndata with model
    JEDVI.setup_anndata(
        model_set,
        batch_key=batch_key,
        labels_key=cls_label
    )
    jedvae = JEDVI(model_set, **config[CONF_KEYS.MODEL].copy())
    logging.info(jedvae)
    # Set training logger
    config[CONF_KEYS.TRAIN]['logger'] = pl.loggers.TensorBoardLogger(step_model_dir)
    # Train the model
    logging.info(f'Running at: {step_model_dir}')
    jedvae.train(
        data_params=config[CONF_KEYS.DATA].copy(), 
        model_params=config[CONF_KEYS.MODEL].copy(), 
        train_params=config[CONF_KEYS.TRAIN].copy(), 
        return_runner=False
    )
    # Save results to lightning directory
    get_model_results(
        model=jedvae, 
        cls_labels=cls_labels, 
        log_dir=step_model_dir, 
        plot=True, 
        max_classes=0
    )

def get_ouptut_dir(config_p: str, output_base_dir: str | None = None) -> str:
    # Create directory based on the input config name
    output_dir = os.path.dirname(config_p) if output_base_dir is None else output_base_dir
    logging.info(f'Run output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == '__main__':
    # Parse cmd line args
    args = parse_args()
    # Load config file
    config = read_config(args.config)
    step_model_dir = get_ouptut_dir(args.config, output_base_dir=args.outdir)
    # Train the model
    run(
        adata_p=args.input, 
        step_model_dir=step_model_dir, 
        config=config
    )
