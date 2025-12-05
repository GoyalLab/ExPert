import os
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from os import PathLike
import random
from copy import deepcopy

import torch
import torch.nn as nn

from src.tune._statics import CONF_KEYS, NESTED_CONF_KEYS

import logging
log = logging.getLogger(__name__)


mutually_excl_keys: dict[str, list[tuple[str]]] = {
    CONF_KEYS.MODEL: [('use_batch_norm', 'use_layer_norm')]
}

def load_json(p: str, mode: str = 'r', **kwargs):
    import json
    if p is None or not isinstance(p, str) or not p.endswith('json'):
        return None
    with open(p, mode) as f:
        return json.load(f, **kwargs)


def to_tensor(m: sp.csr_matrix | torch.Tensor | np.ndarray | pd.DataFrame | None) -> torch.Tensor:
    if m is None:
        return None
    if sp.issparse(m):
        return torch.Tensor(m.toarray())
    if isinstance(m, np.ndarray):
        return torch.Tensor(m)
    if isinstance(m, pd.DataFrame):
        return torch.Tensor(m.values)
    if isinstance(m, torch.Tensor):
        return m
    raise ValueError(f'{m.__class__} is not compatible. Should be either sp.csr_matrix, np.ndarray, or torch.Tensor.')

def tensor_to_df(x: torch.Tensor, index: list, columns: list | None = None, prefix: str = 'dim_') -> pd.DataFrame:
    """Convert tensor to pandas dataframe"""
    if columns is None:
        columns = prefix + pd.Series(np.arange(x.size(1)), dtype=str)
    return pd.DataFrame(x.cpu().numpy(), index=index, columns=columns)

# Recursive function to replace "nn.*" strings with actual torch.nn classes
def replace_nn_modules(d):
    if isinstance(d, dict):
        return {k: replace_nn_modules(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nn_modules(v) for v in d]
    elif isinstance(d, str) and d.startswith("nn."):
        attr = d.split("nn.")[-1]
        return getattr(nn, attr)
    else:
        return d
    
def non_zero(x: torch.Tensor | float | int | None) -> bool:
    if x is None:
        return False
    if x > 0:
        return True
    return False
    
def generate_random_configs(space: dict, n_samples: int | None = None, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    def valid_config(config: dict):
        # Check if there are mutually exclusive parameter values
        for sched_key, mu_kw_list in mutually_excl_keys.items():
            # Get schedule
            sched: dict = config.get(sched_key)
            if sched is None:
                return True
            # Check all mututally exclusive params in schedule
            for mu_tuple in mu_kw_list:
                # Collect all unique params
                vs = set()
                for kw in mu_tuple:
                    vs.add(sched.get(kw))
                # Validate config if they are all None (don't exist in config)
                if len(vs) == 1 and list(vs)[0] is None:
                    return True
                # Reject config if there is less values than number of keywords
                if len(vs) < len(mu_tuple):
                    return False
        return True

    def sample_from_space(subspace):
        if isinstance(subspace, dict):
            return {k: sample_from_space(v) for k, v in subspace.items()}
        if isinstance(subspace, list):
            return random.choice(subspace)
        return subspace

    def random_config(max_iter: int = 100):
        i = 0
        while True and i < max_iter:
            # Keep sampling configs if they are invalid
            config = {k: sample_from_space(v) for k, v in space.items()}
            if valid_config(config):
                break
            i += 1
        return config

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

def sample_configs(space: dict, src_config: dict, N: int = 10, base_dir: str | None = None, verbose: bool = False, **kwargs):
    # Sample configs
    space_configs, config_paths = [], []
    for i, sample in enumerate(generate_random_configs(space, n_samples=N, seed=42, **kwargs)):
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
        if verbose:
            log.info(f'Saving run config to {conf_out}')
        with open(conf_out, 'w') as f:
            yaml.dump(merged_config, f, sort_keys=False)

    # Return configs or configs + saved paths
    if base_dir is None:
        return space_configs
    else:
        return space_configs, pd.DataFrame({'config_path': config_paths})
    
def setup_config(config: dict) -> None:
    # Add schedule params to plan
    config[CONF_KEYS.PLAN][NESTED_CONF_KEYS.SCHEDULES_KEY] = config[CONF_KEYS.SCHEDULES]
    # Add plan to train
    config[CONF_KEYS.TRAIN][NESTED_CONF_KEYS.PLAN_KEY] = config[CONF_KEYS.PLAN]
    # Add encoder and decoder args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.ENCODER_KEY] = config[CONF_KEYS.ENCODER]
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.DECODER_KEY] = config[CONF_KEYS.DECODER]
    # Add classifier args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.CLS_KEY] = config[CONF_KEYS.CLS]
    # Add aligner args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.ALIGN_KEY] = config[CONF_KEYS.ALIGNER]

def read_config(config_p: str, do_setup: bool = False, check_schema: bool = False) -> dict:
    """Read hyperparameter yaml file"""
    log.info(f'Loading config file: {config_p}')
    with open(config_p, 'r') as f:
        config: dict = yaml.safe_load(f)
    # Check for config keys
    if check_schema:
        expected_keys = set(CONF_KEYS._asdict().values())
        if not expected_keys.issubset(config.keys()):
            raise AssertionError(f"Missing keys: {expected_keys - set(config.keys())}")
    # Convert nn modules to actual classes
    config = replace_nn_modules(config)
    if do_setup:
        # Set up nested structure
        setup_config(config=config)
    return config

def ens_to_symbol(adata: ad.AnnData, gene_symbol_keys: list[str] = ['gene_symbol', 'gene_name', 'gene', 'gene symbol', 'gene name']) -> ad.AnnData:
    # Look for possible gene symbol columns
    gscl = adata.var.columns.intersection(set(gene_symbol_keys)).values
    if len(gscl) == 0:
        raise ValueError(f'Could not find a column that describes gene symbol mappings in adata.var, looked for {gene_symbol_keys}')
    # Choose first hit if multiple
    gsh = list(gscl)[0]
    # Convert index
    adata.var.reset_index(names='ensembl_id', inplace=True)
    adata.var.set_index(gsh, inplace=True)
    # Check for duplicate index conflicts
    if adata.var_names.nunique() != adata.shape[0]:
        log.info(f'Found duplicate indices for ensembl to symbol mapping, highest number of conflicts: {adata.var_names.value_counts().max()}')
        # Fix conflicts by choosing the gene with the higher harmonic mean of mean expression and normalized variance out of pool
        if len(set(['means', 'variances_norm']).intersection(adata.var.columns)) == 2:
            adata.var['hm_var'] = (2 * adata.var.means * adata.var.variances_norm) / (adata.var.means + adata.var.variances_norm)
        else:
            adata.var['hm_var'] = np.arange(adata.n_vars)
        idx = adata.var.reset_index().groupby(gsh, observed=True).hm_var.idxmax().values
        adata = adata[:,idx]
    return adata


def filter_min_cells_per_class(adata: ad.AnnData, cls_label: str, min_cells: int = 10) -> None:
    """Filter adata for minimum number of cells in .obs group"""
    if cls_label not in adata.obs:
        log.warning(f'{cls_label} not in adata.obs. Could not filter for cells per class.')
        return
    # Calculate number of cells per class
    cpc = adata.obs[cls_label].value_counts()
    # Save number of classes before filtering
    nc = cpc.shape[0]
    # Calculate class labels with min. number of samples
    valid = cpc[cpc >= min_cells].index
    # Create mask and subset
    cls_mask = adata.obs[cls_label].isin(valid)
    adata._inplace_subset_obs(cls_mask)
    # Save number of classes after filtering
    nc_post_filter = valid.shape[0]
    log.info(f'Filtered adata.obs.{cls_label} for min. {min_cells} cells. Got {nc_post_filter}/{nc} classes.')

def filter_efficiency_score(adata: ad.AnnData, min_score: float = 2.0, col: str = 'mixscale_score') -> None:
    """Filter adata for minimum mixscale score."""
    if col not in adata.obs:
        log.warning(f'{col} not in adata.obs. Could not filter for mixscale score.')
        return
    # Filter for perturbation efficiency score (mixscale or similar)
    efficiency_mask = adata.obs[col] >= min_score
    nc = adata.shape[0]
    adata._inplace_subset_obs(efficiency_mask)
    nc_post_filter = adata.shape[0]
    log.info(f'Filtered adata.obs.{col} >= {min_score}. Got {nc_post_filter}/{nc} cells.')

def read_adata(
        adata_p: str, 
        check_gene_names: bool = True,
        cls_label: str | None = 'cls_label',
        min_cells_per_class: int | None = 10,
        min_efficiency_score: float | None = 2.0,
        efficiency_col: str | None = 'mixscale_score',
        **kwargs
    ) -> ad.AnnData:
    """Wrapper for sc.read()"""
    if adata_p is None or not os.path.exists(adata_p):
        raise FileNotFoundError(f'Cannot find file {adata_p}.')
    # Read adata using scanpy
    adata = sc.read(adata_p, **kwargs)
    # Convert gene names if needed
    if check_gene_names and adata.var_names.str.lower().str.startswith('ens').all():
        log.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        adata = ens_to_symbol(adata).copy()
    # Filter for perturbation efficiency if score is given
    if efficiency_col is not None and min_efficiency_score is not None:
        filter_efficiency_score(adata, min_score=min_efficiency_score, col=efficiency_col)
    # Filter classes if specified
    if cls_label is not None and min_cells_per_class is not None:
        filter_min_cells_per_class(adata, cls_label=cls_label, min_cells=min_cells_per_class)
    return adata
