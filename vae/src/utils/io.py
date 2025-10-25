import os
import glob
import yaml
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

import torch
import torch.nn as nn

from src.tune._statics import CONF_KEYS, NESTED_CONF_KEYS


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

def read_config(config_p: str, setup: bool = True) -> dict:
    """Read hyperparameter yaml file"""
    logging.info(f'Loading config file: {config_p}')
    with open(config_p, 'r') as f:
        config: dict = yaml.safe_load(f)
    # Check for config keys
    expected_keys = set(CONF_KEYS._asdict().values())
    assert expected_keys.issubset(config.keys()), f"Missing keys: {expected_keys - set(config.keys())}"
    # Convert nn modules to actual classes
    config = replace_nn_modules(config)
    if setup:
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
        logging.info(f'Found duplicate indices for ensembl to symbol mapping, highest number of conflicts: {adata.var_names.value_counts().max()}')
        # Fix conflicts by choosing the gene with the higher harmonic mean of mean expression and normalized variance out of pool
        if len(set(['means', 'variances_norm']).intersection(adata.var.columns)) == 2:
            adata.var['hm_var'] = (2 * adata.var.means * adata.var.variances_norm) / (adata.var.means + adata.var.variances_norm)
        else:
            adata.var['hm_var'] = np.arange(adata.n_vars)
        idx = adata.var.reset_index().groupby(gsh, observed=True).hm_var.idxmax().values
        adata = adata[:,idx]
    return adata

def read_adata(adata_p: str, check_gene_names: bool = True) -> ad.AnnData:
    adata = sc.read(adata_p)
    # Convert gene names if needed
    if check_gene_names and adata.var_names.str.lower().str.startswith('ens').all():
        logging.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        adata = ens_to_symbol(adata).copy()
    return adata
