# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

# Annotations as strings (PEP 563) so `ad.AnnData` / `torch.Tensor` in signatures
# don't evaluate the heavy deps at import — lets config sampling stay torch-free.
from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections.abc import Mapping

from expert.tune._statics import CONF_KEYS, NESTED_CONF_KEYS

import importlib
import logging
log = logging.getLogger(__name__)


class _LazyModule:
    """Import-on-first-use proxy for a heavy module.

    torch/anndata are only touched by the data-I/O helpers, never by config sampling
    (sample_configs). Binding them as lazy proxies keeps create_sweep_dir fast — it no
    longer cold-imports torch off the network conda env. Bound as real module globals so
    ordinary `torch.Tensor(...)` LOAD_GLOBAL lookups inside functions resolve here (unlike
    a module-level __getattr__, which only fires for external attribute access). Paired
    with `from __future__ import annotations` so `-> ad.AnnData` signatures stay strings
    and don't trigger the import at definition time.
    """
    __slots__ = ("_name", "_mod")

    def __init__(self, name: str):
        self._name = name
        self._mod = None

    def __getattr__(self, attr):
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Static type-checkers / IDEs resolve the real modules (autocomplete, torch.Tensor
    # types). This branch is False at runtime and never imports anything — typing alone
    # can't make these lazy, because the functions below USE torch/ad at runtime, so the
    # real objects still have to exist. The `else` proxies provide that, imported on first
    # use. Paired with `from __future__ import annotations` so signature hints stay strings.
    import torch
    import torch.nn as nn
    import anndata as ad
else:
    torch = _LazyModule("torch")
    nn = _LazyModule("torch.nn")
    ad = _LazyModule("anndata")


def get_latest_ckpt_dir(dir_path, ckpt_key: str = 'checkpoints') -> str | None:
    import glob
    ckpt_paths = glob.glob(os.path.join(dir_path, '**', ckpt_key, '**', 'model.pt'), recursive=True)
    if ckpt_paths:
        latest = max(ckpt_paths, key=os.path.getmtime)
        return os.path.dirname(latest)
    else:
        return None

def check_adata_cat_cols(adata: ad.AnnData):
    for col in adata.obs.select_dtypes(include="category").columns:
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()

def sanitize_for_logging(obj, max_list_len=20):
    """
    Recursively sanitize hyperparams for logging.
    Keeps only simple types.
    """

    # Allowed primitive types
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    # Small lists/tuples
    if isinstance(obj, (list, np.ndarray, tuple)):
        if len(obj) <= max_list_len:
            return [sanitize_for_logging(x) for x in obj]
        else:
            return f"<list len={len(obj)}>"

    # Dicts
    if isinstance(obj, Mapping):
        o = {
            str(k): sanitize_for_logging(v)
            for k, v in obj.items()
        }
        # Remove None values
        return {k: v for k, v in o.items() if v is not None}

    # Torch tensors
    if isinstance(obj, torch.Tensor):
        return f"<Tensor shape={tuple(obj.shape)}>"

    # Class types, return repr
    if isinstance(obj, type):
        return str(obj)
    
    # Object instances, remove instance information
    if isinstance(obj, object):
        return f"{obj.__class__}"

    # Everything else gets removed
    return None

def load_json(p: str, mode: str = 'r', **kwargs):
    import json
    if p is None or not isinstance(p, str) or not p.endswith('json'):
        return None
    with open(p, mode) as f:
        return json.load(f, **kwargs)

def file_md5(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute md5 hex digest of a file, streamed in chunks."""
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

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
