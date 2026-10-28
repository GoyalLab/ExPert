# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""predict() must not retain the query AnnDatas it is handed.

scvi registers an AnnDataManager per unseen query in ``_per_instance_manager_store[model.id]`` and
never removes it; each entry holds a strong reference to the query, which predict() has just written
an (n_var x width) gene-support placeholder into. A loop over query subsets therefore retained every
one of them -- measured at ~0.7 GB per call, OOM at 128 GB after ~174 calls on a real screen.

These tests are the regression: they fail without ``_release_query_managers`` and pass with it.
"""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from expert.models._expert import ExPert


def _model(tiny_adata, tiny_config, tmp_path):
    """An untrained model that can still run predict(): the forward path is all we exercise."""
    cfg = tiny_config
    ExPert.setup_anndata(tiny_adata, **cfg["model_setup"])
    m = ExPert(tiny_adata, model_log_dir=str(tmp_path / "logs"), **cfg["model"])
    m.is_trained_ = True          # skip training; predict() only needs a built module
    return m


def _query(tiny_adata, i):
    """A fresh query object, distinct from the training adata and from every other call."""
    q = tiny_adata[np.arange(i * 5, i * 5 + 20)].copy()
    q.obs_names = [f"q{i}_{j}" for j in range(q.n_obs)]
    return q


def test_repeated_predict_does_not_retain_queries(tiny_adata, tiny_config, tmp_path):
    m = _model(tiny_adata, tiny_config, tmp_path)
    store = m._per_instance_manager_store
    n_before = len(store.get(m.id, {}))
    own = m.adata_manager.adata_uuid

    refs = []
    for i in range(12):
        q = _query(tiny_adata, i)
        refs.append(weakref.ref(q))
        m.predict(q, obs_only=True, return_latents=True, progress_bar=False,
                  soft_predictions=False, refresh_ctrl_means=False)
        del q
        gc.collect()

    inst = store.get(m.id, {})
    assert len(inst) <= n_before, (
        f"manager store grew from {n_before} to {len(inst)}: predict() is retaining its queries")
    assert own in inst, "predict() released the model's OWN training manager"
    alive = [i for i, r in enumerate(refs) if r() is not None]
    assert not alive, f"query AnnDatas still alive after their loop iteration: {alive}"


def test_predict_keeps_caller_registered_query(tiny_adata, tiny_config, tmp_path):
    """A manager the CALLER registered before the call must survive it."""
    m = _model(tiny_adata, tiny_config, tmp_path)
    q = _query(tiny_adata, 0)
    m._validate_anndata(q, extend_categories=True, add_missing_categories=True)
    uid = q.uns["_scvi_uuid"]
    assert uid in m._per_instance_manager_store.get(m.id, {})

    m.predict(q, obs_only=True, return_latents=True, progress_bar=False,
              soft_predictions=False, refresh_ctrl_means=False)
    assert uid in m._per_instance_manager_store.get(m.id, {}), (
        "predict() released a manager the caller registered itself")
