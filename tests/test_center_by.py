# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""predict(center_by=...) must reproduce the per-group loop EXACTLY.

Centring scripts today call predict() once per group on a subset, which makes
``refresh_ctrl_gene_means`` rewrite ``ctrl_gene_means`` from that subset's own controls -- per-group
input centring achieved by serialising, at the cost of one full predict() per group. ``center_by``
precomputes one control-mean row per (group, dataset) and gathers per cell in a single pass.

The acceptance bar is equality, not similarity: predict() is bit-reproducible, so the two paths must
agree exactly. Anything less means the bank is not reproducing the loop's arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from expert.models._expert import ExPert
from expert.utils.constants import MODULE_KEYS

N_UNITS = 3
CTRL = "control"


@pytest.fixture
def grouped_adata(tiny_adata):
    """tiny_adata + a per-cell grouping column and real control cells in every (unit, dataset)."""
    rng = np.random.default_rng(0)
    q = tiny_adata.copy()
    unit = np.array([f"u{i % N_UNITS}" for i in range(q.n_obs)])
    q.obs["unit"] = unit
    lab = q.obs["perturbation"].astype(str).to_numpy()
    # ~40% controls, spread over every (unit, dataset) pair so each gets its own baseline
    ds = q.obs["dataset"].astype(str).to_numpy()
    for u in np.unique(unit):
        for d in np.unique(ds):
            rows = np.where((unit == u) & (ds == d))[0]
            lab[rows[: max(12, len(rows) // 2)]] = CTRL
    q.obs["perturbation"] = lab
    emb = q.uns["cls_embedding"]
    if CTRL not in emb.index:
        import pandas as pd
        extra = pd.DataFrame(rng.normal(size=(1, emb.shape[1])).astype(np.float32),
                             index=[CTRL], columns=emb.columns)
        q.uns["cls_embedding"] = pd.concat([emb, extra])
    return q


def _model(adata, cfg, tmp_path):
    ExPert.setup_anndata(adata, **cfg["model_setup"])
    m = ExPert(adata, model_log_dir=str(tmp_path / "logs"), **cfg["model"])
    m.is_trained_ = True
    enc = m.module.z_encoder
    # Put the encoder on the control-centring path this feature exists for. Buffers are assigned
    # directly: the test only needs the lookup to be live, not a trained baseline.
    n_b, n_g = len(m.idx_to_batch), adata.n_vars
    enc.center_input = True
    enc.center_mode = "ctrl"
    enc.ds_gene_mean = torch.zeros(n_b, n_g)
    enc.ds_gene_seen = torch.ones(n_b, dtype=torch.bool)
    enc.ctrl_gene_means = torch.zeros(n_b, n_g)
    return m


def _z(out):
    return np.asarray(out[MODULE_KEYS.Z_KEY], dtype=np.float64)


def _predict(m, q, **kw):
    # v0.1.0 (invariant_align) resolves center_mode=None to a z_shared CONTROL baseline, which center_by refuses;
    # an explicit zero baseline is the shipped usage (ALD: per-cell-type centring + center_mode=zeros)
    return m.predict(q, obs_only=True, return_latents=True, progress_bar=False, soft_predictions=False,
                     center_mode=np.zeros(m.module.n_shared, np.float32), control_label=CTRL, **kw)


def test_center_by_matches_the_per_group_loop_exactly(grouped_adata, tiny_config, tmp_path):
    m = _model(grouped_adata, tiny_config, tmp_path)
    q = grouped_adata
    pristine = m.module.z_encoder.ctrl_gene_means.detach().clone()

    # SINGLE PASS over the whole query, each cell centred on its own (unit, dataset) controls.
    z_single = _z(_predict(m, q.copy(), center_by="unit"))

    # THE LOOP this replaces: one predict() per group, each refreshing the buffer from its own cells.
    m.module.z_encoder.ctrl_gene_means = pristine.clone()
    unit = q.obs["unit"].astype(str).to_numpy()
    z_loop = np.empty_like(z_single)
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        z_loop[rows] = _z(_predict(m, q[rows].copy(), refresh_ctrl_means=True))

    assert z_single.shape == z_loop.shape
    d = np.abs(z_single - z_loop).max()
    assert d == 0.0, f"single pass and per-group loop disagree: max|diff| = {d:.3e} (must be exactly 0)"


def test_center_by_actually_groups(grouped_adata, tiny_config, tmp_path):
    """Guard against a silent no-op: per-group centring must differ from one query-wide baseline."""
    m = _model(grouped_adata, tiny_config, tmp_path)
    q = grouped_adata
    pristine = m.module.z_encoder.ctrl_gene_means.detach().clone()
    z_group = _z(_predict(m, q.copy(), center_by="unit"))
    m.module.z_encoder.ctrl_gene_means = pristine.clone()
    z_whole = _z(_predict(m, q.copy(), refresh_ctrl_means=True))
    assert np.abs(z_group - z_whole).max() > 0, (
        "center_by produced the same latents as a single query-wide baseline -- it is a no-op")


def test_center_by_bank_is_keyed_by_group_and_dataset(grouped_adata, tiny_config, tmp_path):
    m = _model(grouped_adata, tiny_config, tmp_path)
    q = m._validate_anndata(grouped_adata.copy(), extend_categories=True, add_missing_categories=True)
    bank, codes, report = m._build_center_by_bank(
        q, np.arange(q.n_obs), "unit", CTRL)
    n_pairs = len({(u, d) for u, d in zip(q.obs["unit"].astype(str), q.obs["dataset"].astype(str))})
    assert bank.shape[0] == n_pairs == len(report)
    assert codes.shape[0] == q.n_obs and codes.max() == n_pairs - 1
    assert all(src == "ctrl" for *_, src in report), f"expected control-derived rows, got {report}"


def test_center_by_refuses_to_half_group_the_latent_baseline(grouped_adata, tiny_config, tmp_path):
    """center_by groups the INPUT baseline; pairing it with a z_shared bank must raise, not approximate."""
    m = _model(grouped_adata, tiny_config, tmp_path)
    if not getattr(m.module, "invariant_align", False):
        pytest.skip("module has no z_shared centring to resolve")
    with pytest.raises(ValueError, match="center_by"):
        _predict_kw = dict(obs_only=True, return_latents=True, progress_bar=False,
                           soft_predictions=False, control_label=CTRL)
        m.predict(grouped_adata.copy(), center_by="unit", center_mode="control", **_predict_kw)
