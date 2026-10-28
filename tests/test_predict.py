# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""predict() on a model trained once on the toy v0.1.0 configuration: prototypes, the deflate readout and its
calibration, f(missing) for untrained genes, and a save / load round trip."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from conftest import N_OBS_PERTS, N_VOCAB, build_model, make_adata, make_config, make_files
from expert.utils.constants import MODULE_KEYS

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trained")
    adata = make_adata()
    cfg = make_config(tmp, make_files(tmp, adata))
    cfg["train"]["max_epochs"] = 2
    model = build_model(adata, cfg)
    model.train(copy.deepcopy(cfg))
    return model, adata, tmp


def _pseudobulk(adata):
    """Per (dataset, perturbation) count sums with the number of cells behind each row."""
    g = adata.obs.groupby(["dataset", "perturbation"], observed=True).indices
    X = np.vstack([np.asarray(adata.X[ix].sum(0)) for ix in g.values()])
    obs = pd.DataFrame([{"dataset": d, "perturbation": p, "n_cells": len(ix), "efficiency": 1.0}
                        for (d, p), ix in g.items()])
    obs.index = obs.index.astype(str)
    import anndata as ad
    pb = ad.AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=adata.var_names))
    for c in ("dataset", "perturbation"):
        pb.obs[c] = pb.obs[c].astype("category")
    return pb


def test_get_prototypes_is_the_scored_bank(trained):
    model, adata, _ = trained
    T, V = model.get_prototypes(), model.get_prototypes(vocabulary=True)
    assert T.shape == (N_OBS_PERTS, model.module.n_shared) and V.shape[0] == N_VOCAB
    assert np.array_equal(V.values[:N_OBS_PERTS], T.values) and list(T.index) == list(model.idx_to_label[:N_OBS_PERTS])
    out = model.predict(adata.copy(), progress_bar=False, obs_only=True)
    assert np.array_equal(np.asarray(out[MODULE_KEYS.CLS_PROJ_KEY], np.float32), V.values)


def test_untrained_genes_carry_the_prior_fill(trained):
    """f(missing) replaces the random init of genes never perturbed in training (predict(obs_only=False))."""
    model, _, _ = trained
    V = model.get_prototypes(vocabulary=True).values
    raw = model.module.cls_emb().detach().cpu().numpy()
    filled = model.module.prior_missing_T.norm(dim=1).cpu().numpy()[N_OBS_PERTS:N_VOCAB] > 0
    assert filled.any()
    untrained = np.arange(N_OBS_PERTS, N_VOCAB)
    assert not np.allclose(V[untrained[filled]], raw[untrained[filled]])
    assert np.array_equal(V[untrained[~filled]], raw[untrained[~filled]])


def test_deflate_needs_the_calibration(trained):
    model, adata, _ = trained
    pb = _pseudobulk(adata)
    with pytest.raises(ValueError, match="readout calibration"):
        model.predict(pb.copy(), progress_bar=False, obs_only=True, readout="deflate", n_cells_key="n_cells")
    rng = np.random.default_rng(0)
    pools = rng.normal(scale=0.05, size=(300, model.module.n_shared))
    info = model.add_readout_calibration(pools, datasets=rng.choice(["ds0", "ds1", "ds2"], 300))
    assert info["n_pools"] == 300 and np.isfinite(info["slope_norm"])
    out = model.predict(pb.copy(), progress_bar=False, obs_only=True, readout="deflate", n_cells_key="n_cells")
    assert np.isfinite(np.asarray(out["readout_effect"], float)).all()


def test_save_load_round_trip(trained, tmp_path):
    from expert import ExPert
    model, adata, _ = trained
    q = adata[:200].copy()
    a = model.predict(q.copy(), progress_bar=False, obs_only=True)
    model.save(str(tmp_path / "m"), save_anndata=False)
    loaded = ExPert.load(str(tmp_path / "m"), use_latest_checkpoint=False)
    b = loaded.predict(q.copy(), progress_bar=False, obs_only=True)
    for k in (MODULE_KEYS.CLS_PROJ_KEY, MODULE_KEYS.Z_SHARED_KEY):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    assert loaded.get_prototypes(vocabulary=True).equals(model.get_prototypes(vocabulary=True))
