# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Training on the toy v0.1.0 configuration (CPU).

* liveness: every loss term of the shipped model is switched on AND reaches the objective -- a mechanism that
  is configured but silently inert would pass any comparison against an equally inert reference;
* the two ablation arms kept in the package (free classifier head instead of proxy anchors; text-initialised
  atoms), and the coherence guard that trips when text init is combined with the prior pull;
* resume: 1 epoch + resume to 3 must equal 3 uninterrupted epochs BIT FOR BIT, from a Lightning checkpoint and
  from a saved model directory.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from conftest import StepRecorder, build_model, make_adata, make_config, make_files

pytestmark = pytest.mark.e2e


def _train(cfg, adata=None, epochs=None, seed=0):
    cfg = copy.deepcopy(cfg)
    if epochs is not None:
        cfg["train"]["max_epochs"] = epochs
    adata = make_adata() if adata is None else adata
    model = build_model(adata, cfg, seed=seed)
    with StepRecorder() as rec:
        model.train(cfg)
    return model, rec


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}


def test_every_v010_loss_term_is_live(tiny_config):
    model, rec = _train(tiny_config)
    pa = tiny_config["model"]["extra_prior_atom_kwargs"]
    assert [e["epoch"] for e in rec.epochs] == [0, 1, 2]
    for k in ("train_loss", "reconstruction_loss_train", "kl_local_train", "pa/loss_train", "eff_loss_train",
              "prior_atom/contrib_train", "prior_neg/loss_train"):
        v = rec.series(k)
        assert len(v) and np.isfinite(v).all() and (v > 0).all(), f"{k} not live: {v}"
    # the configured weight is the weight that reaches the objective
    assert np.all(rec.series("weights/proxy_anchor_train") == 200.0)
    assert np.all(rec.series("weights/prior_atom_train") == pa["weight"])
    assert np.all(rec.series("weights/prior_neg_train") == pa["neg_weight"])
    # the pull is ACTIVE (state 6) from start_epoch on, moves the configured fraction, refits every epoch
    assert np.all(rec.series("prior_atom/state_train") == 6.0)
    assert np.allclose(rec.series("prior_atom/pull_frac_train"), pa["pull_q"], atol=1e-6)
    assert np.all(rec.series("prior_atom/refit_age_train") <= pa["refit_every"])
    assert np.all(rec.series("prior_neg/k_train") == pa["neg_k"])
    # the checkpoint monitor exists in EVERY validation, including the first
    assert [("val_live_macro_top10_recall_A" in v) for v in rec.val] == [True] * 3
    # f(missing) is stored at the end of training for predict(obs_only=False)
    assert model.module.has_prior_missing_fill()


def test_free_classifier_head_arm(tiny_config):
    """Table S3A: proxy anchors off, the cosine head trained at weight 200."""
    cfg = copy.deepcopy(tiny_config)
    cfg["model"]["extra_proxy_anchor_kwargs"]["weight"] = 0.0
    cfg["train"]["plan_kwargs"]["anneal_schedules"]["cls_head_weight"] = {"weight": 200.0}
    model, rec = _train(cfg, epochs=1)
    assert np.all(rec.series("cls_head_weight") == 200.0)
    assert "pa/loss_train" not in rec.epochs[-1] or np.all(rec.series("weights/proxy_anchor_train") == 0.0)
    assert model.cls_head_training_["steps"] > 0 and model.cls_head_training_["weight_max"] == 200.0


def test_text_init_arm_without_pull(tiny_config):
    """Table S3B: atoms initialised from the text bank, no prior pull."""
    cfg = copy.deepcopy(tiny_config)
    cfg["model"]["extra_cls_emb_kwargs"]["atom_init"] = "text"
    cfg["model"]["extra_prior_atom_kwargs"].update(weight=0.0, neg_weight=0.0)
    model = build_model(make_adata(), cfg)
    W0 = model.module.cls_emb.embedding.weight.detach().clone()
    with StepRecorder() as rec:
        model.train(cfg | {"train": cfg["train"] | {"max_epochs": 1}})
    assert np.all(rec.series("pa/loss_train") > 0)
    assert "prior_atom/contrib_train" not in rec.epochs[-1] or np.all(rec.series("prior_atom/contrib_train") == 0)
    # text init is not the random init: the table starts aligned with the text bank, not at randn * 0.02
    assert W0[: model.module.n_labels].norm(dim=1).mean() > 0.1


@pytest.mark.parametrize("atom_init", ["text", "random"])
def test_coherence_guard_trips_on_text_init_only(tmp_path, atom_init):
    """Text-initialised atoms start far more coherent than the guard allows, so the pull switches itself off
    (state 7) instead of collapsing the dictionary -- the reason S3B ran without the pull. Random-init atoms on
    the same data stay below it and are pulled (state 6).

    The guard measures the mean off-diagonal cosine of the fit atoms, which needs >= 32 of them PER CROSS-FIT
    FOLD (fewer -> NaN, and the guard cannot fire): 96 classes at pull_q 0.25 and 2 folds leave 36. Text init centres the bank over the whole vocabulary, so what makes the fit atoms
    coherent is that the SCREENED genes are a biased subset of it (well-studied genes sit together in text
    space). The toy gives the trained rows a common direction; i.i.d. Gaussian rows would never be coherent."""
    adata = make_adata(n_perts=96, n_vocab=120, n_cells=3840)
    bank = adata.uns["cls_embedding"].copy()
    bank.iloc[:96] += np.random.default_rng(2).normal(size=bank.shape[1]) * 1.5
    adata.uns["cls_embedding"] = bank
    cfg = make_config(tmp_path, make_files(tmp_path, adata))
    cfg["model"]["extra_cls_emb_kwargs"]["atom_init"] = atom_init
    # a 32-d toy latent puts random atoms' mean cosine at +-0.007 -- next to coh_max; 512-d brings it to +-0.002
    cfg["model"].update(n_latent=512, n_shared=512)
    _, rec = _train(cfg, adata=adata, epochs=1)
    state = rec.series("prior_atom/state_train")
    assert len(state) == 1 and np.isfinite(rec.series("prior_atom/coh_strong_train")).all()
    if atom_init == "text":
        assert state[0] == 7.0 and rec.series("prior_atom/contrib_train")[0] == 0
    else:
        assert state[0] == 6.0 and rec.series("prior_atom/contrib_train")[0] > 0


@pytest.mark.parametrize("source", ["checkpoint", "model_dir"])
def test_resume_is_bit_identical(tiny_config, tmp_path, source):
    from expert import ExPert
    straight, _ = _train(tiny_config)

    cfg = copy.deepcopy(tiny_config)
    cfg["_model_log_dir"] = str(tmp_path / "resumed")
    adata = make_adata()
    first, _ = _train(cfg, adata=adata, epochs=1)
    if source == "model_dir":
        first.save(str(tmp_path / "final"), save_anndata=False)
        resumed = ExPert.load(str(tmp_path / "final"), adata=adata, use_latest_checkpoint=False)
    else:
        resumed = ExPert.load(cfg["_model_log_dir"], adata=adata, use_latest_checkpoint=True)
    assert resumed._resume_training_state["epochs_completed"] == 1
    resumed.train(cfg)

    a, b = _state(straight), _state(resumed)
    assert a.keys() == b.keys()
    diff = [k for k in a if not torch.equal(a[k], b[k])]
    assert not diff, f"{len(diff)} tensors differ after resume, e.g. {diff[:5]}"
