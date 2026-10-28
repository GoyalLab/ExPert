# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Pre-release configs and checkpoints: removed knobs drop only at their off value, flat knob families fold into
their dicts, removed state is dropped in memory, and the module validates its nested settings."""
from __future__ import annotations

import pytest
import torch

from expert.modules._xpert import XPert
from expert.utils import legacy


def test_removed_knob_at_off_value_is_dropped():
    out = legacy.translate_model_kwargs({"n_latent": 8, "use_group_dro": False, "omp_rollout_weight": 0.0,
                                         "struct_rerank_k": 7})
    assert out == {"n_latent": 8}


def test_removed_knob_switched_on_raises():
    with pytest.raises(ValueError, match="use_group_dro"):
        legacy.translate_model_kwargs({"use_group_dro": True})


def test_flat_families_fold_into_dicts():
    out = legacy.translate_model_kwargs({
        "prior_atom_weight": 3.0, "prior_atom_gate": "eff", "proxy_anchor_weight": 200.0,
        "proxy_anchor_alpha": 32.0, "extra_proxy_anchor_kwargs": {"delta": 0.1}})
    assert out == {"extra_prior_atom_kwargs": {"weight": 3.0, "gate": "eff"},
                   "extra_proxy_anchor_kwargs": {"delta": 0.1, "weight": 200.0, "alpha": 32.0}}


def test_removed_prior_atom_variant_is_not_folded():
    """Removed prior-atom knobs are handled BEFORE folding, so they never reach the dict."""
    out = legacy.translate_model_kwargs({"prior_atom_weight": 1.0, "prior_atom_fwd_blend": False,
                                         "prior_atom_bayes_bins": 10})
    assert out == {"extra_prior_atom_kwargs": {"weight": 1.0}}
    with pytest.raises(ValueError):
        legacy.translate_model_kwargs({"prior_atom_fwd_blend": True})


def test_flat_and_nested_disagreeing_raises():
    with pytest.raises(ValueError, match="both as flat"):
        legacy.translate_model_kwargs({"prior_atom_weight": 1.0, "extra_prior_atom_kwargs": {"weight": 2.0}})
    # agreeing duplicates are fine
    out = legacy.translate_model_kwargs({"prior_atom_weight": 2.0, "extra_prior_atom_kwargs": {"weight": 2.0}})
    assert out["extra_prior_atom_kwargs"] == {"weight": 2.0}


def test_nested_removed_knobs():
    out = legacy.translate_model_kwargs({"proxy_anchor_relax_beta": 0.0, "proxy_anchor_relax_k": 5,
                                         "extra_clip_kwargs": {"use_soft_targets": False, "momentum": 0.1}})
    assert out == {"extra_proxy_anchor_kwargs": {}, "extra_clip_kwargs": {"momentum": 0.1}}
    with pytest.raises(ValueError, match="relax_beta"):
        legacy.translate_model_kwargs({"extra_proxy_anchor_kwargs": {"relax_beta": 0.5}})


def test_schedules():
    cfg = {"train": {"plan_kwargs": {"anneal_schedules": {
        "cls_align_weight": {"weight": 0.0}, "dann_weight": {"weight": 0.0}, "kl_weight": {"max_weight": 0.1}}}}}
    out = legacy.translate_config(cfg)
    assert out["train"]["plan_kwargs"]["anneal_schedules"] == {"kl_weight": {"max_weight": 0.1}}
    cfg["train"]["plan_kwargs"]["anneal_schedules"]["dann_weight"] = {"weight": 0.1}
    with pytest.raises(ValueError, match="dann_weight"):
        legacy.translate_config(cfg)
    # pre-release loss() defaulted CLIP alignment to 1.0: a legacy config without that schedule trained it
    with pytest.raises(ValueError, match="cls_align_weight"):
        legacy.translate_config({"train": {"plan_kwargs": {"anneal_schedules": {"dann_weight": {"weight": 0.0}}}}})


def test_translate_config_does_not_mutate_its_input():
    cfg = {"model": {"prior_atom_weight": 3.0}}
    legacy.translate_config(cfg)
    assert cfg == {"model": {"prior_atom_weight": 3.0}}


def _module(**kw):
    return XPert(n_input=20, n_labels=6, n_batch=2, cls_emb=torch.randn(10, 8), **kw)


def test_removed_state_is_dropped_in_memory_only():
    m = _module()
    sd = m.state_dict()
    legacy_sd = dict(sd) | {"_dann_head.0.weight": torch.zeros(3), "group_dro_q": torch.zeros(2)}
    m.load_state_dict(legacy_sd)                                  # strict: only REMOVED_STATE keys may be extra
    assert "_dann_head.0.weight" in legacy_sd                     # the caller's dict is not rewritten
    with pytest.raises(RuntimeError, match="not_a_removed_key"):
        m.load_state_dict(dict(sd) | {"not_a_removed_key": torch.zeros(1)})


def test_nested_settings_resolve_and_validate():
    m = _module(extra_prior_atom_kwargs={"weight": 3.0, "gate": "eff", "evidence": "nqxnds"},
                extra_proxy_anchor_kwargs={"weight": 200.0})
    assert (m.prior_atom_weight, m.prior_atom_gate, m.prior_atom_evidence) == (3.0, "hard", "nqxnds")
    assert m.prior_atom_pull_q == XPert._PRIOR_ATOM_DEFAULTS["pull_q"] and m.proxy_anchor_weight == 200.0
    for bad in ({"extra_prior_atom_kwargs": {"no_such_key": 1}}, {"extra_prior_atom_kwargs": {"gate": "soft"}},
                {"extra_prior_atom_kwargs": {"evidence": "fisher"}}, {"extra_proxy_anchor_kwargs": {"relax_beta": 0}}):
        with pytest.raises(ValueError):
            _module(**bad)
