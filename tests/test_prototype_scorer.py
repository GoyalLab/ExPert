# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""PrototypeScorer: cosine logits over the prototype table + the diagnostic proto_ce.

Pins the op order the removed ClipLoss used (normalise, matmul, divide by a scalar or per-cell T), the
training-noise RNG draw, the weighted CE reduction, and that nothing it returns carries autograd.
"""
import pytest
import torch
import torch.nn.functional as F

from expert.modules._prototype_scorer import PrototypeScorer


def _data(seed=0, B=12, C=7, D=5):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, generator=g, requires_grad=True)
    emb = torch.randn(C, D, generator=g, requires_grad=True)
    y = torch.randint(0, C, (B,), generator=g)
    w = torch.rand(B, generator=g)
    return z, emb, y, w


@pytest.mark.parametrize("per_cell_T", [False, True])
def test_eval_logits_match_cosine(per_cell_T):
    z, emb, y, w = _data()
    T = torch.full((z.shape[0], 1), 0.07) if per_cell_T else 0.03
    ps = PrototypeScorer().eval()
    out = ps(z=z, z_norm=F.normalize(z, dim=-1), emb_norm=F.normalize(emb, dim=-1), y=y, T=T, weights=w)
    ref = (F.normalize(z, dim=-1) @ F.normalize(emb, dim=-1).T) / T
    assert torch.equal(out['logits'], ref.detach())
    assert torch.equal(PrototypeScorer.logits(z, emb, T), ref)
    assert not out['logits'].requires_grad and not out['proto_ce'].requires_grad
    ce = F.cross_entropy(ref.detach(), y, reduction='none')
    assert torch.equal(out['proto_ce'], (ce * w).sum() / w.sum().clamp_min(1e-12))


def test_train_noise_draw():
    z, emb, y, _ = _data(1)
    ps = PrototypeScorer(noise_std=0.01).train()
    torch.manual_seed(3)
    out = ps(z=z, z_norm=F.normalize(z, dim=-1), emb_norm=F.normalize(emb, dim=-1), y=y, T=0.03)
    torch.manual_seed(3)
    zn = F.normalize(z + torch.randn_like(z) * 0.01, dim=-1)
    assert torch.equal(out['logits'], ((zn @ F.normalize(emb, dim=-1).T) / 0.03).detach())
    assert torch.equal(out['proto_ce'], F.cross_entropy(out['logits'], y, reduction='none').mean())


def test_removed_knobs():
    PrototypeScorer(use_soft_targets=False, use_cls_prototypes=False, momentum=0.1, soft_target_T=0.05)
    for k, v in (('use_soft_targets', True), ('use_cls_prototypes', True), ('arc_margin', 0.2),
                 ('supervision_dropout_rate', 0.1), ('use_reverse', True)):
        with pytest.raises(ValueError):
            PrototypeScorer(**{k: v})
