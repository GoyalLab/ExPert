# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Prototype scorer: cosine logits of z_shared against the prototype (atom) table.

The prototypes are the learnable atom table (``ClassEmbedding`` with ``random_init_proxies``), trained by the
proxy anchor. This module only SCORES against them:

    logits = cos(z_shared, prototypes) / T

These are the A-head logits ``classify()`` / ``predict()`` rank with, the source of every ``proto_*`` recall
metric and of the checkpoint monitor ``val_live_macro_top10_recall_A``. The cross-entropy over the logits is a
DIAGNOSTIC (``proto_ce``): it is never added to the training loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from expert.utils.constants import LOSS_KEYS

# Knobs of the removed CLIP alignment loss (losses/clip_v2.ClipLoss) that ``extra_clip_kwargs`` may still carry.
# Each is accepted only at a value that meant "off" -- every one of them only changed the (now removed) training
# CE or its sampling -- and is otherwise ignored. Anything else not consumed by the scorer is ignored as well:
# ClipLoss swallowed unknown keys through **kwargs.
_REMOVED_OFF = {
    'use_soft_targets': (False, None),
    'use_cls_prototypes': (False, None),
    'use_reverse': (False, None),
    'arc_margin': (0, 0.0, None),
    'supervision_dropout_rate': (0, 0.0, None),
    'ds_sim_weights': (None,),
    'cls_weights': (None,),
}


def logit_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-cell true-class logit minus the best other logit."""
    pos = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    top2 = torch.topk(logits, k=2, dim=-1).values
    neg = torch.where(top2[:, 0] == pos, top2[:, 1], top2[:, 0])
    return pos - neg


def margin(logits, targets):
    B, C = logits.shape
    pos = logits[torch.arange(B), targets]
    neg = logits.clone()
    neg[torch.arange(B), targets] = -1e9
    neg_max = neg.max(dim=1).values
    return (pos - neg_max).mean()


def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()


class PrototypeScorer(nn.Module):
    """Cosine scorer of cells against the prototype table (no parameters, no buffers).

    Parameters
    ----------
    label_smoothing
        Label smoothing of the diagnostic ``proto_ce`` (also read by the supervised head's CE).
    noise_std
        Gaussian noise added to z before scoring in TRAINING mode only (the train-split logits and metrics).
    reduction
        Reduction of the per-cell ``proto_ce`` ('mean', 'batchmean' or 'sum'), set from ``non_elbo_reduction``.
    """

    def __init__(self, label_smoothing: float = 0.0, noise_std: float = 0.01, reduction: str = 'mean', **kwargs):
        super().__init__()
        live = {k: kwargs[k] for k, off in _REMOVED_OFF.items()
                if k in kwargs and not any(kwargs[k] is o or kwargs[k] == o for o in off)}
        if live:
            raise ValueError(f'extra_clip_kwargs switches on removed CLIP-alignment option(s): {live}. '
                             'The prototype scorer has no alignment loss.')
        self.label_smoothing = label_smoothing
        self.noise_std = noise_std
        self.reduction = reduction

    @staticmethod
    def logits(z: torch.Tensor, emb: torch.Tensor, T) -> torch.Tensor:
        """cos(z, emb) / T. ``T`` is a scalar or a per-cell (B, 1) tensor."""
        z = F.normalize(z, dim=-1)
        emb = F.normalize(emb, dim=-1)
        return (z @ emb.T) / T

    def _reduce(self, loss: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if weights is not None:
            w = weights.view(-1, *([1] * (loss.ndim - 1)))
            loss = loss * w
        if self.reduction == "mean":
            if weights is None:
                return loss.mean()
            return loss.sum() / weights.sum().clamp_min(1e-12)
        elif self.reduction == "batchmean":
            loss_sum = loss.sum(dim=tuple(range(1, loss.ndim)))
            if weights is None:
                return loss_sum.mean()
            return loss_sum.sum() / weights.sum().clamp_min(1e-12)
        elif self.reduction == "sum":
            return loss.sum()
        raise ValueError(f"Invalid reduction: {self.reduction}")

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        z_norm: torch.Tensor,
        emb_norm: torch.Tensor,
        y: torch.Tensor,
        T,
        weights: torch.Tensor | None = None,
    ) -> dict:
        """Score ``z`` against the normalised prototypes ``emb_norm``.

        Returns the detached logits, the diagnostic CE (``proto_ce``, weighted by ``weights``) and margin /
        entropy diagnostics. Everything is computed without autograd: nothing here enters the loss.
        """
        y = y.flatten()
        if self.training and self.noise_std > 0:
            z_noisy = z + torch.randn_like(z) * self.noise_std
            z_norm = F.normalize(z_noisy, dim=-1)
        logits = (z_norm @ emb_norm.T) / T
        ce = self._reduce(
            F.cross_entropy(logits, y, reduction='none', label_smoothing=self.label_smoothing),
            weights=weights)
        margin_vals = logit_margins(logits, y)
        return {
            'logits': logits,
            LOSS_KEYS.PROTO_CE: ce,
            'proto/margin': margin_vals,
            'proto/margin_item': margin(logits, y),
            'proto/margin_mean': margin_vals.mean(),
            'proto/margin_std': margin_vals.std(),
            'proto/entropy': entropy(logits),
        }
