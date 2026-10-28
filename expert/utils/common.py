# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import torch


class GradientScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def grad_scale(x: torch.Tensor, scale: float = 1.0):
    """Scale gradients flowing back through x by `scale`. 0.0 = full detach, 1.0 = pass-through."""
    if scale == 1.0:
        return x
    if scale == 0.0:
        return x.detach()
    return GradientScaleFn.apply(x, scale)


def align_x_batch(
    x: torch.Tensor,
    col_map: torch.Tensor,
    n_model_vars: int,
) -> torch.Tensor:
    """Per-batch column remap + zero-fill for query → model var alignment.

    Returns a tensor ``out`` of shape ``(B, n_model_vars)`` where
    ``out[:, j] == x[:, col_map[j]]`` for every ``col_map[j] >= 0`` and
    ``0`` for every ``col_map[j] < 0`` (missing-feature column).

    Used by ``predict()`` to stream column alignment per minibatch
    instead of materializing the aligned adata.X upfront — keeps backed-
    h5ad queries from being forced into RAM and avoids the 2x peak that
    upfront ``align_model_features`` would cause for large queries.

    Parameters
    ----------
    x
        Batch tensor in **query** var space, shape ``(B, n_query_vars)``.
    col_map
        Length-``n_model_vars`` ``torch.long`` tensor. ``col_map[j]`` is
        the column index in ``x`` for output column ``j``, or any negative
        value to insert a zero column there.
    n_model_vars
        Number of output columns (model's var count). Must equal
        ``col_map.numel()``.
    """
    present = col_map >= 0
    out = x.new_zeros((x.shape[0], n_model_vars))
    # index_select is the cheapest gather; needs non-negative indices, so
    # we filter first via the boolean mask. Same device, same dtype as x.
    out[:, present] = x.index_select(1, col_map[present])
    return out
