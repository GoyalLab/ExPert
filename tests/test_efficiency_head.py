# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""EfficiencyHead output-shape contract.

The per-dataset efficiency objective + gate assume eff_hat is 1-D (B,).
The head squeezes its final dim; guard that so a regression there can't
silently broadcast against the (B,) target in the loss.
"""
import torch

from expert.modules._base import EfficiencyHead


def test_output_is_1d():
    head = EfficiencyHead(latent_dim=32)
    z = torch.randn(10, 32)
    out = head(z)
    assert out.shape == (10,), f"expected (B,), got {tuple(out.shape)}"


def test_backprop():
    head = EfficiencyHead(latent_dim=16)
    z = torch.randn(5, 16, requires_grad=True)
    head(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
