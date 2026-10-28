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
import torch.nn as nn


class ProjectionBlock(nn.Module):
    """Simple feedforward block for VAEs."""

    def __init__(
        self,
        n_input: int,
        n_output: int | None = None,
        activation_fn: nn.Module = nn.GELU,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        # Initialize with basic linear layer
        layers = [nn.Linear(n_input, n_output)]
        
        # Normalization (choose one)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(n_output))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(n_output))

        # Nonlinearity
        layers.append(activation_fn())

        # Optional dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # Create full block
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.block(x)

    
class FcBlock(nn.Module):
    """Fully connected bottleneck block with optional residuals."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int | None = None,
        use_layer_norm: bool = True,
        dropout_rate: float = 0.1,
        activation_fn: type[nn.Module] = nn.GELU,
        bias: bool = False,
        noise_std: float = 0.0,
        ff_mult: int = 1,
        use_residuals: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.use_residuals = use_residuals
        self.noise_std = noise_std
        # Setup hidden dimension
        n_hidden = int(n_input * ff_mult) if n_hidden is None else n_hidden
        # Setup layers
        self.ff = nn.Sequential(
            nn.LayerNorm(n_hidden) if use_layer_norm else nn.Identity(),
            nn.Linear(n_input, n_hidden, bias=bias),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, n_input, bias=bias) if n_input != n_hidden else nn.Identity(),
        )

        self.act = activation_fn()

    def forward(self, x: torch.Tensor, **kwargs):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        h = self.ff(x)

        if self.use_residuals:
            return self.act(x + h)
        else:
            return h
