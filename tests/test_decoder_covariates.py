# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""DecoderSCVI categorical-covariate sizing.

Guards the bug where adding a 2nd categorical covariate broke the decoder:
the input projection was sized for one concatenated embedding but the
forward pass concatenated one embedding *per* covariate. The fix made
n_cat_frac a shared budget split proportionally to category count.
"""
import torch

from expert.modules._base import DecoderSCVI


def _decoder(n_cat_list, n_cat_frac=1.0, n_input=64, n_output=120):
    return DecoderSCVI(
        n_input=n_input, n_output=n_output, n_hidden=48,
        n_cat_list=n_cat_list, n_layers=1, decoder_type="mixed",
        use_cat_emb=True, n_cat_frac=n_cat_frac, inject_covariates=True,
    )


def test_two_covariates_forward_shape():
    """The exact crash case: two covariates (19-way + 2-way) must produce a
    decoder whose input layer matches the concatenated per-covariate embeds."""
    B, n_input, n_genes = 8, 64, 120
    dec = _decoder([19, 2], n_cat_frac=1.0, n_input=n_input, n_output=n_genes)
    z = torch.randn(B, n_input)
    lib = torch.zeros(B, 1)
    ds = torch.randint(0, 19, (B, 1))
    tis = torch.randint(0, 2, (B, 1))
    px_scale, px_r, px_rate, px_dropout = dec("gene", z, lib, ds, tis)
    assert px_scale.shape == (B, n_genes)


def test_budget_split_proportional_to_category_count():
    """cat_emb_dims sum to the budget (n_input * n_cat_frac) and the
    larger-cardinality covariate gets the larger slice."""
    n_input, frac = 64, 1.0
    dec = _decoder([19, 2], n_cat_frac=frac, n_input=n_input)
    assert sum(dec.cat_emb_dims) == int(n_input * frac)
    assert dec.cat_emb_dims[0] > dec.cat_emb_dims[1]  # 19-way > 2-way


def test_single_covariate_still_works():
    B, n_input, n_genes = 8, 64, 120
    dec = _decoder([19], n_cat_frac=1.0, n_input=n_input, n_output=n_genes)
    z = torch.randn(B, n_input)
    out = dec("gene", z, torch.zeros(B, 1), torch.randint(0, 19, (B, 1)))
    assert out[0].shape == (B, n_genes)
    assert sum(dec.cat_emb_dims) == n_input
