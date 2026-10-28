# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections.abc import Callable, Iterable
from typing import Literal, Iterable, Optional

from scvi.distributions import Normal

from expert.utils.constants import MODULE_KEYS

from expert.modules._blocks import ProjectionBlock, FcBlock

import logging


class EfficiencyHead(nn.Module):
    """Predicts a dataset-relative efficiency from the CENTRED z_shared.

    `linear=True` is a single Linear. Measured 2026-09-11 on H4/PAGW10: the MLP reaches
    spearman(eff_hat, eff) of only 0.351 on its OWN TRAIN split while its OOD loss RISES over
    training (1.3929 -> 1.4576) -- it overfits and underfits at once. A cross-validated ridge on
    the same input reaches 0.65/0.45, and a plain ||z|| reaches 0.450 against the head's 0.474, so
    the signal is linearly decodable and the extra capacity is buying overfitting, not fit.
    See [[project_cell_aggregation_variance_wall]].

    DEFAULT IS THE MLP so existing checkpoints load and in-flight arms are unchanged. The later
    offline test (2026-09-13, analysis_out/effhead_linear/, H4 + PAGW10) REFUTED switching: a linear
    head fitted on train cells transfers WORSE to OOD than the MLP (essential ~0.44 vs 0.47, gwps
    ~0.21 vs 0.26; train-fitted ridge, exact-loss and online-procedure fits alike), while an
    OOD-fitted ridge reaches 0.65/0.45 -- the signal is linear on OOD but the train-fitted direction
    does not transfer. Keep the MLP.
    """

    def __init__(self, latent_dim: int, linear: bool = False):
        super().__init__()
        self.net = nn.Linear(latent_dim, 1) if linear else nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )

    def forward(self, z: torch.Tensor):
        return self.net(z).squeeze(-1)  # (B,)


def _identity(x):
    return x


class FunnelFCLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = -1,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_activation: bool = True,
        activation_fn: nn.Module = nn.GELU,
    ):
        super().__init__()

        self.use_activation = use_activation
        self.use_layer_norm = use_layer_norm

        # Init modules
        self.layer_norm = nn.LayerNorm(n_out)

        # If n_hidden = -1, set it to n_in
        n_hidden = n_in if n_hidden == -1 else n_hidden

        # Geometric decay over n hidden to output layer
        hidden_dims = np.geomspace(n_hidden, n_out, num=n_layers + 1).astype(int)
        # Set n_hidden as bottleneck first dimension if its not the same as n_in
        e = 0
        if n_hidden != n_in:
            e = 1
            hidden_dims = np.r_[n_in, hidden_dims]
        
        # Init layers
        self.layers = nn.ModuleList()

        for i in range(n_layers + e):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            # Create block for each layer
            block = ProjectionBlock(
                n_input=in_dim, 
                n_output=out_dim, 
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_activation=use_activation,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
            )
            self.layers.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        # Apply layer norm
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x


class MixedLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = -1,
        n_cat_list: Optional[Iterable[int]] = None,
        n_encoder_layers: int = 1,
        n_layers: int = 3,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        use_activation: bool = True,
        use_attention: bool = False,
        bias: bool = True,
        inject_covariates: bool = False,
        activation_fn: nn.Module = nn.GELU,
        noise_std: float = 0.01,
        linear_encoder: bool = False,
        use_residuals: bool = True,
        **kwargs,
    ):
        super().__init__()
        if linear_encoder:
            raise ValueError('MixedLayers: linear_encoder=True is no longer supported.')

        self.inject_covariates = inject_covariates
        self.use_activation = use_activation
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.n_hidden = n_hidden
        self.kwargs = kwargs

        # Init modules
        self.layer_norm = nn.LayerNorm(n_out)

        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in (n_cat_list or [])]
        self.cat_dim = sum(self.n_cat_list)

        # Encode down to bottleneck latent space
        self.encoder = FunnelFCLayers(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_encoder_layers,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
        )
        
        # Introduce multiple attention layers of same dimension
        self.layers = nn.ModuleList()
        for _ in np.arange(n_layers):
            # Create block for each layer
            block = FcBlock(
                n_input=n_out,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn,
                bias=bias,
                noise_std=noise_std,
                use_residuals=use_residuals,
                **kwargs
            )
            self.layers.append(block)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        return x
    

# Encoder
class Encoder(nn.Module):
    """Encode (optionally control-centred) expression into the latent space."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_cat_list: Iterable[int] = None,
        library_mode: str = 'off',      # 'off' | 'feature' | 'regress' -- see self.library_mode
        nuisance_feats: tuple = (),     # extra scalars appended as inputs: 'mito' | 'ribo' | 'ngenes'
        mito_mask: torch.Tensor | None = None,   # (n_input,) bool, MT- genes
        ribo_mask: torch.Tensor | None = None,   # (n_input,) bool, RPS/RPL genes
        n_layers: int = 1,
        n_dim_context_emb: int | None = None,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        use_feature_mask: bool = True,
        feature_mask_eval_full: bool = False,  # train on each dataset's ACTIVE panel only, but let
                                               # every measured gene through at eval. Intended for a
                                               # future DB where each dataset ships its full panel:
                                               # training stays restricted to the informative genes,
                                               # inference uses whatever is available.
        drop_prob: float = 0.0,
        noise_sigma: float = 0.05,
        encoder_type: Literal["funnel", "fc", "mixed"] = "mixed",
        normalize_x: bool = False,
        x_weight: float = 0.0,
        use_emb: bool = False,
        linear_gene_proj: bool = True,
        init_emb_pca: bool = True,
        post_emb_ln: bool = False,
        residual_scale: float = 0.1,
        shared_mask: torch.Tensor | None = None,
        incl_nonshared_prob: float | None = 1.0,
        mask_target_prob: float = 0.0,
        mask_target_reveal_k: int = 1,
        n_labels: int | None = None,
        deterministic: bool = False,
        center_input: bool = False,        # subtract the per-(dataset,gene) mean from the ENCODER input only
        center_mode: Literal["ema", "batch", "ctrl", "ctrl_z"] = "batch",   # 'batch'=in-batch mean; 'ctrl'=precomputed control mean; 'ctrl_z'=(x-mu)/sd
        center_min_cells: int = 16,        # below this many cells of a dataset in-batch, fall back to the EMA
        n_batch: int | None = None,        # sizes the per-dataset gene-mean buffer
        ds_mean_momentum: float = 0.99,
        ctrl_gene_means=None,              # (n_batch, n_gene) CONTROL means (ctrl.h5ad) for center_mode='ctrl'
        ctrl_gene_stds=None,               # (n_batch, n_gene) CONTROL stds (ctrl.h5ad) for center_mode='ctrl_z'
        ctrl_pc_context=None,              # (n_batch, ctx_dim) per-dataset CONTROL-PC context vector
        ctrl_pc_mode: Literal['off', 'film', 'add'] = 'off',   # how that context conditions the encoder
        ctrl_pc_dropout: float = 0.0,      # p(zero the context) -> must degrade to the centering-only baseline
        ctrl_pool=None,                    # (n_batch, pool, n_genes) control cells for the LEARNED context
        ctrl_proj_dim: int = 0,            # learned projection width (0 = off)
        ctrl_proj_n: int = 256,            # control cells sampled PER STEP
        input_topk: int = 0,               # >0: keep only the K largest-|delta| genes/cell (perceiver-style restriction)
        ctrl_topk: int = 0,                # >0: keep only the K genes deviating most from THIS DATASET'S
                                           # CONTROL mean. SELECT-only (values stay raw) and the mask is
                                           # PUBLISHED, so mirror_recon_mask restricts the ELBO to them too.
        offset_rank: int = 0,              # >0: STRUCTURALLY delete the top-r dataset-offset subspace from the input
        **kwargs,
    ):
        super().__init__()
        # Input options of removed mechanisms are still ACCEPTED (callers and checkpoints pass them)
        # but only at their off value; anything else would silently train a different model.
        _removed = {
            'library_mode': library_mode != 'off', 'nuisance_feats': bool(nuisance_feats),
            'normalize_x': bool(normalize_x), 'use_emb': bool(use_emb), 'drop_prob': float(drop_prob or 0) > 0,
            'ctrl_pc_mode': str(ctrl_pc_mode) != 'off', 'ctrl_pc_context': ctrl_pc_context is not None,
            'ctrl_pool': ctrl_pool is not None, 'ctrl_proj_dim': int(ctrl_proj_dim or 0) > 0,
            'input_topk': int(input_topk or 0) > 0, 'ctrl_topk': int(ctrl_topk or 0) > 0,
            'offset_rank': int(offset_rank or 0) > 0, 'shared_mask': shared_mask is not None,
            'encoder_type': encoder_type != 'mixed',
            'center_mode': bool(center_input) and str(center_mode) not in ('ctrl', 'ctrl_z'),
        }
        _on = [k for k, v in _removed.items() if v]
        if _on:
            raise ValueError(f'Encoder: removed option(s) set to a non-off value: {_on}')
        # AE mode: return qz.loc instead of qz.rsample() (see XPert.deterministic).
        self.deterministic = deterministic

        # ---- per-(dataset, gene) input centering -------------------------------------------
        # Dataset effects dominate raw expression, so the encoder spends capacity on "which
        # dataset is this". Feeding x - ctrl_gene_means[dataset] removes the FIRST-ORDER dataset
        # shift before the encoder sees anything. The reconstruction TARGET stays raw x (the decoder
        # conditions on a learnable dataset embedding), so only the encoder's view changes.
        self.center_input = bool(center_input)
        self.center_mode = str(center_mode)
        self.center_min_cells = int(center_min_cells)
        self.ds_mean_momentum = float(ds_mean_momentum)
        if self.center_input and n_batch:
            self.register_buffer("ds_gene_mean", torch.zeros(int(n_batch), int(n_input)))
            self.register_buffer("ds_gene_seen", torch.zeros(int(n_batch), dtype=torch.bool))
        else:
            self.ds_gene_mean = None
            self.ds_gene_seen = None
        if self.center_input and n_batch:
            _c = torch.zeros(int(n_batch), int(n_input))
            if ctrl_gene_means is not None:
                _c = torch.as_tensor(ctrl_gene_means, dtype=torch.float32).clone()
            self.register_buffer("ctrl_gene_means", _c)
            # ONES default: a dataset/gene with no control support divides by 1, so 'ctrl_z'
            # degrades to plain 'ctrl' centering instead of producing inf/nan there.
            _s = torch.ones(int(n_batch), int(n_input))
            if ctrl_gene_stds is not None:
                _s = torch.as_tensor(ctrl_gene_stds, dtype=torch.float32).clone()
            self.register_buffer("ctrl_gene_stds", _s)
            self._register_load_state_dict_pre_hook(self._ctrl_gene_means_load_hook)
        else:
            self.ctrl_gene_means = None
            self.ctrl_gene_stds = None

        self.encoder_class = MixedLayers

        # Save init params
        self.encoder_type = encoder_type
        self.distribution = distribution
        self.var_eps = var_eps
        self.n_input = n_input
        self.n_dim_context_emb = n_dim_context_emb or 0
        self.noise_sigma = noise_sigma
        # Probability of zeroing out a cell's target gene (the gene the
        # perturbation targets) during training. Forces the encoder to
        # predict the perturbation from downstream signal rather than the
        # gene's own knocked-down expression — addresses the "fancy DEG
        # wrapper" failure mode and acts as a strong regularizer.
        # Mapping pert_label_idx -> var_idx is registered as a buffer
        # below; -1 entries mean "no mapped gene" (controls, drugs,
        # non-targeting guides) and are silently skipped.
        self.mask_target_prob = float(mask_target_prob)
        # in-range partial reveal of the masked target (see the class attribute's comment); an
        # instance value so a training config can set it, 1 = pure donor resample
        self.mask_target_reveal_k = int(mask_target_reveal_k)
        if n_labels is not None and n_labels > 0:
            self.register_buffer(
                "pert_to_gene_idx",
                torch.full((n_labels,), -1, dtype=torch.long),
                persistent=True,
            )
        else:
            self.pert_to_gene_idx = None
        # Set encoder output to double the latent dimension or n_hidden
        if n_hidden is None or n_hidden < 0:
            funnel_out_base = 2 * n_output
        else:
            funnel_out_base = n_hidden
        funnel_out_dim = funnel_out_base
       
        self.use_emb = False
        self.shared_mask = None

        # Base encoder layers
        self.encoder = self.encoder_class(
            n_in=n_input,
            n_out=funnel_out_dim,
            n_hidden=-1,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        # Control-PC context conditioning was removed; the attributes stay readable (off / None).
        self.ctrl_pc_mode = 'off'
        self.ctrl_pc_context = None
        self.mean_encoder = nn.Linear(funnel_out_dim, n_output)
        self.var_encoder = nn.Linear(funnel_out_dim, n_output)
        self.var_activation = torch.exp if var_activation is None else var_activation
        self.z_transformation = nn.Softmax(dim=-1) if distribution == "ln" else _identity

        # Debug
        self.c = 0
        
    def set_pert_to_gene_mapping(self, pert_names, var_names) -> int:
        """Populate the ``pert_to_gene_idx`` buffer from name vocabularies.

        Call after both vocabularies exist (typically right after the
        module is constructed). Case-insensitive match on gene symbol.
        Perturbations with no matching var get index ``-1`` and are
        silently skipped at masking time.

        Returns the number of perturbations that got a valid mapping.
        """
        if self.pert_to_gene_idx is None:
            raise RuntimeError(
                'pert_to_gene_idx buffer was not allocated (n_labels was '
                'None at init). Pass n_labels to the Encoder constructor.'
            )
        var_lookup = {str(v).upper(): i for i, v in enumerate(var_names)}
        mapping = torch.full_like(self.pert_to_gene_idx, -1)
        n_mapped = 0
        for i, p in enumerate(pert_names):
            idx = var_lookup.get(str(p).upper(), -1)
            mapping[i] = idx
            if idx >= 0:
                n_mapped += 1
        self.pert_to_gene_idx.copy_(mapping)
        return n_mapped

    # Target-gene masking replaces the target by a DONOR cell's value (see _resample_target_cols);
    # 'zero' keeps the old constant fill. Plain attribute, never a buffer: training-only behaviour,
    # so checkpoints are unaffected either way.
    mask_target_mode = 'resample'
    # PARTIAL REVEAL, IN-RANGE: draw k same-dataset donors and keep the one on the side the real
    # knockdown points (the lowest of k if the cell's true value is below their median, else the
    # highest). Every masked value is then a value the gene really takes in that context -- never an
    # outlier, never the extreme CRISPRi depletion -- and the tilt is bounded: an order statistic of
    # k draws is at most k times likelier than a uniform draw, so the target reads as "somewhat low
    # for this gene" (k=3: ~0.85 sd below its mean for a normal-shaped gene), noticeable only in
    # aggregate. k=1 = pure resample (target hidden). Plain attribute, training-only.
    mask_target_reveal_k = 1

    @staticmethod
    def _draw_donor_vals(x, rows, cols, valid, start, size, order):
        """One donor value per masked entry from its dataset group; `valid(donor)` vetoes donors.
        Returns (values, ok) -- ok False where 4 draws found no valid donor."""
        val = torch.zeros(rows.numel(), dtype=x.dtype, device=x.device)
        todo = torch.ones(rows.numel(), dtype=torch.bool, device=x.device)
        for _ in range(4):                                                # redraw rare collisions
            if not bool(todo.any()):
                break
            pick = start + (torch.rand(rows.numel(), device=x.device) * size).long().clamp_max(size - 1)
            donor = order[pick]
            ok = todo & valid(donor)
            val = torch.where(ok, x[donor, cols], val)
            todo = todo & ~ok
        return val, ~todo

    @staticmethod
    def _tilted_donor_value(x, rows, cols, k, draw, fallback=None):
        """k donor draws -> one value: the lowest if the cell's true value is below their median,
        else the highest (k=1: the single draw). `draw()` returns (values, ok) for all entries.
        Entries with no valid draw at all take `fallback` (default 0 = control baseline in the
        centred view)."""
        vals, oks = zip(*[draw() for _ in range(max(1, int(k)))])
        V, OK = torch.stack(vals, 1), torch.stack(oks, 1)
        fb = torch.zeros_like(V[:, 0]) if fallback is None else fallback.to(V.dtype)
        if V.shape[1] == 1:
            return torch.where(OK[:, 0], V[:, 0], fb)
        inf = torch.tensor(float('inf'), dtype=V.dtype, device=V.device)
        lo = torch.where(OK, V, inf).min(1).values
        hi = torch.where(OK, V, -inf).max(1).values
        med = torch.where(OK, V, torch.full_like(V, float('nan'))).nanmedian(1).values
        out = torch.where(x[rows, cols] < med, lo, hi)
        return torch.where(OK.any(1), out, fb)

    @staticmethod
    def _resample_target_cols(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                              gene_idx: torch.Tensor, bi: torch.Tensor | None,
                              reveal_k: int = 1) -> torch.Tensor:
        """Hide each masked (row, col) by a DONOR value: col's value in a random cell of the same
        dataset in this batch whose own target is not col (reveal_k > 1: the tilted pick of k such
        donors, see mask_target_reveal_k).

        A constant fill is itself a label: a centred 0 almost never occurs naturally (a zero count
        centres to -mu, and the multiplicative input noise keeps an exact 0 exact), so "this gene
        sits exactly at baseline" would mark the target. A donor's value is a draw from the gene's
        real distribution in that context -- real zeros, other perturbations' trans effects -- and
        carries no information about which gene was perturbed. Entries with no valid donor (a
        dataset alone in the batch, or every same-dataset cell targeting col) fall back to 0.
        """
        if rows.numel() == 0:
            return x
        n = x.shape[0]
        ds = bi.view(-1).long() if bi is not None else torch.zeros(n, dtype=torch.long,
                                                                    device=x.device)
        order = torch.argsort(ds, stable=True)
        sds = ds[order]
        uniq, counts = torch.unique_consecutive(sds, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        gpos = torch.searchsorted(uniq, ds[rows])                         # group of each masked row
        size, start = counts[gpos], starts[gpos]
        valid = lambda d: (d != rows) & (gene_idx[d] != cols)
        draw = lambda: Encoder._draw_donor_vals(x, rows, cols, valid, start, size, order)
        val = Encoder._tilted_donor_value(x, rows, cols, reveal_k, draw)
        x = x.clone()
        x[rows, cols] = val
        return x

    def _ctrl_gene_means_load_hook(self, state_dict, prefix, *args):
        """Inject the current ctrl_gene_means buffer when loading a pre-ctrl checkpoint that lacks it,
        so strict load_state_dict doesn't error on the missing key."""
        k = prefix + "ctrl_gene_means"
        if k not in state_dict and getattr(self, "ctrl_gene_means", None) is not None:
            state_dict[k] = self.ctrl_gene_means.detach().clone()
        # Same for ctrl_gene_stds (added later, for center_mode='ctrl_z'). Without this EVERY
        # checkpoint trained with center_input=True before the buffer existed fails a strict load
        # with "Missing key(s): z_encoder.ctrl_gene_stds" -- centering runs are precisely the ones
        # that register the buffer, so they are exactly the checkpoints that break.
        ks = prefix + "ctrl_gene_stds"
        if ks not in state_dict and getattr(self, "ctrl_gene_stds", None) is not None:
            state_dict[ks] = self.ctrl_gene_stds.detach().clone()

    def forward(
        self,
        x: torch.Tensor,
        *cat_list: int,
        feature_masks: torch.Tensor | None = None,
        pert_labels: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        ctrl_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        # EMA the per-(dataset, gene) mean on the RAW x, BEFORE target masking, so a masked
        # target's forced 0 never pollutes the baseline.
        if (self.center_input and self.ds_gene_mean is not None and batch_index is not None
                and self.training and torch.is_grad_enabled()):
            with torch.no_grad():
                bi = batch_index.view(-1).long()
                for d in bi.unique():
                    mu = x[bi == d].mean(0)
                    if not bool(self.ds_gene_seen[d]):
                        self.ds_gene_mean[d] = mu
                        self.ds_gene_seen[d] = True
                    else:
                        self.ds_gene_mean[d].mul_(self.ds_mean_momentum).add_(
                            mu, alpha=1.0 - self.ds_mean_momentum)
        # Target-gene masking (training-only). For each cell, with prob
        # `mask_target_prob`, hide the column corresponding to the
        # gene the cell's perturbation targets. Forces the encoder to
        # learn downstream effects rather than reading off the target
        # gene's own knocked-down expression. Skipped for perts mapped
        # to -1 (controls / drugs / no-gene-association).
        #
        # WHERE the zero goes matters. Under center_input, x is centred on a control mean right
        # below; zeroing RAW x first turns the target into x - mu = -mu, i.e. a PERFECT knockdown --
        # the mask would advertise the label instead of hiding it (the case for every model trained
        # with mask_target_prob > 0 under center_input before this fix, e.g. ExPert v0.1.1).
        # So with centring active the column is replaced in the CENTRED view below; without
        # centring, in raw x here. The replacement is a same-dataset DONOR cell's value
        # (_resample_target_cols), not a constant: an exact baseline value never occurs naturally and
        # would itself mark the target. mask_target_mode='zero' restores the constant fill.
        _tmask = None
        # EVAL-TIME variant (ablation): `encoder.eval_mask_target = True` (plain attribute, never a
        # buffer -> no checkpoint or RNG effect) hides EVERY labelled cell's own target the same way
        # at inference. It reads the cell's TRUE label, so it is a diagnostic, not a deployable readout.
        _eval_mt = (not self.training) and bool(getattr(self, 'eval_mask_target', False))
        if (
            ((self.training and self.mask_target_prob > 0) or _eval_mt)
            and pert_labels is not None
            and self.pert_to_gene_idx is not None
        ):
            labels_flat = pert_labels.view(-1).long()
            gene_idx = self.pert_to_gene_idx[labels_flat]            # (B,)
            if _eval_mt:
                do_mask = gene_idx >= 0
            else:
                dice = torch.rand_like(gene_idx, dtype=torch.float32)
                do_mask = (gene_idx >= 0) & (dice < self.mask_target_prob)
            if _eval_mt:                                            # running (masked, seen) tally
                c = getattr(self, '_eval_mt_counts', (0, 0))
                self._eval_mt_counts = (c[0] + int(do_mask.sum()), c[1] + int(do_mask.numel()))
            if do_mask.any():
                rows = torch.where(do_mask)[0]
                cols = gene_idx[do_mask]
                _tmask = (rows, cols, gene_idx)
                centred = (self.center_input and self.ds_gene_mean is not None
                           and batch_index is not None)
                if not centred:
                    if self.mask_target_mode == 'resample':
                        x = self._resample_target_cols(x, rows, cols, gene_idx, batch_index,
                                                       self.mask_target_reveal_k)
                    else:
                        x = x.clone()
                        x[rows, cols] = 0
                if feature_masks is not None:
                    feature_masks = feature_masks.clone()
                    feature_masks[rows, cols] = 0

        # Remove the per-(dataset, gene) CONTROL mean from the ENCODER's view of x. The target-gene
        # mask is applied to the RESULT (see above). Datasets absent from the control file (zero
        # row) keep the EMA baseline instead of subtracting a zero-filled row.
        if self.center_input and self.ds_gene_mean is not None and batch_index is not None:
            bi = batch_index.view(-1).long()
            mu = self.ds_gene_mean[bi].to(x.dtype)                      # (B, G) default EMA baseline
            _cb = getattr(self, 'center_bank', None)
            _cc = getattr(self, 'center_codes', None)
            if _cb is not None and _cc is not None and int(_cc.numel()) == x.shape[0]:
                # PER-GROUP baseline supplied by predict(center_by=...): one control-mean row per
                # (group, dataset), gathered per cell. Takes precedence over the per-dataset lookup
                # below, which is what looping predict() over the groups achieves by rewriting
                # ctrl_gene_means once per subset. Set and cleared by predict(); never a buffer, so
                # it cannot reach a checkpoint or shift RNG.
                mu = _cb.to(device=x.device, dtype=x.dtype)[_cc.view(-1).long().to(x.device)]
            elif self.center_mode in ("ctrl", "ctrl_z") and getattr(self, "ctrl_gene_means", None) is not None:
                # Precomputed per-dataset CONTROL mean (ctrl.h5ad): clean, dataset-invariant baseline
                # available from step 0. Rows are zero for datasets absent from the control file ->
                # keep the EMA baseline there (never subtract a zero-filled row on an unseen dataset).
                cmu = self.ctrl_gene_means[bi].to(x.dtype)
                has_ctrl = self.ctrl_gene_means[bi].abs().sum(-1, keepdim=True) > 0
                mu = torch.where(has_ctrl, cmu, mu)
            x = x - mu
            # ctrl_z: additionally divide by the per-(dataset,gene) CONTROL std, putting every
            # gene on a control-relative z-scale. Centering removes the dataset OFFSET; this also
            # removes its per-gene SCALE, which differs by protocol/depth/cell type. Genes without
            # control support hold sd=1, so those degrade exactly to 'ctrl'.
            if self.center_mode == "ctrl_z" and getattr(self, "ctrl_gene_stds", None) is not None:
                x = x / self.ctrl_gene_stds[bi].to(x.dtype).clamp_min(1e-3)
            # target-gene mask in the CENTRED view: a donor cell's centred value (default), or
            # "exactly at control baseline" under mask_target_mode='zero'
            if _tmask is not None:
                if self.mask_target_mode == 'resample':
                    x = self._resample_target_cols(x, _tmask[0], _tmask[1], _tmask[2], batch_index,
                                                   self.mask_target_reveal_k)
                else:
                    x = x.clone()
                    x[_tmask[0], _tmask[1]] = 0

        # EVAL-TIME GENE MASK on the encoder's view (plain attribute, never a buffer -> no checkpoint
        # or RNG effect). Zeroes the chosen columns of the CENTERED x, i.e. the model sees "this gene
        # is exactly at its control baseline" -- the MT- masking that helped on disease queries. Set it
        # from a readout/predict script (encoder.input_zero_mask = bool tensor of n_input); None = off.
        _zm = getattr(self, 'input_zero_mask', None)
        if _zm is not None and int(_zm.numel()) == x.shape[1]:
            x = x.masked_fill(_zm.to(x.device).view(1, -1), 0.0)

        # Multiplicative gene-space input noise (training only).
        if self.training and self.noise_sigma:
            x = x * torch.exp(torch.randn_like(x) * self.noise_sigma)

        # Feed x to the encoder
        q = self.encoder(x, *cat_list)

        # Debug
        if self.training:
            self.c = self.c + 1
        # Latent heads
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        qz = Normal(q_m, q_v.sqrt())
        z = self.z_transformation(qz.loc if self.deterministic else qz.rsample())
        return {
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QZM_KEY: q_m,
            MODULE_KEYS.QZV_KEY: q_v,
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.EFFECTIVE_MASK_KEY: feature_masks,
        }
        
        
# Decoder
class DecoderSCVI(nn.Module):
    """Decodes latent space with covariate conditioning and zinb parameters."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        decoder_type: Literal["fc", "funnel", "mixed"] | None = "mixed",
        linear_decoder: bool = False,
        use_cat_emb: bool = True,
        n_cat_dim: int | None = None,
        n_cat_frac: float = 1.0,
        use_film: bool = False,
        init_film_zero: bool = True,
        **kwargs,
    ):
        super().__init__()
        if use_film or linear_decoder or decoder_type != "mixed":
            raise ValueError(f'DecoderSCVI: removed option set (use_film={use_film}, '
                             f'linear_decoder={linear_decoder}, decoder_type={decoder_type!r}); only the '
                             f'"mixed" backbone without FiLM is supported.')
        self.use_film = use_film
        # Get cat input dim. Inactive covariates (≤1 category) map to 0.
        self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in (n_cat_list or [])]
        self.cat_dim = sum(self.n_cat_list)

        # Init covariate embeddings if enabled.
        #
        # ``n_cat_frac`` is the fraction of the latent dim allocated to the
        # *combined* categorical-covariate embedding (NOT per covariate).
        # The budget is split across active covariates proportional to their
        # category count — a covariate with 19 datasets needs more capacity
        # to keep its categories separable than a 2-way tissue-type split.
        # Each active covariate gets its own ``nn.Embedding`` sized to its
        # share; they're concatenated at decode time. ``n_cat_dim`` (if given)
        # overrides the fraction and sets the total budget directly.
        self.use_cat_emb = use_cat_emb and self.cat_dim > 0
        self.cat_emb_dims: list[int] = []
        if self.use_cat_emb:
            total_cat_dim = n_cat_dim if n_cat_dim is not None else int(n_input * n_cat_frac)
            active_counts = [n for n in self.n_cat_list if n > 0]
            total_active = sum(active_counts)
            self.cat_embeddings = nn.ModuleList()
            allocated = 0
            for i, n in enumerate(active_counts):
                if i == len(active_counts) - 1:
                    # Last covariate absorbs rounding remainder so the
                    # concatenated dim hits the budget exactly.
                    dim_i = max(total_cat_dim - allocated, 1)
                else:
                    dim_i = max(round(total_cat_dim * n / total_active), 1)
                allocated += dim_i
                self.cat_emb_dims.append(dim_i)
                emb = nn.Embedding(n, dim_i)
                # Scatter identity-like structure so categories start distinct
                nn.init.orthogonal_(emb.weight, gain=1.0)
                self.cat_embeddings.append(emb)
            decoder_input_dim = n_input + sum(self.cat_emb_dims)
        else:
            self.cat_embeddings = None
            decoder_input_dim = n_input

        # Determine decoder funnel dimensions
        funnel_out_dim = n_output // 2
        # Init decoder module
        self.px_decoder = MixedLayers(
            n_in=decoder_input_dim,
            n_out=funnel_out_dim,
            n_hidden=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inverted=True,
            **kwargs,
        )

        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        else:
            raise ValueError(f"Unknown scale_activation: {scale_activation}")

        # Add last distribution decoder layers
        self.px_scale_decoder = nn.Sequential(nn.Linear(funnel_out_dim, n_output), px_scale_activation)
        self.px_r_decoder = nn.Linear(funnel_out_dim, n_output)
        self.px_dropout_decoder = nn.Linear(funnel_out_dim, n_output)
        
        # FiLM layers: covariates -> scale/shift for decoder hidden dim. FiLM itself (use_film) was
        # removed, but the layers are still CONSTRUCTED: they are in every checkpoint's state dict and
        # their init advances the RNG stream that every later module draws from.
        if self.cat_dim > 0:
            self.film_scale = nn.Linear(self.cat_dim, funnel_out_dim)
            self.film_shift = nn.Linear(self.cat_dim, funnel_out_dim)
            # Init scale near zero so FiLM starts as identity
            if init_film_zero:
                nn.init.zeros_(self.film_scale.weight)
                nn.init.zeros_(self.film_scale.bias)
                nn.init.zeros_(self.film_shift.weight)
                nn.init.zeros_(self.film_shift.bias)
        else:
            self.film_scale = None
            self.film_shift = None
        
    def _get_cat_embedding(self, *cat_list: int) -> torch.Tensor | None:
        """Look up and concatenate per-covariate embeddings.

        Each active covariate has its own ``nn.Embedding`` (sized by category
        count), so indices are 0-based within the covariate — no shared-table
        offset needed. Embeddings are stored in ``n_cat_list`` order for the
        active covariates, so we consume them with a parallel iterator.
        """
        if not self.use_cat_emb or self.cat_embeddings is None:
            return None

        emb_list = []
        emb_iter = iter(self.cat_embeddings)
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat > 1:
                cat = cat.squeeze(-1) if cat.dim() == 2 and cat.size(-1) == 1 else cat
                emb_list.append(next(emb_iter)(cat))

        if emb_list:
            return torch.cat(emb_list, dim=-1)
        return None

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int, **kwargs):
        # Add categorical bias to z for stronger conditional decoding
        cat_emb = self._get_cat_embedding(*cat_list)
        if cat_emb is not None:
            z = torch.cat([z, cat_emb], dim=-1)
        # Decode z without covariates
        px = self.px_decoder(z, *cat_list)
        # Output heads
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


def _whiten_embedding(emb: torch.Tensor, mode: str = 'center') -> torch.Tensor:
    """Remove the common (mean) direction from an ``(n_classes, d)`` embedding matrix.

    Pretrained LLM/text-encoder embeddings sit on a mean-shifted manifold; subtracting the per-dim
    mean removes that shared direction (anisotropy in raw cosine) and leaves the covariance as is.
    Only ``mode='center'`` remains ('standardize' and 'zca' were removed).
    """
    if mode != 'center':
        raise ValueError(f"Unknown whiten mode: {mode!r} (only 'center' is supported)")
    return emb - emb.mean(dim=0, keepdim=True)


def text_atom_init(text: torch.Tensor, dim: int, target_norm: float,
                   fallback: torch.Tensor, whiten: float = 0.0) -> torch.Tensor:
    """PCA of the text bank, rescaled to the random init's row norm.

    WHITENING IS OFF BY DEFAULT, and the measurement is why. `whiten` is the exponent alpha in
    S**(1-alpha): alpha=0 keeps the text spectrum (a pure rotation + truncation, so pairwise text
    geometry is preserved exactly), alpha=1 fully whitens. Whitening was the original proposal --
    raw text is anisotropic, so a few PCs carry most of the variance and a plain PCA hands the
    atoms coarse "transcription factor"-style splits. But flattening the spectrum 1503x left the
    init correlated with the text bank at spearman +0.093 (expert/analysis/test_atom_init.py):
    the text bank's geometry essentially IS its top PCs, so equalising them removes the prior
    itself. Whitening and "keep the prior" are in direct conflict; alpha=0 is the honest text init.

    Anisotropy also does not need pre-correction on the gradient side: `target_norm` already sets
    how fast PA can rotate every atom, uniformly.

    WHY RESCALE. PA is pure cosine (`F.normalize` on both sides) and its gradient
    (I - v v^T)g/||a|| is ORTHOGONAL to v, so the loss can never change ||a||: whatever norm the
    atoms start at is the norm they keep, and the norm sets the rotation RATE via the 1/||a||
    factor. Matching `target_norm` (= 0.02*sqrt(d), the random init) makes the text init exactly as
    easy to reshape as the random one, so the arm tests the DIRECTION prior and nothing else.

    Components beyond the text bank's rank fall back to the random draw, so a latent wider than
    the text rank is padded rather than left degenerate.
    """
    X = text.float()
    X = X - X.mean(0, keepdim=True)
    q = min(dim, X.shape[0], X.shape[1])
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    keep = int((S > (S.max() * 1e-6)).sum().item())
    k = min(q, keep)
    a = float(min(max(whiten, 0.0), 1.0))
    # alpha=0 -> U*S (exact PCA scores, text geometry intact); alpha=1 -> U*sqrt(n-1) (whitened)
    W = U[:, :k] * (S[:k] ** (1.0 - a)) * (math.sqrt(max(X.shape[0] - 1, 1)) ** a)
    out = fallback.clone()
    out[:, :k] = W
    out = out / out.norm(dim=1, keepdim=True).clamp_min(1e-12) * float(target_norm)
    return out.to(fallback.dtype)


class ClassEmbedding(nn.Module):
    """
    Provides class embeddings from a pre-trained embedding matrix (cls_emb), optionally replaced by a
    learnable random-init (or text-PCA-init) atom table, followed by an identity or linear adapter.

    Forward() returns embeddings only for requested class indices. Every __init__ parameter is kept
    (checkpoints replay them); those of removed mechanisms are accepted only at their off value.
    """

    def __init__(
        self,
        pretrained_emb: torch.Tensor | None = None,
        class_texts: dict | None = None,
        n_output: int | None = None,
        transformer_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str = "cpu",
        max_length: int = 256,
        batch_size: int = 64,
        freeze: bool = False,
        train_last_n_layers: int = 1,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_dropout: int = 0.1,
        ignore_texts: bool = False,
        n_prototypes: int = 1,
        use_null_proxy: bool = False,
        pca_init_emb: torch.Tensor | None = None,
        whiten_embedding: bool = True,
        whiten_mode: str = 'center',
        adapter_mode: str = 'linear',      # 'linear' (default) | 'identity' (NO adapter: cosine the
                                           # bank rows directly, needs n_shared == panel width)
                                           # | 'residual' (frozen PCA base + low-rank)
                                           # | 'gated' (per-class softmax over per-source linear adapters)
        extra_src_emb: list | None = None,   # gated: extra prior banks (C, d_s), already label-aligned+centered
        extra_src_present: torch.Tensor | None = None,  # gated: (C, n_src) 0/1 coverage mask, source 0 first
        gate_temp: float = 1.0,            # gated: softmax temperature (<1 sharpens toward one source)
        residual_rank: int = 64,           # rank of the learnable residual (adapter_mode='residual')
        residual_alpha: float = 1.0,       # scale of the residual
        adapter_hidden: int = 1024,        # adapter_mode='mlp': hidden width of the free MLP head
        adapter_depth: int = 2,            # adapter_mode='mlp': number of Linear layers (>=2 => one hidden)
        adapter_dropout: float = 0.1,      # adapter_mode='mlp': dropout between MLP layers (high by design)
        grad_diffuse_beta: float = 0.0,    # >0: diffuse proxy gradients over the kNN so neighbors move together
        grad_diffuse_knn: int = 16,        # candidate-neighbor count for the diffusion graph
        grad_diffuse_mode: str = 'uniform',  # 'uniform' (fixed-k, equal wt) | 'similarity' (cos-weighted, self-normalized)
        grad_diffuse_tau: float = 0.5,     # similarity mode: drop edges below this cosine (loners -> self-loop, no dilution)
        grad_diffuse_power: float = 1.0,   # similarity mode: sharpen weights  w = relu(cos - tau) ** power
        grad_diffuse_steps: int = 1,       # diffusion steps (>1 -> tight clusters approach a shared/component gradient)
        input_emb_dropout: float = 0.0,    # arm-b: feature dropout on the text embedding the adapter projects
        input_emb_noise: float = 0.0,      # arm-b: Gaussian noise std on the text embedding (regularize phi for zero-shot)
        adapter_pca_init: bool = True,     # True: init the linear adapter's first k dims to the text-PCA basis
        adapter_residual_base: bool = False,  # mlp only: out = PCA-init base(x) + alpha*mlp(x), mlp zero-init
        adapter_alpha_init: float = 1.0,      # initial weight on the MLP's nonlinear correction
                                           # (preserves text structure). False: pure xavier — let the adapter learn
                                           # f(text) from scratch (text structure has ~no MI with response).
        proxy_dim: int = 0,                # width of the LEARNABLE proxy table. 0 = keep the text
                                           # embedding width (adapter projects to n_shared);
                                           # n_shared = atoms live directly in z space, which is
                                           # what lets the EMA class centroids initialise them and
                                           # what makes adapter_mode='identity' legal.
        atom_mask: bool = False,           # per-atom sparse gate on LEARNABLE proxies
        bank_mode: bool = False,           # SOURCE THE CLASS EMBEDDING FROM THE ONLINE PSEUDOBULK BANK
                                           # instead of text. `pretrained_emb` then only fixes the
                                           # label ORDER and the adapter's in_features (= panel
                                           # width); its VALUES are never read, because encode()
                                           # returns encode_bank(). Text is not a training signal
                                           # in this mode -- re-attach it later as a regression onto
                                           # observed bank rows.
        extra_src_names: list | None = None,   # display names for extra_src_emb (struct metrics)
        atom_init_seed: int | None = 0,    # RNG for the random atom table, drawn with numpy so it
                                           # is independent of the ambient torch stream. Two runs
                                           # that differ only in a loss knob then share an atom
                                           # init, which makes row-wise comparison across them
                                           # VALID -- measured on BASE vs PULL (unseeded), median
                                           # cos(A_i, B_i) = +0.0008, i.e. no shared basis and
                                           # every per-atom cross-arm diagnostic is dead on
                                           # arrival. None restores the old torch-stream draw.
        atom_init: str = 'random',         # 'random' (randn*0.02) or 'text' (PCA of the text bank,
                                           # rescaled to the SAME row norm). Only meaningful with
                                           # random_init_proxies; the table stays learnable either
                                           # way and no loss pulls it back toward text.
        atom_init_whiten: float = 0.0,     # spectrum exponent for atom_init='text': 0 keeps text
                                           # geometry (default), 1 whitens. See text_atom_init --
                                           # whitening measured away 90% of the prior.
        random_init_proxies: bool = False, # Q3: align to LEARNED proxies from RANDOM init (NO text prior). Replaces
                                           # the frozen text bank with a LEARNABLE random-init embedding + whitening
                                           # off. Tests what a good prior (text) buys vs learning proxies from scratch.
    ):
        super().__init__()
        _removed = {
            'class_texts': class_texts is not None and not ignore_texts, 'bank_mode': bool(bank_mode),
            'n_prototypes': int(n_prototypes) > 1, 'use_null_proxy': bool(use_null_proxy),
            'pca_init_emb': pca_init_emb is not None,
            'adapter_mode': adapter_mode not in ('identity', 'linear'),
            'grad_diffuse_beta': float(grad_diffuse_beta) > 0.0,
            'input_emb_dropout': float(input_emb_dropout) > 0.0, 'input_emb_noise': float(input_emb_noise) > 0.0,
        }
        _on = [k for k, v in _removed.items() if v]
        if _on:
            raise ValueError(f'ClassEmbedding: removed option(s) set to a non-off value: {_on}')

        self.device = device
        self.class_texts = class_texts
        self.max_length = max_length
        self.batch_size = batch_size
        self.freeze = freeze
        self.use_prototypes = False
        self.n_prototypes = 1
        self.adapter_mode = adapter_mode
        self.random_init_proxies = bool(random_init_proxies)
        self.atom_mask = bool(atom_mask)

        assert pretrained_emb is not None, "Provide pretrained_emb."
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
        self.encoder = None
        self.tokenizer = None
        self._num_classes, self._emb_dim = pretrained_emb.shape

        # Q3: LEARNED proxies from RANDOM init (no text prior). Overwrite the frozen text bank with a learnable
        # random-init embedding and turn whitening off (no text structure to whiten). The adapter still trains on
        # top, so the aligned proxies = adapter(learnable random bank) — the model learns the class geometry from
        # the response signal alone, isolating what the text prior actually buys.
        if self.random_init_proxies:
            n_cls_r = int(self._num_classes)
            dim_r = int(self._emb_dim)
            if int(proxy_dim) > 0:
                dim_r = int(proxy_dim)
                self._emb_dim = dim_r
            elif adapter_mode == 'identity' and n_output is not None:
                # DERIVE IT. Under identity the adapter is nn.Identity, so the ONLY legal proxy
                # width is n_output -- any other value raises below. Requiring proxy_dim to be
                # stated as well makes n_shared and proxy_dim two names for one quantity that a
                # config can disagree with, which is exactly how `wide()` at n_shared=256 could
                # leave the atoms at 128 for any base that did not already carry the key.
                # There is no text prior to preserve here (the table is random init), so nothing
                # is lost by sizing it to z.
                dim_r = int(n_output)
                self._emb_dim = dim_r
            if adapter_mode == 'identity' and n_output is not None and dim_r != int(n_output):
                raise ValueError(
                    f"random_init_proxies with adapter_mode='identity' needs proxy_dim == "
                    f"n_shared, got proxy_dim={dim_r} vs n_shared={n_output}. Leave proxy_dim "
                    f"unset to size the atom table to n_shared automatically.")
            # KEEP THE TEXT before it is overwritten. The doubly-supported structure loss needs
            # e_p to gate which atom pairs may be pulled together; once pretrained_emb is replaced
            # by the random table the text is gone and the gate would silently degrade to
            # "pull every pair", which is the failure the gate exists to prevent.
            if pretrained_emb is not None:
                self.register_buffer('_atom_text', pretrained_emb.detach().clone())
                # MULTI-SUPPORT struct loss. Reuse the banks `class_emb_extra_paths` already
                # loads for the gated adapter: they arrive centered on COVERED rows only, scaled
                # to unit mean row-norm, and with a per-class present mask -- which is exactly
                # what a partial-coverage prior needs (text ~19.9k, DepMap ~18k, JUMP 8-12.6k).
                # They are consumed here regardless of adapter_mode, so the atom arms (identity
                # adapter) can use them without switching to 'gated'.
                # GATED REGISTRATION: buffers are only created when extras are supplied, so every
                # existing checkpoint still loads and the RNG stream is unshifted.
                # ATTRIBUTE NAMES, NOT TENSORS. `.to(device)` REPLACES a registered buffer with
                # a new tensor rather than mutating it in place, so a list holding the original
                # objects keeps CPU references alive after the module moves to GPU -- and the
                # struct loss then indexes a CPU tensor with GPU indices. Resolve by getattr at
                # loss time so the list can never go stale.
                _px = [('text', '_atom_text', None)]
                _xe = list(extra_src_emb or [])
                if _xe:
                    _pm = None if extra_src_present is None else torch.as_tensor(extra_src_present)
                    _nms = list(extra_src_names or []) or [f'prior{i}' for i in range(len(_xe))]
                    for _i, _M in enumerate(_xe):
                        _t = torch.as_tensor(_M, dtype=torch.float32).detach().clone()
                        self.register_buffer(f'_atom_prior_{_i}', _t)
                        # present column 0 is the primary bank, so extras start at 1
                        _vn = None
                        if _pm is not None and _pm.shape[1] > _i + 1:
                            _vn = f'_atom_prior_valid_{_i}'
                            self.register_buffer(_vn, (_pm[:, _i + 1] > 0))
                        _px.append((str(_nms[_i]) if _i < len(_nms) else f'prior{_i}',
                                    f'_atom_prior_{_i}', _vn))
                    logging.info(f"[Setup] Atom struct supports: {[n for n, _, _ in _px]} "
                                 f"(coverage "
                                 f"{['n/a' if v is None else f'{float(getattr(self, v).float().mean()):.1%}' for _, _, v in _px]})")
                self._atom_priors = _px
            rand_emb = torch.randn(n_cls_r, dim_r) * 0.02
            # DRAW FIRST, THEN OVERWRITE. Keeping the randn call means the arm differs from its
            # random-init twin in the atom table alone -- every downstream module still gets the
            # same RNG stream, so the comparison is not confounded by a shifted init everywhere.
            # The SEEDED draw follows the same rule: the torch call above is kept purely to
            # advance the stream, then its values are replaced. Drawing from numpy rather than
            # torch.manual_seed keeps this local to the atom table -- seeding torch here would
            # re-align every downstream module too, which is a much larger change than intended.
            if atom_init_seed is not None:
                _g = np.random.default_rng(int(atom_init_seed))
                rand_emb = torch.from_numpy(
                    (_g.standard_normal((n_cls_r, dim_r)) * 0.02).astype(np.float32))
                logging.info('[Setup] atom table seeded with numpy rng %d -- runs differing only '
                             'in a loss knob now share this init, so per-atom cross-arm '
                             'comparison is valid', int(atom_init_seed))
            self.atom_init = str(atom_init or 'random').lower()
            if self.atom_init == 'text':
                if pretrained_emb is None:
                    raise ValueError("atom_init='text' needs a text bank (pretrained_emb is None)")
                _tn = float(0.02 * math.sqrt(dim_r))
                rand_emb = text_atom_init(pretrained_emb[:n_cls_r], dim_r, _tn, rand_emb,
                                          whiten=float(atom_init_whiten))
                logging.info("[Setup] atom_init=TEXT: PCA of %s -> %d dims, whiten alpha=%.2f "
                             "(%s), rows rescaled to %.4f (== the random init's 0.02*sqrt(%d))",
                             tuple(pretrained_emb.shape), dim_r, float(atom_init_whiten),
                             'text geometry PRESERVED' if float(atom_init_whiten) == 0.0
                             else 'spectrum flattened -- this REMOVES the prior',
                             _tn, dim_r)
            self.embedding = nn.Embedding.from_pretrained(rand_emb, freeze=False)  # LEARNABLE proxies
            # SPARSE GATE on the learnable proxies. Disjoint atom supports are what make a SUM
            # identifiable -- OMP recovers k atoms when their supports do not overlap, which
            # matters far more than dictionary coherence (measured: coherence ANTI-correlates
            # with both@2 across our arms). sigmoid(0)=0.5 starts the gate half-open; a closed
            # init would starve the atoms of gradient exactly when they most need it.
            self.atom_mask_logit = (nn.Parameter(torch.zeros_like(rand_emb))
                                    if self.atom_mask else None)
            self.encoder = None
            self.tokenizer = None
            pretrained_emb = rand_emb.detach()  # adapter pca_source (set adapter_pca_init=false in the config)
            whiten_embedding = False
            logging.info(f"[Setup] Q3 random-init LEARNABLE class proxies {tuple(rand_emb.shape)} "
                         f"(no text prior, whitening off).")

        # Whiten pretrained embeddings before they're consumed downstream
        # (forward lookup + adapter PCA init + adapter_geom anchors). Pretrained
        # LLM embeddings sit on an anisotropic, mean-shifted manifold; that
        # mismatches the VAE prior and gives the adapter free directions to
        # distort. Whitening once at init fixes both.
        if whiten_embedding:
            pretrained_emb = _whiten_embedding(pretrained_emb, mode=whiten_mode)
            # Rebuild the frozen lookup from the whitened tensor.
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
            logging.info(
                f"[Setup] Whitened class embedding (mode={whiten_mode}, "
                f"shape={tuple(pretrained_emb.shape)})."
            )

        # Add final output adapter if n output is not None
        if n_output is not None:
            pca_source = pretrained_emb
            # Centering for PCA — when whitening already centered
            # pca_source above, this is a no-op, so skip it. When whitening is off, SVD on uncentered
            # data biases the principal direction, so we still center here.
            if whiten_embedding:
                emb_centered = pca_source
            else:
                emb_centered = pca_source - pca_source.mean(dim=0, keepdim=True)
            # Init on centered embedding PCA
            _, _, Vt = torch.linalg.svd(emb_centered, full_matrices=False)
            k = min(n_output, Vt.shape[0])
            # Create adapter
            if adapter_mode == 'identity':
                # NO ADAPTER. z_shared is cosined DIRECTLY against the bank rows in gene space, so
                # the shared space IS the panel. Removes the learned projection that compressed a
                # ~47-effective-dim bank into 12-30 effective dims, and makes the target an actual
                # measured response profile rather than an arbitrary learned basis. Requires
                # n_shared == panel width; anything else is a silent dimension mismatch.
                if int(n_output) != int(self._emb_dim):
                    raise ValueError(
                        f"adapter_mode='identity' needs n_shared == the source width, got "
                        f"n_shared={n_output} vs source={self._emb_dim}. Set n_shared to the "
                        f"pbulk panel width (bank mode) so the cosine is taken in gene space.")
                self.adapter = torch.nn.Identity()
            else:
                # adapter_mode='linear': one linear map text -> n_output.
                self.adapter = nn.Linear(self._emb_dim, n_output, bias=False)
                # Initialize adapter on random except for pca dimensions (unless adapter_pca_init=False,
                # then pure xavier so the adapter learns f(text) without being pinned to text structure).
                with torch.no_grad():
                    nn.init.xavier_uniform_(self.adapter.weight)
                    if adapter_pca_init:
                        self.adapter.weight[:k].copy_(Vt[:k])
                    else:
                        logging.info("[Setup] adapter_pca_init=False: linear adapter left at xavier init "
                                     "(no text-PCA pinning) — learns f(text) from scratch.")

            # Verify structure is preserved
            with torch.no_grad():
                proj = self.adapter(pretrained_emb)
                sim_orig = F.cosine_similarity(pretrained_emb[:10].unsqueeze(1), pretrained_emb[10:20].unsqueeze(0), dim=-1)
                sim_proj = F.cosine_similarity(proj[:10].unsqueeze(1), proj[10:20].unsqueeze(0), dim=-1)
                logging.info(f"[Setup] Class embedding similarity preserved: {np.corrcoef(sim_orig.detach().cpu().flatten().numpy(), sim_proj.detach().cpu().flatten().numpy())[0,1]:.4f}")
            # Snapshot initial adapter weight for drift diagnostics.
            # Non-persistent: PCA SVD is deterministic given the same
            # pretrained_emb, so __init__ recreates the same reference on
            # every load. Skipping state_dict avoids breaking older
            # checkpoints that predate this buffer.
            # adapter_mode='identity' is an nn.Identity with no parameters at all, so there is
            # no weight to snapshot and no drift to measure -- adapter_drift()
            # returns None rather than crashing on a missing attribute.
            if hasattr(self.adapter, 'weight'):
                self.register_buffer(
                    '_adapter_init_weight',
                    self.adapter.weight.detach().clone(),
                    persistent=False,
                )
            # Set class embedding output dimension to adapter output
            self._emb_dim = n_output
            self.use_adapter = True
        else:
            self.adapter = nn.Identity()
            self.use_adapter = False
        # Move module to device
        self.to(device)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Drop legacy ``_adapter_init_weight`` from older checkpoints — it was
        # registered persistent before being switched to non-persistent. Older
        # checkpoints carry the key and would otherwise raise "Unexpected
        # key(s)" under strict=True.
        legacy_key = prefix + '_adapter_init_weight'
        state_dict.pop(legacy_key, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @torch.no_grad()
    def adapter_drift(self) -> dict[str, torch.Tensor] | None:
        # Per-row cosine to PCA-init = direction preservation per output dim;
        # frobenius ratio = overall magnitude change. Returns None when no
        # adapter (Identity case) so caller can skip logging.
        if not self.use_adapter or not hasattr(self, '_adapter_init_weight'):
            return None
        W = self.adapter.weight.detach()
        W0 = self._adapter_init_weight
        cos = F.cosine_similarity(W, W0, dim=-1)
        frob_ratio = (W - W0).norm() / W0.norm().clamp(min=1e-8)
        return {
            'cos_mean': cos.mean(),
            'cos_min': cos.min(),
            'frob_ratio': frob_ratio,
        }

    # -------------------------------------------------------
    @property
    def shape(self):
        return (self._num_classes, self._emb_dim)

    # -------------------------------------------------------
    def encode(self, class_indices: torch.Tensor | list | None = None) -> torch.Tensor:
        """
        Return raw encoder embeddings **before** the adapter projection.

        Useful for caching: store the detached result once per epoch and
        pass it through ``self.adapter`` fresh on every forward step so
        that the adapter remains in the computation graph.
        """
        embeddings = self.embedding.weight
        # SPARSE GATE, applied here so every consumer sees the SAME atoms: logits, the clip
        # target, the in-train recall readout and the OMP dictionary are all encode(). Gating
        # anywhere else would let the reported geometry drift from the decoded one.
        if getattr(self, "atom_mask_logit", None) is not None:
            embeddings = embeddings * torch.sigmoid(self.atom_mask_logit)
        if class_indices is not None:
            embeddings = embeddings[class_indices]
        return embeddings

    def forward(self, class_indices: torch.Tensor | list | None = None):
        """
        Compute class embeddings = adapter(encode()).

        Args:
            class_indices: Optional indices to return. If None → all classes.
        Returns:
            Tensor of shape (K, emb_dim)
        """
        return self.adapter(self.encode(class_indices))


class EmbeddingAligner(nn.Module):
    """
    Projects the latent (and optionally the class embeddings) into the shared alignment space.
    Logits are computed by the caller; ``return_logits=True`` was removed.
    """

    def __init__(
        self,
        n_input: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        logits: bool = True,
        funnel: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation_fn: nn.Module = nn.GELU,
        class_embed_dim: int = 128,
        shared_projection_dim: int | None = None,
        skip_projection: bool = True,
        temperature: float = 0.1,
        min_temperature: float = 0.02,
        use_learnable_temperature: bool = False,
        mode: Literal["latent_to_class", "class_to_latent", "shared"] = "shared",
        use_cross_attention: bool = False,
        linear_z_proj: bool = True,
        linear_c_proj: bool = True,
        noise_sigma: float | None = None,
        **kwargs,
    ):
        super().__init__()
        if use_learnable_temperature or use_cross_attention or mode != "shared":
            raise ValueError(f'EmbeddingAligner: removed option set (use_learnable_temperature='
                             f'{use_learnable_temperature}, use_cross_attention={use_cross_attention}, '
                             f'mode={mode!r}); only mode="shared" with a fixed temperature is supported.')
        self.logits = logits
        self.class_embed_dim = class_embed_dim
        self.dropout_rate = dropout_rate
        self.funnel = funnel
        self.mode = mode
        self._temperature = temperature
        self._min_temperature = min_temperature
        self.use_learnable_temperature = use_learnable_temperature
        self.use_cross_attention = use_cross_attention
        self.noise_sigma = noise_sigma
        # ---- Helper for building projections ---- #
        def build_projection(n_in: int, n_out: int, linear: bool = False) -> nn.Module:
            """Identity when the dims already match (skip_projection), else a bias-free linear map."""
            if skip_projection and n_in == n_out:
                return nn.Identity()
            if linear:
                return nn.Linear(n_in, n_out, bias=False)
            raise ValueError('EmbeddingAligner: non-linear projections were removed '
                             '(set linear_z_proj / linear_c_proj, or matching dims with skip_projection).')

        # Latent --> shared dim <-- Class embedding space
        if shared_projection_dim is None:
            shared_projection_dim = min(n_input, class_embed_dim)
        # Create z projection
        self.latent_projection = build_projection(n_input, shared_projection_dim, linear=linear_z_proj)
        # Create class projection
        self.class_projection = build_projection(class_embed_dim, shared_projection_dim, linear=linear_c_proj)
        # Update output dimension
        self.n_output = shared_projection_dim

        # ---- Dropout ---- #
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cdropout = nn.Dropout(p=dropout_rate)

    @property
    def temperature(self) -> torch.Tensor:
        # Fixed / annealed T
        return self._temperature

    def project_classes(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """(n_labels, d) class bank in the shared space; a bank already of width d is pre-projected and passes."""
        if cls_emb.size(1) == self.n_output:
            return cls_emb
        return self.class_projection(cls_emb)

    def forward(self, x: torch.Tensor, cls_emb: torch.Tensor = None, return_logits: bool = False, T: float | None = None):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
        cls_emb : Optional[Tensor], shape (n_labels, embed_dim)
        cls_proj : Optional[Tensor], shape (n_labels, shared_dim) — pre-projected class embeddings, skips class_projection if provided
        """
        # Apply latent projection
        z = self.latent_projection(x)               # (batch, d)
        # Class-side projection. cls_emb=None => the caller only needs z_shared and
        # supplies the class bank itself (e.g. the GNN class-encoder), so skip the
        # class_projection forward entirely instead of computing a discarded tensor.
        c = None if cls_emb is None else self.project_classes(cls_emb.to(x.device))
        # Add some dropout to z before alignment
        if self.training and self.dropout_rate > 0:
            z = self.dropout(z)

        if return_logits:
            raise ValueError('EmbeddingAligner.forward(return_logits=True) was removed; '
                             'the caller computes the logits.')
        return {
            MODULE_KEYS.Z_SHARED_KEY: z,
            MODULE_KEYS.CLS_PROJ_KEY: c
        }
