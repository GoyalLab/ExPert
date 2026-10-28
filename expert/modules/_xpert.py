# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

from typing import Iterable

from expert.modules._base import (
    Encoder, 
    DecoderSCVI, 
    EmbeddingAligner,
    ClassEmbedding,
    EfficiencyHead
)
import expert.utils.io as io
import expert.utils.legacy as legacy
from expert.utils.constants import MODULE_KEYS, REGISTRY_KEYS, LOSS_KEYS, PREDICTION_KEYS
import expert.utils.common as co
from expert.modules._prototype_scorer import PrototypeScorer
from expert.losses.proxy_anchor import proxy_anchor_loss
from expert.modules import _prior_atoms as _pa_prior

from typing import Iterable

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

from scvi.module.base import (
    LossOutput,
    auto_move_data,
)
from scvi.module._vae import VAE

from scvi.distributions import (
    NegativeBinomial,
    Normal,
    Poisson,
    ZeroInflatedNegativeBinomial,
)

from collections.abc import Callable
from typing import Literal

from torch.distributions import Distribution

import logging
log = logging.getLogger(__name__)


class XPert(VAE):
    """
    Adaption of scVI and scanVI models to predict Perturb-seq perturbations using context- and class-embedding data.
    """
    # PRIOR-ANCHORED WEAK ATOMS (see expert/modules/_prior_atoms.py): keys and defaults of `extra_prior_atom_kwargs`.
    # Pull atoms of WEAKLY-RESPONDING classes toward f(prior), fit on the STRONG ones. Where a
    # knockdown barely moves the cells, the atom is placed by noise and PA then drags the
    # encoder toward it all run; the prior is the better information in exactly that regime.
    # Off by default. NOTE the encoder is never touched directly -- the target is detached, so
    # the only path to z is "PA's target improved", which is the point (a prior with gradient
    # on z is the struct loss, and that cost the readout every time it was tried).
    _PRIOR_ATOM_DEFAULTS = {
        'weight': 0.0,
        'lam': 0.5,                           # bank fusion; 0.25-0.5 flat-optimal on both screens
        # ONE RATIO. The pulled set is the weakest `pull_q` fraction by |eff| rank and f() is fit
        # on ALL the rest -- disjoint by construction, so no atom both trains the map and is moved
        # by it. Replaces the old strong_q + w_zero_at pair, which I tied together at 0.5 and so
        # pulled 50% of classes: the offline gain lives in the weakest ~5% and is GONE by 50%
        # (+0.0514 at cutoff 5 -> +0.0020 n.s. at 50). Fitting on the other 90% also gives f() far
        # more data, which measurably helps -- the map is sample-size limited (zero-shot 9.20x at
        # 8,697 fit atoms vs 8.10x at 412).
        # 0.075 is MEASURED, not chosen. Sweeping the pull fraction on frozen H4/H8 atoms and
        # counting HITS GAINED on the pulled set (the right objective, since the training
        # version pays ~no distraction cost): the per-class gain falls faster than the class
        # count rises, so the product peaks at 0.075 (+63 hits) -- 0.05 gives +57, 0.10 gives
        # +59, 0.20 gives +54, 0.50 gives +22. Nothing is left on the table above 10%.
        'pull_q': 0.075,                      # fraction of classes pulled; f() fits the rest
        'start_epoch': 5,                     # before this the atoms are randn*0.02: nothing to fit
        'ramp_epochs': 5,                     # 0 -> full weight the moment f() first exists
        'refit_every': 1,                     # epochs. ~1s per refit vs a ~300s epoch, so there is
                                              # no reason to run a staler target than necessary.
                                              # Per STEP would ~3x the run to re-solve a problem
                                              # whose inputs moved negligibly -- and f() is fit on
                                              # the STRONG atoms, which this term barely touches
                                              # (w ~ 0 there), so its fit set is near-stationary.
        # RIDGE ONLY at a FIXED alpha -- no per-bank map selection. Every ridge setting beat every
        # MLP on both uses (zero-shot 8.50x vs 6.70x) and the MLP overfits; the selector could not
        # see the difference that mattered (2% on its own criterion, 27% on zero-shot). 0.5 is the
        # joint optimum: 91% of the best held-out top10 and 98% of the best zero-shot.
        'ridge_alpha': 0.5,
        'min_cells': 8,                       # train cells a class needs before its |eff| score
                                              # is trusted; below this it is left uncovered and
                                              # therefore never pulled
        'min_fit': 300,                       # strong atoms required before f() is fit at all
        # MAP FAMILY, set independently for the two consumers. Measured over 762 (bank x map)
        # configs under the readouts we optimise: soft-alpha OOD is FLAT across families (+62 for
        # the shipped ridge vs +69 for the best of 762, i.e. +0.0007 against a 0.0045 floor), but
        # placement of the MISSING rows is not -- every one of the top-20 zero-shot configs is a
        # graph map and ridge appears zero times (graph25 ~12.0x vs ridge 10.2x). The observed and
        # missing paths are independent, so they take independent maps. 'graph' also needs no
        # fitting: the prior kNN graph is static, so the target is a weighted mean of CURRENT atom
        # positions and there is nothing to refit.
        'map': 'ridge',                       # observed/weak atoms: 'ridge' | 'graph'
        'missing_map': '',                    # untrained rows; '' -> same as 'map'
        'missing_sim_pow': 4.0,               # untrained rows' graph weight = cos^p (4 = measured best)
        'graph_k': 25,                        # neighbours when either map is 'graph'
        # GRAPH-MAP REFINEMENTS (both inert at their defaults, so every existing arm is unchanged):
        #  loo    -- leave-one-out instead of the fold split when the PULL uses the graph map. A
        #            neighbour mean has no parameters to leak; the only leak is an atom being its
        #            own neighbour, which is excluded exactly. Ridge keeps cross-fitting.
        #  ev_pow -- neighbour weight = sim * evidence_rank^ev_pow, so rare genes are pulled toward
        #            their WELL-DETERMINED prior neighbours instead of each other's noise.
        'graph_loo': False,
        'graph_ev_pow': 0.0,
        # EXTRA PA NEGATIVES from f(missing). Filling untrained rows makes them PLAUSIBLE, and
        # measured offline that costs -0.0401 top10 on the TRAINED readout while the same number of
        # RANDOM rows costs exactly 0.0000 -- random directions are never competitive, f() ones are.
        # A model that ships f(missing) must be taught to reject them. Sampled per step, detached.
        'neg_weight': 0.0,                    # 0 = off
        'neg_k': 256,                         # untrained rows sampled per step
        # RUNAWAY GUARD. This term is a positive feedback loop by construction: it pulls atoms
        # together, f() is REFIT on those atoms, and the refit then predicts even more similar
        # targets. MEASURED at weight 200: mean off-diagonal cosine among the fit atoms went
        # 0.0001 -> 0.0150 in 15 epochs, roughly doubling every 2, against H4's FINAL value of
        # 0.0013 -- i.e. the atom table was collapsing, and the readout is cosine retrieval over
        # exactly those atoms. At weight 10 it stayed flat at 0.0003, so the loop is
        # dose-dependent and a ceiling is the right shape of guard. Above this, the pull is
        # switched OFF (state 7) rather than allowed to eat the dictionary.
        'coh_max': 0.01,                      # 0 disables the guard
        'cos_floor': 0.02,                    # per-atom agreement floor; -1 disables
        # GATE: 'hard' pulls the weakest pull_q fraction of the evidence ranking with a uniform weight.
        # 'eff' is the LEGACY NAME of 'hard' and behaves identically (v0.1.0: gate 'eff' + evidence
        # 'nqxnds' = a hard gate on cells x datasets).
        'gate': 'hard',
        # EVIDENCE = the ranking the gate cuts: 'nqxnds' = train cells x datasets (atom_nqxnds_qual);
        # 'ntrain' / 'eff' rank by the per-dataset |eff| percentile (atom_eff_qual).
        'evidence': 'ntrain',
        'folds': 5,                           # cross-fit folds over atoms; 1 = the old split
    }
    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_batch: int,
        n_hidden: int = 256,
        n_latent: int = 128,
        n_layers: int = 2,
        n_shared: int | None = None,
        cls_emb: torch.Tensor | None = None,
        ctrl_gene_means: torch.Tensor | None = None,   # per-(dataset, gene) CONTROL mean -> encoder input centering
        ctrl_gene_stds: torch.Tensor | None = None,    # per-(dataset, gene) CONTROL std  -> center_mode='ctrl_z'
        # PRIOR-ANCHORED WEAK ATOMS: every setting, the weights included, in one dict; keys and defaults (with
        # the measurements behind them) in XPert._PRIOR_ATOM_DEFAULTS. Off unless `weight` or `neg_weight` > 0.
        extra_prior_atom_kwargs: dict | None = None,
        # PROXY ANCHOR (Kim et al., CVPR 2020) over the atom table -- see expert/losses/proxy_anchor.py
        # for why a saturating hinge is the right shape here and an InfoNCE/CE is not.
        # ALL SETTINGS LIVE IN ONE DICT, matching extra_prior_atom_kwargs / extra_cls_emb_kwargs. Keys:
        #   weight, alpha, delta, neg_only, z_grad, center, center_atoms
        extra_proxy_anchor_kwargs: dict | None = None,
        cls_head_dropout: float = 0.0,                 # dropout on the HEAD INPUT (the bare-Linear head has no hidden block)
        ctrl_class_idx: int | None = None,
        use_adapter: bool = False,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.2,
        dispersion: Literal['gene', 'gene-batch', 'gene-label', 'gene-cell'] = 'gene',
        log_variational: bool = True,
        gene_likelihood: Literal['zinb', 'nb', 'poisson', 'normal'] = 'zinb',
        latent_distribution: Literal['normal', 'ln'] = 'normal',
        deterministic: bool = False,
        encode_covariates: bool = False,
        decode_covariates: bool = True,
        deeply_inject_covariates: bool = False,
        use_batch_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'none',
        use_layer_norm: Literal['encoder', 'decoder', 'none', 'both'] = 'both',
        var_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        min_kl: float | None = 1.0,
        reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        non_elbo_reduction: Literal['mean', 'sum', 'batchmean'] = 'mean',
        scale_elbo: bool = False,
        scale_mean_kl: bool = False,
        model_efficiency: bool = True,
        use_efficiency_pa_weight: bool = True,   # per-cell efficiency weight on the PA loss
        efficiency_rank_target: bool = True,     # regress the per-dataset RANK, not raw eff
        efficiency_head_linear: bool = False,    # single Linear; see EfficiencyHead docstring

        efficiency_clip_floor: float = 0.1,
        extra_encoder_kwargs: dict | None = {},
        extra_decoder_kwargs: dict | None = {},
        extra_aligner_kwargs: dict | None = {},
        use_cls_head: bool = False,   # Ablation: plain supervised softmax head on z_shared INSTEAD of prior alignment
        cls_head_cosine: bool = True,      # cosine (normalised) head; the only head implementation
        cls_head_arc_scale: float = 30.0,  # logit scale for the cosine head (ArcFace's `s`)
        extra_cls_emb_kwargs: dict | None = {},
        extra_clip_kwargs: dict = {},
        invariant_align: bool = False,
        align_ema: float = 0.05,
        **vae_kwargs
    ):
        # Initialize base model class

        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            encode_covariates=encode_covariates,
            deeply_inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=False,
            use_observed_lib_size=True,
            var_activation=var_activation,
            **vae_kwargs
        )
        # Setup shared dimensions
        self.n_shared = n_shared if n_shared is not None else n_latent
        # Setup control index
        self.ctrl_class_idx = ctrl_class_idx
        # Update labels, we don't predict control as a class
        self.n_labels = n_labels if self.ctrl_class_idx is None else n_labels - 1

        # Setup extra basic vae args
        self.min_kl = min_kl

        # Remove control embedding if it exists
        if self.ctrl_class_idx is not None and cls_emb is not None:
            no_ctrl_emb_mask = torch.arange(cls_emb.size(0)) != self.ctrl_class_idx
            cls_emb = cls_emb[no_ctrl_emb_mask]

        # Set reduction metrics
        if reduction not in ['batchmean', 'mean']:
            raise ValueError(f'Invalid reduction for elbo loss metrics: {reduction}, choose either "batchmean", or "mean".')
        if non_elbo_reduction not in ['batchmean', 'mean', 'sum']:
            raise ValueError(f'Invalid reduction for extra loss metrics: {non_elbo_reduction}, choose either "batchmean", or "mean".')
        self.reduction = reduction
        self.non_elbo_reduction = non_elbo_reduction
        self.scale_elbo = scale_elbo
        self.scale_mean_kl = scale_mean_kl
        

        # Whether to model the efficiency score (only available if score is given)
        self.model_efficiency = model_efficiency
        
        # Setup external embeddings
        self.use_adapter = use_adapter
        if use_adapter:
            extra_cls_emb_kwargs['n_output'] = self.n_shared
        # Setup class embedding
        if cls_emb is None:
            raise ValueError('Provide pre-trained gene embeddings (cls_emb).')
        self.cls_emb = ClassEmbedding(
            pretrained_emb=cls_emb,
            **extra_cls_emb_kwargs
        )
        self.n_cls = self.cls_emb.shape[0]
        self.n_cls_dim = self.cls_emb.shape[-1]

        # Plain attributes kept for the training plan, which reads cls2pw / cls2module (getattr without
        # a default) and use_learnable_hierarchy_temperatures directly. Hierarchical labels are gone.
        self.cls2module = None
        self.cls2pw = None
        self.use_learnable_hierarchy_temperatures = False
        # No context embedding; n_ctx_dim is read by ExPert.validate_ctx_emb.
        self.n_ctx_dim = 0

        # Setup normalizations for en- and decoder
        use_batch_norm_encoder = use_batch_norm == 'encoder' or use_batch_norm == 'both'
        use_batch_norm_decoder = use_batch_norm == 'decoder' or use_batch_norm == 'both'
        use_layer_norm_encoder = use_layer_norm == 'encoder' or use_layer_norm == 'both'
        use_layer_norm_decoder = use_layer_norm == 'decoder' or use_layer_norm == 'both'
        # Set covaritates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)

        # ----- Setup encoder module -----
        encoder_cat_list = cat_list if encode_covariates else None
        # AE mode: when True the encoder returns the posterior mean (qz.loc)
        # instead of a reparameterized sample, so the latent is deterministic
        # everywhere (reconstruction included). Pair with kl_weight=0 and the
        # plan's use_posterior_mean='both' (checked at training-plan setup).
        self.deterministic = deterministic
        self.default_encoder_kwargs = {
            'n_input': n_input,
            'n_output': n_latent,
            'n_hidden': n_hidden,
            'n_cat_list': encoder_cat_list,
            'dropout_rate': dropout_rate,
            'use_batch_norm': use_batch_norm_encoder,
            'use_layer_norm': use_layer_norm_encoder,
            'var_activation': var_activation,
            'n_batch': n_batch,          # sizes the per-dataset gene-mean buffer
            'ctrl_gene_means': ctrl_gene_means,   # control baseline for center_mode='ctrl' (None -> zeros)
            'ctrl_gene_stds': ctrl_gene_stds,     # control scale for center_mode='ctrl_z' (None -> ones = plain centering)
            'n_labels': n_labels,
            'deterministic': deterministic,
        }
        # Set base decoder input dimension
        base_decoder_input_dim = n_latent
        extra_encoder_kwargs.update(self.default_encoder_kwargs)
        # Create encoder class
        self.z_encoder = Encoder(**extra_encoder_kwargs)

        # ----- Setup decoder module -----
        # Covariate params
        self.decode_covariates = decode_covariates
        # Whether to decode covariates
        n_input_decoder = base_decoder_input_dim + n_continuous_cov * decode_covariates
        # Set decoder input to global if not None
        decoder_cat_list = cat_list if decode_covariates else None
        # Setup aligner
        self.aligner = EmbeddingAligner(
            n_input=n_latent,
            shared_projection_dim=self.n_shared,
            class_embed_dim=self.n_cls_dim,
            **extra_aligner_kwargs
        )
        
        # Init raw x (scvi) decoder
        base_decoder_kwargs = {
            'n_input': n_input_decoder,
            'n_output': n_input,
            'n_cat_list': decoder_cat_list,
            'n_hidden': n_hidden,
            'use_batch_norm': use_batch_norm_decoder, 
            'use_layer_norm': use_layer_norm_decoder,
            'scale_activation': 'softmax',
            'inject_covariates': deeply_inject_covariates,
        }
        base_decoder_kwargs.update(extra_decoder_kwargs)
        self.decoder = DecoderSCVI(**base_decoder_kwargs)
        
        # Prototype scorer: cos(z_shared, prototypes) / T -- the A-head logits and the diagnostic proto_ce.
        # ``extra_clip_kwargs`` keeps its (config / checkpoint) name; the scorer takes label_smoothing, noise_std
        # and reduction from it and accepts the removed CLIP-alignment knobs only at their off values.
        extra_clip_kwargs['reduction'] = self.non_elbo_reduction
        self.proto_scorer = PrototypeScorer(**extra_clip_kwargs)

        # Optional supervised COSINE head on z_shared (cls_head_w): scaled cosine logits, trained by CE in
        # loss() and published under CLS_HEAD_LOGITS_KEY. Built on n_shared (NOT n_latent) so it scores the
        # same space as the prototype/atom path. cls_head_w is constructed whenever cls_head_cosine is set
        # (independent of use_cls_head) -- that is the construction every checkpoint was trained with.
        self.use_cls_head = bool(use_cls_head)
        self.cls_head_cosine = bool(cls_head_cosine)
        if not self.cls_head_cosine:
            raise ValueError("cls_head_cosine=False (the Linear Classifier head) is no longer supported; "
                             "the cosine head is the only head implementation.")
        self.cls_head_arc_scale = float(cls_head_arc_scale)
        _head_in = self.n_shared if self.use_cls_head else n_latent
        # DROPOUT ON THE HEAD INPUT (see _head_input).
        self.cls_head_dropout = float(cls_head_dropout or 0.0)
        self.cls_head_w = torch.nn.Parameter(torch.empty(self.n_labels, _head_in))
        torch.nn.init.xavier_uniform_(self.cls_head_w)

        # Setup efficiency head if enabled. A single head predicts a
        # dataset-relative efficiency; the responder gate is derived from it
        # (responder = high efficiency) rather than a separate classifier.
        # SET BEFORE THE HEAD IS BUILT: the other efficiency_* flags are assigned further down,
        # after this block, so reading one of those here would always see its default.
        self.efficiency_head_linear = bool(efficiency_head_linear)
        if self.model_efficiency:
            self.efficiency_head = EfficiencyHead(
                self.n_shared, linear=self.efficiency_head_linear)
            # Responder gate from standardized efficiency: responder if the
            # per-dataset z-scored eff_hat exceeds ``responder_thr`` SDs above
            # the dataset mean. ``responder_steepness`` controls the soft
            # sigmoid sharpness around that threshold.
            self.responder_thr = 0.5
            self.responder_steepness = 2.0

        self.use_efficiency_pa_weight = bool(use_efficiency_pa_weight)
        self.efficiency_rank_target = bool(efficiency_rank_target)
        self.efficiency_clip_floor = float(efficiency_clip_floor)
        # ---- prior-anchored weak atoms ----
        _pk = dict(extra_prior_atom_kwargs or {})
        _unknown = set(_pk) - set(self._PRIOR_ATOM_DEFAULTS)
        if _unknown:
            raise ValueError(f'unknown extra_prior_atom_kwargs: {sorted(_unknown)}; '
                             f'valid keys are {sorted(self._PRIOR_ATOM_DEFAULTS)}')
        # each key lands on `self.prior_atom_<key>`, cast to its default's type
        for _k, _d in self._PRIOR_ATOM_DEFAULTS.items():
            _v = _pk.get(_k, _d)
            setattr(self, f'prior_atom_{_k}', type(_d)(_v if _v is not None else _d))
        # 'eff' -> 'hard' is a pure rename (v0.1.0 says 'eff'). The agree/soft/bayes gates were removed.
        _gate_cfg = self.prior_atom_gate
        _gate = {'eff': 'hard'}.get(_gate_cfg.lower(), _gate_cfg.lower())
        if _gate != 'hard':
            raise ValueError(f"extra_prior_atom_kwargs gate must be 'hard' (legacy 'eff'), got {_gate_cfg!r}; "
                             f"the agree/soft/bayes gates were removed")
        self.prior_atom_gate = _gate
        self.prior_atom_evidence = self.prior_atom_evidence.lower()
        if self.prior_atom_evidence not in ('ntrain', 'eff', 'nqxnds'):
            raise ValueError(f"extra_prior_atom_kwargs evidence must be 'nqxnds', 'ntrain' or 'eff', got "
                             f"{self.prior_atom_evidence!r}; the fisher / nqxnds_q evidence sources were removed")
        if self.prior_atom_weight > 0:
            log.info(f'[prior-atoms] gate={self.prior_atom_gate} (config {_gate_cfg!r}), '
                     f'evidence={self.prior_atom_evidence} -> ranked by {self.prior_atom_ranking()}')
        # HEAD TRAINING RECORD (plain attributes, never buffers). Configs can say use_cls_head=True while
        # the head loss weight is 0 for the whole run (every PA arm), so "does the model have a head" and
        # "was the head trained" are different questions. The model copies these into
        # `cls_head_training_` after train() so predict(head='auto') can tell them apart.
        self._cls_head_steps = 0
        self._cls_head_w_max = 0.0
        # f(missing) FILL for the untrained vocabulary rows (`prior_missing_T`). NOT registered here:
        # store_prior_missing() adds it at the end of training (or post-hoc via
        # ExPert.add_prior_missing_fill), and the pre-hook registers it at load only when the
        # checkpoint carries the key -- so older checkpoints load unchanged and the RNG is untouched.
        self._register_load_state_dict_pre_hook(self._prior_missing_load_hook)
        # HUBNESS CORRECTION (`atom_hub_pen`, (2, n_vocab): row 0 = popularity of +atom, row 1 = of -atom).
        # Same gating as prior_missing_T: never registered here, added post hoc by store_hub_correction()
        # and re-registered at load only when the checkpoint carries it. use_hub_correction is a plain
        # attribute so predict() can toggle it per call.
        self._register_load_state_dict_pre_hook(self._hub_pen_load_hook)
        self.use_hub_correction = True
        # READOUT NOISE CALIBRATION (`readout_noise`, (4,): log-noise95 intercept/slope vs log n_cells for
        # ||z|| and for the atom-span norm). Same gating: added post hoc by ExPert.add_readout_calibration()
        # and re-registered at load only when the checkpoint carries it.
        self._register_load_state_dict_pre_hook(self._readout_noise_load_hook)
        # EFFECT-SIZE READOUT (`effect_size_w`, (5,) over [eff_hat, max_cos, margin, p_top1, ||z_c||];
        # `effect_size_meta`, (4,) = [b, k, T, n_labels]): the trained efficiency head combined post hoc with the cosine
        # confidence of the call (ExPert.add_effect_head). Same gating: never registered here, re-registered at load
        # only when the checkpoint carries it.
        self._register_load_state_dict_pre_hook(self._effect_head_load_hook)
        # Pre-release checkpoints carry buffers of mechanisms this package no longer has (legacy.REMOVED_STATE).
        # Dropped in memory before the strict load; nothing on disk is ever rewritten.
        self._register_load_state_dict_pre_hook(self._legacy_state_hook)
        self._prior_atom_tripped = False
        self._prior_atom_fitter = None
        self._prior_atom_epoch = -1
        self._prior_atom_flips = 0
        # THE |eff| GATE IS COMPUTED AT SETUP, not accumulated: ExPert._compute_prior_atom_eff_gate
        # attaches `atom_eff_qual` from TRAIN cells only, exactly as the offline gate was built.
        self.atom_eff_qual = None
        self.atom_ntrain_qual = None
        _pak = dict(extra_proxy_anchor_kwargs or {})
        _DEF = {'weight': 0.0, 'alpha': 32.0, 'delta': 0.1, 'neg_only': False, 'z_grad': 1.0,
                'center': True, 'center_atoms': True}
        _unknown = set(_pak) - set(_DEF)
        if _unknown:
            raise ValueError(f'unknown extra_proxy_anchor_kwargs: {sorted(_unknown)}; '
                             f'valid keys are {sorted(_DEF)}')
        _g = lambda k: _pak.get(k, _DEF[k])                                    # noqa: E731
        # loss() falls back to this when the plan passes no proxy_anchor_weight schedule
        self.proxy_anchor_weight = float(_g('weight'))
        self.proxy_anchor_alpha = float(_g('alpha'))
        self.proxy_anchor_delta = float(_g('delta'))
        self.proxy_anchor_neg_only = bool(_g('neg_only'))
        self.proxy_anchor_z_grad = float(_g('z_grad'))
        self.proxy_anchor_center = bool(_g('center'))
        self.proxy_anchor_center_atoms = bool(_g('center_atoms'))
        # Per-class centroids in z_shared space. Never updated any more (the centre loss is gone), but the
        # buffer is in every checkpoint and anchors the module device for the prior-atom code.
        self.register_buffer(
            'class_centers',
            torch.zeros(self.n_labels, self.n_shared),
        )

        # ---- Optional dataset-invariant alignment (control-free "differential") ----
        # EMA per-dataset mean of z_shared, subtracted before every alignment term so each cell is a
        # deviation-from-baseline in the alignment space. Reconstruction (from z) is untouched, so the
        # count/ZINB model is unaffected. At predict on a novel dataset the center is a reference-set
        # mean (control cells / query mean) passed in via ``align_center`` — see classify()/predict().
        self.invariant_align = invariant_align
        self.align_ema = float(align_ema)
        self.register_buffer('align_ds_mean', torch.zeros(n_batch, self.n_shared))
        # How many training batches each dataset's EMA mean has seen. 0 => never
        # trained on (a held-out / OOD-test dataset), so its stored mean is a
        # meaningless zero; at eval such datasets self-center on the batch query
        # mean instead (mirrors predict(center_mode='query_mean')).
        self.register_buffer('align_ds_count', torch.zeros(n_batch, dtype=torch.long))

        # Check parameter setup
        self._check_setup()

        # ----- Debug -----
        self._step = 0

    def _check_setup(self, error_on_fail: bool = False):
        # Error message definition
        def e(msg):
            if error_on_fail:
                raise ValueError(msg)
            else:
                log.warning(msg)
        # Log all batch transformations
        self.transformations = []
        if self.log_variational:
            self.transformations.append('log1p')
        if self.transformations:
            log.info(f'Applying in-batch transformations: {self.transformations}')
        
    def _get_module_by_name(self, name: str):
        module = self
        for part in name.split("."):
            if not hasattr(module, part):
                return None
            module = getattr(module, part)
        return module

    def freeze_module(
        self,
        key: str,
        soft_lr: float | None = None,
        optim: torch.nn.Module | None = None
    ) -> None:

        module = self._get_module_by_name(key)

        if module is None or not isinstance(module, torch.nn.Module):
            log.warning(f'{key} is not a valid module, skipped freeze.')
            return
        # Module already frozen
        if getattr(module, 'frozen', False):
            return

        # HARD FREEZE
        if soft_lr is None or optim is None:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            log.info(f'Successfully froze {key} parameters.')
            module.frozen = True

        # SOFT FREEZE (reduce LR)
        else:
            log.info(f'Successfully soft froze {key} parameters with lr: {soft_lr}.')
            module_params = set(module.parameters())

            for group in optim.param_groups:
                if any(p in module_params for p in group["params"]):
                    group["lr"] = soft_lr
                    
    @property
    def cached_cls_emb(self) -> torch.Tensor:
        """Adapter applied fresh each access so gradients flow to the adapter,
        while the (potentially expensive) encoder output is cached per epoch."""
        return self.cls_emb.adapter(self._cached_raw_cls_emb)

    @property
    def cached_cls_proj(self) -> torch.Tensor:
        """Return the per-epoch cached class projection (adapter + class_projection, detached)."""
        if not hasattr(self, '_cached_cls_proj'):
            self.reset_cached_cls_emb()
        return self._cached_cls_proj

    def reset_cached_cls_emb(self, verbose: bool = False, reset_labels: bool = False):
        """Reset cached encoder output and projected class embeddings on epoch start.

        The pre-adapter encoder output is stored (detached) and the adapter +
        class projection are applied once per epoch.  The projected embeddings
        are cached so the aligner can skip re-projecting every batch.
        """
        with torch.no_grad():
            self._cached_raw_cls_emb = self.cls_emb.encode().detach()
            # Cache the fully projected class embeddings (adapter + class_projection)
            cls_emb = self.cls_emb.adapter(self._cached_raw_cls_emb)
            self._cached_cls_proj = self.aligner.class_projection(cls_emb).detach()

    def _transform_x(self, x: torch.Tensor, batch_index: torch.Tensor):
        # Apply log1p transformation to input batch
        if self.log_variational:
            x = torch.log1p(x)
        return x
        
    @auto_move_data
    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
        full_forward_pass: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        
        # Collect batch data. X may arrive as int16 (backed dense-cache path
        # keeps native dtype to cut CPU work + H2D bytes); cast to float on GPU
        # here so all downstream ops (library size, log1p, recon) see float.
        # No-op when X is already float (in-memory path).
        x = tensors[REGISTRY_KEYS.X_KEY].float()
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        label = tensors[REGISTRY_KEYS.LABELS_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        # Determine library size
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        else:
            library = None
        # Construct batch from data
        o = {
            MODULE_KEYS.BATCH_INDEX_KEY: batch_index,
            MODULE_KEYS.LABEL_KEY: label,
            MODULE_KEYS.CONT_COVS_KEY: cont_covs,
            MODULE_KEYS.CAT_COVS_KEY: cat_covs,
            MODULE_KEYS.LIBRARY_KEY: library,
            MODULE_KEYS.CLS_EFF_KEY: tensors.get(REGISTRY_KEYS.CLS_EFF_KEY, None),
        }
        # Apply x transformations
        o[MODULE_KEYS.X_KEY] = self._transform_x(x, batch_index)
        return o

    def _cosine_head_logits(self, z):
        """Scaled cosine logits of the supervised head: s * cos(z, w_c)."""
        w = F.normalize(self.cls_head_w, dim=-1)
        zc = F.normalize(z, dim=-1)
        cos = (zc @ w.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return cos * self.cls_head_arc_scale

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        label: torch.Tensor,
        library: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
        **kwargs
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        # Feature masks are not supported
        feat_mask = None

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x, cont_covs), dim=-1)
        else:
            encoder_input = x
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # Perform forward pass through encoder. ``label`` is forwarded as
        # ``pert_labels`` so the encoder can apply target-gene masking
        # during training; encoders that don't consume it absorb it via
        # **kwargs and ignore it at inference.
        inference_out: dict[str, torch.Tensor] = self.z_encoder(
            encoder_input, *categorical_input,
            feature_masks=feat_mask,
            pert_labels=label,
            batch_index=batch_index,   # per-dataset control baseline for input centring
            # Baseline for center_input. Train/val: omitted -> the encoder uses the in-batch
            # per-dataset mean (diverse perturbations cancel, so it approximates a control mean).
            # predict(): the CALLER supplies a mask marking that query's own reference group
            # (e.g. healthy vs diseased) -- this is query metadata, NOT the model's trained control
            # class, and a homogeneous query needs it: centering such a query on its own all-cell
            # mean deletes the effect (measured: 84% of a known signature lost vs 2% via a
            # reference group).
            ctrl_mask=kwargs.get('ctrl_mask', None),
        )
        # Unpack encoder output
        qz = inference_out[MODULE_KEYS.QZ_KEY]
        z = inference_out[MODULE_KEYS.Z_KEY]
        effective_mask = inference_out.get(MODULE_KEYS.EFFECTIVE_MASK_KEY, feat_mask)
        ql = None
        if not self.use_observed_lib_size:
            if self.batch_representation == 'embedding':
                ql, library_encoded = self.l_encoder(encoder_input, *categorical_input)
            else:
                ql, library_encoded = self.l_encoder(
                    encoder_input, batch_index, *categorical_input
                )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        # Construct output object
        o = {
            MODULE_KEYS.X_KEY: x,
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
            MODULE_KEYS.LABEL_KEY: label,
            MODULE_KEYS.BATCH_INDEX_KEY: batch_index,
            MODULE_KEYS.CONT_COVS_KEY: cont_covs,
            MODULE_KEYS.CAT_COVS_KEY: cat_covs,
            MODULE_KEYS.EFFECTIVE_MASK_KEY: effective_mask,
        }
        # Forward other inputs
        o.update(kwargs)
        return o

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(MODULE_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        # Return generative data
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY],
            MODULE_KEYS.Y_KEY: inference_outputs[MODULE_KEYS.LABEL_KEY],
            MODULE_KEYS.CONT_COVS_KEY: inference_outputs.get(MODULE_KEYS.CONT_COVS_KEY),
            MODULE_KEYS.CAT_COVS_KEY: inference_outputs.get(MODULE_KEYS.CAT_COVS_KEY),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
            MODULE_KEYS.CTX_PROJ_KEY: inference_outputs.get(MODULE_KEYS.CTX_PROJ_KEY),
        }
    
    def _head_input(self, z: torch.Tensor) -> torch.Tensor:
        """The head input: z_shared, with dropout during training (cls_head_dropout)."""
        h = z
        # DROPOUT ON THE HEAD INPUT. The head is built with n_layers=0 -- a bare Linear -- so the
        # Classifier's own dropout_rate has no hidden block to act on and is inert here. Applying it
        # to the head INPUT is the version that does something, and the motivation is a measured
        # result: giving the head a hidden layer (h_mlp) made things WORSE, i.e. head capacity
        # SUBSTITUTES for encoder work rather than adding to it. Dropping units forces the encoder
        # to make class identity redundant across dimensions instead of leaning on a few. Train-only
        # by construction (F.dropout respects self.training), so eval is untouched.
        if self.cls_head_dropout > 0:
            h = F.dropout(h, p=self.cls_head_dropout, training=self.training)
        return h

    def effective_atoms(self, raw: bool = False):
        """The atom dictionary every consumer uses: the class-embedding table (`cls_emb.encode()`).

        `raw` is kept for callers that ask for the unblended table explicitly; there is no blend any more,
        so both forms return the same tensor.
        """
        ce = getattr(self, 'cls_emb', None)
        if ce is None:
            return None
        return ce.encode()

    def _prior_missing_load_hook(self, state_dict, prefix, *args):
        k = prefix + 'prior_missing_T'
        if k in state_dict and 'prior_missing_T' not in self._buffers:
            self.register_buffer('prior_missing_T', torch.zeros_like(state_dict[k]))

    def _prior_missing_apply(self, A):
        """Untrained rows -> saved f(missing) where a prediction exists (non-zero row); live rows untouched."""
        n = int(self.n_labels)
        if A.shape[0] <= n:
            return A
        T = self.prior_missing_T[n:A.shape[0]].to(device=A.device, dtype=A.dtype)
        fill = T.norm(dim=1) > 0
        return torch.cat([A[:n], torch.where(fill[:, None], T, A[n:])], 0)

    def _legacy_state_hook(self, state_dict, prefix, *args, **kwargs):
        legacy.drop_removed_state(self, state_dict, prefix)

    def _hub_pen_load_hook(self, state_dict, prefix, *args, **kwargs):
        k = prefix + 'atom_hub_pen'
        if k in state_dict and 'atom_hub_pen' not in self._buffers:
            self.register_buffer('atom_hub_pen', torch.zeros_like(state_dict[k]))

    def has_hub_correction(self) -> bool:
        return 'atom_hub_pen' in self._buffers

    def _readout_noise_load_hook(self, state_dict, prefix, *args, **kwargs):
        k = prefix + 'readout_noise'
        if k in state_dict and 'readout_noise' not in self._buffers:
            self.register_buffer('readout_noise', torch.zeros_like(state_dict[k]))

    def has_readout_noise(self) -> bool:
        return 'readout_noise' in self._buffers

    def store_readout_noise(self, vec) -> None:
        """Persist the readout noise curve: [icpt_norm, slope_norm, icpt_span, slope_span] (natural log)."""
        vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
        if vec.numel() != 4:
            raise ValueError(f'readout_noise needs 4 values, got {vec.numel()}')
        if 'readout_noise' in self._buffers:
            self.readout_noise.copy_(vec.to(self.readout_noise.device))
        else:
            self.register_buffer('readout_noise', vec)

    def _effect_head_load_hook(self, state_dict, prefix, *args, **kwargs):
        for name in ('effect_size_w', 'effect_size_meta'):
            k = prefix + name
            if k in state_dict and name not in self._buffers:
                self.register_buffer(name, torch.zeros_like(state_dict[k]))

    def has_effect_head(self) -> bool:
        return 'effect_size_w' in self._buffers and 'effect_size_meta' in self._buffers

    def store_effect_head(self, w, meta) -> None:
        """Persist the effect-size readout: raw-space weights w (5,) over [eff_hat, max_cos, margin, p_top1, ||z_c||]
        and meta [b, k, T, n_labels] (see ExPert.add_effect_head)."""
        w = torch.as_tensor(w, dtype=torch.float32).flatten()
        meta = torch.as_tensor(meta, dtype=torch.float32).flatten()
        if w.numel() != 5 or meta.numel() != 4:
            raise ValueError(f'effect_size_w needs 5 values and effect_size_meta 4, got {w.numel()} / {meta.numel()}')
        for name, v in (('effect_size_w', w), ('effect_size_meta', meta)):
            if name in self._buffers:
                del self._buffers[name]
            self.register_buffer(name, v)

    def store_hub_correction(self, pen: torch.Tensor) -> None:
        """Persist the per-atom popularity offset (cosine units). pen: (2, n_rows) -- [+atom, -atom]."""
        dev = self.cls_emb.embedding.weight.device if getattr(self, 'cls_emb', None) is not None else 'cpu'
        pen = torch.as_tensor(pen, dtype=torch.float32, device=dev)
        if pen.dim() == 1:
            pen = torch.stack([pen, torch.zeros_like(pen)])
        if 'atom_hub_pen' in self._buffers and self.atom_hub_pen.shape == pen.shape:
            self.atom_hub_pen.copy_(pen)
        else:
            if 'atom_hub_pen' in self._buffers:
                del self._buffers['atom_hub_pen']
            self.register_buffer('atom_hub_pen', pen)

    def _hub_offset(self, n_rows: int, idx: torch.Tensor | None = None, sign: int = 0):
        """Offset (cosine units) for the scored class rows, or None when off / absent / mismatched.
        idx = the row indices that were scored (local predictions); rows past the buffer get 0."""
        if not getattr(self, 'use_hub_correction', True) or 'atom_hub_pen' not in self._buffers:
            return None
        pen = self.atom_hub_pen[0 if sign >= 0 else 1]
        if idx is None:
            return pen if pen.shape[0] == n_rows else None
        out = torch.zeros(idx.shape[0], device=pen.device, dtype=pen.dtype)
        ok = idx < pen.shape[0]
        out[ok] = pen[idx[ok]]
        return out

    def has_prior_missing_fill(self) -> bool:
        """True when predict(obs_only=False) scores untrained rows against f(missing), not the random init."""
        return 'prior_missing_T' in self._buffers and bool((self.prior_missing_T.norm(dim=1) > 0).any())

    @torch.no_grad()
    def store_prior_missing(self, fitter=None) -> int:
        """Write a fitter's f(missing) into the persistent `prior_missing_T` (n_vocab, d) buffer.

        Zero rows = no prediction (no bank covers that gene), which _prior_missing_apply leaves at the
        table value. Returns the number of filled rows (0 = nothing to store)."""
        f = fitter if fitter is not None else getattr(self, '_prior_atom_fitter', None)
        M, rows = getattr(f, 'missing', None), getattr(f, 'missing_rows', None)
        ce = getattr(self, 'cls_emb', None)
        if ce is None or M is None or rows is None or not len(rows):
            return 0
        dev = self.class_centers.device
        T = torch.zeros(int(ce.shape[0]), int(M.shape[1]), device=dev)
        T[rows.to(dev).long()] = M.detach().to(device=dev, dtype=T.dtype)
        if 'prior_missing_T' in self._buffers and self.prior_missing_T.shape == T.shape:
            self.prior_missing_T.copy_(T)
        else:
            self.register_buffer('prior_missing_T', T)
        log.info('[prior-missing] stored f(missing) for %d untrained rows (buffer %s)', int(len(rows)),
                 tuple(T.shape))
        return int(len(rows))

    @torch.no_grad()
    def rebuild_prior_missing(self) -> int:
        """Refit the prior map on the CURRENT atoms, exactly as training does, and store f(missing).

        Needs the evidence ranks (`atom_eff_qual` etc.) that ExPert._compute_prior_atom_eff_gate attaches;
        the prior banks themselves are checkpoint buffers. Runs the training refit through
        _prior_atom_loss, with the attributes that call writes restored afterwards, so only
        `prior_missing_T` changes."""
        _attrs = {k: getattr(self, k, None) for k in
                  ('prior_atom_neg_weight', '_cur_epoch', '_prior_atom_fitter', '_prior_atom_epoch',
                   '_prior_atom_tripped', '_prior_refit_n')}
        was_training = self.training
        try:
            if not io.non_zero(getattr(self, 'prior_atom_neg_weight', 0.0)):
                self.prior_atom_neg_weight = 1.0          # f(missing) is only predicted when negatives are on
            self._cur_epoch = max(int(getattr(self, '_cur_epoch', 0) or 0), int(self.prior_atom_start_epoch))
            self._prior_atom_fitter, self._prior_atom_tripped = None, False
            self.train()
            self._prior_atom_loss()
            f = self._prior_atom_fitter
        finally:
            self.train(was_training)
            for k, v in _attrs.items():
                setattr(self, k, v)
        n = self.store_prior_missing(f)
        if not n:
            log.warning('[prior-missing] refit produced no f(missing) -- no priors, no evidence ranks, or the fit '
                        'was skipped (see [prior-atoms] lines above)')
        return n

    def _prior_neg_loss(self, z_shared, real_mask=None):
        """Repel cells from f(missing) -- the UNTRAINED rows a zero-shot deployment would fill.

        WHY THIS IS NEEDED AND NOT A NICETY. Measured offline on H8/fold_4: adding the f()-filled
        untrained rows to the readout dictionary costs -0.0401 top10 on the TRAINED classes, while
        adding the SAME NUMBER of random-init rows costs exactly 0.0000. Random directions are
        never competitive; f() ones are plausible-but-wrong, which is strictly worse than useless.
        So a model that ships f(missing) has to be taught to reject them, and training is the only
        place that can happen -- a frozen table cannot re-separate anything.

        Deliberately a SEPARATE term rather than extra columns inside proxy_anchor_loss: the PA
        negative term is normalised by |P| = L for batch-composition independence, so appending
        columns would rescale it and break comparability with every arm trained so far.

        Same shape as PA's negative half (softplus-free logsumexp-over-cells with the PA margin),
        detached targets, and a fresh random subset each step -- f() refits every epoch, so a fixed
        set would let the encoder memorise these particular directions instead of the region.
        """
        dev = z_shared.device
        zero = torch.zeros((), device=dev)
        f = self._prior_atom_fitter
        M = None if f is None else getattr(f, 'missing', None)
        if M is None or not len(M) or getattr(self, '_prior_atom_tripped', False):
            return zero, {}
        v = torch.ones(z_shared.shape[0], dtype=torch.bool, device=dev)
        if real_mask is not None:
            v = v & real_mask.view(-1).bool()
        if int(v.sum()) < 4:
            return zero, {}
        k = min(int(self.prior_atom_neg_k), M.shape[0])
        idx = torch.randperm(M.shape[0], device=dev)[:k]
        with torch.autocast(device_type=dev.type, enabled=False):
            zc = z_shared.float()[v]
            zc = zc - zc.mean(0, keepdim=True)          # PA's centring convention
            zc = F.normalize(zc, dim=-1)
            Mv = M[idx].to(dev).float().detach()
            Mv = F.normalize(Mv - Mv.mean(0, keepdim=True), dim=-1)
            sim = zc @ Mv.T                              # (B, k)
            u = float(self.proxy_anchor_alpha) * (sim + float(self.proxy_anchor_delta))
            z0 = torch.zeros(1, u.shape[1], device=dev, dtype=u.dtype)
            loss = torch.logsumexp(torch.cat([z0, u], dim=0), dim=0).mean()
        with torch.no_grad():
            met = {'prior_neg/loss': loss.detach(),
                   'prior_neg/k': torch.tensor(float(k), device=dev),
                   'prior_neg/sim_mean': sim.mean(),
                   # the number that says whether the distractors are actually close: if max
                   # similarity to a filled row approaches a cell's own-atom similarity, the
                   # readout WILL confuse them.
                   'prior_neg/sim_max': sim.max()}
        return loss, met

    def prior_atom_ranking(self) -> str:
        """The ranking the prior-atom selection uses, as resolved in `_prior_atom_loss`: 'nqxnds' (train
        cells x datasets) when that is the evidence, otherwise the per-dataset |eff| rank ('eff')."""
        return 'nqxnds' if self.prior_atom_evidence == 'nqxnds' else 'eff'

    def _prior_atom_loss(self):
        """Pull WEAK-response atoms toward f(prior), fit on the STRONG ones. See _prior_atoms.py.

        WHICH atoms: the hard gate (legacy name 'eff') pulls the weakest pull_q fraction of the
        prior_atom_ranking() ranking with a uniform weight -- v0.1.0: the bottom 11.2% by cells x
        datasets ('nqxnds'), NOT by |eff|. The atoms f() is fit on are never the atoms it moves (one
        threshold for both roles, plus cross-fitting over atoms).

        THE TARGET IS DETACHED and its scale is exogenous, so this term can only rotate a weak
        atom toward the prior; it can neither reach the encoder directly nor change atom norm.
        """
        dev = self.class_centers.device
        zero = torch.zeros((), device=dev)
        ce = getattr(self, 'cls_emb', None)
        # 1=no atoms 2=no gate 3=pre-start 4=no priors 5=fit failed 6=applied 7=runaway guard
        if ce is None or not getattr(ce, 'random_init_proxies', False):
            return zero, {'prior_atom/state': torch.tensor(1.0, device=dev)}
        _q = getattr(self, 'atom_eff_qual', None)
        if _q is None:
            # LOUD: the gate is built at setup from the fold's train split. Without it the term
            # would silently fall back to a constant weight, i.e. pull EVERY atom equally.
            if not getattr(self, '_prior_atom_gate_warned', False):
                log.warning('[prior-atoms] prior_atom_weight>0 but no atom_eff_qual -- the |eff| '
                            'gate was never built (needs a fold_dir). Term REFUSES.')
                self._prior_atom_gate_warned = True
            return zero, {'prior_atom/state': torch.tensor(2.0, device=dev)}
        # EVIDENCE SOURCE. `atom_eff_qual` is the |eff| percentile; `atom_nqxnds_qual` the cells x datasets
        # rank. Both are in [0,1] with 1 = MOST evidence.
        if self.prior_atom_evidence == 'nqxnds':
            # CELLS x N_DATASETS. Offline +75 self / +72 overall vs raw depth's +64/+62; cross-
            # screen support is not raw depth. Built at setup next to atom_ntrain_qual.
            _qn = getattr(self, 'atom_nqxnds_qual', None)
            if _qn is None:
                if not getattr(self, '_prior_atom_nqwarn', False):
                    log.warning('[prior-atoms] evidence=nqxnds but atom_nqxnds_qual was never '
                                'built -- falling back to |eff|, which measures +11 vs +72. '
                                'Check the setup hook.')
                    self._prior_atom_nqwarn = True
            else:
                _q = _qn
        _q = _q.to(dev)
        ep = int(getattr(self, '_cur_epoch', 0))
        if ep < self.prior_atom_start_epoch:
            # NOT a misconfiguration: at epoch 0 the atoms are randn*0.02 and a map fit on them
            # regresses onto the init, not onto response structure.
            return zero, {'prior_atom/state': torch.tensor(3.0, device=dev)}

        n_live = int(getattr(self, 'n_labels', 0) or 0)
        A_all = self.effective_atoms(raw=True)
        if A_all is None or n_live <= 0:
            return zero, {'prior_atom/state': torch.tensor(1.0, device=dev)}
        A = A_all[:n_live]

        due = (self._prior_atom_fitter is None
               or (ep - self._prior_atom_epoch) >= max(1, self.prior_atom_refit_every))
        if due and self.training:
            # REFIT IS REQUIRED, not an optimisation: this term MOVES the atoms f() was fit on, so
            # a map frozen at epoch 5 is regressing onto a table that no longer exists.
            def _res(x):
                return getattr(ce, x, None) if isinstance(x, str) else x
            pri = getattr(ce, '_atom_priors', None) or []
            banks = [(nm, _res(e), _res(v)) for nm, e, v in pri]
            banks = [(nm, e[:n_live] if e is not None else None,
                      v[:n_live] if v is not None else None) for nm, e, v in banks]
            banks = [b for b in banks if b[1] is not None]
            if not banks:
                log.warning('[prior-atoms] prior_atom_weight>0 but cls_emb has no _atom_priors -- '
                            'the term is INERT. Set class_emb_extra_paths / a text bank.')
                return zero, {'prior_atom/state': torch.tensor(4.0, device=dev)}
            rank = _q[:n_live]
            cov = torch.isfinite(rank)
            if int(cov.sum()) >= 2:
                thr = torch.quantile(rank[cov], self.prior_atom_pull_q)
                strong = torch.nonzero(cov & (rank > thr), as_tuple=False).flatten()
                # ONE threshold for both roles, so "the atoms f() is FIT on are the atoms f()
                # never MOVES" holds exactly, whatever shape the score distribution has. A
                # literal 0.5 only coincides with it because the score is a median of percentiles.
                self._prior_atom_thr = float(thr)
            else:
                strong = torch.zeros(0, dtype=torch.long, device=dev)
            # PULLED = the weakest pull_q fraction only. Uniform weight inside it: measured
            # offline, a flat alpha beats a ramp across the gate 6x (+0.3000 vs +0.0500 on the
            # (0.1,0.2] band), because a ramp spends most of its budget on classes that barely move.
            targets = torch.nonzero(cov & (rank <= thr), as_tuple=False).flatten()
            _miss = (torch.arange(n_live, A_all.shape[0], device=dev)
                     if io.non_zero(self.prior_atom_neg_weight) else None)
            # Shared graph-map options. The k-NN cache lives on the MODULE because a fresh
            # predictor is built every refit; a plain attribute, never a buffer, so checkpoints and
            # the RNG are untouched ([[feedback_new_buffers_break_checkpoints]]).
            if getattr(self, '_prior_graph_cache', None) is None:
                self._prior_graph_cache = {}
            _gev = torch.zeros(A_all.shape[0], device=dev)
            _gev[:n_live] = torch.nan_to_num(_q[:n_live].to(dev).float(), nan=0.0)
            _gkw = dict(graph_k=int(getattr(self, 'prior_atom_graph_k', 25)),
                        graph_loo=bool(getattr(self, 'prior_atom_graph_loo', False)),
                        graph_evidence=_gev,
                        graph_ev_pow=float(getattr(self, 'prior_atom_graph_ev_pow', 0.0)),
                        graph_cache=self._prior_graph_cache)
            f = _pa_prior.PriorAtomPredictor(
                lam=self.prior_atom_lam, ridge_alpha=self.prior_atom_ridge_alpha,
                min_fit=self.prior_atom_min_fit,
                cos_floor=getattr(self, 'prior_atom_cos_floor', 0.02),
                map=getattr(self, 'prior_atom_map', 'ridge'),
                missing_map=getattr(self, 'prior_atom_missing_map', '') or None,
                missing_sim_pow=float(getattr(self, 'prior_atom_missing_sim_pow', 1.0)),
                seed=int(getattr(self, '_prior_atom_seed', 0)), **_gkw)
            # AUTOCAST OFF FOR THE FIT. loss() runs inside autocast, and a 4,096-wide
            # `linalg.solve` in fp16 is at best inaccurate and at worst unsupported. The smokes are
            # precision=32, so this is exactly the class of bug they cannot catch
            # ([[feedback_smoke_blind_spots]]).
            with torch.autocast(device_type=A.device.type, enabled=False):
                _bk_full = [(nm, e, v) for nm, e, v in
                            [(nm, _res(e), _res(v)) for nm, e, v in (pri or [])] if e is not None]
                _At = A_all.detach().float() if _miss is not None else A.detach().float()
                _bk = _bk_full if _miss is not None else banks
                _nf = max(1, int(self.prior_atom_folds))
                if (_nf > 1 and str(getattr(self, 'prior_atom_map', 'ridge')) == 'graph'
                        and bool(getattr(self, 'prior_atom_graph_loo', False))):
                    # the leave-one-out inside the graph map replaces the fold split; for this map
                    # the only possible leak (self as neighbour) is excluded row by row
                    if not getattr(self, '_prior_graph_loo_logged', False):
                        log.info('[prior-atoms] graph map with leave-one-out: fold split (%d) '
                                 'replaced by self-exclusion; ev_pow=%.2f', _nf,
                                 float(getattr(self, 'prior_atom_graph_ev_pow', 0.0)))
                        self._prior_graph_loo_logged = True
                    _nf = 1
                if _nf <= 1:
                    ok = f.fit(_At, _bk, strong, targets, missing=_miss, n_trained=n_live)
                else:
                    # CROSS-FIT OVER ATOMS. Each atom's target comes from a map that never saw it,
                    # so the leak invariant holds for EVERY atom rather than only across a
                    # fit/pull threshold -- the precondition for gating on cos(A,T), where an atom
                    # can be both a plausible fit row and a pull target.
                    _g = torch.Generator(device='cpu').manual_seed(
                        int(getattr(self, '_prior_atom_seed', 0)))
                    _fold = torch.randperm(n_live, generator=_g).to(dev) % _nf
                    _acc = torch.zeros_like(_At[:n_live])
                    _hit = torch.zeros(n_live, dtype=torch.bool, device=dev)
                    ok = False
                    for _k in range(_nf):
                        _hold = targets[_fold[targets] == _k]
                        _fit = strong[_fold[strong] != _k]
                        if not len(_hold) or len(_fit) < self.prior_atom_min_fit:
                            continue
                        _fk = _pa_prior.PriorAtomPredictor(
                            lam=self.prior_atom_lam, ridge_alpha=self.prior_atom_ridge_alpha,
                            min_fit=self.prior_atom_min_fit,
                            cos_floor=getattr(self, 'prior_atom_cos_floor', 0.02),
                            map=getattr(self, 'prior_atom_map', 'ridge'),
                            missing_map=getattr(self, 'prior_atom_missing_map', '') or None,
                            missing_sim_pow=float(getattr(self, 'prior_atom_missing_sim_pow', 1.0)),
                            seed=int(getattr(self, '_prior_atom_seed', 0)),
                            **{**_gkw, 'graph_cache': None})   # fold rows differ per fold
                        if not _fk.fit(_At, _bk, _fit, _hold,
                                       missing=_miss if _k == 0 else None, n_trained=n_live):
                            continue
                        _m = _fk.mask[:n_live].to(dev)
                        _acc[_m] = _fk.target[:n_live][_m].to(_acc.dtype)
                        _hit |= _m
                        ok = True
                        if _k == 0:
                            f.missing = getattr(_fk, 'missing', None)
                            f.missing_rows = getattr(_fk, 'missing_rows', None)
                            f.chosen, f.scores = _fk.chosen, _fk.scores
                            f.scale, f.coh = _fk.scale, _fk.coh
                    if ok:
                        f.target = torch.zeros_like(_At)
                        f.target[:n_live] = _acc
                        _mk = torch.zeros(_At.shape[0], dtype=torch.bool, device=dev)
                        _mk[:n_live] = _hit
                        f.mask = _mk
                        log.info('[prior-atoms] cross-fit %d folds: %d/%d atoms have a '
                                 'held-out target', _nf, int(_hit.sum()), n_live)
            _cs = float((getattr(f, 'coh', None) or {}).get('strong', 0.0) or 0.0)
            if io.non_zero(self.prior_atom_coh_max) and _cs > self.prior_atom_coh_max:
                if not getattr(self, '_prior_atom_tripped', False):
                    log.warning('[prior-atoms] RUNAWAY GUARD TRIPPED at epoch %d: fit-atom mean '
                                'off-diagonal cosine %.4f exceeds prior_atom_coh_max=%.4g '
                                '(H4 reference 0.0013). The pull is feeding itself and collapsing '
                                'the dictionary -- term DISABLED for the rest of the run.',
                                ep, _cs, self.prior_atom_coh_max)
                self._prior_atom_tripped = True
            self._prior_atom_epoch = ep
            self._prior_atom_fitter = f if ok else None
            # monotonic refit counter: id(f) can be recycled by the allocator after the old fitter is freed
            self._prior_refit_n = int(getattr(self, '_prior_refit_n', 0)) + 1

        if getattr(self, '_prior_atom_tripped', False):
            return zero, {'prior_atom/state': torch.tensor(7.0, device=dev),
                          'prior_atom/tripped': torch.tensor(1.0, device=dev)}
        f = self._prior_atom_fitter
        if f is None or f.target is None:
            return zero, {'prior_atom/state': torch.tensor(5.0, device=dev)}

        m = f.mask
        # SLICE TO THE LIVE BLOCK, do not reject. When prior_atom_neg_weight > 0 the fit runs on
        # A_all so the UNTRAINED vocabulary rows get predictions too, and mask/target then carry
        # A_all.shape[0] rows while this pull applies to the live block alone. Pull targets are all
        # < n_live by construction (they come from the |eff| gate, which is only defined there), so
        # the tail is all-False and dropping it loses nothing. The old `!=` comparison instead
        # returned state 5 EVERY step whenever the table exceeded n_labels -- i.e. the whole term
        # was silently inert in exactly the shipping configuration. Caught by P14.
        if m.shape[0] < A.shape[0]:
            return zero, {'prior_atom/state': torch.tensor(5.0, device=dev)}
        m = m[:A.shape[0]]
        if not bool(m.any()):
            return zero, {'prior_atom/state': torch.tensor(5.0, device=dev)}
        # The pulled set IS the weakest pull_q fraction and f() was fit on all the rest, so the
        # leak invariant -- no atom both trains the map and is moved by it -- holds by construction
        # rather than by a matched threshold. See [[project_struct_loss_is_the_text_leak]].
        # UNIFORM inside the pulled set (m already restricts to it), 0 elsewhere.
        _rank_all = torch.nan_to_num(_q[:n_live], nan=1.0)
        w = torch.ones(int(m.sum()), device=dev, dtype=A.dtype)
        tot = w.sum()
        if float(tot) <= 0:
            return zero, {'prior_atom/state': torch.tensor(5.0, device=dev)}
        # target is sliced the same way before masking: it too may carry the full table.
        T = f.target[:A.shape[0]][m].to(device=A.device, dtype=A.dtype).detach()
        Am = A[m]
        # DIRECTIONAL PULL: L2 between UNIT vectors, i.e. exactly 2(1 - cos).
        #
        # A raw L2 to a fixed-norm target is NOT direction-only, and the difference is first-order,
        # not a subtlety. Expanding ||A - T||^2 = ||A||^2 - 2||A||*s*cos + s^2, the radial force is
        # 2(s*cos - ||A||): inward whenever ||A|| > s*cos. Early on cos is small, so the term would
        # SHRINK precisely the weak atoms it is meant to reposition -- and atom norm sets the OMP
        # coefficients, so that is a readout effect disguised as a placement effect.
        # It is also not the operation the offline evidence covers: the offline readout L2-normalises
        # every atom before scoring, so magnitude was irrelevant there BY CONSTRUCTION. Normalising
        # both sides here makes the gradient purely tangential (zero radial component, asserted in
        # test_prior_atoms P8) and makes the term literally the quantity that was measured.
        loss = ((F.normalize(Am, dim=-1) - F.normalize(T, dim=-1)).pow(2).sum(-1) * w).sum() / tot
        # RAMP LIVES HERE, not in the annealer. The term cannot exist before start_epoch anyway,
        # and a bare `weight:` key in an anneal block is returned verbatim and silently ignores
        # max_weight/schedule -- a trap this repo has already paid for twice.
        r = min(1.0, (ep - self.prior_atom_start_epoch + 1)
                / max(1, self.prior_atom_ramp_epochs))
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(Am, T, dim=-1)
            met = {
                'prior_atom/state': torch.tensor(6.0, device=dev),
                'prior_atom/loss_raw': loss.detach(),
                # ramped, UNWEIGHTED loss. The weighted share of the objective is emitted at the
                # call site as prior_atom/contrib, which needs the weight this method never sees.
                'prior_atom/ramped': (r * loss).detach(),
                'prior_atom/scale': torch.tensor(f.scale, device=dev),
                'prior_atom/n': torch.tensor(float(int(m.sum())), device=dev),
                'prior_atom/w_mean': w.mean(),
                # THE LEAK-EXPOSURE NUMBER: share of the pull landing on atoms f() was FIT on.
                # Must read exactly 0.0 -- the sets are disjoint by construction.
                'prior_atom/w_frac_strong': (w[_rank_all[m] > float(self._prior_atom_thr)].sum()
                                             / tot),
                'prior_atom/pull_frac': torch.tensor(float(m.sum()) / max(n_live, 1), device=dev),
                'prior_atom/thr': torch.tensor(float(self._prior_atom_thr), device=dev),
                'prior_atom/cos': (cos * w).sum() / tot,
                # The pull is tangential by construction, so these are a CHECK, not a control:
                # if pulled atoms shrink relative to the untouched ones anyway, something else
                # (PA, CE) is reacting to the move and that is worth seeing.
                'prior_atom/norm_pulled': Am.norm(dim=1).mean(),
                'prior_atom/norm_free': A[:n_live][~m].norm(dim=1).mean()
                if bool((~m).any()) else A[m].norm(dim=1).mean(),
                'prior_atom/ramp': torch.tensor(float(r), device=dev),
                'prior_atom/cov_frac': torch.isfinite(_q[:n_live]).float().mean(),
                'prior_atom/eff_rank_mean': torch.nan_to_num(_q[:n_live], nan=1.0).mean(),
                # A rising flip count means the two maps are within noise for some bank, so the
                # target is jumping between refits and the fusion is weaker than it reads.
                # NOT `map_flips`: `prior_atom/map_<bank>` is a namespace, and a counter named
                # map_flips parses as a bank called "flips" -- the smoke duly reported one.
                'prior_atom/refit_age': torch.tensor(float(ep - self._prior_atom_epoch), device=dev),
            }
            for nm, k in (f.chosen or {}).items():
                met[f'prior_atom/map_{nm}'] = torch.tensor(
                    1.0 if k == 'mlp' else 0.0, device=dev)
                # THE SELECTION MUST BE AUDITABLE OVER THE RUN, not just in the text log. If the
                # two maps score within noise the choice is a coin flip and the fusion is weaker
                # than it reads -- offline the margins were decisive (DepMap ridge 0.0732 vs mlp
                # 0.0340; text mlp 0.0668 vs ridge 0.0129), and that is the thing to check held.
                _sc = (f.scores or {}).get(nm, {})
                for _k, _v in _sc.items():
                    met[f'prior_atom/fit_{nm}_{_k}'] = torch.tensor(float(_v), device=dev)
            for _k, _v in (getattr(f, 'coh', None) or {}).items():
                met[f'prior_atom/coh_{_k}'] = torch.tensor(float(_v), device=dev)
        return r * loss, met

    @auto_move_data
    def generative(self, *args, **kwargs) -> dict[str, Distribution | None]:
        # Reconstruct raw data using scvi-style setup
        return self._generative_scvi(*args, **kwargs)
    
    def _generative_scvi(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        ctx_proj: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
        **kwargs
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        z_dec = z

        # Likelihood distribution
        if cont_covs is None:
            decoder_input = z_dec
        elif z_dec.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z_dec, cont_covs.unsqueeze(0).expand(z_dec.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z_dec, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library
        # Perform decoder forward pass
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            *categorical_input,
            context_emb=None
        )

        if self.dispersion == "gene-label":
            px_r = linear(
                F.one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(rate=px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            px = Normal(px_rate, px_r, normal_mu=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
        }
    
    def classify(
        self,
        x: torch.Tensor | None,
        batch_index: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        eff_scores: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
        inference_outputs: dict[str, torch.Tensor] | None = None,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        local_predictions: bool = True,
        align_center: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Note: caller controls the grad context. predict() wraps its loop in
        # torch.no_grad(); tent_adapt() needs grad to flow to LayerNorm γ/β.
        """Forward pass through the encoder and classifier.

        Parameters
        ----------
        x
            Tensor of shape ``(n_obs, n_vars)``.
        batch_index
            Tensor of shape ``(n_obs,)`` denoting batch indices.
        cont_covs
            Tensor of shape ``(n_obs, n_continuous_covariates)``.
        cat_covs
            Tensor of shape ``(n_obs, n_categorical_covariates)``.
        use_posterior_mean
            Whether to use the posterior mean of the latent distribution for
            classification.
      
        Returns
        -------
        Tensor of shape ``(n_obs, n_labels)`` denoting logit scores per label.
        Before v1.1, this method by default returned probabilities per label,
        see #2301 for more details.
        """
        # Try caching inference and recalculate if missing
        if inference_outputs is None:
            # Get inference inputs
            batch = {
                REGISTRY_KEYS.X_KEY: x,
                REGISTRY_KEYS.BATCH_KEY: batch_index,
                REGISTRY_KEYS.LABELS_KEY: label,
                REGISTRY_KEYS.CONT_COVS_KEY: cont_covs,
                REGISTRY_KEYS.CAT_COVS_KEY: cat_covs,
                REGISTRY_KEYS.CLS_EFF_KEY: eff_scores,
            }
            inference_outputs = self.inference(**self._get_inference_input(batch))
        # Get inference outputs
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]
        z = inference_outputs[MODULE_KEYS.Z_KEY]
        # Aligner forward pass using either qz mean or sampled z
        _z = qz.loc if use_posterior_mean else z
        
        # Do alignment forward pass. The model's OWN bank iff the caller passed none -- captured BEFORE the
        # reassignment below, which is what decides whether the saved f(missing) fill applies in _class_reps
        _own_bank = cls_emb is None
        if cls_emb is None:
            cls_emb = self.cls_emb()
        align_out = self.aligner(x=_z, cls_emb=cls_emb, return_logits=False)
        # Get aligned spaces
        z2c = align_out.get(MODULE_KEYS.Z_SHARED_KEY)
        # PUBLISH THE RAW LATENT TOO, BEFORE centring. `_resolve_align_center` builds the
        # inference baseline by averaging z_shared per dataset, and it needs the UN-CENTRED space
        # -- the same space `align_ds_mean` lives in. It calls classify(align_center=None), which
        # at eval still applies the stored-EMA / query-mean branch, so reading the centred tensor
        # would average an already-centred latent and return a baseline of ~0. That is not
        # hypothetical: it produced a resolved bank with max|z| 2.4e-4 (3e-6 relative) instead of
        # the expected O(1).
        align_out['z_shared_raw'] = z2c
        z2c = self._center_z_shared(z2c, batch_index, center=align_center)  # dataset-baseline removal (no-op unless invariant_align)
        # PUBLISH THE CENTRED LATENT. z2c feeds the efficiency head, the prototype logits and
        # _head_input, but align_out[Z_SHARED_KEY] still held the RAW aligner output -- so
        # predict() returned an UNCENTRED z_shared and every consumer of it (ood_readout_probe
        # included) scored the uncentred space. The whole `align_center` / _resolve_align_center
        # path existed to supply a baseline that then never reached the caller.
        # SAFE FOR EVERY EXISTING ARM: without invariant_align `_center_z_shared` is the identity,
        # so z2c IS the raw tensor and this write-back changes nothing. Only invariant_align
        # models see a different value -- which is the point.
        align_out[MODULE_KEYS.Z_SHARED_KEY] = z2c
        c2z = self._class_reps(align_out, apply_prior_fwd=_own_bank)  # aligner class projection (+ saved f(missing) fill)
        # Publish the resolved class bank so downstream (predict()'s cls_proj collection, reports) reads
        # the bank that is actually scored.
        align_out[MODULE_KEYS.CLS_PROJ_KEY] = c2z
        # Subset class embedding to local (trained) classes
        if local_predictions:
            local_targets = torch.arange(self.n_labels, device=c2z.device)
            _c2z = c2z[local_targets]
        else:
            _c2z = c2z
        # Get cell-wise T predictions if enabled
        if self.model_efficiency:
            eff_hat = self.efficiency_head(z2c).flatten()
            align_out['eff_hat'] = eff_hat
            # Log raw |weights| only. Per-dataset z-scoring is done ONCE
            # post-aggregation in the predict loop (see _expert.py), not
            # per-batch — each batch has its own (mean, std) per dataset,
            # so per-batch z-scores aggregated across batches end up with
            # inconsistent normalization and a noisy diagnostic. Doing it
            # across the full eval set gives a single consistent axis.
            weights: torch.Tensor | None = inference_outputs.get(MODULE_KEYS.CLS_EFF_KEY, None)
            if weights is not None:
                align_out[MODULE_KEYS.CLS_EFF_KEY] = weights.abs().flatten()
            # Apply current T scaling if active. eff_hat is now a raw
            # (un-standardized) per-cell score; standardize per dataset for
            # the temperature signal so it matches the training-time gate.
            if hasattr(self, 'last_t_gamma') and self.last_t_gamma > 0:
                ctx_flat = batch_index.flatten().long()
                eff_z = torch.zeros_like(eff_hat)
                for c in ctx_flat.unique():
                    m = ctx_flat == c
                    e = eff_hat[m]
                    if e.numel() >= 2:
                        eff_z[m] = (e - e.mean()) / (e.std() + 1e-4)
                eff_norm = (eff_z.detach().clamp(min=0.0) / 3.0).clamp(max=1.0)
                T_i = self.aligner.temperature * (1 + self.last_t_gamma * (1 - eff_norm))
                T_i = T_i.clamp(min=0.07, max=1.0).unsqueeze(-1)
                align_out['T_i'] = T_i
            else:
                T_i = self.aligner.temperature
        # Use global training alignment temperature
        else:
            T_i = self.aligner.temperature
        # Get logits — A (adapter/text) cosine over the class bank owns CLS_LOGITS_KEY (the
        # full-bank / held-out scorer used for zero-shot + far-OOD recall).
        align_out[MODULE_KEYS.CLS_LOGITS_KEY] = self.proto_scorer.logits(z2c, _c2z, T=T_i)
        # HUBNESS CORRECTION (eval/publish path only): score = cos - pen_a, i.e. logits - pen_a / T. A
        # per-atom constant, identical for every cell, so it only re-ranks atoms within a query.
        if not self.training:
            _off = self._hub_offset(_c2z.shape[0], local_targets if local_predictions else None)
            _lg = align_out[MODULE_KEYS.CLS_LOGITS_KEY]
            if _off is not None and _lg.dim() == 2 and _lg.shape[-1] == _off.shape[0]:
                align_out[MODULE_KEYS.CLS_LOGITS_KEY] = _lg - _off.to(_lg)[None, :] / T_i
        # Two-head: publish H (free classifier) separately so both heads can be scored at eval
        # (the far-OOD contract A_recall > max(H, null) needs H's tahoe/zero-shot recall too).
        # H only spans the trained block, so it is emitted unsliced (n_labels wide) regardless
        # of local_predictions — downstream ranks it against trained targets only.
        if self.use_cls_head:
            _h2c = self._head_input(z2c)
            align_out[MODULE_KEYS.CLS_HEAD_LOGITS_KEY] = self._cosine_head_logits(_h2c)

        # Add alignment output to inference
        inference_outputs.update(align_out)
        return inference_outputs

    @staticmethod
    def _eff_zscore_per_dataset(
        eff_hat: torch.Tensor,
        weights: torch.Tensor,
        ctx_flat: torch.Tensor,
        min_cells: int = 8,
        min_target_std: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Per-dataset z-score of ``eff_hat`` and (already-abs-valued)
        ``weights`` within the batch.

        The efficiency loss minimizes ``MSE(z(eff_hat), z(|weights|))`` per
        dataset — equivalent to ``2(1 - Pearson_r)`` — so the head's
        absolute scale and offset are unconstrained by design (we don't
        encode dataset identity). Centralizing the standardization here
        keeps the training objective and the predict-time diagnostic in
        sync.

        Datasets with fewer than ``min_cells`` cells in this batch or with
        no measurable target variance are skipped: their cells get
        ``z = 0`` and they contribute no MSE term.

        Returns
        -------
        eff_z, weights_z
            Per-cell z-scores within each cell's dataset; zero for cells in
            skipped datasets. Same shape as inputs.
        per_dataset_mse
            ``F.mse_loss(pz, tz)`` per valid dataset — averaged at the
            caller so each dataset contributes equally regardless of cell
            count.
        """
        eff_z = torch.zeros_like(eff_hat)
        weights_z = torch.zeros_like(weights)
        per_dataset_mse: list[torch.Tensor] = []
        for c in ctx_flat.unique():
            m = ctx_flat == c
            if m.sum() < min_cells:
                continue
            p, t = eff_hat[m], weights[m]
            t_std = t.std()
            if t_std < min_target_std:
                continue
            pz = (p - p.mean()) / (p.std() + 1e-4)
            tz = (t - t.mean()) / (t_std + 1e-4)
            eff_z[m] = pz
            weights_z[m] = tz
            per_dataset_mse.append(F.mse_loss(pz, tz))
        return eff_z, weights_z, per_dataset_mse

    def _class_reps(self, alignment_output, apply_prior_fwd: bool = True):
        """Pre-scored class bank in the shared space: the aligner class projection (CLS_PROJ), with the
        saved f(missing) fill (`prior_missing_T`) on the untrained rows when the model carries one.

        Under the identity adapter CLS_PROJ is the atom table, so without the fill the untrained rows are
        still the random init, which predict(obs_only=False) would score as zero-shot classes. Only for the
        model's OWN bank: an external cls_emb passed to classify() has rows that do not line up with the
        buffer (apply_prior_fwd=False), and a bank of a different shape is left alone."""
        _cp = alignment_output.get(MODULE_KEYS.CLS_PROJ_KEY)
        if _cp is None:
            return None
        if (apply_prior_fwd and 'prior_missing_T' in self._buffers and _cp.dim() == 2
                and _cp.shape[1] == self.prior_missing_T.shape[1]
                and int(self.n_labels) < _cp.shape[0] <= self.prior_missing_T.shape[0]):
            _cp = self._prior_missing_apply(_cp)
        return _cp

    def _center_z_shared(self, z_shared, ctx, center=None):
        """Subtract the per-dataset baseline from z_shared (control-free 'differential' in the
        alignment space). No-op unless ``invariant_align``.

        Training: EMA-update each present dataset's mean (and its seen-count) and subtract it.
        Inference: subtract an explicit ``center`` (predict()'s reference-set / control mean) if
        given. Otherwise subtract the stored EMA mean for datasets seen in training, but for
        datasets NEVER trained on (OOD-test / ood_ood_pert held-out datasets — align_ds_count==0,
        stored mean is a meaningless zero) self-center on the batch's per-dataset query mean. This
        makes the in-training OOD-test metric take the same control-free path as classify()/predict()
        instead of collapsing on an uncentered latent."""
        if not self.invariant_align:
            return z_shared
        # dtype note: align_ds_mean is a fp32 buffer but z_shared is bf16 under bf16-mixed autocast.
        # Scalar-indexed buffer assignment (buffer[c] = ...) rejects a bf16 source ("Index put
        # requires the source and destination dtypes match"), so all EMA/query-mean updates run in
        # fp32 (autocast disabled), and the final subtraction casts the fp32 center back to z_shared's
        # dtype so the centered latent stays bf16 for the downstream alignment matmuls.
        if center is not None:
            # (n_shared,) single reference vector -> broadcast; (n_batch, n_shared)
            # per-dataset bank -> gather each cell's dataset row (predict() builds
            # this from the reference/control set for control-free inference).
            center = center.to(z_shared.dtype)
            if center.dim() == 2:
                return z_shared - center[ctx.flatten().long()]
            return z_shared - center
        ctx_flat = ctx.flatten().long()
        if self.training:
            with torch.no_grad(), torch.autocast(device_type=z_shared.device.type, enabled=False):
                zf = z_shared.float()
                for c in torch.unique(ctx_flat):
                    m = ctx_flat == c
                    self.align_ds_mean[c] = (
                        (1 - self.align_ema) * self.align_ds_mean[c]
                        + self.align_ema * zf[m].mean(0))
                    self.align_ds_count[c] += 1
            return z_shared - self.align_ds_mean[ctx_flat].to(z_shared.dtype)
        # Eval: stored EMA mean for trained datasets; batch query mean for never-trained ones.
        # Device-safe: on ExPert.load() the align_ds_* buffers can be on cpu while z_shared/ctx are on cuda.
        _dev = z_shared.device
        _mean = self.align_ds_mean.to(_dev)
        _count = self.align_ds_count.to(_dev)
        ctx_flat = ctx_flat.to(_dev)
        centers = _mean[ctx_flat]
        unseen = _count[ctx_flat] == 0
        if unseen.any():
            centers = centers.clone()
            zf = z_shared.float()
            for c in torch.unique(ctx_flat[unseen]):
                m = ctx_flat == c
                centers[m] = zf[m].mean(0, keepdim=True)  # fp32 -> fp32 clone
        return z_shared - centers.to(z_shared.dtype)

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        rl_weight: float = 1.0,
        kl_weight: float = 1.0,
        cls_align_weight: float = 0.0,   # REMOVED (CLIP alignment): must be 0; read only to reject old configs
        use_posterior_mean: bool = True,
        local_predictions: bool = True,
        T_align: float | None = None,
        lambda_eff: float = 0.1,
        T_gamma: float = 0.0,
        gate_eff: float = 0.0,
        clip_grad_scale: float = 1.0,
        cls_head_weight: float = 0.0,   # supervised cosine head on z_shared (needs use_cls_head=True)
        # ATOM WEIGHTS ARE SCHEDULED LIKE EVERY OTHER LOSS WEIGHT. None means "not scheduled": fall back
        # to the module attribute, so every existing config and checkpoint keeps its current behaviour.
        prior_atom_weight: float | None = None,
        proxy_anchor_weight: float | None = None,
        **kwargs,
    ) -> LossOutput:
        """Compute the total training loss.

        Orchestrator only: the ELBO (reconstruction + KL), the prior-atom pull and its f(missing)
        negatives, the proxy anchor, the efficiency head, the prototype scorer (diagnostic only) and the supervised
        cosine head, accumulated here in a fixed order. Unknown (removed) loss weights arrive in **kwargs and
        are ignored.
        """
        if io.non_zero(cls_align_weight):
            raise ValueError(f'cls_align_weight={float(cls_align_weight)}: CLIP alignment has been removed; the '
                             'prototype logits are trained by the proxy anchor. Set cls_align_weight to 0.')
        self._step = self._step + 1

        # ---- Unpack ----
        ctx: torch.Tensor = inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY]
        cls: torch.Tensor = inference_outputs[MODULE_KEYS.LABEL_KEY]
        z: torch.Tensor = inference_outputs[MODULE_KEYS.Z_KEY]
        px: Distribution = generative_outputs[MODULE_KEYS.PX_KEY]
        qz: Distribution = inference_outputs[MODULE_KEYS.QZ_KEY]
        pz: Distribution = generative_outputs[MODULE_KEYS.PZ_KEY]
        weights: torch.Tensor | None = inference_outputs.get(MODULE_KEYS.CLS_EFF_KEY, None)
        real_mask: torch.Tensor | None = inference_outputs.get('real_mask', None)

        extra_metrics: dict = {}
        extra_outputs: dict = {}

        # ---- ELBO: reconstruction + KL ----
        elbo = self._elbo_loss(
            tensors, inference_outputs, px, qz, pz,
            ctx=ctx, cls=cls, real_mask=real_mask, rl_weight=rl_weight, kl_weight=kl_weight,
        )
        total_loss = elbo['total']
        extra_metrics[MODULE_KEYS.KL_Z_PER_LATENT_KEY] = elbo['kl_div_per_latent']

        lo_kwargs = {
            'reconstruction_loss': elbo['reconst_loss'],
            'kl_local': {MODULE_KEYS.KL_Z_KEY: elbo['kl_divergence_z']},
            'true_labels': cls,
            'n_obs_minibatch': elbo['n_obs_minibatch'],
        }

        # ---- Re-format labels; pick the latent the alignment objectives use ----
        ctx = ctx.flatten()
        cls = cls.flatten()
        logits = None
        extra_outputs['ctx_labels'] = ctx

        _qz = qz
        z_ = z
        # Posterior mean or sampled latent, with optional grad-scaling back into
        # the encoder (0.0 = encoder trains on ELBO only, 1.0 = full alignment grad).
        _z = _qz.loc if use_posterior_mean else z_
        _z_align = co.grad_scale(_z, clip_grad_scale)

        # ---- Alignment forward (shared by every alignment term) ----
        cls2z = self.cls_emb()
        alignment_output = self.aligner(x=_z_align, cls_emb=cls2z, return_logits=False)
        z_shared_raw = alignment_output.get(MODULE_KEYS.Z_SHARED_KEY)
        z_shared = self._center_z_shared(z_shared_raw, ctx)      # control-free dataset-baseline removal (no-op unless invariant_align)
        cls2z = self._class_reps(alignment_output)               # aligner class-projection (+ saved f(missing) fill)
        z_shared_norm = F.normalize(z_shared, dim=-1)
        cls2z_norm = F.normalize(cls2z, dim=-1)
        extra_outputs[MODULE_KEYS.Z_SHARED_KEY] = z_shared_norm.detach()
        # RAW z_shared, published ALONGSIDE the normalised one. The key above is UNIT vectors, so
        # anything that pseudobulks it computes a directional mean, not a pseudobulk --
        # mean-of-unit-vectors weights every cell equally while `ood_readout_probe` averages RAW
        # z_shared, weighting each cell by its latent magnitude. Both tensors are n_shared wide, so
        # a dimension check cannot tell them apart and the divergence is silent. The group-readout
        # monitor reads THIS key.
        extra_outputs['z_shared_raw'] = z_shared.detach()
        extra_outputs[MODULE_KEYS.CLS_PROJ_KEY] = cls2z_norm.detach()

        # ---- H head (free classifier) on the LIVE z_shared (NO detach) ----
        # Computed here so the dropout RNG draw stays at this point of the step; its CE / metrics are
        # added below in the dedicated H block.
        head_logits = None
        if self.use_cls_head:
            _h_in = self._head_input(z_shared)
            head_logits = self._cosine_head_logits(_h_in)

        # ---- Learnable perturbation ATOMS: diagnostics, LOGGED UNCONDITIONALLY ----
        # Cosine objectives pin atom direction but not scale, so the norm is watched here.
        _ce_at = getattr(self, 'cls_emb', None)
        if getattr(_ce_at, 'random_init_proxies', False):
            with torch.no_grad():
                _A_dbg = self.effective_atoms()
                _nl_dbg = int(min(self.n_labels, _A_dbg.shape[0]))
                extra_metrics['atom/norm'] = _A_dbg[:_nl_dbg].norm(dim=1).mean()
                extra_metrics['atom/norm_std'] = _A_dbg[:_nl_dbg].norm(dim=1).std()
                if _A_dbg.shape[0] > _nl_dbg:
                    extra_metrics['atom/norm_dead'] = _A_dbg[_nl_dbg:].norm(dim=1).mean()
                extra_metrics['atom/n_live'] = torch.tensor(float(_nl_dbg), device=_A_dbg.device)

        # ---- PRIOR-ANCHORED WEAK ATOMS ----
        # An L2 on the atom table; it never touches z.
        _prior_atom_w = (self.prior_atom_weight if prior_atom_weight is None
                         else float(prior_atom_weight))
        # THE NEGATIVES MUST NOT DEPEND ON THE PULL. f(missing) is built inside _prior_atom_loss (it owns
        # the fitter), so the fit runs whenever either term is on; only its LOSS is withheld when the pull
        # weight is 0.
        _prior_any = (io.non_zero(_prior_atom_w) or io.non_zero(self.prior_atom_neg_weight))
        if _prior_any:
            _pt, _pm_met = self._prior_atom_loss()
            if io.non_zero(_prior_atom_w):
                total_loss = total_loss + _prior_atom_w * _pt
            if io.non_zero(self.prior_atom_neg_weight):
                _nt, _nm = self._prior_neg_loss(z_shared, real_mask=real_mask)
                total_loss = total_loss + self.prior_atom_neg_weight * _nt
                extra_metrics.update(_nm)
                extra_metrics['weights/prior_neg'] = self.prior_atom_neg_weight
            extra_metrics.update(_pm_met)
            extra_metrics['weights/prior_atom'] = _prior_atom_w
            # THE INERTNESS CHECK, as one number: what this term actually adds to the objective.
            extra_metrics['prior_atom/contrib'] = (_prior_atom_w * _pt).detach()

        # ---- PROXY ANCHOR on the atoms -------------------------------------------------
        # Decides how far apart the atoms sit, and turns itself off once they are far enough.
        _pa_w = (self.proxy_anchor_weight if proxy_anchor_weight is None
                 else float(proxy_anchor_weight))
        if io.non_zero(_pa_w):
            _ce_pa = getattr(self, 'cls_emb', None)
            if getattr(_ce_pa, 'random_init_proxies', False):
                _A_pa = self.effective_atoms()
                # LIVE ROWS ONLY. The table spans the vocabulary (19,895 at small) so zero-shot
                # labels can index it; the untrained tail would otherwise contribute ~19.7k random
                # negatives per proxy and swamp both the loss and the coherence metric.
                _nl_pa = int(min(self.n_labels, _A_pa.shape[0]))
                _pa_valid = None if real_mask is None else real_mask.view(-1).bool()
                # PER-CELL EFFICIENCY WEIGHT ON PA. PA is the only live supervision in the
                # PA arms (cls_head_weight is 0.0), so the existing efficiency machinery -- which
                # only ever fed the per-cell `cell_weights` -- was inert there.
                # Same helper and same floor as the efficiency gate: within-batch PER-DATASET rank of
                # the raw efficiency mapped to [efficiency_clip_floor, 1]. Per-dataset because the
                # raw medians span 0.0-3.3 across datasets, and RANK because the distribution is
                # ratio-like with a heavy tail (median 1.035, max 18.24). Weights are handed to the
                # LOSS, never carried as a model kwarg.
                _pa_w_eff = None
                # STATE IS LOGGED UNCONDITIONALLY. A weighting that silently does not fire is this
                # repo's characteristic failure, and "the metric is absent" does not say WHY.
                # 1=flag off, 2=no efficiency tensor, 3=shape mismatch, 4=applied.
                _eff_state = 1.0
                if getattr(self, 'use_efficiency_pa_weight', False):
                    if weights is None:
                        _eff_state = 2.0
                    else:
                        _e_pa = weights.abs().flatten()
                        # `weights` is not always batch-shaped (it can arrive filtered), so
                        # size-gate rather than let a broadcast silently mis-align cells.
                        if _e_pa.numel() != z_shared.shape[0]:
                            _eff_state = 3.0
                        else:
                            _pa_w_eff = self._efficiency_rank_weight(
                                _e_pa, ctx.flatten().long())
                            extra_metrics['pa/eff_w_mean'] = _pa_w_eff.mean().detach()
                            _eff_state = 4.0
                extra_metrics['pa/eff_w_state'] = torch.tensor(_eff_state, device=z_shared.device)
                _pl, _pm = proxy_anchor_loss(
                    z_shared, _A_pa[:_nl_pa], cls,
                    weights=_pa_w_eff,
                    alpha=self.proxy_anchor_alpha,
                    delta=self.proxy_anchor_delta,
                    neg_only=self.proxy_anchor_neg_only,
                    z_grad=self.proxy_anchor_z_grad,
                    center=self.proxy_anchor_center,
                    center_atoms=self.proxy_anchor_center_atoms,
                    valid=_pa_valid)
                if _pm:
                    total_loss = total_loss + _pa_w * _pl
                    extra_metrics.update(_pm)
                    extra_metrics['weights/proxy_anchor'] = _pa_w
                    # RESIDUAL POOLED OFFSET. z_shared already had its PER-DATASET baseline removed
                    # by _center_z_shared (invariant_align), and then proxy_anchor_loss(center=True)
                    # subtracts the BATCH mean of that centred tensor on top. If the per-dataset
                    # step is tracking, this residual is ~0 and the second subtraction is a no-op;
                    # if it is not, it is a POOLED subtraction over a multi-dataset batch, which
                    # re-couples datasets that were just decoupled. The per-dataset baselines are
                    # heterogeneous enough for this to matter (N ep59: ||baseline|| 11.1-163.9,
                    # spread/size 1.10), so measure it rather than assume.
                    with torch.no_grad():
                        _zc = z_shared.detach().float()
                        _zn = _zc.norm(dim=-1).mean().clamp_min(1e-9)
                        extra_metrics['pa/z_batch_offset_frac'] = _zc.mean(0).norm() / _zn
                    # ONE-TIME ACTIVATION LINE. This model has a history of knobs that read as
                    # set in the config and are inert at the consuming line, and the smoke can
                    # only check what the log says. Print what the term ACTUALLY received.
                    if not getattr(self, '_pa_announced', False):
                        self._pa_announced = True
                        # SHORT LINES ON PURPOSE, AT MOST TWO VALUES EACH. The rich log
                        # handler wraps at ~38 characters of message and a wrapped line is
                        # invisible to the smoke's `grep -m1`. A five-line version of this
                        # still wrapped and made the pa1_neg smoke report "neg_only never
                        # reached the loss" for a term that was working correctly. Measure
                        # before lengthening any of these.
                        log.info('[proxy-anchor] w=%.4g alpha=%.3g',
                                 _pa_w, self.proxy_anchor_alpha)
                        log.info('[proxy-anchor] delta=%.3g neg_only=%s',
                                 self.proxy_anchor_delta, bool(self.proxy_anchor_neg_only))
                        log.info('[proxy-anchor] loss=%.4f pos=%.4f',
                                 float(_pl), float(_pm['pa/pos']))
                        log.info('[proxy-anchor] neg=%.4f margin=%.4f',
                                 float(_pm['pa/neg']), float(_pm['pa/margin']))
                        log.info('[proxy-anchor] vpos=%.3f vneg=%.3f',
                                 float(_pm['pa/viol_pos']), float(_pm['pa/viol_neg']))
                        log.info('[proxy-anchor] coh=%.4f/%.4f',
                                 float(_pm['pa/coh_mean']), float(_pm['pa/coh_max']))
                        log.info('[proxy-anchor] live=%d zgrad=%.2f',
                                 _nl_pa, self.proxy_anchor_z_grad)

        # ---- DICTIONARY COHERENCE of the head input, LOGGED UNCONDITIONALLY ----
        # (median over classes of the max |cos| between batch class means). Note the head-input dropout
        # draws RNG here.
        if self.use_cls_head and cls is not None:
            with torch.no_grad():
                _y = cls.view(-1).long()
                _v = (_y >= 0) & (_y < self.n_labels)
                if real_mask is not None:
                    _v = _v & real_mask.view(-1).bool()
                _i = torch.nonzero(_v).flatten()
                if _i.numel() >= 8:
                    _zc = self._head_input(z_shared)[_i]
                    _lb, _inv = torch.unique(_y[_i], return_inverse=True)
                    if _lb.numel() > 3:
                        _s = torch.zeros(_lb.numel(), _zc.shape[1], device=_zc.device,
                                         dtype=_zc.dtype).index_add_(0, _inv, _zc)
                        _n = torch.zeros(_lb.numel(), device=_zc.device,
                                         dtype=_zc.dtype).index_add_(
                            0, _inv, torch.ones_like(_inv, dtype=_zc.dtype))
                        _mu = _s / _n.unsqueeze(1).clamp_min(1.0)
                        _mu = _mu - _zc.mean(0, keepdim=True)
                        _u = _mu - _mu.mean(1, keepdim=True)
                        _u = _u / _u.norm(dim=1, keepdim=True).clamp_min(1e-12)
                        _C = (_u @ _u.T).abs()
                        _C.fill_diagonal_(0)
                        _mc = _C.max(1).values.median()
                        extra_metrics['dict/coherence'] = _mc
                        extra_metrics['dict/k_max'] = 0.5 * (1.0 + 1.0 / _mc.clamp_min(1e-3))
                        extra_metrics['dict/n_atoms'] = torch.tensor(
                            float(_lb.numel()), device=_zc.device)

        # ---- Efficiency head + responder gate (produces cell_weights, T_i) ----
        T_align = T_align if T_align is not None else self.aligner.temperature
        eff = self._efficiency_gate(
            z_shared, weights, ctx,
            T_align=T_align, lambda_eff=lambda_eff, T_gamma=T_gamma, gate_eff=gate_eff)
        total_loss = total_loss + eff['term']
        cell_weights = eff['cell_weights']
        T_i = eff['T_i']
        extra_metrics.update(eff['metrics'])
        extra_outputs.update(eff['outputs'])
        extra_metrics['T_align'] = T_align

        # ---- Prototype scorer: A-head logits + diagnostic proto_ce ----
        # The zero-shot arms (local_predictions=False) score the FULL bank; every other split scores the
        # trained block. The logits are the published CLS_LOGITS_KEY -- the scorer every proto_* metric, the
        # checkpoint monitor and the zero-shot arms read. Computed without autograd: nothing here enters the loss.
        if local_predictions:
            _cls_emb_norm = cls2z_norm[torch.arange(self.n_labels, device=cls2z.device)]
        else:
            _cls_emb_norm = cls2z_norm
        ps = self.proto_scorer(
            z=z_shared, z_norm=z_shared_norm, emb_norm=_cls_emb_norm, y=cls,
            T=T_i if T_i is not None else T_align, weights=cell_weights)
        logits = ps.pop('logits')
        extra_outputs['proto/margin'] = ps.pop('proto/margin')
        extra_metrics.update(ps)

        # ---- H head (free supervised classifier on z_shared): CE + publication ----
        # A owns CLS_LOGITS_KEY; H is published separately under CLS_HEAD_LOGITS_KEY. head_logits was
        # computed early (above) on the LIVE z_shared.
        if self.use_cls_head and head_logits is not None:
            # Guard: the zero-shot arms pass local_predictions=False and carry HELD-OUT labels
            # y >= n_labels. This head only spans the trained block, so a CE here would index OOB.
            if local_predictions and io.non_zero(cls_head_weight):
                _ls = self.proto_scorer.label_smoothing
                _yh = cls.flatten().long()
                head_ce = F.cross_entropy(head_logits, _yh, reduction='none',
                                          label_smoothing=_ls)
                _hw = cell_weights
                if _hw is not None:
                    head_loss = (head_ce * _hw).sum() / _hw.sum().clamp_min(1.0)
                else:
                    head_loss = head_ce.mean()
                total_loss = total_loss + cls_head_weight * head_loss
                if self.training:
                    self._cls_head_steps = int(getattr(self, '_cls_head_steps', 0)) + 1
                    self._cls_head_w_max = max(float(getattr(self, '_cls_head_w_max', 0.0)), float(cls_head_weight))
                extra_metrics['cls_head_loss'] = head_loss
                extra_metrics['weights/cls_head'] = cls_head_weight
            # Coexist: publish H separately (the plan computes H's metrics from this key, so it
            # must NOT depend on cls_head_weight); only fall back to H when A produced nothing.
            extra_outputs[MODULE_KEYS.CLS_HEAD_LOGITS_KEY] = head_logits.detach()
            if logits is None:
                logits = head_logits

        # ---- Metrics: score on the trained (local) logit block by default ----
        # Soft-CE puts pressure over the full vocab, but IN-DISTRIBUTION validation
        # cells only carry trained labels — argmax over the full vocab could land on
        # a held-out class and report artificial low F1. So slice to n_labels when
        # local_predictions is True (train/val/test/ood_test). The zero-shot arms
        # (id_ood_pert / ood_ood_pert) pass local_predictions=False: their TRUE label
        # IS a held-out gene (index >= n_labels), so they need the FULL-bank logits
        # to rank it — slicing would drop the answer column and log nothing.
        if logits is not None:
            if local_predictions and logits.shape[-1] > self.n_labels:
                metric_logits = logits[:, :self.n_labels].detach()
            else:
                metric_logits = logits.detach()
            if PREDICTION_KEYS.PREDICTION_KEY not in extra_outputs:
                extra_outputs[PREDICTION_KEYS.PREDICTION_KEY] = metric_logits.argmax(dim=-1).flatten()
            extra_outputs[MODULE_KEYS.CLS_LOGITS_KEY] = metric_logits

        extra_metrics['extra_outputs'] = extra_outputs
        lo_kwargs['extra_metrics'] = extra_metrics
        lo_kwargs[LOSS_KEYS.LOSS] = total_loss
        return LossOutput(**lo_kwargs)

    def _elbo_loss(
        self, tensors, inference_outputs, px, qz, pz, *,
        ctx, cls, real_mask, rl_weight, kl_weight,
    ) -> dict:
        """Reconstruction (NB/ZINB log-prob) + KL(qz||pz), with the min-KL clamp and reduction."""
        kl_divergence_z_mat = kl_divergence(qz, pz)

        # Likelihood reconstruction. Cast to float (X may be int16 from the
        # backed dense-cache path) for px.log_prob.
        x = tensors[REGISTRY_KEYS.X_KEY].float()
        reconst_loss_mat = -px.log_prob(x)

        # Reconstruction/KL only on real cells.
        n_obs_minibatch = x.size(0)
        if real_mask is not None:
            _real_mask = real_mask.unsqueeze(-1)
            reconst_loss_mat = reconst_loss_mat * _real_mask
            kl_divergence_z_mat = kl_divergence_z_mat * _real_mask
            n_obs_minibatch = real_mask.sum()

        if self.min_kl is not None:
            kl_divergence_z_mat = torch.clamp(kl_divergence_z_mat, min=self.min_kl)

        kl_divergence_z = kl_divergence_z_mat.sum(-1)          # (b,)
        kl_div_per_latent = kl_divergence_z_mat.sum(0)         # (l,)
        reconst_loss = reconst_loss_mat.sum(-1)                # (b,)
        rl_denom = reconst_loss_mat.size(1)

        if self.reduction == "batchmean":
            pass
        elif self.reduction == "mean":
            kl_divergence_z = kl_divergence_z / self.n_latent
            kl_div_per_latent = kl_div_per_latent / n_obs_minibatch
            reconst_loss = reconst_loss / rl_denom
            if self.scale_mean_kl:
                kl_divergence_z = kl_divergence_z * (self.n_latent / rl_denom)
        else:
            raise ValueError(f'reduction has to be either "batchmean" or "mean", got {self.reduction}')

        L_elbo = rl_weight * reconst_loss + kl_weight * kl_divergence_z
        if self.scale_elbo and self.reduction == 'batchmean':
            L_elbo = L_elbo / rl_denom
        total = L_elbo.mean()
        return {
            'total': total,
            'reconst_loss': reconst_loss,
            'kl_divergence_z': kl_divergence_z,
            'kl_div_per_latent': kl_div_per_latent,
            'n_obs_minibatch': n_obs_minibatch,
        }

    def _efficiency_rank_weight(self, eff: torch.Tensor, ctx_flat: torch.Tensor) -> torch.Tensor:
        """Within-batch, PER-DATASET rank of ``eff`` -> a per-cell weight in [floor, 1].

        Each dataset is ranked against ITS OWN cells in the batch, so the huge cross-dataset efficiency
        bias (raw per-dataset median 0.0-3.3) is removed — a strong cell in low-efficiency Replogle gets
        the same top weight as a strong cell in high-efficiency Marson. Rank (not z-score) is robust to
        the heavy tail + ~11% exact-zero efficiencies. Floored so weak-signal cells still contribute.
        Uses the same ctx grouping as ``_eff_zscore_per_dataset`` (balanced loader gives ~100 cells/dataset
        per batch, enough for a stable percentile)."""
        w = torch.ones_like(eff)
        for d in ctx_flat.unique():
            m = ctx_flat == d
            n = int(m.sum())
            if n < 2:
                continue
            e = eff[m]
            order = e.argsort()
            rank = torch.empty(n, device=eff.device, dtype=eff.dtype)
            rank[order] = torch.arange(n, device=eff.device, dtype=eff.dtype)
            w[m] = rank / (n - 1)                       # within-dataset percentile in [0, 1]
        return self.efficiency_clip_floor + (1.0 - self.efficiency_clip_floor) * w

    def _efficiency_gate(self, z_shared, weights, ctx, *,
                         T_align, lambda_eff, T_gamma, gate_eff) -> dict:
        """Per-dataset-relative efficiency regression + responder gate. Produces
        the per-cell cell_weights and per-cell temperature T_i that the prototype /
        anchor terms consume. Returns its weighted loss contribution in ``term``
        (0.0 when the efficiency head is off / no valid datasets)."""
        out = {'term': 0.0, 'cell_weights': None, 'T_i': None, 'metrics': {}, 'outputs': {}}
        if not (self.model_efficiency and weights is not None and lambda_eff > 0):
            return out
        weights = weights.abs().flatten()
        ctx_flat = ctx.flatten().long()
        eff_hat = self.efficiency_head(z_shared.detach()).flatten()
        # REGRESS THE RANK, NOT RAW EFFICIENCY. `efficiency` is ratio-like with a heavy right tail
        # (median 1.035, max 18.24) and ~11% exact zeros, so a z-score of the raw value is set by a
        # few extreme cells and compresses everyone else toward 0. `_efficiency_rank_weight` was
        # built to avoid exactly that for the LOSS WEIGHT ("Rank (not z-score) is robust to the
        # heavy tail + ~11% exact-zero efficiencies") -- the same argument applies verbatim to the
        # head's TARGET, and it was simply never applied here.
        # It also makes the head predict the quantity we actually CONSUME: the per-dataset
        # percentile in [floor, 1] that weights the PA loss, rather than a raw value we would have
        # to rank-transform afterwards anyway.
        # Still z-scored per dataset below, so the objective stays scale/offset-free
        # (2*(1 - Pearson) on ranks == a Spearman-like criterion).
        _target = (self._efficiency_rank_weight(weights, ctx_flat)
                   if getattr(self, 'efficiency_rank_target', False) else weights)
        eff_z, eff_z_target, per_dataset_mse = self._eff_zscore_per_dataset(eff_hat, _target, ctx_flat)
        if per_dataset_mse:
            eff_loss = torch.stack(per_dataset_mse).mean()
            out['term'] = lambda_eff * eff_loss
            out['metrics']['eff_loss'] = eff_loss
        # Gate/temperature consume the standardized prediction but must not flow
        # grad back through the standardization.
        eff_z = eff_z.detach()
        eff_z_target = eff_z_target.detach()
        gate = torch.sigmoid(self.responder_steepness * (eff_z - self.responder_thr))
        eff_norm = (eff_z.clamp(min=0.0) / 3.0).clamp(max=1.0)
        T_i = T_align * (1 + T_gamma * (1 - eff_norm))
        T_i = T_i.clamp(min=0.07, max=1.0).unsqueeze(-1)
        self.last_t_gamma = T_gamma
        gate_eff = gate_eff.clamp(min=0.0, max=1.0)
        cell_weights = (1 - gate_eff) + gate * gate_eff
        out['cell_weights'] = cell_weights
        out['T_i'] = T_i
        out['outputs'] = {
            'proto/weights': cell_weights,
            'responder_prob': gate,
            'eff_hat': eff_hat,
            REGISTRY_KEYS.CLS_EFF_KEY: weights,
            'eff_hat_scaled': eff_z,
            'eff_scaled': eff_z_target,
            'T_align_dist': T_i,
        }
        return out
