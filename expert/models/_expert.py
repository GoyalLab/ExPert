# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import os
import re
import json
import functools
import contextlib
from expert.utils.constants import REGISTRY_KEYS, PREDICTION_KEYS, MODULE_KEYS
import expert.utils.io as io
import expert.utils.ckpt_compat as ckpt_compat
import expert.utils.legacy as legacy
import expert.utils.performance as pf
from expert.utils._callbacks import DelayedEarlyStopping, SaveCheckPointWithIndices
import expert.utils.plotting as pl
from expert.utils.readout import (calibrated_readout as ro_calibrated_readout, noise_level as ro_noise_level,
                                  deflate_rows as ro_deflate_rows, transplant_scores as ro_transplant_scores,
                                  fit_noise_curve as ro_fit_noise_curve, span_basis as ro_span_basis,
                                  perturbation_prototypes as ro_perturbation_prototypes)
from expert.modules._xpert import XPert
from expert.modules._splitter import DataSplitter
from expert._train.expert_plans import ContrastiveSupervisedTrainingPlan
from expert.data._manager import EmbAnnDataManager

import torch
from torch import Tensor
from torch.distributions import Distribution
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
import numpy.typing as npt


from collections.abc import Sequence
from typing import Literal, Any, Iterator

from anndata import AnnData

# scvi imports
from scvi.model._utils import get_max_epochs_heuristic
from scvi.train import TrainRunner
from scvi.data._utils import _get_adata_minify_type, _check_if_view, _is_minified
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp
from scvi.model.base import (
    ArchesMixin,
    VAEMixin,
    RNASeqMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass
)
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    StringUnsField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi._types import AnnOrMuData

import logging
log = logging.getLogger(__name__)


class ZeroShotFoldMismatch(RuntimeError):
    """The zero-shot subset's ``zs_split`` cannot be derived for the fold being trained, so the
    tag baked in at build time would be used blind. Raised (never warned) because the failure is
    silent-but-wrong: the run completes and reports its headline zero-shot number under the wrong
    arm."""


def _release_query_managers(fn):
    """Drop the AnnDataManagers a call registered for one-off QUERY AnnDatas.

    scvi registers a manager per unseen query in ``_per_instance_manager_store[model.id]`` (via
    ``_validate_anndata`` -> ``transfer_fields``) and never removes it. Each entry holds a strong
    reference to the query AnnData -- which this call has just written an (n_var x width) gene-support
    placeholder into -- so a loop over query subsets retains every one of them. Measured at ~0.7 GB per
    call: 12.7 GB after 5 calls, 127 GB after ~174, then OOM.

    Kept deliberately: managers that already existed before the call (so a query the caller registered
    itself survives) and the model's own training manager. Releasing only means the NEXT call on the
    same object re-runs ``transfer_fields``, which is cheap -- the expensive placeholder is already on
    the object and is not recomputed.

    Re-entrancy matters: ``predict`` and ``get_latent_representation`` can nest, and an inner frame
    releasing its caller's manager would break the outer call, so only the outermost frame releases.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        store = self._per_instance_manager_store
        depth = getattr(self, '_query_mgr_depth', 0)
        self._query_mgr_depth = depth + 1
        before = set(store.get(self.id, {})) if depth == 0 else None
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._query_mgr_depth = depth
            if depth == 0:
                inst = store.get(self.id, {})
                try:
                    _own = self.adata_manager
                    keep = _own.adata_uuid if _own is not None else None
                except Exception:          # no adata registered at all
                    keep = None
                for uid in set(inst) - before:
                    if uid != keep:
                        inst.pop(uid, None)
    return wrapper


class ExPert(
        RNASeqMixin,
        VAEMixin,
        ArchesMixin, 
        UnsupervisedTrainingMixin,
        BaseMinifiedModeModelClass
    ):
    _module_cls = XPert
    _name = __qualname__
    _training_plan_cls = ContrastiveSupervisedTrainingPlan
    _LATENT_QZM_KEY = f"{__qualname__}_latent_qzm"
    _LATENT_QZV_KEY = f"{__qualname__}_latent_qzv"
    _LATENT_Z_KEY = f"{__qualname__}_latent_z"
    _LATENT_Z_SHARED_KEY = f"{__qualname__}_latent_z_shared"
    _LATENT_CTX_PROJ_KEY = f"{__qualname__}_latent_ctx_proj"
    _LATENT_CLS_PROJ_KEY = f"{__qualname__}_latent_cls_proj"
    _LATENT_Z_SHARED_UMAP = f"{__qualname__}_latent_h_umap"
    _LATENT_UMAP = f"{__qualname__}_latent_umap"
    _TRAINING_STATE_FNAME = 'training_state.pt'
    # Python-side training state carried in training_state.pt so a resumed run continues exactly (see save()).
    _RESUME_MODULE_ATTRS = ('_prior_atom_epoch', '_prior_atom_thr', '_prior_refit_n', '_prior_atom_tripped',
                            'last_t_gamma', '_step', '_cur_epoch')
    _RESUME_PLAN_ATTRS = ('loss_kwargs', 'kl_kwargs', '_ctx_seen_ever', '_ctx_n_learned', '_excluded_ctx',
                          'class_batch_counts', 'train_class_counts', '_atom_prev', '_hist_logged_epoch')
    _FINAL_MODEL_DNAME = 'model'
    # Filename for the class/gene embedding persisted alongside the model so
    # it can be re-loaded without depending on the original (often machine-
    # specific) ``class_emb_path``. Layout matches ``_load_class_embedding_df``
    # (.npz with 'embeddings' and 'genes' arrays).
    _GENE_EMB_FNAME = 'gene_embeddings.npz'
    # Set transiently during ``load`` so ``setup_anndata`` can find the
    # embedding saved next to the model. ``None`` outside a load.
    _load_class_emb_dir = None
    # Keys ExPert reads out of its own ``**model_kwargs`` bag. A POPped key never reaches the
    # module, so it appears in NO __init__ signature -- without this declaration the
    # load-time compat filter would drop it. Keep in sync with every
    # ``self._model_kwargs[...]`` / ``.get`` / ``.pop`` in this file.
    _model_level_kwargs = frozenset({
        'class_emb_extra_paths', 'cls_align_loss', 'extra_clip_kwargs', 'extra_cls_emb_kwargs',
        'extra_encoder_kwargs', 'use_cls_head',
    })
    # Set transiently by ``load`` so ``__init__`` drops checkpoint kwargs the current
    # schema no longer accepts. OFF for ordinary construction, so a typo'd knob in a
    # fresh training config still fails loudly instead of being silently discarded.
    _loading_ckpt = False
    _skip_control_db = False   # set by load(load_control_db=False): restore ctrl means from the checkpoint, never read ctrl.h5ad

    # Per-field placeholder defaults for query data missing required columns.
    # Anything not listed here falls through to per-slot defaults (categorical
    # obs: first known category; uns: None).
    _QUERY_NUMERIC_OBS_DEFAULTS = {
        REGISTRY_KEYS.SIZE_FACTOR_KEY: 1.0,
        REGISTRY_KEYS.CLS_EFF_KEY: 1.0,
    }
    _QUERY_JOINT_OBS_DEFAULTS = {
        REGISTRY_KEYS.CAT_COVS_KEY: 'unknown',
        REGISTRY_KEYS.CONT_COVS_KEY: np.nan,
    }


    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers: int = 2,
        n_shared: int | None = None,
        dropout_rate: float = 0.2,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        log_variational: bool = True,
        use_full_cls_emb: bool = False,
        ctrl_class: str | None = None,
        model_log_dir: str | None = None,
        **model_kwargs,
    ):
        super().__init__(adata)
        self._model_kwargs = dict(model_kwargs)
        # CHECKPOINT COMPAT. scvi replays a saved model's recorded __init__ kwargs verbatim,
        # so any parameter deleted from the current schema makes every older checkpoint
        # unloadable. On the LOAD path only, drop what the current classes no longer accept.
        # Gated on the flag rather than run unconditionally so that a typo'd knob in a fresh
        # training config still raises instead of being silently discarded.
        if type(self)._loading_ckpt:
            # Knobs of REMOVED mechanisms are dropped only at their off value (legacy.REMOVED); a checkpoint
            # that trained with one switched on raises here instead of loading as a different model.
            self._model_kwargs = legacy.translate_model_kwargs(self._model_kwargs, source='checkpoint')
            self._model_kwargs, self._dropped_ckpt_kwargs = ckpt_compat.filter_init_kwargs(
                self._model_kwargs, type(self), self._module_cls, context='checkpoint',
            )
            model_kwargs = self._model_kwargs
        else:
            self._dropped_ckpt_kwargs = []
        # Set number of classes
        self.n_labels = self.summary_stats.n_labels
        # Whether to use the full class embedding. Auto-reconcile with the
        # CLIP loss objective: soft-target CE *needs* the full vocab (held-
        # out anchors get supervised via target-distribution mass), and
        # hard CE on the full vocab is actively harmful (every held-out
        # anchor is a negative for every cell, pushing them away from
        # sensible positions). Force consistency rather than asking the
        # user to keep two flags in sync.
        _extra_clip = self._model_kwargs.get('extra_clip_kwargs', {}) or {}
        _soft_on = bool(_extra_clip.get('use_soft_targets', False))
        if _soft_on and not use_full_cls_emb:
            log.info(
                '[Setup] use_soft_targets=True → auto-enabling '
                'use_full_cls_emb=True so held-out anchors enter the loss '
                'via target-distribution mass.'
            )
            use_full_cls_emb = True
        elif (not _soft_on) and use_full_cls_emb:
            # Keep the FULL bank with hard CE — do NOT auto-disable. The old
            # auto-disable feared held-out anchors acting as negatives for
            # training cells, but that never happens: hard-CE training scores
            # ONLY the local trained block via local_predictions=True (see loss
            # `_cls_emb = cls2z[local_targets]`). The full bank is used solely by
            # the zero-shot-pert val arm (local_predictions=False), which needs
            # held-out rows to exist or it indexes OOB and crashes the run.
            log.info(
                '[Setup] use_full_cls_emb=True with hard CE: keeping the FULL bank. '
                'Training scores only the local trained block via local_predictions=True '
                '(held-out are NOT training negatives); held-out anchors are scored only '
                'on the zero-shot-pert val arm.'
            )
        self.use_full_cls_emb = use_full_cls_emb
        self.ctrl_class = ctrl_class
        self.log_variational = log_variational
        # Check if adata.X is raw data or not
        self.raw_x = self._x_is_raw()

        # Initialize indices and labels for this model
        self._setup()
        # Get covariates
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            # Get covariate regsitry
            cov_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            # Get number of categories per covariate
            n_cats_per_cov = cov_registry.n_cats_per_key
            # Get summary string
            cov_summary_string = '\n\t - '.join([f'{k} ({len(v)})' for k, v in cov_registry.mappings.items()])
        else:
            n_cats_per_cov = None
            cov_summary_string = None
        # Determine number of batches/datasets and fix library method
        n_batch = self.summary_stats.n_batch
        # Check number of hidden neurons, set to input features if < 0 else use given number
        n_hidden = self.summary_stats.n_vars if n_hidden < 0 else n_hidden
        # Get class embeddings
        cls_emb, _ = self.get_cls_emb()
        # Extra PRIOR banks for adapter_mode='gated'. These deliberately do NOT go through
        # class_emb_path: that path REQUIRES full label coverage and raises on any missing row, and
        # partial coverage is the defining property here (text ~19.9k genes, DepMap ~18k, JUMP ORF
        # 12.6k, JUMP CRISPR 8.0k). A missing class gets a zero row and present=0, so the gate can
        # learn to ignore it instead of scoring a fake embedding.
        # POP, not get: self._model_kwargs is spread into the module constructor (**), so a
        # leftover model-level key reaches XPertVAE as an unexpected kwarg and kills the run.
        _extra_paths = self._model_kwargs.pop('class_emb_extra_paths', None)
        if _extra_paths:
            if isinstance(_extra_paths, (str, os.PathLike)):
                _extra_paths = [_extra_paths]
            # FLATTEN one level. This key needs DIFFERENT nesting on the two entry points:
            # create_sweep (the run.sh path) treats every list as a sweep axis, so a genuine
            # multi-prior list must be double-wrapped [[a, b]]; but `expert.tune.run --config`
            # does NOT run create_sweep, so the same file reaches here still nested and each
            # element is a LIST. It then str()s to "['....npz']" and dies on the extension.
            # Accepting both makes one config work on both paths.
            _flat = []
            for _p in _extra_paths:
                _flat.extend(_p if isinstance(_p, (list, tuple)) else [_p])
            _extra_paths = _flat
            _n_cls = int(cls_emb.shape[0])
            _labels = [str(x) for x in self.idx_to_label[:_n_cls]]
            _banks, _pres_cols, _names = [], [torch.ones(_n_cls)], []
            for _ep in _extra_paths:
                _df = self._load_class_embedding_df(_ep)
                _df.index = _df.index.astype(str)
                _df = _df[~_df.index.duplicated(keep='first')]
                _sub = _df.reindex(_labels)
                _hit = torch.as_tensor(_sub.notna().all(axis=1).to_numpy(), dtype=torch.float32)
                _M = torch.as_tensor(_sub.fillna(0.0).to_numpy(np.float32))
                # CENTER on the covered rows only. Centering with the zero-filled misses pulls the
                # mean toward zero and reintroduces the shared offset that has repeatedly collapsed
                # the cosine here (measured shared-offset fraction: text 0.477, DepMap 0.314).
                if _hit.sum() > 1:
                    _M = _M - _M[_hit > 0].mean(0, keepdim=True)
                _M = _M * _hit.unsqueeze(1)          # misses go back to exactly zero
                # Scale each block to unit mean row-norm so one source cannot win the gate purely by
                # having a larger native magnitude (DepMap effect sizes vs SapBERT logits).
                _rn = _M[_hit > 0].norm(dim=1).mean() if _hit.sum() > 0 else torch.tensor(1.0)
                _M = _M / _rn.clamp(min=1e-6)
                _banks.append(_M)
                _pres_cols.append(_hit)
                _names.append(os.path.basename(str(_ep)))
                log.info(f'[Setup] Prior source {os.path.basename(str(_ep))}: '
                         f'{int(_hit.sum())}/{_n_cls} classes covered ({float(_hit.mean()):.1%}), '
                         f'{_M.shape[1]} dims.')
            _ext_kw = self._model_kwargs.get('extra_cls_emb_kwargs', {})
            _ext_kw['extra_src_emb'] = _banks
            _ext_kw['extra_src_present'] = torch.stack(_pres_cols, dim=1)
            _ext_kw['extra_src_names'] = _names          # labels the per-source struct metrics
            self._model_kwargs['extra_cls_emb_kwargs'] = _ext_kw
            self._prior_source_names = ['primary'] + _names
            if _ext_kw.get('adapter_mode') != 'gated' and _ext_kw.get('random_init_proxies'):
                # The atom-struct / prior-atom paths consume these banks directly (see
                # ClassEmbedding._atom_priors) while the class projection stays the identity adapter.
                log.info(f"[Setup] class_emb_extra_paths -> atom-struct supports "
                         f"({len(_banks)} extra); the GATED ADAPTER is off "
                         f"(adapter_mode={_ext_kw.get('adapter_mode')!r}), so they shape the "
                         f"struct loss only, not the class projection.")
        # Initialize model
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.n_labels,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_shared=n_shared,
            cls_emb=cls_emb,
            ctrl_class_idx=self.ctrl_class_idx,
            ctrl_gene_means=self.ctrl_gene_means,
            ctrl_gene_stds=self.ctrl_gene_stds,
            log_variational=self.log_variational,
            dropout_rate=dropout_rate,
            n_continuous_cov=self.summary_stats.get('n_extra_continuous_covs', 0),
            n_cats_per_cov=n_cats_per_cov,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            **self._model_kwargs,
        )
        # Wire up the encoder's pert→gene mapping so target-gene masking
        # (set via ``mask_target_prob`` in the encoder kwargs) can find
        # the var-space column to zero per cell. Silently a no-op when
        # the encoder doesn't have ``set_pert_to_gene_mapping`` (e.g. an
        # older encoder type or a buffer wasn't allocated). The mapping
        # ships with the checkpoint as a buffer.
        z_enc = getattr(self.module, 'z_encoder', None)
        if z_enc is not None and hasattr(z_enc, 'set_pert_to_gene_mapping'):
            try:
                n_mapped = z_enc.set_pert_to_gene_mapping(
                    self._label_mapping, self.adata.var_names,
                )
                if getattr(z_enc, 'mask_target_prob', 0.0) > 0:
                    log.info(
                        f'[Setup] Target-gene mask wired: {n_mapped}/{self.n_labels} '
                        f'perturbations mapped to a var column; '
                        f'mask_target_prob={z_enc.mask_target_prob:.2f}.'
                    )
            except Exception as e:
                log.warning(f'[Setup] Could not set pert_to_gene_idx mapping: {e}')
        # Fresh initialization, set params accordingly
        self.supervised_history_ = None
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.is_evaluated = False
        # Stage counter: tracks how many checkpoint-resume cycles this model has been through
        self._training_stage = 0
        self._from_checkpoint = False
        # Setup versioned log directory and logger
        # Defer log-dir creation until training actually starts (see
        # _ensure_log_dir, called from train()). Creating it eagerly here
        # breaks load() when the saved model_log_dir is an absolute path
        # from another machine that can't be created locally.
        self._base_log_dir = model_log_dir
        self._version_dir = None
        self.model_log_dir = None
        self._logger = None
        self._ckpt_dir = None
        # Give model summary
        self.n_unseen_labels = self.adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['n_unseen_labels'] if REGISTRY_KEYS.CLS_EMB_INIT in adata.uns else None
        self._model_summary_string = (
            f"{self.__class__} Model with the following params: \n"
            f"adata shape: {self.adata.shape}, "
            f"n_classes: {self.n_labels}, "
            f"n_unseen_classes: {self.n_unseen_labels}, \n"
            f"n_contexts: {n_batch}"
        )
        # Include covariate information
        if cov_summary_string is not None:
            self._model_summary_string += f"\ncat_covariates: {cov_summary_string}\n"
        # Include control class information
        if self.ctrl_class is not None and self.ctrl_class_idx is not None:
            self._model_summary_string += f"\nctrl_class: {self.ctrl_class}\n"

    def _x_is_raw(self, n_check: int = 100_000):
        # For backed data, sample a small chunk to check
        if self.adata.isbacked:
            sample = self.adata.X[:min(1000, self.adata.n_obs)]
            if sp.issparse(sample):
                data = sample.data
            else:
                data = sample[sample != 0]
        else:
            X = self.adata.X
            if sp.issparse(X):
                data = X.data  # nonzero values only
            else:
                data = X[X != 0]
        # Subsample to avoid allocating a full copy via np.round
        if len(data) > n_check:
            idx = np.random.default_rng(0).choice(len(data), n_check, replace=False)
            data = data[idx]
        # Raw counts: all integers
        return np.allclose(data, np.round(data), atol=1e-6)

    def _setup_batch_labels(self):
        # Assign batch labels
        batch_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        self.original_batch_key = batch_state_registry.original_key
        self._batch_mapping = batch_state_registry.categorical_mapping
        self.idx_to_batch = pd.Series(self._batch_mapping)

    def _setup_labels(self):
        labels_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.original_label_key = labels_state_registry.original_key
        self._label_mapping = labels_state_registry.categorical_mapping
        self._code_to_label = dict(enumerate(self._label_mapping))
        self.idx_to_label = np.array(self._label_mapping)
        self.n_labels = len(self._label_mapping)

    def _setup_cat_cov_labels(self):
        cat_cov_reg = self.adata_manager.registry.get('field_registries').get(REGISTRY_KEYS.CAT_COVS_KEY).get('state_registry')
        if not cat_cov_reg:
            self.idx_to_cat_cov = None
        else:
            idx_to_cat_cov = []
            for k, vs in cat_cov_reg['mappings'].items():
                idx_to_cat_cov.extend(vs)
            self.idx_to_cat_cov = np.array(idx_to_cat_cov)

    def _setup_class_embeddings(self):
        cls_emb_registry = self.adata_manager.registry["field_registries"].get(
            REGISTRY_KEYS.CLS_EMB_KEY, {}
        )

        if len(cls_emb_registry) == 0:
            self.cls_emb = None
            self.train_cls_emb = None
            self.cls_sim = None
            self.train_cls_sim = None
            self.n_train_labels = self.n_labels
            self.n_unseen_labels = 0
            return

        ext_key = cls_emb_registry.get("original_key")
        cls_emb_key = (
            cls_emb_registry["data_registry"].get("attr_key")
            if ext_key is None
            else ext_key
        )

        cls_emb_obj = self.adata.uns[cls_emb_key]
        if isinstance(cls_emb_obj, pd.DataFrame):
            cls_emb_df = cls_emb_obj
        elif isinstance(cls_emb_obj, np.ndarray):
            # Backwards compat: older saves stored cls_emb as ndarray (lost
            # the label index). Reconstruct using idx_to_label; rows beyond
            # n_train are assumed to be unseen labels and named accordingly.
            n_rows = cls_emb_obj.shape[0]
            n_train = len(self.idx_to_label)
            if n_rows < n_train:
                raise ValueError(
                    f'[Setup] Stored class embedding has {n_rows} rows but model expects at least {n_train}.'
                )
            extra = [f'unseen_{i}' for i in range(n_rows - n_train)]
            log.warning(
                f'[Setup] Class embedding stored as ndarray; reconstructing DataFrame from idx_to_label '
                f'({n_train} training labels + {len(extra)} unseen placeholder names).'
            )
            cls_emb_df = pd.DataFrame(cls_emb_obj, index=np.concatenate([self.idx_to_label, extra]))
        else:
            raise ValueError(f'Class embedding must be a DataFrame or ndarray, got {type(cls_emb_obj)}.')

        # Align embedding to label order
        label_series = pd.Series(self.idx_to_label)
        shared = label_series[label_series.isin(cls_emb_df.index)].values
        unseen = cls_emb_df.index.difference(shared).values

        if len(shared) != len(label_series):
            missing = label_series[~label_series.isin(shared)]
            raise ValueError(f"[Setup] Missing {len(missing)} class embeddings for observed labels.")

        ordered_df = pd.concat(
            [cls_emb_df.loc[shared], cls_emb_df.loc[unseen]], axis=0
        )

        self.cls_emb = torch.tensor(ordered_df.values, dtype=torch.float32)
        self.train_cls_emb = self.cls_emb[: len(shared)]
        self.cls_sim = None
        self.train_cls_sim = None

        self.n_train_labels = len(shared)
        self.n_unseen_labels = len(unseen)
        self.idx_to_label = ordered_df.index.values

        self.ext_class_embed_dim = self.cls_emb.shape[1]

        log.info(f"[Setup] Registered {len(self.idx_to_label)} class embeddings with {cls_emb_df.shape[1]} dims.")
        
    def _setup_control_class(self):
        self.ctrl_class_idx = None

        if self.ctrl_class is None:
            return

        # Ensure labels exist
        if self.idx_to_label is None:
            raise RuntimeError("[Setup] Labels must be initialized before setting control class.")

        matches = np.where(self.idx_to_label == self.ctrl_class)[0]

        if len(matches) == 0:
            log.warning(
                f"[Setup] Specified control class '{self.ctrl_class}' not found in labels. Ignoring."
            )
            self.ctrl_class = None
            return

        self.ctrl_class_idx = int(matches[0])

        # Optional: if embeddings exist and control embedding should be learned
        if self.cls_emb is not None:
            if self.ctrl_class_idx >= self.n_train_labels:
                log.info(
                    "[Setup] Control class is not in training labels; embedding will be learned."
                )

        log.info(
            f"[Setup] Registered control class '{self.ctrl_class}' at index {self.ctrl_class_idx}."
        )
        
    def _ctrl_setup_skipped(self, reason: str) -> None:
        """Leave _setup_control_means WITHOUT reading the control DB. The persistent encoder buffers
        (ctrl_gene_means/stds) restore from the checkpoint regardless."""
        self.ctrl_gene_means = None
        self.ctrl_gene_stds = None
        log.info('[Setup] control DB not read (%s): control-derived setup skipped', reason)

    def _setup_control_means(self):
        """Per-(dataset, gene) mean of CONTROL cells, for encoder input centering.

        Loaded from the ``control_adata_path`` setup arg (a small ctrl.h5ad of control/unperturbed
        cells for every dataset, incl. held-out ones), aligned to THIS model's gene order and dataset
        (batch) categories, and put in the SAME space the encoder sees (log1p). Passed to the module as a frozen (n_datasets, n_genes) buffer that the
        token stage subtracts, so a token carries "deviation from control" rather than the dataset
        baseline. ENCODER-ONLY: reconstruction still targets the raw perturbation x.
        """
        from scvi.data._constants import _SETUP_ARGS_KEY
        self.ctrl_gene_means = None
        self.ctrl_gene_stds = None
        _setup_args = self.adata_manager.registry.get(_SETUP_ARGS_KEY, {}) or {}
        ctrl_path = _setup_args.get('control_adata_path', None)
        if not ctrl_path:
            self._ctrl_setup_skipped('no control_adata_path')
            return
        if type(self)._skip_control_db:
            # LOAD FOR INFERENCE: everything predict() needs from the control DB is a PERSISTENT buffer
            # (encoder ctrl_gene_means / ctrl_gene_stds) that load_state_dict restores, and a shipped model
            # usually cannot see the training filesystem anyway. Only features that consume NON-persistent
            # control-derived tensors would need the file, and none of those remain.
            log.info('[Load] skipping control DB %r: ctrl_gene_means/stds restore from the checkpoint; '
                     'predict() centres datasets without a trained baseline on their query controls, '
                     'else the query mean', ctrl_path)
            self._ctrl_setup_skipped('load_control_db=False')
            return
        if not os.path.exists(str(ctrl_path)):
            log.warning(f"[Setup] control_adata_path {ctrl_path!r} not found; skipping control-mean "
                        f"computation (a ctrl_gene_means buffer saved in a checkpoint still restores).")
            self._ctrl_setup_skipped('control_adata_path not found')
            return
        if self.original_batch_key not in (ctrl := ad.read_h5ad(ctrl_path)).obs:
            raise ValueError(f"[Setup] control adata missing batch column '{self.original_batch_key}'.")
        log.info(f"[Setup] Computing per-dataset control gene means from {ctrl_path} "
                 f"({ctrl.n_obs} control cells).")
        # --- gene alignment: control var order -> this model's var_names (unmatched genes stay 0) ---
        model_genes = list(self.adata.var_names)
        n_genes = len(model_genes)
        cvar = {str(g): i for i, g in enumerate(ctrl.var_names)}
        col = np.array([cvar.get(str(g), -1) for g in model_genes])   # control column per model gene
        hit = col >= 0
        if not hit.any():
            log.warning("[Setup] No gene overlap between control adata and model; skipping centering.")
            return
        # --- into the encoder's space: log1p (opt) ---
        Xc = ctrl.X
        if self.raw_x:
            if self.log_variational:
                if sp.issparse(Xc):
                    Xc = Xc.copy(); Xc.data = np.log1p(Xc.data)
                else:
                    Xc = np.log1p(Xc)
        cbatch = ctrl.obs[self.original_batch_key].astype(str).values
        n_datasets = len(self.idx_to_batch)
        means = np.zeros((n_datasets, n_genes), dtype=np.float32)
        # Per-(dataset, gene) control STD, for center_mode='ctrl_z'. Stored as ONES where a dataset
        # or gene has no control support, so dividing by it is a no-op there and 'ctrl_z' degrades
        # exactly to 'ctrl' rather than blowing up on a zero divisor.
        stds = np.ones((n_datasets, n_genes), dtype=np.float32)
        n_filled = 0
        for i, ds in enumerate(self.idx_to_batch):
            m = cbatch == str(ds)
            if not m.any():
                continue
            Xd = Xc[m]
            mu = np.asarray(Xd.mean(axis=0)).flatten()      # mean over this dataset's control cells
            means[i, hit] = mu[col[hit]]                      # scatter onto model gene order
            # E[X^2] - E[X]^2, computed this way so it works for sparse Xd without densifying.
            ex2 = np.asarray(Xd.multiply(Xd).mean(axis=0)).flatten() if sp.issparse(Xd) \
                else np.asarray((Xd ** 2).mean(axis=0)).flatten()
            sd = np.sqrt(np.clip(ex2 - mu ** 2, 0.0, None))
            # A gene with (near-)zero control variance carries no scale information; leaving it at
            # 1.0 avoids amplifying pure noise by a huge 1/sd factor.
            sd = np.where(sd > 1e-3, sd, 1.0)
            stds[i, hit] = sd[col[hit]]
            n_filled += 1
        self.ctrl_gene_means = io.to_tensor(means)
        self.ctrl_gene_stds = io.to_tensor(stds)
        log.info(f"[Setup] control means: {n_filled}/{n_datasets} datasets filled, "
                 f"{int(hit.sum())}/{n_genes} genes matched. "
                 f"stds: median {float(np.median(stds[stds != 1.0])) if (stds != 1.0).any() else float('nan'):.4f}")

    def _setup(self):
        """Setup adata for training. Prepare embeddings."""
        self.cls_emb = None
        # Setup batch labels
        self._setup_batch_labels()
        # Setup class labels
        self._setup_labels()
        # Setup covariate labels
        self._setup_cat_cov_labels()
        # Setup class embeddings
        self._setup_class_embeddings()
        # Setup control class
        self._setup_control_class()
        # Per-dataset CONTROL gene means for encoder input centering (optional)
        self._setup_control_means()
        
    def _setup_log_dir(self, base_log_dir: str | None) -> None:
        """Setup versioned log directory under base_log_dir.

        Layout::

            <base_log_dir>/
              {name}_v0/                      ← version_dir (shared across stages)
                stage_0/                      ← model_log_dir (per-stage: tb logs, evaluate output)
                stage_1/
              {name}_v1/                      ← new version (fresh run)
                ...

        - ``self._version_dir``: ``{base}/{name}_vN/`` — shared across stages.
        - ``self.model_log_dir``: ``{base}/{name}_vN/stage_M/`` — per-stage outputs and tb logs.
        """
        if base_log_dir is None:
            return
        import re
        import lightning.pytorch as _pl
        os.makedirs(base_log_dir, exist_ok=True)
        # Find existing version directories
        pattern = re.compile(rf'^{re.escape(self._name)}_v(\d+)$')
        existing = []
        for entry in os.listdir(base_log_dir):
            m = pattern.match(entry)
            if m and os.path.isdir(os.path.join(base_log_dir, entry)):
                existing.append(int(m.group(1)))
        version = (max(existing) + 1) if existing else 0
        # Set logging output version and path
        self._version = version
        self._version_str = f'{self._name}_v{version}'
        self._logger = _pl.loggers.TensorBoardLogger(
            save_dir=base_log_dir, name='', version=self._version_str,
        )
        self.model_log_dir = self._logger.log_dir
        self._ckpt_dir = os.path.join(self.model_log_dir, 'checkpoints')

    def _ensure_log_dir(self) -> None:
        """Lazily create the versioned log dir on the first ``train`` call.

        No-op when logging is disabled (no base dir) or already set up — e.g.
        restored from a resumed training state in :meth:`load`."""
        if self._logger is None and self._base_log_dir is not None:
            self._setup_log_dir(self._base_log_dir)

    def get_cls_emb(
            self, 
            use_full_cls_emb: bool | None = None, 
            return_dataframe: bool = False,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Helper getter to return either training or full class embedding"""
        if self.cls_emb is None:
            return None, None
        # Check if we want to train on full or observed embedding only
        use_full_cls_emb = use_full_cls_emb if use_full_cls_emb is not None else self.use_full_cls_emb
        if use_full_cls_emb:
            cls_emb, cls_sim = self.cls_emb, self.cls_sim
        else:
            cls_emb, cls_sim = self.train_cls_emb, self.train_cls_sim
        # Return copy of tensor as dataframe
        if return_dataframe:
            return io.tensor_to_df(cls_emb, index=self.idx_to_label), None
        else:
            return cls_emb, cls_sim
        
    @torch.inference_mode()
    def get_prototypes(self, vocabulary: bool = False) -> pd.DataFrame:
        """Perturbation prototypes: the vectors predict() scores a response against.

        One row per perturbation, indexed by name, in the shared latent space (the space of the z_shared
        that ``predict(..., return_latents=True)`` returns). The readout ranks a cell or group response by
        cosine similarity to these rows.

        Parameters
        ----------
        vocabulary
            False: the perturbations seen in training -- what ``predict(obs_only=True)`` ranks.
            True: the whole gene vocabulary -- what ``predict(obs_only=False)`` ranks. Rows of genes never
            perturbed in training hold the prior-graph fill f(missing) when the model carries one (see
            ``add_prior_missing_fill``), otherwise their untrained initialisation.
        """
        mod = self.module
        was_training = mod.training
        mod.eval()
        try:
            c = mod.aligner.project_classes(mod.cls_emb())
            P = mod._class_reps({MODULE_KEYS.CLS_PROJ_KEY: c})
        finally:
            mod.train(was_training)
        if not vocabulary:
            P = P[:int(mod.n_labels)]
        labels = np.asarray(self.idx_to_label).astype(str)[:P.shape[0]]
        return pd.DataFrame(P.detach().float().cpu().numpy(), index=pd.Index(labels, name='perturbation'))

    @torch.inference_mode()
    @_release_query_managers
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5_000,
        batch_size: int | None = 1024,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
        progress_bar: bool = True,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Compute the latent representation of the data.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            :class:`~anndata.AnnData` object with :attr:`~anndata.AnnData.var_names` in the same
            order as the ones used to train the model. If ``None`` and ``dataloader`` is also
            ``None``, it defaults to the object used to initialize the model.
        indices
            Indices of observations in ``adata`` to use. If ``None``, defaults to all observations.
            Ignored if ``dataloader`` is not ``None``
        give_mean
            If ``True``, returns the mean of the latent distribution. If ``False``, returns an
            estimate of the mean using ``mc_samples`` Monte Carlo samples.
        mc_samples
            Number of Monte Carlo samples to use for the estimator for distributions with no
            closed-form mean (e.g., the logistic normal distribution). Not used if ``give_mean`` is
            ``True`` or if ``return_dist`` is ``True``.
        batch_size
            Minibatch size for the forward pass. If ``None``, defaults to
            ``scvi.settings.batch_size``. Ignored if ``dataloader`` is not ``None``
        return_dist
            If ``True``, returns the mean and variance of the latent distribution. Otherwise,
            returns the mean of the latent distribution.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.

        Returns
        -------
        An array of shape ``(n_obs, n_latent)`` if ``return_dist`` is ``False``. Otherwise, returns
        a tuple of arrays ``(n_obs, n_latent)`` with the mean and variance of the latent
        distribution.
        """
        from torch.distributions import Normal
        from torch.nn.functional import softmax

        # Check if model has been trained
        self._check_if_trained(warn=False)
        # Check if both adata and another dataloader are given
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        # Check if model is minified TODO: only works if using model's minified adata
        if self.is_minified and adata is None:
            qzm = self.adata.obsm[self._LATENT_QZM_KEY]
            qzv = self.adata.obsm[self._LATENT_QZV_KEY]
            qz: Distribution = Normal(qzm, qzv.sqrt())
            if return_dist:
                return qzm, qzv
            else:
                return qz

        # Adata is not minified
        if dataloader is None:
            adata = self._validate_anndata(adata, extend_categories=True)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

        zs: list[Tensor] = []
        zs_shared: list[Tensor] = []
        qz_means: list[Tensor] = []
        qz_vars: list[Tensor] = []
        if progress_bar:
            from tqdm import tqdm
            try:
                total = len(dataloader)
            except TypeError:
                total = None
            dl_it = tqdm(dataloader, total=total, unit='batch', desc='Encoding')
        else:
            dl_it = dataloader
        with torch.no_grad():
            for tensors in dl_it:
                outputs: dict[str, Tensor | Distribution | None] = self.module.inference(
                    **self.module._get_inference_input(tensors)
                )

                if MODULE_KEYS.QZ_KEY in outputs:
                    qz: Distribution = outputs.get(MODULE_KEYS.QZ_KEY)
                    qzm: Tensor = qz.loc
                    qzv: Tensor = qz.scale.square()
                else:
                    qzm: Tensor = outputs.get(MODULE_KEYS.QZM_KEY)
                    qzv: Tensor = outputs.get(MODULE_KEYS.QZV_KEY)
                    qz: Distribution = Normal(qzm, qzv.sqrt())
                # Collect distribution parameters
                qz_means.append(qzm.cpu())
                qz_vars.append(qzv.cpu())
                # Get z latent space
                z: Tensor = qzm if give_mean else outputs.get(MODULE_KEYS.Z_KEY)
                # Get shared latent space
                zshared: Tensor = outputs.get(MODULE_KEYS.Z_SHARED_KEY)
                if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                    samples = qz.sample([mc_samples])
                    z = softmax(samples, dim=-1).mean(dim=0)
                # Collect tensors
                zs.append(z.cpu())
                if zshared is not None:
                    zs_shared.append(zshared.cpu())
        # Return output dictionary. .float() upcasts bf16 → fp32 since numpy
        # has no bfloat16 dtype (forward outputs are bf16 under autocast).
        o = {
            MODULE_KEYS.Z_KEY: torch.cat(zs).float().numpy(),
        }
        # Add shared latent space if aligner is on
        if len(zs_shared) > 0:
            o[MODULE_KEYS.Z_SHARED_KEY] = torch.cat(zs_shared).float().numpy()
        # Return latent distributions if possible
        if return_dist:
            o.update({
                MODULE_KEYS.QZM_KEY: torch.cat(qz_means).float().numpy(),
                MODULE_KEYS.QZV_KEY: torch.cat(qz_vars).float().numpy()
            })
        return o
        
    def validate_ctx_emb(self, ctx_emb: torch.Tensor):
        # Extract labels if it's a dataframe
        if isinstance(ctx_emb, pd.DataFrame):
            labels = ctx_emb.index.values
        else:
            labels = None
        # Convert to tensor
        ctx_emb = io.to_tensor(ctx_emb)
        assert self.module.n_ctx_dim != ctx_emb.shape[-1], f'Dimension mismatch, ctx_emb has to have {self.module.n_ctx_dim} dimensions.'
        return labels, ctx_emb

    def validate_cls_emb(self, cls_emb: torch.Tensor):
        """Validate user-supplied class embeddings.

        Accepts either:
          - already-projected embeddings of shape ``(n_cls, n_cls_dim)`` —
            same space the trained class anchors live in (z_shared).
          - raw embeddings of shape ``(n_cls, adapter.in_features)`` —
            the original LLM/text-encoder space; predict() will apply
            the model's adapter to project them into z_shared.

        Row ordering matters at prediction time. ``classify(..., local_predictions=True)``
        slices ``c2z[0:n_labels]`` assuming the first ``n_labels`` rows
        correspond to the model's trained classes in
        ``self.idx_to_label`` order. We enforce that here when the input
        is a labelled DataFrame: trained classes are sorted first (in
        ``idx_to_label`` order), held-out classes follow. Mirrors the
        reorder done in ``_setup_class_embeddings`` at setup time.

        For unlabelled tensor / array input we can't reorder — the caller
        is trusted to have already aligned rows to the model's trained
        label order.
        """
        # Extract labels if it's a dataframe
        if isinstance(cls_emb, pd.DataFrame):
            # Reorder so trained classes come first in ``idx_to_label``
            # order, then held-out classes (in their original order).
            # Same logic as _setup_class_embeddings at setup time.
            # ``get_training_classes()`` returns ONLY the observed labels
            # (n_train_labels) — important when use_full_cls_emb=True, where
            # idx_to_label also contains the n_unseen_labels with-embedding-
            # but-never-in-data classes that should count as held-out here.
            train_labels = pd.Series(self.get_training_classes())
            present_train = train_labels[train_labels.isin(cls_emb.index)].values
            missing_train = train_labels[~train_labels.isin(cls_emb.index)].values
            held_out = cls_emb.index.difference(present_train).values
            if len(missing_train) > 0:
                preview = ', '.join(map(str, missing_train[:5]))
                more = '' if len(missing_train) <= 5 else f', … (+{len(missing_train) - 5} more)'
                log.warning(
                    f'[validate_cls_emb] cls_emb is missing {len(missing_train)}'
                    f'/{len(train_labels)} trained class labels: {preview}{more}. '
                    f'classify(local_predictions=True) will not be able to score '
                    f'cells against those classes — held-out classes will pad the '
                    f'local block instead, which is almost certainly not what you want.'
                )
            cls_emb = pd.concat(
                [cls_emb.loc[present_train], cls_emb.loc[held_out]],
                axis=0,
            )
            labels = cls_emb.index.values
            log.info(
                f'[validate_cls_emb] Reordered cls_emb: {len(present_train)} '
                f'trained + {len(held_out)} held-out classes (total {len(labels)}).'
            )
        else:
            labels = None
        cls_emb = io.to_tensor(cls_emb)
        n_dim = cls_emb.shape[-1]
        n_proj = self.module.n_cls_dim
        adapter = getattr(self.module.cls_emb, 'adapter', None)
        n_raw = getattr(adapter, 'in_features', None)
        if n_dim != n_proj and n_dim != n_raw:
            raise ValueError(
                f'cls_emb dim {n_dim} must match either projected dim '
                f'{n_proj} (z_shared space) or raw dim {n_raw} (LLM space '
                'before the adapter, for auto-projection in predict).'
            )
        return labels, cls_emb

    @torch.no_grad()
    def _extend_batch_buffers(self, dataset_name: str) -> int:
        """Append a row to every per-batch ENCODER buffer for a novel dataset; return its index.

        Only the encoder's lookup buffers are extended (`ctrl_gene_means`, `ctrl_gene_stds`,
        `ds_gene_mean`, `ds_gene_seen`). That is sufficient at PREDICT time:
        `encode_covariates=False`, so no encoder layer is SHAPED by n_batch, and no losses are
        computed -- the decoder covariate embedding and the DANN head keep their trained size.

        New rows start neutral (zeros for means, ones for stds, seen=False) and the caller
        immediately overwrites the control mean from the query's own control cells.
        """
        enc = getattr(self.module, 'z_encoder', None)
        new_idx = None
        for name, fill in (('ctrl_gene_means', 0.0), ('ctrl_gene_stds', 1.0),
                           ('ds_gene_mean', 0.0)):
            buf = getattr(enc, name, None)
            if buf is None:
                continue
            pad = torch.full((1,) + tuple(buf.shape[1:]), fill,
                             dtype=buf.dtype, device=buf.device)
            setattr(enc, name, torch.cat([buf, pad], dim=0))
            new_idx = getattr(enc, name).shape[0] - 1
        seen = getattr(enc, 'ds_gene_seen', None)
        if seen is not None:
            setattr(enc, 'ds_gene_seen',
                    torch.cat([seen, torch.zeros(1, dtype=seen.dtype, device=seen.device)]))
        # invariant_align's per-dataset z_shared baselines are indexed by the SAME batch index in
        # classify() (_center_z_shared). A novel dataset gets count 0, which makes the eval branch
        # self-centre it on the query mean (or on predict()'s resolved align_center when one is passed).
        for name in ('align_ds_mean', 'align_ds_count'):
            buf = getattr(self.module, name, None)
            if isinstance(buf, torch.Tensor) and buf.dim() >= 1 and new_idx is not None and buf.shape[0] <= new_idx:
                pad = torch.zeros((new_idx + 1 - buf.shape[0],) + tuple(buf.shape[1:]), dtype=buf.dtype, device=buf.device)
                setattr(self.module, name, torch.cat([buf, pad], dim=0))
        # keep the model-side mapping in step so idx_to_batch round-trips
        try:
            self.idx_to_batch = pd.Series(list(self.idx_to_batch) + [dataset_name])
        except Exception:
            pass
        log.info('[Predict] extended per-batch encoder buffers for novel dataset %r -> index %s',
                 dataset_name, new_idx)
        return int(new_idx)

    def refresh_ctrl_gene_means(self, adata, control_label: str | None = None,
                                restore: bool = False, min_ctrl: int = 10,
                                fallback: str | None = 'query_mean') -> dict:
        """Recompute the encoder's per-dataset CONTROL means from THIS query's own control cells.

        WHY THIS EXISTS. ``ctrl_gene_means`` is a ``(n_batch, n_gene)`` buffer built ONCE at setup
        from the TRAINING control database. A query dataset the model never saw has no row, so the
        encoder falls back to the EMA ``ds_gene_mean`` -- a TRAINING baseline. Since the encoder is
        trained exclusively on control-centred input, a wrong baseline is a silent distribution
        shift that reads as catastrophic far-OOD performance rather than as a bug. For any novel
        dataset the centering must come from the query's OWN controls.

        The transform mirrors ``_setup_control_means`` exactly (log1p per ``self.raw_x`` /
        ``log_variational``) and scatters onto the model's var order, so the values
        live in the same space as the trained buffer.

        Returns a per-dataset report; pass ``restore=True`` to get the previous buffer back so the
        model is not permanently mutated.
        """
        import scipy.sparse as _sp
        enc = getattr(self.module, 'z_encoder', None)
        buf = getattr(enc, 'ctrl_gene_means', None)
        if buf is None:
            log.warning('[Predict] model has no ctrl_gene_means buffer; nothing to refresh.')
            return {}
        label = control_label if control_label is not None else self.ctrl_class
        if label is None and fallback != 'query_mean':
            raise ValueError('refresh_ctrl_gene_means needs control_label (or model.ctrl_class).')
        cls_col = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY).original_key
        if cls_col not in adata.obs or self.original_batch_key not in adata.obs:
            raise KeyError(f'query needs obs[{cls_col!r}] and obs[{self.original_batch_key!r}]')
        prev = buf.detach().clone() if restore else None

        # query var -> model var (unmatched model genes keep their existing value)
        model_genes = list(self.adata.var_names)
        qvar = {str(g): i for i, g in enumerate(adata.var_names)}
        col = np.array([qvar.get(str(g), -1) for g in model_genes])
        hit = col >= 0

        lab = adata.obs[cls_col].astype(str).values
        dsq = adata.obs[self.original_batch_key].astype(str).values
        is_ctrl = (lab == str(label)) if label is not None else np.zeros(len(lab), dtype=bool)
        report = {}
        idx_of = {str(d): i for i, d in enumerate(self.idx_to_batch)}
        if not is_ctrl.any():
            if fallback != 'query_mean':
                log.warning('[Predict] no cells with %s == %r in the query -- control means NOT '
                            'refreshed; a novel dataset will use a TRAINING baseline.', cls_col, label)
                return {'_no_control_cells': True}
            report['_no_control_cells'] = True

        def _has_trained_row(d):
            bi_ = idx_of.get(str(d))
            return bi_ is not None and bi_ < buf.shape[0] and bool(buf[bi_].abs().sum() > 0)

        for d in np.unique(dsq):
            m = is_ctrl & (dsq == d)
            source = 'ctrl'
            # ``min_ctrl`` is a CELL-count heuristic. A query of PSEUDOBULKS carries far more
            # control signal per row (8 rows x 742-4,823 cells each here), so the caller can lower
            # it deliberately; the default is unchanged.
            if m.sum() < min_ctrl:
                if _has_trained_row(d) or fallback != 'query_mean':
                    # a dataset seen in training keeps its stored control baseline (restored from the
                    # checkpoint); only NOVEL datasets need a query-derived one
                    report[str(d)] = {'n_ctrl': int(m.sum()), 'updated': False,
                                      'reason': (f'fewer than {min_ctrl} control cells; keeping the trained baseline'
                                                 if _has_trained_row(d) else f'fewer than {min_ctrl} control cells')}
                    continue
                # QUERY-MEAN FALLBACK for a novel dataset without controls: measured almost
                # indistinguishable from control centring on the OOD readout (0.0599 vs 0.0611)
                m = dsq == d
                source = 'query_mean'
            X = adata[m].X
            X = X.toarray() if _sp.issparse(X) else np.asarray(X)
            if self.raw_x and self.log_variational:
                X = np.log1p(X)
            mu = X.mean(0)
            bi = idx_of.get(str(d))
            if bi is None or bi >= buf.shape[0]:
                # NOVEL dataset -> EXTEND the per-batch buffers rather than skip. Safe because the
                # encoder never takes the dataset as a categorical input (encode_covariates=False),
                # so nothing is SHAPED by n_batch on the encode path -- only these lookup buffers
                # are indexed by it. predict() computes no losses, so the decoder/DANN heads (which
                # ARE sized to the trained n_batch) are irrelevant here.
                bi = self._extend_batch_buffers(str(d))
                idx_of[str(d)] = bi
                buf = enc.ctrl_gene_means          # rebind: the buffer object was replaced
                report.setdefault('_extended', []).append(str(d))
            row = buf[bi].detach().clone()
            row[torch.as_tensor(np.flatnonzero(hit))] = torch.as_tensor(
                mu[col[hit]], dtype=row.dtype, device=row.device)
            buf[bi] = row
            report[str(d)] = {'n_ctrl': int(m.sum()) if source == 'ctrl' else 0, 'updated': True,
                              'batch_index': int(bi), 'genes_matched': int(hit.sum()), 'source': source,
                              'n_cells': int(m.sum())}
        # `report` mixes per-dataset DICTS with bookkeeping entries under underscore keys
        # ('_extended' is a LIST, '_no_control_cells' a bool), so every consumer must filter by
        # type -- an unguarded v.get() here raised "'list' object has no attribute 'get'" and took
        # down any predict(refresh_ctrl_means=True) on a novel dataset that needed extending.
        _per_ds = {k: v for k, v in report.items()
                   if not str(k).startswith('_') and isinstance(v, dict)}
        log.info('[Predict] refreshed ctrl_gene_means for %d/%d query datasets',
                 sum(1 for v in _per_ds.values() if v.get('updated')), len(_per_ds))
        if restore and prev is not None:
            report['_prev'] = prev
        return report

    def _build_center_by_bank(self, adata, indices, center_by, control_label, min_ctrl: int = 10):
        """Per-(group, dataset) CONTROL-mean bank + per-cell codes, for ``predict(center_by=...)``.

        THE LOOP THIS REPLACES. Calling predict() once per group on a subset makes
        ``refresh_ctrl_gene_means`` rewrite ``ctrl_gene_means`` from that subset's own controls, so a
        cell ends up centred on the control mean of ITS group -- per-group centring achieved by
        serialising. Here every group's row is computed up front and the encoder gathers one per cell,
        which is the same arithmetic in one pass.

        Keyed by the (group, dataset) PAIR, not the group: a subset spanning two datasets updates two
        buffer rows in the loop, and the pair key reproduces that exactly. Everything else mirrors
        ``refresh_ctrl_gene_means`` -- the same CPM/log1p transform, the same query->model var scatter,
        the same ``min_ctrl`` threshold, the same 'a dataset with a trained row keeps it' rule and the
        same query-mean fallback -- so the two paths agree to the bit.

        Returns ``(bank, codes, report)``: bank is (n_pairs, n_gene), codes is aligned to ``indices``.
        """
        import scipy.sparse as _sp
        enc = getattr(self.module, 'z_encoder', None)
        buf = getattr(enc, 'ctrl_gene_means', None)
        if buf is None:
            raise ValueError('center_by needs a model whose encoder has ctrl_gene_means '
                             "(trained with center_mode='ctrl'/'ctrl_z').")
        if center_by not in adata.obs:
            raise KeyError(f'center_by={center_by!r} is not a column of the query .obs')
        cls_col = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        bkey = self.original_batch_key
        if bkey not in adata.obs:
            raise KeyError(f'query needs obs[{bkey!r}] to build a per-(group, dataset) bank')
        grp_all = adata.obs[center_by].astype(str).values
        ds_all = adata.obs[bkey].astype(str).values
        label = control_label if control_label is not None else getattr(self, 'ctrl_class', None)
        lab_all = adata.obs[cls_col].astype(str).values if cls_col in adata.obs else None
        is_ctrl = (lab_all == str(label)) if (lab_all is not None and label is not None) \
            else np.zeros(len(grp_all), dtype=bool)

        model_genes = list(self.adata.var_names)
        qvar = {str(g): i for i, g in enumerate(adata.var_names)}
        col = np.array([qvar.get(str(g), -1) for g in model_genes])
        hit = col >= 0
        hit_idx = torch.as_tensor(np.flatnonzero(hit))
        idx_of = {str(d): i for i, d in enumerate(self.idx_to_batch)}

        sel = np.asarray(indices)
        in_sel = np.zeros(adata.n_obs, dtype=bool)
        in_sel[sel] = True
        order, codes = {}, np.empty(len(sel), dtype=np.int64)
        for j, i in enumerate(sel):
            key = (str(grp_all[i]), str(ds_all[i]))
            if key not in order:
                order[key] = len(order)
            codes[j] = order[key]
        keys = [k for k, _ in sorted(order.items(), key=lambda kv: kv[1])]

        bank = torch.zeros(len(keys), buf.shape[1], dtype=buf.dtype)
        report = []
        for k, (g, d) in enumerate(keys):
            bi = idx_of.get(str(d))
            has_row = bi is not None and bi < buf.shape[0]
            base = buf[bi].detach().clone() if has_row else torch.zeros(buf.shape[1], dtype=buf.dtype)
            m = in_sel & (grp_all == g) & (ds_all == d)
            mc = m & is_ctrl
            src = 'ctrl'
            if int(mc.sum()) < min_ctrl:
                if has_row and bool(buf[bi].abs().sum() > 0):
                    # a dataset seen in training keeps its stored baseline -- same rule as the loop
                    bank[k] = base
                    report.append((g, d, int(mc.sum()), 'trained_baseline'))
                    continue
                mc, src = m, 'query_mean'
            X = adata[mc].X
            X = X.toarray() if _sp.issparse(X) else np.asarray(X)
            if self.raw_x and self.log_variational:
                X = np.log1p(X)
            row = base.clone()
            row[hit_idx] = torch.as_tensor(X.mean(0)[col[hit]], dtype=row.dtype)
            bank[k] = row
            report.append((g, d, int(mc.sum()), src))
        return bank, codes, report

    def _auto_align_center(self, adata, center_mode, control_label, indices=None):
        """Pick a center_mode automatically when the query contains UNTRAINED datasets.

        Returns `center_mode` unchanged if the caller already chose one, if the module was not
        trained with invariant_align, or if every dataset in the query already has a trained
        baseline. Otherwise returns 'control' (preferred) or 'query_mean'.

        Deciding this from obs is deliberate: it costs one value_counts() rather than a pass over
        the data, and the pass only happens if a baseline is actually missing.
        """
        if center_mode is not None:
            return center_mode
        if not getattr(self.module, 'invariant_align', False):
            return center_mode
        cnt = getattr(self.module, 'align_ds_count', None)
        if cnt is None:
            return center_mode
        _q = adata if adata is not None else self.adata
        bkey = getattr(self, 'original_batch_key', None)
        # RESTRICT TO THE ROWS ACTUALLY BEING PREDICTED. predict(indices=...) with adata=None
        # scores a SUBSET of self.adata, but reading obs over the whole object sees every
        # registered dataset -- including held-out ones that have no trained baseline. That made
        # an ID-only query auto-resolve to 'query_mean', replacing the trained align_ds_mean for
        # the trained datasets too, and per-cell accuracy fell to CHANCE (0.004 vs the training
        # loop's clip_accuracy_local of 0.20) while the cosine pseudobulk readout, which is
        # invariant to a per-dataset offset, showed nothing.
        _obs = _q.obs if _q is not None else None
        if _obs is not None and indices is not None:
            _ix = np.asarray(indices)
            # NOT `adata is None`: predict() has already reassigned `adata` to self.adata via
            # _validate_anndata by the time this runs, so that test never fires.
            if _ix.dtype != bool and len(_ix) != len(_obs):
                _obs = _obs.iloc[_ix]
            elif _ix.dtype == bool:
                _obs = _obs.iloc[np.flatnonzero(_ix)]
        if _q is None or bkey is None or bkey not in _obs:
            log.warning('[Predict] invariant_align is on but the query has no %r column; cannot '
                        'check which datasets need a z_shared baseline.', bkey)
            return center_mode
        try:
            mapping = list(self.adata_manager.get_state_registry(
                REGISTRY_KEYS.BATCH_KEY).categorical_mapping)
        except Exception as e:                                        # noqa: BLE001
            log.warning('[Predict] could not read the batch mapping (%s); skipping the '
                        'z_shared baseline check.', e)
            return center_mode
        pos = {str(v): i for i, v in enumerate(mapping)}
        present = {pos[str(v)] for v in _obs[bkey].astype(str).unique() if str(v) in pos}
        cnt_l = cnt.detach().cpu().long().tolist()
        missing = sorted(i for i in present if i < len(cnt_l) and cnt_l[i] == 0)
        # A query that arrived WITHOUT a batch column had one fabricated for it: every cell was
        # assigned mapping[0] plus a sidecar `_placeholder_<key>` flag (see
        # _add_missing_query_registry). Those cells are not from that dataset, so its trained
        # align_ds_mean is the wrong baseline -- a different tissue and platform -- yet its count
        # is non-zero, so the check above sees nothing missing and the centering silently uses it.
        # Treat a fabricated batch like a never-trained dataset and derive the baseline from the
        # query itself.
        ph_key = f'_placeholder_{bkey}'
        forced = bool(ph_key in _obs
                      and np.asarray(_obs[ph_key]).astype(bool).any())
        if not missing and not forced:
            return center_mode
        names = [str(mapping[i]) for i in missing]
        # Prefer CONTROLS: the query mean is contaminated by perturbation signal.
        lbl = control_label if control_label is not None else getattr(self, 'ctrl_class', None)
        lkey = None
        try:
            lkey = self.adata_manager.get_state_registry(
                REGISTRY_KEYS.LABELS_KEY).original_key
        except Exception:                                             # noqa: BLE001
            pass
        has_ctrl = bool(lbl is not None and lkey is not None and lkey in _obs
                        and (_obs[lkey].astype(str) == str(lbl)).any())
        # For a fabricated batch, self-centre on the query mean even when controls exist. A
        # caller in this position is running inference on a novel dataset and typically applies
        # its own control contrast downstream (see expert.utils.readout); subtracting the control
        # mean here as well would fold that comparison into the latent and conflate the two
        # stages. The genuinely-never-trained case below keeps preferring controls, because the
        # in-training OOD-test metric that path serves has no downstream contrast of its own.
        chosen = 'query_mean' if forced else ('control' if has_ctrl else 'query_mean')
        if forced and not missing:
            log.warning(
                '[Predict] invariant_align: the query had no %r column, so every cell was assigned '
                'the placeholder category %r. That dataset\'s trained z_shared baseline belongs to '
                'different cells, so it is NOT used. Auto-resolving with center_mode=%r over one '
                'extra pass (pass center_mode explicitly to override).',
                bkey, str(mapping[0]), chosen)
            return chosen
        log.warning(
            '[Predict] invariant_align: %d/%d query dataset(s) have NO trained z_shared baseline '
            '(%s). Their stored align_ds_mean row is zero, so centring would fall back to the '
            'batch query mean. Auto-resolving with center_mode=%r over one extra pass.%s',
            len(missing), len(present), ', '.join(names[:6]) + ('...' if len(names) > 6 else ''),
            chosen,
            '' if has_ctrl else ' NOTE no control cells found in the query for '
                                f'control_label={lbl!r}; the query mean carries perturbation '
                                'signal and is the weaker baseline.')
        return chosen

    def _resolve_align_center(
        self,
        scdl,
        center_mode,
        control_label,
        align_col_map,
        align_n_model_vars,
        ctx_emb,
        cls_emb,
        use_posterior_mean,
    ) -> torch.Tensor | None:
        """Compute the ``z_shared`` baseline subtracted at inference (control-free
        differential). Returns ``None`` (no centering) unless the module trained with
        ``invariant_align`` AND ``center_mode`` is given; see :meth:`predict`.

        For ``'query_mean'`` / ``'control'`` it runs ONE extra no-grad pass over the
        same dataloader, accumulating the RAW (un-centered) ``z_shared`` — the exact
        space the module's EMA ``align_ds_mean`` lives in — summed per dataset, then
        returns a ``(n_batch, n_shared)`` bank. Datasets absent from the reference set
        keep their trained ``align_ds_mean`` row, so mixed seen+novel queries work.
        """
        if center_mode is None:
            return None
        if not getattr(self.module, 'invariant_align', False):
            log.warning(
                '[Predict] center_mode=%r ignored: module was not trained with '
                'invariant_align, so z_shared centering is a no-op.', center_mode)
            return None
        # A ready-made center (vector or per-dataset bank) passes straight through.
        if isinstance(center_mode, (np.ndarray, torch.Tensor)):
            return io.to_tensor(center_mode).float().to(self.device)
        if center_mode not in ('query_mean', 'control'):
            raise ValueError(
                "center_mode must be 'query_mean', 'control', an array/tensor, or "
                f"None; got {center_mode!r}.")
        # Resolve the control label index (into idx_to_label) for 'control' mode.
        ctrl_idx = None
        if center_mode == 'control':
            lbl = control_label if control_label is not None else getattr(self, 'ctrl_class', None)
            if lbl is None:
                raise ValueError(
                    "center_mode='control' needs control_label (or a set self.ctrl_class).")
            hits = np.where(np.asarray(self.idx_to_label) == lbl)[0]
            if len(hits) == 0:
                raise ValueError(f"control_label {lbl!r} not found in idx_to_label.")
            ctrl_idx = int(hits[0])
        n_batch, n_shared = self.module.align_ds_mean.shape
        sums = torch.zeros(n_batch, n_shared, device=self.device)
        counts = torch.zeros(n_batch, device=self.device)
        from tqdm import tqdm
        try:
            total = len(scdl)
        except TypeError:
            total = None
        for tensors in tqdm(scdl, total=total, unit='batch', desc=f'Centering ({center_mode})'):
            x = tensors.get(REGISTRY_KEYS.X_KEY)
            if align_col_map is not None and x is not None:
                from expert.utils.common import align_x_batch
                x = align_x_batch(x, align_col_map, align_n_model_vars)
            batch = tensors.get(REGISTRY_KEYS.BATCH_KEY)
            l = tensors.get(REGISTRY_KEYS.LABELS_KEY)
            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
            eff_scores = tensors.get(REGISTRY_KEYS.CLS_EFF_KEY)
            out = self.module.classify(
                x, batch_index=batch, label=l, cat_covs=cat_covs, cont_covs=cont_covs,
                eff_scores=eff_scores, use_posterior_mean=use_posterior_mean,
                inference_outputs=None, ctx_emb=ctx_emb, cls_emb=cls_emb,
                return_logits=True, local_predictions=False, align_center=None)
            # RAW, un-centred -- and it must be read from the dedicated key. Z_SHARED_KEY now
            # carries the CENTRED latent (so predict() returns what the model actually scores),
            # and averaging that would yield a ~zero baseline: the eval branch already centres
            # when align_center is None.
            zs = out.get('z_shared_raw')
            if zs is None:
                zs = out.get(MODULE_KEYS.Z_SHARED_KEY)
                log.warning('[Predict] classify() did not publish z_shared_raw; falling back to '
                            'the (possibly centred) Z_SHARED_KEY -- the resolved baseline may be '
                            'degenerate.')
            if zs is None:
                continue
            b = batch.flatten().long().to(self.device)
            zs = zs.float().to(self.device)
            if ctrl_idx is not None:
                if l is None:
                    raise ValueError("center_mode='control' needs labels; query adata has none.")
                keep = l.flatten().long().to(self.device) == ctrl_idx
                if not keep.any():
                    continue
                b, zs = b[keep], zs[keep]
            sums.index_add_(0, b, zs)
            counts.index_add_(0, b, torch.ones_like(b, dtype=sums.dtype))
        # Seed with the trained per-dataset means; overwrite only observed datasets.
        bank = self.module.align_ds_mean.detach().clone().to(self.device)
        seen = counts > 0
        if seen.any():
            bank[seen] = sums[seen] / counts[seen].unsqueeze(1)
        n_seen = int(seen.sum())
        if n_seen == 0:
            log.warning(
                '[Predict] center_mode=%r found no reference cells; falling back to '
                'the trained align_ds_mean.', center_mode)
        else:
            log.info(
                '[Predict] center_mode=%r: z_shared baseline computed for %d/%d datasets '
                'over %d reference cells.', center_mode, n_seen, n_batch, int(counts.sum().item()))
        return bank

    def _resolve_head(self, head: str = 'auto') -> tuple[str, str]:
        """Resolve predict()'s `head` to 'A' (cosine over the class bank / atoms) or 'H' (free classifier).

        'A' / 'H' are returned as given. For 'auto', in order of evidence:
          1. no free head (use_cls_head False)                         -> 'A'
          2. saved training record `cls_head_training_` (set by train()) -> 'H' iff the head loss was applied
             on at least one training step with a non-zero weight, else 'A'
          3. no record (checkpoints trained before the record existed): the head's weight rows. Random init
             gives per-row norms with coefficient of variation ~0.5*sqrt(0.8/d) (d = row length; 0.0099 at
             2048); measured untrained heads (GRAPH0, PA_F4) sit at 0.0090-0.0097, trained ones at
             0.017 (weight 20) - 0.036 (weight 200). CV > 1.4x the untrained value -> 'H', else 'A'.
        """
        head = str(head)
        if head in ('A', 'H'):
            return head, 'explicit'
        mod = self.module
        if not getattr(mod, 'use_cls_head', False):
            return 'A', 'model has no free classifier head (use_cls_head=False)'
        rec = getattr(self, 'cls_head_training_', None)
        if isinstance(rec, dict) and 'steps' in rec:
            st, wm = int(rec.get('steps', 0)), float(rec.get('weight_max', 0.0))
            if st > 0 and wm > 0:
                return 'H', f'training record: head loss applied on {st:,} steps (max weight {wm:g})'
            return 'A', (f'training record: head loss never applied (steps {st}, max weight {wm:g}) -- '
                         f'use_cls_head=True but the head is UNTRAINED')
        W = getattr(mod, 'cls_head_w', None)
        if W is None and getattr(mod, 'classifier', None) is not None:
            _lin = [m for m in mod.classifier.modules() if isinstance(m, torch.nn.Linear)
                    and m.weight.shape[0] == int(getattr(mod, 'n_labels', -1))]
            W = _lin[-1].weight if _lin else None
        if W is None or W.dim() != 2 or W.shape[0] < 2:
            return 'H', 'no training record and no head weight matrix to inspect -- legacy rule (use_cls_head)'
        with torch.no_grad():
            rn = W.detach().float().norm(dim=1)
            cv = float(rn.std() / rn.mean().clamp_min(1e-12))
        exp_cv = 0.5 * float(np.sqrt(0.8 / W.shape[1]))
        if cv > 1.4 * exp_cv:
            return 'H', f'no training record; head row-norm CV {cv:.4f} > 1.4x the untrained value {exp_cv:.4f}'
        return 'A', (f'no training record; head row-norm CV {cv:.4f} ~ untrained value {exp_cv:.4f} -- '
                     f'head looks UNTRAINED')

    @_release_query_managers
    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = 4096,
        use_posterior_mean: bool = True,
        ctx_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        return_latents: bool = True,
        progress_bar: bool = True,
        obs_only: bool = False,
        add_report: bool = False,
        add_missing_categories: bool = True,
        feature_mask_strategy: Literal['inferred', 'shared', 'all'] = 'inferred',   # accepted, no effect (feature masks were removed)
        head: Literal['auto', 'A', 'H'] = 'auto',
        hub_correction: bool = False,             # subtract the stored per-atom popularity offset (add_hub_correction); no-op without it
        tent_iters: int = 0,
        tent_lr: float = 1e-4,
        inplace: bool = False,
        copy: bool = False,
        n_cores: int | None = None,
        minify: bool = False,
        soft_predictions: bool = True,
        top_k_logits: int | None = None,
        center_mode: str | np.ndarray | torch.Tensor | None = None,
        # CENTRE EACH CELL ON ITS OWN GROUP'S CONTROLS, IN ONE PASS. An obs column (e.g. cell type,
        # or cell type x donor). Without it the only way to get a per-group input baseline is to call
        # predict() once per group on a subset, which is what every centring script does today; this
        # precomputes one control-mean row per (group, dataset) and has the encoder gather per cell,
        # giving the same arithmetic in a single pass. See _build_center_by_bank.
        center_by: str | None = None,
        control_label: str | None = None,
        refresh_ctrl_means: bool | None = None,  # None = AUTO: mirror the encoder's training-time centering
        # DECODE THE QUERY'S RESPONSE over the perturbation prototypes: expert.utils.readout.calibrated_readout --
        # iterated full cosine readout with deflation, a depth-matched noise stop (add_readout_calibration) and
        # PM-2 merging with the first `decode_head` slots held by round 1, so the top-`decode_head` list IS the
        # cosine readout and ranks after it carry what the dominant mechanism left unexplained. Replaced the OMP
        # decode (2026-10-15): OMP's depth-200 profiles carry ~no set enrichment on ALD (0-0.5 Reactome sets per
        # group vs 39), and its doubles edge on GSE146194 is the target transcript's free label (all methods at the
        # floor once targets are masked). OMP stays available as readout.nnomp.
        decode_perturbations: bool = False,
        decode_center: str = 'auto',             # 'control' | 'none' | 'auto'
        # WHETHER TO SUBTRACT A CONTROL MEAN FROM z_shared BEFORE DECODING. This is NOT the same
        # centring the encoder does, and it was applied unconditionally. With invariant_align the
        # module has ALREADY expressed z_shared as a difference from the dataset baseline
        # (_center_z_shared, EMA align_ds_mean) and the atoms were CLIP-aligned to THAT tensor --
        # so subtracting controls again is a second, training-unseen centring that adds the SAME
        # -ctrl_z term to every group, i.e. it injects a shared axis rather than removing one.
        # Measured on GSE146194 (72 labelled conditions, 8,964 atoms): full-vocabulary top-1
        # 0.167 with the subtraction, 0.278 without. 'auto' = 'none' when the module was trained
        # with invariant_align, else 'control' (an uncentred model still needs the reference).
        decode_group_key: str | None = None,     # obs column to pseudobulk over; None = whole query
        # decode_head = slots held by round 1 (the cosine readout), decode_rounds = max rounds,
        # decode_top = merged list length per tail.
        decode_head: int = 10,
        decode_rounds: int = 3,
        decode_top: int = 200,
        # PER-CELL readout confidence: obs 'readout_confidence' = max cosine of the cell to the perturbation
        # prototypes (validated per cell: AUROC .76 for truth-in-top-10, 11x top-10 rate from the lowest to
        # the highest quintile) and 'readout_effect' = ||z|| / noise95 of a single control cell (needs
        # add_readout_calibration; NaN otherwise). Uses the same baseline as the decode (decode_center).
        readout_confidence: bool = True,
        # 'cosine' (DEFAULT) returns the plain logits. 'deflate' (expert.utils.readout.deflate_rows) judges every row
        # at its own depth and decodes past round 1 only while its explainable residual exceeds the noise of a
        # control delta of that many cells; the merged order is transplanted onto the row's own logits. A single
        # cell (n = 1) almost never clears it, so cell-level deflated predictions equal the cosine logits.
        # 'deflate' is the combinatorial readout: it recovers the SECOND perturbation of a double and is worth using when the query is known to be a combination,
        # but it does not beat cosine on single perturbations -- over 10,061 fold_4 OOD singles no swept
        # configuration improved on cosine, and its leading set contains a truly perturbed gene for only ~26%
        # of doubles against the full atom table. Needs add_readout_calibration.
        readout: Literal['deflate', 'cosine'] = 'cosine',
        n_cells_key: str | None = None,          # obs column: cells behind each row (pseudobulk / bulk); None = 1
        # EFFECT SIZE: obs 'effect_size' = the stored readout (add_effect_head): the trained efficiency head combined with
        # the cosine confidence of the call, cosines taken on each row centred on the control cells of its own
        # `effect_center_key` group (default: the batch key; falls back to all query controls, then the group mean
        # with a warning). No-op without a stored readout.
        effect_size: bool = True,
        effect_center_key: str | None = None,
        # NON-HUMAN QUERIES (expert.utils.orthologs): 'auto' moves a mouse / rat symbol space onto human symbols when
        # its var_names match the model only once upper-cased; 'off' never converts. ortholog_table = an MGI-style
        # homology table (HOM_MouseHumanSequence.rpt); without one the conversion upper-cases (logged as a fallback).
        orthologs: Literal['auto', 'off'] = 'auto',
        ortholog_table: str | None = None,
        ortholog_taxon: int = 10090,
        # INPUT GENE MASK: model genes matching any of these glob patterns (case-insensitive, e.g. 'MT-*') are set
        # to their control baseline in the encoder input (the centred input is zeroed) for every row of this call,
        # controls included; the model is restored afterwards. Use 'MT-*' for queries that are NOT Perturb-seq
        # screens (drug screens, disease atlases, other non-CRISPR data): there the mitochondrial fraction mostly
        # tracks cell stress and quality rather than the perturbation, and masking it helped in almost every such
        # case (Tahoe-100M: every MT-masked arm beat every unmasked one, ~1.7x recall@10). Keep it off for
        # CRISPR screens like the training data, where MT- genes carry signal (fold_4 OOD top-10 -0.0085 when
        # masked). Masking genes that are already zeroed upstream is a no-op.
        mask_genes: str | Sequence[str] | None = None,
        **kwargs
    ) -> dict[str, pd.DataFrame | np.ndarray] | AnnData | None:
        """Encode a query and read it out against the model's perturbation prototypes.

        Every cell (or pseudobulk row) is encoded to ``z_shared`` and scored against one prototype per
        trained perturbation. The scores come back as ``soft_predictions`` (rows x perturbations); with
        ``decode_perturbations=True`` the cells are also averaged per group and read out as
        groups x perturbations.

        Parameters
        ----------
        adata
            Query AnnData (counts in ``X``, gene symbols as ``var_names``, ``obs['perturbation']``; the
            label may be a placeholder for unlabelled data). ``None`` predicts on the training data.
        indices
            Subset of ``adata`` rows to predict on (default: all).
        batch_size
            Minibatch size for encoding.
        use_posterior_mean
            ``True`` encodes with the posterior mean (deterministic); ``False`` samples.
        return_latents
            Also return ``z_shared`` (needed by every readout below; switched on automatically when used).
        inplace
            Write predictions onto ``adata`` (``obsm`` / ``obs`` / ``uns``) and return ``None``.
        center_mode
            Reference-set baseline subtracted from ``z_shared`` before scoring, for control-free
            "differential" inference on a NOVEL dataset (the module's trained per-dataset ``align_ds_mean``
            has no row for an unseen dataset). Only active when the module was trained with
            ``invariant_align=True``; ignored (with a warning) otherwise. ``None`` = the trained
            per-dataset mean; ``'query_mean'`` = the query's own per-dataset mean; ``'control'`` = the mean
            of the cells labelled ``control_label``; an array = a ready ``(n_shared,)`` or
            ``(n_batch, n_shared)`` centre.
        center_by
            obs column (e.g. cell type) whose groups are each centred on their OWN controls in one pass.
        control_label
            The perturbation label of control cells (default ``self.ctrl_class``).
        readout
            ``'cosine'`` (default) or ``'deflate'``; see READOUT below.
        n_cells_key
            obs column with the number of cells behind each row (pseudobulk / bulk input). Default: every
            row is one cell.
        readout_confidence
            Add per-row ``obs['readout_confidence']`` and ``obs['readout_effect']`` (see below).
        decode_perturbations, decode_group_key, decode_head, decode_rounds, decode_top, decode_center
            Group-level readout; see GROUP READOUT below.
        orthologs, ortholog_table, ortholog_taxon
            ``'auto'`` converts a non-human symbol query (e.g. mouse ``Actb``) to human symbols before anything
            else (`expert.utils.orthologs`): through ``ortholog_table`` (MGI ``HOM_MouseHumanSequence.rpt``;
            one-to-one human classes, several query genes of one class summed; ``ortholog_taxon`` = the query's
            NCBI taxon, 10090 mouse) or, without a table, by upper-casing. Rows and ``obs`` are unchanged, so
            ``inplace`` writes still land on the caller's object. ``'off'`` never converts.

        Returns
        -------
        A dict (or ``None`` with ``inplace=True``) with the soft predictions (rows x perturbations), the
        latents, the per-row readout columns and, with ``decode_perturbations=True``, the group readout under
        ``PREDICTION_KEYS.DECODE_KEY`` (``'decoded_perturbations'``; with ``inplace`` in ``adata.uns``).

        READOUT: WHAT THE SCORES ARE
        ----------------------------
        Round 1 is the plain signed cosine readout: every prototype scored against the row's response. The
        top of the list (``'loss'`` tail) names perturbations whose knock-down the row RESEMBLES; the
        bottom (``'gain'`` tail) names those whose knock-down it looks OPPOSITE to.

        ``readout='deflate'`` (opt-in; the default is ``'cosine'``) then asks what the dominant mechanism left
        unexplained: it projects out the subspace spanned by round 1's leading prototypes, reads the residual
        against the WHOLE table again (round 2, then 3), and merges the rounds by proportional slot allocation (PM-2),
        each round weighted by the share of the response it explained. Two rules keep this safe:

        * The first ``decode_head`` (10) slots of each tail are held by round 1, so the top-10 is ALWAYS
          the cosine readout. Later rounds only reorder ranks 11+.
        * A further round runs only while the dictionary-explainable residual is above the noise of a
          CONTROL delta with the same number of cells. Single cells essentially never clear it, so
          cell-level soft predictions are the cosine logits exactly; pseudobulks and groups of tens to
          hundreds of cells do.

        Measured (2026-10-15): deep OOD singles top-10 identical to cosine at every depth;
        GSE146194 doubles both-targets-in-top-20 .200 -> .314; on alcoholic liver disease, healthy-null
        false-positive Reactome sets per group 10.1 -> 6.8, and a replicated AH Pol II pause signal
        (0 -> 5 sets) that cosine missed. OMP is no longer the decode: on disease its rankings carry
        almost no set enrichment. It stays available as
        ``expert.utils.readout.nnomp``.

        The stop and the effect sizes need a one-off calibration, stored with the model::

            model.add_readout_calibration('ctrl_pools.npz')   # 10-cell TRAINING control pools, 'Q' + 'dataset'
            model.save(model_dir, overwrite=True)

        (``paper/scripts/fig5/fig5_hubness.py --stage ctrlref`` writes that file.) Without it the deflated soft
        predictions fall back to cosine, the group readout uses every round, and effect sizes are NaN.
        ``readout='cosine'`` always returns the plain logits.

        INPUT GENE MASK (``mask_genes``)
        --------------------------------
        ``mask_genes='MT-*'`` (glob patterns, case-insensitive; a list for several) puts the matching model genes at
        their control baseline for this call: the encoder's centred input for them is zeroed, in query and control
        rows alike, through ``z_encoder.input_zero_mask``, which is restored on return. Recommended for queries
        that are not Perturb-seq screens (drug screens, disease atlases); leave it off for CRISPR screens.

        PER-ROW OUTPUTS (obs)
        ---------------------
        ``readout_confidence``  the largest cosine to any prototype: the best predictor of whether the
                                readout is right (per cell AUROC .76 for truth-in-top-10; top-10 hit rate
                                rises ~11x from the lowest to the highest quintile).
        ``readout_effect``      ||z|| divided by the 95th percentile of a control delta of the same number
                                of cells. 1 = indistinguishable from control noise. Single cells are almost
                                always below 1; it is meaningful for pseudobulks and groups.
        ``readout_rounds``      rounds used for that row (1 = cosine).
        ``effect_size``         predicted within-dataset effect-size percentile (~0-1; add_effect_head): the trained
                                efficiency head (eff_hat) combined linearly with the cosine confidence of the call
                                (max cosine, top-1 minus top-2 margin, top-1 softmax probability over the top 10, and
                                ||z_c||). Held-out cells of the production model: mean Spearman 0.44 with |Mixscale|
                                (essential screens 0.70) vs 0.40 / 0.65 for eff_hat alone.

        GROUP READOUT (``decode_perturbations=True``)
        --------------------------------------------
        The cells of each ``decode_group_key`` group (whole query if ``None``) are averaged IN z, then read
        out as above at the group's total cell count. This is the recommended way to read a condition,
        donor or cell type: averaging encoded cells keeps more perturbation signal than encoding summed
        counts, and the noise calibration is built from exactly such averages. Result (dict):

        ``scores``       DataFrame groups x perturbations, in the cosine scale and ordered by the merged
                         readout; the group-level analogue of ``soft_predictions``.
        ``group_stats``  DataFrame per group: ``n_cells``, ``n_rounds``, ``effect_size``, ``explainable``
                         (dictionary-explainable residual / its noise), ``max_cos``.
        ``loss`` / ``gain``  merged ranked labels per group (``decode_top`` long); ``round_loss`` /
                         ``round_gain`` each round's own list; ``energy`` the share each round explained.

        ``decode_head`` (10) slots held by round 1, ``decode_rounds`` (3) maximum rounds, ``decode_top``
        (200) list length. ``decode_center='auto'`` subtracts a control mean only for models trained
        without ``invariant_align``. For those, pass control cells with the query and set
        ``control_label``, or the query mean is used, which deletes most of a homogeneous query's signal.

        RECIPES
        -------
        Perturb-seq cells, one ranking per cell::

            out = model.predict(q)                                   # soft_predictions == cosine per cell

        Perturb-seq / disease conditions, one ranking per group (pass control cells too)::

            out = model.predict(q, decode_perturbations=True, decode_group_key='condition',
                                control_label='control')
            S = out['decoded_perturbations']['scores']                # groups x perturbations

        Pre-made pseudobulk or bulk rows (encode summed counts; needs the row depth)::

            out = model.predict(pb, n_cells_key='n_cells', readout='deflate')   # deflated per row at its own depth

        For a disease atlas, build the per-donor x cell-type contrast yourself from the returned group
        scores or z (as in Fig. 5), and judge hits against a healthy donor-vs-donor null: a group well
        above control noise is not automatically above healthy donor variation.

        Notes
        -----
        CENTERING A DISEASE / NOVEL QUERY (measured 2026-09-14 on Weinberg Leigh PBMC with
        GRAPH0 and EFFGATE3; harness data/inference/metabolic_weinberg/centering_audit.py). Two
        centerings act on the query and only the first is a real choice:

        * ENCODER INPUT. A ``center_mode='ctrl'`` encoder subtracts the control mean of
          log1p(x), refreshed from the cells labelled ``control_label`` IN THIS CALL. Best
          readout: call predict() once per cell type (per batch x cell type when every batch has
          controls of that type), each call carrying that cell type's own controls -- or use
          ``center_by``. One pooled call left the absolute read of Leigh monocytes
          indistinguishable from control monocytes (Spearman 0.96 over 8,964 atoms, both models)
          and blind to the known signal (mTOR-autophagy p 0.07-0.09, Complex III 0.18-1.0). Per
          cell type the similarity fell to 0.50-0.55 and both came back (p 0.013-0.017 and
          0.007-0.010) with control rows staying null. Cell-type-matched deltas agreed either way:
          pooled input centering costs the absolute read, not the contrast.
        * ``center_mode`` (z_shared). A shift constant within a cell type cancels exactly in a
          cell-type-matched delta (max rel. diff 6e-8 over five choices), so this only moves an
          absolute, uncontrasted read. Centre on the WHOLE query's mean -- the inference analogue
          of the training running mean. With per-cell-type calls, ``'query_mean'`` would silently
          become each cell type's own all-cell mean, which under disease-skewed composition
          manufactures a disease-vs-mimic anti-correlation (-0.7 measured). So pass
          ``center_mode=np.zeros(n_shared)`` in each call and subtract the mean of the
          concatenated z_shared afterwards (matches ``'query_mean'`` on one call to ~8e-5 relative).
        """
        if mask_genes:
            # Scope the mask to THIS call: set the encoder's eval-time input mask, run predict() unmasked-by-argument,
            # and restore whatever mask the encoder carried before (normally None) on every exit path.
            from fnmatch import fnmatch
            _args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs', 'mask_genes', '__class__')}
            _pats = [mask_genes] if isinstance(mask_genes, str) else list(mask_genes)
            _pats = [str(p).strip().upper() for p in _pats if str(p).strip()]
            _vn = np.asarray(self.adata.var_names).astype(str)
            _hit = np.array([any(fnmatch(v.upper(), p) for p in _pats) for v in _vn])
            _enc = self.module.z_encoder
            if not _hit.any():
                log.warning(f'[Predict] mask_genes {_pats}: no model gene matches; running unmasked.')
            else:
                log.info(f'[Predict] mask_genes {_pats}: {int(_hit.sum())} of {len(_vn)} model genes held at their '
                         f'control baseline ({", ".join(_vn[_hit][:10])}{" ..." if _hit.sum() > 10 else ""}).')
            _prev_mask = getattr(_enc, 'input_zero_mask', None)
            _enc.input_zero_mask = torch.as_tensor(_hit) if _hit.any() else _prev_mask
            try:
                return self.predict(**_args, mask_genes=None, **kwargs)
            finally:
                _enc.input_zero_mask = _prev_mask
        _gone = sorted(set(kwargs) & {'decode_k', 'decode_stop', 'decode_stop_t', 'decode_min_gain',
                                      'decode_resid_frac', 'decode_dict', 'decode_mode', 'decode_lam',
                                      'min_class_efficiency', 'class_efficiency_power'})
        if _gone:
            # **kwargs is otherwise swallowed: without this an OMP-era call would silently run the new readout
            raise TypeError(f'predict() no longer runs the OMP decode ({_gone}); the decode is now the deflated '
                            f'readout (decode_head / decode_rounds / decode_top). For OMP call '
                            f'expert.utils.readout.nnomp on the returned z_shared.')
        if readout not in ('deflate', 'cosine'):
            raise ValueError(f"readout must be 'deflate' | 'cosine', got {readout!r}")
        if decode_perturbations:
            # Routed below, after the latents exist. Validate the PRECONDITIONS here, at the top,
            # so a 40-minute pass over a 100M-cell query cannot end in an argument error -- and so
            # the flag can never read as functional while returning undecoded results.
            if not return_latents:
                log.info('[Decode] decode_perturbations=True: auto-enabling return_latents '
                         '(the decode consumes z_shared).')
                return_latents = True
            if str(decode_center).lower() not in ('control', 'none', 'auto'):
                raise ValueError(f"decode_center must be 'control' | 'none' | 'auto', got "
                                 f"{decode_center!r}")
            if int(decode_head) < 0 or int(decode_rounds) < 1 or int(decode_top) < max(1, int(decode_head)):
                raise ValueError(f'decode needs decode_head >= 0, decode_rounds >= 1, decode_top >= decode_head; '
                                 f'got {decode_head}, {decode_rounds}, {decode_top}')
        from tqdm import tqdm
        # Check if model has been trained
        self._check_if_trained(warn=False)
        # minify is a post-write op that needs the caller's adata to land
        # back as the user's reference and a full latent payload in obsm.
        # Both require inplace=True + return_latents=True.
        if minify:
            if not inplace:
                log.warning('[Predict] minify=True ignored: requires inplace=True. Skipping minification.')
                minify = False
            elif not return_latents:
                log.info('[Predict] minify=True: auto-enabling return_latents so qzm/qzv land in obsm.')
                return_latents = True
        # Cap intra-op CPU thread count when requested. No-op for GPU runs
        # since all heavy ops execute on-device; set once at the top so any
        # downstream tensor work (alignment, dataloader collate) sees it.
        # Use the call-level value vs. process-global default — torch
        # ``set_num_threads`` is global, but we restore on exit below.
        if n_cores is not None:
            _prev_threads = torch.get_num_threads()
            torch.set_num_threads(int(n_cores))
            log.info(f'[Predict] Capping torch intra-op threads to {n_cores} (was {_prev_threads}).')
        else:
            _prev_threads = None
        # Ensure adata has the correct features. Keep a handle to the user's
        # original adata so the ``inplace`` write target is the object the
        # caller is holding — align_model_features / _validate_anndata
        # rebind ``adata`` to internal copies that the caller can't see.
        is_query = adata is not None
        user_adata = adata
        if is_query and orthologs != 'off':
            from expert.utils.orthologs import guess_species, to_human
            sp_rep = guess_species(adata.var_names, self.adata.var_names)
            if sp_rep['non_human']:
                if sp_rep['ensembl_non_human'] > 0.5:
                    raise ValueError(f'[Predict] query var_names are non-human Ensembl IDs ({sp_rep}); map them to '
                                     f'gene symbols first, then predict() converts symbols to human.')
                adata, conv = to_human(adata, table=ortholog_table, query_taxon=ortholog_taxon)
                (log.warning if ortholog_table is None else log.info)(
                    f'[Predict] non-human gene space detected (exact match {sp_rep["exact"]:.1%}, upper-cased '
                    f'{sp_rep["upper"]:.1%}); converted to human by {conv["method"]}: {conv["n_mapped_query_genes"]:,} '
                    f'of {conv["n_query_genes"]:,} genes -> {conv["n_human_genes"]:,} human genes, '
                    f'{conv["frac_counts_kept"]:.1%} of counts kept.')
        # Var alignment is now per-batch, not upfront — adata.X stays in
        # query's var space (so backed h5ads stay on disk and we never
        # materialize a 2x-sized in-memory copy). ``align_col_map`` is the
        # length-n_model_vars index plan; ``align_n_model_vars`` is the
        # target column count for the per-batch zero-fill. Both None when
        # adata == self.adata.
        align_col_map: torch.Tensor | None = None
        align_n_model_vars: int | None = None
        if is_query:
            align_col_map, align_n_model_vars = self._build_var_alignment(adata)
        # Validate adata or get it from model. When the query lives in a
        # different var space we have to relax scvi's strict X-layer var-
        # count check during transfer_fields — categorical fields still
        # carry training mappings through, and the per-batch remap below
        # makes X model-compatible before it reaches the encoder.
        if align_col_map is not None:
            with self._relax_layer_var_check():
                adata = self._validate_anndata(adata, extend_categories=True, add_missing_categories=add_missing_categories)
        else:
            adata = self._validate_anndata(adata, extend_categories=True, add_missing_categories=add_missing_categories)
        # Take all cells by default
        use_full = False
        if indices is None:
            use_full = True
            indices = np.arange(adata.n_obs)
        # Create dataloader
        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size
        )
        # WHICH HEAD SCORES, resolved ONCE and logged. 'auto' must not trust use_cls_head: PA arms set it
        # while training the head at weight 0, so the old rule silently returned an untrained head's
        # chance-level scores (and only the trained block, so no zero-shot columns either).
        _head_choice, _head_why = self._resolve_head(head)
        self.module.use_hub_correction = bool(hub_correction)
        if hasattr(self.module, 'has_hub_correction') and self.module.has_hub_correction():
            log.info(f'[Predict] hubness correction {"ON" if hub_correction else "OFF"} '
                     f'(stored per-atom offset; pass hub_correction=False for plain cosine)')
        if (not obs_only and cls_emb is None and _head_choice == 'A'
                and getattr(getattr(self.module, 'cls_emb', None), '_atom_priors', None)
                and hasattr(self.module, 'has_prior_missing_fill') and not self.module.has_prior_missing_fill()):
            log.warning('[Predict] obs_only=False but this model has no saved f(missing) fill: the untrained '
                        'columns score against RANDOM-INIT atoms. Run model.add_prior_missing_fill(config) once '
                        '(needs the training DB obs + fold split) and save.')
        log.info(f"[Predict] scoring head: {_head_choice} ({'atoms / class bank' if _head_choice == 'A' else 'free classifier'}); "
                 f"requested head={head!r}: {_head_why}")
        
        # Use new context embedding at inference
        if ctx_emb is not None:
            log.info(f'Using new context embedding: {ctx_emb.shape}')
            ctx_labels, ctx_emb = self.validate_ctx_emb(ctx_emb)
            ctx_emb = ctx_emb.to(self.device)
        # Use model's context labels and embedding
        else:
            ctx_labels = self.idx_to_batch
        # Use new class embedding at inference
        cls_zero_shot_info: pd.DataFrame | None = None
        # Keep handles to the raw (pre-adapter) cls embedding + matching
        # labels so the post-loop report can compute embedding-space
        # prediction quality and adapter-fidelity metrics. Both stay None
        # when the caller didn't supply an expanded cls_emb.
        cls_emb_raw_np = None
        if cls_emb is not None:
            log.info(f'Using new class embedding: {cls_emb.shape}')
            cls_labels, cls_emb = self.validate_cls_emb(cls_emb)
            cls_emb = cls_emb.to(self.device)
            # Zero-shot prediction is only possible when held-out class
            # anchors actually enter the prediction softmax. ``obs_only=True``
            # (the default) maps to ``local_predictions=True`` in classify(),
            # which slices c2z[0:n_labels] — held-out classes are dropped
            # from the candidate set entirely. Warn loudly: the caller is
            # paying the cost of an expanded cls_emb for nothing.
            n_extra = int(cls_emb.shape[0]) - int(self.n_labels)
            if obs_only and n_extra > 0:
                log.warning(
                    f'[Predict] cls_emb has {n_extra} class(es) beyond the '
                    f'{self.n_labels} trained — but obs_only=True restricts '
                    f'predictions to the trained set (local_predictions=True). '
                    f'Pass obs_only=False to let zero-shot classes appear in '
                    f'the prediction softmax.'
                )
            # Capture the raw (pre-adapter) embedding for zero-shot
            # similarity scoring. We do this *before* the adapter projection
            # because the adapter is a learned linear map: it can collapse
            # semantically distinct classes onto similar points in shared
            # space, washing out the novelty signal we want to quantify.
            cls_emb_raw_np = cls_emb.detach().cpu().numpy()
            try:
                log.info(f'[Predict] Computing class similarities')
                from expert.utils.performance import class_similarity_to_trained
                cls_zero_shot_info = class_similarity_to_trained(
                    cls_emb=cls_emb_raw_np,
                    cls_labels=cls_labels,
                    trained_labels=self.get_training_classes(),
                )
            except Exception as e:
                log.warning(f'[Predict] Could not compute zero-shot similarity: {e}')
            # If the user supplied embeddings in the raw LLM space (matching
            # adapter.in_features rather than the projected n_cls_dim), apply
            # the trained adapter so they live in the same z_shared geometry
            # as the in-training class anchors. This is the canonical path
            # for zero-shot inference over a *larger* class set than was seen
            # during training: pass raw LLM embeddings of all desired classes
            # (including unseen ones), let the adapter project them.
            adapter = getattr(self.module.cls_emb, 'adapter', None)
            in_features = getattr(adapter, 'in_features', None)
            if (
                adapter is not None
                and in_features is not None
                and cls_emb.shape[-1] == in_features
                and cls_emb.shape[-1] != self.module.n_cls_dim
            ):
                log.info(
                    f'[Predict] Projecting raw cls_emb through trained adapter: '
                    f'{tuple(cls_emb.shape)} -> '
                    f'(*, {self.module.n_cls_dim}).'
                )
                with torch.no_grad():
                    cls_emb = adapter(cls_emb)
        # Use model's class labels and embedding
        else:
            cls_labels = self.idx_to_label

        # Optional TENT pass on the target cells before classifying — adapts
        # LayerNorm γ/β by entropy minimization on the cells we're about to
        # predict over. Useful for OOD / cross-dataset queries.
        if tent_iters > 0:
            log.info(f'[Predict] Running TENT ({tent_iters} iter, lr={tent_lr}) on {len(indices)} target cells.')
            self.tent_adapt(adata=adata, indices=indices, n_iters=tent_iters, lr=tent_lr, batch_size=batch_size)

        log.info(f'[Predict] Classifying.')
        # Pre-allocate large output arrays and fill batch-by-batch
        n_total = len(indices)
        cursor = 0
        cls_logits = None
        z_cls_logits = None
        ctx_logits = None
        zs = None
        z_shared = None
        qz_means = None
        qz_vars = None
        # Small per-batch buffers stay as lists — concatenated cheaply at end.
        ctx2z, cls2z = [], []
        data_ctx_labels = []
        data_cls_labels = []
        predictions = []
        response_probs = []
        eff_hats = []
        true_weights = []
        # *_scaled views are computed once post-aggregation (per dataset across
        # the full eval set) below, not collected per batch — per-batch z-scores
        # have inconsistent (mean, std) across batches and produce a noisy axis.
        # Top-K sparse path: when top_k_logits is set, accumulate per-batch
        # CSR rows instead of writing into a dense (n_total, n_classes) buffer.
        # For 18K classes × 1.9M cells, dense float16 is ~64GB and OOM-prone
        # at the aggregation step (both np.nan_to_num and post-loop argmax
        # do full-buffer scans, faulting every page into RAM at once). Top-K
        # CSR stays at ~3GB for K=200 — softmax mass is concentrated enough
        # that K=100-500 captures essentially all the signal.
        cls_logits_csr_rows = [] if top_k_logits is not None else None

        def _fill(buf, chunk_np, fill=np.nan):
            """Allocate ``buf`` (sized to (n_total, *chunk[1:])) on first call,
            then write the next chunk at ``cursor``. Returns the (possibly
            new) buffer."""
            if buf is None:
                shape = (n_total,) + chunk_np.shape[1:]
                buf = np.full(shape, fill, dtype=np.float16)
            buf[cursor:cursor + chunk_np.shape[0]] = chunk_np
            return buf

        # Add progress bar to classification if toggled
        if progress_bar:
            dl_it = enumerate(tqdm(scdl, total=len(scdl), unit='batch', desc='Predicting'))
        else:
            dl_it = enumerate(scdl)
        # Run through each batch of data
        # Move col_map to the model's device once so the per-batch gather
        # below runs on the same device as x (no host↔device hop per batch).
        if align_col_map is not None:
            align_col_map = align_col_map.to(self.device)
        # Force eval mode for the inference passes below (after any TENT adaptation,
        # which manages its own mode). Guarantees dropout/noise are off AND that
        # _center_z_shared takes its EVAL branch — otherwise a freshly-trained module
        # left in train() would mutate align_ds_mean / class_centers / prototypes and
        # apply stochastic dropout during prediction.
        self.module.eval()
        # Resolve the reference-set baseline (control-free differential inference).
        # No-op / None unless the module was trained with invariant_align and the
        # caller passed center_mode. Uses the SAME dataloader + var-alignment plan
        # as the main loop so the pre-pass sees identical inputs.
        # INPUT-SPACE control centering for a novel query. This is a DIFFERENT mechanism from
        # `center_mode` below: that centres z_shared and is gated on invariant_align, whereas the
        # ENCODER's center_mode='ctrl' subtracts a per-dataset control mean from x. Without this
        # refresh a dataset absent from training keeps a TRAINING baseline -- a silent distribution
        # shift for an encoder trained only on control-centred input.
        # ---- INPUT control-centring: mirror EXACTLY what the encoder was trained with -------
        # A model trained with center_mode='ctrl' subtracts a per-dataset control mean from x. If
        # the query's datasets have no ctrl_gene_means row, the encoder silently falls back to a
        # TRAINING baseline, which is a distribution shift that reads as catastrophic far-OOD
        # performance rather than as a bug. So this is AUTOMATIC (refresh_ctrl_means=None) rather
        # than opt-in, and it FAILS LOUDLY: a warning in a 500 KB log is not a safeguard.
        # NOTE this is the ENCODER-INPUT path and has nothing to do with `center_mode`/
        # align_center below, which centres z_shared and is retired -- the two are not comparable
        # and must not be substituted for one another.
        _enc = getattr(self.module, 'z_encoder', None)
        _needs_ctrl = (getattr(_enc, 'center_input', False)
                       and str(getattr(_enc, 'center_mode', '')).startswith('ctrl'))
        _do_refresh = _needs_ctrl if refresh_ctrl_means is None else bool(refresh_ctrl_means)
        # Stale per-group centring state can never survive into another call: cleared on EVERY
        # predict, not just the ones that set it, so an exception mid-loop cannot poison the next.
        if _enc is not None:
            _enc.center_bank = None
            _enc.center_codes = None
        _center_codes = None
        if center_by is not None:
            # the per-group bank SUPERSEDES the whole-query refresh -- running both would overwrite
            # ctrl_gene_means with a query-wide mean that nothing then reads
            _bank, _center_codes, _rep = self._build_center_by_bank(
                adata, indices, center_by, control_label)
            _enc.center_bank = _bank.to(self.device)
            _srcs = {}
            for _g, _d, _n, _s in _rep:
                _srcs[_s] = _srcs.get(_s, 0) + 1
            log.info('[Predict] center_by=%r: %d (group, dataset) baselines in ONE pass %s',
                     center_by, len(_rep), _srcs)
        elif _do_refresh:
            _q = adata if adata is not None else self.adata
            _lab = control_label if control_label is not None else getattr(self, 'ctrl_class', None)
            if _lab is None:
                # try to find one rather than fail on a naming detail
                _cc = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
                _cand = {str(x) for x in _q.obs[_cc].astype(str).unique()} if _cc in _q.obs else set()
                for _w in ('control', 'ctrl', 'non-targeting', 'NT', 'DMSO_TF'):
                    _hit = [x for x in _cand if str(x).lower() == _w.lower()]
                    if _hit:
                        _lab = _hit[0]; break
            _rep = self.refresh_ctrl_gene_means(_q, control_label=_lab, fallback='query_mean')
            _per = {k: v for k, v in (_rep or {}).items() if not str(k).startswith('_') and isinstance(v, dict)}
            _src = {}
            for _d, _v in _per.items():
                _src.setdefault(_v.get('source') if _v.get('updated') else 'trained_baseline', []).append(_d)
            log.info('[Predict] input control-centring per query dataset: %s',
                     {k: len(v) for k, v in _src.items()})
            if _src.get('query_mean'):
                log.warning('[Predict] no usable control cells for NOVEL dataset(s) %s: input centred on their '
                            'QUERY MEAN (pass control cells + control_label for a control baseline)',
                            _src['query_mean'])
            # only fail when some query dataset ends with NO baseline at all (never happens with the
            # query-mean fallback; kept for fallback=None callers)
            _bad = any((not v.get('updated')) and 'trained baseline' not in str(v.get('reason', ''))
                       for v in _per.values())
            if _bad and _needs_ctrl:
                raise ValueError(
                    f"This model was trained with center_mode="
                    f"{getattr(_enc, 'center_mode', None)!r} (control-centred input), but no usable "
                    f"control cells were found in the query (control_label={_lab!r}). Encoding it "
                    f"anyway would centre on a TRAINING baseline and produce silently wrong "
                    f"results. Pass control cells with the query and set control_label, or pass "
                    f"refresh_ctrl_means=False to acknowledge and override.")
            for _d, _v in (_rep or {}).items():
                if (not str(_d).startswith('_') and isinstance(_v, dict) and not _v.get('updated')
                        and 'trained baseline' not in str(_v.get('reason', ''))):
                    log.warning('[Predict] ctrl means NOT refreshed for %r: %s',
                                _d, _v.get('reason'))
        elif _needs_ctrl:
            log.warning('[Predict] model trained with center_mode=%r but refresh_ctrl_means=False '
                        '-- the query will be centred on a TRAINING baseline.',
                        getattr(_enc, 'center_mode', None))

        # ---- AUTO-FILL the z_shared baseline for datasets absent from training ----------
        # Mirrors the refresh_ctrl_means policy directly above: a missing baseline is a silent
        # distribution shift, so it is fixed AUTOMATICALLY and reported LOUDLY rather than left
        # opt-in. Under invariant_align, `_center_z_shared` subtracts `align_ds_mean[dataset]`; a
        # dataset never seen in training has align_ds_count == 0 and its stored row is a
        # meaningless ZERO, so the eval branch falls back to self-centring on the batch QUERY MEAN.
        # That is defined behaviour but it is the WEAKER option: predict()'s own decode path
        # measures an all-cell mean deleting 84% of a known signature against 2% via a control
        # reference. So when the query contains untrained datasets we resolve a real baseline
        # first -- from the query's CONTROLS when it has them, else the query mean.
        # This matters concretely: splits/200 fold_0 holds out whole DATASETS
        # (donor D2 + cell_line HepG2 -> 4 datasets), so every ood_test query hits this path.
        # an explicit array is a FIXED shift, identical for every group -- combining it with center_by equals
        # looping predict() per group with that array (the Fig. 5 recipe), so only QUERY-DERIVED baselines refuse
        _explicit_center = isinstance(center_mode, (np.ndarray, torch.Tensor))
        center_mode = self._auto_align_center(adata, center_mode, control_label, indices=indices)

        align_center = self._resolve_align_center(
            scdl, center_mode, control_label, align_col_map, align_n_model_vars,
            ctx_emb, cls_emb, use_posterior_mean)
        if center_by is not None and align_center is not None and not _explicit_center:
            # center_by groups the INPUT baseline. A z_shared baseline resolved here is per-DATASET
            # over the WHOLE query, so combining the two would centre x per group and z per dataset
            # -- which is NOT what looping predict() over the groups does, and the difference would
            # be invisible. Refuse rather than approximate.
            raise ValueError(
                f'center_by={center_by!r} cannot be combined with a resolved z_shared baseline '
                f'(center_mode={center_mode!r}). center_by groups the ENCODER-INPUT centring only; '
                f'the z_shared bank would still be per-dataset over the whole query, which differs '
                f'from looping predict() per group. Pass center_mode=None or an explicit array.')
        with torch.no_grad():
            for _, tensors in dl_it:
                # Unpack batch tensor
                x = tensors.get(REGISTRY_KEYS.X_KEY)
                # Per-batch var-space alignment: gather query → model
                # columns + zero-fill missing. Cheap (single index_select
                # + scatter), no full-X materialization needed upstream.
                if align_col_map is not None and x is not None:
                    from expert.utils.common import align_x_batch
                    x = align_x_batch(x, align_col_map, align_n_model_vars)
                l = tensors.get(REGISTRY_KEYS.LABELS_KEY)
                batch = tensors.get(REGISTRY_KEYS.BATCH_KEY)
                # Get continous covariates
                cont_key = REGISTRY_KEYS.CONT_COVS_KEY
                cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None
                # Get cell efficiency
                eff_scores = tensors.get(REGISTRY_KEYS.CLS_EFF_KEY)
                # Get categorical covariates
                cat_key = REGISTRY_KEYS.CAT_COVS_KEY
                cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
                # Cache inference if adata is minified
                if _is_minified(adata):
                    inference_outputs = {
                        MODULE_KEYS.QZM_KEY: tensors[self._LATENT_QZM_KEY],
                        MODULE_KEYS.QZV_KEY: tensors[self._LATENT_QZV_KEY],
                    }
                else:
                    inference_outputs = None
                # Per-group input baseline: hand the encoder THIS minibatch's rows. The loader is
                # sequential over `indices`, so `cursor` is the batch's row offset (the same counter
                # the output buffers are filled at).
                if _center_codes is not None:
                    _n = int(x.shape[0]) if x is not None else 0
                    _enc.center_codes = torch.as_tensor(
                        _center_codes[cursor:cursor + _n], dtype=torch.long, device=self.device)
                # Run classification process, i.e. full / cached / partial inference
                cls_output = self.module.classify(
                    x,
                    batch_index=batch,
                    label=l,
                    cat_covs=cat_covs,
                    cont_covs=cont_covs,
                    eff_scores=eff_scores,
                    use_posterior_mean=use_posterior_mean,
                    inference_outputs=inference_outputs,
                    ctx_emb=ctx_emb,
                    cls_emb=cls_emb,
                    return_logits=True,
                    local_predictions=obs_only,
                    align_center=align_center,
                )

                # Get regular classifier logits (only present when classifier head is wired).
                # .float() upcasts bf16 → fp32 since numpy has no bfloat16 dtype.
                if 'logits' in cls_output:
                    z_cls_logits = _fill(z_cls_logits, cls_output['logits'].float().cpu().numpy(), fill=-np.inf)
                # Unpack context and perturbation classification output
                if MODULE_KEYS.CTX_LOGITS_KEY in cls_output:
                    ctx_logits = _fill(ctx_logits, cls_output[MODULE_KEYS.CTX_LOGITS_KEY].float().cpu().numpy(), fill=-np.inf)
                # Class logits: pull once, populate predictions + buffer from
                # the same array. Dense path keeps a single pre-allocated
                # buffer; top-K sparse path accumulates per-batch CSR rows.
                # WHICH HEAD scores. Two-head models publish BOTH: CLS_LOGITS_KEY = A (adapter(text),
                # full bank, the zero-shot/far-OOD scorer) and CLS_HEAD_LOGITS_KEY = H (the free
                # Linear(n_shared->n_labels) classifier, trained-block only).
                # 'auto' picks the head the model actually TRAINED: H when use_cls_head is set, else A.
                # This matters for backward compat — a classifier-only checkpoint (use_cls_head=True,
                # cls_align_weight=0) has a NEVER-TRAINED adapter, so reading CLS_LOGITS_KEY scores a
                # random head and returns chance (~10/n_labels). Callers evaluating a two-head model,
                # where BOTH heads are trained, must pass head='A' or head='H' explicitly.
                _head_key = MODULE_KEYS.CLS_LOGITS_KEY
                if _head_choice == 'H':
                    if MODULE_KEYS.CLS_HEAD_LOGITS_KEY in cls_output:
                        _head_key = MODULE_KEYS.CLS_HEAD_LOGITS_KEY
                    elif head == 'H':
                        raise KeyError(
                            f"head='H' requested but {MODULE_KEYS.CLS_HEAD_LOGITS_KEY} is absent — this "
                            f"model has no free classifier head (use_cls_head is off).")
                batch_cls = cls_output[_head_key].float().cpu().numpy()
                # Per-batch top-1 predictions — populate from classify() if
                # it provided them, otherwise compute argmax in-loop. Either
                # path frees the aggregation step from a post-loop argmax
                # over the full cls_logits buffer (which page-faults every
                # byte at once on the dense path → OOM).
                if PREDICTION_KEYS.PREDICTION_KEY in cls_output:
                    predictions.extend(cls_output[PREDICTION_KEYS.PREDICTION_KEY].flatten().cpu().numpy())
                else:
                    predictions.extend(np.argmax(batch_cls, axis=-1).tolist())
                if cls_logits_csr_rows is None:
                    # Dense path (unchanged behavior)
                    cls_logits = _fill(cls_logits, batch_cls, fill=-np.inf)
                else:
                    # Top-K sparse path: keep only the K largest per row,
                    # zero out the rest. CSR is ~3*K*B bytes per batch.
                    k = min(int(top_k_logits), batch_cls.shape[-1])
                    n_rows = batch_cls.shape[0]
                    # argpartition is O(N) per row (vs O(N log N) for full sort)
                    topk_idx = np.argpartition(batch_cls, -k, axis=-1)[:, -k:]
                    topk_vals = np.take_along_axis(batch_cls, topk_idx, axis=-1)
                    cls_logits_csr_rows.append(sp.csr_matrix(
                        (topk_vals.flatten().astype(np.float32),
                         topk_idx.flatten().astype(np.int32),
                         np.arange(0, (n_rows + 1) * k, k, dtype=np.int64)),
                        shape=batch_cls.shape,
                    ))
                # Unpack shared latent spaces
                if use_posterior_mean and MODULE_KEYS.QZ_KEY in cls_output and cls_output[MODULE_KEYS.QZ_KEY] is not None:
                    _z_store = cls_output[MODULE_KEYS.QZ_KEY].loc
                elif use_posterior_mean and MODULE_KEYS.QZM_KEY in cls_output:
                    _z_store = cls_output[MODULE_KEYS.QZM_KEY]
                else:
                    _z_store = cls_output[MODULE_KEYS.Z_KEY]
                zs = _fill(zs, _z_store.float().cpu().numpy())
                z_shared = _fill(z_shared, cls_output[MODULE_KEYS.Z_SHARED_KEY].float().cpu().numpy())
                # Capture latent distribution params from the same forward
                # pass — avoids a second pass for _register_latent_variables.
                if MODULE_KEYS.QZ_KEY in cls_output and cls_output[MODULE_KEYS.QZ_KEY] is not None:
                    _qz = cls_output[MODULE_KEYS.QZ_KEY]
                    qz_means = _fill(qz_means, _qz.loc.float().cpu().numpy())
                    qz_vars = _fill(qz_vars, _qz.scale.square().float().cpu().numpy())
                elif MODULE_KEYS.QZM_KEY in cls_output and MODULE_KEYS.QZV_KEY in cls_output:
                    qz_means = _fill(qz_means, cls_output[MODULE_KEYS.QZM_KEY].float().cpu().numpy())
                    qz_vars = _fill(qz_vars, cls_output[MODULE_KEYS.QZV_KEY].float().cpu().numpy())
                # Class / context projections are computed from the (frozen)
                # cls_emb and ctx_emb — deterministic per model state, so all
                # batches produce identical tensors in eval mode. Storing only
                # the first avoids accumulating ~41GB of duplicates over a
                # ~4M-cell run (3884 batches × 7000 × 384 × float32).
                if not cls2z:
                    if MODULE_KEYS.CTX_PROJ_KEY in cls_output:
                        ctx2z.append(cls_output[MODULE_KEYS.CTX_PROJ_KEY].cpu())
                    cls2z.append(cls_output[MODULE_KEYS.CLS_PROJ_KEY].cpu())
                # Collect labels
                if batch is not None:
                    data_ctx_labels.append(batch.flatten().cpu().numpy())
                if l is not None:
                    data_cls_labels.append(l.flatten().cpu().numpy())
                # Collect cell response probability. .float() upcasts bf16 →
                # fp32 since numpy has no bfloat16 dtype (autocast outputs).
                if 'response_prob' in cls_output:
                    response_probs.append(cls_output['response_prob'].flatten().float().cpu().numpy())
                # Collect predicted efficiency and true weights (if registered)
                if 'eff_hat' in cls_output:
                    eff_hats.append(cls_output['eff_hat'].flatten().float().cpu().numpy())
                w = tensors.get(REGISTRY_KEYS.CLS_EFF_KEY, None)
                if w is not None:
                    true_weights.append(w.flatten().cpu().numpy())
                # Advance the row cursor for the pre-allocated buffers
                cursor += int(cls_output[MODULE_KEYS.CLS_LOGITS_KEY].shape[0])
                # ---- CLEANUP ----
                del cls_output, x, l, batch
                torch.cuda.empty_cache()

        log.info(f'[Predict] Aggregating predictions')
        # Top-K sparse path: vstack per-batch CSRs into a single
        # (n_total, n_classes) CSR. Memory peak here is just the result
        # (~3GB for K=200, 1.9M cells) plus the brief transient during
        # vstack. Dropping the per-batch list immediately afterward frees
        # it. No nan_to_num / argmax pass needed — sparse zero entries
        # correctly represent "below top-K" and predictions were already
        # populated in-loop.
        if cls_logits_csr_rows is not None:
            cls_logits = sp.vstack(cls_logits_csr_rows, format='csr')
            del cls_logits_csr_rows
        # NaN handling for dense path: post-loop argmax on NaN logits picks
        # the NaN row. Now skipped entirely since predictions is always
        # populated in-loop. ctx_logits is small (~Bx19) so always cheap.
        has_z_cls_logits = z_cls_logits is not None
        has_ctx_logits = ctx_logits is not None
        if has_ctx_logits:
            np.nan_to_num(ctx_logits, copy=False, nan=-np.inf)
        # cls_logits nan_to_num: only fires if predictions is empty (it never
        # is anymore, but the guard stays for safety). For sparse CSR it's
        # a no-op since zeros are correctly represented.
        if cls_logits is not None and not predictions and not sp.issparse(cls_logits):
            np.nan_to_num(cls_logits, copy=False, nan=-np.inf)
        if _enc is not None:
            _enc.center_bank = None
            _enc.center_codes = None
        # Truncate buffers if cursor underran n_total (e.g. drop_last yielded
        # fewer rows than indices). With our standard loaders cursor == n_total.
        if cursor < n_total:
            if cls_logits is not None:
                cls_logits = cls_logits[:cursor]
            if z_cls_logits is not None:
                z_cls_logits = z_cls_logits[:cursor]
            if ctx_logits is not None:
                ctx_logits = ctx_logits[:cursor]
            if zs is not None:
                zs = zs[:cursor]
            if z_shared is not None:
                z_shared = z_shared[:cursor]
            if qz_means is not None:
                qz_means = qz_means[:cursor]
            if qz_vars is not None:
                qz_vars = qz_vars[:cursor]
            n_total = cursor
        # cls2z / ctx2z are now single-element lists (we only stored the
        # first, since all batches produce identical projections in eval).
        # Just unwrap to numpy.
        # .float() upcasts bf16 → fp32 since numpy has no bfloat16 dtype.
        # DETACH before numpy. Under `random_init_proxies` the class projection is a function of
        # LEARNABLE atoms, so cls2z carries grad whenever predict() is called outside no_grad --
        # and a returned numpy array can never want the graph. Without this, decoding a screen
        # through an atoms model raises "Can't call numpy() on Tensor that requires grad".
        ctx2z = ctx2z[0].detach().float().numpy() if ctx2z else None
        cls2z = cls2z[0].detach().float().numpy() if cls2z else None
        # Map indices back to labels
        ctx_ls, cls_ls = None, None
        if data_ctx_labels:
            data_ctx_labels = np.concatenate(data_ctx_labels)
            # ctx_labels was captured BEFORE a novel query dataset was appended to idx_to_batch by the
            # control-mean refresh (_extend_batch_buffers), so its new index is not in the stale copy
            if (ctx_emb is None and len(data_ctx_labels)
                    and int(np.max(data_ctx_labels)) >= len(ctx_labels) and len(self.idx_to_batch) > len(ctx_labels)):
                ctx_labels = self.idx_to_batch
            ctx_ls = ctx_labels[data_ctx_labels]
        if data_cls_labels:
            data_cls_labels = np.concatenate(data_cls_labels)
            cls_ls = cls_labels[data_cls_labels]
        # Stack response probabilities
        if response_probs:
            response_probs = np.concatenate(response_probs)
        # Aggregate efficiency tensors
        eff_hat_arr = np.concatenate(eff_hats) if eff_hats else None
        true_weights_arr = np.concatenate(true_weights) if true_weights else None
        # Per-dataset z-scores computed ONCE across the full eval set. Doing
        # this per-batch (as the inference path used to) gave each batch its
        # own (mean, std) per dataset, so the concatenated values had
        # inconsistent normalization — Spearman survived but R² got shredded.
        # One pass over the aggregated raws fixes that: every cell in dataset
        # D is z-scored against D's full empirical (mean, std).
        eff_hat_scaled_arr = None
        eff_scaled_arr = None
        if eff_hat_arr is not None and true_weights_arr is not None and ctx_ls is not None:
            abs_weights = np.abs(true_weights_arr).astype(eff_hat_arr.dtype)
            eff_hat_scaled_arr = np.zeros_like(eff_hat_arr)
            eff_scaled_arr = np.zeros_like(abs_weights)
            ctx_arr = np.asarray(ctx_ls)
            for ds in np.unique(ctx_arr):
                m = ctx_arr == ds
                if m.sum() < 8:
                    continue
                p, t = eff_hat_arr[m], abs_weights[m]
                t_std = float(t.std())
                if t_std < 1e-4:
                    continue
                p_std = float(p.std())
                eff_hat_scaled_arr[m] = (p - p.mean()) / (p_std + 1e-4)
                eff_scaled_arr[m] = (t - t.mean()) / (t_std + 1e-4)

        # Memory-efficient label lookup: argmax + columns indexing on the
        # raw numpy buffer avoids the giant (n_cells, n_classes) DataFrame
        # construction that previously OOM'd at the aggregation step.
        # DataFrames are materialized lazily below only when
        # ``soft_predictions=True`` (the default) AND there's a consumer.

        # Add classifier predictions (Z classifier head)
        if has_z_cls_logits:
            z_cls_cols = np.asarray(cls_labels[:z_cls_logits.shape[-1]])
            z_pred_idx = np.argmax(z_cls_logits, axis=-1)
            z_cls_predictions = z_cls_cols[z_pred_idx]
        else:
            z_cls_cols = None
            z_cls_predictions = ''
        # Add context predictions
        if has_ctx_logits:
            ctx_cols = np.asarray(ctx_labels[:ctx_logits.shape[-1]])
            ctx_pred_idx = np.argmax(ctx_logits, axis=-1)
            ctx_predictions = ctx_cols[ctx_pred_idx]
        else:
            ctx_cols = None
            ctx_predictions = ''

        # Extend class labels with null-proxy names if logits include them
        n_logit_cols = cls_logits.shape[-1]
        if n_logit_cols > len(cls_labels):
            extra = np.array([f'null_{i}' for i in range(n_logit_cols - len(cls_labels))])
            cls_columns = np.concatenate([np.asarray(cls_labels), extra])
        else:
            cls_columns = np.asarray(cls_labels[:n_logit_cols])
        # Determine prediction labels directly from cls_logits + cls_columns
        # — no DataFrame needed for the argmax path. Skip the argmax pass
        # when per-batch ``predictions`` were already collected during the
        # loop (avoids a full scan over the ~64GB softmax buffer).
        if predictions:
            pred_cls_idx = np.array(predictions)
        else:
            pred_cls_idx = np.argmax(cls_logits, axis=-1)
        cls_predictions = cls_columns[pred_cls_idx]

        # Materialize soft predictions lazily. Dense cls_logits → DataFrame
        # with copy=False (wraps the buffer). Sparse cls_logits (top-K path)
        # → keep as scipy CSR; anndata.obsm accepts CSR directly and column
        # names get spilled to uns below for label recovery.
        cls_soft_predictions = z_cls_soft_predictions = ctx_soft_predictions = None
        if soft_predictions:
            obs_index = adata.obs_names[indices]
            if sp.issparse(cls_logits):
                # CSR doesn't go through pd.DataFrame — pass through directly.
                cls_soft_predictions = cls_logits
            else:
                cls_soft_predictions = pd.DataFrame(
                    cls_logits, columns=cls_columns, index=obs_index, copy=False,
                )
            if has_z_cls_logits:
                z_cls_soft_predictions = pd.DataFrame(
                    z_cls_logits, columns=z_cls_cols, index=obs_index, copy=False,
                )
            if has_ctx_logits:
                ctx_soft_predictions = pd.DataFrame(
                    ctx_logits, columns=ctx_cols, index=obs_index, copy=False,
                )

        # Return all model predictions
        # TODO: add all keys to statics
        o = {
            PREDICTION_KEYS.Z_PREDICTION_KEY: z_cls_predictions,
            PREDICTION_KEYS.Z_SOFT_PREDICTION_KEY: z_cls_soft_predictions,
            PREDICTION_KEYS.CTX_PREDICTION_KEY: ctx_predictions,
            PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY: ctx_soft_predictions,
            PREDICTION_KEYS.PREDICTION_KEY: cls_predictions,
            PREDICTION_KEYS.SOFT_PREDICTION_KEY: cls_soft_predictions,
            REGISTRY_KEYS.BATCH_KEY: ctx_ls,
            REGISTRY_KEYS.LABELS_KEY: cls_ls,
            'label_idx': data_cls_labels,
            'indices': indices,
            'response_probs': response_probs,
            'eff_hat': eff_hat_arr,
            'raw_weights': true_weights_arr,
            'eff_hat_scaled': eff_hat_scaled_arr,
            'eff_scaled': eff_scaled_arr,
            'cls_zero_shot_info': cls_zero_shot_info,
            'cls_emb_raw': cls_emb_raw_np,
            'cls_labels': np.asarray(cls_labels),
        }
        # Add latent spaces to prediction output
        if return_latents:
            o.update({
                MODULE_KEYS.Z_KEY: zs,
                MODULE_KEYS.Z_SHARED_KEY: z_shared,
                MODULE_KEYS.CTX_PROJ_KEY: ctx2z,
                MODULE_KEYS.CLS_PROJ_KEY: cls2z,
            })
            if qz_means is not None and qz_vars is not None:
                o[MODULE_KEYS.QZM_KEY] = qz_means
                o[MODULE_KEYS.QZV_KEY] = qz_vars

        # ---- Sparse decode of the query's RESPONSE over the perturbation dictionary ----
        # This is the readout the atom design exists for, so it runs on z_shared -- the same
        # geometry the atoms live in -- and never on the CE head's W. OMP over a CE-trained W was
        # measured as a dead wash (0.1116 vs 0.1115): CE shapes W for winner-take-all, not as a
        # basis, so the head is not a dictionary no matter how well it classifies.
        # ---- Readout baseline, shared by the per-cell confidence and the decode ----
        # THE BASELINE IS THE MEASUREMENT. A perturbation signature is a DIFFERENCE from unperturbed, so
        # scoring raw z would score "what this cell is" rather than "what was done to it". Whether a control
        # mean is subtracted at all is decode_center's call (see its comment); when it is, prefer the query's
        # OWN controls and fall back to the query mean only with a loud warning, since on a homogeneous query
        # that mean is mostly signal (measured: 84% of a known signature deleted vs 2% via a reference).
        _Zr, _ro_base = None, None
        _obs_rows = adata.obs.iloc[np.asarray(indices)] if indices is not None else adata.obs
        if n_cells_key is not None:
            if n_cells_key not in _obs_rows:
                raise KeyError(f'n_cells_key {n_cells_key!r} not in adata.obs')
            _nrow = _obs_rows[n_cells_key].to_numpy(dtype=np.float64)
        else:
            _nrow = np.ones(len(_obs_rows), np.float64)
        if decode_perturbations or readout_confidence or readout == 'deflate':
            _Zr = z_shared if z_shared is not None else zs
            if _Zr is None:
                if decode_perturbations:
                    raise RuntimeError('decode_perturbations: no latent was produced to decode.')
            else:
                _Zr = np.asarray(_Zr, dtype=np.float64)
                _cmode = str(decode_center).lower()
                if _cmode == 'auto':
                    _cmode = 'none' if getattr(self.module, 'invariant_align', False) else 'control'
                if _cmode == 'control':
                    _obs = _obs_rows
                    _lbl = control_label if control_label is not None else getattr(self, 'ctrl_class', None)
                    if _lbl is not None and self.original_label_key in _obs:
                        _m = (_obs[self.original_label_key].astype(str) == str(_lbl)).to_numpy()
                        if _m.sum() >= 1:
                            _ro_base = _Zr[_m].mean(0)
                            log.info(f'[Readout] control baseline from {int(_m.sum())} {_lbl!r} cells in the query.')
                    if _ro_base is None:
                        _ro_base = _Zr.mean(0)
                        log.warning(
                            f'[Readout] no control cells found (control_label={_lbl!r}); centring on the '
                            f'QUERY MEAN. On a homogeneous query this deletes most of the signature -- '
                            f'pass control cells with the query and set control_label.')
                else:
                    log.info('[Readout] decode_center=%r: NOT subtracting a control mean -- the module centres '
                             'z_shared on the dataset baseline itself and the atoms were trained against that '
                             'frame.', decode_center)
        _noise = (self.module.readout_noise.detach().cpu().numpy().astype(np.float64)
                  if hasattr(self.module, 'has_readout_noise') and self.module.has_readout_noise() else None)

        # ---- Per-cell readout confidence ----
        if readout_confidence and _Zr is not None:
            try:
                _P, _ = self._readout_prototypes(cls2z)
            except ValueError as _e:
                log.info(f'[Readout] per-cell confidence skipped: {_e}')
                _P = None
            if _P is not None:
                _dev = next(self.module.parameters()).device
                _Pn = torch.nn.functional.normalize(torch.as_tensor(_P, dtype=torch.float32, device=_dev), dim=1)
                _conf = np.empty(_Zr.shape[0], np.float32)
                _nrm = np.empty(_Zr.shape[0], np.float64)
                for _s in range(0, _Zr.shape[0], 8192):
                    _zb = _Zr[_s:_s + 8192] - (_ro_base if _ro_base is not None else 0.0)
                    _nrm[_s:_s + 8192] = np.linalg.norm(_zb, axis=1)
                    _zt = torch.nn.functional.normalize(torch.as_tensor(_zb, dtype=torch.float32, device=_dev), dim=1)
                    _conf[_s:_s + 8192] = (_zt @ _Pn.T).max(1).values.cpu().numpy()
                o['readout_confidence'] = _conf
                o['readout_effect'] = (_nrm / ro_noise_level(_nrow[:len(_nrm)], _noise[0], _noise[1])
                                       if _noise is not None else np.full(_Zr.shape[0], np.nan))

        # ---- Effect size (add_effect_head): trained efficiency head + cosine confidence of the call ----
        if (effect_size and hasattr(self.module, 'has_effect_head') and self.module.has_effect_head()
                and eff_hat_arr is not None and (z_shared is not None or zs is not None)):
            from expert.utils.readout import perturbation_prototypes as _pprot
            _Ze = np.asarray(z_shared if z_shared is not None else zs, dtype=np.float32)
            _meta = self.module.effect_size_meta.detach().cpu().numpy().astype(np.float64)
            _b, _k, _T = float(_meta[0]), int(_meta[1]), float(_meta[2])
            _w = self.module.effect_size_w.detach().cpu().numpy().astype(np.float64)
            _, _, _Pe = _pprot(self)
            if int(_meta[3]) != _Pe.shape[0]:
                raise ValueError(f'effect head was stored against {int(_meta[3])} prototypes; model has {_Pe.shape[0]}')
            _Pe = _Pe / np.clip(np.linalg.norm(_Pe, axis=1, keepdims=True), 1e-12, None)
            # per-group control baseline (the readout was fitted on dataset-control-centred latents)
            _lbl = control_label if control_label is not None else getattr(self, 'ctrl_class', None)
            _ckey = effect_center_key if effect_center_key is not None else self.original_batch_key
            _ctl = ((_obs_rows[self.original_label_key].astype(str) == str(_lbl)).to_numpy()
                    if _lbl is not None and self.original_label_key in _obs_rows else np.zeros(len(_Ze), bool))
            _grp = (_obs_rows[_ckey].astype(str).to_numpy() if _ckey in _obs_rows
                    else np.zeros(len(_Ze), dtype=object))
            _Zc = np.empty_like(_Ze)
            for _g in pd.unique(_grp):
                _m = _grp == _g
                if (_m & _ctl).sum() >= 20:
                    _base = _Ze[_m & _ctl].mean(0)
                elif _ctl.sum() >= 20:
                    _base = _Ze[_ctl].mean(0)
                    log.warning(f'[effect-size] group {_g!r}: < 20 control cells; centred on ALL query controls.')
                else:
                    _base = _Ze[_m].mean(0)
                    log.warning(f'[effect-size] group {_g!r}: no control cells in the query; centred on the GROUP '
                                f'MEAN, which removes the shared response -- pass controls (control_label).')
                _Zc[_m] = _Ze[_m] - _base
            _dev = next(self.module.parameters()).device
            _Pt = torch.as_tensor(_Pe, dtype=torch.float32, device=_dev)
            _feat = np.empty((len(_Zc), 5), np.float64)
            _feat[:, 0] = np.asarray(eff_hat_arr, np.float64)[:len(_Zc)]
            for _s in range(0, len(_Zc), 8192):
                _zc = torch.as_tensor(_Zc[_s:_s + 8192], device=_dev)
                _v = torch.topk(torch.nn.functional.normalize(_zc, dim=1) @ _Pt.T, _k, dim=1).values
                _feat[_s:_s + 8192, 1] = _v[:, 0].cpu().numpy()
                _feat[_s:_s + 8192, 2] = (_v[:, 0] - _v[:, 1]).cpu().numpy()
                _feat[_s:_s + 8192, 3] = torch.softmax(_v / _T, dim=1)[:, 0].cpu().numpy()
                _feat[_s:_s + 8192, 4] = _zc.norm(dim=1).cpu().numpy()
            o['effect_size'] = (_feat @ _w + _b).astype(np.float32)

        # ---- Deflated soft predictions (default) ----
        if readout == 'deflate' and _Zr is not None and soft_predictions:
            _skip = None
            if _noise is None:
                _skip = 'no readout calibration on this checkpoint (add_readout_calibration)'
            elif cls_logits is None or sp.issparse(cls_logits):
                _skip = 'soft predictions are sparse (top_k_logits)'
            elif hub_correction and hasattr(self.module, 'has_hub_correction') and self.module.has_hub_correction():
                _skip = 'hub_correction reorders round 1 (combination not validated)'
            if _skip:
                # An explicit readout='deflate' that cannot be honoured is an error, not a silent fallback to the
                # plain cosine logits (which is what readout='cosine', the default, returns).
                raise ValueError(f"readout='deflate' cannot run: {_skip}. Use readout='cosine', or fix the cause.")
            else:
                _P, _labs = self._readout_prototypes(cls2z)
                _dr = ro_deflate_rows(_Zr - (_ro_base if _ro_base is not None else 0.0), _P, _nrow[:_Zr.shape[0]],
                                      _noise, rounds=int(decode_rounds), head=int(decode_head), k=int(decode_top))
                _colpos = pd.Index(np.asarray(cls_columns).astype(str)).get_indexer(_labs)
                _valid = _colpos >= 0
                _pmap = np.cumsum(_valid) - 1                      # prototype index -> position among valid columns
                _cols = _colpos[_valid]
                _h = int(decode_head)
                for _j, _i in enumerate(_dr['rows']):
                    _row = np.asarray(cls_logits[_i, _cols], np.float64)[None]
                    # ROUND 1 IS THE MODEL'S OWN LOGITS: their top/bottom `decode_head` stay exactly in place (the
                    # head's order differs from plain cos(z_shared, prototypes) on ~8.5% of cells), the deflated
                    # ranks follow in merged order
                    _hl = list(np.argsort(-_row[0])[:_h])
                    _hg = list(np.argsort(_row[0])[:_h])
                    _L = _hl + [x for x in (_pmap[y] for y in _dr['loss'][_j] if _valid[y]) if x not in set(_hl)]
                    _G = _hg + [x for x in (_pmap[y] for y in _dr['gain'][_j] if _valid[y]) if x not in set(_hg)]
                    _kk = min(len(_L), len(_G))
                    cls_logits[_i, _cols] = ro_transplant_scores(_row, np.asarray([_L[:_kk]]),
                                                                 np.asarray([_G[:_kk]]))[0]
                o['readout_rounds'] = _dr['n_rounds']
                if isinstance(cls_soft_predictions, pd.DataFrame):
                    cls_soft_predictions = pd.DataFrame(cls_logits, columns=cls_columns,
                                                        index=cls_soft_predictions.index, copy=False)
                    o[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = cls_soft_predictions
                log.info(f"[Readout] deflate: {len(_dr['rows'])}/{_Zr.shape[0]} rows decoded past round 1 "
                         f"(depth-matched noise stop; median n_cells {float(np.median(_nrow)):.0f}).")

        # ---- Decode of the query's RESPONSE over the perturbation dictionary ----
        # This is the readout the atom design exists for, so it runs on z_shared -- the same
        # geometry the atoms live in -- and never on the CE head's W. OMP over a CE-trained W was
        # measured as a dead wash (0.1116 vs 0.1115): CE shapes W for winner-take-all, not as a
        # basis, so the head is not a dictionary no matter how well it classifies.
        if decode_perturbations:
            _Z, _obs = _Zr, _obs_rows
            # Pseudobulk before decoding when asked. Per-cell latents are noisy and the decode is a
            # least-squares fit, so a group mean is the operating point the combination results
            # were measured at; None decodes the query as one profile.
            if decode_group_key is not None:
                if decode_group_key not in _obs:
                    raise KeyError(f'decode_group_key {decode_group_key!r} not in adata.obs')
                _g = _obs[decode_group_key].astype(str).to_numpy()
                _keys = list(dict.fromkeys(_g))
                _Q = np.stack([_Z[_g == kk].mean(0) for kk in _keys])
                _ncell = [float(_nrow[_g == kk].sum()) for kk in _keys]
            else:
                _keys = ['__query__']
                _Q = _Z.mean(0, keepdims=True)
                _ncell = [float(_nrow.sum())]

            _P, _labs = self._readout_prototypes(cls2z)
            if _noise is None:
                log.warning('[Decode] no readout noise calibration on this checkpoint (add_readout_calibration): '
                            'the depth-matched stop is off, every round is used; the top-%d is still exactly '
                            'the cosine readout, and effect sizes are NaN.', int(decode_head))
            _rr = ro_calibrated_readout(_Q - (_ro_base if _ro_base is not None else 0.0), _P, _ncell,
                                        noise=_noise, rounds=int(decode_rounds), head=int(decode_head),
                                        k=int(decode_top))
            _res = {'mode': 'deflate', 'calibrated': _noise is not None,
                    'dictionary': 'atoms' if getattr(getattr(self.module, 'cls_emb', None),
                                                     'random_init_proxies', False) else 'adapter',
                    'n_atoms': int(_P.shape[0]), 'head': int(decode_head),
                    'loss': _labs[_rr['loss']], 'gain': _labs[_rr['gain']],
                    'loss_idx': _rr['loss'], 'gain_idx': _rr['gain'],
                    'round_loss': _labs[_rr['top_loss']], 'round_gain': _labs[_rr['top_gain']],
                    'n_rounds': _rr['n_rounds'], 'energy': _rr['energy'],
                    'effect_size': _rr['effect_size'], 'explainable': _rr['explainable'],
                    'max_cos': _rr['max_cos']}
            _res['groups'] = _keys
            _res['group_key'] = decode_group_key
            _res['n_cells'] = _ncell
            # GROUPS x PERTURBATIONS, the per-row soft-prediction layout at group level: the group's own cosine
            # values, re-ordered by the merged readout (readout.transplant_scores). The top/bottom decode_head of
            # every group are the plain cosine readout; a group the stop never let past round 1 is its cosine
            # row unchanged. Rank-based downstream steps (top-k, enrichment) read this exactly like
            # soft_predictions.
            _Qc = _Q - (_ro_base if _ro_base is not None else 0.0)
            _S1 = ((_Qc / np.clip(np.linalg.norm(_Qc, axis=1, keepdims=True), 1e-12, None))
                   @ (_P / np.clip(np.linalg.norm(_P, axis=1, keepdims=True), 1e-12, None)).T)
            _res['scores'] = pd.DataFrame(ro_transplant_scores(_S1, _rr['loss'], _rr['gain']).astype(np.float32),
                                          index=pd.Index(_keys, name=decode_group_key or 'group'),
                                          columns=pd.Index(_labs, name='perturbation'))
            _res['group_stats'] = pd.DataFrame({'n_cells': np.asarray(_ncell, np.float64),
                                                'n_rounds': np.asarray(_rr['n_rounds'], np.int64),
                                                'effect_size': np.asarray(_rr['effect_size'], np.float64),
                                                'explainable': np.asarray(_rr['explainable'], np.float64),
                                                'max_cos': np.asarray(_rr['max_cos'], np.float64)},
                                               index=_res['scores'].index)
            o[PREDICTION_KEYS.DECODE_KEY] = _res
            log.info(f"[Decode] {len(_keys)} group(s) over {_res['n_atoms']} {_res['dictionary']} prototypes, "
                     f"head {decode_head}, <= {decode_rounds} rounds (mean used "
                     f"{float(np.mean(_res['n_rounds'])):.2f}), calibrated={_res['calibrated']}.")
        # Add classification report
        if add_report:
            log.info(f'[Predict] Adding classification report.')
            s, r = self.get_prediction_output_report(o)
            o[PREDICTION_KEYS.SUMMARY_KEY] = s
            o[PREDICTION_KEYS.REPORT_KEY] = r
        # Restore default thread settings
        if _prev_threads is not None:
            torch.set_num_threads(_prev_threads)
        # Inplace mode: write predictions/latents directly into the adata
        # the user passed in. Only kicks in when an adata was supplied (we
        # don't want to mutate self.adata implicitly).
        if inplace and is_query and user_adata is not None:
            if use_full:
                # Predicted over the full adata — mutate the user's adata
                # in place. ``align_model_features(inplace=True)`` above
                # has already realigned ``user_adata``'s var space to the
                # model's, so ``adata is user_adata`` and per-cell writes
                # land 1:1 on the caller's reference (no view, no copy).
                target = user_adata
            else:
                # Predicted over a subset — return a copy of just those cells.
                target = user_adata[np.asarray(indices)].copy()
            # Mark the query
            if is_query:
                target.uns['_query'] = {'model': self._name}
            # obs columns
            target.obs[PREDICTION_KEYS.PREDICTION_KEY] = cls_predictions
            target.obs['pred_cls_idx'] = pred_cls_idx
            if has_ctx_logits:
                target.obs[PREDICTION_KEYS.CTX_PREDICTION_KEY] = ctx_predictions
            if has_z_cls_logits:
                target.obs[PREDICTION_KEYS.Z_PREDICTION_KEY] = z_cls_predictions
            if isinstance(response_probs, np.ndarray) and response_probs.size > 0:
                target.obs['response_probs'] = response_probs
            if eff_hat_arr is not None:
                target.obs['eff_hat'] = eff_hat_arr
            if 'readout_confidence' in o:
                target.obs['readout_confidence'] = o['readout_confidence']
                target.obs['readout_effect'] = o['readout_effect']
            if 'readout_rounds' in o:
                target.obs['readout_rounds'] = o['readout_rounds']
            if 'effect_size' in o:
                target.obs['effect_size'] = o['effect_size']
            if PREDICTION_KEYS.DECODE_KEY in o:
                # inplace returns None, so the group decode must land on the object or it is silently lost.
                # Only h5ad-writable members: the score matrix, the per-group table, merged lists as str arrays.
                _d = o[PREDICTION_KEYS.DECODE_KEY]
                target.uns[PREDICTION_KEYS.DECODE_KEY] = {
                    'scores': _d['scores'], 'group_stats': _d['group_stats'],
                    'loss': np.asarray(_d['loss']).astype(str), 'gain': np.asarray(_d['gain']).astype(str),
                    'group_key': '' if _d['group_key'] is None else str(_d['group_key']),
                    'head': int(_d['head']), 'calibrated': bool(_d['calibrated']), 'dictionary': str(_d['dictionary'])}
            if true_weights_arr is not None:
                target.obs[REGISTRY_KEYS.CLS_EFF_KEY] = true_weights_arr
            # Per-class novelty info (zero-shot similarity to trained set).
            # Stored on uns so the dual-input ``get_prediction_output_report``
            # can reconstruct the same zero-shot breakdown from an inplace
            # adata it would have produced from the dict output.
            if cls_zero_shot_info is not None:
                target.uns['cls_zero_shot_info'] = cls_zero_shot_info
            # obsm. Soft predictions are only stored when soft_predictions=True;
            # otherwise skip the obsm write entirely. Column names land in uns
            # so downstream code can still recover label order without the
            # DataFrame wrapping.
            if soft_predictions and cls_soft_predictions is not None:
                target.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = cls_soft_predictions
                if has_ctx_logits and ctx_soft_predictions is not None:
                    target.obsm[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY] = ctx_soft_predictions
                if has_z_cls_logits and z_cls_soft_predictions is not None:
                    target.obsm[PREDICTION_KEYS.Z_SOFT_PREDICTION_KEY] = z_cls_soft_predictions
            # Always persist the column-name vectors so callers can recover
            # the class-name order from a saved adata even if the soft
            # prediction matrix was dropped.
            target.uns['cls_columns'] = cls_columns
            if has_ctx_logits and ctx_cols is not None:
                target.uns['ctx_columns'] = ctx_cols
            if has_z_cls_logits and z_cls_cols is not None:
                target.uns['z_cls_columns'] = z_cls_cols
            if return_latents:
                target.obsm[self._LATENT_Z_KEY] = zs
                target.obsm[self._LATENT_Z_SHARED_KEY] = z_shared
                if qz_means is not None and qz_vars is not None:
                    target.obsm[self._LATENT_QZM_KEY] = qz_means
                    target.obsm[self._LATENT_QZV_KEY] = qz_vars
            # uns (single-row tables)
            if return_latents:
                if ctx2z is not None:
                    target.uns[self._LATENT_CTX_PROJ_KEY] = ctx2z
                target.uns[self._LATENT_CLS_PROJ_KEY] = cls2z
            # Classification summary / per-class report from add_report=True.
            # Mirrors evaluate()'s layout (uns[SUMMARY_KEY]/uns[REPORT_KEY])
            # so downstream code reading either inplace adata uniformly works.
            # The zero-shot breakdowns live on ``summary.attrs`` which doesn't
            # survive h5ad round-trip, so we also spill them to their own uns
            # keys when present.
            if add_report and PREDICTION_KEYS.SUMMARY_KEY in o:
                target.uns[PREDICTION_KEYS.SUMMARY_KEY] = o[PREDICTION_KEYS.SUMMARY_KEY]
                target.uns[PREDICTION_KEYS.REPORT_KEY] = o[PREDICTION_KEYS.REPORT_KEY]
                # Spill report .attrs entries into uns so they survive
                # h5ad round-trip (DataFrame .attrs is in-memory only).
                _attrs_keys = (
                    'zero_shot_summary', 'zero_shot_per_class',
                    'emb_quality_proj_summary', 'emb_quality_proj_per_class',
                    'emb_quality_raw_summary', 'emb_quality_raw_per_class',
                    'adapter_fidelity_summary', 'adapter_fidelity_per_class',
                )
                for _k in _attrs_keys:
                    _v = o[PREDICTION_KEYS.SUMMARY_KEY].attrs.get(_k)
                    if _v is not None:
                        target.uns[_k] = _v
            # Raw cls embedding + labels — feeding these to uns lets a
            # later ``get_prediction_output_report(adata)`` reproduce the
            # soft-quality and adapter-fidelity metrics from disk.
            if cls_emb_raw_np is not None:
                target.uns['cls_emb_raw'] = cls_emb_raw_np
                target.uns['cls_labels'] = np.asarray(cls_labels)
            # Optional post-write minification: drop X now that latents are
            # in obsm[QZM/QZV]. The downstream model can re-classify against
            # this adata via the cached-inference path without ever needing
            # raw counts again — huge memory win for stored query results.
            if minify:
                log.info(f'[Predict] Minifying adata.')
                self.minify_query(target)
            return None if use_full and not copy else target
        return o
        
    def get_training_plan(self, **plan_kwargs):
        # Create training plan
        return self._training_plan_cls(
            module=self.module, 
            n_classes=self.n_labels,
            **plan_kwargs
        )
    
    @property
    def is_minified(self) -> bool:
        return _get_adata_minify_type(self.adata) is not None
    
    @torch.no_grad()
    def _readout_prototypes(self, cls2z=None) -> tuple[np.ndarray, np.ndarray]:
        """`(P, labels)` -- the trained prototype block the readout scores against, cut to the trained rows.

        Prefers `cls2z`, the class bank the A head ACTUALLY scores (`_class_reps`: the atom table after any
        prior forward blend), so the readout ranks exactly like the soft predictions -- the raw atom table
        agreed with the logits' top-10 on only 91.5% of GSE146194 cells. Without a forward pass (cls2z None)
        it falls back to the atom table (A (*) U)."""
        ce = getattr(self.module, 'cls_emb', None)
        if cls2z is None and bool(getattr(ce, 'random_init_proxies', False)):
            _, labels, P = ro_perturbation_prototypes(self, which='atoms')
            return np.asarray(P, np.float64), np.asarray(labels).astype(str)
        if cls2z is None:
            raise ValueError('no learnable atom table and no cls2z from a forward pass')
        P = np.asarray(cls2z, np.float64)
        n = int(getattr(self.module, 'n_labels', P.shape[0]))
        if 0 < n < P.shape[0]:
            P = P[:n]
        return P, np.asarray(self.idx_to_label).astype(str)[:P.shape[0]]

    def add_readout_calibration(self, ctrl_pools, datasets=None, pool_cells: int = 10, span: int = 256,
                                cls2z=None) -> dict:
        """POST-TRAINING readout calibration: store the noise level of a CONTROL delta as a function of cell
        count, which the deflate readout's stopping rule and the effect sizes are measured against.

        ``ctrl_pools``: the model's own encoding of TRAINING control cells pooled `pool_cells` at a time,
        each minus its dataset's control mean -- an .npz with 'Q' and 'dataset' (paper/scripts/
        fig5_hubness.py --stage ctrlref writes exactly that), or an (n, d) array plus ``datasets``.
        Two curves are fitted (log noise95 linear in log n): ||z|| (effect size) and its norm in the top-
        `span` prototype directions (the stop). Saved with the model; predict() uses it automatically.
        """
        if isinstance(ctrl_pools, (str, os.PathLike)):
            d = np.load(ctrl_pools, allow_pickle=True)
            C, datasets = d['Q'], d['dataset']
        else:
            C = ctrl_pools
        if datasets is None:
            raise ValueError("add_readout_calibration needs the dataset of every control pool "
                             "(pools are averaged only within a dataset)")
        C = np.asarray(C, np.float64)
        P, _ = self._readout_prototypes(cls2z)
        a = ro_fit_noise_curve(C, datasets, None, pool_cells=pool_cells)
        b = ro_fit_noise_curve(C, datasets, ro_span_basis(P, span), pool_cells=pool_cells)
        self.module.store_readout_noise([a[0], a[1], b[0], b[1]])
        info = dict(n_pools=int(C.shape[0]), n_datasets=int(len(np.unique(np.asarray(datasets).astype(str)))),
                    slope_norm=a[1], slope_span=b[1], noise95_norm_1cell=float(np.exp(a[0])),
                    noise95_norm_100cells=float(np.exp(a[0] + a[1] * np.log(100))),
                    noise95_span_100cells=float(np.exp(b[0] + b[1] * np.log(100))))
        log.info(f'[readout-calibration] stored noise curve: {info}')
        return info

    def add_effect_head(self, head) -> dict:
        """POST-TRAINING effect-size readout, read out by predict() as ``obs['effect_size']``.

        Combines the trained efficiency head with the cosine confidence of the cell's call: effect_size =
        [eff_hat, max_cos, margin, p_top1, ||z_c||] @ w + b, where the cosines are those of the dataset-control-centred
        z_shared to the trained prototypes and p_top1 = softmax(top-k cos / T)[0]. ``head``: the .npz written by
        analysis/effect_size/effect_stack.py (keys w, b, k, T, inputs) or a dict with the same keys, fitted on training
        cells held out of every earlier fit, target = within-dataset percentile of |Mixscale| (so the output is a
        percentile-like score, ~0-1). Needs the efficiency head (model_efficiency). Save the model to keep it.
        """
        h = np.load(head, allow_pickle=True) if isinstance(head, (str, os.PathLike)) else head
        w = np.asarray(h['w'], np.float64).ravel()
        order = ('eff_hat', 'max_cos', 'margin', 'p_top1', 'z_norm')
        if 'inputs' in h and tuple(np.asarray(h['inputs']).astype(str)) != order:
            raise ValueError(f'effect head inputs {tuple(np.asarray(h["inputs"]).astype(str))} != {order}')
        if w.size != len(order):
            raise ValueError(f'effect head needs {len(order)} weights, got {w.size}')
        if not getattr(self.module, 'model_efficiency', False):
            raise ValueError('effect-size readout needs the efficiency head (model_efficiency=True)')
        from expert.utils.readout import perturbation_prototypes
        _, labels, _ = perturbation_prototypes(self)
        meta = [float(np.asarray(h['b']).item()), float(np.asarray(h['k']).item()), float(np.asarray(h['T']).item()),
                float(len(labels))]
        self.module.store_effect_head(w, meta)
        info = dict(w=dict(zip(order, np.round(w, 6).tolist())), b=meta[0], k=int(meta[1]), T=meta[2],
                    n_labels=len(labels))
        log.info(f'[effect-head] stored: {info}')
        return info

    def add_hub_correction(self, bank, form: Literal['isoft', 'csls'] = 'isoft', beta: float = 30.0,
                           k: int = 25, fill: Literal['bank', 'nbr'] = 'bank', fill_k: int = 25,
                           chunk: int = 8192) -> dict:
        """POST-TRAINING hubness correction: store a per-atom popularity offset that predict() and the
        readout subtract from the cosine (score = cos - pen_a; the same offset for every query).

        ``bank``: unlabelled REFERENCE queries in z space -- the model's own encoding of its TRAINING
        pseudobulks (e.g. analysis/readout/cache_queries.py on id_train capped at 10 cells per
        dataset x perturbation): an .npz path with 'Q' or an (n, d) array. Never test/OOD queries.

        form='isoft' (inverted softmax, validated on ALD): pen_a = logsumexp_r(beta cos(r,a)) / beta.
        form='csls': pen_a = mean of a's top-k cosines / 2 (CSLS 2cos - r_a, halved to cosine units).
        Two rows are stored: popularity of +atom (classification, 'loss' tail) and of -atom ('gain').
        Untrained (zero-shot) rows: fill='nbr' gives each filled row the mean penalty of its fill_k
        nearest trained atoms (the fill sits on top of the corrected trained table); fill='bank' scores
        them against the bank like trained rows. Rows still at random init get 0.
        Save the model to keep it; predict(hub_correction=False) scores plain cosine.
        """
        import torch.nn.functional as _F
        mod = self.module
        if isinstance(bank, (str, os.PathLike)):
            bank = np.load(bank, allow_pickle=True)['Q']
        R = _F.normalize(torch.as_tensor(np.asarray(bank, np.float32)), dim=1)
        A = mod.effective_atoms()
        if A is None:
            raise ValueError('add_hub_correction needs a learnable atom table (cls_emb)')
        n_lab = int(mod.n_labels)
        if mod.has_prior_missing_fill() and 'prior_missing_T' in mod._buffers:
            A = mod._prior_missing_apply(A)
        A = A.detach().float().cpu()
        filled = torch.zeros(A.shape[0], dtype=torch.bool)
        if A.shape[0] > n_lab and 'prior_missing_T' in mod._buffers:
            filled[n_lab:] = mod.prior_missing_T[n_lab:A.shape[0]].detach().float().cpu().norm(dim=1) > 0
        An = _F.normalize(A, dim=1)
        rows = (torch.arange(n_lab) if fill == 'nbr'
                else torch.cat([torch.arange(n_lab), torch.nonzero(filled).flatten()]))
        pen = torch.zeros(2, A.shape[0])
        for s, sgn in ((0, 1.0), (1, -1.0)):
            Ar = sgn * An[rows]
            if form == 'isoft':
                acc = torch.full((len(rows),), -float('inf'))
                for i in range(0, R.shape[0], chunk):
                    acc = torch.logaddexp(acc, torch.logsumexp(beta * (R[i:i + chunk] @ Ar.T), dim=0))
                pen[s, rows] = acc / beta
            elif form == 'csls':
                top = None
                for i in range(0, R.shape[0], chunk):
                    blk = R[i:i + chunk] @ Ar.T
                    top = blk if top is None else torch.cat([top, blk])
                    top = torch.topk(top, min(k, top.shape[0]), dim=0).values
                pen[s, rows] = top.mean(0) / 2.0
            else:
                raise ValueError(f'form must be isoft or csls, got {form!r}')
        if fill == 'nbr' and bool(filled.any()):
            fi = torch.nonzero(filled).flatten()
            for i in range(0, len(fi), chunk):
                f = fi[i:i + chunk]
                nb = torch.topk(An[f] @ An[:n_lab].T, fill_k, dim=1).indices
                pen[:, f] = pen[:, :n_lab][:, nb].mean(-1)
        mod.store_hub_correction(pen)
        info = dict(form=form, beta=beta, k=k, fill=fill, fill_k=fill_k, n_bank=int(R.shape[0]),
                    n_trained=n_lab, n_filled=int(filled.sum()),
                    pen_trained_median=float(pen[0, :n_lab].median()),
                    pen_filled_median=float(pen[0, filled].median()) if bool(filled.any()) else None)
        log.info(f'[hub-correction] stored {tuple(pen.shape)} offset: {info}')
        return info

    def add_prior_missing_fill(self, config: str | dict | None = None, adata_path: str | None = None,
                               fold_dir: str | None = None) -> int:
        """POST-HOC f(missing) fill for a trained prior-graph model that predates the saved buffer.

        Refits the prior map on the model's atoms exactly as training does and stores the untrained-row
        predictions in ``module.prior_missing_T``; predict(obs_only=False) then scores zero-shot columns
        against them. Save the model (or append the key to model.pt) to keep it.

        The prior banks are checkpoint buffers, but the fit set comes from the train-split evidence ranks,
        so this reads the TRAINING DB's obs columns (efficiency, label, dataset; h5py, no X) and the fold's
        ``id_train``. Paths come from the run config (``adata_path`` / ``fold_dir`` anywhere in it) unless
        passed. Deliberately NEVER called automatically: a shipped model usually cannot reach those files.
        Returns the number of filled rows.
        """
        if isinstance(config, str):
            import yaml
            with open(config) as fh:
                config = yaml.safe_load(fh)

        def _find(d, key):
            if isinstance(d, dict):
                if d.get(key):
                    return d[key]
                for v in d.values():
                    r = _find(v, key)
                    if r:
                        return r
            return None
        adata_path = adata_path or _find(config, 'adata_path')
        fold_dir = fold_dir or _find(config, 'fold_dir')
        if not adata_path or not fold_dir:
            raise ValueError(f'add_prior_missing_fill needs adata_path and fold_dir (got {adata_path!r}, {fold_dir!r})')
        import h5py
        from anndata.io import read_elem
        cols = ['efficiency', self.original_label_key, self.original_batch_key]
        with h5py.File(adata_path, 'r') as fh:
            obs = pd.DataFrame({c: read_elem(fh['obs'][c]) for c in cols})
        md = io.load_json(os.path.join(os.path.dirname(os.path.normpath(fold_dir)), 'metadata.json'))
        tr = np.loadtxt(os.path.join(fold_dir, f"id_train.{md.get('index_format', 'txt.gz')}"), dtype=np.int64, ndmin=1)
        log.info(f'[prior-missing] evidence ranks from {len(tr):,} train cells of {len(obs):,} ({fold_dir})')
        self._compute_prior_atom_eff_gate(obs=obs, train_indices=tr)
        return self.module.rebuild_prior_missing()

    def _compute_prior_atom_eff_gate(self, obs: pd.DataFrame | None = None,
                                     train_indices: np.ndarray | None = None) -> None:
        """Per-class |eff| quality score for the prior-atom gate, from TRAIN CELLS ONLY.

        MIRRORS THE OFFLINE GATE EXACTLY (`analysis/readout/band_by_dataset.py`, gate_mode
        'absrank'): per-cell |eff| -> percentile WITHIN DATASET -> per-class MEDIAN. Every number
        the arm rests on was measured on this quantity, so it is computed the same way rather than
        estimated:
          * |eff|, not raw. 13.6% of train cells are NEGATIVE, which is a strong effect in the
            other direction, not a weak one. The model takes .abs() at the two consuming sites.
          * per-DATASET percentile. Raw per-dataset medians span 0.0-3.3, so a global threshold on
            raw efficiency mostly measures which screen a class came from.
          * per-class MEDIAN of the percentiles, over ALL of that class's train cells.

        WHY AT SETUP AND NOT ACCUMULATED. The first attempt kept a per-class running mean of
        per-BATCH percentiles. A batch holds ~100 cells per dataset, so each observation is a noisy
        estimate of the class's whole-dataset percentile, and the running mean converges only
        slowly -- it read w_mean 0.0609 at ~15 observations against an offline-equivalent ~0.096.
        Computed here it is exact from step 1: no warmup, no coverage gate, no accumulation, and
        no dependence on how many times a class happened to be sampled.

        TRAIN CELLS ONLY. `self.train_indices` comes from the fold's `id_train`, so val/test/
        ood_test never enter. Efficiency is measured per cell, so folding in held-out cells would
        put held-out response strength into a weight that steers training -- the same family as
        the class_centers and GECO leaks, and just as invisible.
        """
        self.module.atom_eff_qual = None
        self.module.atom_ntrain_qual = None
        self.module.atom_nqxnds_qual = None
        self.module.atom_ntrain_count = None
        self.module.atom_nds_count = None
        # ALSO build when only the NEGATIVES are on. f(missing) needs the fitter, the fitter
        # needs this gate, and gating on the pull weight alone made `prior_atom_neg_weight`
        # unusable without a pull -- which is precisely the isolating control arm.
        if not (io.non_zero(getattr(self.module, 'prior_atom_weight', 0.0))
                or io.non_zero(getattr(self.module, 'prior_atom_neg_weight', 0.0))):
            return
        # obs / train_indices default to the training setup; add_prior_missing_fill passes them explicitly
        # for a loaded model, which has neither the full DB nor the split.
        tr = train_indices if train_indices is not None else getattr(self, 'train_indices', None)
        if tr is None or not len(tr):
            log.warning('[prior-atoms] no train_indices at setup -- the |eff| gate CANNOT be '
                        'built and the term will refuse (state 2). Pass a fold_dir.')
            return
        try:
            obs = self.adata.obs if obs is None else obs
            tr = np.asarray(tr)
            # CATEGORICAL CODES, NEVER .astype(str). At 19.5M rows `.astype(str)` materialises
            # 19.5M individual Python str objects (~1.2 GB), the train slice copies ~0.9 GB more,
            # and that is per column -- then np.argsort runs on an OBJECT array, comparing strings
            # in Python. These arms already sit AT the 250 GiB cgroup ceiling without this term
            # (MM200 peaked at 250.0 GiB and completed), so a few spare GB at setup is enough to
            # OOM the run hours later -- which is exactly what killed PA50 at epoch 12.
            # Codes are int32: 78 MB per column, and the grouping below is pure integer work.
            eff = np.asarray(obs['efficiency'].to_numpy(), dtype=np.float32)[tr]
            _lc, _dc = obs[self.original_label_key], obs[self.original_batch_key]
            lab = (_lc.cat.codes.to_numpy() if hasattr(_lc, 'cat')
                   else pd.Categorical(_lc).codes)[tr]
            ds = (_dc.cat.codes.to_numpy() if hasattr(_dc, 'cat')
                  else pd.Categorical(_dc).codes)[tr]
            lab_names = [str(x) for x in (_lc.cat.categories if hasattr(_lc, 'cat')
                                          else pd.Categorical(_lc).categories)]
        except Exception as e:                                    # noqa: BLE001
            log.warning('[prior-atoms] could not read the |eff| gate inputs (%r) -- the term '
                        'will refuse rather than run on a wrong gate', e)
            return
        ok = np.isfinite(eff)
        eff, lab, ds = np.abs(eff[ok]), lab[ok], ds[ok]
        rk = np.empty_like(eff)
        for d in np.unique(ds):                                   # percentile WITHIN dataset
            m = ds == d
            e = eff[m]
            o = np.argsort(e, kind='stable')
            r = np.empty(len(e)); r[o] = np.arange(len(e))
            rk[m] = r / max(len(e) - 1, 1)
        order = np.argsort(lab, kind='stable')                     # integer sort, not object
        ls, rs = lab[order], rk[order]
        med, cnt, st = {}, {}, 0
        for i in range(1, len(ls) + 1):                            # per-class MEDIAN percentile
            if i == len(ls) or ls[i] != ls[st]:
                med[lab_names[ls[st]]] = float(np.median(rs[st:i]))
                cnt[lab_names[ls[st]]] = i - st
                st = i
        # A class with a handful of train cells has a median percentile set by noise. Leaving it
        # UNCOVERED (NaN) means it is never pulled, which is the safe default -- a prior should not
        # reposition an atom on the strength of three cells.
        _mc = int(getattr(self.module, 'prior_atom_min_cells', 8))
        labs = [str(x) for x in self.idx_to_label]
        q = np.array([med.get(l, np.nan) if cnt.get(l, 0) >= _mc else np.nan for l in labs],
                     dtype=np.float32)
        n_lab = int(getattr(self.module, 'n_labels', len(q)) or len(q))
        q = q[:n_lab]
        fin = np.isfinite(q)
        self.module.atom_eff_qual = torch.as_tensor(q, dtype=torch.float32)

        # TRAIN-CELL DEPTH, the evidence measure for the SOFT pull weight. `cnt` is already built
        # above for the min_cells filter, so this is free. Ranked to [0,1] with 1 = MOST cells, to
        # match atom_eff_qual's orientation; the pull weight is (1 - rank)^gamma either way.
        #
        # NO min_cells FILTER HERE, deliberately. For the |eff| gate a thin class is UNCOVERED
        # because a median percentile over three cells is noise. For this measure a thin class is
        # the POINT -- little data is exactly when the prior should be trusted -- so a NaN would
        # invert the term's intent. Classes absent from train get count 0 and rank 0, i.e. maximum
        # pull, which is the correct reading of "no evidence".
        #
        # Depth, not effect size: |eff| measures +11 as a continuous weight against this one's +64
        # ([[project_soft_alpha_flips_the_sign]]). Counting ALL train cells rather than only cells
        # with a finite efficiency is worth +2 and is what the offline optimum used.
        _nq = np.array([float(cnt.get(l, 0)) for l in labs], dtype=np.float32)[:n_lab]
        _o = np.argsort(_nq, kind='stable')
        _r = np.empty(len(_nq), dtype=np.float32)
        _r[_o] = np.arange(len(_nq), dtype=np.float32)
        _r = _r / max(len(_nq) - 1, 1)
        self.module.atom_ntrain_qual = torch.as_tensor(_r, dtype=torch.float32)

        # CELLS x N_DATASETS, measured +75 self / +72 overall offline against raw depth's +64/+62
        # (evidence_zoo, 2026-09-10) and never adopted -- production still runs evidence='ntrain'.
        # Cross-screen support is not the same thing as raw depth: 2,000 cells from ONE screen
        # pins an atom far less than 2,000 spread over six, and the product is the cheapest
        # statistic that says so. Train-split only, so it is deployable as a training-time weight.
        # VECTORISED: a Python loop over the ~14M train cells here would stall setup. Unique
        # (label, dataset) PAIRS then bincount over the label column gives the same counts.
        _pairs = np.unique(np.stack([lab, ds], 1), axis=0)
        _ndc = np.bincount(_pairs[:, 0], minlength=len(lab_names)).astype(np.float32)
        _n2i = {str(s): i for i, s in enumerate(lab_names)}
        _ndv = np.array([_ndc[_n2i[l]] if l in _n2i else 0.0 for l in labs],
                        dtype=np.float32)[:n_lab]
        _sc = _nq * np.maximum(_ndv, 1.0)
        _o2 = np.argsort(_sc, kind='stable')
        _r2 = np.empty(len(_sc), dtype=np.float32)
        _r2[_o2] = np.arange(len(_sc), dtype=np.float32)
        _r2 = _r2 / max(len(_sc) - 1, 1)
        self.module.atom_nqxnds_qual = torch.as_tensor(_r2, dtype=torch.float32)
        # RAW COUNTS as well as ranks, for gate='bayes': its weight 1/(1 + n/kappa) is a function of
        # the evidence AMOUNT, which a rank discards. Plain attributes, never buffers.
        self.module.atom_ntrain_count = torch.as_tensor(_nq, dtype=torch.float32)
        self.module.atom_nds_count = torch.as_tensor(_ndv, dtype=torch.float32)
        log.info('[prior-atoms] nq x ndatasets gate: n_datasets median %.0f (max %.0f); '
                 'spearman with raw depth %.3f', float(np.median(_ndv)), float(_ndv.max()),
                 float(np.corrcoef(np.argsort(np.argsort(_nq)),
                                   np.argsort(np.argsort(_sc)))[0, 1]))
        log.info('[prior-atoms] n_train gate: %d classes, cells/class median %.0f IQR %.0f-%.0f '
                 '(%d classes with 0 train cells -> maximum pull)',
                 n_lab, float(np.median(_nq)), float(np.percentile(_nq, 25)),
                 float(np.percentile(_nq, 75)), int((_nq == 0).sum()))
        log.info('[prior-atoms] |eff| gate from %s TRAIN cells over %d datasets: %d/%d classes '
                 'covered, score median %.3f IQR %.3f-%.3f (min %d train cells; offline '
                 'reference: median 0.500 IQR 0.424-0.587)'
                 , f'{len(eff):,}', len(np.unique(ds)), int(fin.sum()), n_lab,
                 float(np.nanmedian(q)), float(np.nanpercentile(q[fin], 25)),
                 float(np.nanpercentile(q[fin], 75)), _mc)

    def _load_splits_from_dir(self, fold_dir: str, verify_md5: bool = False) -> None:
        """Load pre-calculated data splitter indices from a fold directory (e.g. ``.../splits/fold_0``)."""
        if not os.path.isdir(fold_dir):
            raise FileNotFoundError(f'Fold directory does not exist: {fold_dir}')
        # Infer fold index from the directory name (e.g. "fold_0" -> 0)
        fold_name = os.path.basename(os.path.normpath(fold_dir))
        if not fold_name.startswith('fold_') or not fold_name[len('fold_'):].isdigit():
            raise ValueError(f'Fold directory name must match "fold_<int>", got {fold_name!r}.')
        fold = int(fold_name[len('fold_'):])
        # Locate metadata in the parent split directory
        split_dir = os.path.dirname(os.path.normpath(fold_dir))
        mf = os.path.join(split_dir, 'metadata.json')
        if not os.path.exists(mf):
            raise FileNotFoundError(f'Parent split directory {split_dir} has no metadata.json.')
        md = io.load_json(mf)
        # Validate fold index against metadata
        n_folds = md.get('n_folds', 0)
        if not (0 <= fold < n_folds):
            raise ValueError(f'Fold {fold} out of range [0, {n_folds}) for split dir {split_dir}.')
        # Validate adata shape against source referenced in metadata
        src_shape = md.get('source_shape')
        if src_shape is not None:
            # COMPARE ONLY THE AXES THE METADATA ACTUALLY RECORDED. Splits are row-index files, so
            # the builder often stores source_shape as [n_cells, None] -- the gene count is not part
            # of what a split asserts. Comparing the tuples verbatim turns "not recorded" into a
            # mismatch and rejects the very store the splits were built from (671242 vs 671242,
            # 3096 vs None). A recorded axis that disagrees is still a hard error.
            _bad = [(i, a, b) for i, (a, b) in enumerate(zip(src_shape, self.adata.shape))
                    if a is not None and int(a) != int(b)]
            if _bad:
                raise ValueError(
                    f'[Splits] adata shape {tuple(self.adata.shape)} does not match source shape '
                    f'{tuple(src_shape)} from {mf} on axis/axes '
                    f'{[i for i, _, _ in _bad]} (recorded {[a for _, a, _ in _bad]}, '
                    f'got {[b for _, _, b in _bad]}).'
                )
        # Optional md5 verification against the source h5ad file
        if verify_md5:
            src_md5 = md.get('source_md5')
            adata_path = getattr(self.adata, 'filename', None)
            adata_path = str(adata_path) if adata_path is not None else None
            if src_md5 and adata_path and os.path.exists(adata_path):
                actual_md5 = io.file_md5(adata_path)
                if actual_md5 != src_md5:
                    raise ValueError(
                        f'[Splits] adata md5 {actual_md5} does not match source md5 {src_md5}.'
                    )
                log.info(f'[Splits] Verified adata md5 matches source ({src_md5}).')
            else:
                log.warning('[Splits] md5 verification requested but source file is unavailable; skipping.')
        # Load all available indices
        idx_fmt = md.get('index_format', 'txt.gz')
        def _load_idx(name: str) -> np.ndarray | None:
            fp = os.path.join(fold_dir, f'{name}.{idx_fmt}')
            if not os.path.exists(fp):
                return None
            return np.loadtxt(fp, dtype=np.int64, ndmin=1)
        self.train_indices = _load_idx('id_train')
        self.validation_indices = _load_idx('id_val')
        self.test_indices = _load_idx('id_test')
        self.ood_test_indices = _load_idx('ood_test')
        # Validate counts against per-fold metadata
        fold_meta = (md.get('folds') or [{}] * n_folds)[fold]
        expected_counts = {
            'train': (fold_meta.get('n_train'), self.train_indices),
            'val': (fold_meta.get('n_val'), self.validation_indices),
            'test': (fold_meta.get('n_test'), self.test_indices),
            'ood_test': (fold_meta.get('n_ood'), self.ood_test_indices),
        }
        for name, (expected, idx) in expected_counts.items():
            got = 0 if idx is None else int(len(idx))
            if expected is not None and expected != got:
                raise ValueError(
                    f'[Splits] {name}: expected {expected} indices, got {got} in {fold_name}.'
                )
            # Guard against out-of-bounds indices
            if idx is not None and len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= self.adata.n_obs):
                raise ValueError(
                    f'[Splits] {name} contains out-of-bounds indices for adata.n_obs={self.adata.n_obs}.'
                )
        # Validate total cell count matches adata
        total = sum(0 if v[1] is None else len(v[1]) for v in expected_counts.values())
        if total != self.adata.n_obs:
            raise ValueError(
                f'[Splits] Loaded {total} indices across splits but adata has {self.adata.n_obs} cells.'
            )
        log.info(
            f'[Splits] Loaded {fold_name} from {split_dir}: ' + ', '.join(
                # A split file may legitimately be ABSENT (-> None): a production split has no ood_test
                # and no test at all. Only ood_test was guarded here, so a missing id_test crashed the
                # whole run on the LOG LINE, after the split had loaded fine.
                f'{_n}={0 if _v is None else len(_v):,}' for _n, _v in
                (('train', self.train_indices), ('val', self.validation_indices),
                 ('test', self.test_indices), ('ood_test', self.ood_test_indices)))
        )

    def _set_data_split_indices(self, data_splitter: torch.nn.Module):
        """Manually set split indices."""
        self.train_indices = getattr(data_splitter, "train_idx", None)
        self.validation_indices = getattr(data_splitter, "val_idx", None)
        self.test_indices = getattr(data_splitter, "test_idx", None)
        self.ood_test_indices = getattr(data_splitter, "ood_test_idx", None)
        
    @devices_dsp.dedent
    @staticmethod
    def _retag_zs_split_for_fold(zs, zs_path: str, fold_dir: str | None) -> None:
        """Re-derive the zs subset's ``zs_split`` for the fold being trained, IN MEMORY.

        ``zs_split`` is a pure function of ``dataset`` GIVEN a fold: the fold's held_out_contexts
        are ``ood_ood_pert`` (unseen pert in an UNSEEN dataset), everything else is
        ``id_ood_pert`` (unseen pert in a SEEN dataset). The CELLS are fold-independent:
        build_zs_subset5 caps every (dataset x pert) group at 5 WITHIN its arm, and every group in
        the written file sits AT the cap (55,225 groups x 5 cells), so the arm assignment only
        picks the per-arm RNG offset — never which groups survive. A fold switch therefore needs a
        RE-TAG, not a rebuild. The ACTIVE fold's metadata.json is the source of truth here; the tag
        baked in at build time is advisory and is overwritten. The file on disk is not modified."""
        if not fold_dir:
            return
        meta_p = os.path.join(os.path.dirname(os.path.normpath(fold_dir)), 'metadata.json')
        m = re.search(r'fold_(\d+)', os.path.basename(os.path.normpath(fold_dir)))
        if not (os.path.exists(meta_p) and m):
            log.warning(f'[ZS] cannot derive zs_split (no {meta_p} / unparsable {fold_dir!r}); trusting the '
                        f'tag baked into the file. If it was built for another fold the zero-shot arms are '
                        f'mislabelled.')
            return
        with open(meta_p) as fh:
            meta = json.load(fh)
        fold_i = int(m.group(1))
        folds = meta.get('folds') or []
        if fold_i >= len(folds):
            log.warning(f'[ZS] fold_{fold_i} absent from {meta_p} ({len(folds)} folds); trusting the tag baked '
                        f'into the file. If it was built for another fold the zero-shot arms are mislabelled.')
            return
        ctx_col = meta.get('context_col', 'dataset')
        if ctx_col not in zs.obs:
            # Without the context column the tag cannot be derived, and the baked one belongs to
            # whichever fold built the file — trusting it is the silent mislabel this class exists for.
            raise ZeroShotFoldMismatch(
                f"zs subset {zs_path!r} has no {ctx_col!r} obs column, so zs_split cannot be derived for "
                f"fold_{fold_i} ({fold_dir!r}). The tag baked in at build time belongs to whichever fold "
                f"built the file, so trusting it would silently mislabel the zero-shot arms (id_ood_pert vs "
                f"ood_ood_pert inverted) and report the headline number under the wrong arm.")
        # Split builders name the held-out set either 'held_out_contexts' (production / context splits) or
        # 'held_out_datasets' (the cell-line / donor fold splits under splits/full). Reading only the former left
        # every fold of the latter with an EMPTY ood_ood_pert arm (all zero-shot cells tagged id_ood_pert).
        _f = folds[fold_i]
        held = set(_f.get('held_out_contexts') or _f.get('held_out_datasets') or [])
        ds = zs.obs[ctx_col].astype(str).to_numpy()
        expected = np.where(np.isin(ds, list(held)), 'ood_ood_pert', 'id_ood_pert')
        got = zs.obs['zs_split'].astype(str).to_numpy() if 'zs_split' in zs.obs else None
        if got is not None and not np.array_equal(expected, got):
            flipped = sorted({d for d, e, g in zip(ds, expected, got) if e != g})
            log.warning(f'[ZS] re-tagging zs_split for fold_{fold_i}: {len(flipped)} dataset(s) change arm '
                        f'{flipped}. The file on disk is NOT modified.')
        zs.obs['zs_split'] = expected
        log.info(f'[ZS] zs_split derived for fold_{fold_i} ({len(held)} held-out datasets).')

    def _build_zs_manager(self, zs_path: str, fold_dir: str | None = None):
        """Build a 2nd AnnDataManager for the unseen-perturbation subset, so DataSplitter can
        serve id_ood_pert / ood_ood_pert as live val loaders. LABELS_KEY is injected as full
        BANK indices (held-out genes -> rows n_labels..n_cls) so the model's full-vocab loss
        scores them correctly (a plain transfer would map them to 'unknown'). Verified with
        verify_zs_loader.py (batch labels == bank indices; gene names line up)."""
        import anndata as ad
        from scvi import REGISTRY_KEYS
        zs = ad.read_h5ad(zs_path)
        # Before anything else: derive zs_split for the fold being trained. Must stay ahead of the
        # gene gate below (which .copy()s obs through) and of the manager build — the splitter reads
        # zs_split off this in-memory object, not off the file.
        self._retag_zs_split_for_fold(zs, zs_path, fold_dir)
        bank = np.asarray(self.idx_to_label).astype(str)
        n_labels = int(self.n_labels)
        row_of = {g: i for i, g in enumerate(bank.tolist())}
        genes = zs.obs[self.original_label_key].astype(str).to_numpy()
        row = np.array([row_of.get(g, -1) for g in genes])
        keep = row >= n_labels                                   # held-out block only (safety re-gate)
        if not keep.any():
            log.warning('[ZS] no held-out cells after gene gate; skipping zs val loaders.')
            return None
        zs = zs[keep].copy(); row = row[keep]
        # add_missing_categories=True fills placeholders for obs fields the zs DB lacks
        # (e.g. 'efficiency'), same as predict(), then transfers + registers a manager.
        self._validate_anndata(zs, add_missing_categories=True)
        zsm = self.get_anndata_manager(zs)
        lab_attr = zsm.data_registry[REGISTRY_KEYS.LABELS_KEY].attr_key
        zs.obs[lab_attr] = row.astype('int64')                   # OVERWRITE encoded labels = bank indices
        vc = zs.obs['zs_split'].value_counts().to_dict() if 'zs_split' in zs.obs else {}
        log.info(f'[ZS] eval manager built: {zs.n_obs:,} held-out cells | splits {vc}')
        return zsm

    def train(
        self,
        config: dict[str, Any],
        verbose: bool = True,
        clean_tmp: bool = False,
    ):
        """Train the model.

        Parameters
        ----------
        config: dict[str, dict]
            full config with subconfigs for model setup and training
        """
        from expert.tune._statics import CONF_KEYS
        # Unpack run config dictionary
        data_params: dict[str, Any] = config.get(CONF_KEYS.DATA, {})
        model_params: dict[str, Any] = config.get(CONF_KEYS.MODEL, {})
        train_params: dict[str, Any] = config.get(CONF_KEYS.TRAIN, {})
        # Get max epochs specified in params
        epochs = train_params.get('max_epochs')
        # Train for suggested number of epochs if no specification is give
        if epochs is None:
            # Determine number of epochs needed for complete training
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))
            epochs = max_epochs
            log.info(f'Epochs suggested: {max_epochs}, training for {epochs} epochs.')
        # Get training split fraction
        train_size: int = data_params.pop('train_size', 0.9)
        if not train_size < 1.0 and train_size > 0:
            raise ValueError(f'Parameter train_size should be between 0 and 1, got {train_size}')
        
        # Check if batch size is defined or not
        if data_params.get("batch_size") is None:
            # Fall back to product of max cells and max classes per batch
            msb = data_params.get("max_cells_per_batch", 32)
            mcb = data_params.get("max_classes_per_batch", 16)
            batch_size = int(msb * mcb)
            data_params["batch_size"] = batch_size
        
        # Add control class index to data params
        data_params['ctrl_class'] = self.ctrl_class

        # TODO: fix cache indices from pre-training path
        # Cache indices if model was pre-trained/ resumed from checkpoint or pre-split
        fold_dir = data_params.pop('fold_dir', None)
        if self.was_pretrained or fold_dir is not None:
            if fold_dir is not None:
                # Check if indices can be loaded from pre-computed split
                log.info(f'[Splits] Re-using pre-computed data split from {fold_dir}.')
                self._load_splits_from_dir(fold_dir, verify_md5=bool(data_params.get('verify_md5', False)))
                # AFTER the splits, so train_indices is populated and the gate is train-only.
                self._compute_prior_atom_eff_gate()
        
            # Pass loaded indices to data loader. ``getattr`` with a default
            # only fires when the attribute is *missing*; scvi's
            # ``BaseModelClass`` and our own ``_set_data_split_indices`` set
            # these to ``None`` on a fresh model, so the default never fires
            # and ``len(None)`` crashes the caching guard below. Coerce
            # ``None`` → empty so the guard treats "no cache yet" the same
            # whether the attribute is missing or explicitly ``None``.
            def _idx_or_empty(name):
                v = getattr(self, name, None)
                return v if v is not None else np.array([], dtype=np.int64)
            cache_indices = {
                'train':    _idx_or_empty('train_indices'),
                'val':      _idx_or_empty('validation_indices'),
                'test':     _idx_or_empty('test_indices'),
                'ood_test': _idx_or_empty('ood_test_indices'),
            }
            # Add to splitter params if caching worked
            if len(cache_indices['train']) > 0:
                data_params['cache_indices'] = cache_indices

        # Optional: 2nd manager for the unseen-perturbation subset -> live id_ood_pert /
        # ood_ood_pert val loaders (loss curves). Path from cluster.unseen_pert_adata_path
        # (shared with the post-training evaluate_zero_shot_perturbations eval in tune/run.py);
        # silently skipped when unset/missing.
        zs_adata_manager = None
        _cluster = config.get('cluster', {}) or {}
        _zs_p = _cluster.get('unseen_pert_adata_path')
        if _zs_p and os.path.exists(_zs_p):
            try:
                # `fold_dir` is the LOCAL popped at the top of this block, NOT data_params['fold_dir']
                # — that key no longer exists here, and reading it back would silently pass None and
                # turn the fold guard into a no-op.
                zs_adata_manager = self._build_zs_manager(_zs_p, fold_dir=fold_dir)
            except ZeroShotFoldMismatch:
                # NEVER downgrade this one to a skip: an underivable zs_split does not break the run,
                # it silently reports the zero-shot headline under the wrong arm. Kill the run instead.
                raise
            except Exception as e:
                log.warning(f'[ZS] could not build zeroshot eval manager from {_zs_p}: {e}; skipping zs val loaders.')
        elif _zs_p:
            log.warning(f'[ZS] zs_subset path set but not found: {_zs_p}; skipping zs val loaders.')

        # Create data splitter
        data_splitter = DataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            zs_adata_manager=zs_adata_manager,
            **data_params,
        )
        # The prior-atom gate is built from TRAIN cells only. A fold_dir supplies them above; a random split exists
        # only once the splitter is set up, so without this the pull and its negatives refuse (state 2). The split
        # is seeded and local-RNG only, so Lightning's own setup() recomputes the identical one.
        _mod = self.module
        _prior_on = (io.non_zero(getattr(_mod, 'prior_atom_weight', 0.0))
                     or io.non_zero(getattr(_mod, 'prior_atom_neg_weight', 0.0)))
        if _prior_on and getattr(_mod, 'atom_eff_qual', None) is None and getattr(_mod, 'atom_nqxnds_qual', None) is None:
            data_splitter.setup()
            self._set_data_split_indices(data_splitter)
            self._compute_prior_atom_eff_gate()
        # Setup training plan
        plan_kwargs: dict = train_params.pop('plan_kwargs', {})

        # Share code to label mapping with training plan
        plan_kwargs['_code_to_label'] = self.idx_to_label.copy()
        # Share batch labels with training plan
        plan_kwargs['batch_labels'] = self._batch_mapping.copy()
        # Share covariate labels with training plan
        plan_kwargs['cov_labels'] = self.idx_to_cat_cov.copy()
        # Create training plan
        training_plan = self.get_training_plan(**plan_kwargs)
        # Create the versioned log dir now that training is actually starting
        # (deferred from __init__ so load() doesn't touch a stale path).
        self._ensure_log_dir()
        # Reuse cached logger or pick up from train_params
        logger = train_params.get('logger') or self._logger
        if logger is not None:
            # Cache for subsequent train() calls (resumes append to same log dir)
            self._logger = logger
            train_params['logger'] = logger
            log.info(f'[Logging] model_log_dir={logger.log_dir}')
        callbacks = []
        # Manage early stopping callback
        early_stopping_start_epoch = train_params.pop('early_stopping_start_epoch')
        if train_params.get('early_stopping', False) and early_stopping_start_epoch is not None:
            # Disable scvi internal early stopping
            train_params['early_stopping'] = False
            # Create custom delayed early stopping using the start epoch parameter
            early_stopping_callback = DelayedEarlyStopping(
                start_epoch=early_stopping_start_epoch,
                warmup_epochs=plan_kwargs.get('n_epochs_warmup', 400),
                monitor=train_params.get('early_stopping_monitor', 'validation_loss'),
                min_delta=train_params.get('early_stopping_min_delta', 0.0),
                patience=train_params.get('early_stopping_patience', 0.0),
                mode=train_params.get('early_stopping_mode', 0.0),
            )
            # Add to callbacks
            callbacks.append(early_stopping_callback)
        # Manage checkpoint callback
        use_checkpoint = train_params.pop('checkpoint', False)
        checkpoint_monitor = train_params.pop('checkpoint_monitor', 'validation_loss')
        checkpoint_mode = train_params.pop('checkpoint_mode', 'max')
        # save_top_k: 1 = best-by-monitor only (default/legacy); -1 = keep all; k = k best.
        # every_n_epochs: when set with save_top_k=-1, save a *staged* checkpoint every N
        # epochs and keep them all (trajectory sweeps). N should be a
        # multiple of check_val_every_n_epoch so the monitor value in the filename is fresh.
        checkpoint_save_top_k = train_params.pop('checkpoint_save_top_k', 1)
        checkpoint_every_n_epochs = train_params.pop('checkpoint_every_n_epochs', None)
        # Create checkpoint — uses our overridden save() which includes training state
        if use_checkpoint and self._ckpt_dir is not None:
            checkpoint_dir = self._ckpt_dir
            # Start with fresh checkpoint dir since the current one is already loaded
            if clean_tmp and self._from_checkpoint and os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
                import shutil
                log.info(f'Cleaning up temporary checkpoints.')
                shutil.rmtree(checkpoint_dir)
            ckpt_fn = "{epoch}-{step}-{"+checkpoint_monitor+":.4f}"
            checkpoint_callback = SaveCheckPointWithIndices(
                dirpath=checkpoint_dir,
                filename=ckpt_fn,
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                save_top_k=checkpoint_save_top_k,
                every_n_epochs=checkpoint_every_n_epochs,
                load_best_on_end=False,
            )
            log.info(f'Saving model checkpoints to: {checkpoint_dir}')
            callbacks.append(checkpoint_callback)
        else:
            checkpoint_dir = None
        # Add resume callback to restore optimizer/scheduler/epoch state from a loaded checkpoint
        resume_state = getattr(self, '_resume_training_state', None)
        if resume_state is not None and resume_state.get('optimizer_state_dict') is not None:
            from expert.utils._callbacks import ResumeTrainingStateCallback
            callbacks.append(ResumeTrainingStateCallback(resume_state))
            self._resume_training_state = None
            # The f(missing) fill is written when training ENDS (store_prior_missing below); an uninterrupted run
            # never trains with it, but the zero-shot negatives read it when present. Set it aside for the resumed
            # segment -- it is rewritten from the final atoms when this training call ends.
            if 'prior_missing_T' in self.module._buffers:
                del self.module._buffers['prior_missing_T']
                log.info('[Resume] set aside prior_missing_T (rewritten at the end of training)')

        # Optional throughput diagnostic: logs per-batch GPU compute vs
        # pipeline wait (dataloader fetch + H2D + loop overhead) so you can
        # tell whether training is compute-bound or starved for input.
        profile_throughput = train_params.pop('profile_throughput', False)
        profile_warmup = train_params.pop('profile_warmup', 20)
        profile_window = train_params.pop('profile_window', 200)
        if profile_throughput:
            from expert.utils._callbacks import ThroughputProbe
            callbacks.append(ThroughputProbe(warmup=profile_warmup, window=profile_window))

        # Add callbacks to Trainer kwargs
        train_params['callbacks'] = callbacks
        # If callbacks (checkpoint/early-stop) are registered they need validation to have run — but HONOR the
        # config's check_val_every_n_epoch when set (setdefault), only falling back to per-epoch. Previously this
        # UNCONDITIONALLY forced 1, silently overriding check_val_every_n_epoch:10 and running every held-out
        # loader (incl. the zero-shot arms) every epoch — a large, hidden validation-overhead amplification.
        if len(callbacks) > 0:
            train_params.setdefault('check_val_every_n_epoch', 1)
     
        # Single device only (multi-GPU / DDP was removed). ``devices`` defaults to 'auto', which
        # respects the SLURM allocation.
        accelerator = train_params.pop('accelerator', 'auto')
        devices = train_params.pop('devices', 'auto')
        strategy = train_params.pop('strategy', None)
        n_dev = devices if isinstance(devices, int) else (
            len(devices) if isinstance(devices, (list, tuple)) else None
        )
        if isinstance(n_dev, int) and n_dev > 1:
            raise ValueError(f'devices={devices!r}: multi-GPU training is no longer supported; use devices: 1.')
        if strategy is None:
            strategy = 'auto'

        # Create training runner
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            enable_checkpointing=use_checkpoint,
            default_root_dir=checkpoint_dir,
            **train_params
        )
        # Save hyperparameters to model output dir
        if logger is not None:
            # save hyper-parameters to lightning logs
            hparams = {
                'data_params': data_params,
                'model_params': model_params,
                'plan_params': plan_kwargs,
                'train_params': train_params
            }
            # Clean up hparams before logging
            hparams = io.sanitize_for_logging(hparams)
            runner.trainer.logger.log_hyperparams(hparams)
        # Save training plan to model
        self.training_plan = training_plan
        # Print model summary
        if verbose:
            self.__repr__()
        # Save runner in model
        self.last_runner = runner
        # Set module to train mode
        self.module.train()
        # Train model
        self._fit_end_rng = None          # checkpoints written DURING fitting save the live streams
        runner()
        self._fit_end_rng = self._rng_snapshot()
        # HEAD TRAINING RECORD -> a saved model attribute (trailing underscore = persisted in attr_dict), so
        # predict(head='auto') resolves from what training DID rather than from use_cls_head. Accumulates
        # across resumed runs.
        _prev = getattr(self, 'cls_head_training_', None) or {}
        self.cls_head_training_ = {
            'steps': int(_prev.get('steps', 0)) + int(getattr(self.module, '_cls_head_steps', 0)),
            'weight_max': max(float(_prev.get('weight_max', 0.0)), float(getattr(self.module, '_cls_head_w_max', 0.0))),
        }
        self.module._cls_head_steps = 0
        # PRESERVE f(missing) so predict(obs_only=False) scores untrained rows against the graph fill instead of
        # the random init. The last refit's copy is what the PA negatives trained against; arms without
        # negatives never predicted it, so refit once on the final atoms.
        if getattr(self.module, '_prior_atom_fitter', None) is not None:
            try:
                if not self.module.store_prior_missing():
                    self.module.rebuild_prior_missing()
            except Exception as e:                                # noqa: BLE001
                log.warning(f'[Train] could not store f(missing) ({e!r}); add it later with add_prior_missing_fill()')
        log.info(f"[Train] free classifier head record: {self.cls_head_training_} "
                 f"(use_cls_head={getattr(self.module, 'use_cls_head', False)})")
        # Auto-save after training (includes training state for resume)
        if self.model_log_dir is not None:
            save_dir = os.path.join(self.model_log_dir, self._FINAL_MODEL_DNAME)
            self.save(save_dir, overwrite=True)
        # Add latent representation to model.
        # COST-GATED. `_register_latent_variables` runs the encoder over EVERY cell and
        # accumulates four full-size lists (zs, zs_shared, qz_means, qz_vars), then concatenates
        # them -- so it peaks at ~2x n_obs * n_latent * 4B * 4. It sits OUTSIDE the `--eval` guard,
        # so it runs on every training job.
        #
        # That was invisible at n_latent=128 and is fatal at 1024. Measured, on the 19.5M-cell
        # full DB:
        #     n_latent  128 ->  ~40 GB accumulate   (historical, worked)
        #     n_latent 1024 -> ~320 GB accumulate, ~640 GB at the cat  (OOM-killed at a 300 GB
        #                      cap; GPU nodes top out at ~503 GB, so no node can run it)
        # The 64 GB gate is bracketed by those two numbers rather than guessed: above the
        # known-working case, below the known-fatal one. Every existing arm is far under it
        # (small runs are ~1.3 GB) and so is bit-identical.
        #
        # Nothing is lost by skipping: latents are recomputed on demand from a checkpoint via
        # predict()/get_latent_representation(), which is bit-reproducible.
        _nl = int(getattr(self.module, 'n_latent', 0) or 0)
        _peak_gb = (self.adata.n_obs * _nl * 4 * 4) / 1024 ** 3
        if _peak_gb <= 64.0:
            self._register_latent_variables()
        else:
            log.warning(
                f'[Latent] Skipping post-train latent registration: {self.adata.n_obs:,} cells x '
                f'n_latent {_nl} would accumulate ~{_peak_gb:.0f} GB (~{2 * _peak_gb:.0f} GB at '
                f'the concatenate) and OOM. Recompute from a checkpoint with predict() if needed.'
            )


    def _add_missing_query_registry(self, adata: ad.AnnData) -> None:
        """Populate placeholder values in `adata` for any obs/uns columns the
        model's registry expects but the query is missing.

        Categorical obs placeholders use the first known category from training,
        which is arbitrary but lets the dataloader produce a valid batch index.
        """
        for field, reg in self.adata_manager.registry.get('field_registries', {}).items():
            data_reg  = reg.get('data_registry') or {}
            state_reg = reg.get('state_registry') or {}
            attr_key  = data_reg.get('attr_key', '')
            slot      = data_reg.get('attr_name', '')

            # Joint covariate fields carry a list of source columns
            if field in self._QUERY_JOINT_OBS_DEFAULTS:
                for col in state_reg.get('field_keys', []):
                    if col not in adata.obs:
                        log.info(f'Adding placeholder for missing .obs[{col}]')
                        adata.obs[col] = self._QUERY_JOINT_OBS_DEFAULTS[field]
                continue

            # Single-column numeric obs fields with a known sensible default
            if field in self._QUERY_NUMERIC_OBS_DEFAULTS:
                if attr_key and attr_key not in adata.obs:
                    log.info(f'Adding placeholder for missing .obs[{attr_key}]')
                    adata.obs[attr_key] = self._QUERY_NUMERIC_OBS_DEFAULTS[field]
                continue

            # Single-column obs fields (categorical or numeric)
            if slot == 'obs':
                key = state_reg.get('original_key') or attr_key
                if not key or key in adata.obs:
                    continue
                cat_map = state_reg.get('categorical_mapping')
                # Categorical (batch_key, labels_key, etc.) — we have to use
                # the first known training category as the obs value because
                # scvi's transfer-fields auto-extends ``cat_map`` for any
                # unseen strings, which then produces out-of-range codes that
                # crash the model's batch-embedding lookup at forward and the
                # label-mapping at predict aggregation. So the obs column
                # itself still carries ``cat_map[0]`` (and gets coded as int
                # 0), but we write a sidecar boolean ``_placeholder_<key>``
                # alongside so downstream code can unambiguously filter or
                # color these placeholder cells. This is the best we can do
                # without modifying scvi's transfer-setup machinery.
                if cat_map is not None and len(cat_map) > 0:
                    log.warning(
                        f'Adding placeholder for missing .obs[{key!r}]: using '
                        f'{cat_map[0]!r} (first known training category) so '
                        f'the dataloader produces a valid integer index. '
                        f"Filter by .obs['_placeholder_{key}'] == True to "
                        f'identify these cells; do not use .obs[{key!r}] as '
                        f'ground truth for them.'
                    )
                    adata.obs[key] = cat_map[0]
                    adata.obs[f'_placeholder_{key}'] = True
                # scvi-tools library size: register expects per-cell sum of X
                elif attr_key == REGISTRY_KEYS.OBSERVED_LIB_SIZE:
                    log.info(f'Computing .obs[{key}] from query .X (library size)')
                    x_sum = np.asarray(adata.X.sum(axis=1)).flatten()
                    adata.obs[key] = x_sum.astype(np.float32)
                # Generic numeric obs — NaN placeholder
                else:
                    log.warning(f'Adding NaN placeholder for missing numeric .obs[{key}]')
                    adata.obs[key] = np.nan
                continue

            # Uns fields (embeddings, etc.) — None signals "use the model's stored copy"
            if slot == 'uns' and attr_key and attr_key not in adata.uns:
                log.info(f'Adding empty placeholder for missing .uns[{attr_key}]')
                adata.uns[attr_key] = None
                continue

            # Obsm fields — typically cached latents from a minified training
            # adata. Field validation requires the array to exist; for a fresh
            # query, content doesn't matter because non-minified inference
            # recomputes everything. Fill with zeros of the right width.
            if slot == 'obsm' and attr_key and attr_key not in adata.obsm:
                cols = state_reg.get('column_names')
                width = len(cols) if cols is not None else None
                if width is None:
                    log.warning(f'[Setup] Cannot infer width for missing .obsm[{attr_key}]; skipping placeholder.')
                    continue
                log.info(f'Adding zero placeholder for missing .obsm[{attr_key}] (shape: ({adata.n_obs}, {width}))')
                adata.obsm[attr_key] = np.zeros((adata.n_obs, width), dtype=np.float32)
                continue

            # Varm fields — per-gene metadata (e.g. feature_mask_per_dataset in
            # older registries). Validation requires the key to exist. We
            # compute the actual gene support from adata.X (gene is active iff
            # it has any nonzero count in the query) and broadcast across all
            # dataset columns; the model never reads it -- this just satisfies
            # scvi validation with an honest per-gene support summary.
            if slot == 'varm' and attr_key and attr_key not in adata.varm:
                cols = state_reg.get('column_names')
                width = len(cols) if cols is not None else None
                if width is None:
                    log.warning(f'[Setup] Cannot infer width for missing .varm[{attr_key}]; skipping placeholder.')
                    continue
                X = adata.X
                try:
                    col_sums = np.asarray(X.sum(axis=0)).flatten()
                except (AttributeError, TypeError):
                    # BACKED QUERY. anndata's on-disk `_CSRDataset`/`_CSCDataset` has no .sum(),
                    # so a backed h5ad with a missing .varm key crashed predict() outright --
                    # which is every external screen loaded with --backed. Accumulate in row
                    # chunks instead; the result is identical, just streamed.
                    col_sums = np.zeros(adata.n_vars, dtype=np.float64)
                    _step = 20000
                    for _i in range(0, adata.n_obs, _step):
                        _blk = X[_i:_i + _step]
                        _blk = _blk.toarray() if hasattr(_blk, 'toarray') else np.asarray(_blk)
                        col_sums += np.asarray(_blk).sum(axis=0).ravel()
                mask_1d = (col_sums > 0).astype(np.float32)
                log.info(
                    f'Computing gene-support placeholder for missing .varm[{attr_key}] '
                    f'(shape: ({adata.n_vars}, {width}), '
                    f'{int(mask_1d.sum())} / {adata.n_vars} genes active in query).'
                )
                adata.varm[attr_key] = np.tile(mask_1d[:, None], (1, width))
                continue


    def _validate_anndata(
        self, adata: AnnOrMuData | None = None, copy_if_view: bool = True, extend_categories: bool = True, add_missing_categories: bool = False,
    ) -> AnnData:
        """Validate anndata has been properly registered, transfer if necessary."""
        if adata is None:
            adata = self.adata

        _check_if_view(adata, copy_if_view=copy_if_view)
        # TODO: Fix bug that model can't find the manager for this class
        # AssertionError: Unable to find instance specific manager store. The model has likely not been initialized with an AnnData object.
        # cls._per_instance_manager_store is empty
        adata_manager = self.get_anndata_manager(adata)
        if add_missing_categories:
                self._add_missing_query_registry(adata)
        if adata_manager is None:
            log.info(
                "Input AnnData not setup with scvi-tools. "
                + "attempting to transfer AnnData setup"
            )
            self._register_manager_for_instance(self.adata_manager.transfer_fields(adata, extend_categories=extend_categories))
        else:
            # Case where correct AnnDataManager is found, replay registration as necessary.
            adata_manager.validate()
        return adata
    
    @property
    def eval_adata(self) -> ad.AnnData:
        """AnnData used for storing / reading eval results.

        Defaults to ``self.adata``. ``evaluate()`` sets ``self._eval_adata``
        to a lightweight in-memory AnnData (just obs/var/uns of the kept
        cells, no X) when ``skip_splits`` is non-empty — keeping the
        original (potentially backed) ``self.adata`` untouched while
        still storing eval outputs in obsm/obs at the right size.
        """
        return getattr(self, '_eval_adata', None) or self.adata

    def _get_available_splits(self) -> dict[str, np.ndarray]:
        """Get list of available data splits in the model.
        
        Returns
        -------
        dict[str, np.ndarray]
            List of available split names ('train', 'val', 'test')
        """
        return {
            split: indices for split, indices in {
                'train': getattr(self, 'train_indices', None),
                'val': getattr(self, 'validation_indices', None), 
                'test': getattr(self, 'test_indices', None),
                'ood_test': getattr(self, 'ood_test_indices', None)
            }.items() if indices is not None and len(indices) > 0
        }
    
    def _get_split_indices(self, split: str) -> np.ndarray:
        """Get model split indices by split key (train, val/validation, test)"""
        avail_splits = self._get_available_splits()
        if split not in avail_splits:
            raise ValueError(f'Model does not have indices for split: {split}, has to be one of {self._get_available_splits()}')
        return avail_splits.get(split)
        
    def _get_split_adata(self, split: str, ignore_ctrl: bool = True) -> ad.AnnData:
        """Get AnnData subset for a specific split and optionally filter control cells.

        Reads from ``self.eval_adata`` so eval flows operating on a subset
        (when ``skip_splits`` was set) get the right cells. When
        ``eval_adata.obs`` carries the SPLIT_KEY column we filter by that
        (works regardless of whether eval_adata == self.adata or a smaller
        subset); otherwise we fall back to positional split indices on
        ``self.adata``.
        """
        ad_obj = self.eval_adata
        if REGISTRY_KEYS.SPLIT_KEY in ad_obj.obs:
            adata = ad_obj[ad_obj.obs[REGISTRY_KEYS.SPLIT_KEY] == split]
        else:
            adata = self.adata[self._get_split_indices(split)]
        if self.ctrl_class is not None and ignore_ctrl:
            adata = adata[adata.obs[self.original_label_key] != self.ctrl_class]
        return adata
    
    def _get_model_classification_report(
        self, split: str, ignore_ctrl: bool = True, responder_threshold: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.metrics import classification_report, precision_recall_fscore_support

        # Check if internal predictions are available
        if PREDICTION_KEYS.PREDICTION_KEY not in self.eval_adata.obs:
            raise ValueError('Run self.evaluate() before calculating statistics.')
        # Get mode adata
        mode_adata = self._get_split_adata(split, ignore_ctrl=ignore_ctrl)
        y_true = mode_adata.obs[self.original_label_key]
        y_pred = mode_adata.obs[PREDICTION_KEYS.PREDICTION_KEY]
        # Generate a classification report for the relevant mode
        report = classification_report(
            y_true,
            y_pred,
            zero_division=np.nan,
            output_dict=True
        )
        # Format output accordingly
        report_df = pd.DataFrame(report).transpose()
        summary = report_df[report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        report_data = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        report_data['log_count'] = np.log(report_data['support'])

        # Efficiency-weighted F1 columns (true weights + predicted eff_hat), if available
        eff_state_registry = self.adata_manager.registry.get('field_registries', {}).get(REGISTRY_KEYS.CLS_EFF_KEY, {})
        eff_obs_key = eff_state_registry.get('data_registry', {}).get('attr_key', None)
        weight_sources = {}
        if eff_obs_key is not None and eff_obs_key in mode_adata.obs:
            w_true = np.abs(mode_adata.obs[eff_obs_key].values.astype(float))
            weight_sources['f1-score_eff_weighted'] = w_true
            # Responder-only mask: cells with |w| > threshold (default 1 std from control)
            weight_sources[f'f1-score_responder_only'] = (w_true > responder_threshold).astype(float)
        if 'eff_hat' in mode_adata.obs:
            w_pred = np.abs(mode_adata.obs['eff_hat'].values.astype(float))
            weight_sources['f1-score_eff_hat_weighted'] = w_pred
            weight_sources[f'f1-score_responder_only_pred'] = (w_pred > responder_threshold).astype(float)
        for col, w in weight_sources.items():
            class_labels = report_data.index.values
            _, _, f1_w, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=class_labels, average=None,
                sample_weight=w, zero_division=np.nan,
            )
            report_data[col] = f1_w
            # Macro + weighted-by-support averages into summary (NaN → skipped)
            _, _, macro_f1_w, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', sample_weight=w, zero_division=np.nan,
            )
            _, _, wavg_f1_w, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', sample_weight=w, zero_division=np.nan,
            )
            summary.loc['macro avg', col] = macro_f1_w
            summary.loc['weighted avg', col] = wavg_f1_w

        # Add mode to results
        report_data[REGISTRY_KEYS.SPLIT_KEY] = split
        summary[REGISTRY_KEYS.SPLIT_KEY] = split
        return summary, report_data
    
    def _prediction_output_from_adata(self, adata: AnnData) -> dict:
        """Reconstruct the dict view of predict()'s output from an adata
        that was written to by ``predict(..., inplace=True)``.

        Pulls ground-truth labels from the registered label column and the
        per-cell predictions / extras from the obs/obsm/uns slots the
        inplace path wrote. Missing optional fields (no labels at query
        time, no zero-shot info if cls_emb wasn't expanded) just stay
        absent — downstream code already handles those via .get(...).
        """
        out: dict = {}
        # Ground-truth labels — only present when the query was annotated.
        if self.original_label_key in adata.obs:
            out[REGISTRY_KEYS.LABELS_KEY] = np.asarray(adata.obs[self.original_label_key])
        else:
            raise KeyError(
                f"Cannot build a classification report from this adata: no "
                f"obs[{self.original_label_key!r}] column with ground-truth "
                f"labels. Either annotate the query before predict() or pass "
                f"the dict returned by predict(inplace=False) instead."
            )
        # Predicted labels — always written by inplace predict()
        if PREDICTION_KEYS.PREDICTION_KEY not in adata.obs:
            raise KeyError(
                f"adata missing obs[{PREDICTION_KEYS.PREDICTION_KEY!r}] — "
                f"did predict(inplace=True) actually run on this object?"
            )
        out[PREDICTION_KEYS.PREDICTION_KEY] = np.asarray(adata.obs[PREDICTION_KEYS.PREDICTION_KEY])
        # Optional extras — only included when present so .get() returns None.
        if 'cls_zero_shot_info' in adata.uns:
            out['cls_zero_shot_info'] = adata.uns['cls_zero_shot_info']
        # Raw cls embedding + labels (for soft-quality + adapter-fidelity
        # metrics). Stored on the inplace adata's uns by predict() when
        # the user passed an expanded cls_emb.
        if 'cls_emb_raw' in adata.uns:
            out['cls_emb_raw'] = adata.uns['cls_emb_raw']
        if 'cls_labels' in adata.uns:
            out['cls_labels'] = adata.uns['cls_labels']
        # Projected cls embedding (post-aligner): from latent_cls_proj if
        # return_latents=True was used during inplace predict().
        if self._LATENT_CLS_PROJ_KEY in adata.uns:
            out[MODULE_KEYS.CLS_PROJ_KEY] = adata.uns[self._LATENT_CLS_PROJ_KEY]
        return out

    def get_prediction_output_report(self, prediction_output: dict | AnnData):
        from sklearn.metrics import classification_report

        # Accept both shapes: predict() dict OR an adata that was written
        # by predict(inplace=True). Adata input is reconstructed into the
        # same dict shape via the obs/uns slots written by the inplace path.
        if isinstance(prediction_output, AnnData):
            prediction_output = self._prediction_output_from_adata(prediction_output)

        # Generate a classification report for the relevant mode
        report = classification_report(
            y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
            y_pred=prediction_output[PREDICTION_KEYS.PREDICTION_KEY],
            zero_division=np.nan,
            output_dict=True
        )
        # Format output accordingly
        report_df = pd.DataFrame(report).transpose()
        summary = report_df[report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        report_data = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
        # Filter for observed classes
        report_data = report_data[report_data.support > 0].copy()
        report_data['log_count'] = np.log(report_data.support)
        # Attach threshold-free soft F1 on efficiency if available
        eff_soft_f1 = prediction_output.get('eff_soft_f1', None)
        if eff_soft_f1 is not None:
            summary.loc['eff_soft_f1', 'f1-score'] = eff_soft_f1
        # Zero-shot split: when predict() was called with an expanded cls_emb
        # (so some target labels are outside the trained class set), surface
        # seen-vs-zero-shot metrics + a similarity-binned breakdown. Stored as
        # extra entries on the returned objects so callers that already
        # consume (summary, report_data) keep working unchanged.
        zsi = prediction_output.get('cls_zero_shot_info', None)
        if zsi is not None:
            try:
                from expert.utils.performance import zero_shot_classification_report
                zs_summary, zs_per_class = zero_shot_classification_report(
                    y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
                    y_pred=prediction_output[PREDICTION_KEYS.PREDICTION_KEY],
                    trained_labels=self.get_training_classes(),
                    cls_similarities=zsi,
                )
                # Join the per-class similarity / is_zero_shot onto the
                # existing per-class report so a single DataFrame describes
                # both performance and novelty.
                report_data = report_data.join(
                    zsi[['is_trained', 'similarity_to_trained', 'nearest_trained']],
                    how='left',
                )
                report_data['is_zero_shot'] = ~report_data['is_trained'].fillna(False)
                # Stash zero-shot summary under a separate attribute on the
                # DataFrame so the user can inspect it without breaking the
                # existing (summary, report) return contract.
                summary.attrs['zero_shot_summary'] = zs_summary
                summary.attrs['zero_shot_per_class'] = zs_per_class
            except Exception as e:
                log.warning(f'[Predict] Zero-shot report skipped: {e}')

        # Embedding-space soft prediction quality + adapter-fidelity checks.
        # Strict label-match F1 is brutal at 18K candidates — these metrics
        # tell you whether wrong predictions land near the true class
        # (soft quality) and whether the adapter actually preserves the
        # held-out class geometry it inherits from the raw embedding.
        cls_emb_raw = prediction_output.get('cls_emb_raw', None)
        cls_proj = prediction_output.get(MODULE_KEYS.CLS_PROJ_KEY, None)
        cls_labels = prediction_output.get('cls_labels', None)
        if cls_labels is not None and cls_proj is not None:
            try:
                from expert.utils.performance import embedding_prediction_quality
                # Soft quality in the **projection** space (where CLIP operates).
                # Optionally also computed in raw space if available — gives
                # the model-free semantic baseline as a comparison.
                eq_summary, eq_per_class = embedding_prediction_quality(
                    y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
                    y_pred=prediction_output[PREDICTION_KEYS.PREDICTION_KEY],
                    cls_emb=cls_proj,
                    cls_labels=cls_labels,
                    trained_labels=self.get_training_classes(),
                )
                summary.attrs['emb_quality_proj_summary'] = eq_summary
                summary.attrs['emb_quality_proj_per_class'] = eq_per_class
                # Attach to per-class report. Pull BOTH directions:
                #   _when_true: per-y_true class (limited to classes in y_true)
                #   _when_pred: per-y_pred class (covers predicted-only
                #     classes too — exactly the "model predicted something
                #     novel" case the metric is meant to capture).
                report_data = report_data.join(
                    eq_per_class[['mean_cosine_when_true', 'mean_cosine_when_pred']].rename(
                        columns={
                            'mean_cosine_when_true': 'mean_cosine_proj_when_true',
                            'mean_cosine_when_pred': 'mean_cosine_proj_when_pred',
                        }
                    ),
                    how='left',
                )
                # Back-compat alias: ``mean_cosine_proj`` = per-y_true variant.
                report_data['mean_cosine_proj'] = report_data['mean_cosine_proj_when_true']
                if cls_emb_raw is not None:
                    eq_raw_summary, eq_raw_per_class = embedding_prediction_quality(
                        y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
                        y_pred=prediction_output[PREDICTION_KEYS.PREDICTION_KEY],
                        cls_emb=cls_emb_raw,
                        cls_labels=cls_labels,
                        trained_labels=self.get_training_classes(),
                    )
                    summary.attrs['emb_quality_raw_summary'] = eq_raw_summary
                    summary.attrs['emb_quality_raw_per_class'] = eq_raw_per_class
                    report_data = report_data.join(
                        eq_raw_per_class[['mean_cosine_when_true', 'mean_cosine_when_pred']].rename(
                            columns={
                                'mean_cosine_when_true': 'mean_cosine_raw_when_true',
                                'mean_cosine_when_pred': 'mean_cosine_raw_when_pred',
                            }
                        ),
                        how='left',
                    )
                    report_data['mean_cosine_raw'] = report_data['mean_cosine_raw_when_true']
                    # Per-cell view for the "distribution of soft quality
                    # binned by zero-shot similarity" plot. Each cell is
                    # one data point; cls_similarities (the cls_zero_shot_info
                    # DataFrame) carries the y_true-class similarity-to-trained
                    # so cells can be binned in the plot without re-deriving it.
                    from expert.utils.performance import per_cell_prediction_cosine, random_cosine_baseline
                    summary.attrs['emb_quality_raw_per_cell'] = per_cell_prediction_cosine(
                        y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
                        y_pred=prediction_output[PREDICTION_KEYS.PREDICTION_KEY],
                        cls_emb=cls_emb_raw,
                        cls_labels=cls_labels,
                        trained_labels=self.get_training_classes(),
                        cls_similarities=zsi,
                    )
                    summary.attrs['emb_quality_raw_per_cell_centered'] = per_cell_prediction_cosine(
                        y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
                        y_pred=prediction_output[PREDICTION_KEYS.PREDICTION_KEY],
                        cls_emb=cls_emb_raw,
                        cls_labels=cls_labels,
                        trained_labels=self.get_training_classes(),
                        cls_similarities=zsi,
                        center=True,
                    )
                    # Random-pair baseline: average cosine between two
                    # random class anchors in the same space. We compute
                    # BOTH the raw and mean-centered baselines because
                    # LLM embeddings are typically anisotropic — raw mean
                    # pairwise cosine often sits at 0.5-0.8, making "model
                    # achieves 0.66" deceptive. The centered baseline
                    # strips that DC offset so model lift over random is
                    # visible directly. If the gap between raw baseline
                    # and model cosine is small but the gap between
                    # centered baseline (~0) and centered model cosine
                    # is also small, the embedding space is doing most
                    # of the work, not the model.
                    summary.attrs['emb_quality_raw_random_baseline'] = random_cosine_baseline(
                        cls_emb_raw, n_samples=10_000,
                    )
                    summary.attrs['emb_quality_raw_random_baseline_centered'] = random_cosine_baseline(
                        cls_emb_raw, n_samples=10_000, center=True,
                    )
            except Exception as e:
                log.warning(f'[Predict] Embedding-prediction-quality report skipped: {e}')

        # Top-K recall: per-cell hit indicators ("is true class in top K
        # predictions?") aggregated per-perturbation. Discrete, gene-
        # specific like F1 but tolerant of small reorderings — gives a
        # cleaner diagnostic per zero-shot bin than mean-cosine, which is
        # subject to LLM-space anisotropy. K=5 / 50 / 100 give gene-level
        # / pathway-ish / family-ish resolution at 18K candidates.
        soft_pred = prediction_output.get(PREDICTION_KEYS.SOFT_PREDICTION_KEY, None)
        if soft_pred is not None and cls_labels is not None:
            try:
                from expert.utils.performance import per_cell_topk_recall
                # cls_columns for soft_pred: same as cls_labels but might
                # include null_proxy padding when n_logit_cols > n_cls.
                # For top-K recall we don't care about that — true labels
                # are in cls_labels which is a prefix of cls_columns.
                tk = per_cell_topk_recall(
                    y_true=prediction_output[REGISTRY_KEYS.LABELS_KEY],
                    soft_predictions=soft_pred,
                    cls_columns=cls_labels,
                    ks=(5, 50, 100),
                )
                # Attach per-cell hit columns to the per-cell DataFrame
                # so the existing per-cell plot can bin them by zero-shot
                # similarity.
                if 'emb_quality_raw_per_cell' in summary.attrs:
                    per_cell_df = summary.attrs['emb_quality_raw_per_cell']
                    for col in tk.columns:
                        per_cell_df[col] = tk[col].values
                # Per-perturbation aggregation: recall@K = mean hit across
                # cells of each true class. Each y_true class gets one
                # value per K — the natural input for the boxen plot
                # binned by similarity-to-trained.
                y_true_arr = np.asarray(prediction_output[REGISTRY_KEYS.LABELS_KEY]).astype(str)
                tk_with_true = tk.copy()
                tk_with_true['y_true'] = y_true_arr
                per_class_tk = tk_with_true.groupby('y_true').mean(numeric_only=True)
                per_class_tk = per_class_tk.rename(columns={
                    c: c.replace('hit@', 'recall@') for c in per_class_tk.columns
                })
                # Join onto the per-class classification report
                report_data = report_data.join(per_class_tk, how='left')
                summary.attrs['topk_recall_per_class'] = per_class_tk
                # Bin-level summary (cell-weighted, ignoring class identity)
                # — convenient for at-a-glance reporting.
                cell_summary = {f'mean_recall_at_{k}_overall': float(np.nanmean(tk[f'hit@{k}']))
                                for k in (5, 50, 100)}
                summary.attrs['topk_recall_summary'] = cell_summary
            except Exception as e:
                log.warning(f'[Predict] Top-K recall report skipped: {e}')

        if cls_emb_raw is not None and cls_proj is not None and cls_labels is not None:
            try:
                from expert.utils.performance import adapter_fidelity
                af_summary, af_per_class = adapter_fidelity(
                    raw_emb=cls_emb_raw,
                    proj_emb=cls_proj,
                    cls_labels=cls_labels,
                    trained_labels=self.get_training_classes(),
                )
                summary.attrs['adapter_fidelity_summary'] = af_summary
                summary.attrs['adapter_fidelity_per_class'] = af_per_class
                # Join per-class adapter-fidelity stats onto the per-class
                # report for the held-out (zero-shot) rows. Trained classes
                # get NaN for these columns (they're not in af_per_class).
                report_data = report_data.join(
                    af_per_class[['pearson_r', 'proj_minus_raw', 'nearest_agree']].rename(
                        columns={
                            'pearson_r': 'adapter_pearson_r',
                            'proj_minus_raw': 'adapter_proj_minus_raw',
                            'nearest_agree': 'adapter_nearest_agree',
                        }
                    ),
                    how='left',
                )
            except Exception as e:
                log.warning(f'[Predict] Adapter-fidelity report skipped: {e}')
        return summary, report_data
    
    def _register_classification_report(
        self,
        force: bool = False,
        ignore_ctrl: bool = True,
        responder_threshold: float = 1.0,
        skip_splits: list[str] | None = None,
    ) -> None:
        """Run classification report for all registered modes."""
        # Check if already given
        if self.is_evaluated and not force:
            return
        skip = set(skip_splits or [])
        # Generate classification summaries and reports
        summaries, reports = [], []
        for split in self._get_available_splits():
            if split in skip:
                continue
            summary, report = self._get_model_classification_report(
                split, ignore_ctrl=ignore_ctrl, responder_threshold=responder_threshold,
            )
            summaries.append(summary)
            reports.append(report)
        summaries = pd.concat(summaries, axis=0) if summaries else pd.DataFrame()
        reports = pd.concat(reports, axis=0).reset_index(names=['cls_label']) if reports else pd.DataFrame()
        # Reports go on eval_adata so they reflect the cells we actually
        # evaluated (defaults to self.adata when no subset is in play).
        self.eval_adata.uns[PREDICTION_KEYS.SUMMARY_KEY] = summaries
        self.eval_adata.uns[PREDICTION_KEYS.REPORT_KEY] = reports

    def _register_top_n_predictions(self) -> None:
        """Calculate performance metrics for each data split and optionally also for each context"""
        top_n_predictions = pf.compute_top_n_predictions(
            adata=self.eval_adata,
            split_key=REGISTRY_KEYS.SPLIT_KEY,
            context_key=self.original_batch_key,
            labels_key=self.original_label_key,
            ctrl_key=self.ctrl_class,
            predictions_key=PREDICTION_KEYS.SOFT_PREDICTION_KEY,
        )
        # Add data to eval_adata's uns (defaults to self.adata)
        self.eval_adata.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY] = top_n_predictions

    def _get_top_n_predictions(self) -> pd.DataFrame:
        if PREDICTION_KEYS.TOP_N_PREDICTION_KEY not in self.eval_adata.obsm:
            raise KeyError('No model evaluation top prediction metrics available. Run model.evaluate()')
        return self.adata.obsm[PREDICTION_KEYS.TOP_N_PREDICTION_KEY]

    # ------------------------------------------------------------------
    # Token-level interpretability (SapBERT pathway retrieval)
    # ------------------------------------------------------------------

    def _register_latent_variables(
        self,
        force: bool = True,
        indices: np.ndarray | None = None,
        progress_bar: bool = True,
        batch_size: int = 1024,
    ) -> None:
        """Register latent space with self.adata.obsm.

        When ``indices`` is given, only those cells are run through the
        encoder; cells not in ``indices`` get NaN rows in qzm / qzv. Used
        by ``evaluate(skip_splits=...)`` to avoid forward passes over
        skipped (e.g. train) cells.
        """
        # Check if it is already present, recalculate if forced
        if self._LATENT_QZM_KEY in self.adata.obsm and not force:
            return
        # Run inference and assign it to model adata
        latent_out = self.get_latent_representation(
            indices=indices, return_dist=True, progress_bar=progress_bar, batch_size=batch_size,
        )
        qzm = latent_out[MODULE_KEYS.QZM_KEY]
        qzv = latent_out[MODULE_KEYS.QZV_KEY]
        if indices is None:
            self.adata.obsm[self._LATENT_QZM_KEY] = qzm
            self.adata.obsm[self._LATENT_QZV_KEY] = qzv
        else:
            # Scatter into NaN-filled full-size arrays so downstream code
            # that indexes by absolute cell id still works.
            n = self.adata.n_obs
            full_qzm = np.full((n, qzm.shape[1]), np.nan, dtype=qzm.dtype)
            full_qzv = np.full((n, qzv.shape[1]), np.nan, dtype=qzv.dtype)
            full_qzm[indices] = qzm
            full_qzv[indices] = qzv
            self.adata.obsm[self._LATENT_QZM_KEY] = full_qzm
            self.adata.obsm[self._LATENT_QZV_KEY] = full_qzv

    def _register_split_labels(
        self, force: bool = True, skip_splits: list[str] | None = None,
    ) -> None:
        """Add split label to eval_adata.obs (defaults to self.adata).

        Cells in ``skip_splits`` keep a None label so iterators that group
        by split (classification report, top-N, plotting) skip them
        naturally rather than seeing rows with NaN predictions. When
        ``eval_adata`` is a subset built by ``evaluate()``, it already
        carries the SPLIT_KEY column from construction — this method
        becomes a no-op in that case unless ``force=True``.
        """
        ad_obj = self.eval_adata
        # Skip if already populated and not forced
        if REGISTRY_KEYS.SPLIT_KEY in ad_obj.obs and not force:
            return
        skip = set(skip_splits or [])
        # If eval_adata is a different object (subset), the model's
        # split_indices arrays still point into self.adata coords; we can't
        # use them directly. The eval_adata should already have SPLIT_KEY
        # pre-populated by the caller (evaluate()) in that case.
        if ad_obj is not self.adata:
            return
        # Create array of split labels matching adata size
        split_labels = np.full(ad_obj.n_obs, None, dtype=object)
        # Fill in split labels based on indices
        for split in self._get_available_splits():
            if split in skip:
                continue
            split_indices = self._get_split_indices(split)
            split_labels[split_indices] = split
        # Add split labels to adata.obs
        ad_obj.obs[REGISTRY_KEYS.SPLIT_KEY] = pd.Categorical(split_labels)
    
    def _register_model_predictions(
        self,
        force: bool = True,
        use_z: bool = False,
        indices: np.ndarray | None = None,
        skip_splits: list[str] | None = None,
        calc_top_n: bool = False,
        **kwargs,
    ) -> None:
        """Add detailed model classification outputs to internal self.adata object.

        ``indices`` selects rows of self.adata to run the forward pass on.
        Results are stored on ``self.eval_adata`` (defaults to
        ``self.adata``). When ``self.eval_adata`` is a smaller subset
        built by ``evaluate(skip_splits=...)``, ``len(indices)`` should
        match ``eval_adata.n_obs`` so results can be written directly. If
        instead ``eval_adata is self.adata`` and ``indices`` is a subset,
        results are scattered back with NaN for the missing rows.
        """
        # Check if model has been trained
        self._check_if_trained(warn=False)
        # Storage target — eval_adata defaults to self.adata, or is a
        # smaller in-memory subset built by evaluate(skip_splits=...).
        dest = self.eval_adata
        # Predictions have not yet been made
        if PREDICTION_KEYS.PREDICTION_KEY in dest.obs and not force:
            return
        # Add split information to dest (no-op when eval_adata is a
        # pre-populated subset).
        self._register_split_labels(force=force, skip_splits=skip_splits)
        # Forward pass on self.adata (the source of X) at `indices`. predict()
        # defaults adata=None to self.adata.
        po = self.predict(return_latents=True, indices=indices, **kwargs)

        n = dest.n_obs
        # Scatter only when storing back into self.adata at a subset of
        # rows (legacy NaN-padded path). When dest is a separate eval_adata
        # the predict output is sized 1:1 with dest, write directly.
        needs_scatter = (
            dest is self.adata
            and indices is not None
            and len(indices) < n
        )
        idx = np.asarray(indices) if needs_scatter else None

        def _scatter(arr):
            """Place ``arr`` (sized to len(indices)) into a full-dest-size
            array with NaN/None for cells not in ``indices``. No-op when
            ``indices`` is None — direct write."""
            if arr is None:
                return None
            if idx is None:
                return arr
            arr = np.asarray(arr)
            if arr.ndim == 1:
                if np.issubdtype(arr.dtype, np.floating):
                    full = np.full(n, np.nan, dtype=arr.dtype)
                else:
                    full = np.full(n, None, dtype=object)
                full[idx] = arr
            else:
                fill = np.nan if np.issubdtype(arr.dtype, np.floating) else 0
                full = np.full((n,) + arr.shape[1:], fill, dtype=arr.dtype)
                full[idx] = arr
            return full

        # Save context predictions
        dest.obs[PREDICTION_KEYS.CTX_PREDICTION_KEY] = _scatter(po[PREDICTION_KEYS.CTX_PREDICTION_KEY])
        if PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY in po and po[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY] is not None:
            dest.obsm[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY] = _scatter(po[PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY])
        # Save regular classification predictions to model adata
        if use_z:
            # Use classifier predictions
            predictions = po[PREDICTION_KEYS.Z_PREDICTION_KEY]
            soft_predictions = po[PREDICTION_KEYS.Z_SOFT_PREDICTION_KEY]
        else:
            # Use aligner predictions
            predictions = po[PREDICTION_KEYS.PREDICTION_KEY]
            soft_predictions = po[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
        dest.obs[PREDICTION_KEYS.PREDICTION_KEY] = _scatter(predictions)
        dest.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = _scatter(soft_predictions)
        # Register efficiency scores
        if po.get('response_probs', []):
            dest.obs['response_probs'] = _scatter(po['response_probs'])
        # Register predicted efficiency and threshold-free soft F1 if available
        if po.get('eff_hat', None) is not None:
            dest.obs['eff_hat'] = _scatter(po['eff_hat'])
        if po.get(MODULE_KEYS.CLS_EFF_KEY, None) is not None:
            dest.obs[MODULE_KEYS.CLS_EFF_KEY] = _scatter(po[MODULE_KEYS.CLS_EFF_KEY])
        # Per-dataset scaled views — these match the loss's training axis,
        # so this is the pair to read the eff_hat-vs-eff diagonal against.
        if po.get('eff_hat_scaled', None) is not None:
            dest.obs['eff_hat_scaled'] = _scatter(po['eff_hat_scaled'])
        if po.get('eff_scaled', None) is not None:
            dest.obs['eff_scaled'] = _scatter(po['eff_scaled'])
        # Register aligned latent spaces with model's adata
        dest.obsm[self._LATENT_Z_KEY] = _scatter(po[MODULE_KEYS.Z_KEY])
        dest.obsm[self._LATENT_Z_SHARED_KEY] = _scatter(po[MODULE_KEYS.Z_SHARED_KEY])
        # Register the latent distribution params captured during the same
        # forward pass (replaces the separate _register_latent_variables call).
        if po.get(MODULE_KEYS.QZM_KEY) is not None and po.get(MODULE_KEYS.QZV_KEY) is not None:
            dest.obsm[self._LATENT_QZM_KEY] = _scatter(po[MODULE_KEYS.QZM_KEY])
            dest.obsm[self._LATENT_QZV_KEY] = _scatter(po[MODULE_KEYS.QZV_KEY])
        if MODULE_KEYS.CTX_PROJ_KEY in po and po[MODULE_KEYS.CTX_PROJ_KEY] is not None:
            dest.uns[self._LATENT_CTX_PROJ_KEY] = po[MODULE_KEYS.CTX_PROJ_KEY]
        dest.uns[self._LATENT_CLS_PROJ_KEY] = po[MODULE_KEYS.CLS_PROJ_KEY]
        # Calculate top N predictions over model data
        if calc_top_n:
            self._register_top_n_predictions()
        # Set evaluation as completed
        self.is_evaluated = True

    def _save_eval_logits(self, output_dir: str | None = None, skip_splits: list[str] | None = None):
        """Save logits from eval split cells.

        Per split, writes one HDF5 file per prediction matrix:
          - ``<output_dir>/<split>/logits.h5`` — class soft predictions
          - ``<output_dir>/<split>/ctx_logits.h5`` — context soft predictions
            (only when a context prediction matrix is registered)

        Each file has the same flat top-level layout:
          - ``/logits`` — 2-d float dataset, dense. CSR top-K sources are
            densified at write time with ``-inf`` fill on non-active cells
            (matches the dense predict-path) so ``np.argmax(axis=1)`` works
            correctly even when stored top-K logits are negative.
          - ``/columns`` — per-column labels (only if the source was a
            DataFrame with labelled columns).
          - ``/obs_names`` — cell barcodes for each row in ``/logits``,
            suitable for joining against any other AnnData by barcode
            rather than positional index.

        Top-level attrs: ``format_version``, ``shape``. Skip if
        ``output_dir`` is None.
        """
        if output_dir is None:
            return
        import h5py

        # Read from eval_adata (defaults to self.adata) so we operate on the
        # same data that was registered with predictions.
        ad_obj = self.eval_adata
        cls_logits = ad_obj.obsm.get(PREDICTION_KEYS.SOFT_PREDICTION_KEY, None)
        ctx_logits = ad_obj.obsm.get(PREDICTION_KEYS.CTX_SOFT_PREDICTION_KEY, None)
        if cls_logits is None and ctx_logits is None:
            log.warning('[Eval] No soft predictions registered; skipping logit save.')
            return

        skip = set(skip_splits or [])
        str_dt = h5py.string_dtype('utf-8')

        def _dense_and_columns(arr, idx):
            """Returns (dense ``np.ndarray``, columns_or_None) for the row
            subset. Top-K CSR sources are densified with ``-inf`` fill so
            downstream argmax sees the non-active classes as strictly worse
            than any stored top-K value, matching the dense predict-path."""
            if hasattr(arr, 'iloc'):  # pandas DataFrame — already dense + -inf filled
                sub = arr.iloc[idx]
                return np.asarray(sub.values), np.asarray(sub.columns)
            if sp.issparse(arr):
                sub = arr[idx].tocoo()
                dense = np.full(sub.shape, -np.inf, dtype=np.float32)
                dense[sub.row, sub.col] = sub.data.astype(np.float32, copy=False)
                return dense, None
            return np.asarray(arr[idx]), None

        def _write_logits_file(path, values, columns, names):
            """Write ``/logits`` (2-d dense), optional ``/columns``, and
            ``/obs_names`` (cell barcodes), all at the file's top level."""
            with h5py.File(path, 'w') as f:
                f.attrs['format_version'] = '2'
                f.attrs['shape'] = np.asarray(values.shape, dtype=np.int64)
                f.create_dataset(
                    'logits', data=values,
                    compression='gzip', compression_opts=4,
                )
                if columns is not None:
                    f.create_dataset(
                        'columns', data=np.asarray(columns, dtype=str_dt),
                        compression='gzip', compression_opts=4,
                    )
                f.create_dataset(
                    'obs_names', data=np.asarray(names, dtype=str_dt),
                    compression='gzip', compression_opts=4,
                )

        # Filter splits via the SPLIT_KEY column (works whether eval_adata
        # is the full self.adata or a smaller subset).
        if REGISTRY_KEYS.SPLIT_KEY in ad_obj.obs:
            split_col = ad_obj.obs[REGISTRY_KEYS.SPLIT_KEY]
            for split in pd.unique(split_col.dropna()):
                if split in skip:
                    continue
                split_idx = np.where((split_col == split).values)[0]
                split_names = np.asarray(ad_obj.obs_names[split_idx])
                split_dir = os.path.join(output_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                if cls_logits is not None:
                    vals, cols = _dense_and_columns(cls_logits, split_idx)
                    _write_logits_file(
                        os.path.join(split_dir, 'logits.h5'),
                        vals, cols, split_names,
                    )
                if ctx_logits is not None:
                    vals, cols = _dense_and_columns(ctx_logits, split_idx)
                    _write_logits_file(
                        os.path.join(split_dir, 'ctx_logits.h5'),
                        vals, cols, split_names,
                    )
        log.info(f'[Eval] Saved per-split logits (h5) under: {output_dir}')


    def knn_rerank(
        self,
        k: int = 50,
        alpha: float = 0.5,
        metric: Literal['cosine', 'euclidean'] = 'cosine',
        z_key: str | None = None,
        splits: list[str] | None = None,
        knn_temperature: float = 0.1,
        update_predictions: bool = True,
        force: bool = True,
    ) -> pd.DataFrame:
        """k-NN reranking on the shared latent space.

        For each cell in ``splits`` (default: every non-train split), find
        the top-k training cells by ``metric`` in ``z_key`` (default
        ``z_shared``), build a class-vote distribution from neighbor true
        labels weighted by similarity, and ensemble with the model's soft
        predictions: ``alpha * model_softmax + (1 - alpha) * knn_softmax``.

        Useful when ``recall@k`` is high but top-1 accuracy is low — the
        model's representations are good but its prototype/head selection
        is noisy. kNN over labeled train neighbors fixes the selection.

        Updates ``adata.obs[PREDICTION_KEY]`` (argmax of blended) and
        rewrites soft predictions for the queried cells in-place.
        Re-runs ``_register_classification_report`` so summaries reflect
        the rerank.
        """
        from sklearn.neighbors import NearestNeighbors

        if z_key is None:
            z_key = self._LATENT_Z_SHARED_KEY
        if z_key not in self.adata.obsm:
            raise ValueError(f'Run evaluate() first; missing {z_key} in adata.obsm.')
        if PREDICTION_KEYS.SOFT_PREDICTION_KEY not in self.adata.obsm:
            raise ValueError('Run evaluate() first; missing soft predictions.')

        Z = np.asarray(self.adata.obsm[z_key])
        train_idx = getattr(self, 'train_indices', None)
        if train_idx is None or len(train_idx) == 0:
            raise ValueError('No train_indices available to build kNN index.')
        train_idx = np.asarray(train_idx)

        if splits is None:
            splits = [s for s in self._get_available_splits() if s != 'train']
        if not splits:
            log.warning('[kNN] No non-train splits to rerank.')
            return pd.DataFrame()
        query_idx = np.concatenate([self._get_split_indices(s) for s in splits])

        # Build kNN index over train cells
        log.info(f'[kNN] Building index on {len(train_idx):,} train cells (z={Z.shape[1]}, metric={metric}).')
        nn = NearestNeighbors(
            n_neighbors=k,
            metric='cosine' if metric == 'cosine' else 'minkowski',
            p=2 if metric == 'euclidean' else 2,
            n_jobs=-1,
        )
        nn.fit(Z[train_idx])
        log.info(f'[kNN] Querying {len(query_idx):,} cells with k={k}.')
        dists, neigh_pos = nn.kneighbors(Z[query_idx], n_neighbors=k, return_distance=True)

        # Map class names ↔ columns of soft predictions
        soft_df = self.adata.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
        if not hasattr(soft_df, 'columns'):
            raise ValueError('Soft predictions must be a DataFrame with class-named columns.')
        classes = soft_df.columns.values
        cls_to_col = {c: i for i, c in enumerate(classes)}

        # Train-cell label → column index (drop labels not in current class universe)
        labels_arr = self.adata.obs[self.original_label_key].values
        train_label_cols = np.array([cls_to_col.get(l, -1) for l in labels_arr[train_idx]], dtype=np.int64)

        # Distance → similarity weights, softmax-normalized per query
        sims = (1.0 - dists) if metric == 'cosine' else (-dists)
        sims = sims / max(knn_temperature, 1e-6)
        sims = sims - sims.max(axis=1, keepdims=True)
        weights = np.exp(sims)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12

        # Aggregate per-class kNN scores (vectorized scatter-add per row)
        n_query, n_classes = len(query_idx), len(classes)
        neigh_cols = train_label_cols[neigh_pos]              # (n_query, k)
        valid = neigh_cols >= 0
        knn_scores = np.zeros((n_query, n_classes), dtype=np.float64)
        # Vectorized: stack rows then np.add.at on (row, col)
        rows = np.repeat(np.arange(n_query), k)
        cols = neigh_cols.ravel()
        w = weights.ravel()
        keep = valid.ravel()
        np.add.at(knn_scores, (rows[keep], cols[keep]), w[keep])

        # Model soft predictions → softmax (in case they're stored as logits)
        m = soft_df.iloc[query_idx].values.astype(np.float64)
        m = m - m.max(axis=1, keepdims=True)
        m_soft = np.exp(m)
        m_soft /= m_soft.sum(axis=1, keepdims=True) + 1e-12

        blended = alpha * m_soft + (1.0 - alpha) * knn_scores
        out = pd.DataFrame(blended, columns=classes, index=soft_df.index[query_idx])

        if update_predictions:
            preds_obs = self.adata.obs[PREDICTION_KEYS.PREDICTION_KEY].values.astype(object)
            preds_obs[query_idx] = classes[blended.argmax(axis=1)]
            self.adata.obs[PREDICTION_KEYS.PREDICTION_KEY] = preds_obs
            soft_vals = soft_df.values.astype(np.float64).copy()
            soft_vals[query_idx] = blended
            self.adata.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY] = pd.DataFrame(
                soft_vals, columns=classes, index=soft_df.index
            )
            self._register_classification_report(force=True)
        log.info(f'[kNN] Done (alpha={alpha}, k={k}, T={knn_temperature}).')
        return out


    @_release_query_managers
    def tent_adapt(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_iters: int = 1,
        lr: float = 1e-4,
        batch_size: int | None = 1024,
    ) -> None:
        """Test-time entropy minimization (TENT) on LayerNorm affine params.

        Adapts only the LayerNorm γ/β of the encoder + aligner by minimizing
        prediction entropy over (unlabeled) target data — typically the OOD
        split before scoring it. All other params stay frozen, the rest of
        the model is unchanged after this call. Re-run for each target
        distribution; effect is small but nearly free.

        Parameters
        ----------
        adata : optional override for the data to adapt on; defaults to ``self.adata``.
        indices : cells to adapt on (e.g. ``self.ood_test_indices``).
        n_iters : passes over the target data (1-3 is typical).
        lr : learning rate for the LN params.
        batch_size : minibatch size for adaptation.
        """
        self._check_if_trained(warn=False)
        if adata is None:
            adata = self.adata
        # Probe var alignment first so we know whether to relax scvi's
        # X-layer var-count check during validation.
        align_col_map, align_n_model_vars = self._build_var_alignment(adata)
        if align_col_map is not None:
            with self._relax_layer_var_check():
                adata = self._validate_anndata(adata, extend_categories=True, add_missing_categories=True)
        else:
            adata = self._validate_anndata(adata, extend_categories=True, add_missing_categories=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        loader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        if align_col_map is not None:
            align_col_map = align_col_map.to(self.device)

        # Freeze every param, then unfreeze only LayerNorm γ/β
        prev_requires_grad = {n: p.requires_grad for n, p in self.module.named_parameters()}
        for p in self.module.parameters():
            p.requires_grad = False
        ln_params = []
        for m in self.module.modules():
            if isinstance(m, torch.nn.LayerNorm):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    ln_params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    ln_params.append(m.bias)
        if not ln_params:
            log.warning('[TENT] No LayerNorm params found; skipping.')
            for n, p in self.module.named_parameters():
                p.requires_grad = prev_requires_grad[n]
            return
        opt = torch.optim.Adam(ln_params, lr=lr)
        # Train mode for dropout/noise off via flag flip is unnecessary here:
        # we only need the LN affines updated. Keep eval() so dropout stays off.
        self.module.eval()
        log.info(f'[TENT] Adapting {sum(p.numel() for p in ln_params)} LN params over {len(indices)} cells × {n_iters} iter(s).')
        for it in range(n_iters):
            for tensors in loader:
                # module.classify is decorated with @torch.no_grad for normal
                # inference paths; enable_grad here so the LayerNorm γ/β
                # updates we want actually flow through autograd.
                x_in = tensors.get(REGISTRY_KEYS.X_KEY)
                if align_col_map is not None and x_in is not None:
                    from expert.utils.common import align_x_batch
                    x_in = align_x_batch(x_in, align_col_map, align_n_model_vars)
                cls_out = self.module.classify(
                    x=x_in,
                    batch_index=tensors.get(REGISTRY_KEYS.BATCH_KEY),
                    label=tensors.get(REGISTRY_KEYS.LABELS_KEY),
                    cont_covs=tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                    cat_covs=tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
                    use_posterior_mean=True,
                    return_logits=True,
                    local_predictions=True,
                )
                logits = cls_out[MODULE_KEYS.CLS_LOGITS_KEY]
                # Mean prediction entropy
                p = torch.softmax(logits, dim=-1)
                logp = torch.log_softmax(logits, dim=-1)
                ent = -(p * logp).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                ent.backward()
                opt.step()
        # Restore original requires_grad map
        for n, param in self.module.named_parameters():
            param.requires_grad = prev_requires_grad[n]
        self.module.eval()
        log.info('[TENT] Done.')

    def evaluate(
            self,
            plot: bool = False,
            save_anndata: bool = False,
            output_dir: str | None = None,
            results_mode: Literal['return', 'save'] | None | list[str] = 'save',
            force: bool = True,
            ignore_ctrl: bool = True,
            z_shared: bool = True,
            use_z: bool = False,
            minify_adata: bool = True,
            progress_bar: bool = True,
            knn_rerank: bool = False,
            knn_k: int = 50,
            knn_alpha: float = 0.5,
            knn_temperature: float = 0.1,
            skip_splits: list[str] | tuple[str, ...] | str | None = None,
            batch_size: int = 1024,
            obs_only: bool = False,
            predict_kwargs: dict | None = None,
        ) -> dict[pd.DataFrame] | None:
        """Run model evaluation for all available data splits registered with the model.

        ``skip_splits`` (default ``None``) skips forward passes,
        classification reports, and per-split logit saving for the listed
        splits. The train split is skipped by default because predicting
        over it can be the bulk of the eval cost. Pass ``skip_splits=None``
        or ``skip_splits=[]`` to evaluate every available split.

        When ``train`` is skipped, ``minify_adata`` and ``knn_rerank`` are
        disabled (both depend on having latents and labels for train cells).
        """
        if self.is_evaluated and not force:
            log.info('[Eval] This model has already been evaluated, pass force=True to re-evaluate.')
            return
        # Set module to evaluation mode
        self.module.eval()
        # Save model to tensorboard logger directory if registered
        if getattr(self, 'model_log_dir', None) is not None:
            base_output_dir = self.model_log_dir
            model_output_dir = os.path.join(self.model_log_dir, 'model')
        # Try to fall back to output_dir parameter and skip if not specified
        else:
            if output_dir is not None:
                base_output_dir = output_dir
                model_output_dir = os.path.join(output_dir, 'model')
            else:
                log.warning('[Eval] Could not find model tensorboard log directory or a specified "output_dir", skipping model saving.')
                base_output_dir = None
                model_output_dir = None
        # Refactor results mode to always be a list of options or None
        results_mode: list[str] | None = [results_mode] if results_mode is not None and not isinstance(results_mode, list) else None

        # Resolve skip_splits to a list and compute the indices that *will*
        # be evaluated. ``kept_indices=None`` means "all cells" (i.e. don't
        # skip anything); a non-None value triggers the scatter-back path
        # in the registration helpers so skipped cells get NaN.
        if isinstance(skip_splits, str):
            skip_splits = [skip_splits]
        skip_splits = list(skip_splits) if skip_splits else []
        available_splits = self._get_available_splits()
        skip_set = set(skip_splits) & set(available_splits)
        kept_split_names = [s for s in available_splits if s not in skip_set]
        if skip_set:
            log.info(
                f'[Eval] Skipping splits: {sorted(skip_set)} '
                f'(use skip_splits=None to evaluate all). Evaluating: {kept_split_names}'
            )
            # These features need full-adata coverage / train labels.
            if minify_adata and 'train' in skip_set:
                log.warning('[Eval] minify_adata disabled — needs latents over the full adata, but train is skipped.')
                minify_adata = False
            if knn_rerank and 'train' in skip_set:
                log.warning('[Eval] knn_rerank disabled — needs train latents to build the kNN index, but train is skipped.')
                knn_rerank = False
            kept_indices = (
                np.concatenate([np.asarray(available_splits[s]) for s in kept_split_names])
                if kept_split_names else np.array([], dtype=int)
            )
        else:
            kept_indices = None
        skip_list = sorted(skip_set)

        # When skipping splits, build a lightweight eval_adata to receive
        # obsm/obs results. Anndata enforces n_obs alignment, and full-adata
        # scatter on a 14M-cell × 7K-class softmax would need ~400GB. We can't
        # `.copy()` a backed adata in memory, so we construct a *new*
        # in-memory AnnData containing just obs/var/uns of the kept cells —
        # no X. self.adata stays untouched (preserves train cells, original
        # backing); downstream registration/plotting helpers route reads and
        # writes through self.eval_adata, which falls back to self.adata
        # when not set. The forward pass below reads X from self.adata at
        # ``kept_indices``; results land 1:1 on eval_adata.
        if kept_indices is not None and len(kept_indices) > 0:
            log.info(
                f'[Eval] Building lightweight eval_adata: '
                f'{self.adata.n_obs:,} -> {len(kept_indices):,} cells '
                f'(kept splits: {kept_split_names}). self.adata stays '
                f'untouched; eval results will be stored on self.eval_adata.'
            )
            # Pre-populate the SPLIT_KEY column on the new adata so split
            # filtering (reports / save_logits / plotting) works without
            # remapping the model's positional split-indices arrays.
            # kept_indices is the concat of available_splits[s] in
            # kept_split_names order — labels are repeats of the split
            # names by their sizes.
            split_sizes = [len(np.asarray(available_splits[s])) for s in kept_split_names]
            split_labels = np.repeat(kept_split_names, split_sizes)
            eval_adata = ad.AnnData(
                obs=self.adata.obs.iloc[kept_indices].copy(),
                var=self.adata.var.copy(),
                uns=dict(self.adata.uns),
            )
            eval_adata.obs_names = self.adata.obs_names[kept_indices]
            eval_adata.obs[REGISTRY_KEYS.SPLIT_KEY] = pd.Categorical(split_labels)
            self._eval_adata = eval_adata
        
        # Run full prediction — qzm/qzv are captured during the same forward
        # pass and registered alongside z/z_shared, so no separate
        # _register_latent_variables call is needed. The forward pass reads X
        # from self.adata at ``kept_indices``; the helper writes results to
        # self.eval_adata which is either self.adata (full eval) or the
        # smaller subset built above (skip-splits eval).
        log.info('[Eval] Running model predictions on data splits.')
        # predict_kwargs reaches predict() verbatim. WITHOUT IT A ctrl-CENTRED MODEL CANNOT BE
        # EVALUATED ON A HELD-OUT-DATASET SPLIT AT ALL: predict() refuses when the query carries
        # no control cells, and the eval subset is built from test/ood_test indices only, which
        # by construction exclude them. `center_mode='query_mean'` is the same resolution
        # ood_readout_probe arrives at automatically for these folds.
        self._register_model_predictions(
            force=force, use_z=use_z, progress_bar=progress_bar,
            indices=kept_indices, skip_splits=skip_list, batch_size=batch_size,
            obs_only=obs_only, **(predict_kwargs or {})
        )
        # Run classification report for all data splits
        log.info('[Eval] Generating reports.')
        self._register_classification_report(
            force=force, ignore_ctrl=ignore_ctrl, skip_splits=skip_list,
        )

        # Optional kNN reranking on z_shared (post-hoc, no retrain). Runs
        # AFTER the initial report so the standard summary reflects the
        # plain model, then knn_rerank() re-runs the report internally.
        if knn_rerank:
            log.info(f'[Eval] kNN reranking (k={knn_k}, alpha={knn_alpha}, T={knn_temperature}).')
            self.knn_rerank(
                k=knn_k, alpha=knn_alpha, knn_temperature=knn_temperature,
                update_predictions=True, force=force,
            )

        # Minify the model if enabled and adata is not already minified
        if minify_adata and not self.is_minified:
            log.info(f'[Eval] Minifying adata using latent distribution')
            self.minify_adata(use_latent_qzm_key=self._LATENT_QZM_KEY, use_latent_qzv_key=self._LATENT_QZV_KEY)
        # Save model only if we have an output directory
        if model_output_dir is not None:
            # Always save anndata if its minified else fall back to parameter
            save_anndata = True if self.is_minified else save_anndata
            save_ad_txt = 'with' if save_anndata else 'without'
            adata_txt = 'adata (minified)' if self.is_minified else 'adata'
            log.info(f'[Eval] Saving model {save_ad_txt} {adata_txt} to: {model_output_dir}')
            # NOTE: no keep_emb=True. The full class-embedding DataFrame lives in
            # adata.uns['cls_embedding'] (up to ~19895 x 4096); its 4096 column-name
            # strings overflow the HDF5 ~64KB object-header limit on write_h5ad
            # ("object header message is too large"). Letting keep_emb default to
            # False re-enables the pop-guard in save() (pops cls_embedding/_sim from
            # uns before write, restores after), and reload is unaffected — save()
            # always writes the full bank to gene_embeddings.npz and setup rebuilds
            # adata.uns['cls_embedding'] from it.
            self.save(dir_path=model_output_dir, save_anndata=save_anndata, overwrite=True)
        # Create evalutaion return object
        return_obj = None
        if results_mode is not None:
            # Collect evaluation result metrics
            eval_report = self.get_eval_report()
            eval_summary = self.get_eval_summary()
            # Return evaluation results as dictionary
            if 'return' in results_mode:
                return_obj = {
                    PREDICTION_KEYS.REPORT_KEY: eval_report,
                    PREDICTION_KEYS.SUMMARY_KEY: eval_summary,
                }
            # Save reports as seperate files in output directory
            if 'save' in results_mode:
                log.info(f'[Eval] Saving evaluation metrics to: {base_output_dir}')
                sum_o = os.path.join(base_output_dir, f'{self._name}_eval_summary.csv')
                rep_o = os.path.join(base_output_dir, f'{self._name}_eval_report.csv')
                eval_summary.to_csv(sum_o)
                eval_report.to_csv(rep_o)
                # Save logits for each eval split
                self._save_eval_logits(output_dir=base_output_dir, skip_splits=skip_list)

        log.info(f'[Eval] Evaluation done.')
        # Plot evalutation results if specified
        if plot:
            log.info('[Eval] Plotting evaluation results.')
            # _plot_evalutation reads uns[TOP_N_PREDICTION_KEY]; the
            # prediction registration above leaves it off by default
            # (calc_top_n=False) since only plotting needs it. Compute now.
            if PREDICTION_KEYS.TOP_N_PREDICTION_KEY not in self.eval_adata.uns:
                self._register_top_n_predictions()
            self._plot_evalutation(output_dir=base_output_dir, z_shared=z_shared, skip_splits=skip_list)
        # Return 
        return return_obj
    
    def get_eval_summary(self) -> pd.DataFrame | None:
        if PREDICTION_KEYS.SUMMARY_KEY not in self.eval_adata.uns:
            log.warning('No model evaluation summary available.')
            return None
        return self.eval_adata.uns[PREDICTION_KEYS.SUMMARY_KEY]

    def get_eval_report(self) -> pd.DataFrame | None:
        if PREDICTION_KEYS.REPORT_KEY not in self.eval_adata.uns:
            log.warning('No model evaluation report available.')
            return None
        return self.eval_adata.uns[PREDICTION_KEYS.REPORT_KEY]

    def evaluate_zero_shot_perturbations(
        self,
        adata: "AnnData | str | os.PathLike",
        label_key: str | None = None,
        batch_key: str | None = None,
        family_tau: float = 0.6,
        top_k: tuple[int, ...] = (1, 5, 10),
        chunk_size: int = 50_000,
        batch_size: int = 1024,
        save_path: str | None = None,
        save_topk_path: str | None = None,
        save_topk: int = 25,
        progress_bar: bool = False,
        center_mode: str | np.ndarray | torch.Tensor | None = None,
        control_label: str | None = None,
    ) -> pd.DataFrame:
        """Zero-shot evaluation on held-out (unseen) perturbations.

        Scores cells whose perturbed gene was NEVER trained against the FULL
        class-embedding vocabulary (the held-out anchors enter the softmax), so
        the model has to place each cell purely from the alignment geometry. Two
        splits are reported, distinguished by whether the cell's dataset was seen
        in training:

            id_ood_pert   — trained dataset,  unseen perturbation (hard)
            ood_ood_pert  — held-out dataset, unseen perturbation (very hard)

        This is the missing pair of the seen/unseen × dataset/perturbation 2x2
        (val / ID-test / OOD-test cover the seen-perturbation column).

        Cells are gated on load by their gene's position in the class bank:
        keep iff ``n_labels <= idx < n_cls`` — one test that guarantees (a) not
        a trained class (no leakage), (b) present in the embedding bank (so it is
        a candidate at all), and (c) in the held-out block the full-vocab metric
        can reach. Trained-gene cells (leakage) and genes absent from the bank
        are dropped with a logged count. Matching is by gene NAME against
        ``idx_to_label`` — canonicalize the adata's perturbation labels to the
        bank's symbol namespace beforehand (aliases won't match).

        Parameters
        ----------
        adata
            Pre-computed merged AnnData of unseen-perturbation cells (or a path
            to an ``.h5ad``). Must carry the perturbation and dataset obs columns
            and be reconcilable to the model's gene panel (predict() reindexes).
        label_key / batch_key
            obs columns for the perturbed gene / dataset. Default to the model's
            registered ``original_label_key`` / ``original_batch_key``.
        family_tau
            Cosine threshold (in the whitened+centered teacher space, matching
            ``soft_target_min_cos``) for the ``family_hit@1`` metric: a top-1
            prediction counts as a family hit if the predicted gene is within
            ``family_tau`` cosine of the true gene — credit for landing on a
            functional sibling even when the exact gene is wrong.
        top_k
            Recall@k cutoffs (full-vocab and held-out-only).
        center_mode / control_label
            Forwarded to :meth:`predict` for control-free differential inference on a
            NOVEL dataset (subtract a per-dataset ``z_shared`` baseline before scoring;
            only active if the module trained with ``invariant_align``). Computed
            per-chunk, so keep ``chunk_size`` large enough to estimate the baseline.

        Returns
        -------
        pd.DataFrame indexed by split (``id_ood_pert``, ``ood_ood_pert``,
        ``overall``) with n_cells, n_perts, recall@k (full vocab + held-out
        only), median rank, and family_hit@1.
        """
        import anndata as ad
        import torch.nn.functional as F
        self._check_if_trained(warn=False)
        if isinstance(adata, (str, os.PathLike)):
            log.info(f'[ZeroShot] Reading unseen-perturbation adata: {adata}')
            adata = ad.read_h5ad(adata)
        label_key = label_key or self.original_label_key
        batch_key = batch_key or self.original_batch_key
        for k in (label_key, batch_key):
            if k not in adata.obs:
                raise KeyError(f"[ZeroShot] adata.obs is missing required column {k!r}.")

        # --- Bank membership sets (gene-name namespace = idx_to_label) ---
        bank = np.asarray(self.idx_to_label).astype(str)
        n_labels = int(self.n_labels)
        n_cls = int(bank.shape[0])
        trained_set = set(bank[:n_labels].tolist())
        heldout_set = set(bank[n_labels:].tolist())
        label_to_row = {g: i for i, g in enumerate(bank.tolist())}

        genes = adata.obs[label_key].astype(str).to_numpy()
        in_heldout = np.array([g in heldout_set for g in genes])
        in_trained = np.array([g in trained_set for g in genes])
        in_bank = in_heldout | in_trained
        is_ctrl = (genes == str(self.ctrl_class)) if getattr(self, 'ctrl_class', None) is not None else np.zeros(len(genes), bool)
        keep = in_heldout & ~is_ctrl
        log.info(
            f'[ZeroShot] Gene gate on {len(genes):,} cells: keep(held-out)={int(keep.sum()):,} | '
            f'drop leakage(trained)={int((in_trained & ~is_ctrl).sum()):,} | '
            f'drop not-in-bank={int((~in_bank & ~is_ctrl).sum()):,} | drop ctrl={int(is_ctrl.sum()):,}. '
            f'({len(heldout_set):,} held-out / {len(trained_set):,} trained genes in bank.)'
        )
        if not keep.any():
            raise ValueError('[ZeroShot] No cells left after the held-out gene gate — check that the '
                             'adata labels use the same gene symbols as the embedding bank and are not trained.')
        adata = adata[keep].copy()
        genes = adata.obs[label_key].astype(str).to_numpy()

        # --- Split by dataset membership (trained batch categories) ---
        trained_batches = set(np.asarray(self._batch_mapping).astype(str).tolist())
        ds = adata.obs[batch_key].astype(str).to_numpy()
        split = np.where(np.isin(ds, list(trained_batches)), 'id_ood_pert', 'ood_ood_pert')
        log.info(f'[ZeroShot] Split by dataset membership: id_ood_pert={int((split=="id_ood_pert").sum()):,} | '
                 f'ood_ood_pert={int((split=="ood_ood_pert").sum()):,}.')

        # --- Teacher-space geometry for the family-hit metric (whiten+center) ---
        raw_bank, _ = self.get_cls_emb(use_full_cls_emb=True)
        raw_bank = F.normalize(raw_bank.float(), dim=-1)
        rc = F.normalize(raw_bank - raw_bank.mean(dim=0, keepdim=True), dim=-1).cpu().numpy()  # (n_cls, d)
        heldout_cols_mask = np.zeros(n_cls, bool)
        heldout_cols_mask[n_labels:] = True

        # --- Stream chunks through predict() (bounds the cells x n_cls softmax) ---
        n = adata.n_obs
        ks = tuple(int(k) for k in top_k)
        acc = {c: {'n': 0, 'rank': [], 'rank_ho': [],
                   **{f'r{k}': 0 for k in ks}, **{f'rho{k}': 0 for k in ks},
                   'fam': 0, 'perts': set()} for c in ('id_ood_pert', 'ood_ood_pert')}
        # Optional: stash per-cell top-K ranked predictions so any annotation
        # metric (pathway/family/GO) can be scored POST-HOC without re-running
        # inference. Column order = idx_to_label, so held-out block is >= n_labels.
        # `dataset` is stored alongside `split` because split only separates id_ood_pert from
        # ood_ood_pert, which pools all 21 datasets together. Those datasets are not exchangeable:
        # the 12 Marson ones have a DEG-response target with cross-donor reliability 0.0037 (no
        # donor-invariant component => a zero-shot score there measures donor, not transfer), and
        # they are 54% of database_zeroshot_subset.h5ad by cell count, while the four datasets where
        # the text prior demonstrably works contribute ~1.3%. A pooled number is therefore ~94%
        # composed of cases where the prior cannot work. Keeping the dataset per cell makes the
        # per-dataset breakdown scoreable post-hoc without re-running inference.
        store = {'true_gene': [], 'split': [], 'dataset': [], 'topk_genes': [], 'topk_scores': []} if save_topk_path else None
        kmax = int(min(save_topk, n_cls)) if save_topk_path else 0
        for start in range(0, n, chunk_size):
            sub = adata[start:start + chunk_size]
            out = self.predict(sub, obs_only=False, soft_predictions=True,
                               return_latents=False, add_report=False, inplace=False,
                               progress_bar=progress_bar, batch_size=batch_size,
                               center_mode=center_mode, control_label=control_label)
            soft = out[PREDICTION_KEYS.SOFT_PREDICTION_KEY]
            cols = np.asarray(soft.columns).astype(str)
            L = np.asarray(soft.values, dtype=np.float32)                 # (m, n_cols)
            col_to_j = {g: j for j, g in enumerate(cols)}
            g_chunk = genes[start:start + chunk_size]
            sp_chunk = split[start:start + chunk_size]
            ds_chunk = ds[start:start + chunk_size]
            t_j = np.array([col_to_j[g] for g in g_chunk])               # true gene column (guaranteed present)
            m = L.shape[0]
            true_score = L[np.arange(m), t_j]
            rank_full = (L > true_score[:, None]).sum(1) + 1
            ho_cols = np.array([j for j, g in enumerate(cols) if g in heldout_set])
            rank_ho = (L[:, ho_cols] > true_score[:, None]).sum(1) + 1
            pred_gene = cols[L.argmax(1)]
            pred_row = np.array([label_to_row[g] for g in pred_gene])
            true_row = np.array([label_to_row[g] for g in g_chunk])
            fam = (rc[pred_row] * rc[true_row]).sum(1) >= float(family_tau)
            for c in ('id_ood_pert', 'ood_ood_pert'):
                sel = sp_chunk == c
                if not sel.any():
                    continue
                a = acc[c]
                a['n'] += int(sel.sum())
                a['rank'].append(rank_full[sel]); a['rank_ho'].append(rank_ho[sel])
                for k in ks:
                    a[f'r{k}'] += int((rank_full[sel] <= k).sum())
                    a[f'rho{k}'] += int((rank_ho[sel] <= k).sum())
                a['fam'] += int(fam[sel].sum())
                a['perts'].update(g_chunk[sel].tolist())
            # Stash top-K ranked predictions for post-hoc annotation scoring.
            if store is not None:
                part = np.argpartition(-L, kmax - 1, axis=1)[:, :kmax]           # unordered top-K
                order = np.argsort(-np.take_along_axis(L, part, axis=1), axis=1)  # sort within
                tk = np.take_along_axis(part, order, axis=1)                      # (m, kmax) col idx
                store['true_gene'].append(g_chunk)
                store['split'].append(sp_chunk)
                store['dataset'].append(ds_chunk)
                store['topk_genes'].append(cols[tk])                              # gene names
                store['topk_scores'].append(np.take_along_axis(L, tk, axis=1).astype(np.float32))

        # --- Aggregate ---
        rows = {}
        def _agg(a):
            if a['n'] == 0:
                return None
            ranks = np.concatenate(a['rank']); ranks_ho = np.concatenate(a['rank_ho'])
            row = {'n_cells': a['n'], 'n_perts': len(a['perts']),
                   'median_rank': float(np.median(ranks)),
                   'median_rank_heldout_only': float(np.median(ranks_ho)),
                   'family_hit@1': a['fam'] / a['n']}
            for k in ks:
                row[f'recall@{k}'] = a[f'r{k}'] / a['n']
                row[f'recall@{k}_heldout_only'] = a[f'rho{k}'] / a['n']
            return row
        for c in ('id_ood_pert', 'ood_ood_pert'):
            r = _agg(acc[c])
            if r is not None:
                rows[c] = r
        # Overall = pooled
        pooled = {'n': sum(acc[c]['n'] for c in acc),
                  'rank': [x for c in acc for x in acc[c]['rank']],
                  'rank_ho': [x for c in acc for x in acc[c]['rank_ho']],
                  'fam': sum(acc[c]['fam'] for c in acc),
                  'perts': set().union(*(acc[c]['perts'] for c in acc)),
                  **{f'r{k}': sum(acc[c][f'r{k}'] for c in acc) for k in ks},
                  **{f'rho{k}': sum(acc[c][f'rho{k}'] for c in acc) for k in ks}}
        ro = _agg(pooled)
        if ro is not None:
            rows['overall'] = ro
        df = pd.DataFrame(rows).T
        df.index.name = 'split'
        if save_path is not None:
            df.to_csv(save_path)
            log.info(f'[ZeroShot] Saved zero-shot perturbation metrics to: {save_path}')
        if store is not None:
            np.savez_compressed(
                save_topk_path,
                true_gene=np.concatenate(store['true_gene']),
                split=np.concatenate(store['split']),
                dataset=np.concatenate(store['dataset']),
                topk_genes=np.concatenate(store['topk_genes']),      # (N, kmax) str
                topk_scores=np.concatenate(store['topk_scores']),    # (N, kmax) f32
                idx_to_label=bank, n_labels=np.int64(n_labels),
            )
            log.info(f'[ZeroShot] Saved per-cell top-{kmax} predictions to: {save_topk_path} '
                     f'(for post-hoc pathway/annotation scoring).')
        log.info(f'[ZeroShot] Done.\n{df}')
        return df

    def _plot_eval_split(self, split: str, plt_dir: str) -> None:
        """Plot confusion matrix per split."""
        # Get split data without control cells
        adata = self._get_split_adata(split, ignore_ctrl=False)
        # Get actual and predicted class labels
        y = adata.obs[self.original_label_key].values.astype(str)
        y_hat = adata.obs[PREDICTION_KEYS.PREDICTION_KEY].values.astype(str)
        # Create output file path
        o = os.path.join(plt_dir, f'cm_{split}.png')
        pl.plot_confusion(y, y_hat, plt_file=o)
        # Plot mode umap using combined projection
        groups = [self.original_batch_key, self.original_label_key]
        # Check if efficiency score is given
        eff_state_registry = self.adata_manager.registry.get('field_registries').get(REGISTRY_KEYS.CLS_EFF_KEY, {})
        eff_key = eff_state_registry.get('data_registry', {}).get('attr_key', None)
        if eff_key is not None:
            groups.append(eff_key)
        for group in groups:
            # Plot a umap projection for every group
            o = os.path.join(plt_dir, f'umap_{split}_{group}.png')
            pl.plot_umap(adata, slot=self._LATENT_UMAP, hue=group, output_file=o)

    def _plot_eval_splits(self, plt_dir: str, skip_splits: list[str] | None = None) -> None:
        """Generate all split-specific plots — iterate via eval_adata's SPLIT_KEY."""
        skip = set(skip_splits or [])
        ad_obj = self.eval_adata
        if REGISTRY_KEYS.SPLIT_KEY in ad_obj.obs:
            splits_in_data = [s for s in pd.unique(ad_obj.obs[REGISTRY_KEYS.SPLIT_KEY].dropna()) if s not in skip]
        else:
            splits_in_data = [s for s in self._get_available_splits() if s not in skip]
        for split in splits_in_data:
            self._plot_eval_split(split, plt_dir=plt_dir)

    def _plot_evalutation(
        self,
        output_dir: str | None,
        z_shared: bool = True,
        metric: str = 'f1-score',
        skip_splits: list[str] | None = None,
    ) -> None:
        """Save plots associated to model evaluation."""
        if output_dir is None:
            log.warning('Evalutaion output directory is not available. Skipping plots.')
            return
        # Set plotting directory and create if needed
        plt_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plt_dir, exist_ok=True)
        # All eval-side plots read from eval_adata (defaults to self.adata).
        ad_obj = self.eval_adata
        # Plot performance - support correlations
        support_corr_o = os.path.join(plt_dir, f'support_correlations.png')
        pl.plot_performance_support_corr(ad_obj.uns[PREDICTION_KEYS.REPORT_KEY], o=support_corr_o, hue=REGISTRY_KEYS.SPLIT_KEY)
        # Plot performance metric over top N predictions in all splits
        top_n_o = os.path.join(plt_dir, f'top_n_{metric}.png')
        pl.plot_top_n_performance(
            ad_obj.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY],
            output_file=top_n_o,
            metric=metric,
            mean_split='test',
            N=ad_obj.obsm[PREDICTION_KEYS.SOFT_PREDICTION_KEY].shape[-1]
        )
        # Plot top N performance over datasets
        top_n_ds_o = os.path.join(plt_dir, f'top_n_ds_{metric}.png')
        pl.plot_top_n_performance_per_dataset(
            ad_obj.uns[PREDICTION_KEYS.TOP_N_PREDICTION_KEY],
            context_key=self.original_batch_key,
            split_key='split',
            metric=metric,
            output_file=top_n_ds_o
        )
        # Efficiency comparison: predicted (eff_hat, z-scored) vs reference
        # (raw weights standardized per-dataset on the eval data so both live
        # in the same space). Skipped if the model didn't train an
        # efficiency head or the registered eff cols aren't present.
        eff_hat_col = 'eff_hat'
        ref_raw_col = REGISTRY_KEYS.CLS_EFF_KEY
        if (
            eff_hat_col in ad_obj.obs.columns
            and ref_raw_col in ad_obj.obs.columns
        ):
            cols = [ref_raw_col, eff_hat_col, REGISTRY_KEYS.SPLIT_KEY, self.original_batch_key]
            eff_df = ad_obj.obs[cols].copy().dropna(subset=[ref_raw_col, eff_hat_col])
            # Per-dataset z-score the reference so its scale matches eff_hat.
            grouped = eff_df.groupby(self.original_batch_key, observed=True)[ref_raw_col]
            std = grouped.transform('std').clip(lower=0.5)
            eff_df['eff_ref'] = (eff_df[ref_raw_col] - grouped.transform('mean')) / std
            try:
                pl.plot_efficiency_comparison(
                    eff_df, ref_col='eff_ref', pred_col=eff_hat_col,
                    split_col=REGISTRY_KEYS.SPLIT_KEY,
                    plt_file=os.path.join(plt_dir, 'efficiency_per_split.png'),
                )
                pl.plot_efficiency_comparison(
                    eff_df, ref_col='eff_ref', pred_col=eff_hat_col,
                    split_col=REGISTRY_KEYS.SPLIT_KEY,
                    dataset_col=self.original_batch_key,
                    plt_file=os.path.join(plt_dir, 'efficiency_per_dataset.png'),
                )
            except Exception as e:
                log.warning(f'[Eval] Skipped efficiency comparison plots: {e}')
        # Latent-based plots run on cells with valid latents. When eval_adata
        # is the full self.adata and some splits were skipped, those cells
        # have NaN rows in obsm[z_key]; UMAP can't handle NaN, so subset to
        # the evaluated cells. When eval_adata is already the subset, this
        # is a no-op.
        skip = set(skip_splits or [])
        if skip and REGISTRY_KEYS.SPLIT_KEY in ad_obj.obs:
            eval_mask = ad_obj.obs[REGISTRY_KEYS.SPLIT_KEY].notna().values
            plot_adata = ad_obj[eval_mask].copy() if eval_mask.sum() < ad_obj.n_obs else ad_obj
        else:
            plot_adata = ad_obj
        # Get latent space key
        z_key = self._LATENT_Z_SHARED_KEY if z_shared else self._LATENT_Z_KEY
        # Calculate UMAP for latent space based on latent mean
        pl.calc_umap(plot_adata, rep=z_key, slot_key=self._LATENT_UMAP)
        # Plot full UMAP colored by data split
        umap_split_o = os.path.join(plt_dir, f'full_umap_{REGISTRY_KEYS.SPLIT_KEY}.png')
        pl.plot_umap(plot_adata, slot=self._LATENT_UMAP, hue=REGISTRY_KEYS.SPLIT_KEY, output_file=umap_split_o)
        # Plot full UMAP colored by batch label
        umap_batch_o = os.path.join(plt_dir, f'full_umap_{self.original_batch_key}.png')
        pl.plot_umap(plot_adata, slot=self._LATENT_UMAP, hue=self.original_batch_key, output_file=umap_batch_o)
        # Plot full UMAP colored by batch label
        umap_label_o = os.path.join(plt_dir, f'full_umap_{self.original_label_key}.png')
        pl.plot_umap(plot_adata, slot=self._LATENT_UMAP, hue=self.original_label_key, output_file=umap_label_o)
        # Plot shared z umap with context and class embedding markers on it
        umap_proxy_o = os.path.join(plt_dir, f'full_umap_proxy.png')
        pl.plot_umap_with_proxies(
            plot_adata,
            latent_key=z_key,
            proxy_key=self._LATENT_CLS_PROJ_KEY,
            perturbation_col=self.original_label_key,
            batch_key=self.original_batch_key,
            idx_to_label=self.idx_to_label,
            output_file=umap_proxy_o
        )
        # Plot individual splits
        self._plot_eval_splits(plt_dir, skip_splits=skip_splits)
        return
    
    @staticmethod
    @contextlib.contextmanager
    def _relax_layer_var_check():
        """Disable scvi-tools' LayerField n_vars-equality check.

        scvi's ``LayerField.transfer_field`` asserts ``target.n_vars ==
        source.n_vars`` — which blocks us from registering a query adata
        that lives in a different var space than the training adata. We
        per-batch align X to the model's var space inside the predict
        loop, so the upfront var-count enforcement is the wrong check
        here. Categorical fields (batch_key, labels_key, …) still
        transfer normally inside this context — only the X-layer var
        check is relaxed, so training categorical mappings are preserved.
        """
        from scvi.data.fields._layer_field import LayerField
        from scvi.data.fields._base_field import BaseAnnDataField
        original = LayerField.transfer_field
        def _relaxed_transfer(self, state_registry, adata_target, **kwargs):
            # Skip the n_vars equality assertion; everything else is the
            # same as LayerField.transfer_field upstream.
            BaseAnnDataField.transfer_field(self, state_registry, adata_target, **kwargs)
            return self.register_field(adata_target)
        LayerField.transfer_field = _relaxed_transfer
        try:
            yield
        finally:
            LayerField.transfer_field = original

    def _build_var_alignment(self, adata: AnnData) -> tuple[torch.Tensor | None, int]:
        """Build a per-batch column-remap plan without materializing X.

        Returns ``(col_map, n_model_vars)``:
          - ``col_map`` is a ``torch.long`` tensor of length ``n_model_vars``;
            ``col_map[j]`` is the column index in ``adata.X`` for output
            column ``j``, or ``-1`` to insert a zero column there.
          - Returns ``(None, n_model_vars)`` if the query already matches
            the model's var space exactly (caller skips the per-batch
            remap entirely).

        Used by ``predict()`` to stream column alignment per minibatch
        instead of materializing the full aligned X up front. Keeps backed
        queries on-disk and avoids 2× memory peaks for large in-memory
        queries.
        """
        model_genes = self.adata.var_names
        n_model_vars = len(model_genes)
        # Fast path: nothing to remap, caller can skip the per-batch hop.
        if adata.var_names.equals(model_genes):
            return None, n_model_vars
        col_map_np = adata.var_names.get_indexer(model_genes)  # -1 for missing
        n_missing = int((col_map_np < 0).sum())
        n_extra = int(adata.n_vars - int((col_map_np >= 0).sum()))
        if n_missing > 0:
            log.info(
                f'[Predict] Aligning {n_missing} missing features in-batch '
                f'(zero-filled).'
            )
        if n_extra > 0:
            log.info(
                f'[Predict] Dropping {n_extra} query features not used by '
                f'the model (in-batch).'
            )
        return torch.as_tensor(col_map_np, dtype=torch.long), n_model_vars

    def minify_query(self, adata: ad.AnnData) -> None:
        """Minify a query adata: zero out X, keeping the latent qzm/qzv in
        obsm so the model can still cached-inference against it.

        Requires that the latents are already in obsm (e.g. from a
        ``predict(..., inplace=True, return_latents=True)`` call). Drops
        any layers too — once X is gone, any per-gene layer is just
        stale.
        """
        if self._LATENT_QZM_KEY not in adata.obsm or self._LATENT_QZV_KEY not in adata.obsm:
            raise KeyError(
                f'Query data missing latent representation '
                f'({self._LATENT_QZM_KEY} / {self._LATENT_QZV_KEY} in obsm). '
                f'Run predict(inplace=True, return_latents=True) first.'
            )
        # Empty sparse matrix in the original shape — avoids allocating a
        # dense n_obs × n_vars buffer just to discard it. Stays CSR for
        # consistency with the rest of the pipeline.
        dtype = adata.X.dtype if hasattr(adata.X, 'dtype') else np.float32
        adata.X = sp.csr_matrix(adata.X.shape, dtype=dtype)
        # Drop layers — they were aligned to the old X and are now misleading.
        if len(adata.layers) > 0:
            log.info(f'[Minify] Dropping {len(adata.layers)} layer(s) along with X.')
            for k in list(adata.layers.keys()):
                del adata.layers[k]

    def get_training_classes(self) -> np.ndarray:
        """Get trained class label names. Class labels are sorted by observed and unobserved."""
        return self.idx_to_label[:self.n_train_labels]
    
    @staticmethod
    def _synthesize_registry_adata(dir_path: str) -> AnnData | None:
        """Build a minimal adata from the SAVED REGISTRY so load() needs no training data.

        Everything inference actually consumes is already inside model.pt: `var_names`, and the
        registry's per-field `categorical_mapping` (the full label and dataset category lists).
        The control means, class centres, alignment EMA and px_r are BUFFERS in the state dict,
        so they are restored after setup and are unaffected by what this shell holds.

        The shell carries no real cells, so it is for inference only (the caller records that). It
        spans every category by construction, which setup's full-category indexing requires.

        Returns None when the registry is unreadable, so the caller falls back to the old paths.
        """
        import scipy.sparse as _sp
        model_pt = next((os.path.join(dir_path, n) for n in ('model.pt', 'model.pth')
                         if os.path.exists(os.path.join(dir_path, n))), None)
        if model_pt is None:
            return None
        try:
            blob = torch.load(model_pt, map_location='cpu', weights_only=False)
            var_names = np.asarray(blob['var_names']).astype(str)
            reg = blob['attr_dict']['registry_']
            fields = reg['field_registries']
        except Exception as e:
            log.warning(f'[Load] cannot synthesize a registry adata: {type(e).__name__}: {e}')
            return None

        # READ THE REGISTRY, do not hardcode field names. Every field records where it comes
        # from: `data_registry.attr_name` ('obs' / 'obsm' / 'uns' / 'varm') and `attr_key` (the
        # column). Hardcoding 'labels'/'batch' missed `cls_efficiency`, whose column is `efficiency`
        # under attr_key with an empty state_registry -- and the shell died on it.
        cats: dict[str, np.ndarray] = {}
        conts: list[str] = []
        for fld, fr in fields.items():
            dr = fr.get('data_registry', {}) or {}
            sr = fr.get('state_registry', {}) or {}
            attr_name, attr_key = dr.get('attr_name'), dr.get('attr_key')
            if attr_name == 'obs':
                orig, mapping = sr.get('original_key'), sr.get('categorical_mapping')
                if orig is not None and mapping is not None:
                    cats[str(orig)] = np.asarray(mapping).astype(str)   # setup re-encodes this
                elif isinstance(attr_key, str):
                    conts.append(attr_key)                              # e.g. efficiency
            elif attr_name == 'obsm' and fld == 'extra_categorical_covs':
                # built from obs columns named in field_keys, with their own category lists
                # `mappings` is a DICT keyed by field name here (not a positional list) --
                # indexing it positionally yields the KEY string and a 0-d array. Accept both.
                fkeys = list(sr.get('field_keys', []) or [])
                maps = sr.get('mappings', {}) or {}
                for _i, _k in enumerate(fkeys):
                    mv = maps.get(_k) if isinstance(maps, dict) else (
                        maps[_i] if _i < len(maps) else None)
                    if mv is None:
                        continue
                    mv = np.asarray(mv).astype(str).ravel()
                    if mv.size:
                        cats[str(_k)] = mv
        if not cats:
            log.warning('[Load] registry exposes no obs categoricals; cannot synthesize.')
            return None

        cats = {k: np.asarray(v).astype(str).ravel() for k, v in cats.items()}
        cats = {k: v for k, v in cats.items() if v.size}
        if not cats:
            log.warning('[Load] no non-empty categorical mappings; cannot synthesize.')
            return None
        n = int(max(v.size for v in cats.values()))
        obs = pd.DataFrame(index=pd.Index([f'synthetic_{i}' for i in range(n)]))
        for k, v in cats.items():
            # tile so EVERY category appears at least once -- that is the whole point, and it is
            # what satisfies setup's full-category indexing
            obs[k] = pd.Categorical(v[np.arange(n) % v.size], categories=list(v))
        for k in conts:
            obs[k] = np.zeros(n, dtype=np.float32)   # never read at inference

        adata = AnnData(X=_sp.csr_matrix((n, len(var_names)), dtype=np.float32), obs=obs)
        adata.var_names = var_names
        adata.obsm['_scvi_latent_qzm'] = np.zeros((n, 0), dtype=np.float32)
        adata.obsm['_scvi_latent_qzv'] = np.zeros((n, 0), dtype=np.float32)
        log.info(f'[Load] synthesized a {n}-row registry adata from model.pt '
                 f'({", ".join(f"{k}:{len(v)}" for k, v in cats.items())}, {len(var_names)} vars). '
                 f'INFERENCE ONLY -- the count-derived loss weights are uniform placeholders.')
        return adata

    @staticmethod
    def _load_shell_adata_for_inference(shell_path: str) -> AnnData | None:
        """Load the saved adata, injecting empty minify slots if it's a shell.

        Returns ``None`` when there's no file to load or the adata is a normal
        (non-shell) save — in which case the caller lets scvi read it. When the
        saved ``X`` is an emptied shell (counts stripped at save time) with no
        stored latents and no minify registry, we inject empty
        ``_scvi_latent_qzm``/``_scvi_latent_qzv`` slots so scvi's minified-adata
        guard passes. The adata is only a structural placeholder; run inference
        (``predict``/``get_latent_representation``) on your own data.
        """
        if not os.path.exists(shell_path):
            return None
        # Cheaply check whether X is an emptied shell before a full read.
        try:
            import h5py
            with h5py.File(shell_path, 'r') as f:
                x = f.get('X')
                is_shell = isinstance(x, h5py.Group) and 'data' in x and x['data'].shape[0] == 0
        except Exception:
            is_shell = False
        if not is_shell:
            return None
        adata = ad.read_h5ad(shell_path)
        has_latents = '_scvi_latent_qzm' in adata.obsm and '_scvi_latent_qzv' in adata.obsm
        if has_latents or _is_minified(adata):
            return adata  # already loadable; reuse to avoid a second read
        n = adata.n_obs
        adata.obsm['_scvi_latent_qzm'] = np.zeros((n, 0), dtype=np.float32)
        adata.obsm['_scvi_latent_qzv'] = np.zeros((n, 0), dtype=np.float32)
        log.warning(
            '[Load] Saved adata.X is an empty shell (counts stripped); injected '
            'empty latent qzm/qzv slots so the model loads. This adata is a '
            'placeholder — run predict()/get_latent_representation() on your own data.'
        )
        return adata

    @classmethod
    def load(cls, dir_path: str, use_latest_checkpoint: bool = True, load_state: bool = True,
             load_control_db: bool = False, **kwargs):
        # Resolve to the latest checkpoint directory if requested
        used_ckpt = False
        if use_latest_checkpoint:
            ckpt_path = io.get_latest_ckpt_dir(dir_path)
            if ckpt_path is not None:
                dir_path = ckpt_path
                log.info(f'[Load] Resolved to latest checkpoint: {dir_path}')
                used_ckpt = True
        # Look for final model directory
        else:
            dir_path = io.get_latest_ckpt_dir(dir_path, ckpt_key='**')
        # When no adata is supplied, scvi reads the adata.h5ad saved with the
        # model. Snapshots saved with counts stripped have an emptied X but no
        # stored latents and no minify registry, which trips scvi's
        # "non-minified model with minified adata" guard. We only need the
        # model for inference on new data (predict / get_latent_representation),
        # so load that shell ourselves and inject empty latent slots to satisfy
        # the guard. A caller-supplied adata is left untouched.
        _synth = False
        if kwargs.get('adata', None) is None:
            prefix = kwargs.get('prefix', None)
            shell_path = os.path.join(dir_path, f'{prefix}adata.h5ad' if prefix else 'adata.h5ad')
            shell = cls._load_shell_adata_for_inference(shell_path)
            if shell is None:
                # No adata supplied and none saved beside the model: build one from the registry.
                # Everything inference needs is already in model.pt, so requiring the TRAINING data
                # just to call predict() was an accident of setup reading obs, not a real dependency.
                shell = cls._synthesize_registry_adata(dir_path)
                _synth = shell is not None
            if shell is not None:
                kwargs['adata'] = shell
        # Point setup_anndata at the class embedding saved next to the model so a
        # stale (machine-specific) class_emb_path in the registry is ignored.
        # Setup args of removed setup_anndata inputs recorded by older checkpoints
        # (edge_graph, deg_*, gene_emb_varm_key, ...) are replayed by scvi into
        # setup_anndata(**kwargs) and passed on as (ignored) field-transfer kwargs.
        cls._load_class_emb_dir = dir_path
        # Older checkpoints record __init__ kwargs that the current schema has since
        # dropped; scvi replays them verbatim into cls(...), so without this they raise
        # `unexpected keyword argument`. The flag makes __init__ filter them (and log
        # every drop). Scoped to this call so ordinary construction still validates.
        cls._loading_ckpt = True
        cls._skip_control_db = not bool(load_control_db)
        try:
            # Load latest model
            model = super().load(dir_path, **kwargs)
        finally:
            cls._load_class_emb_dir = None
            cls._loading_ckpt = False
            cls._skip_control_db = False
        # Mark checkpoint usage in model
        model._from_checkpoint = used_ckpt
        # INFERENCE ONLY when the registry shell was synthesized: that shell holds one placeholder
        # row per category and no real cells, so record it rather than let a resumed run look healthy.
        model._registry_synthesized = bool(_synth)

        # Restore training state if saved alongside the model
        state_path = os.path.join(dir_path, cls._TRAINING_STATE_FNAME)
        if load_state and os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            prev_stage = state.get('_training_stage', 0)
            model._version = state.get('_version', 0)
            model._training_stage = prev_stage + 1
            model.was_pretrained = True
            model._resume_training_state = {
                'optimizer_state_dict': state.get('optimizer_state_dict'),
                'scheduler_state_dict': state.get('scheduler_state_dict'),
                'epoch': state.get('epoch', 0),
                'epochs_completed': state.get('epochs_completed'),
                'global_step': state.get('global_step', 0),
                'optimizer_steps': state.get('optimizer_steps'),
                'rng_state': state.get('rng_state'),
                'module_state': state.get('module_state'),
                'plan_state': state.get('plan_state'),
            }
            # Restore split indices
            for key, attr in [('train', 'train_indices'), ('val', 'validation_indices'),
                              ('test', 'test_indices'), ('ood_test', 'ood_test_indices')]:
                idx = state.get('split_indices', {}).get(key)
                if idx is not None:
                    setattr(model, attr, np.asarray(idx))
            # Restore version_dir and setup logger for the next training stage
            saved_version_dir = state.get('model_log_dir')
            if saved_version_dir is not None:
                import lightning.pytorch as _pl
                model.model_log_dir = saved_version_dir
                model._ckpt_dir = os.path.join(saved_version_dir, 'checkpoints')
                model._logger = _pl.loggers.TensorBoardLogger(
                    save_dir=saved_version_dir, name='', version='',
                )
            epoch = model._resume_training_state['epoch']
            step = model._resume_training_state['global_step']
            log.info(f'[Load] Restored training run {prev_stage} → run {model._training_stage} (epoch {epoch}, step {step})')
        return model

    @staticmethod
    def _rng_snapshot() -> dict:
        import random as _random
        rng = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': _random.getstate()}
        if torch.cuda.is_available():
            rng['cuda'] = torch.cuda.get_rng_state_all()
        return rng

    def save(self, *args, **kwargs) -> None:
        """Save model weights, embeddings, and training state for full resume."""
        keep_emb = bool(kwargs.pop('keep_emb', None))
        # Handle external model embeddings if anndata is saved with model
        popped: dict = {}
        if self.cls_emb is not None and bool(kwargs.get('save_anndata', None)) and not keep_emb:
            for k in (REGISTRY_KEYS.CLS_EMB_KEY, REGISTRY_KEYS.CLS_SIM_KEY):
                if k in self.adata.uns:
                    popped[k] = self.adata.uns.pop(k)
        # Save model as scvi normally does (model.pt with state_dict + registry)
        try:
            super().save(*args, **kwargs)
        finally:
            # Restore any temporarily-popped uns entries so the in-memory model
            # remains usable regardless of whether save succeeded.
            for k, v in popped.items():
                self.adata.uns[k] = v
        # Save training state alongside the model for full resume
        dir_path = args[0] if args else kwargs.get('dir_path')
        if dir_path is not None:
            # Persist the class/gene embedding next to the model so it can be
            # re-loaded without depending on the original (often machine-
            # specific) ``class_emb_path`` baked into the saved setup args.
            if self.cls_emb is not None:
                emb_df, _ = self.get_cls_emb(use_full_cls_emb=True, return_dataframe=True)
                if emb_df is not None:
                    emb_fpath = os.path.join(dir_path, self._GENE_EMB_FNAME)
                    np.savez(
                        emb_fpath,
                        embeddings=emb_df.to_numpy(dtype=np.float32),
                        genes=emb_df.index.to_numpy(),
                    )
                    log.info(f'[Save] Wrote class embedding to {emb_fpath}.')
            state = {
                'model_log_dir': getattr(self, 'model_log_dir', None),
                '_training_stage': getattr(self, '_training_stage', 0),
                '_version': getattr(self, '_version', None),
                'split_indices': {
                    'train': getattr(self, 'train_indices', None),
                    'val': getattr(self, 'validation_indices', None),
                    'test': getattr(self, 'test_indices', None),
                    'ood_test': getattr(self, 'ood_test_indices', None),
                },
            }
            # Include optimizer/scheduler state if training plan is available
            training_plan = getattr(self, 'training_plan', None)
            if training_plan is not None:
                optim = training_plan.optimizers()
                optim = getattr(optim, 'optimizer', optim)    # the real optimizer, not the Lightning wrapper
                if optim is not None:
                    state['optimizer_state_dict'] = optim.state_dict()
                schedulers = training_plan.lr_schedulers()
                if schedulers is not None:
                    sched = schedulers if not isinstance(schedulers, list) else schedulers[0]
                    if hasattr(sched, 'scheduler'):
                        sched = sched['scheduler'] if isinstance(sched, dict) else sched.scheduler
                    state['scheduler_state_dict'] = sched.state_dict()
                state['epoch'] = training_plan.current_epoch
                # EPOCHS COMPLETED, unambiguously. `current_epoch` is the INDEX of the running epoch while a
                # Lightning checkpoint is written at an epoch's end (SaveCheckPointWithIndices sets
                # _saving_at_epoch_end), but already the COUNT once fit() has returned (the final save).
                # Restoring the index as a count re-ran the saved epoch (a 1-epoch checkpoint restarted at 0).
                _at_end = bool(getattr(self, '_saving_at_epoch_end', False))
                state['epochs_completed'] = int(training_plan.current_epoch) + (1 if _at_end else 0)
                # Both step counters: optimizer steps (LightningModule.global_step, what step-keyed code
                # reads) and the logger's x-axis counter (_batches_that_stepped).
                _tr = getattr(training_plan, 'trainer', None)
                _bts = getattr(getattr(getattr(_tr, 'fit_loop', None), 'epoch_loop', None),
                               '_batches_that_stepped', None)
                state['global_step'] = (int(_bts) if _bts is not None
                                        else int(training_plan.global_step))
                state['optimizer_steps'] = int(training_plan.global_step)
            # IN-MEMORY TRAINING STATE that a fresh process would otherwise start empty: the module's prior-refit
            # bookkeeping and the training plan's carried-over loss/KL kwargs and per-epoch caches. Without them a
            # resumed run diverges at its first step although weights, buffers, optimizer and RNG are identical.
            if training_plan is not None:
                import copy as _copy
                _mod = getattr(training_plan, 'module', None)
                state['module_state'] = {k: _copy.deepcopy(getattr(_mod, k)) for k in self._RESUME_MODULE_ATTRS
                                         if _mod is not None and hasattr(_mod, k)}
                state['plan_state'] = {k: _copy.deepcopy(getattr(training_plan, k)) for k in self._RESUME_PLAN_ATTRS
                                       if hasattr(training_plan, k)}
            # RNG STREAMS, so a resumed run draws the same batches, dropout masks and permutations as an
            # uninterrupted one: the balanced sampler draws from the global torch RNG (numpy's above 2**24
            # cells), dropout and the prior refit from torch.
            # After train() the streams are those of the END OF FITTING, not of now: train() itself draws after
            # fitting (latent registration), and so may anything the caller runs before saving.
            state['rng_state'] = getattr(self, '_fit_end_rng', None) or self._rng_snapshot()
            torch.save(state, os.path.join(dir_path, self._TRAINING_STATE_FNAME))
    

    @staticmethod
    def _load_class_embedding_df(path: str | os.PathLike) -> pd.DataFrame:
        """Load a class-embedding DataFrame from disk for A/B comparison.

        Expected layout: rows indexed by perturbation / class names, columns
        are embedding dimensions. Supported by suffix:
          - .npz                        → numpy.load (keys 'embeddings', 'genes')
          - .pkl / .pickle              → pandas.read_pickle
          - .parquet / .pq              → pandas.read_parquet
          - .feather                    → pandas.read_feather
          - .csv                        → pandas.read_csv (first column is index)
          - .tsv / .txt                 → pandas.read_csv (tab-separated)
          - .h5 / .hdf / .hdf5          → pandas.read_hdf (uses 'cls_embedding' key)

        ``.npz`` is the preferred format: numpy's own format survives the
        numpy 1.x ↔ 2.x pickle module-path rename (``numpy.core.numeric`` →
        ``numpy._core.numeric``) that breaks cross-env pickle loads.
        """
        p = str(path)
        suffix = os.path.splitext(p)[1].lower()
        if suffix == '.npz':
            data = np.load(p, allow_pickle=True)
            if 'embeddings' not in data or 'genes' not in data:
                raise ValueError(
                    f"Class embedding .npz at {p} must contain arrays 'embeddings' "
                    f"and 'genes'; got keys {list(data.keys())}."
                )
            df = pd.DataFrame(data['embeddings'], index=list(data['genes']))
        elif suffix in ('.pkl', '.pickle'):
            df = pd.read_pickle(p)
        elif suffix in ('.parquet', '.pq'):
            df = pd.read_parquet(p)
        elif suffix == '.feather':
            df = pd.read_feather(p)
            # feather drops the index by default — first col is treated as index.
            if df.columns[0] in ('index', 'gene', 'label'):
                df = df.set_index(df.columns[0])
        elif suffix == '.csv':
            df = pd.read_csv(p, index_col=0)
        elif suffix in ('.tsv', '.txt'):
            df = pd.read_csv(p, sep='\t', index_col=0)
        elif suffix in ('.h5', '.hdf', '.hdf5'):
            df = pd.read_hdf(p, key='cls_embedding')
        else:
            raise ValueError(
                f'Unsupported class-embedding file extension {suffix!r}. '
                f'Use one of: .pkl/.pickle, .parquet/.pq, .feather, .csv, .tsv/.txt, .h5/.hdf/.hdf5.'
            )
        # Accept dict-style pickles too (common shape from precomputed runs):
        # {label: 1D array-like}. Convert to DataFrame indexed by label.
        if isinstance(df, dict):
            if not df:
                raise ValueError(f'Class embedding dict at {p} is empty.')
            sample_key = next(iter(df))
            sample_val = df[sample_key]
            try:
                arr = np.asarray(sample_val)
            except Exception as e:
                raise TypeError(
                    f'Class embedding dict at {p}: values must be array-like; '
                    f'got {type(sample_val).__name__} for key {sample_key!r} ({e}).'
                )
            if arr.ndim != 1:
                raise ValueError(
                    f'Class embedding dict at {p}: each value must be a 1D vector; '
                    f'got shape {arr.shape} for key {sample_key!r}.'
                )
            df = pd.DataFrame.from_dict(df, orient='index')
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'Loaded class embedding from {p} is {type(df).__name__}, expected DataFrame or dict.')
        if df.shape[1] < 2:
            raise ValueError(f'Class embedding at {p} has only {df.shape[1]} column(s); need ≥2 dims.')
        # Normalize columns to ``dim_<i>`` strings unconditionally. Column
        # names don't carry information downstream (we only use the matrix
        # values + the row index for label alignment), and standardizing
        # avoids dtype hiccups we've hit before (int columns, tuple columns
        # from MultiIndex flatten, h5ad string-coercion warnings, …).
        df = df.copy()
        df.columns = [f'dim_{i}' for i in range(df.shape[1])]
        return df

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        class_emb_uns_key: str | None = 'cls_embedding',
        class_emb_path: str | os.PathLike | None = None,
        control_adata_path: str | os.PathLike | None = None,   # ctrl.h5ad of control cells -> per-dataset gene means for encoder input centering
        size_factor_key: str | None = None,
        efficiency_key: str | None = None,
        cast_to_csr: bool = True,
        raw_counts: bool = True,
        check_cat_cols: bool = True,
        layer: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Setup AnnData object for training an ExPert model.
        """
        if isinstance(categorical_covariate_keys, str):
            categorical_covariate_keys = [categorical_covariate_keys]
        if isinstance(continuous_covariate_keys, str):
            continuous_covariate_keys = [continuous_covariate_keys]
        if not isinstance(adata.X, sp.csr_matrix) and cast_to_csr and not adata.isbacked:
            log.info('[Registry] Converting adata.X to csr matrix to boost training efficiency.')
            adata.X = sp.csr_matrix(adata.X)
        # Warn if sparse count data is stored in a larger dtype than necessary
        if raw_counts and sp.issparse(adata.X) and not adata.isbacked and adata.X.nnz > 0:
            if adata.X.dtype.itemsize > 2:
                sample = adata.X.data[:min(100_000, adata.X.nnz)]
                if np.allclose(sample, np.round(sample), atol=1e-6):
                    max_val = adata.X.data.max()
                    suggested = 'int16' if max_val <= 32767 else 'int32'
                    log.warning(
                        f'[Registry] adata.X contains counts stored as {adata.X.dtype} '
                        f'(max={max_val:.0f}). Consider downcasting before setup_anndata with: '
                        f'adata.X.data = adata.X.data.astype(np.{suggested})'
                    )
        if check_cat_cols:
            io.check_adata_cat_cols(adata)
        # Optional: override the class embedding in-place from disk before
        # registry. Lets you A/B different precomputed embeddings without
        # rebuilding the adata. Validated to cover every observed training
        # label so the existing alignment in _setup_class_embeddings still
        # raises a clean error if anything's missing.
        # On load, default to the embedding saved next to the model. This
        # overrides a stale ``class_emb_path`` (e.g. an absolute path from the
        # training machine) and covers the case where it was never set.
        if cls._load_class_emb_dir is not None:
            local_emb = os.path.join(cls._load_class_emb_dir, cls._GENE_EMB_FNAME)
            if os.path.exists(local_emb) and (
                class_emb_path is None or not os.path.exists(str(class_emb_path))
            ):
                if class_emb_path is not None:
                    log.warning(
                        f'[Registry] Saved class_emb_path {class_emb_path!r} not found; '
                        f'falling back to {local_emb}.'
                    )
                else:
                    log.info(f'[Registry] Defaulting class embedding to {local_emb}.')
                class_emb_path = local_emb
        if class_emb_path is not None:
            ext_df = cls._load_class_embedding_df(class_emb_path)
            if labels_key is None or labels_key not in adata.obs:
                raise ValueError(
                    f'[Registry] class_emb_path requires a valid labels_key; got {labels_key!r}.'
                )
            obs_labels = pd.Index(pd.unique(adata.obs[labels_key].astype(str)))
            missing = obs_labels.difference(ext_df.index.astype(str))
            if len(missing) > 0:
                preview = ', '.join(map(str, missing[:5]))
                more = '' if len(missing) <= 5 else f', … (+{len(missing) - 5} more)'
                raise ValueError(
                    f'[Registry] class_emb file at {class_emb_path} is missing '
                    f'{len(missing)}/{len(obs_labels)} observed labels: {preview}{more}'
                )
            ext_df.index = ext_df.index.astype(str)
            target_key = class_emb_uns_key or 'cls_embedding'
            adata.uns[target_key] = ext_df
            log.info(
                f'[Registry] Overrode adata.uns[{target_key}] with {ext_df.shape[0]} × '
                f'{ext_df.shape[1]} embedding from {class_emb_path}.'
            )
            class_emb_uns_key = target_key  # ensure downstream registration picks it up
        setup_method_args = cls._get_setup_method_args(**locals())
        
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=raw_counts),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            NumericalObsField(REGISTRY_KEYS.CLS_EFF_KEY, efficiency_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # Add class embedding matrix with shape (n_labels, class_emb_dim)
        if class_emb_uns_key is not None and class_emb_uns_key in adata.uns:
            log.info(f'[Registry] Registered class embedding from adata.uns[{class_emb_uns_key}].')
            anndata_fields.append(StringUnsField(REGISTRY_KEYS.CLS_EMB_KEY, class_emb_uns_key))
        # Register latent fields of adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        # Create AnnData manager
        adata_manager = EmbAnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
