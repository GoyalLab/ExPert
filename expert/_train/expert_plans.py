# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import pandas as pd
import numpy as np
import torch
import torchmetrics.functional as tmf

from expert.utils.constants import MODULE_KEYS, REGISTRY_KEYS, PREDICTION_KEYS
from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass, LossOutput
from scvi.train._metrics import ElboMetric
from scvi.train._constants import METRIC_KEYS

from typing import Literal
from collections import defaultdict

import logging


def _sigmoid_schedule(t, T, k):
    if T == 0:
        return 0
    """Normalized sigmoid: 0 at t=0, 1 at t=T."""
    midpoint = T / 2
    raw = 1 / (1 + np.exp(-k * (t - midpoint) / T))
    min_val = 1 / (1 + np.exp(k / 2))       # sigmoid(0)
    max_val = 1 / (1 + np.exp(-k / 2))      # sigmoid(T)
    return (raw - min_val) / (max_val - min_val)

def _exponential_schedule(t, T, k):
    if T == 0:
        return 0
    """Normalized exponential: 0 at t=0, 1 at t=T."""
    raw = 1 - np.exp(-k * t / T)
    max_val = 1 - np.exp(-k)
    return raw / max_val


class ContrastiveSupervisedTrainingPlan(TrainingPlan):
    """Lightning module task for Contrastive Supervised Training.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    n_classes
        The number of classes in the labeled dataset.
    classification_ratio
        Weight of the classification_loss in loss function
    lr
        Learning rate used for optimization :class:`~torch.optim.Adam`.
    weight_decay
        Weight decay used in :class:`~torch.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        n_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_warmup: int | None = None,
        n_epochs_warmup: int | None = 400,
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation", "validation_loss"
        ] = "elbo_validation",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        optimizer_cls: torch.optim.Optimizer = torch.optim.AdamW,
        average: str = "macro",
        top_k: int = 1,
        use_posterior_mean: Literal["train", "val", "both"] | None = "val",
        log_class_distribution: bool = True,
        anneal_schedules: dict[str, dict[str | float]] | None = None,
        kl_stage_offset: int = 0,
        batch_labels: np.ndarray | None = None,
        cov_labels: np.ndarray | None = None,
        # Full log options
        log_full: list[str] | None = ['val'],
        full_log_every_n_epoch: int = 1,
        # Per-dataset metrics
        live_macro_exclude: list[str] | tuple[str, ...] = (),
        grp_min_cells: int = 5,        # min cells per (dataset,class) group for the readout proxy.
        # 5, NOT 15: eval splits are capped by `eval_cap_per_group` (shipped default 5), so a
        # threshold of 15 can NEVER be met and the pseudobulk monitor emits its empty sentinel
        # (grp_n=0) on every run -- which is what left every arm checkpointing on the per-cell
        # macro that correlates +0.22 / -0.51 with the shipped readout.
        per_ds_min_cells: int = 25,
        # Plotting options
        plot_umap: bool = False,
        plot_umap_key: str | list[str] = 'z_shared',
        plot_f1_dist: bool = True,
        plot_cm: bool = True,
        plot_clip_geom: bool = True,
        plot_recall_at_k: bool = True,
        plot_cat_emb: bool = True,
        log_histograms: bool = True,                     # extra-output histograms (throttled to full_log_every_n_epoch)
        plot_every_n_epochs: int = 10,
        # Caching options
        n_max_cache: int = 10_000,
        save_cache: bool = False,
        use_cosine_restart: bool = False,
        # CosineAnnealingWarmRestarts shape (epoch interval). Defaults are the previously hard-coded values, so
        # every existing arm is unchanged. A single decay over N resumed epochs: T0=N, Tmult=1, eta_min ~0.
        cosine_T0: int = 15,
        cosine_Tmult: int = 2,
        cosine_eta_min: float = 5e-6,
        **loss_kwargs,
    ):
        self.optimizer_cls = optimizer_cls
        self.use_cosine_restart = use_cosine_restart
        self.cosine_T0, self.cosine_Tmult, self.cosine_eta_min = int(cosine_T0), int(cosine_Tmult), float(cosine_eta_min)
        # Custom optimizer creator (one named 'default' param group)
        optimizer, optimizer_fn = self._optimizer_creator_fn_custom()
        if optimizer == 'Custom':
            loss_kwargs['optimizer'] = optimizer
            loss_kwargs['optimizer_creator'] = optimizer_fn
        # Init parent
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_warmup,
            n_epochs_kl_warmup=n_epochs_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            compile=compile,
            compile_kwargs=compile_kwargs,
            **loss_kwargs,
        )
        # Save annealing schedules and default params
        self.anneal_schedules = {
            'rl_weight': {'weight': 1.0},
            'kl_weight': {'min_weight': 0.0, 'max_weight': 1.0}
        }
        self.anneal_schedules.update(anneal_schedules)
        # CLIP alignment is removed: the prototype logits are only scored (proto_ce is a diagnostic), so a
        # cls_align_weight schedule may be present (it is still logged as a schedule weight) but must stay 0.
        _ca = self.anneal_schedules.get('cls_align_weight')
        if _ca is not None and any(
                float((_ca if not isinstance(_ca, dict) else _ca.get(k, 0)) or 0) != 0.0
                for k in ('weight', 'max_weight', 'min_weight', 'post_min_weight')):
            raise ValueError(f'anneal_schedules.cls_align_weight={_ca}: CLIP alignment has been removed; '
                             'its weight must be 0.')
        self.n_epochs_warmup = n_epochs_warmup
        self.n_steps_warmup = n_steps_warmup
        self.kl_stage_offset = kl_stage_offset
        self.kl_reset_epochs = self._get_stall_epochs()
        # DEDUP: multiple alignment terms sharing a stall epoch must count as ONE kl stage, not N. Without the
        # set(), e.g. 4 terms stalled to ep10 give reset_at=[0,10,10,10,10] -> 4 stages -> kl_stage_offset*4 pins
        # kl_weight near 0 for ~130 epochs (the KL-warmup-explosion precondition). (Also gated by kl_stage_offset>0.)
        self.reset_kl_at = np.array(sorted(set(self.kl_reset_epochs.values())))
        self.kl_kwargs = self.anneal_schedules.get('kl_weight')
        # CLS params
        self.n_classes = n_classes
        self.use_posterior_mean = use_posterior_mean if use_posterior_mean is not None else "none"
        # Contrastive params
        self.log_class_distribution = log_class_distribution

        # Misc
        self.average = average                                                      # How to reduce the classification metrics
        self.top_k = top_k                                                          # How many top predictions to consider for metrics
        # Plotting options
        self.plot_umap = plot_umap
        self.plot_umap_key = plot_umap_key if isinstance(plot_umap_key, list) else [plot_umap_key]
        self.plot_f1_dist = plot_f1_dist
        self.plot_cm = plot_cm
        self.plot_clip_geom = plot_clip_geom
        self.plot_recall_at_k = plot_recall_at_k
        self.plot_cat_emb = plot_cat_emb
        self.log_histograms = log_histograms
        # Throttle latch: maps split mode -> last epoch its histograms were written,
        # so the otherwise per-step / per-batch histogram calls fire at most once
        # per epoch (and only every full_log_every_n_epoch).
        self._hist_logged_epoch = {}
        self.log_full = log_full
        self.full_log_every_n_epoch = full_log_every_n_epoch
        self.plot_every_n_epochs = plot_every_n_epochs
        self.batch_labels = batch_labels
        self.cov_labels = cov_labels
        # Dataset-name PREFIXES excluded from the LIVE macro (the Marson donors replicate each
        # other, so pooling them inflates the dataset-mean).
        # ACCEPT BOTH NESTINGS, AND REFUSE A BARE STRING. This key has bitten twice:
        #   * a bare 'Marson' is ITERATED PER CHARACTER, so the exclusion silently became
        #     ('M','a','r',...) and every dataset starting with 'M' was dropped -- which is what
        #     collapsed v0.1.4's live monitor to 0.0 for a whole run.
        #   * the fix for that, writing [['Marson']], is required on the create_sweep (run.sh)
        #     path because create_sweep collapses every list to its first element -- but
        #     `expert.tune.run --config` does NOT run create_sweep, so the same file arrives here
        #     still nested and `n.startswith(['Marson'])` raises TypeError at the first validation,
        #     after training has already burned an epoch. Same dual-entry-point problem that
        #     class_emb_extra_paths solves by flattening in _expert.py; solved the same way.
        _lme = live_macro_exclude or ()
        if isinstance(_lme, str):
            _lme = (_lme,)
        _flat = []
        for _p in _lme:
            _flat.extend(_p if isinstance(_p, (list, tuple)) else [_p])
        self.live_macro_exclude = tuple(str(x) for x in _flat)
        self.grp_min_cells = int(grp_min_cells)
        self.per_ds_min_cells = per_ds_min_cells
        self._excluded_ctx = None
        # Save classes that have been seen during training
        self.train_class_counts = []
        self.class_batch_counts = defaultdict(list)
        self.observed_classes = set()
        # ----- Cache -----
        self.cache = {}
        self.n_max_cache = n_max_cache
        self.save_cache = save_cache
        # Cache for embeddings and metadata within epoch
        self.epoch_cache = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list),
            'ood_test': defaultdict(list),
        }

        # Setup test / ood_test metrics. Both routed through validation_step
        # via separate ElboMetric collections so scvi's ``elbo_{mode}`` key
        # lookup resolves for each split prefix.
        self._n_obs_test = None
        self.initialize_test_metrics()
        self.initialize_ood_test_metrics()
        
    def _cos_lr_scheduler(self, opt: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=getattr(self, "cosine_T0", 15),
            T_mult=getattr(self, "cosine_Tmult", 2),
            eta_min=getattr(self, "cosine_eta_min", 5e-6),
        )

    def _optimizer_creator_fn_custom(self):

        logging.info('Creating custom optimizer')

        def _creator(params):

            params_list = list(params)

            if not hasattr(self, "module"):
                return self.optimizer_cls(
                    params_list,
                    lr=self.lr,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )

            # One named group holding every trainable parameter (layout kept stable so optimizer state
            # saved by earlier runs still restores on resume).
            param_groups = [{"params": params_list, "lr": self.lr, "weight_decay": self.weight_decay,
                             "name": 'default'}]

            return self.optimizer_cls(
                param_groups,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

        return "Custom", _creator
    
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.module.parameters())

        optimizer = self.get_optimizer_creator()(params)

        config = {"optimizer": optimizer}

        # --------------------------------
        # ReduceLROnPlateau
        # --------------------------------
        if self.reduce_lr_on_plateau:

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )

            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": self.lr_scheduler_metric,
                "interval": "epoch",
            }

        # --------------------------------
        # Cosine Restart Scheduler
        # --------------------------------
        elif self.use_cosine_restart:

            scheduler = self._cos_lr_scheduler(optimizer)

            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return config
    
    @property
    def n_obs_test(self):
        """Number of observations in the tset set.

        This will update the loss kwargs for loss rescaling.

        Notes
        -----
        This can get set after initialization
        """
        return self._n_obs_test

    @n_obs_test.setter
    def n_obs_test(self, n_obs: int):
        if "n_obs" in self._loss_args:
            self.loss_kwargs.update({"n_obs": n_obs})
        self._n_obs_test = n_obs
        self.initialize_test_metrics()
        self.initialize_ood_test_metrics()

    def initialize_test_metrics(self):
        """Initialize test related metrics."""
        (
            self.elbo_test,
            self.rec_loss_test,
            self.kl_local_test,
            self.kl_global_test,
            self.test_metrics,
        ) = self._create_elbo_metric_components(mode="test", n_total=self.n_obs_test)
        self.elbo_test.reset()

    def initialize_ood_test_metrics(self):
        """Initialize ood_test metrics. scvi's ``compute_and_log_metrics``
        indexes the collection by ``elbo_{mode}`` — so the ood_test prefix
        needs its own ElboMetric collection with that key. Uses the same
        ``n_obs_test`` as the in-distribution test collection (currently
        unwired in this project, so both fall back to n=1)."""
        (
            self.elbo_ood_test,
            self.rec_loss_ood_test,
            self.kl_local_ood_test,
            self.kl_global_ood_test,
            self.ood_test_metrics,
        ) = self._create_elbo_metric_components(mode="ood_test", n_total=self.n_obs_test)
        self.elbo_ood_test.reset()

    def _cache_step_data(self, mode: str, batch: dict, inference_outputs: dict, loss_output: LossOutput):
        """Cache data from a single step.

        Per-context cap so each dataset gets a fair slice of the cache,
        rather than the first dataset filling it (val/test loaders run
        with shuffle=False, so cell order is dataset-clustered if
        val_idx happens to be sorted by adata position).
        """
        # Only cache every n epoch
        if self.full_log_every_n_epoch <= 0 or (self.current_epoch + 1) % self.full_log_every_n_epoch != 0:
            return
        cache = self.epoch_cache[mode]

        # Per-context cap: each dataset gets up to n_max_cache / n_ctx cells.
        # n_ctx IS THE NUMBER OF DATASETS PRESENT IN *THIS SPLIT*, not the registry total. Dividing
        # by the registry total starves any split that holds only a few datasets: on fold_4
        # ood_test is the 2 held-out K562 sets, so at n_batch=22 it received 2 x (50000//22) =
        # 4,544 of its 100,610 cells while 20 dataset slots went permanently unused. Spread over
        # 8,964 classes that is <1 cell per (dataset,class) group, so NO group reached
        # grp_min_cells and every ood_test_grp_* metric emitted its sentinel (-1) for the whole
        # run. val was not healthy either, just non-zero: 55 groups out of 893k cells.
        # Counting only observed contexts gives ood_test 50000//2 = 25,000 per dataset at the SAME
        # memory budget. Small-DB arms were unaffected (200 classes -> ~45 cells/group), which is
        # why this only ever showed up at full scale.
        # CONTEXT COUNT MUST NOT BE THE RUNNING TALLY. `cache` is .clear()ed every epoch, so a
        # per-epoch `_ctx_seen` restarts empty and n_ctx grows 1, 2, ... as batches arrive. val/test
        # loaders run shuffle=False and are DATASET-CLUSTERED, so the first batches are all one
        # dataset: n_ctx=1 -> per_ctx_cap = n_max_cache // 1 = the WHOLE budget, that dataset fills
        # it, and the total guard below then rejects every later dataset. Measured at full scale in
        # v0.1.4: the val cache held 2 datasets (ds_val/* = 6 tags) where the pre-group-aware cache
        # held 20 -- and because both survivors happened to start with 'M', live_macro_exclude
        # (a bare str, iterated per CHARACTER) excluded both and val_live_macro_top10_recall_A
        # logged 0.0 for the entire run, freezing save_top_k on epochs 0-2.
        #
        # The split's dataset composition is IDENTICAL every epoch, so learn it once and reuse it.
        # Until it is known, divide by the REGISTRY total: fair to every dataset, and at worst
        # starves a narrow split for one epoch (the pre-fix behaviour) instead of collapsing a wide
        # one for the whole run.
        _seen_all = getattr(self, '_ctx_seen_ever', None)
        if _seen_all is None:
            _seen_all = {}
            self._ctx_seen_ever = _seen_all
        _seen = _seen_all.setdefault(mode, set())
        _ctx_pre = inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY].detach().cpu().flatten().long()
        _seen.update(int(v) for v in torch.unique(_ctx_pre).tolist())
        _learned = getattr(self, '_ctx_n_learned', {}).get(mode)
        n_ctx = int(_learned) if _learned else max(1, int(getattr(self.module, 'n_batch', 1) or 1))
        per_ctx_cap = max(1, self.n_max_cache // n_ctx)
        # Hard total guard: never let the cache exceed the budget it was given.
        if int(cache.get('n', 0)) >= self.n_max_cache:
            return
        _n_ctx_buf = max(n_ctx, int(getattr(self.module, 'n_batch', 1) or 1))
        ctx_counts = cache.get('_ctx_counts', None)
        if ctx_counts is None:
            ctx_counts = torch.zeros(_n_ctx_buf, dtype=torch.long)
            cache['_ctx_counts'] = ctx_counts

        # Per-cell context (cpu) — used to build the keep mask
        ctx_arr = inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY].detach().cpu().flatten().long()
        keep = torch.zeros_like(ctx_arr, dtype=torch.bool)

        # GROUP-AWARE FILL. Taking the FIRST cells of each dataset would spread the budget over
        # every (dataset,class) group it happens to meet: at 8,964 classes that is ~0.25 cells per
        # group, so almost no group reaches grp_min_cells and the pseudobulk grp_* readout emits
        # its sentinel. Filling each group to grp_min_cells instead turns the SAME budget into
        # complete groups (50k cells -> ~10k groups at cap 5).
        _lab_t = inference_outputs.get(MODULE_KEYS.LABEL_KEY, None)
        _n_lab_c = int(getattr(self.module, 'n_labels', 0) or 0)
        if _lab_t is not None and _n_lab_c > 0:
            lab_arr = _lab_t.detach().cpu().flatten().long()
            gcap = max(1, int(getattr(self, 'grp_min_cells', 5)))
            gcounts = cache.get('_grp_counts', None)
            if gcounts is None:
                gcounts = torch.zeros(_n_ctx_buf * _n_lab_c, dtype=torch.int32)
                cache['_grp_counts'] = gcounts
            ok_lab = (lab_arr >= 0) & (lab_arr < _n_lab_c)
            gid = ctx_arr.clamp(0, _n_ctx_buf - 1) * _n_lab_c + lab_arr.clamp(0, _n_lab_c - 1)
            # GROUP QUOTA -- the part that makes this work. Admitting the first cell of every new
            # group spreads a 2,272-cell dataset budget over ~2,272 distinct groups at one cell
            # each and completes NONE. A budget of B fills at most B // gcap groups, so cap the
            # number of admitted groups at exactly that and feed only those. 2272 // 5 = 454
            # complete groups per dataset instead of 2,272 singletons.
            # rank of each cell within its group IN THIS BATCH (stable, vectorised)
            srt = torch.argsort(gid, stable=True)
            gs = gid[srt]
            first = torch.ones(gs.numel(), dtype=torch.bool)
            if gs.numel() > 1:
                first[1:] = gs[1:] != gs[:-1]
            ordinal = torch.cumsum(first.long(), 0) - 1
            starts = torch.zeros(int(ordinal[-1].item()) + 1, dtype=torch.long)
            starts.scatter_(0, ordinal[first], torch.arange(gs.numel())[first])
            rank = torch.empty_like(gid)
            rank[srt] = torch.arange(gs.numel()) - starts[ordinal]
            quota = max(1, per_ctx_cap // gcap)
            admitted = cache.get('_grp_admitted', None)
            if admitted is None:
                admitted = {}                      # dict, NOT a list: see _process_epoch_cache
                cache['_grp_admitted'] = admitted
            room = (gcounts[gid].long() + rank) < gcap
            cand = room & ok_lab
            # a cell is usable only if its group is already admitted, or its dataset still has
            # quota left to admit a new one (decided in first-seen order, which is arbitrary but
            # unbiased under the balanced loader)
            gid_l = gid.tolist()
            ctx_l = ctx_arr.tolist()
            cand_l = cand.tolist()
            for i in range(len(gid_l)):
                if not cand_l[i]:
                    continue
                c_i, g_i = ctx_l[i], gid_l[i]
                a = admitted.setdefault(c_i, set())
                if g_i not in a:
                    if len(a) < quota:
                        a.add(g_i)
                    else:
                        cand[i] = False
            # still respect the per-dataset budget
            # TWO PHASES. Phase 1 spends the budget on COMPLETING groups. Phase 2 tops the
            # dataset up with whatever is left over, so the cache never holds FEWER cells than the
            # default would. Without phase 2 this REGRESSES wherever groups are scarce: the small
            # DB has 200 classes against a quota of 2272//5 = 454, so every group is admitted and
            # capped at 5, caching 1,000 cells per dataset out of a 2,272 budget and making each
            # group mean noisier than the default's ~11 cells. Group-awareness must be a
            # PRIORITISATION, not a reduction.
            for c in torch.unique(ctx_arr):
                slot = int((per_ctx_cap - ctx_counts[c]).clamp(min=0).item())
                if slot <= 0:
                    continue
                ci = torch.where((ctx_arr == c) & cand)[0]           # phase 1: completes groups
                n_take = min(slot, ci.numel())
                if n_take > 0:
                    keep[ci[:n_take]] = True
                    ctx_counts[c] += n_take
                    slot -= n_take
                if slot > 0:                                          # phase 2: use the remainder
                    rest = torch.where((ctx_arr == c) & ~keep)[0]
                    n2 = min(slot, rest.numel())
                    if n2 > 0:
                        keep[rest[:n2]] = True
                        ctx_counts[c] += n2
            if bool(keep.any()):
                gcounts.scatter_add_(0, gid[keep],
                                     torch.ones(int(keep.sum()), dtype=torch.int32))
        if not bool(keep.any()):
            return
        keep_dev = keep.to(inference_outputs[MODULE_KEYS.Z_KEY].device)

        # Apply mask to a tensor field while moving to cpu
        def _slice(t: torch.Tensor) -> torch.Tensor:
            return t[keep_dev].detach().cpu()

        # Increment cumulative cache size
        cache['n'] = cache.get('n', 0) + int(keep.sum().item())
        # Core fields
        cache[MODULE_KEYS.Z_KEY].append(_slice(inference_outputs[MODULE_KEYS.Z_KEY]))
        cache[REGISTRY_KEYS.LABELS_KEY].append(_slice(inference_outputs[MODULE_KEYS.LABEL_KEY]))
        cache[REGISTRY_KEYS.BATCH_KEY].append(_slice(inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY]))
        # Logits (LossOutput.logits, fallback path)
        if loss_output.logits is not None:
            cache[PREDICTION_KEYS.PREDICTION_KEY].append(
                _slice(torch.argmax(loss_output.logits, dim=-1))
            )
        # Handle extra outputs
        extra_outputs = loss_output.extra_metrics.get("extra_outputs", {})
        # Read (do NOT pop) logits and store as sparse (B, n_labels) with only in-batch classes
        # filled. This dict is the SAME object compute_and_log_metrics reads later in the same
        # step, so popping here would blank proto_top*_recall for every batch that reaches this
        # line — the key must survive; the double-append it used to prevent is handled below.
        logits = extra_outputs.get(MODULE_KEYS.CLS_LOGITS_KEY, None)
        if logits is not None:
            labels = inference_outputs[MODULE_KEYS.LABEL_KEY].detach().flatten()
            n_labels = self.module.n_labels
            full_logits = torch.zeros(logits.size(0), n_labels, device=logits.device)
            unique_cls = labels.unique()
            if logits.size(1) == n_labels:
                full_logits = logits
            else:
                full_logits[:, unique_cls] = logits[:, :unique_cls.size(0)]
            cache[MODULE_KEYS.CLS_LOGITS_KEY].append(_slice(full_logits))
        # Automatically cache everything tensor-like, sliced to the kept cells
        for k, v in extra_outputs.items():
            # Already cached above as the densified (B, n_labels) form — re-appending the raw
            # (possibly in-batch-width) tensor under the same key would break the concat.
            if k == MODULE_KEYS.CLS_LOGITS_KEY:
                continue
            if v is None or not isinstance(v, torch.Tensor):
                continue
            # 0-d / non-per-cell tensors (e.g. global cls_proj) cache as-is
            if v.dim() == 0 or v.size(0) != ctx_arr.numel():
                cache[k].append(v.detach().cpu())
            else:
                cache[k].append(_slice(v))

    def _process_epoch_cache(self, mode: str) -> dict[str, np.ndarray]:
        """Process and clear the epoch cache."""
        import numpy as np
        import torch
        # Get mode cache from training plan
        cache = self.epoch_cache.get(mode, None)
        if not cache:
            return {}
        # Collect processed data
        processed = {}

        # keys that should be averaged instead of concatenated
        mean_keys = {
            MODULE_KEYS.CLS_PROJ_KEY,
        }
        # Collect each cache item
        for k, tensors in cache.items():
            # BOOKKEEPING KEYS ARE '_'-PREFIXED AND NEVER DATA. The old guard only rejected
            # non-lists, which held by luck: _ctx_counts/_grp_counts are tensors and _ctx_seen is a
            # set. A list-of-sets (_grp_admitted) sailed through it into torch.cat and killed the
            # run with "expected Tensor as element 0 in argument 0, but got set". Reject by name.
            if isinstance(k, str) and k.startswith('_'):
                continue
            if not isinstance(tensors, list) or len(tensors) == 0:
                continue
            # Mean aggregation
            if k in mean_keys:
                t = torch.stack(tensors, dim=0).mean(dim=0)
            # Concatenation
            else:
                t = torch.cat(tensors, dim=0)
            # Upcast floats: numpy has no bfloat16 dtype, so bf16 forward
            # outputs under autocast would fail .numpy() here. Integer
            # tensors (labels, batch IDs, predictions) pass through as-is.
            if t.is_floating_point():
                t = t.float()
            processed[k] = t.numpy()

            # squeeze labels/batch/preds
            if k in {
                REGISTRY_KEYS.LABELS_KEY,
                REGISTRY_KEYS.BATCH_KEY,
                PREDICTION_KEYS.PREDICTION_KEY,
            }:
                processed[k] = processed[k].squeeze()

        # Add split info
        processed[REGISTRY_KEYS.SPLIT_KEY] = np.repeat(
            mode, processed[REGISTRY_KEYS.LABELS_KEY].shape[0]
        )
        # LEARN THE SPLIT'S DATASET COUNT for the next epoch's per-context cap. `_ctx_seen_ever`
        # survives the clear below (it lives on self, not in the cache dict), so after one full
        # pass it holds every context in this split and the cap stops depending on arrival order.
        _seen_all = getattr(self, '_ctx_seen_ever', None)
        if _seen_all and _seen_all.get(mode):
            if not hasattr(self, '_ctx_n_learned'):
                self._ctx_n_learned = {}
            _n_now = len(_seen_all[mode])
            if self._ctx_n_learned.get(mode) != _n_now:
                self._ctx_n_learned[mode] = _n_now
                # module-level `logger` does NOT exist here -- only self.logger, which is the
                # Lightning logger and has no .info. Use the stdlib logger explicitly.
                logging.getLogger(__name__).info(
                    "[epoch-cache] %s: learned %d contexts in this split; per-context cap "
                    "is now %d cells", mode, _n_now, max(1, self.n_max_cache // _n_now))
        # Clear cache
        self.epoch_cache[mode].clear()
        return processed

    def _get_stall_epochs(self) -> dict[str, int]:
        stalls = {'kl': 0}
        for name, schedule in self.anneal_schedules.items():
            if name == 'kl_weight':
                continue
            stall = schedule.get('n_epochs_stall', 0)
            if stall > 0:
                stalls[name] = stall
        return stalls
    
    def _compute_weight(
        self,
        n_epochs_warmup: int | None,
        n_steps_warmup: int | None,
        n_epochs_stall: int | None = None,
        epoch: int | None = None,
        max_weight: float | None = 0.0,
        min_weight: float | None = 0.0,
        post_min_weight: float | None = None,
        schedule: Literal["linear", "sigmoid", "exponential"] = "sigmoid",
        anneal_k: float = 10.0,
        hard_stall: bool = True,
        invert: bool = False,
        weight: float | None = None,
        **kwargs
    ) -> float:
        # Return weight if it is already fixed
        if weight is not None:
            return weight
        # Pull current epoch
        if epoch is None:
            epoch = self.current_epoch
        
        # Set undefined weights to
        if min_weight is None:
            min_weight = 0.0
        if max_weight is None:
            max_weight = 0.0
        # Auto-invert if min > max
        if min_weight > max_weight:
            min_weight, max_weight = max_weight, min_weight
            invert = not invert

        # Apply stall
        if n_epochs_stall is not None and epoch < n_epochs_stall:
            # Disable loss during stall or return registered minimum weight
            if not invert:
                w = 0.0 if hard_stall else min_weight
            # Invert loss and return maximum weight during stall period
            else:
                w = max_weight
            return w

        # Reset epochs if stalling is enabled
        if n_epochs_stall:
            epoch -= n_epochs_stall
            n_epochs_warmup -= n_epochs_stall
        # Calculate slope for function
        slope = max_weight - min_weight
        # Calculate time points
        if n_epochs_warmup is not None:
            t = epoch
            T = n_epochs_warmup
        elif n_steps_warmup is not None:
            t = self.global_step
            T = n_steps_warmup
        else:
            return 0.0 if invert else max_weight
        # Set start point
        t = min(t, T)
        # Select schedule
        if schedule == "linear":
            w = t / T
        elif schedule == "sigmoid":
            w = _sigmoid_schedule(t, T, anneal_k)
        elif schedule == "exponential":
            w = _exponential_schedule(t, T, anneal_k)
        else:
            raise ValueError(f"Invalid schedule type: '{schedule}'")
        # Get final weight
        weight = min_weight + slope * w
        # Invert function if needed
        if invert:
            weight = max_weight - weight + min_weight

        # Reverse anneal if a new post warmup minimum is given
        if (
            post_min_weight is not None
            and self.n_epochs_warmup is not None
            and epoch >= n_epochs_warmup
            and self.n_epochs_warmup > n_epochs_warmup
        ):
            # progress from end of warmup → total duration
            t2 = epoch - n_epochs_warmup
            T2 = self.n_epochs_warmup - n_epochs_warmup
            if schedule == "linear":
                w2 = t2 / T2
            elif schedule == "sigmoid":
                w2 = _sigmoid_schedule(t2, T2, anneal_k)
            elif schedule == "exponential":
                w2 = _exponential_schedule(t2, T2, anneal_k)
            else:
                w2 = t2 / T2
            # decay from max_weight → post_min_weight
            weight = max_weight - (max_weight - post_min_weight) * w2
        # Return weight on epoch
        return weight

    def get_kl_weight(self):
        """KL weight with epoch offset reset at each stage."""
        kwargs = self.kl_kwargs
        reset_at = self.reset_kl_at
        
        # Compute cumulative offset from stages
        if reset_at.shape[0] > 1 and self.kl_stage_offset > 0:
            n_stages_passed = max((self.current_epoch > reset_at).sum() - 1, 0)
            kl_epoch_offset = n_stages_passed * self.kl_stage_offset
            epoch = max(self.current_epoch - kl_epoch_offset, 10)
        else:
            epoch = self.current_epoch
        # Shift effective epoch back
        kwargs['epoch'] = epoch
        kwargs['n_epochs_warmup'] = self.n_epochs_warmup
        kwargs['n_steps_warmup'] = None
        kwargs['n_epochs_stall'] = 0

        klw = self._compute_weight(**kwargs)
        
        return klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
    
    def _get_schedule_weight(self, key: str, default: float = 0.0) -> torch.Tensor:
        # Return default value if key is not in schedules
        if key not in self.anneal_schedules:
            return default
        # Return non-default weights
        if key == 'kl_weight':
            return self.get_kl_weight()
        # Init default args
        sched_kwargs = {'n_epochs_warmup': self.n_epochs_warmup, 'n_steps_warmup': self.n_steps_warmup}
        # Add scheduling params if given
        kwargs = {}
        kwargs.update(self.anneal_schedules.get(key, {}))
        # Return default value of the mode is not correct
        if bool(kwargs.pop('train_only', None)) and not self.training:
            return default
        sched_kwargs.update(kwargs)
        klw = self._compute_weight(**sched_kwargs)
        return (
            # float32: schedule math yields a numpy float64; torch.tensor(klw) would
            # build a float64 tensor that can't move to MPS (no float64 there). These
            # are scalar loss weights, so float32 is correct on every backend.
            klw if type(self).__name__ == "JaxTrainingPlan"
            else torch.tensor(float(klw), dtype=torch.float32).to(self.device)
        )
        
    @property
    def weights(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of all current weights."""
        return {k: self._get_schedule_weight(k) for k in self.anneal_schedules}
    
    # Substrings that mark per-cell index/label arrays — they bloat the
    # tensorboard event file (multi-GB) without being useful as histograms.
    # Substring-matched against extra_outputs keys (lowercased). Anything
    # matching is *skipped* by ``_log_extra_output_histograms``. These are
    # per-cell ID/categorical arrays that bloat TB logs without informing
    # training — their distributions are already summarized by other plots
    # (cm, class-count histograms in epoch/samples_per_*, etc.).
    _hist_blacklist: tuple[str, ...] = (
        'index', 'label', 'idx',         # cell / class / batch identifiers
        'prediction', 'pred',            # argmax outputs, per-cell ints
    )

    def _log_extra_output_histograms(self, mode: str, extra_outputs: dict, do_log: bool = True):
        """
        Log TensorBoard histograms for all non-scalar values in extra_outputs.

        Values are removed from the dictionary using pop() to avoid downstream issues.
        Keys whose name contains any substring in ``self._hist_blacklist`` are skipped.

        ``do_log=False`` still drains ``extra_outputs`` (the pop side-effect that
        downstream code relies on) but skips all CPU transfer and TensorBoard writes.
        Used to throttle these otherwise per-step / per-batch histograms.
        """
        import numpy as np
        import torch
        # Skip if no exra outputs are given
        if extra_outputs is None:
            return
        # Throttled off: drain the bulky arrays without any CPU transfer / TB writes.
        if not do_log:
            extra_outputs.clear()
            return
        # Get all extra keys
        keys = list(extra_outputs.keys())  # avoid dict size change during iteration
        # Log each key
        for k in keys:
            v = extra_outputs.pop(k, None)
            if v is None:
                continue
            # Drop bulky per-cell index/label arrays
            kl = k.lower() if isinstance(k, str) else ''
            if any(b in kl for b in self._hist_blacklist):
                continue
            # Convert tensors to numpy
            if isinstance(v, torch.Tensor):
                if v.ndim > 1:
                    continue
                # Skip integer / boolean dtypes — these are class / batch IDs
                # masquerading as histogram-worthy arrays. Cheap dynamic
                # check that catches any new int-valued extra_output added
                # in loss() without needing to extend the name blacklist.
                if not v.is_floating_point():
                    continue
                # Upcast: numpy has no bfloat16 dtype, so .numpy() rejects
                # bf16 tensors under autocast. float() is a no-op for fp32.
                v = v.detach().float().cpu().numpy()
            # Convert lists to numpy
            if isinstance(v, list):
                v = np.asarray(v)
            # Skip scalars
            if np.isscalar(v):
                continue
            # Skip empty
            if hasattr(v, "__len__") and len(v) == 0:
                continue
            try:
                self.logger.experiment.add_histogram(
                    f"{k}_hist_{mode}",
                    v,
                    global_step=self.current_epoch,
                )
            except Exception:
                # Avoid crashing training if tensorboard rejects shape
                continue

    def compute_and_log_metrics(
        self, loss_output: LossOutput, metrics: dict[str, ElboMetric], mode: str
    ):
        """Computes and logs metrics."""
        # Log individual reconstruction losses if there are multiple
        if isinstance(loss_output.reconstruction_loss, dict) and len(loss_output.reconstruction_loss.keys()) > 1:
            for k, rl in loss_output.reconstruction_loss.items():
                self.log(
                    f"{mode}_{k}",
                    rl.mean() if isinstance(rl, torch.Tensor) else rl,
                    on_epoch=True,
                    batch_size=loss_output.n_obs_minibatch,
                    prog_bar=True,
                )
        # Log individual kl local losses if there are multiple
        if isinstance(loss_output.kl_local, dict) and len(loss_output.kl_local) > 1:
            for k, kl in loss_output.kl_local.items():
                is_tensor = isinstance(kl, torch.Tensor)
                metric_name = f"{mode}_{k}"
                self.log(
                    metric_name,
                    kl.mean() if is_tensor else kl,
                    on_epoch=True,
                    batch_size=loss_output.n_obs_minibatch,
                    prog_bar=False,
                )
        # Drop the per-latent KL tensor from the metrics (not logged)
        loss_output.extra_metrics.pop(MODULE_KEYS.KL_Z_PER_LATENT_KEY)
        
        # Responder probability from the efficiency gate (if active)
        responder_prob = loss_output.extra_metrics['extra_outputs'].get('responder_prob', None)
        
        # Initialize default classification metrics. NOTE: classification_loss is deliberately
        # NOT seeded here — this model's A head is the prototype scorer (no
        # classification_loss), so the scvi-inherited classification_loss slot was never written and logged a
        # constant np.inf trace on every split. Dropping the seed stops that spurious inf curve; it
        # is purely a logging metric and never enters total_loss/backprop (assembled in _xpert.loss()).
        # The BARE names carry head H (the free classifier, CLS_HEAD_LOGITS_KEY); every proto_*
        # name carries head A (the prototype scorer, CLS_LOGITS_KEY). Seeded because H's key is
        # gated on cls_head_weight != 0 / use_cls_head — a monitor that goes MISSING after the
        # val loop has run raises MisconfigurationException instead of failing quietly.
        metrics_dict = {
            METRIC_KEYS.ACCURACY_KEY: 0.0,
            METRIC_KEYS.F1_SCORE_KEY: 0.0,
            'top5_recall': 0.0,
            'top10_recall': 0.0,
            f'proto_{METRIC_KEYS.ACCURACY_KEY}': 0.0,
            f'proto_{METRIC_KEYS.F1_SCORE_KEY}': 0.0,
            # Seed top-K recall too (like proto_f1_score above) so they're ALWAYS logged, even on a batch
            # where proto_pred is None -- a monitor that goes MISSING breaks checkpointing / early stopping.
            # Overwritten with the real value below.
            'proto_top5_recall': 0.0,
            'proto_top10_recall': 0.0,
        }
        # Per-cell context labels (batch-composition histogram below)
        ctx_labels = loss_output.extra_metrics['extra_outputs'].get('ctx_labels', None)

        # Get true class labels
        true_labels = loss_output.true_labels.squeeze(-1)
        # Prototype (A-head) classification metrics
        proto_pred = loss_output.extra_metrics['extra_outputs'].get(PREDICTION_KEYS.PREDICTION_KEY, None)
        if proto_pred is not None:
            # Get in-batch labels
            unique = list(torch.unique(true_labels))
            n_cls = len(unique)
            n_total_cls = self.n_classes
            base_proto_pred = proto_pred
            # Check if any classes are outside of training classes, i.e null classes or zero-shot candidates
            unknown_mask = (1-(proto_pred == torch.unique(true_labels).reshape(-1, 1)).sum(0)).bool()
            # Check null predictions
            null_mask = (proto_pred >= self.n_classes)
            if unknown_mask.any():
                # Assign anything outside of training to placeholder and add to list of predictions
                proto_pred = proto_pred.masked_fill(unknown_mask, n_cls)
                unique.append(n_cls)
                n_cls += 1
                n_total_cls += 1
            # Get local index mapping
            mapping = {int(u): i for i, u in enumerate(unique)}
            # Y in-batch labels
            true_local = torch.tensor(
                [mapping[int(v)] for v in true_labels],
                device=true_labels.device
            )
            # Local yhat
            pred_local = torch.tensor(
                [mapping[int(v)] for v in proto_pred],
                device=proto_pred.device
            )

            # Calculate accuracy over predicted and true labels
            accuracy_local = tmf.classification.multiclass_accuracy(
                pred_local,
                true_local,
                num_classes=n_cls,
                ignore_index=n_cls,
                average=self.average,
            )
            # Calculate F1 score over predicted and true labels
            f1_local = tmf.classification.multiclass_f1_score(
                pred_local,
                true_local,
                num_classes=n_cls,
                ignore_index=n_cls,
                average=self.average,
            )
            # Calculate total F1 score over all classes
            f1 = tmf.classification.multiclass_f1_score(
                base_proto_pred,
                true_labels,
                num_classes=n_total_cls,
                average=self.average,
            )

            # Log fraction of predicted responders
            null_frac = responder_prob.mean() if responder_prob is not None else 0.0
            metrics_dict[f'responder_fraction'] = null_frac
            # Threshold-free soft F1 directly on continuous efficiency scores
            true_eff = loss_output.extra_metrics['extra_outputs'].get(REGISTRY_KEYS.CLS_EFF_KEY, None)
            pred_eff = loss_output.extra_metrics['extra_outputs'].get('eff_hat', None)
            if true_eff is not None and pred_eff is not None:
                y_true = true_eff.flatten().float().abs()
                y_pred = pred_eff.flatten().float().abs()
                # Robust per-batch normalization to [0,1] using a shared scale
                scale = torch.quantile(torch.cat([y_true, y_pred]), 0.99).clamp(min=1e-6)
                y_true = (y_true / scale).clamp(0, 1)
                y_pred = (y_pred / scale).clamp(0, 1)
                soft_tp = torch.minimum(y_true, y_pred).sum()
                soft_fp = torch.clamp(y_pred - y_true, min=0).sum()
                soft_fn = torch.clamp(y_true - y_pred, min=0).sum()
                denom = 2 * soft_tp + soft_fp + soft_fn
                soft_f1 = (2 * soft_tp / denom) if denom > 0 else torch.tensor(0.0, device=y_pred.device)
                metrics_dict[f'eff_soft_{METRIC_KEYS.F1_SCORE_KEY}'] = soft_f1
            # Efficiency-weighted macro F1 on classification predictions —
            # shared implementation with evaluate() via sklearn (NaN-skipping macro)
            if true_eff is not None:
                from expert.utils.performance import weighted_f1_macro
                metrics_dict[f'proto_{METRIC_KEYS.F1_SCORE_KEY}_eff_weighted'] = weighted_f1_macro(
                    y_true=true_local,
                    y_pred=pred_local,
                    weights=true_eff,
                    n_classes=n_cls,
                    ignore_index=n_cls,
                )
            # Log proto f1 within not-null predicted cells (if there are enough responders in batch)
            if null_frac > 0 and null_frac < 0.9:
                null_mask = responder_prob < 0.5
                if (~null_mask).sum() > 0:
                    f1_responder = tmf.classification.multiclass_f1_score(
                        pred_local[~null_mask],
                        true_local[~null_mask],
                        num_classes=n_cls,
                        ignore_index=n_cls,
                        average=self.average,
                    )
                else:
                    f1_responder = 0.0
                metrics_dict[f'proto_{METRIC_KEYS.F1_SCORE_KEY}_responder'] = f1_responder
                # Do the same for non-responder fraction
                if null_mask.sum() > 0:
                    f1_non_responder = tmf.classification.multiclass_f1_score(
                        pred_local[null_mask],
                        true_local[null_mask],
                        num_classes=n_cls,
                        ignore_index=n_cls,
                        average=self.average,
                    )
                else:
                    f1_non_responder = 0.0
                metrics_dict[f'proto_{METRIC_KEYS.F1_SCORE_KEY}_non_responder'] = f1_non_responder
                
            # Update metrics. The PRIMARY proto_f1_score (what checkpoints / early-stop
            # monitor) is the SUPPORT-FILTERED local F1 — macro over only the classes
            # present in this batch, so chance ≈ 1/(observed classes) ≈ 1/200 on the
            # small split. The old full-observed-vocab macro (num_classes=n_total_cls)
            # counts every observed class absent from a val batch as F1=0 — badly
            # deflated on the full run (macro over ~9k) and still deflated here since
            # the (unbalanced) val loader doesn't cover all 200 per batch. Kept under
            # _global for reference; monitors should use the local one.
            metrics_dict[f'proto_{METRIC_KEYS.ACCURACY_KEY}_local'] = accuracy_local
            metrics_dict[f'proto_{METRIC_KEYS.F1_SCORE_KEY}_local'] = f1_local
            metrics_dict[f'proto_{METRIC_KEYS.F1_SCORE_KEY}'] = f1_local
            metrics_dict[f'proto_{METRIC_KEYS.F1_SCORE_KEY}_global'] = f1
            # Top-K recall (= top-K accuracy in multi-class). Gene-specific
            # like F1 but robust to small reorderings — exactly what the
            # geom-loss warmup causes when siblings move in and out of the
            # #1 slot. Single topk@10 + two cheap membership checks gives
            # both K=5 and K=10 for ~one O(B·n_classes) op per batch
            # (sub-ms on GPU, negligible vs forward/backward).
            proto_logits = loss_output.extra_metrics['extra_outputs'].get(
                MODULE_KEYS.CLS_LOGITS_KEY, None
            )
            if proto_logits is not None:
                n_logit_cols = proto_logits.shape[-1]
                t_idx = true_labels.long().flatten()
                in_local = t_idx < n_logit_cols
                if in_local.any():
                    yt = t_idx[in_local]
                    k_max = min(10, n_logit_cols)
                    top_idx = proto_logits[in_local].topk(k_max, dim=-1).indices  # (B, ≤10)
                    eq = top_idx == yt.unsqueeze(-1)
                    metrics_dict['proto_top5_recall'] = eq[:, :min(5, k_max)].any(dim=-1).float().mean()
                    metrics_dict['proto_top10_recall'] = eq.any(dim=-1).float().mean()

        # Head H (free classifier) metrics under the BARE names. Deliberately outside the
        # proto_pred block: H is trained by CE and must stay measurable when A produces no
        # prediction. Same support-filtered local mapping as the A path above, so the two
        # heads' f1_score/accuracy are directly comparable.
        head_logits = loss_output.extra_metrics['extra_outputs'].get(
            MODULE_KEYS.CLS_HEAD_LOGITS_KEY, None
        )
        if head_logits is not None:
            t_idx = true_labels.long().flatten()
            n_head_cols = head_logits.shape[-1]
            head_pred = head_logits.argmax(dim=-1).flatten()
            head_unique = list(torch.unique(t_idx))
            n_head_cls = len(head_unique)
            # Predictions outside the in-batch classes collapse to one placeholder class
            head_unknown = (1 - (head_pred == t_idx.unique().reshape(-1, 1)).sum(0)).bool()
            if head_unknown.any():
                head_pred = head_pred.masked_fill(head_unknown, n_head_cls)
                head_unique.append(n_head_cls)
                n_head_cls += 1
            head_map = {int(u): i for i, u in enumerate(head_unique)}
            head_true_local = torch.tensor([head_map[int(v)] for v in t_idx], device=t_idx.device)
            head_pred_local = torch.tensor([head_map[int(v)] for v in head_pred], device=head_pred.device)
            metrics_dict[METRIC_KEYS.ACCURACY_KEY] = tmf.classification.multiclass_accuracy(
                head_pred_local,
                head_true_local,
                num_classes=n_head_cls,
                ignore_index=n_head_cls,
                average=self.average,
            )
            metrics_dict[METRIC_KEYS.F1_SCORE_KEY] = tmf.classification.multiclass_f1_score(
                head_pred_local,
                head_true_local,
                num_classes=n_head_cls,
                ignore_index=n_head_cls,
                average=self.average,
            )
            in_local = t_idx < n_head_cols
            if in_local.any():
                yt = t_idx[in_local]
                k_max = min(10, n_head_cols)
                eq = head_logits[in_local].topk(k_max, dim=-1).indices == yt.unsqueeze(-1)
                metrics_dict['top5_recall'] = eq[:, :min(5, k_max)].any(dim=-1).float().mean()
                metrics_dict['top10_recall'] = eq.any(dim=-1).float().mean()

        # Extra-output + batch-composition histograms. These land on EVERY train
        # step and EVERY val/test/ood batch, so throttle to at most once per epoch
        # and only every full_log_every_n_epoch (matching the figures). Without this
        # the tfevents file balloons to tens of GB (~99% of bytes are histograms).
        log_hist = (
            self.log_histograms
            and self.full_log_every_n_epoch is not None
            and self.full_log_every_n_epoch > 0
            and (self.current_epoch + 1) % self.full_log_every_n_epoch == 0
            and self._hist_logged_epoch.get(mode) != self.current_epoch
        )
        if log_hist:
            self._hist_logged_epoch[mode] = self.current_epoch
        self._log_extra_output_histograms(mode, loss_output.extra_metrics['extra_outputs'], do_log=log_hist)
        # Log batch composition histograms
        self._log_extra_output_histograms(mode, {MODULE_KEYS.BATCH_INDEX_KEY: ctx_labels, MODULE_KEYS.LABEL_KEY: true_labels}, do_log=log_hist)
        # Add metrics to extra metrics for internal logging
        loss_output.extra_metrics.update(metrics_dict)
        # Remove all extra outputs from metrics for super logging
        loss_output.extra_metrics.pop('extra_outputs')
        # Compute and log all metrics
        super().compute_and_log_metrics(loss_output, metrics, mode)

    def _log_temperatures(self, batch_size: int | None = None):
        # Log the aligner's noise sigma (extra_aligner_kwargs.noise_sigma) if set
        sigma = getattr(self.module.aligner, 'noise_sigma', None)
        if sigma:
            self.log(
                "noise_sigma",
                sigma,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )

    def step(self, split, batch, batch_idx):
        """Step for supervised training"""
        # Update loss kwargs with schedules weights
        _w = self.weights
        self.loss_kwargs.update(_w)
        # Setup kwargs
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)
        
        # Add split-specific kwargs.
        # Honor the configured ``use_posterior_mean`` policy (constructor arg)
        # instead of hardcoding. Default 'val' preserves the VAE behavior:
        # posterior mean on eval splits (val / test / ood_test read qz.loc so
        # metrics are deterministic across re-runs) while train stays stochastic
        # (qz.rsample) so KL keeps its reparameterization-trick signal. 'both'
        # uses the mean on train too (deterministic alignment latent — e.g. AE
        # mode, pair with kl_weight=0); 'train' only on train; 'none' always
        # samples.
        _pm = self.use_posterior_mean
        input_kwargs['use_posterior_mean'] = (
            _pm in ('train', 'both') if split == 'train' else _pm in ('val', 'both')
        )

        # Zero-shot arms score a HELD-OUT gene (true label index >= n_labels), so they
        # need full-bank logits — local_predictions=False keeps the loss over the full
        # vocab and (paired with the gated slice in loss()) exposes the full-bank
        # CLS_LOGITS the recall metric ranks against. Every other split scores only the
        # trained block (local_predictions=True).
        input_kwargs['local_predictions'] = split not in ('id_ood_pert', 'ood_ood_pert')

        # Perform full forward pass of model
        return self.forward(batch, loss_kwargs=input_kwargs)
        
    def training_step(self, batch, batch_idx):
        """Training step for supervised training."""
        # Log batch composition
        self._epoch_labels.extend(batch[REGISTRY_KEYS.LABELS_KEY].flatten().cpu().numpy())
        self._epoch_datasets.extend(batch[REGISTRY_KEYS.BATCH_KEY].flatten().cpu().numpy())
        # Perform full forward pass of model
        inference_outputs, _, loss_output = self.step(split='train', batch=batch, batch_idx=batch_idx)
        loss = loss_output.loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        # Log all schedule weights
        for k, w in self.weights.items():
            self.log(k, w, on_epoch=True, prog_bar=False)
        # Log learnable temperatures if given
        self._log_temperatures()
        # Save in cache
        with torch.no_grad():
            self._cache_step_data('train', batch, inference_outputs, loss_output)
        # Log other metrics
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        y = loss_output.true_labels.squeeze(-1)
        # Count occurrences of each class in this batch
        unique, counts = torch.unique(y, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            self.class_batch_counts[u].append(c)
        # Save number of classes / batch
        self.train_class_counts.append(len(unique))
        # Return final loss value
        return loss

    def log(self, name, value, *args, **kwargs):
        """Default ``add_dataloader_idx=False`` so multi-dataloader val
        (val + test + ood_test) doesn't auto-suffix metric names with
        ``/dataloader_idx_N``. We split-prefix names ourselves."""
        kwargs.setdefault('add_dataloader_idx', False)
        return super().log(name, value, *args, **kwargs)

    def log_dict(self, dictionary, *args, **kwargs):
        """Same ``add_dataloader_idx=False`` default as :meth:`log`. scvi's
        base ``compute_and_log_metrics`` logs the ELBO collection
        (``elbo_``/``reconstruction_loss_``/``kl_local_``/``kl_global_<mode>``)
        through ``log_dict`` rather than ``log`` — without this override those
        four names get the ``/dataloader_idx_N`` suffix under multi-dataloader
        val, which makes the default ``lr_scheduler_metric='elbo_validation'``
        (and any monitor on them) unreachable. The per-mode collections are
        already prefix-distinct, so dropping the suffix can't collide."""
        kwargs.setdefault('add_dataloader_idx', False)
        return super().log_dict(dictionary, *args, **kwargs)

    def _split_for_dataloader_idx(self, dataloader_idx: int) -> str:
        """Resolve dataloader_idx -> split name via the data module."""
        names = getattr(self.trainer.datamodule, 'val_split_names', None) or ['val']
        return names[dataloader_idx] if dataloader_idx < len(names) else 'val'

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Validation step. Routes by ``dataloader_idx`` so val + test +
        ood_test loaders (returned as a list from
        ``DataSplitter.val_dataloader``) get logged under split-distinct
        metric names. The 'val' split keeps the legacy ``validation_*``
        prefix to preserve early-stopping monitor names."""
        split = self._split_for_dataloader_idx(dataloader_idx)
        # Backward-compat prefix: val cells log under 'validation_*' so
        # existing monitors (e.g. proto_f1_score_eff_weighted_validation)
        # don't break. test / ood_test get their own prefixes.
        prefix = 'validation' if split == 'val' else split
        # Perform full forward pass of model
        inference_outputs, _, loss_output = self.step(split=split, batch=batch, batch_idx=batch_idx)
        loss = loss_output.loss
        self.log(
            f"{prefix}_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=(split == 'val'),
        )
        # Cache step data for val and ood_test (test still skipped during fit
        # to keep overhead down — flip on by adding 'test' to log_full / here
        # if you also want cm/umap for in-distribution test during training).
        if split in ('val', 'ood_test'):
            with torch.no_grad():
                self._cache_step_data(split, batch, inference_outputs, loss_output)
        if split == 'val':
            self.compute_and_log_metrics(loss_output, self.val_metrics, prefix)
        elif split in ('id_ood_pert', 'ood_ood_pert'):
            # Zeroshot arms: log RECALL over the FULL bank (rank of the true HELD-OUT gene) —
            # the well-defined zero-shot metric — plus the loss ({prefix}_loss already logged
            # above). Computed inline from the prototype logits so we skip the per-mode ElboMetric
            # collection (which would KeyError). Same math as compute_and_log_metrics: a
            # held-out label (>= n_labels, < n_cls) is in-range when logits span the full bank.
            proto_logits = loss_output.extra_metrics.get('extra_outputs', {}).get(MODULE_KEYS.CLS_LOGITS_KEY, None)
            if proto_logits is not None:
                t_idx = loss_output.true_labels.long().flatten()
                in_local = t_idx < proto_logits.shape[-1]
                if in_local.any():
                    yt = t_idx[in_local]
                    k_max = min(10, proto_logits.shape[-1])
                    top_idx = proto_logits[in_local].topk(k_max, dim=-1).indices
                    eq = top_idx == yt.unsqueeze(-1)
                    bs = int(in_local.sum())
                    self.log(f"{prefix}_proto_top5_recall", eq[:, :min(5, k_max)].any(dim=-1).float().mean(),
                             on_epoch=True, batch_size=bs)
                    self.log(f"{prefix}_proto_top10_recall", eq.any(dim=-1).float().mean(),
                             on_epoch=True, batch_size=bs)
        else:
            # Route test / ood_test to their own ElboMetric collections.
            # scvi's compute_and_log_metrics indexes by ``elbo_{mode}``, so
            # each prefix needs a collection whose ``elbo`` is registered
            # under that exact key — sharing one collection raises KeyError
            # for whichever prefix wasn't its creation mode.
            metric_set = self.ood_test_metrics if split == 'ood_test' else self.test_metrics
            self.compute_and_log_metrics(loss_output, metric_set, prefix)

    def on_train_epoch_start(self):
        # The module has no view of the epoch inside loss(); anything epoch-dependent there reads
        # this. Without it an epoch-annealed knob silently evaluates at epoch 0 forever.
        self.module._cur_epoch = int(self.current_epoch)
        # Monitor batch composition
        self._epoch_labels = []
        self._epoch_datasets = []
        # Re-sample epoch dataset if toggled. In Lightning 2.x, train_dataloader
        # is a CombinedLoader; accessing unknown attrs before iter() raises
        # RuntimeError (not AttributeError), bypassing hasattr — unwrap to the
        # underlying loader via .iterables.
        loader = getattr(self.trainer, 'train_dataloader', None)
        if loader is not None and hasattr(loader, 'iterables'):
            loader = loader.iterables
        if loader is not None and hasattr(loader, 'set_epoch'):
            loader.set_epoch(self.current_epoch)

    def _plt_cat_emb(self):
        """Plot per-covariate decoder embedding weights as clustermaps.

        Each categorical covariate now has its own embedding (sized to its
        category count), so they live in different-dimensional spaces and
        can't share one heatmap — emit one clustermap per covariate.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        decoder = self.module.decoder
        embeddings = getattr(decoder, 'cat_embeddings', None)
        if not embeddings:
            return
        # cov_labels is the flat concatenation of category names across
        # covariates, in n_cat_list order. Slice per active covariate.
        active_counts = [n for n in decoder.n_cat_list if n > 0]
        labels_ok = (
            self.cov_labels is not None
            and len(self.cov_labels) == sum(active_counts)
        )
        offset = 0
        for i, (emb, n) in enumerate(zip(embeddings, active_counts)):
            if labels_ok:
                labels = self.cov_labels[offset:offset + n]
            else:
                labels = [f'cat_{j}' for j in range(n)]
            offset += n
            w = emb.weight.detach().cpu().numpy()
            colnames = 'd' + pd.Series(np.arange(w.shape[-1]), dtype=str)
            df = pd.DataFrame(w, index=labels, columns=colnames)
            g = sns.clustermap(
                df,
                row_cluster=True,
                col_cluster=w.shape[-1] > 1,
                figsize=(8, 6),
            )
            self.logger.experiment.add_figure(
                f"cat_cov_emb/cov_{i}", g.figure, self.current_epoch
            )
            plt.close(g.figure)

    @property
    def _is_main_process(self) -> bool:
        """True on rank 0 (or single-GPU). Used to guard plotting/figure
        writes so only one rank renders to TensorBoard under DDP. Metric
        ``self.log`` calls are NOT guarded — those run on all ranks."""
        return int(getattr(self, 'global_rank', 0)) == 0

    def on_train_epoch_end(self):
        # Class embedding adapter drift vs PCA init — zero-shot canary.
        # NOTE: no sync_dist — these are CPU-side values and NCCL (the GPU DDP
        # backend) cannot all-reduce CPU tensors. Cross-rank aggregation isn't
        # needed anyway: with use_distributed_sampler=False each rank holds the
        # same (grad-synced) weights, so per-rank values are identical.
        cls_emb = getattr(self.module, 'cls_emb', None)
        if cls_emb is not None and hasattr(cls_emb, 'adapter_drift'):
            drift = cls_emb.adapter_drift()
            if drift is not None:
                for k, v in drift.items():
                    self.log(f'adapter/{k}', v, on_epoch=True, prog_bar=False)

        # ATOM-SIDE progress series, the dictionary's analogue of the reconstruction rate.
        # Logged (not acted on) so that an atom-lr controller can be CALIBRATED on a real
        # per-epoch series. The only currently-fitted trigger borrows cls_head_loss, which
        # tracks JOINT encoder+head progress; these two are the dictionary's own.
        # Across ad2's checkpoints drift FLATTENS after 10k (0.0472 -> 0.0328 -> 0.0325 ->
        # 0.0313) while norm dispersion keeps CLIMBING (0.36 -> 1.78) -- so they disagree, and
        # which one a controller should watch is exactly what this series will settle.
        # `_atom_prev` is a plain attribute, NOT a register_buffer: an unconditional persistent
        # buffer makes existing checkpoints unloadable and shifts the RNG stream.
        if cls_emb is not None:
            _emb = getattr(cls_emb, 'embedding', None)
            _W = getattr(_emb, 'weight', None) if _emb is not None else None
            if _W is not None and _W.ndim == 2:
                import torch as _t
                # SLICE to n_labels. The table is padded to the full vocab and the unused rows
                # stay at their random init, so a drift computed over all ~19.7k rows is mostly
                # noise from rows the readout never touches.
                _n = int(getattr(self.module, 'n_labels', _W.shape[0]) or _W.shape[0])
                _n = max(1, min(_n, _W.shape[0]))
                with _t.no_grad():
                    cur = _W.detach()[:_n].float().cpu()
                    prev = getattr(self, '_atom_prev', None)
                    if prev is not None and prev.shape == cur.shape:
                        cs = _t.nn.functional.cosine_similarity(cur, prev, dim=1).mean()
                        self.log('atoms/drift', float(1.0 - cs), on_epoch=True, prog_bar=False)
                    nm = cur.norm(dim=1)
                    self.log('atoms/norm_disp', float(nm.std() / nm.mean().clamp_min(1e-12)),
                             on_epoch=True, prog_bar=False)
                    self._atom_prev = cur

        # Plotting / figure writes: rank 0 only.
        if (self.full_log_every_n_epoch > 0
                and (self.current_epoch + 1) % self.full_log_every_n_epoch == 0
                and self._is_main_process):
            # Plot heatmap of categorical covariate embedding in decoder
            if self.plot_cat_emb and getattr(self.module.decoder, 'cat_embeddings', None):
                # Plot and save in tensorboard logger
                self._plt_cat_emb()
            # Plot class distribution in batches
            if self.log_class_distribution:
                # Now log once with full epoch data
                _, label_counts = np.unique(self._epoch_labels, return_counts=True)
                self.logger.experiment.add_histogram(
                    "epoch/samples_per_label", label_counts, self.current_epoch
                )
                _, ds_counts = np.unique(self._epoch_datasets, return_counts=True)
                self.logger.experiment.add_histogram(
                    "epoch/samples_per_dataset", ds_counts, self.current_epoch
                )
        
    def _full_log_on_epoch_end(self):
        """Calculate metrics synchronously, then offload heavy plotting to a background thread."""
        from concurrent.futures import ThreadPoolExecutor
        # Get modes to do a full log on
        modes = self.log_full
        if isinstance(modes, str):
            modes = [modes]
        if len(self.log_full) == 0:
            return
        # Auto-include ood_test when its dataloader actually fed cache data
        # this epoch. Keeps log_full configurable for other splits but means
        # users don't have to manually list ood_test when the split exists.
        modes = list(modes)
        ood_cache = self.epoch_cache.get('ood_test')
        if ood_cache and len(ood_cache) > 0 and 'ood_test' not in modes:
            modes.append('ood_test')
        # Get cached steps from all modes
        data = {}
        _modes = []
        # Plotting is rank-0 only under DDP (figure writes to a single
        # TensorBoard). Metric logging below (`_log_full_metrics`) still runs
        # on every rank. With use_distributed_sampler=False each rank sees the
        # full split, so rank 0's plots are complete, not a 1/N shard.
        # Plot at the caller's full-log cadence. This method is only reached from
        # on_validation_epoch_end under the gate `(current_epoch+1) % full_log_every_n_epoch == 0`,
        # so the old separate `current_epoch % plot_every_n_epochs == 0` throttle (no +1,
        # default 10) never coincided with the validation epochs (check_val>1 runs val at
        # epoch N-1) — the UMAP figure was silently never emitted. Tie plotting to the
        # full-log gate that already governs this method.
        should_plot = self._is_main_process
        # Per-mode caches kept separately so each mode's plots see only its data;
        # `data` (combined across modes) is reserved for the umap which mixes splits.
        mode_datas: dict[str, dict[str, np.ndarray]] = {}
        for mode in modes:
            mode_data = self._process_epoch_cache(mode)
            if mode_data is None or len(mode_data.keys()) == 0:
                continue
            # Metrics must stay synchronous — they call self.log()
            self._log_full_metrics(mode, mode_data)
            _modes.append(mode)
            mode_datas[mode] = mode_data
            # Combined data — only built when umap is on, to merge splits in one figure
            if not self.plot_umap:
                continue
            if len(data) == 0:
                data = {k: v for k, v in mode_data.items()}
                continue
            for k, v in data.items():
                new_v = mode_data.get(k)
                if new_v is not None:
                    data[k] = np.concatenate((v, new_v), axis=0)

        # Offload all plotting to a background thread so training continues immediately
        if should_plot and _modes:
            # Snapshot per-mode data so plotting threads see immutable copies
            plot_mode_datas = {
                m: {k: v.copy() if hasattr(v, 'copy') else v for k, v in mode_datas[m].items()}
                for m in _modes
            }
            # Combined snapshot only needed for the umap path
            plot_data_combined = {k: v.copy() if hasattr(v, 'copy') else v for k, v in data.items()}
            plot_modes = list(_modes)
            epoch = self.current_epoch

            def _plot():
                import matplotlib.pyplot as plt
                # ISOLATE EVERY PLOT. This used to be one try around the whole block, so the first
                # plot to raise took every LATER plot with it -- and since umap runs last, one bad
                # figure meant a run with NO plots at all (v0.1.3: N has frames to ep59, W2 has
                # none). Each call now fails on its own and names itself, so a broken figure costs
                # exactly that figure and the log says which one.
                def _try(_name, _fn, *a, **kw):
                    try:
                        _fn(*a, **kw)
                    except Exception as _e:
                        import traceback
                        logging.warning(
                            '[Plot] %s failed at epoch %s: %s: %s\n%s',
                            _name, epoch, type(_e).__name__, _e,
                            ''.join(traceback.format_exception(type(_e), _e, _e.__traceback__))[-1500:])
                try:
                    for mode in plot_modes:
                        md = plot_mode_datas[mode]
                        if self.plot_f1_dist:
                            _try(f'f1_boxplots[{mode}]', self._plt_f1_per_ctx_boxplots, md, weighted=False)
                            _try(f'f1_boxplots_w[{mode}]', self._plt_f1_per_ctx_boxplots, md, weighted=True)
                            _try(f'f1_support_coverage[{mode}]', self._plt_f1_support_coverage, mode, md)
                        if self.plot_cm:
                            _try(f'confusion[{mode}]', self._plt_confusion, mode, md)
                        if self.plot_recall_at_k:
                            _try(f'recall_at_k[{mode}]', self._plt_recall_at_k, md)
                        if self.plot_clip_geom:
                            _try(f'proto_geometry[{mode}]', self._plt_proto_geometry, mode, md)
                            _try(f'eff_weight_corr[{mode}]', self._plt_eff_weight_corr, mode, md)
                    # UMAP intentionally uses combined data so all splits share one figure
                    if self.plot_umap:
                        full_name = '-'.join(plot_modes) if len(plot_modes) > 1 else plot_modes[0]
                        for z_key in self.plot_umap_key:
                            if z_key in plot_data_combined:
                                _try(f'umap[{z_key}]', self._plt_umap, full_name, plot_data_combined,
                                     current_epoch=epoch, z_key=z_key)
                finally:
                    plt.close('all')

            # Synchronous plotting
            _plot()

    # Calculate accuracy & f1 for entire validation set, not just per batch
    def on_validation_epoch_end(self):
        """Calculate metrics and plot charts for full data split(s)."""
        if self.full_log_every_n_epoch > 0 and (self.current_epoch + 1) % self.full_log_every_n_epoch == 0:
            self._full_log_on_epoch_end()
                    
    def _log_full_metrics(self, mode: str, mode_data: dict[str, np.ndarray]) -> None:
        # Skip logging if no classification is available yet
        if PREDICTION_KEYS.PREDICTION_KEY not in mode_data:
            return
        true_labels = torch.tensor(mode_data.get(REGISTRY_KEYS.LABELS_KEY))
        predicted_labels = torch.tensor(mode_data.get(PREDICTION_KEYS.PREDICTION_KEY)).squeeze(-1)
        # Per-cell context, read PRE-mask so it stays row-aligned with the label arrays
        ctx = mode_data.get(REGISTRY_KEYS.BATCH_KEY, None)
        ctx = torch.tensor(ctx).flatten() if ctx is not None else None
        # Row masks below must be applied to EVERY per-cell array, ctx included
        row_mask = None
        # Remove control cells from BOTH arrays so pairs stay aligned
        if self.module.ctrl_class_idx is not None:
            no_ctrl_mask = (true_labels != self.module.ctrl_class_idx)
            true_labels = true_labels[no_ctrl_mask]
            predicted_labels = predicted_labels[no_ctrl_mask]
            row_mask = no_ctrl_mask
            if ctx is not None:
                ctx = ctx[no_ctrl_mask]
        n_classes = self.n_classes
 
        # Check if any classes are outside of training classes, i.e
        unknown_mask = (predicted_labels >= n_classes)
        if unknown_mask.any():
            # Assign unknown to classes that are outside of the training classes
            predicted_labels = predicted_labels.masked_fill(unknown_mask, n_classes)
            n_classes += 1
        
        # Calculate classification metrics
        accuracy = tmf.classification.multiclass_accuracy(
            predicted_labels,
            true_labels,
            n_classes,
            average=self.average
        )
        # Macro F1 over classes PRESENT in this split (matches sklearn's classification_report)
        if self.average == 'macro':
            f1_per_class = tmf.classification.multiclass_f1_score(
                predicted_labels,
                true_labels,
                n_classes,
                average='none',
            )
            present = torch.zeros(n_classes, dtype=torch.bool, device=f1_per_class.device)
            present[torch.unique(torch.cat([true_labels.flatten(), predicted_labels.flatten()]))] = True
            f1 = f1_per_class[present].mean() if present.any() else torch.tensor(0.0, device=f1_per_class.device)
        else:
            f1 = tmf.classification.multiclass_f1_score(
                predicted_labels,
                true_labels,
                n_classes,
                average=self.average,
            )
        # No sync_dist: these are CPU tensors (from cached per-rank arrays) and
        # NCCL can't all-reduce CPU tensors. With use_distributed_sampler=False
        # each rank runs full validation, so the values are already identical.
        self.log(
            f"{mode}_full_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{mode}_full_f1",
            f1,
            on_epoch=True,
            prog_bar=False,
        )
        # Stratified F1 by per-split support (track rare-class stability over epochs)
        self._log_rare_class_metrics(mode, true_labels, predicted_labels, n_classes)
        # Per-dataset recall + the LIVE (non-excluded) dataset macro, for both heads
        self._log_per_dataset_metrics(mode, mode_data, true_labels, ctx, row_mask)
        self._log_group_readout_metrics(mode, mode_data, true_labels, ctx, row_mask)
        # Include top k predictions as well
        if self.top_k > 1:
            logits = torch.tensor(mode_data.get(MODULE_KEYS.CLS_LOGITS_KEY))
            top_k_acc_key = f'{mode}_full_accuracy_top_{self.top_k}'
            top_k_f1_key = f'{mode}_full_f1{self.top_k}'
            top_k_accuracy = tmf.classification.multiclass_accuracy(
                logits,
                true_labels,
                n_classes,
                average=self.average,
                top_k=self.top_k,
            )
            top_k_f1 = tmf.classification.multiclass_f1_score(
                logits,
                true_labels,
                n_classes,
                average=self.average,
                top_k=self.top_k,
            )
            self.log(
                top_k_acc_key,
                top_k_accuracy,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                top_k_f1_key,
                top_k_f1,
                on_epoch=True,
                prog_bar=False,
            )

    def _resolve_excluded_ctx(self) -> set[int]:
        """Context indices whose dataset name matches a ``live_macro_exclude`` prefix."""
        if self._excluded_ctx is None:
            if self.batch_labels is None or not self.live_macro_exclude:
                self._excluded_ctx = set()
            else:
                names = np.asarray(self.batch_labels).astype(str)
                self._excluded_ctx = {
                    i for i, n in enumerate(names)
                    if any(n.startswith(p) for p in self.live_macro_exclude)
                }
        return self._excluded_ctx

    def _log_group_readout_metrics(
        self,
        mode: str,
        mode_data: dict,
        true_labels: torch.Tensor,
        ctx: torch.Tensor | None,
        row_mask: torch.Tensor | None,
    ) -> None:
        """PSEUDOBULK A-readout proxy: the in-training metric that actually tracks the OMP readout.

        WHY THIS EXISTS. `{mode}_live_macro_top10_recall_A` is a PER-CELL macro, and it does not
        track the readout that ships: over 12 arms it correlates +0.22 with `ood_readout_probe`'s
        w_top10 and **-0.51 with the tail**. Early-stopping or checkpointing on it would therefore
        select systematically WORSE dictionaries. The cause is the aggregation, not the metric --
        the readout scores 566 control-centred pseudobulk GROUPS, and pseudobulking averages away
        precisely the per-cell noise the per-cell metric rewards fixing.

        This mirrors `expert/analysis/ood_readout_probe.py` exactly:
            base_d = mean z over ALL cached cells of dataset d
            Q      = mean z over (d, class) - base_d
            score  = cos(Q, atom),  rank of the true atom
        so `{mode}_grp_top10_A` is comparable to the probe's `top10` and is a VALID monitor.

        CANNOT be computed from the cached logits: `_logits_z2c` normalises z and the atom before
        the dot product, so mean-of-cosines != cosine-of-mean. It needs z, which the epoch cache
        already holds.

        TRAINED CLASSES ONLY. The atom table spans the 19,895-row VOCABULARY; scoring against it
        would rank the true atom among ~19.7k never-trained random rows. Sliced to n_labels, the
        same slice the OMP dictionary takes.

        Fully vectorised: group means via a single index_add_, and rank via `(S > S_true).sum(1)`
        rather than an argsort. Cost is one (G x n_labels) matmul once per logged epoch.
        """
        # UNCONDITIONAL EMISSION. This is intended as a checkpoint/early-stopping monitor, and
        # Lightning RAISES on a monitor that is merely ABSENT -- so any epoch that cannot form
        # groups (a smoke with too few cells, a dim mismatch, an arm without atoms) would KILL the
        # run rather than score it. Same convention as _log_per_dataset_metrics: fall back to 0.0,
        # which under mode='max' simply never wins, and read `{mode}_grp_n` to tell "no groups this
        # epoch" from "genuinely 0.0".
        def _emit(top1=0.0, top10=0.0, rankmed=-1.0, n=0.0):
            self.log(f'{mode}_grp_top1_A', float(top1), on_epoch=True, prog_bar=False)
            self.log(f'{mode}_grp_top10_A', float(top10), on_epoch=True,
                     prog_bar=(mode in ('val', 'ood_test')))
            self.log(f'{mode}_grp_rankmed_A', float(rankmed), on_epoch=True, prog_bar=False)
            self.log(f'{mode}_grp_n', float(n), on_epoch=True, prog_bar=False)

        if ctx is None or ctx.numel() == 0:
            _emit(); return
        ce = getattr(self.module, 'cls_emb', None)
        if ce is None or not getattr(ce, 'random_init_proxies', False):
            _emit(); return
        # 'z_shared_raw' FIRST. MODULE_KEYS.Z_SHARED_KEY in the epoch cache is the UNIT-
        # NORMALISED z (_xpert.py publishes z_shared_norm under it), and averaging unit
        # vectors is a directional mean, not the pseudobulk ood_readout_probe computes.
        Z = mode_data.get('z_shared_raw', None)
        if Z is None:
            Z = mode_data.get(MODULE_KEYS.Z_SHARED_KEY, mode_data.get(MODULE_KEYS.Z_KEY, None))
        if Z is None:
            _emit(); return
        Z = torch.as_tensor(Z).float()
        if row_mask is not None:
            Z = Z[row_mask]
        y = true_labels.long().flatten()
        c = ctx.long().flatten()
        if Z.shape[0] != y.numel() or Z.shape[0] != c.numel():
            _emit(); return
        with torch.no_grad():
            A = ce.encode().detach().float().cpu()
            n_lab = int(min(getattr(self.module, 'n_labels', 0) or 0, A.shape[0]))
            if n_lab < 2:
                _emit(); return
            A = A[:n_lab]                                   # TRAINED BLOCK ONLY
            # SPACE GUARD. The cache stores MODULE_KEYS.Z_KEY; under a split latent that is the
            # FULL z while the atoms live in z_shared, so a fallback to 'z' would silently compute
            # cosines between two different spaces and log a plausible-looking number. Refuse
            # instead, once, loudly.
            # NORMALISATION GUARD. A dimension check cannot distinguish raw z from unit-
            # normalised z -- both are n_shared wide -- so check the norms directly. If they are
            # all ~1 the cache handed us the normalised key and every group mean would be a
            # directional mean; refuse rather than log a plausible wrong number.
            _nrm = Z.norm(dim=1)
            if float((_nrm - 1.0).abs().max()) < 1e-3:
                if not getattr(self, '_grp_norm_warned', False):
                    self._grp_norm_warned = True
                    print('[grp-readout] DISABLED: cached z is UNIT-NORMALISED, so a group mean '
                          'would be a directional mean, not the pseudobulk ood_readout_probe '
                          "computes. Publish/cache raw z_shared ('z_shared_raw') to enable it.")
                _emit(); return
            if Z.shape[1] != A.shape[1]:
                if not getattr(self, '_grp_dim_warned', False):
                    self._grp_dim_warned = True
                    print(f'[grp-readout] DISABLED: cached z is {Z.shape[1]}-d but atoms are '
                          f'{A.shape[1]}-d (split latent?). Cache z_shared to enable it.')
                _emit(); return
            keep = (y >= 0) & (y < n_lab)
            if int(keep.sum()) < self.grp_min_cells:
                _emit(); return
            Z, y, c = Z[keep], y[keep], c[keep]

            # ---- per-dataset baseline (vectorised) ----
            uc, cinv = torch.unique(c, return_inverse=True)
            d_sum = torch.zeros(uc.numel(), Z.shape[1]).index_add_(0, cinv, Z)
            d_cnt = torch.zeros(uc.numel()).index_add_(0, cinv, torch.ones_like(cinv, dtype=Z.dtype))
            base = d_sum / d_cnt.clamp_min(1).unsqueeze(-1)

            # ---- (dataset, class) group means (vectorised) ----
            gid = cinv * n_lab + y
            ug, ginv, gcnt = torch.unique(gid, return_inverse=True, return_counts=True)
            g_sum = torch.zeros(ug.numel(), Z.shape[1]).index_add_(0, ginv, Z)
            g_mean = g_sum / gcnt.clamp_min(1).unsqueeze(-1).to(Z.dtype)
            g_ctx, g_lab = torch.div(ug, n_lab, rounding_mode='floor'), ug % n_lab
            ok = gcnt >= self.grp_min_cells
            if int(ok.sum()) < 2:
                _emit(); return
            Q = (g_mean - base[g_ctx])[ok]
            truth = g_lab[ok]

            S = torch.nn.functional.normalize(Q, dim=-1) @ torch.nn.functional.normalize(A, dim=-1).T
            s_true = S.gather(1, truth.unsqueeze(1))
            rank = (S > s_true).sum(dim=1)                  # 0-based, no argsort
            _emit(top1=(rank == 0).float().mean(), top10=(rank < 10).float().mean(),
                  rankmed=rank.float().median(), n=truth.numel())

    def _log_per_dataset_metrics(
        self,
        mode: str,
        mode_data: dict[str, np.ndarray],
        true_labels: torch.Tensor,
        ctx: torch.Tensor | None,
        row_mask: torch.Tensor | None,
    ) -> None:
        # Per-dataset class-macro recall@10 for BOTH heads, plus the dataset-mean over the
        # non-excluded ("LIVE") datasets — the pooled number is dominated by the replicate
        # donors, so the LIVE macro is the comparable one. Runs on the epoch-end cache, whose
        # arrays are the full-epoch concat, so self.log takes no batch_size (a single scalar
        # per epoch, already throttled by full_log_every_n_epoch).
        import re
        if ctx is None or self.batch_labels is None or ctx.numel() == 0:
            return
        heads = {}
        for name, key in (('A', MODULE_KEYS.CLS_LOGITS_KEY), ('H', MODULE_KEYS.CLS_HEAD_LOGITS_KEY)):
            arr = mode_data.get(key, None)
            if arr is None:
                continue
            lg = torch.tensor(arr)
            if row_mask is not None:
                lg = lg[row_mask]
            if lg.shape[0] == ctx.numel():
                heads[name] = lg
        if not heads:
            return
        excl = self._resolve_excluded_ctx()
        names = np.asarray(self.batch_labels).astype(str)
        live, marson = defaultdict(list), defaultdict(list)
        for c in torch.unique(ctx):
            m = ctx == c
            n_d = int(m.sum())
            # Skip (never log 0.0) datasets too small for a trustworthy class-macro
            if n_d < self.per_ds_min_cells:
                continue
            ci = int(c)
            safe = re.sub(r'[^0-9A-Za-z_.-]', '_', names[ci]) if ci < len(names) else str(ci)
            yt = true_labels[m].long().flatten()
            self.log(f"ds_{mode}/{safe}/n_cells", float(n_d), on_epoch=True, prog_bar=False)
            for hname, lg in heads.items():
                lg_d = lg[m]
                in_local = yt < lg_d.shape[-1]
                if not bool(in_local.any()):
                    continue
                yt_d = yt[in_local]
                k_max = min(10, lg_d.shape[-1])
                hit = (
                    lg_d[in_local].topk(k_max, dim=-1).indices == yt_d.unsqueeze(-1)
                ).any(dim=-1).float()
                # Class-macro (not cell-micro) within the dataset
                r10 = torch.stack([hit[yt_d == u].mean() for u in torch.unique(yt_d)]).mean()
                self.log(f"ds_{mode}/{safe}/{hname}_top10_recall", r10, on_epoch=True, prog_bar=False)
                (marson if ci in excl else live)[hname].append(r10)
        for hname in heads:
            # Emitted UNCONDITIONALLY: this is a checkpoint/early-stopping monitor, and Lightning
            # RAISES on a monitor that is merely absent, so a val epoch with no live dataset would
            # kill the run rather than score it 0. Falls back to 0.0, which under mode='max' simply
            # never wins — read `n_live_ds` to tell "no live data this epoch" from "genuinely 0.0".
            self.log(
                f"{mode}_live_macro_top10_recall_{hname}",
                torch.stack(live[hname]).mean() if live[hname] else torch.tensor(0.0),
                on_epoch=True,
                # OOD ON THE BAR TOO. This is the metric arms are judged on, and watching only val
                # is how a run looks healthy while OOD says otherwise. Head A only: _H is the
                # donor-leak-prone head and would just crowd the bar.
                prog_bar=(mode in ('val', 'ood_test') and hname == 'A'),
            )
            self.log(f"{mode}_n_live_ds_{hname}", float(len(live[hname])), on_epoch=True, prog_bar=False)
            # Logged alongside LIVE so the leak size (excluded - live) is a tracked quantity
            if marson[hname]:
                self.log(
                    f"{mode}_excluded_macro_top10_recall_{hname}",
                    torch.stack(marson[hname]).mean(),
                    on_epoch=True,
                    prog_bar=False,
                )

    def _log_rare_class_metrics(
        self,
        mode: str,
        true_labels: torch.Tensor,
        predicted_labels: torch.Tensor,
        n_classes: int,
    ) -> None:
        # Stratifies macro F1 by per-split class support so the rare tail is
        # tracked separately from common classes — under WeightedRandomSampler
        # the rare-tier curve is the signal that weighting is actually helping.
        if true_labels.numel() == 0:
            return
        support = torch.bincount(true_labels.long().flatten(), minlength=n_classes)
        f1_per_class = tmf.classification.multiclass_f1_score(
            predicted_labels, true_labels, n_classes, average='none',
        )
        tiers = {
            'rare':   (support > 0) & (support <= 50),
            'mid':    (support > 50) & (support <= 200),
            'common': support > 200,
        }
        for name, mask in tiers.items():
            n_cls = int(mask.sum().item())
            if n_cls == 0:
                continue
            self.log(f"{mode}_f1_{name}", f1_per_class[mask].mean(), on_epoch=True, prog_bar=False)
            self.log(f"{mode}_n_classes_{name}", float(n_cls), on_epoch=True, prog_bar=False)
            self.log(f"{mode}_n_cells_{name}", float(support[mask].sum().item()), on_epoch=True, prog_bar=False)

    def _plt_confusion(self, mode: str, mode_data: dict[str, np.ndarray]):
        """
        Plot confusion matrices:
        - Per context: cm/{mode}/{ctx}
        - Combined: cm/{mode}/all
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        preds = mode_data.get(PREDICTION_KEYS.PREDICTION_KEY)
        if preds is None:
            return

        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        contexts = mode_data[REGISTRY_KEYS.BATCH_KEY]

        df = pd.DataFrame({
            "pred": preds,
            "label": labels,
            "context": contexts,
        })

        # Optional context names
        if self.batch_labels is not None:
            df["context"] = self.batch_labels[df["context"]]

        # -----------------------------------
        # Helper to plot one confusion matrix
        # -----------------------------------
        def _plot_cm(cm, title, tag):
            fig, ax = plt.subplots(figsize=(6, 5))

            sns.heatmap(
                cm,
                cmap="Blues",
                vmin=0,
                vmax=1,
                square=True,
                cbar=True,
                ax=ax,
            )

            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            plt.tight_layout()

            self.logger.experiment.add_figure(
                tag=tag,
                figure=fig,
                global_step=self.current_epoch,
            )
            plt.close(fig)

        # -----------------------------------
        # Global confusion matrix
        # -----------------------------------
        cm_all = confusion_matrix(
            df["label"],
            df["pred"],
            labels=list(range(self.n_classes)),
        ).astype(float)

        # Row normalize
        row_sums = cm_all.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_all_norm = cm_all / row_sums

        _plot_cm(
            cm_all_norm,
            title=f"{mode} | Confusion Matrix (all contexts)",
            tag=f"cm/{mode}/all",
        )

        # -----------------------------------
        # Per-context confusion matrices
        # -----------------------------------
        for ctx, sdf in df.groupby("context"):
            if len(sdf) == 0:
                continue

            cm_ctx = confusion_matrix(
                sdf["label"],
                sdf["pred"],
                labels=list(range(self.n_classes)),
            ).astype(float)

            row_sums = cm_ctx.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_ctx_norm = cm_ctx / row_sums

            _plot_cm(
                cm_ctx_norm,
                title=f"{mode} | Confusion Matrix ({ctx})",
                tag=f"cm/{mode}/{ctx}",
            )
            
    def _plt_eff_weight_corr(self, mode: str, mode_data: dict[str, np.ndarray]):
        """Predicted vs reference per-cell efficiency for a single split.

        Uses ``plot_efficiency_comparison`` from the plotting utils (ridge
        density on top, per-context hexbin grid below with Spearman/Pearson/
        R²/Wasserstein in the titles). One figure per split — logged under
        ``proto/efficiency_comparison_{mode}``.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from expert.utils.plotting import plot_efficiency_comparison

        # Read RAW eff_hat + RAW |weights| from the cache and z-score per
        # dataset across the full epoch cache. The per-batch ``eff_*_scaled``
        # logged from the loss path are correct for MSE (each batch standardizes
        # against its own (mean, std)) but aggregating those across batches gives
        # inconsistent normalization — Spearman survives, R² gets shredded.
        # Doing the standardization once over the full split here gives the
        # consistent diagonal the plot needs.
        eff = mode_data.get(REGISTRY_KEYS.CLS_EFF_KEY, None)
        eff_hat = mode_data.get("eff_hat", None)
        if eff is None or eff_hat is None:
            return
        eff = np.asarray(eff).reshape(-1)
        eff_hat = np.asarray(eff_hat).reshape(-1)
        if len(eff) == 0 or len(eff) != len(eff_hat):
            return

        ctx = mode_data.get(REGISTRY_KEYS.BATCH_KEY, None)
        if ctx is None:
            return
        ctx = np.asarray(ctx).reshape(-1)
        if self.batch_labels is not None:
            ctx = self.batch_labels[ctx]
        # Per-dataset z-score (same min_cells / min_target_std guards as the
        # loss-side helper). Cells in skipped datasets get z = 0.
        eff_z = np.zeros_like(eff_hat, dtype=np.float32)
        ref_z = np.zeros_like(eff, dtype=np.float32)
        for ds in np.unique(ctx):
            m = ctx == ds
            if m.sum() < 8:
                continue
            p, t = eff_hat[m], eff[m]
            t_std = float(t.std())
            if t_std < 1e-4:
                continue
            eff_z[m] = (p - p.mean()) / (float(p.std()) + 1e-4)
            ref_z[m] = (t - t.mean()) / (t_std + 1e-4)
        df = pd.DataFrame({'eff': ref_z, 'eff_hat': eff_z, 'split': mode, 'dataset': ctx})

        try:
            fig, _ = plot_efficiency_comparison(
                obs=df,
                ref_col='eff',
                pred_col='eff_hat',
                split_col='split',
                dataset_col='dataset',
                return_fig=True,
            )
        except Exception as e:
            logging.warning(f'[Plot] efficiency_comparison failed for {mode}: {e}')
            return

        if hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(
                f"proto/efficiency_comparison_{mode}",
                fig,
                global_step=self.current_epoch,
            )
        plt.close(fig)
            
    def _plt_proto_geometry(self, mode: str, mode_data: dict[str, np.ndarray]):
        """
        Plot prototype geometry sanity scatter:
        x = hardest negative similarity
        y = positive similarity

        One subplot per condition.
        Logged to TensorBoard: proto/geom_{mode}
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import math

        logits = mode_data.get(MODULE_KEYS.CLS_LOGITS_KEY)
        if logits is None:
            return

        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        contexts = mode_data[REGISTRY_KEYS.BATCH_KEY]
        weights = mode_data.get("proto/weights", None)

        logits = np.asarray(logits)
        labels = np.asarray(labels).astype(int).flatten()
        contexts = np.asarray(contexts)

        # ----- positive similarities -----
        pos = logits[np.arange(len(labels)), labels]

        # ----- hardest negative -----
        neg_logits = logits.copy()
        neg_logits[np.arange(len(labels)), labels] = -np.inf
        hardest_neg = neg_logits.max(axis=1)

        df = pd.DataFrame({
            "pos": pos,
            "neg": hardest_neg,
            "context": contexts,
            "label": labels,
        })

        if weights is not None:
            df["weight"] = np.asarray(weights)

        # Optional mapping
        if self.batch_labels is not None:
            df["context"] = self.batch_labels[df["context"]]

        # ----- define conditions to plot -----
        conditions = [c for c in ["context", "label", "weight"] if c in df.columns]

        n_cond = len(conditions)
        if n_cond == 0:
            return

        n_cols = min(3, n_cond)
        n_rows = math.ceil(n_cond / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 5 * n_rows),
            squeeze=False
        )

        axes = axes.flatten()

        vmin = min(df["neg"].min(), df["pos"].min())
        vmax = max(df["neg"].max(), df["pos"].max())

        for i, cond in enumerate(conditions):
            ax = axes[i]

            if pd.api.types.is_numeric_dtype(df[cond]):
                # continuous coloring
                sc = ax.scatter(
                    df["neg"],
                    df["pos"],
                    c=df[cond],
                    s=6,
                    alpha=0.4,
                )
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            else:
                # categorical coloring
                for val, sdf in df.groupby(cond):
                    ax.scatter(
                        sdf["neg"],
                        sdf["pos"],
                        s=6,
                        alpha=0.4,
                        label=str(val),
                    )
                if df[cond].nunique() <= 10:
                    ax.legend(markerscale=2, fontsize=8)

            ax.plot([vmin, vmax], [vmin, vmax], linestyle="--")
            ax.set_xlabel("Hardest negative similarity")
            ax.set_ylabel("Positive similarity")
            ax.set_title(f"{cond}")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{mode} | prototype geometry sanity", fontsize=14)
        plt.tight_layout()

        self.logger.experiment.add_figure(
            tag=f"proto/geom_{mode}",
            figure=fig,
            global_step=self.current_epoch,
        )

        plt.close(fig)

    def _plt_f1_per_ctx_boxplots(self, mode_data: dict[str, np.ndarray], weighted: bool = False):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import torch
        import numpy as np
        from torchmetrics.functional.classification import multiclass_f1_score
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        preds = mode_data.get(PREDICTION_KEYS.PREDICTION_KEY)
        if preds is None:
            return
        # Fetch efficiency weights for weighted variant
        weights = mode_data.get(REGISTRY_KEYS.CLS_EFF_KEY) if weighted else None
        if weighted and weights is None:
            return

        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        contexts = mode_data[REGISTRY_KEYS.BATCH_KEY]
        splits = mode_data[REGISTRY_KEYS.SPLIT_KEY]

        df = pd.DataFrame({
            "pred": preds,
            "label": labels,
            "context": contexts,
            "split": splits,
        })
        if weighted:
            df["weight"] = np.abs(np.asarray(weights).flatten())

        # Optional context names
        if self.batch_labels is not None:
            df["context"] = self.batch_labels[df["context"]]

        records = []

        # (split, context) → per-class F1 + support
        for (split, ctx), g in df.groupby(["split", "context"]):
            if g["label"].nunique() < 2:
                continue  # F1 meaningless

            labels_t = torch.tensor(g["label"].values)
            preds_t = torch.tensor(g["pred"].values)
            cm_n_cls = self.n_classes
            unknown_mask = (preds_t >= self.n_classes)
            if unknown_mask.any():
                # Assign unknown to classes that are outside of the training classes
                preds_t = preds_t.masked_fill(unknown_mask, self.n_classes)
                cm_n_cls += 1

            if weighted:
                from expert.utils.performance import weighted_f1_per_class
                f1_per_class, support = weighted_f1_per_class(
                    y_true=g["label"].values,
                    y_pred=g["pred"].values,
                    weights=g["weight"].values,
                    n_classes=self.n_classes,
                    ignore_index=self.n_classes,
                )
            else:
                f1_per_class = multiclass_f1_score(
                    preds_t,
                    labels_t,
                    num_classes=cm_n_cls,
                    ignore_index=self.n_classes,
                    average="none",
                ).cpu().numpy()

                # class support within this context
                support = (
                    g["label"]
                    .value_counts()
                    .reindex(range(self.n_classes), fill_value=0)
                    .values
                )

            for cls_id, (f1, sup) in enumerate(zip(f1_per_class, support)):
                if sup == 0:
                    continue
                # NaN IS EXPECTED HERE, ONLY ON THE WEIGHTED PATH. weighted_f1_per_class uses
                # zero_division=np.nan so the metric path can np.nanmean past undefined classes.
                # Weighted `support` counts TRUE samples, but the NaN comes from the PRECISION
                # denominator -- a class nothing was predicted as -- so a class can have support
                # > 0 and still be NaN. Early in training that is most classes, every point in a
                # split is NaN, and seaborn gets an all-masked array: "ConversionError: Failed to
                # convert value(s) to axis units". The unweighted path never hits this because
                # torchmetrics returns 0.0 rather than NaN.
                if not np.isfinite(f1):
                    continue
                records.append({
                    "split": split,
                    "class": cls_id,
                    "context": ctx,
                    "f1": f1,
                    "support": sup,
                })

        if not records:
            return

        f1_df = pd.DataFrame(records)

        n_ctx = f1_df.context.nunique()

        for split, sdf in f1_df.groupby("split"):
            if sdf.empty or not np.isfinite(sdf["f1"].values).any():
                continue
            fig, ax = plt.subplots(figsize=(max(8, n_ctx * 0.25), 4))

            # --- boxplot (distribution) ---
            sns.boxplot(
                data=sdf,
                x="context",
                y="f1",
                ax=ax,
                showfliers=False,
                boxprops=dict(alpha=.5)
            )

            # --- scatter with support-based coloring ---
            ctx_order = list(sdf["context"].unique())
            ctx_to_x = {ctx: i for i, ctx in enumerate(ctx_order)}

            x = sdf["context"].map(ctx_to_x).values
            y = sdf["f1"].values
            support = sdf["support"].values

            norm = Normalize(vmin=support.min(), vmax=support.max())
            cmap = plt.cm.viridis

            # --- size scaling based on support ---
            s_min, s_max = 10, 80
            sizes = np.sqrt(support)
            sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
            sizes = s_min + sizes * (s_max - s_min)
            # add jitter points
            sc = ax.scatter(
                x + np.random.uniform(-0.15, 0.15, size=len(x)),  # jitter
                y,
                c=support,
                s=sizes,
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                linewidths=0,
            )

            # --- colorbar ---
            cbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                pad=0.01,
            )
            cbar.set_label("Class support within context", rotation=90)

            title_prefix = "eff-weighted per-context F1" if weighted else "per-context F1"
            ax.set_title(f"{split} | {title_prefix} (colored by class support) | N={support.sum()}")
            ax.set_xlabel("Context")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(ctx_order)))
            ax.set_xticklabels(ctx_order, rotation=45, ha="right")

            plt.tight_layout()
            tag_suffix = "_eff_weighted" if weighted else ""
            self.logger.experiment.add_figure(
                tag=f"f1_per_class_boxplot{tag_suffix}/{split}",
                figure=fig,
                global_step=self.current_epoch,
            )
            plt.close(fig)

    
    def _plt_recall_at_k(self, mode_data: dict[str, np.ndarray]):
        """
        Plot recall@k curves per context, one figure per split.
        For each k in {1, 10, 100}, compute the fraction of samples whose
        true label appears in the top-k predicted classes (by logit score).
        Logged to TensorBoard: ``recall_at_k/{split}``.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        logits = mode_data.get(MODULE_KEYS.CLS_LOGITS_KEY)
        if logits is None:
            return

        labels = np.asarray(mode_data[REGISTRY_KEYS.LABELS_KEY]).astype(int).flatten()
        contexts = np.asarray(mode_data[REGISTRY_KEYS.BATCH_KEY])
        logits = np.asarray(logits)
        # Per-cell split labels — required so train/val/test each get their
        # own figure rather than all sharing the last mode's data
        splits = mode_data.get(REGISTRY_KEYS.SPLIT_KEY)
        if splits is None:
            return
        splits = np.asarray(splits)

        n_classes = logits.shape[1]
        ks = [k for k in [1, 10, 100] if k <= n_classes]
        max_k = max(ks)

        # Top-k once over the full batch — slice per split below
        top_indices = np.argsort(logits, axis=1)[:, -max_k:][:, ::-1]
        hits = {k: np.any(top_indices[:, :k] == labels[:, None], axis=1).astype(float) for k in ks}

        ctx_arr = self.batch_labels[contexts] if self.batch_labels is not None else contexts

        for split in np.unique(splits):
            mask = splits == split
            if mask.sum() == 0:
                continue
            df = pd.DataFrame({"context": ctx_arr[mask]})
            for k in ks:
                df[f"hit@{k}"] = hits[k][mask]

            ctx_order = sorted(df["context"].unique())
            fig, ax = plt.subplots(figsize=(max(8, len(ctx_order) * 1.2), 5))
            x = np.arange(len(ctx_order))
            width = 0.8 / len(ks)
            for i, k in enumerate(ks):
                vals = [df.loc[df["context"] == ctx, f"hit@{k}"].mean() for ctx in ctx_order]
                ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=f"R@{k}", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(ctx_order, rotation=45, ha="right")
            ax.set_ylabel("Recall@k")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Context")
            ax.legend(fontsize=8)

            overall = {k: df[f"hit@{k}"].mean() for k in ks}
            title_parts = [f"R@{k}={v:.3f}" for k, v in overall.items()]
            ax.set_title(f"{split} | Recall@k per context | Overall: {', '.join(title_parts)}")

            plt.tight_layout()
            self.logger.experiment.add_figure(
                tag=f"recall_at_k/{split}",
                figure=fig,
                global_step=self.current_epoch,
            )
            plt.close(fig)


    def _plt_f1_support_coverage(self, mode: str, mode_data: dict[str, np.ndarray]):
        """Scatter of per-class F1 vs support, colored by dataset coverage (n_contexts)."""
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from torchmetrics.functional.classification import multiclass_f1_score

        preds = mode_data.get(PREDICTION_KEYS.PREDICTION_KEY)
        if preds is None:
            return

        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        contexts = mode_data[REGISTRY_KEYS.BATCH_KEY]

        df = pd.DataFrame({"label": labels, "pred": preds, "context": contexts})

        n_classes = self.n_classes

        # Per-class F1
        labels_t = torch.tensor(df["label"].values)
        preds_t = torch.tensor(df["pred"].values).squeeze(-1)
        unknown_mask = preds_t >= n_classes
        cm_n_cls = n_classes
        if unknown_mask.any():
            preds_t = preds_t.masked_fill(unknown_mask, n_classes)
            cm_n_cls += 1

        f1_per_class = multiclass_f1_score(
            preds_t, labels_t, num_classes=cm_n_cls,
            ignore_index=n_classes, average="none",
        ).cpu().numpy()[:n_classes]

        # Per-class support
        support = df["label"].value_counts().reindex(range(n_classes), fill_value=0).values

        # Per-class dataset coverage = number of distinct contexts the class appears in
        coverage = df.groupby("label")["context"].nunique().reindex(range(n_classes), fill_value=0).values

        # Filter to classes that actually appear
        mask = support > 0
        f1_vals = f1_per_class[mask]
        sup_vals = support[mask]
        cov_vals = coverage[mask]
        cls_ids = np.where(mask)[0]

        if len(f1_vals) == 0:
            return

        # --- Main scatter: F1 vs support, colored by coverage ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: F1 vs Support (colored by coverage)
        ax = axes[0]
        norm = Normalize(vmin=cov_vals.min(), vmax=cov_vals.max())
        cmap = plt.cm.plasma
        sc = ax.scatter(
            sup_vals, f1_vals, c=cov_vals, s=20,
            cmap=cmap, norm=norm, alpha=0.7, linewidths=0,
        )
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01).set_label("Dataset coverage")
        ax.set_xlabel("Support (n cells)")
        ax.set_ylabel("F1")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{mode} | F1 vs Support (color = coverage)")

        # Panel 2: F1 vs Coverage
        ax = axes[1]
        ax.scatter(
            cov_vals + np.random.uniform(-0.15, 0.15, size=len(cov_vals)),
            f1_vals, c=np.log1p(sup_vals), s=20,
            cmap="viridis", alpha=0.7, linewidths=0,
        )
        fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=np.log1p(sup_vals).min(), vmax=np.log1p(sup_vals).max()), cmap="viridis"),
            ax=ax, pad=0.01,
        ).set_label("log(support)")
        ax.set_xlabel("Dataset coverage (n contexts)")
        ax.set_ylabel("F1")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{mode} | F1 vs Coverage (color = support)")

        # Panel 3: Coverage vs Support (colored by F1)
        ax = axes[2]
        f1_norm = Normalize(vmin=0, vmax=1)
        sc3 = ax.scatter(
            sup_vals, cov_vals + np.random.uniform(-0.15, 0.15, size=len(cov_vals)),
            c=f1_vals, s=20, cmap="RdYlGn", norm=f1_norm, alpha=0.7, linewidths=0,
        )
        fig.colorbar(ScalarMappable(norm=f1_norm, cmap="RdYlGn"), ax=ax, pad=0.01).set_label("F1")
        ax.set_xlabel("Support (n cells)")
        ax.set_ylabel("Dataset coverage (n contexts)")
        ax.set_xscale("log")
        ax.set_title(f"{mode} | Coverage vs Support (color = F1)")

        plt.tight_layout()
        self.logger.experiment.add_figure(
            tag=f"f1_support_coverage/{mode}",
            figure=fig,
            global_step=self.current_epoch,
        )
        plt.close(fig)

    def _plt_umap(self, mode: str, mode_data: dict[str, np.ndarray], current_epoch: int, z_key: str = MODULE_KEYS.Z_KEY):
        """Plot UMAP of latent space (with proxies) into TensorBoard."""
        import os
        import umap
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get shared latent embedding and annotation data, fall back to z if not in cached data
        embeddings = mode_data[z_key]
        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        covs = mode_data[REGISTRY_KEYS.BATCH_KEY]
        modes = pd.Categorical(mode_data[REGISTRY_KEYS.SPLIT_KEY])
        # Optional per-cell efficiency annotations for the eff/eff_hat row.
        # Reshaped to 1D and masked alongside embeddings/labels below.
        eff_ref = mode_data.get(REGISTRY_KEYS.CLS_EFF_KEY, None)
        eff_pred = mode_data.get('eff_hat', None)
        # abs to match the loss target (|weights|); predict-path weights are signed.
        eff_ref = np.abs(np.asarray(eff_ref).reshape(-1)) if eff_ref is not None else None
        eff_pred = np.asarray(eff_pred).reshape(-1) if eff_pred is not None else None

        # --- Subset: remove control cells
        if getattr(self.module, 'ctrl_class_idx') is not None:
            mask = labels != self.module.ctrl_class_idx
            embeddings, labels, covs, modes = embeddings[mask], labels[mask], covs[mask], modes[mask]
            if eff_ref is not None and len(eff_ref) == len(mask):
                eff_ref = eff_ref[mask]
            if eff_pred is not None and len(eff_pred) == len(mask):
                eff_pred = eff_pred[mask]
        # Drop eff annotations if lengths no longer line up with labels — the
        # row will be skipped rather than risk a plotting crash on bad shape.
        if eff_ref is not None and len(eff_ref) != len(labels):
            eff_ref = None
        if eff_pred is not None and len(eff_pred) != len(labels):
            eff_pred = None
        has_eff = eff_ref is not None and eff_pred is not None
        # Add batch annotation
        if self.batch_labels is not None:
            cov_idx = self.batch_labels
            covs = cov_idx[covs]
        else:
            cov_idx = sorted(np.unique(covs))
        # --- Add class proxies if they exist
        # The class proxies live in the FULL shared space, so they can only be overlaid on an
        # embedding of that width. Under a split latent z_key is a BLOCK (z_p 128, z_n 32) while
        # the proxies stay n_shared -- concatenating them raised a shape error that killed every
        # umap in the run, reported only as a caught "[Plot] Plotting failed" warning. Width check
        # rather than a name check, so any future sub-block is covered too.
        _proj_ok = (MODULE_KEYS.CLS_PROJ_KEY in mode_data
                    and np.asarray(mode_data[MODULE_KEYS.CLS_PROJ_KEY]).shape[-1]
                    == np.asarray(embeddings).shape[-1])
        if (MODULE_KEYS.CLS_PROJ_KEY in mode_data and z_key != MODULE_KEYS.Z_KEY
                and not _proj_ok and not getattr(self, '_umap_proj_warned', False)):
            self._umap_proj_warned = True
            logging.info(f'[Plot] {z_key!r} is {np.asarray(embeddings).shape[-1]}-d but the class '
                         f'proxies are '
                         f'{np.asarray(mode_data[MODULE_KEYS.CLS_PROJ_KEY]).shape[-1]}-d; '
                         f'plotting cells only (proxies are undefined inside a latent block).')
        if _proj_ok and z_key != MODULE_KEYS.Z_KEY:
            cls_proxies = mode_data[MODULE_KEYS.CLS_PROJ_KEY]
            # Get proxies (excluding null)
            cls_proxies = cls_proxies[:self.module.n_cls]
            # Aggregate into smaller shapes if needed
            if cls_proxies.ndim == 3:
                cls_proxies = cls_proxies.mean(1)
            # TRAINED PREFIX ONLY. cls_emb holds n_cls rows but only the first n_labels are ever
            # trained -- the rest are random-init placeholders for perturbations this model has
            # never seen (nothing feeds them from a prior yet). Including them puts a large blob of
            # random vectors into the UMAP fit, which dominates the embedding and pushes the real
            # atoms out into a ring -- the 'empty centre' is a plotting artifact, not structure.
            # Same reason the decode path must slice to [:n_labels] before running OMP.
            _n_tr = int(min(getattr(self.module, 'n_labels', cls_proxies.shape[0]),
                            cls_proxies.shape[0]))
            cls_proxies = cls_proxies[:_n_tr]
            # Get current number of available proxies for plotting
            n_proxies = cls_proxies.shape[0]

            # proxy label ids
            proxy_labels = np.arange(n_proxies)
            proxy_covs = np.repeat("proxy", n_proxies)
            proxy_modes = np.repeat("proxy", n_proxies)
            proxy_is_obs = np.array(
                [True] * self.module.n_labels + [False] * (self.module.n_cls - self.module.n_labels)
            )[:n_proxies]

            # --- combine into a unified DataFrame
            df = pd.DataFrame({
                "UMAP_input_idx": np.arange(len(labels) + n_proxies),
                "label_id": np.concatenate((labels, proxy_labels)),
                "cov": np.concatenate((covs, proxy_covs)),
                "mode": np.concatenate((modes, proxy_modes)),
                "is_proxy": np.concatenate((np.zeros(len(labels), dtype=bool), np.ones(n_proxies, dtype=bool))),
                "is_observed": np.concatenate((np.ones(len(labels), dtype=bool), proxy_is_obs))
            })
            # Combine latent embeddings with class proxies
            embeddings_all = np.concatenate((embeddings, cls_proxies), axis=0)
        else:
            # --- combine into a unified DataFrame
            df = pd.DataFrame({
                "UMAP_input_idx": np.arange(len(labels)),
                "label_id": labels,
                "cov": covs,
                "mode": modes,
                "is_proxy": np.zeros(len(labels), dtype=bool),
                "is_observed": np.ones(len(labels), dtype=bool)
            })
            # Combine latent embeddings with class proxies
            embeddings_all = embeddings

        # --- optional label mapping
        if "_code_to_label" in self.loss_kwargs:
            df["label"] = self.loss_kwargs["_code_to_label"][df['label_id']]
        else:
            df["label"] = df["label_id"].astype(str)
        # Attach per-cell eff annotations. Proxy rows (if present) get NaN so
        # the continuous scatter naturally skips them via the non-proxy mask.
        if has_eff:
            n_extra = len(df) - len(labels)
            pad = np.full(max(n_extra, 0), np.nan)
            df['eff_ref'] = np.concatenate([eff_ref, pad]) if n_extra > 0 else eff_ref
            df['eff_pred'] = np.concatenate([eff_pred, pad]) if n_extra > 0 else eff_pred

        # --- cached PCA → UMAP transform
        # UMAP's PyNNDescent knn construction scales with input dimension,
        # so on high-dim embeddings (e.g. ``z_shared`` at 4096) the fit
        # peaks at ~15-25 GB for ~50k rows and dominates the epoch-end
        # memory budget. PCA to ~50 dims first cuts UMAP's peak ~5× and
        # its time ~3× while preserving local structure (standard
        # scanpy/scvi recipe). The cache stores the fitted PCA + UMAP
        # reducers and the PCA-projected reference; drift detection now
        # runs on the projected reference (10s of MB) instead of the raw
        # high-dim one (hundreds of MB).
        cache = self.cache.setdefault("umap_cache", {})
        pca_key         = f"{z_key}_{mode}_pca"
        transformer_key = f"{z_key}_{mode}_umap_transformer"
        ref_key         = f"{z_key}_{mode}_umap_reference"

        umap_pca_dim = 50
        if embeddings_all.shape[1] > umap_pca_dim:
            from sklearn.decomposition import PCA
            cached_pca = cache.get(pca_key)
            # Refit PCA if the input dim changed; otherwise reuse the
            # cached fit and just transform.
            if (
                cached_pca is None
                or getattr(cached_pca, "components_", None) is None
                or cached_pca.components_.shape[1] != embeddings_all.shape[1]
            ):
                cached_pca = PCA(n_components=umap_pca_dim, random_state=42)
                cached_pca.fit(embeddings_all)
                cache[pca_key] = cached_pca
            umap_input = cached_pca.transform(embeddings_all).astype(np.float32)
        else:
            umap_input = embeddings_all.astype(np.float32)

        def _fresh_umap(x):
            r = umap.UMAP(n_components=2)
            return r, r.fit_transform(x)

        if transformer_key not in cache:
            reducer, emb_2d = _fresh_umap(umap_input)
            cache[transformer_key] = reducer
            cache[ref_key] = umap_input
        else:
            ref = cache[ref_key]
            if umap_input.shape == ref.shape:
                rel_diff = np.mean(np.abs(umap_input - ref)) / (np.mean(np.abs(ref)) + 1e-6)
                if rel_diff > 0.1:
                    reducer, emb_2d = _fresh_umap(umap_input)
                    cache[transformer_key] = reducer
                    cache[ref_key] = umap_input
                else:
                    reducer = cache[transformer_key]
                    emb_2d = reducer.transform(umap_input)
            else:
                reducer, emb_2d = _fresh_umap(umap_input)
                cache[transformer_key] = reducer
                cache[ref_key] = umap_input

        # Add umap coordinates to dataframe
        df["UMAP1"], df["UMAP2"] = emb_2d[:, 0], emb_2d[:, 1]
        # Add covariate ordering
        df["cov"] = pd.Categorical(df["cov"], cov_idx, ordered=True)
        
        # --- plotting helper
        def _scatter_base(ax, data, hue, title, legend=True, plot_proxy=True, proxy_color="black", proxy_legend=True, umap_key='UMAP'):
            data_sorted = pd.concat([
                data[~data.is_proxy],                                     # normal cells
                data[(data.is_proxy) & (~data.is_observed)],              # unobserved proxies
                data[(data.is_proxy) & (data.is_observed)],               # observed proxies (TOP)
            ])
            u1, u2 = f'{umap_key}1', f'{umap_key}2'
            sns.scatterplot(
                data=data_sorted[~data_sorted.is_proxy],
                x=u1, y=u2, hue=hue,
                s=6, alpha=0.5, ax=ax, legend=legend
            )
            # overlay proxies as crosses (always plot observed on top of unobserved)
            if plot_proxy:
                sns.scatterplot(
                    data=data_sorted[(data_sorted.is_proxy) & (~data_sorted.is_observed)],
                    x=u1, y=u2, hue=hue,
                    s=40, marker="X", ax=ax, legend=False,
                    edgecolor=proxy_color, linewidth=0.5
                )
                sns.scatterplot(
                    data=data_sorted[(data_sorted.is_proxy) & (data_sorted.is_observed)],
                    x=u1, y=u2, hue=hue,
                    s=40, marker="X", ax=ax, legend=proxy_legend,
                    edgecolor=proxy_color, linewidth=0.8
                )
            ax.set_title(title)
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            # Adjust legend
            if legend or proxy_legend:
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0,
                    markerscale=2,
                    fontsize="small"
                )

        # --- create figure grid
        N = 2
        if has_eff:
            N += 1  # extra row for eff (ref) and eff_hat (pred) continuous scatters
        D = int(N * 4)
        fig, axes = plt.subplots(N, 2, figsize=(12, D), dpi=150)

        # 1. by modes (splits)
        _scatter_base(axes[0, 0], df, "mode", f"Splits @ Epoch {current_epoch}", plot_proxy=False, proxy_legend=False)

        # 2. by class (overlay proxies)
        _scatter_base(axes[0, 1], df, "label", f"Classes @ Epoch {current_epoch}", legend=False, plot_proxy=False, proxy_legend=False)

        # 3. by observed vs unseen
        df_obs = df.copy()
        df_obs["obs_label"] = np.where(df_obs.is_observed, "Observed proxy", "Unseen proxy")
        # Ensure observed are plotted last
        df_obs["obs_label"] = pd.Categorical(df_obs["obs_label"], ["Unseen proxy", "Observed proxy"], ordered=True)
        _scatter_base(axes[1, 0], df_obs, "obs_label", f"Observed vs Unseen @ Epoch {current_epoch}")

        # 4. by covariates (contexts)
        _scatter_base(axes[1, 1], df, "cov", f"Contexts @ Epoch {current_epoch}", plot_proxy=False, proxy_legend=False)

        # 5. eff (reference) vs eff_hat (predicted) — shared vmin/vmax so the
        # two panels are directly comparable, robust to outliers via 1st/99th
        # percentile clipping.
        if has_eff:
            eff_row = N - 1
            cells = df[~df.is_proxy]
            vals = np.concatenate([cells['eff_ref'].to_numpy(), cells['eff_pred'].to_numpy()])
            vals = vals[~np.isnan(vals)]
            if vals.size:
                vmin, vmax = np.quantile(vals, [0.01, 0.99])
            else:
                vmin, vmax = None, None
            for ax, col, title in (
                (axes[eff_row, 0], 'eff_ref', f"Eff (ref |z|) @ Epoch {current_epoch}"),
                (axes[eff_row, 1], 'eff_pred', f"Eff (pred) @ Epoch {current_epoch}"),
            ):
                sc = ax.scatter(
                    cells['UMAP1'], cells['UMAP2'],
                    c=cells[col], cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    s=6, alpha=0.6,
                )
                ax.set_title(title)
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                fig.colorbar(sc, ax=ax, shrink=0.85)
        plt.tight_layout()
        self.logger.experiment.add_figure(f"{mode}_{z_key}_umap", fig, current_epoch)
        plt.close(fig)

        # Save data to file
        if self.save_cache:
            import expert.utils.plotting as pl
            umap_dir = os.path.join(self.logger.log_dir, 'data', 'umap')
            os.makedirs(umap_dir, exist_ok=True)
            csv_path = os.path.join(umap_dir, f"umap_data_epoch{current_epoch}.csv")
            df.to_csv(csv_path, index=False)
            # Update gif
            pl.create_umap_gif(umap_dir, hue='cov', duration=500, max_labels=10, loop=0, verbose=False)

