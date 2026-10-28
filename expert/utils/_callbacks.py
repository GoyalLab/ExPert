# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import time

import lightning.pytorch as pl
import torch

from lightning.pytorch.callbacks import Callback
from scvi.train._callbacks import LoudEarlyStopping, SaveCheckpoint

import logging
log = logging.getLogger(__name__)


class DelayedEarlyStopping(LoudEarlyStopping):
    def __init__(self, start_epoch: int = 5, *args, **kwargs):
        super().__init__(check_on_train_epoch_end=False, *args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        # Wait until we reach the start epoch
        if trainer.current_epoch < self.start_epoch:
            return  # do nothing yet
        super().on_validation_end(trainer, pl_module)


class SaveCheckPointWithIndices(SaveCheckpoint):
    """
    Extends existing checkpointing logic to also keep track of used indices to keep indices constant for resuming.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_save_checkpoint(self, trainer: pl.Trainer, *args) -> None:
        """Saves the model state on Lightning checkpoint saves."""
        # set a post-training state before saving
        model = trainer._model
        model.module.eval()
        model.is_trained_ = True
        model.trainer = trainer

        # Save split indices in checkpoint so resuming reuses the same split
        data_splitter = trainer.datamodule if trainer.datamodule is not None else getattr(model, 'data_splitter_', None)
        if data_splitter is not None:
            model._set_data_split_indices(data_splitter)

        monitor_candidates = self._monitor_candidates(trainer)
        save_path = self.format_checkpoint_name(monitor_candidates)
        # by default, the function above gives a .ckpt extension
        save_path = save_path.split(".ckpt")[0]
        # tells save() this is an epoch-end checkpoint: current_epoch is the running epoch's INDEX here
        model._saving_at_epoch_end = True
        try:
            model.save(save_path, save_anndata=False, overwrite=True)
        finally:
            model._saving_at_epoch_end = False

        model.module.train()
        model.is_trained_ = False


class ResumeTrainingStateCallback(Callback):
    """Restores optimizer, scheduler, and epoch/step counters from a saved checkpoint.

    Applied once on train start, then removes itself from the callback list.
    """
    def __init__(self, state: dict):
        super().__init__()
        self._state = state

    def on_train_start(self, trainer, pl_module):
        # Restore optimizer state
        optim = pl_module.optimizers()
        # Load into the REAL optimizer: pl_module.optimizers() is a LightningOptimizer wrapper, and
        # load_state_dict on the wrapper rebinds the wrapper's own `state` while the optimizer that step() uses
        # stays empty (a resumed run then restarted Adam from zero moments).
        optim = getattr(optim, 'optimizer', optim)
        if optim is not None and self._state.get('optimizer_state_dict'):
            optim.load_state_dict(self._state['optimizer_state_dict'])
            log.info('[Resume] Restored optimizer state')
        # Restore scheduler state
        schedulers = trainer.lr_scheduler_configs
        if schedulers and self._state.get('scheduler_state_dict'):
            sched_state = self._state['scheduler_state_dict']
            if isinstance(sched_state, dict):
                schedulers[0].scheduler.load_state_dict(sched_state)
                log.info('[Resume] Restored scheduler state')
        # Epoch counter: the number of COMPLETED epochs (= the index of the next one). Older training states only
        # carry 'epoch', whose meaning depended on where it was written; it is used as-is for those.
        done = self._state.get('epochs_completed')
        if done is None:
            done = self._state.get('epoch', 0)
        trainer.fit_loop.epoch_progress.current.completed = done
        trainer.fit_loop.epoch_progress.current.processed = done
        # Step counters: the logger's x-axis (_batches_that_stepped) and the optimizer steps behind
        # trainer.global_step, which step-keyed code reads.
        logged_step = self._state.get('global_step', 0)
        opt_steps = self._state.get('optimizer_steps')
        trainer.fit_loop.epoch_loop._batches_that_stepped = logged_step
        if opt_steps is not None:
            prog = trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total
            prog.ready = prog.completed = int(opt_steps)
            if hasattr(pl_module, 'module') and hasattr(pl_module.module, '_step'):
                pl_module.module._step = int(opt_steps)
        # Python-side state of the module and the training plan (prior-refit bookkeeping, carried-over loss/KL
        # kwargs, per-epoch caches) -- what a fresh process would otherwise start empty
        for owner, key in ((getattr(pl_module, 'module', None), 'module_state'), (pl_module, 'plan_state')):
            for k, v in (self._state.get(key) or {}).items():
                if owner is not None:
                    setattr(owner, k, v)
        # The plan's atom snapshot for `atoms/drift` is taken in its on_train_epoch_end, which runs AFTER an epoch-end
        # checkpoint is written; the weights do not change in between, so the restored atom table is that snapshot.
        _mod = getattr(pl_module, 'module', None)
        _ce = getattr(_mod, 'cls_emb', None)
        _W = getattr(getattr(_ce, 'embedding', None), 'weight', None)
        if _W is not None:
            _n = max(1, min(int(getattr(_mod, 'n_labels', _W.shape[0]) or _W.shape[0]), _W.shape[0]))
            pl_module._atom_prev = _W.detach()[:_n].float().cpu()
        # RNG streams, last: anything earlier in fit start may have drawn from them
        rng = self._state.get('rng_state')
        if rng:
            import random
            import numpy as np
            torch.set_rng_state(rng['torch'])
            np.random.set_state(rng['numpy'])
            random.setstate(rng['python'])
            if 'cuda' in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
        log.info(f'[Resume] Restored epochs_completed={done}, global_step={logged_step}, '
                 f'optimizer_steps={opt_steps}, rng={"yes" if rng else "no (legacy training state)"}')


class ThroughputProbe(Callback):
    """Diagnostic: is training GPU-compute-bound or stalling on the input
    pipeline (dataloader fetch + host->device copy + loop overhead)?

    Per train batch it times two spans:
      * ``compute`` = ``on_train_batch_start`` -> ``on_train_batch_end``, with a
        CUDA sync on both ends so it reflects real GPU work, not just kernel
        launches.
      * ``wait`` = previous ``on_train_batch_end`` -> this
        ``on_train_batch_start`` — Lightning fetches and moves the next batch
        to the device *before* ``on_train_batch_start``, so this captures
        everything the GPU idles on between steps.

    Every ``window`` batches it logs the means and the data-bound fraction
    ``wait / (wait + compute)``: ~0% means compute-bound (the GPU is the limit,
    more dataloader workers won't help); a large fraction means the GPU is
    idling for input. The first ``warmup`` batches are skipped (lazy init /
    cudnn autotune), and the validation/epoch-boundary gap is not counted.
    """

    def __init__(self, warmup: int = 20, window: int = 200):
        super().__init__()
        self.warmup = warmup
        self.window = window
        self._i = 0          # batches seen (across epochs)
        self._n = 0          # batches counted in the current window
        self._t_prev_end = None
        self._t_start = None
        self._wait = 0.0
        self._compute = 0.0

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_epoch_start(self, trainer, pl_module):
        # Drop the carry-over from validation / the epoch boundary so it isn't
        # misattributed as dataloader wait on the first batch of the epoch.
        self._t_prev_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        now = time.perf_counter()
        if self._t_prev_end is not None and self._i >= self.warmup:
            self._wait += now - self._t_prev_end
        self._t_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        now = time.perf_counter()
        if self._i >= self.warmup and self._t_start is not None:
            self._compute += now - self._t_start
            self._n += 1
        self._t_prev_end = now
        self._i += 1
        if self._n and self._n % self.window == 0:
            total = self._wait + self._compute
            frac = self._wait / total if total > 0 else 0.0
            if trainer.is_global_zero:
                log.info(
                    f"[ThroughputProbe] {self._n} batches | "
                    f"wait={self._wait / self._n * 1e3:.1f} ms/batch, "
                    f"compute={self._compute / self._n * 1e3:.1f} ms/batch | "
                    f"data-bound={frac:.0%} "
                    f"({'pipeline-stalled' if frac > 0.25 else 'compute-bound'})"
                )
            self._wait = 0.0
            self._compute = 0.0
            self._n = 0
