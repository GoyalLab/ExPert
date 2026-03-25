from lightning.pytorch.callbacks import Callback
from scvi.train._callbacks import LoudEarlyStopping

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
        # Restore epoch and global_step counters
        resumed_epoch = self._state.get('epoch', 0)
        resumed_step = self._state.get('global_step', 0)
        # Epoch counter
        trainer.fit_loop.epoch_progress.current.completed = resumed_epoch
        trainer.fit_loop.epoch_progress.current.processed = resumed_epoch
        # Global step — this is what the logger uses for the x-axis
        trainer.fit_loop.epoch_loop._batches_that_stepped = resumed_step
        log.info(f'[Resume] Restored epoch={resumed_epoch}, global_step={resumed_step}')


class PeriodicTestCallback(Callback):
    def __init__(self, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        # TODO: debug this
        return
        if trainer.current_epoch % self.every_n_epochs == 0 and trainer.is_global_zero:
            trainer.test(model=pl_module, ckpt_path=None)
