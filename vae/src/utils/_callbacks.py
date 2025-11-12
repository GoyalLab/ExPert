from lightning.pytorch.callbacks import Callback
from scvi.train._callbacks import LoudEarlyStopping

import logging


class DelayedEarlyStopping(LoudEarlyStopping):
    def __init__(self, start_epoch: int = 5, *args, **kwargs):
        super().__init__(check_on_train_epoch_end=False, *args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        # Wait until we reach the start epoch
        if trainer.current_epoch < self.start_epoch:
            return  # do nothing yet
        super().on_validation_end(trainer, pl_module)

class PeriodicTestCallback(Callback):
    def __init__(self, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        # TODO: debug this
        return
        if trainer.current_epoch % self.every_n_epochs == 0 and trainer.is_global_zero:
            trainer.test(model=pl_module, ckpt_path=None)
