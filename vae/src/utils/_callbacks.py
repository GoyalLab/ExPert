from scvi.train._callbacks import LoudEarlyStopping


class DelayedEarlyStopping(LoudEarlyStopping):
    def __init__(self, start_epoch: int = 5, *args, **kwargs):
        super().__init__(check_on_train_epoch_end=False, *args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        # Wait until we reach the start epoch
        if trainer.current_epoch < self.start_epoch:
            return  # do nothing yet
        super().on_validation_end(trainer, pl_module)
