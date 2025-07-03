import logging
from IPython import embed
import pandas as pd
import numpy as np
from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS
import torch
import torchmetrics.functional as tmf
from torchmetrics.classification import MulticlassConfusionMatrix

from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass, LossOutput
from scvi.train._metrics import ElboMetric
from scvi.train._constants import METRIC_KEYS

from typing import TYPE_CHECKING

from typing import Literal, Any
from umap import UMAP
from collections import Counter, defaultdict


def _compute_weight(
    epoch: int,
    step: int,
    n_epochs_warmup: int | None,
    n_steps_warmup: int | None,
    n_epochs_stall: int | None,
    max_weight: float | None = 1.0,
    min_weight: float | None = 0.0,
) -> float:
    """Computes the classification weight for the current step or epoch.

    If both `n_epochs_warmup` and `n_steps_warmup` are None, `max_weight` is returned.

    Parameters
    ----------
    epoch
        Current epoch.
    step
        Current step.
    n_epochs_warmup
        Number of training epochs to scale weight on classification loss from
        `min_weight` to `max_weight`.
    n_steps_warmup
        Number of training steps (minibatches) to scale weight on classification loss from
        `min_weight` to `max_weight`.
    max_weight
        Maximum scaling factor on classification loss during training.
    min_weight
        Minimum scaling factor on classification loss during training.
    """
    # Check min and max weights
    if min_weight is None:
        min_weight = 0.0
    if max_weight is None:
        max_weight = 0.0
    if min_weight > max_weight:
        raise ValueError(
            f"min_weight={min_weight} is larger than max_weight={max_weight}."
        )
    # Start warmup at this epoch
    if n_epochs_stall is not None:
        if epoch < n_epochs_stall:
            return 0.0
        else:
            epoch -= n_epochs_stall

    slope = max_weight - min_weight
    if n_epochs_warmup is not None:
        if epoch < n_epochs_warmup:
            return slope * (epoch / n_epochs_warmup) + min_weight
    elif n_steps_warmup is not None:
        if step < n_steps_warmup:
            return slope * (step / n_steps_warmup) + min_weight
    return max_weight


class SemiSupervisedTrainingPlan(TrainingPlan):
    """Lightning module task for SemiSupervised Training.

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
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 400,
        n_steps_cls_warmup: int | None = None,
        n_epochs_cls_warmup: int | None = 400,
        n_epochs_cls_stall: int | None = 100,
        max_cls_weight: float = 1.0,
        min_cls_weight: float = 0.0,
        n_steps_contr_warmup: int | None = None,
        n_epochs_contr_warmup: int | None = 400,
        n_epochs_contr_stall: int | None = 100,
        max_contr_weight: float = 1.0,
        min_contr_weight: float = 0.0,
        n_steps_align_warmup: int | None = None,
        n_epochs_align_warmup: int | None = None,
        n_epochs_align_stall: int | None = None,
        max_align_weight: float | None = None,
        min_align_weight: float | None = None,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation", "validation_classification_loss"
        ] = "elbo_validation",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        average: str = "macro",
        use_posterior_mean: Literal["train", "val", "both"] = "train",
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            compile=compile,
            compile_kwargs=compile_kwargs,
            **loss_kwargs,
        )
        # CLS params
        self.n_steps_cls_warmup = n_steps_cls_warmup
        self.n_epochs_cls_warmup = n_epochs_cls_warmup
        self.n_epochs_cls_stall = n_epochs_cls_stall
        self.max_cls_weight = max_cls_weight
        self.min_cls_weight = min_cls_weight
        self.use_posterior_mean = use_posterior_mean
        # Contrastive params
        self.n_steps_contr_warmup = n_steps_contr_warmup
        self.n_epochs_contr_warmup = n_epochs_contr_warmup
        self.n_epochs_contr_stall = n_epochs_contr_stall
        self.max_contr_weight = max_contr_weight
        self.min_contr_weight = min_contr_weight
        # Alignment params
        self.n_steps_align_warmup = n_steps_align_warmup
        self.n_epochs_align_warmup = n_epochs_align_warmup
        self.n_epochs_align_stall = n_epochs_align_stall
        self.max_align_weight = max_align_weight
        self.min_align_weight = min_align_weight
        # Misc
        self.n_classes = n_classes
        self.average = average
        self.plot_cm = self.loss_kwargs.pop("plot_cm", False)
        self.plot_umap = self.loss_kwargs.pop("plot_umap", False)
        # Initialize confusion matrix
        self.train_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
    
    def log_with_mode(self, key: str, value: Any, mode: str, **kwargs):
        """Log with mode."""
        # TODO: Include this with a base training plan
        self.log(f"{mode}_{key}", value, **kwargs)

    @property
    def classification_ratio(self):
        """Scaling factor on classification weight during training. Consider Jax"""
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_cls_warmup,
            self.n_steps_cls_warmup,
            self.n_epochs_cls_stall,
            self.max_cls_weight,
            self.min_cls_weight,
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )
    
    @property
    def alignment_loss_weight(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_align_warmup,
            self.n_steps_align_warmup,
            self.n_epochs_align_stall,
            self.max_align_weight,
            self.min_align_weight,
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    @property
    def contrastive_loss_weight(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_contr_warmup,
            self.n_steps_contr_warmup,
            self.n_epochs_contr_stall,
            self.max_contr_weight,
            self.min_contr_weight,
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    def compute_and_log_metrics(
        self, loss_output: LossOutput, metrics: dict[str, ElboMetric], mode: str
    ):
        """Computes and logs metrics."""
        super().compute_and_log_metrics(loss_output, metrics, mode)

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
        if isinstance(loss_output.kl_local, dict) and len(loss_output.kl_local.keys()) > 1:
            for k, kl in loss_output.kl_local.items():
                self.log(
                    f"{mode}_{k}",
                    kl.mean() if isinstance(kl, torch.Tensor) else kl,
                    on_epoch=True,
                    batch_size=loss_output.n_obs_minibatch,
                    prog_bar=True,
                )

        # no labeled observations in minibatch
        if loss_output.classification_loss is None:
            return

        classification_loss = loss_output.classification_loss
        true_labels = loss_output.true_labels.squeeze(-1)
        logits = loss_output.logits
        predicted_labels = torch.argmax(logits, dim=-1)

        # Update confusion matrix
        if mode == 'train':
            confusion_metric = self.train_confmat
        elif mode == 'val':
            confusion_metric = self.val_confmat
        else:
            confusion_metric = self.test_confmat
        confusion_metric.update(predicted_labels, true_labels)

        accuracy = tmf.classification.multiclass_accuracy(
            predicted_labels,
            true_labels,
            self.n_classes,
            average=self.average
        )
        f1 = tmf.classification.multiclass_f1_score(
            predicted_labels,
            true_labels,
            self.n_classes,
            average=self.average,
        )
        ce = tmf.classification.multiclass_calibration_error(
            logits,
            true_labels,
            self.n_classes,
        )

        self.log_with_mode(
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY,
            classification_loss,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.ACCURACY_KEY,
            accuracy,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.F1_SCORE_KEY,
            f1,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.CALIBRATION_ERROR_KEY,
            ce,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )

    def training_step(self, batch, batch_idx):
        """Training step for semi-supervised training."""
        # Potentially dangerous if batch is from a single dataloader with two keys
        if len(batch) == 2:
            full_dataset = batch[0]
            labelled_dataset = batch[1]
        else:
            full_dataset = batch
            labelled_dataset = None

        # Compute KL warmup
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        # Compute CL warmup
        self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
        self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight})
        if self.max_align_weight is not None and self.max_align_weight > 0:
            self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
        self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == "train" or self.use_posterior_mean == "both"})
        # Add external embedding
        if self.loss_kwargs.get('use_ext_emb', False) and REGISTRY_KEYS.CLS_EMB_KEY in self.loss_kwargs and labelled_dataset is not None:
            labelled_dataset[REGISTRY_KEYS.CLS_EMB_KEY] = self.loss_kwargs.pop(REGISTRY_KEYS.CLS_EMB_KEY)
        input_kwargs = {
            "labelled_tensors": labelled_dataset,
        }
        input_kwargs.update(self.loss_kwargs)
        _, _, loss_output = self.forward(full_dataset, loss_kwargs=input_kwargs)
        loss = loss_output.loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.log(
            "kl_weight",
            self.loss_kwargs['kl_weight'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.log(
            "classification_ratio",
            self.loss_kwargs['classification_ratio'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for semi-supervised training."""
        # Potentially dangerous if batch is from a single dataloader with two keys
        if len(batch) == 2:
            full_dataset = batch[0]
            labelled_dataset = batch[1]
        else:
            full_dataset = batch
            labelled_dataset = None

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
        self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight})
        if self.max_align_weight is not None and self.max_align_weight > 0:
            self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
        self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == "val" or self.use_posterior_mean == "both"})
        
        if self.loss_kwargs.get('use_ext_emb', False) and REGISTRY_KEYS.CLS_EMB_KEY in self.loss_kwargs and labelled_dataset is not None:
            labelled_dataset[REGISTRY_KEYS.CLS_EMB_KEY] = self.loss_kwargs.pop(REGISTRY_KEYS.CLS_EMB_KEY)

        input_kwargs = {
            "labelled_tensors": labelled_dataset,
        }
        input_kwargs.update(self.loss_kwargs)
        _, _, loss_output = self.forward(full_dataset, loss_kwargs=input_kwargs)
        loss = loss_output.loss
        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.compute_and_log_metrics(loss_output, self.val_metrics, "validation")

    def get_umap(self, embeddings, labels, title="UMAP Projection", figsize=(10, 8), plt_file=None):
        """Plots a UMAP projection of embeddings colored by labels."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        umap = UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        embeddings_2d = umap.fit_transform(embeddings)

        plt.figure(figsize=figsize)
        scatter = sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=labels,
            palette="tab10",
            legend="full",
            alpha=0.7,
        )
        scatter.set_title(title)
        scatter.set_xlabel("UMAP-1")
        scatter.set_ylabel("UMAP-2")
        plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def on_validation_epoch_end(self):
        """Plot class label confusion matrix and UMAP projection."""
        import pytorch_lightning as pl
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.current_epoch % 10 == 0:
            if self.plot_cm:
                confusion_matrix = self.val_confmat.compute()
                self.val_confmat.reset()
                
                # Log confusion matrix as a heatmap if using tensorboard
                if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                    fig = plt.figure(figsize=(10, 10), dpi=150)
                    sns.heatmap(confusion_matrix.cpu().numpy(), annot=False)
                    plt.title(f'Validation Confusion Matrix (Epoch: {self.current_epoch})')
                    plt.close()
                    
                    # Log figure to tensorboard
                    self.logger.experiment.add_figure(
                        'validation_confusion_matrix',
                        fig,
                        global_step=self.current_epoch
                    )
            if self.plot_umap:
                # Perform forward pass on the validation set to get embeddings
                full_dataset = self.trainer.datamodule.val_dataloader()
                embeddings = []
                predicted_labels = []

                for _, batch in enumerate(full_dataset):
                    if len(batch) == 2:
                        full_data = batch[0]
                    else:
                        full_data = batch

                    with torch.no_grad():
                        inference_outputs, _, loss_output = self.forward(full_data)
                        embeddings.append(inference_outputs[MODULE_KEYS.Z_KEY].cpu())
                        if self.loss_kwargs['classification_ratio'] != 0:
                            predicted_labels.append(torch.argmax(loss_output.logits, dim=-1).cpu())

                embeddings = torch.cat(embeddings, dim=0)
                if self.loss_kwargs['classification_ratio'] != 0:
                    predicted_labels = torch.cat(predicted_labels, dim=0)
                    predicted_labels = predicted_labels.cpu().numpy()
                else:
                    predicted_labels = None

                embeddings = embeddings.cpu().numpy()

                fig_umap = self.get_umap(embeddings, predicted_labels, title=f"UMAP Projection (Predicted Labels), Epoch: {self.current_epoch}")
                self.logger.experiment.add_figure("val_umap_projection", fig_umap, global_step=self.current_epoch)

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
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 400,
        n_steps_cls_warmup: int | None = None,
        n_epochs_cls_warmup: int | None = 400,
        n_epochs_cls_stall: int | None = 100,
        max_cls_weight: float = 1.0,
        min_cls_weight: float = 0.0,
        n_steps_contr_warmup: int | None = None,
        n_epochs_contr_warmup: int | None = 400,
        n_epochs_contr_stall: int | None = 100,
        max_contr_weight: float = 1.0,
        min_contr_weight: float = 0.0,
        use_contr_in_val: bool = False,
        n_steps_align_warmup: int | None = None,
        n_epochs_align_warmup: int | None = None,
        n_epochs_align_stall: int | None = None,
        max_align_weight: float | None = None,
        min_align_weight: float | None = None,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation", "validation_classification_loss"
        ] = "elbo_validation",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        average: str = "macro",
        use_posterior_mean: Literal["train", "val", "both"] | None = "val",
        log_class_distribution: bool = False,
        log_full_val: bool = True,
        freeze_encoder_epoch: int | None = None,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            compile=compile,
            compile_kwargs=compile_kwargs,
            **loss_kwargs,
        )
        # CLS params
        self.n_steps_cls_warmup = n_steps_cls_warmup
        self.n_epochs_cls_warmup = n_epochs_cls_warmup
        self.n_epochs_cls_stall = n_epochs_cls_stall
        self.max_cls_weight = max_cls_weight
        self.min_cls_weight = min_cls_weight
        self.use_posterior_mean = use_posterior_mean
        # Contrastive params
        self.n_steps_contr_warmup = n_steps_contr_warmup
        self.n_epochs_contr_warmup = n_epochs_contr_warmup
        self.n_epochs_contr_stall = n_epochs_contr_stall
        self.max_contr_weight = max_contr_weight
        self.min_contr_weight = min_contr_weight
        self.use_contr_in_val = use_contr_in_val
        self.log_class_distribution = log_class_distribution
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.encoder_freezed = False
        # Alignment params
        self.n_steps_align_warmup = n_steps_align_warmup
        self.n_epochs_align_warmup = n_epochs_align_warmup
        self.n_epochs_align_stall = n_epochs_align_stall
        self.max_align_weight = max_align_weight
        self.min_align_weight = min_align_weight
        # Misc
        self.n_classes = n_classes
        self.average = average
        self.plot_cm = self.loss_kwargs.pop("plot_cm", False)
        self.plot_umap = self.loss_kwargs.pop("plot_umap", None)
        self.log_full_val = log_full_val
        # Initialize confusion matrix
        self.train_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.full_val_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        # Save classes that have been seen during training
        self.train_class_counts = []
        self.class_batch_counts = defaultdict(list)
    
    def log_with_mode(self, key: str, value: Any, mode: str, **kwargs):
        """Log with mode."""
        # TODO: Include this with a base training plan
        self.log(f"{mode}_{key}", value, **kwargs)

    @property
    def classification_ratio(self):
        """Scaling factor on classification weight during training. Consider Jax"""
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_cls_warmup,
            self.n_steps_cls_warmup,
            self.n_epochs_cls_stall,
            self.max_cls_weight,
            self.min_cls_weight,
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )
    
    @property
    def alignment_loss_weight(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_align_warmup,
            self.n_steps_align_warmup,
            self.n_epochs_align_stall,
            self.max_align_weight,
            self.min_align_weight,
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    @property
    def contrastive_loss_weight(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_contr_warmup,
            self.n_steps_contr_warmup,
            self.n_epochs_contr_stall,
            self.max_contr_weight,
            self.min_contr_weight,
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    def compute_and_log_metrics(
        self, loss_output: LossOutput, metrics: dict[str, ElboMetric], mode: str
    ):
        """Computes and logs metrics."""
        super().compute_and_log_metrics(loss_output, metrics, mode)

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
        if isinstance(loss_output.kl_local, dict) and len(loss_output.kl_local.keys()) > 1:
            for k, kl in loss_output.kl_local.items():
                self.log(
                    f"{mode}_{k}",
                    kl.mean() if isinstance(kl, torch.Tensor) else kl,
                    on_epoch=True,
                    batch_size=loss_output.n_obs_minibatch,
                    prog_bar=True,
                )
        
        # no classification loss (yet)
        if loss_output.classification_loss is None:
            return

        classification_loss = loss_output.classification_loss
        true_labels = loss_output.true_labels.squeeze(-1)
        logits = loss_output.logits
        predicted_labels = torch.argmax(logits, dim=-1)

        # Update confusion matrix
        if mode == 'train':
            confusion_metric = self.train_confmat
        elif mode == 'val':
            confusion_metric = self.val_confmat
        else:
            confusion_metric = self.test_confmat
        confusion_metric.update(predicted_labels, true_labels)

        accuracy = tmf.classification.multiclass_accuracy(
            predicted_labels,
            true_labels,
            self.n_classes,
            average=self.average
        )
        f1 = tmf.classification.multiclass_f1_score(
            predicted_labels,
            true_labels,
            self.n_classes,
            average=self.average,
        )
        ce = tmf.classification.multiclass_calibration_error(
            logits,
            true_labels,
            self.n_classes,
        )

        self.log_with_mode(
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY,
            classification_loss,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.ACCURACY_KEY,
            accuracy,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.F1_SCORE_KEY,
            f1,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.log_with_mode(
            METRIC_KEYS.CALIBRATION_ERROR_KEY,
            ce,
            mode,
            on_step=False,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )

    def training_step(self, batch, batch_idx):
        """Training step for supervised training."""
        # Compute KL warmup
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        # Compute CL warmup
        self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
        self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight})
        if self.max_align_weight is not None and self.max_align_weight > 0:
            self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
        self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == "train" or self.use_posterior_mean == "both"})
        # Setup kwargs
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)
        # Add external embedding to batch
        if input_kwargs.get('use_ext_emb', False) and REGISTRY_KEYS.CLS_EMB_KEY in input_kwargs:
            batch[REGISTRY_KEYS.CLS_EMB_KEY] = input_kwargs.pop(REGISTRY_KEYS.CLS_EMB_KEY)
        _, _, loss_output = self.forward(batch, loss_kwargs=input_kwargs)
        loss = loss_output.loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.log(
            "kl_weight",
            self.loss_kwargs['kl_weight'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.log(
            "classification_ratio",
            self.loss_kwargs['classification_ratio'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        y = loss_output.true_labels.squeeze(-1)
        # Count occurrences of each class in this batch
        unique, counts = torch.unique(y, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            self.class_batch_counts[u].append(c)
        # Save number of classes / batch
        self.train_class_counts.append(len(unique))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for supervised training."""
        # Compute KL warmup
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        # Compute CL warmup
        self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
        # Compute contrastive weight
        self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight if self.use_contr_in_val else 0.0})
        if self.max_align_weight is not None and self.max_align_weight > 0:
            self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
        self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == "train" or self.use_posterior_mean == "both"})
        # Setup kwargs
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)
        # Add external embedding to batch
        if input_kwargs.get('use_ext_emb', False) and REGISTRY_KEYS.CLS_EMB_KEY in input_kwargs:
            batch[REGISTRY_KEYS.CLS_EMB_KEY] = input_kwargs.pop(REGISTRY_KEYS.CLS_EMB_KEY)
        _, _, loss_output = self.forward(batch, loss_kwargs=input_kwargs)
        loss = loss_output.loss
        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.compute_and_log_metrics(loss_output, self.val_metrics, "validation")

    def on_train_epoch_start(self):
        if self.freeze_encoder_epoch is not None and self.current_epoch >= self.freeze_encoder_epoch and not self.encoder_frozen:
            self.freeze_encoder()
            self.encoder_frozen = True

    def freeze_encoder(self):
        self.module.z_encoder.eval()
        for param in self.module.z_encoder.parameters():
            param.requires_grad = False
    
    def on_train_epoch_end(self):
        if self.log_class_distribution:
            import matplotlib.pyplot as plt
            # Convert counter to dataframe and pad missing values with nan
            max_len = max(len(v) for v in self.class_batch_counts.values())
            df = pd.DataFrame({
                k: v + [np.nan] * (max_len - len(v))
                for k, v in self.class_batch_counts.items()
            })
            # Sort by class key to make comparable
            df = df[sorted(df.columns)]
            fig, ax = plt.subplots(figsize=(10, 5))
            df.boxplot(ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Samples per Batch")
            ax.set_title("Per-Class Distribution of Samples per Batch")
            self.logger.experiment.add_figure("train_class_batch_distribution", fig, self.current_epoch)
            plt.close(fig)

            # Reset for next epoch
            self.class_batch_counts = defaultdict(list)

            # Plot distribution of number of classes / batch
            df = pd.DataFrame({'n_classes': self.train_class_counts})
            fig, ax = plt.subplots(figsize=(5, 10))
            df.boxplot(ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Number of classes in Batch")
            ax.set_title("Distribution of Number of Classes per Batch")
            self.logger.experiment.add_figure("train_class_distribution", fig, self.current_epoch)
            plt.close(fig)
            # Reset for next epoch
            self.train_class_counts = []

        if self.plot_umap in ['train', 'both'] and self.current_epoch % 10 == 0:
            train_data = self._get_mode_data('train')
            self._plt_umap('train', train_data)

    # Calculate accuracy & f1 for entire validation set, not just per batch
    def on_validation_epoch_end(self):
        """Plot class label confusion matrix and UMAP projection."""
        import pytorch_lightning as pl
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.current_epoch % 10 == 0:
            if self.plot_cm:
                confusion_matrix = self.val_confmat.compute()
                self.val_confmat.reset()
                
                # Log confusion matrix as a heatmap if using tensorboard
                if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                    fig = plt.figure(figsize=(10, 10), dpi=150)
                    sns.heatmap(confusion_matrix.cpu().numpy(), annot=False)
                    plt.title(f'Validation Confusion Matrix (Epoch: {self.current_epoch})')
                    plt.close()
                    
                    # Log figure to tensorboard
                    self.logger.experiment.add_figure(
                        'validation_confusion_matrix',
                        fig,
                        global_step=self.current_epoch
                    )
            plt_umap = self.plot_umap in ['val', 'both']
            if plt_umap or self.log_full_val:
                # Get data from forward pass
                val_data = self._get_mode_data('val')
                # Plot UMAP for validation set
                if plt_umap:
                    self._plt_umap('val', val_data)
                # Calculate performance metrics over entire validation set, not just batch
                if self.log_full_val and self.classification_ratio > 0:
                    self._log_full_val_metrics(val_data)
                    
    def _log_full_val_metrics(self, mode_data: dict[str, np.ndarray]) -> None:
        true_labels = torch.tensor(mode_data.get('labels'))
        predicted_labels = torch.tensor(mode_data.get('predicted_labels'))

        accuracy = tmf.classification.multiclass_accuracy(
            predicted_labels,
            true_labels,
            self.n_classes,
            average=self.average
        )
        f1 = tmf.classification.multiclass_f1_score(
            predicted_labels,
            true_labels,
            self.n_classes,
            average=self.average,
        )
        self.log(
            "validation_full_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "validation_full_f1",
            f1,
            on_epoch=True,
            prog_bar=False,
        )
        

    def _get_mode_data(self, mode: str = 'val') -> dict[str, np.ndarray]:
        if mode == 'train':
            full_dataset = self.trainer.datamodule.train_dataloader()
        elif mode == 'val':
            full_dataset = self.trainer.datamodule.val_dataloader()
        else:
            full_dataset = None

        # Move batches to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def move_to_device(batch):
            if isinstance(batch, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                return type(batch)(move_to_device(b) for b in batch)
            elif isinstance(batch, torch.Tensor):
                return batch.to(device)
            else:
                return batch

        if full_dataset is not None:
            pass

        embeddings = []
        labels = []
        predicted_labels = []
        covs = []
        # Collect results for all batches
        for _, batch in enumerate(full_dataset):
            with torch.no_grad():
                # Compute KL warmup
                if "kl_weight" in self.loss_kwargs:
                    self.loss_kwargs.update({"kl_weight": self.kl_weight})
                # Compute CL warmup
                self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
                # Compute contrastive weight
                self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight if self.use_contr_in_val else 0.0})
                if self.max_align_weight is not None and self.max_align_weight > 0:
                    self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
                self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == mode or self.use_posterior_mean == "both"})
                # Setup kwargs
                input_kwargs = {}
                input_kwargs.update(self.loss_kwargs)
                # Add external embedding to batch
                if input_kwargs.get('use_ext_emb', False) and REGISTRY_KEYS.CLS_EMB_KEY in input_kwargs:
                    batch[REGISTRY_KEYS.CLS_EMB_KEY] = input_kwargs.pop(REGISTRY_KEYS.CLS_EMB_KEY)
                # Perform actual forward pass
                inference_outputs, _, loss_output = self.forward(move_to_device(batch), loss_kwargs=input_kwargs)
                # Save inference and generative output
                embeddings.append(inference_outputs[MODULE_KEYS.Z_KEY].cpu())
                covs.append(batch[REGISTRY_KEYS.BATCH_KEY].cpu())
                labels.append(loss_output.true_labels)
                # No classification (yet), skip batch
                if loss_output.logits is not None:
                    # Add predicted labels from each batch
                    predicted_labels.append(torch.argmax(loss_output.logits, dim=-1))
        # Concat results pull to cpu, and reformat
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy().squeeze()
        predicted_labels = torch.cat(predicted_labels, dim=0).cpu().numpy().squeeze() if len(predicted_labels) > 0 else np.array([])
        covs = torch.cat(covs, dim=0).cpu().numpy().squeeze()
        
        # Package data for return
        return {
            'embeddings': embeddings,
            'labels': labels,
            'predicted_labels': predicted_labels,
            'covs': covs
        }

    def _plt_umap(self, mode: str, mode_data: dict[str, np.ndarray]):
        # Extract data from forward pass
        embeddings = mode_data['embeddings']
        labels = mode_data['labels']
        # Look for actual label encoding
        if "_code_to_label" in self.loss_kwargs:
            labels = [self.loss_kwargs["_code_to_label"][l] for l in labels]
        labels = pd.Categorical(labels)
        covs = pd.Categorical(mode_data['covs'])

        """Plots a UMAP projection of Z latent space."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Use sc.tl.umap default arguments, apart from n_neighbors
        umap = UMAP(
            n_neighbors=5, min_dist=0.5, metric="euclidean",
            n_components=2, spread=1.0, negative_sample_rate=5
        )
        embeddings_2d = umap.fit_transform(embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        # Plot classes
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=labels,
            alpha=0.7,
            ax=axes[0]
        )
        axes[0].set_title(f"Classes @ Epoch: {self.current_epoch}")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")
        axes[0].legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            title="Classes",
        )
        # Plot batch keys
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=covs,
            alpha=0.7,
            ax=axes[1]
        )
        axes[1].set_title(f"Batch Keys @ Epoch: {self.current_epoch}")
        axes[1].set_xlabel("UMAP-1")
        axes[1].set_ylabel("UMAP-2")
        axes[1].legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            title="BATCH_KEY",
        )
        self.logger.experiment.add_figure(f"{mode}_z_umap", fig, self.current_epoch)
        plt.close(fig)
