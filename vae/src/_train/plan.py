import logging
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
        *,
        classification_ratio: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 400,
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
        plot_cm: bool = True,
        plot_umap: bool = False,
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
        self.loss_kwargs.update({"classification_ratio": classification_ratio})
        self.n_classes = n_classes
        self.average = average
        self.plot_cm = plot_cm
        self.plot_umap = plot_umap
        # Initialize confusion matrix
        self.train_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=n_classes)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=n_classes)

    def log_with_mode(self, key: str, value: Any, mode: str, **kwargs):
        """Log with mode."""
        # TODO: Include this with a base training plan
        self.log(f"{mode}_{key}", value, **kwargs)

    def plot_confusion(self, y_true, y_pred, figsize=(10, 8), hm_kwargs={'annot': False}, verbose=False, plt_file=None):
        from sklearn.metrics import confusion_matrix
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get class labels (for multiclass classification, this will correspond to unique labels)
        class_labels = np.unique(y_pred)
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        # Plot confusion matrix using seaborn heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(cm, xticklabels=class_labels, yticklabels=class_labels, **hm_kwargs)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        # Return the plot instead of showing it
        fig = plt.gcf()  # Get the current figure
        plt.close(fig)  # Close the figure to prevent it from displaying
        return fig

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

        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
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
                    sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
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
