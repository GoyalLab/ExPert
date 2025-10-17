import pandas as pd
import numpy as np
import torch
import torchmetrics.functional as tmf
import torch.nn.functional as F

from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS
from src.utils.common import to_tensor
from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass, LossOutput
from scvi.train._metrics import ElboMetric
from scvi.train._constants import METRIC_KEYS

from typing import Literal, Any
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

def _compute_weight(
    epoch: int,
    step: int,
    n_epochs_warmup: int | None,
    n_steps_warmup: int | None,
    n_epochs_stall: int | None = None,
    max_weight: float | None = 0.0,
    min_weight: float | None = 0.0,
    schedule: Literal["linear", "sigmoid", "exponential"] = "sigmoid",
    anneal_k: float = 10.0,
    invert: bool = False
) -> float:
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
        return max_weight if invert else 0.0
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
        t = step
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
        return max_weight - weight + min_weight
    else:
        return weight


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
        top_k: int = 1,
        use_posterior_mean: Literal["train", "val", "both"] | None = "val",
        use_cls_scores: Literal["train", "val", "both"] | None = "train",
        log_class_distribution: bool = False,
        log_full_val: bool = True,
        freeze_encoder_epoch: int | None = None,
        gene_emb: torch.Tensor | None = None,
        cls_emb: torch.Tensor | None = None,
        cls_sim: torch.Tensor | None = None,
        use_full_cls_emb: bool = False,
        incl_n_unseen: int | None = None,
        anneal_schedules: dict[str, dict[str | float]] | None = None,
        full_val_log_every_n_epoch: int = 10,
        use_contr_in_val: bool = False,
        multistage_kl_frac: float = 0.1,
        **loss_kwargs,
    ):
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
        self.anneal_schedules = anneal_schedules
        self.n_epochs_warmup = n_epochs_warmup
        self.n_steps_warmup = n_steps_warmup
        self.multistage_kl_frac = multistage_kl_frac
        self.kl_reset_epochs = self._get_stall_epochs()
        # CLS params
        self.n_classes = n_classes
        self.use_posterior_mean = use_posterior_mean
        self.gene_emb = to_tensor(gene_emb)
        self.cls_emb = to_tensor(cls_emb)
        self.cls_sim = self._calc_cls_sim(self.cls_emb) if cls_sim is None else to_tensor(cls_sim)
        self.use_full_cls_emb = use_full_cls_emb
        # Add train class embedding
        if not use_full_cls_emb and self.cls_emb is not None:
            self.train_cls_emb = self.cls_emb[:self.n_classes,:]
            self.train_cls_sim = self.train_cls_emb @ self.train_cls_emb.T
        # Get number of total classes in embedding
        self.n_all_classes = self.cls_emb.shape[0] if self.cls_emb is not None else self.n_classes
        if self.cls_emb is None:
            logging.warning(f'No external class embeddings provided, falling back to internal class embeddings.')
        self.incl_n_unseen = incl_n_unseen
        self.use_cls_scores = use_cls_scores
        # Contrastive params
        self.log_class_distribution = log_class_distribution
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.encoder_freezed = False
        self.use_contr_in_val = use_contr_in_val
        # Misc
        self.average = average                                                      # How to reduce the classification metrics
        self.top_k = top_k                                                          # How many top predictions to consider for metrics
        self.plot_cm = self.loss_kwargs.pop("plot_cm", False)
        self.plot_umap = self.loss_kwargs.pop("plot_umap", None)
        self.log_full_val = log_full_val
        self.full_val_log_every_n_epoch = full_val_log_every_n_epoch
        # Save classes that have been seen during training
        self.train_class_counts = []
        self.class_batch_counts = defaultdict(list)
        self.observed_classes = set()
        # Cache
        self.cache = {}

    def _get_stall_epochs(self) -> dict[str, int]:
        stalls = {'kl': 0}
        for name, schedule in self.anneal_schedules.items():
            stall = schedule.get('n_epochs_stall', 0)
            if stall > 0:
                stalls[name] = stall
        return stalls
    
    def log_with_mode(self, key: str, value: Any, mode: str, **kwargs):
        """Log with mode."""
        if isinstance(value, torch.Tensor):
            value = value.detach()
        # TODO: Include this with a base training plan
        self.log(f"{mode}_{key}", value, **kwargs)

    @property
    def rl_weight(self):
        """Scaling factor on classification weight during training. Consider Jax"""
        kwargs = self.anneal_schedules.get('rl_weight', {'min_weight': 1.0, 'max_weight': 1.0})
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            self.n_epochs_warmup,
            self.n_steps_warmup,
            **kwargs
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    @property
    def kl_weight_multistage(self):
        """KL weight for forward process. Resets every time a new loss objective activates."""
        reset_at = np.array(sorted(self.kl_reset_epochs.values()))
        idx = (self.current_epoch > reset_at).sum() - 1
        n_epochs_stall = reset_at[idx]
        n_epochs_warmup = reset_at[idx+1] if idx+1 < len(reset_at) else self.n_epochs_warmup
        # No new loss
        if n_epochs_stall < 1:
            min_weight = self.min_kl_weight
        # New loss
        else:
            min_weight = self.max_kl_weight * self.multistage_kl_frac if self.multistage_kl_frac > 0 else self.min_kl_weight
        # Set KL weight to restart at multistage min (0.1) for every new loss that gets activated
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            n_epochs_warmup,
            self.n_steps_warmup,
            n_epochs_stall=n_epochs_stall,
            min_weight=min_weight,
            max_weight=self.max_kl_weight
        )
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    @property
    def classification_ratio(self):
        """Scaling factor on classification weight during training. Consider Jax"""
        # Init basic args
        sched_kwargs = {'epoch': self.current_epoch, 'step': self.global_step, 'n_epochs_warmup': self.n_epochs_warmup, 'n_steps_warmup': self.n_steps_warmup}
        # Add scheduling params if given
        kwargs = self.anneal_schedules.get('classification_ratio', {})
        sched_kwargs.update(kwargs)
        klw = _compute_weight(**sched_kwargs)
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )
    
    @property
    def alignment_loss_weight(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        # Init basic args
        sched_kwargs = {'epoch': self.current_epoch, 'step': self.global_step, 'n_epochs_warmup': self.n_epochs_warmup, 'n_steps_warmup': self.n_steps_warmup}
        # Add scheduling params if given
        kwargs = self.anneal_schedules.get('alignment_loss_weight', {})
        sched_kwargs.update(kwargs)
        klw = _compute_weight(**sched_kwargs)
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    @property
    def contrastive_loss_weight(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        # Init basic args
        sched_kwargs = {'epoch': self.current_epoch, 'step': self.global_step, 'n_epochs_warmup': self.n_epochs_warmup, 'n_steps_warmup': self.n_steps_warmup}
        # Add scheduling params if given
        kwargs = self.anneal_schedules.get('contrastive_loss_weight', {})
        sched_kwargs.update(kwargs)
        klw = _compute_weight(**sched_kwargs)
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )
    
    @property
    def class_kl_temperature(self):
        """Scaling factor on contrastive weight during training. Consider Jax"""
        # Init basic args
        sched_kwargs = {'epoch': self.current_epoch, 'step': self.global_step, 'n_epochs_warmup': self.n_epochs_warmup, 'n_steps_warmup': self.n_steps_warmup}
        # Add scheduling params if given
        kwargs = self.anneal_schedules.get('class_kl_temperature', {})
        sched_kwargs.update(kwargs)
        klw = _compute_weight(**sched_kwargs)
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )
    
    @property
    def adversarial_context_lambda(self):
        """Lambda for adversial context loss during training. Consider Jax"""
        # Init basic args
        sched_kwargs = {'epoch': self.current_epoch, 'step': self.global_step, 'n_epochs_warmup': self.n_epochs_warmup, 'n_steps_warmup': self.n_steps_warmup}
        # Add scheduling params if given
        kwargs = self.anneal_schedules.get('adversarial_context_lambda', {})
        sched_kwargs.update(kwargs)
        klw = _compute_weight(**sched_kwargs)
        return (
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

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
        if isinstance(loss_output.kl_local, dict) and len(loss_output.kl_local.keys()) > 1:
            for k, kl in loss_output.kl_local.items():
                self.log(
                    f"{mode}_{k}",
                    kl.mean() if isinstance(kl, torch.Tensor) else kl,
                    on_epoch=True,
                    batch_size=loss_output.n_obs_minibatch,
                    prog_bar=True,
                )
        # Initialize default classification metrics
        metrics_dict = {
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY: np.inf,
            METRIC_KEYS.ACCURACY_KEY: 0.0,
            METRIC_KEYS.F1_SCORE_KEY: 0.0,
        }
        # Add classification-based metrics if they exist
        if loss_output.classification_loss is not None:
            classification_loss = loss_output.classification_loss
            true_labels = loss_output.true_labels.squeeze(-1)
            logits = loss_output.logits
            # Take argmax of logits to get highest predicted label
            predicted_labels = torch.argmax(logits, dim=-1).squeeze(-1)
            n_classes = self.n_classes
            # Exclude control class for multiclass metrics
            if not self.module.use_classification_control:
                n_classes -= 1
            # Check if any classes are outside of training classes, i.e
            unknown_mask = (predicted_labels >= n_classes)
            if unknown_mask.any():
                # Assign unknown to classes that are outside of the training classes
                predicted_labels = predicted_labels.masked_fill(unknown_mask, n_classes)
                n_classes += 1
            # Assign unknown to classes that are outside of the training classes
            predicted_labels = predicted_labels.masked_fill((predicted_labels >= n_classes), n_classes)
            # Calculate accuracy over predicted and true labels
            accuracy = tmf.classification.multiclass_accuracy(
                predicted_labels,
                true_labels,
                n_classes,
                average=self.average
            )
            # Calculate F1 score over predicted and true labels
            f1 = tmf.classification.multiclass_f1_score(
                predicted_labels,
                true_labels,
                n_classes,
                average=self.average,
            )
            # Update metrics
            metrics_dict[METRIC_KEYS.CLASSIFICATION_LOSS_KEY] = classification_loss
            metrics_dict[METRIC_KEYS.ACCURACY_KEY] = accuracy
            metrics_dict[METRIC_KEYS.F1_SCORE_KEY] = f1
            # Include top k predictions as well
            if self.top_k > 1:
                top_k_acc_key = f'{METRIC_KEYS.ACCURACY_KEY}_top_{self.top_k}'
                top_k_f1_key = f'{METRIC_KEYS.F1_SCORE_KEY}_top_{self.top_k}'
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
                metrics_dict[top_k_acc_key] = top_k_accuracy
                metrics_dict[top_k_f1_key] = top_k_f1
        
        # Add metrics to extra metrics for internal logging
        loss_output.extra_metrics.update(metrics_dict)
        # Compute and log all metrics
        super().compute_and_log_metrics(loss_output, metrics, mode)

    def _get_batch_class_embedding(self, batch_idx: int | None = None, **kwargs) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cls_emb is None:
            return None, None
        # Use full embedding
        cls_emb = self.cls_emb
        cls_sim = self.cls_sim
        if self.use_full_cls_emb:
            # Randomly subsample embeddings outside of training classes if there are more available
            if self.incl_n_unseen is not None and self.incl_n_unseen > 0 and self.n_all_classes > self.n_classes:
                # Seen class indices
                seen_idx = torch.arange(self.n_classes, device=cls_emb.device)
                # Unseen class indices
                unseen_idx = torch.arange(self.n_classes, self.n_all_classes, device=cls_emb.device)
                # Fix RNG seed based on batch_idx to make sampling deterministic per batch (optional)
                if batch_idx is not None:
                    g = torch.Generator(device=cls_emb.device)
                    g.manual_seed(batch_idx + 1234)  # offset for reproducibility
                    unseen_sample = unseen_idx[
                        torch.randperm(len(unseen_idx), generator=g)[:self.incl_n_unseen]
                    ]
                else:
                    unseen_sample = unseen_idx[
                        torch.randperm(len(unseen_idx))[:self.incl_n_unseen]
                    ]
                # Combine seen + sampled unseen
                selected_idx = torch.cat([seen_idx, unseen_sample], dim=0)
                # Subset embeddings and similarity matrix
                cls_emb = cls_emb[selected_idx]
                cls_sim = cls_sim[selected_idx][:, selected_idx]

            return cls_emb, cls_sim
        # Or use fixed part of embedding
        else:
            cls_emb = self.train_cls_emb
            cls_sim = self.train_cls_sim
        # Add learnable control embedding if given and re-calculate class similarities
        if self.module.ctrl_class_idx is not None:
            # Clone to avoid modifying the buffer in-place
            cls_emb = cls_emb.clone()
            # Ensure control_emb is on same device and normalized
            control_emb = F.normalize(self.module.control_emb, dim=-1).to(cls_emb.device)
            # Replace single control embedding with learnable embedding
            cls_emb[self.module.ctrl_class_idx] = control_emb.squeeze(0)
            # Normalize entire embedding matrix for cosine-similarity
            cls_emb = F.normalize(cls_emb, dim=-1)
            cls_sim = cls_emb @ cls_emb.T
        return cls_emb, cls_sim

    def _calc_cls_sim(self, cls_emb: torch.Tensor | None) -> torch.Tensor | None:
        if cls_emb is None:
            return None
        cls_emb = F.normalize(cls_emb, dim=-1)
        return cls_emb @ cls_emb.T

    def step(self, mode, batch, batch_idx):
        """Step for supervised training"""
        # Compute RL weight
        self.loss_kwargs.update({"rl_weight": self.rl_weight})
        # Compute KL warmup
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight_multistage})
        # Compute CL warmup
        self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
        if mode=='train' or self.use_contr_in_val:
            self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight})
        else:
            self.loss_kwargs.update({"contrastive_loss_weight": 0.0})
        self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
        self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == mode or self.use_posterior_mean == "both"})
        self.loss_kwargs.update({"class_kl_temperature": self.class_kl_temperature})
        self.loss_kwargs.update({"adversarial_context_lambda": self.adversarial_context_lambda})
        # Setup kwargs
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)
        # Add external gene embedding to batch
        if self.gene_emb is not None:
            # Add gene embedding matrix to batch
            batch[REGISTRY_KEYS.GENE_EMB_KEY] = self.gene_emb
        # Add external embedding to batch
        if self.cls_emb is not None:
            # Add embedding for classes
            batch[REGISTRY_KEYS.CLS_EMB_KEY], batch[REGISTRY_KEYS.CLS_SIM_KEY] = self._get_batch_class_embedding(batch_idx=batch_idx)
        # Choose to use class scores for scaling or not
        input_kwargs['use_cls_scores'] = self.use_cls_scores in [mode, 'both']
        # Perform full forward pass of model
        inference, generative, loss_output = self.forward(batch, loss_kwargs=input_kwargs)
        return inference, generative, loss_output
        
    def training_step(self, batch, batch_idx):
        """Training step for supervised training."""
        # Perform full forward pass of model
        _, _, loss_output = self.step(mode='train', batch=batch, batch_idx=batch_idx)
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
        self.log(
            "class_kl_temperature",
            self.loss_kwargs['class_kl_temperature'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.log(
            "constrastive_loss_weight",
            self.loss_kwargs['contrastive_loss_weight'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.log(
            "alignment_loss_weight",
            self.loss_kwargs['alignment_loss_weight'],
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=False,
        )
        self.log(
            "logit_temperature",
            1.0 / self.module.logit_scale.clamp(0, 4.6).exp(),
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
        # Perform full forward pass of model
        _, _, loss_output = self.step(mode='val', batch=batch, batch_idx=batch_idx)
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
        """Plot full validation set UMAP projection."""
        if self.full_val_log_every_n_epoch > 0 and self.current_epoch % self.full_val_log_every_n_epoch == 0:
            plt_umap = self.plot_umap in ['val', 'both']
            if plt_umap or self.log_full_val:
                # Get data from forward pass
                self.module.use_counter = True
                val_data = self._get_mode_data('val')
                self.module.use_counter = False
                # Plot UMAP for validation set
                if plt_umap:
                    self._plt_umap('val', val_data)
                # Calculate performance metrics over entire validation set, not just batch
                if self.log_full_val and self.classification_ratio > 0:
                    self._log_full_val_metrics(val_data)
                    
    def _log_full_val_metrics(self, mode_data: dict[str, np.ndarray]) -> None:
        true_labels = torch.tensor(mode_data.get('labels'))
        predicted_labels = torch.tensor(mode_data.get('predicted_labels')).squeeze(-1)
        n_classes = self.n_classes
        # Exclude control class for multiclass metrics
        if not self.module.use_classification_control:
            n_classes -= 1
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
        f1 = tmf.classification.multiclass_f1_score(
            predicted_labels,
            true_labels,
            n_classes,
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
        # Include top k predictions as well
        if self.top_k > 1:
            logits = torch.tensor(mode_data.get('logits'))
            top_k_acc_key = f'validation_full_accuracy_top_{self.top_k}'
            top_k_f1_key = f'validation_full_f1{self.top_k}'
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
        logits = []
        # Collect results for all batches
        for batch_idx, batch in enumerate(full_dataset):
            with torch.no_grad():
                # Compute KL warmup
                if "kl_weight" in self.loss_kwargs:
                    self.loss_kwargs.update({"kl_weight": self.kl_weight})
                # Compute CL warmup
                self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
                # Compute contrastive weight
                self.loss_kwargs.update({"contrastive_loss_weight": self.contrastive_loss_weight if self.use_contr_in_val else 0.0})
                self.loss_kwargs.update({"alignment_loss_weight": self.alignment_loss_weight})
                self.loss_kwargs.update({"use_posterior_mean": self.use_posterior_mean == mode or self.use_posterior_mean == "both"})
                self.loss_kwargs.update({"class_kl_temperature": self.class_kl_temperature})
                # Setup kwargs
                input_kwargs = {}
                input_kwargs.update(self.loss_kwargs)
                # Add external embedding to batch
                if self.cls_emb is not None:
                    # Add embedding for classes that are not in the training data
                    batch[REGISTRY_KEYS.CLS_EMB_KEY], batch[REGISTRY_KEYS.CLS_SIM_KEY] = self._get_batch_class_embedding(batch_idx=batch_idx)
                # Add external gene embedding to batch
                if self.gene_emb is not None:
                    # Add gene embedding matrix to batch
                    batch[REGISTRY_KEYS.GENE_EMB_KEY] = self.gene_emb
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
                    logits.append(loss_output.logits)
        # Concat results pull to cpu, and reformat
        embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy().squeeze()
        predicted_labels = torch.cat(predicted_labels, dim=0).cpu().numpy().squeeze() if len(predicted_labels) > 0 else np.array([])
        logits = torch.cat(logits, dim=0).cpu().numpy().squeeze() if len(logits) > 0 else np.array([])
        covs = torch.cat(covs, dim=0).cpu().numpy().squeeze()
        
        # Package data for return
        return {
            'embeddings': embeddings,
            'labels': labels,
            'predicted_labels': predicted_labels,
            'logits': logits,
            'covs': covs
        }

    def _plt_umap(self, mode: str, mode_data: dict[str, np.ndarray]):
        import anndata as ad
        import scanpy as sc
        # Extract data from forward pass
        embeddings = mode_data['embeddings']
        labels = mode_data['labels']
        # Look for actual label encoding
        if "_code_to_label" in self.loss_kwargs:
            labels = [self.loss_kwargs["_code_to_label"][l] for l in labels]
        labels = pd.Categorical(labels)
        covs = pd.Categorical(mode_data['covs'])
        # Plot umap consistent with scanpy settings
        obs = pd.DataFrame({'labels': labels, 'covs': covs})
        x = ad.AnnData(X=embeddings, obs=obs)
        sc.pp.neighbors(x, use_rep='X')
        sc.tl.umap(x)
        embeddings_2d = x.obsm['X_umap']

        """Plots a UMAP projection of Z latent space."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        # Plot classes
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=labels,
            alpha=0.7,
            s=4,
            ax=axes[0],
            legend=False
        )
        axes[0].set_title(f"Classes @ Epoch: {self.current_epoch}")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")
        # Plot batch keys
        sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=covs,
            alpha=0.7,
            s=4,
            ax=axes[1],
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
