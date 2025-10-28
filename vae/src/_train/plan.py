import pandas as pd
import numpy as np
import torch
import torchmetrics.functional as tmf
import torch.nn.functional as F

from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS
from src.utils.io import to_tensor
from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass, LossOutput
from scvi.train._metrics import ElboMetric
from scvi.train._constants import METRIC_KEYS

from typing import Literal, Any
from collections import defaultdict


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
        incl_n_unseen: int | None = None,
        anneal_schedules: dict[str, dict[str | float]] | None = None,
        full_val_log_every_n_epoch: int = 10,
        use_contrastive_loader: Literal["train", "val", "both"] | None = "train",
        multistage_kl_frac: float = 0.1,
        multistage_kl_min: float = 1e-2,
        use_local_stage_warmup: bool = False,
        batch_labels: np.ndarray | None = None,
        # Caching options
        cache_n_epochs: int | None = 1,
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
        self.multistage_kl_min = multistage_kl_min
        self.kl_reset_epochs = self._get_stall_epochs()
        self.reset_kl_at = np.array(sorted(self.kl_reset_epochs.values()))
        self.kl_kwargs = self.anneal_schedules.get('kl_weight', {'min_weight': 0.0, 'max_weight': 1.0})
        self.use_local_stage_warmup = use_local_stage_warmup
        # CLS params
        self.n_classes = n_classes
        self.use_posterior_mean = use_posterior_mean
        self.gene_emb = to_tensor(gene_emb)
   
        self.incl_n_unseen = incl_n_unseen
        self.use_cls_scores = use_cls_scores
        # Contrastive params
        self.log_class_distribution = log_class_distribution
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.encoder_freezed = False
        self.use_contrastive_loader = use_contrastive_loader
        # Misc
        self.average = average                                                      # How to reduce the classification metrics
        self.top_k = top_k                                                          # How many top predictions to consider for metrics
        self.plot_cm = self.loss_kwargs.pop("plot_cm", False)
        self.plot_umap = self.loss_kwargs.pop("plot_umap", None)
        self.plot_kl_distribution = self.loss_kwargs.pop("plot_kl_distribution", False)
        self.plot_context_emb = self.loss_kwargs.pop("plot_context_emb", False)
        self.log_full_val = log_full_val
        self.full_val_log_every_n_epoch = full_val_log_every_n_epoch
        self.batch_labels = batch_labels
        # Save classes that have been seen during training
        self.train_class_counts = []
        self.class_batch_counts = defaultdict(list)
        self.observed_classes = set()
        # ----- Cache -----
        self.cache = {}
        # Cache for embeddings and metadata within epoch
        self.epoch_cache = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
        # Cache for historical data across epochs TODO: implement properly
        self.history_cache = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
        # Save n epochs in cache
        self.cache_n_epochs = cache_n_epochs

    def _cache_step_data(self, mode: str, batch: dict, inference_outputs: dict, loss_output: LossOutput):
        """Cache data from a single step."""
        # Only cache every n epoch
        if self.full_val_log_every_n_epoch > 0 and self.current_epoch % self.full_val_log_every_n_epoch == 0:
            self.epoch_cache[mode]['embeddings'].append(inference_outputs[MODULE_KEYS.Z_KEY].cpu())
            self.epoch_cache[mode]['labels'].append(loss_output.true_labels.cpu())
            self.epoch_cache[mode]['covs'].append(batch[REGISTRY_KEYS.BATCH_KEY].cpu())
            if loss_output.logits is not None:
                self.epoch_cache[mode]['predicted_labels'].append(torch.argmax(loss_output.logits, dim=-1).cpu())
                self.epoch_cache[mode]['logits'].append(loss_output.logits.cpu())

    def _process_epoch_cache(self, mode: str) -> dict[str, np.ndarray]:
        """Process and clear the epoch cache, returning concatenated arrays."""
        cache = self.epoch_cache[mode]
        if not cache:
            return {}
            
        # Concatenate all cached tensors
        processed = {
            'embeddings': torch.cat(cache['embeddings'], dim=0).numpy(),
            'labels': torch.cat(cache['labels'], dim=0).numpy().squeeze(),
            'covs': torch.cat(cache['covs'], dim=0).numpy().squeeze(),
        }
        
        if cache.get('predicted_labels', []):
            processed['predicted_labels'] = torch.cat(cache['predicted_labels'], dim=0).numpy().squeeze()
            processed['logits'] = torch.cat(cache['logits'], dim=0).numpy().squeeze()
        
        # Add to historical cache
        for k, v in processed.items():
            self.history_cache[mode][k].append(v)
            # Keep only last N epochs
            if len(self.history_cache[mode][k]) > self.cache_n_epochs:
                self.history_cache[mode][k].pop(0)
        
        # Clear epoch cache
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
        # Get extra kl args from 
        kwargs = self.kl_kwargs
        # Determine reset params
        reset_at = self.reset_kl_at
        n_epochs_stall = 0
        n_epochs_warmup = self.n_epochs_warmup
        # Handle stall schedules
        if reset_at.shape[0] > 1:
            # Check in which state we currently are
            idx = (self.current_epoch > reset_at).sum() - 1
            # Get next stage starting epoch
            n_epochs_stall = reset_at[idx]
            
            # KL goes from min to max within each stage
            if self.use_local_stage_warmup:
                n_epochs_warmup = reset_at[idx+1] if idx+1 < len(reset_at) else self.n_epochs_warmup
                min_weight = kwargs['min_weight'] if n_epochs_stall < 1 else kwargs['max_weight'] * self.multistage_kl_frac
                n_epochs_warmup = n_epochs_warmup - n_epochs_stall
            # Global behavior - KL resets to minimum and warms up continuously
            else:
                # Get starting KL weight
                min_weight = kwargs['min_weight']
                # Extend warmup period
                n_epochs_warmup = n_epochs_warmup + n_epochs_stall
                # Reset KL to multistage minimum at new stage
                if n_epochs_stall > 0:
                    min_weight = self.multistage_kl_min
                
            # Update min weight
            kwargs['min_weight'] = min_weight

        # Update warmup and stall epochs
        kwargs['n_epochs_warmup'] = n_epochs_warmup
        kwargs['n_steps_warmup'] = None
        kwargs['n_epochs_stall'] = n_epochs_stall
        
        # Calculate final KL weight
        klw = _compute_weight(
            self.current_epoch,
            self.global_step,
            **kwargs
        )
        
        return klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)

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
    
    def _plot_context_embedding_heatmap(self) -> None:
        """Plot context embedding"""
        if self.module.use_context_emb:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Get embedding weights and convert to numpy array for plotting 
            weights = self.module.context_emb.weight.clone().detach().cpu().numpy()
            # Get context labels
            batch_labels = self.batch_labels
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(weights, cmap='viridis', ax=ax, annot=True, fmt=".3f")
            ax.set_title(f'Context Embedding Weights @ Epoch {self.current_epoch}')
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Context Index')

            # Add context labels if available
            if batch_labels is not None:
                ax.set_yticks(np.arange(len(batch_labels)) + 0.5)
                ax.set_yticklabels(batch_labels, rotation=0)
            # Add tight layout to fit everything into the plot
            plt.tight_layout()

            # Log to tensorboard
            self.logger.experiment.add_figure('context_embedding_heatmap', fig, self.current_epoch)
            plt.close(fig)
    
    def _plot_kl_distribution_over_latents(self, kl_tensor: torch.Tensor, metric: str) -> None:
        """Plot distribution of KL divergence over latent dimensions."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert to numpy for plotting
        kl_values = kl_tensor.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot density of KL values
        sns.kdeplot(data=kl_values.flatten(), ax=ax, fill=True)

        # Set y lim to 0-latent dims
        plt.ylim((0, kl_values.shape[0]))
        # Set x lim to 2 std 
        plt.xlim((-2, 2))
        
        # Calculate mean
        mean_kl = np.mean(kl_values)
        
        # Add vertical line for mean
        ax.axvline(mean_kl, color='red', linestyle=':', label=f'Mean: {mean_kl:.2f}')
        
        # Add text annotation for mean
        ax.text(mean_kl + 0.5, kl_values.shape[0]*0.9, f'Mean KL: {mean_kl:.2f}', 
            rotation=0, verticalalignment='top')
        
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of KL Divergence over Latents (Epoch {self.current_epoch}) N latent = {kl_values.shape[0]}')
        # Log to tensorboard
        self.logger.experiment.add_figure(f"{metric}_distribution", fig, self.current_epoch)
        plt.close(fig)

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
        # Only log kl distribtion every n epochs on validation
        kl_per_latent = loss_output.extra_metrics.pop(MODULE_KEYS.KL_Z_PER_LATENT_KEY)
        if (
            self.plot_kl_distribution
            and mode != 'train'
            and self.full_val_log_every_n_epoch is not None 
            and self.current_epoch % self.full_val_log_every_n_epoch == 0
        ):
            self._plot_kl_distribution_over_latents(kl_tensor=kl_per_latent, metric=MODULE_KEYS.KL_Z_PER_LATENT_KEY)
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

            # Exclude control class for multiclass metrics if its actually in the data
            if not self.module.use_classification_control and self.module.ctrl_class_idx is not None:
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

    def _calc_cls_sim(self, cls_emb: torch.Tensor | None) -> torch.Tensor | None:
        if cls_emb is None:
            return None
        cls_emb = F.normalize(cls_emb, dim=-1)
        return cls_emb @ cls_emb.T
    
    def _log_temperatures(self, batch_size: int | None = None):
        # Log classifier temperature
        if self.module.use_learnable_temperature:
            # Check classifier temperature
            cls_temp = getattr(self.module.classifier, 'temperature')
            if cls_temp:
                self.log(
                    "T_classifier_logit",
                    cls_temp,
                    on_epoch=True,
                    batch_size=batch_size,
                    prog_bar=False,
                )
            # Log other learnable temperatures
            self.log(
                "T_contrastive_z",
                self.module.contr_temp,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )
            self.log(
                "T_clip_z",
                self.module.clip_temp,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )
            self.log(
                "T_proxy_z",
                self.module.proxy_temp,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )

    def step(self, mode, batch, batch_idx):
        """Step for supervised training"""
        # Compute RL weight
        self.loss_kwargs.update({"rl_weight": self.rl_weight})
        # Compute KL warmup
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight_multistage})
        # Compute CL warmup
        self.loss_kwargs.update({"classification_ratio": self.classification_ratio})
        if self.use_contrastive_loader in ['both', mode]:
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
        # Choose to use class scores for scaling or not
        input_kwargs['use_cls_scores'] = self.use_cls_scores in [mode, 'both']
        # Perform full forward pass of model
        inference, generative, loss_output = self.forward(batch, loss_kwargs=input_kwargs)
        return inference, generative, loss_output
        
    def training_step(self, batch, batch_idx):
        """Training step for supervised training."""
        # Perform full forward pass of model
        inference_outputs, _, loss_output = self.step(mode='train', batch=batch, batch_idx=batch_idx)
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
        # Log learnable temperatures if given
        self._log_temperatures()
        # Log other metrics
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        y = loss_output.true_labels.squeeze(-1)
        # Count occurrences of each class in this batch
        unique, counts = torch.unique(y, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            self.class_batch_counts[u].append(c)
        # Save number of classes / batch
        self.train_class_counts.append(len(unique))
        # Save in cache
        with torch.no_grad():
            self._cache_step_data('train', batch, inference_outputs, loss_output)
        # Return final loss value
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for supervised training."""
        # Perform full forward pass of model
        inference_outputs, _, loss_output = self.step(mode='val', batch=batch, batch_idx=batch_idx)
        loss = loss_output.loss
        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.compute_and_log_metrics(loss_output, self.val_metrics, "validation")
        # Cache the data
        with torch.no_grad():
            self._cache_step_data('val', batch, inference_outputs, loss_output)

    def on_train_epoch_start(self):        
        # Freeze module encoder at a certain epoch if option is enabled
        if self.freeze_encoder_epoch is not None and self.current_epoch >= self.freeze_encoder_epoch and not self.encoder_frozen:
            self.module.freeze_module('z_encoder')
    
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
        # Plot umap
        if self.plot_umap in ['train', 'both'] and self.current_epoch % 10 == 0:
            train_data = self._process_epoch_cache('train')
            if train_data:
                self._plt_umap('train', train_data)

    # Calculate accuracy & f1 for entire validation set, not just per batch
    def on_validation_epoch_end(self):
        """Plot full validation set UMAP projection."""
        if self.full_val_log_every_n_epoch > 0 and self.current_epoch % self.full_val_log_every_n_epoch == 0:
            plt_umap = self.plot_umap in ['val', 'both']
            if plt_umap or self.log_full_val:
                # Get data from forward pass
                val_data = self._process_epoch_cache('val')
                if val_data:
                    # Plot UMAP for validation set
                    if plt_umap:
                        self._plt_umap('val', val_data)
                    # Calculate performance metrics over entire validation set, not just batch
                    if self.log_full_val and self.classification_ratio > 0:
                        self._log_full_val_metrics(val_data)
            # Plot context embedding
            if self.plot_context_emb:
                self._plot_context_embedding_heatmap()
                    
    def _log_full_val_metrics(self, mode_data: dict[str, np.ndarray]) -> None:
        true_labels = torch.tensor(mode_data.get('labels'))
        predicted_labels = torch.tensor(mode_data.get('predicted_labels')).squeeze(-1)
        n_classes = self.n_classes
        # Exclude control class for multiclass metrics if it exists in data
        if not self.module.use_classification_control and self.module.ctrl_class_idx is not None:
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

    def _plt_umap(self, mode: str, mode_data: dict[str, np.ndarray], use_history: bool = True):
        """Plot umap of latent space in tensorboard logger"""
        import umap
        # Extract data from forward pass
        embeddings = mode_data['embeddings']
        labels = mode_data['labels']
        # Look for actual label encoding
        if "_code_to_label" in self.loss_kwargs:
            labels = [self.loss_kwargs["_code_to_label"][l] for l in labels]
        labels = pd.Categorical(labels)
        covs = pd.Categorical(mode_data['covs'])
        
        # Create cache structure if it doesn't exist
        if 'umap_cache' not in self.cache:
            self.cache['umap_cache'] = {}
        
        # Check if we have a cached UMAP transformer for this mode
        transformer_key = f'{mode}_umap_transformer'
        embedding_key = f'{mode}_umap_reference'
        
        if transformer_key not in self.cache['umap_cache']:
            # First time - fit UMAP and cache the transformer and reference embedding
            self.cache['umap_cache'][transformer_key] = umap.UMAP(n_components=2)
            embeddings_2d = self.cache['umap_cache'][transformer_key].fit_transform(embeddings)
            self.cache['umap_cache'][embedding_key] = embeddings  # Store reference embedding
        else:
            # Check if embeddings have changed significantly
            ref_embeddings = self.cache['umap_cache'][embedding_key]
            if embeddings.shape == ref_embeddings.shape:
                # Calculate relative difference
                rel_diff = np.mean(np.abs(embeddings - ref_embeddings)) / (np.mean(np.abs(ref_embeddings)) + 1e-6)
                if rel_diff > 0.1:  # Refit if embeddings have changed significantly (10% threshold)
                    self.cache['umap_cache'][transformer_key] = umap.UMAP(n_components=2)
                    embeddings_2d = self.cache['umap_cache'][transformer_key].fit_transform(embeddings)
                    self.cache['umap_cache'][embedding_key] = embeddings
                else:
                    # Use cached transformer for similar embeddings
                    embeddings_2d = self.cache['umap_cache'][transformer_key].transform(embeddings)
            else:
                # Refit if embedding dimensions have changed
                self.cache['umap_cache'][transformer_key] = umap.UMAP(n_components=2)
                embeddings_2d = self.cache['umap_cache'][transformer_key].fit_transform(embeddings)
                self.cache['umap_cache'][embedding_key] = embeddings
    
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
