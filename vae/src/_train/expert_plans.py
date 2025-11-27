import pandas as pd
import numpy as np
import torch
import torchmetrics.functional as tmf
import torch.nn.functional as F

from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS, PREDICTION_KEYS
from src.utils.io import to_tensor
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
        log_class_distribution: bool = False,
        freeze_modules: list[str] | None = None,
        soft_freeze_lr: float | None = None,
        freeze_encoder_epoch: int | None = None,
        freeze_decoder_epoch: int | None = None,
        gene_emb: torch.Tensor | None = None,
        anneal_schedules: dict[str, dict[str | float]] | None = None,
        multistage_kl_frac: float = 0.1,
        multistage_kl_min: float = 1e-2,
        use_local_stage_warmup: bool = False,
        batch_labels: np.ndarray | None = None,
        # Full log options
        log_full: list[str] | None = ['val', 'test'],
        full_log_every_n_epoch: int = 10,
        # Plotting options
        plot_kl_distribution: bool = True,
        log_random_predictions: bool = True,            # Monitor zero-shot capability
        plot_umap: bool = False,
        plot_umap_key: str = 'z_shared',
        # Caching options
        cache_n_epochs: int | None = 1,
        n_train_step_cache: int = 64,
        save_cache: bool = False,
        **loss_kwargs,
    ):
        # Get custom optimizer for soft freezing if argument is provided
        optimizer, optimizer_fn = self._optimizer_creator_fn_custom(freeze_modules=freeze_modules, freeze_lr=soft_freeze_lr)
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
            'rl_weight': {'min_weight': 1.0, 'max_weight': 1.0},
            'kl_weight': {'min_weight': 0.0, 'max_weight': 1.0}
        }
        self.anneal_schedules.update(anneal_schedules)
        self.n_epochs_warmup = n_epochs_warmup
        self.n_steps_warmup = n_steps_warmup
        self.multistage_kl_frac = multistage_kl_frac
        self.multistage_kl_min = multistage_kl_min
        self.kl_reset_epochs = self._get_stall_epochs()
        self.reset_kl_at = np.array(sorted(self.kl_reset_epochs.values()))
        self.kl_kwargs = self.anneal_schedules.get('kl_weight')
        self.use_local_stage_warmup = use_local_stage_warmup
        # CLS params
        self.n_classes = n_classes
        self.use_posterior_mean = use_posterior_mean if use_posterior_mean is not None else "none"
        self.gene_emb = to_tensor(gene_emb)
   
        # Contrastive params
        self.log_class_distribution = log_class_distribution
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.freeze_decoder_epoch = freeze_decoder_epoch
        # Misc
        self.average = average                                                      # How to reduce the classification metrics
        self.top_k = top_k                                                          # How many top predictions to consider for metrics
        self.plot_umap = plot_umap
        self.plot_umap_key = plot_umap_key
        self.plot_kl_distribution = plot_kl_distribution
        self.log_full = log_full
        self.full_log_every_n_epoch = full_log_every_n_epoch
        self.batch_labels = batch_labels
        self.log_random_predictions = log_random_predictions
        # Save classes that have been seen during training
        self.train_class_counts = []
        self.class_batch_counts = defaultdict(list)
        self.observed_classes = set()
        # ----- Cache -----
        self.cache = {}
        self.n_train_step_cache = n_train_step_cache
        self.save_cache = save_cache
        # Cache for embeddings and metadata within epoch
        self.epoch_cache = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        # Cache for historical data across epochs TODO: implement properly
        self.history_cache = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        # Save n epochs in cache
        self.cache_n_epochs = cache_n_epochs

        # Setup test metrics
        self._n_obs_test = None
        self.initialize_test_metrics()


    def _optimizer_creator_fn_custom(
        self,
        optimizer_cls: torch.optim.Adam | torch.optim.AdamW = torch.optim.Adam,
        freeze_modules: list[str] | None = None,
        freeze_lr: float | None = 1e-5,
    ):
        """
        Create an optimizer for the model, applying a reduced LR to any parameter
        belonging to a module whose name matches entries in `freeze_modules`.

        Parameters
        ----------
        optimizer_cls : torch.optim.Optimizer class
            e.g., torch.optim.Adam or torch.optim.AdamW
        freeze_modules : list of str, optional
            Names (substrings) of modules to 'soft freeze' (e.g. ["encoder", "decoder"])
        freeze_lr : float
            Learning rate for frozen modules; default 1e-5
        """
        # Create a custom optimizer if modules should be soft frozen
        if freeze_modules is None or freeze_lr is None or not float(freeze_lr) > 0:
            return "Adam", None

        # Ensure it's cast to float
        freeze_lr = float(freeze_lr)
        
        logging.info(f'Creating custom optimizer. lr: {freeze_lr}')

        def _creator(params):
            # collect all parameters (params is usually a generator)
            params_list = list(params)
            if not hasattr(self, "module"):
                # fallback: no module context, behave normally
                return optimizer_cls(params_list, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)

            # map parameter -> module name for filtering
            param_to_name = {}
            for module_name, module in self.module.named_modules():
                for p in module.parameters(recurse=False):
                    param_to_name[p] = module_name

            # build param groups
            param_groups = []
            for p in params_list:
                name = param_to_name.get(p, "")
                if any(fm in name for fm in freeze_modules):
                    param_groups.append({"params": [p], "lr": freeze_lr})
                else:
                    param_groups.append({"params": [p], "lr": self.lr})

            return optimizer_cls(
                param_groups,
                lr=self.lr,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

        return "Custom", _creator


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

    def _cache_step_data(self, mode: str, batch: dict, inference_outputs: dict, loss_output: LossOutput):
        """Cache data from a single step."""
        # Only cache every n epoch
        if self.full_log_every_n_epoch > 0 and self.current_epoch % self.full_log_every_n_epoch == 0:
            # Stop caching training steps after limit is reached
            if mode == 'train' and len(self.epoch_cache[mode][MODULE_KEYS.Z_KEY]) > self.n_train_step_cache:
                return
            self.epoch_cache[mode][MODULE_KEYS.Z_KEY].append(inference_outputs[MODULE_KEYS.Z_KEY].cpu())
            self.epoch_cache[mode][REGISTRY_KEYS.LABELS_KEY].append(inference_outputs[REGISTRY_KEYS.LABELS_KEY].cpu())
            self.epoch_cache[mode][REGISTRY_KEYS.BATCH_KEY].append(inference_outputs[REGISTRY_KEYS.BATCH_KEY].cpu())
            if loss_output.logits is not None:
                self.epoch_cache[mode][PREDICTION_KEYS.PREDICTION_KEY].append(torch.argmax(loss_output.logits, dim=-1).cpu())
                self.epoch_cache[mode][MODULE_KEYS.CLS_LOGITS_KEY].append(loss_output.logits.cpu())
            # Add shared latent space to cache
            if MODULE_KEYS.Z_SHARED_KEY in loss_output.extra_metrics['extra_outputs']:
                self.epoch_cache[mode][MODULE_KEYS.Z_SHARED_KEY].append(loss_output.extra_metrics['extra_outputs'][MODULE_KEYS.Z_SHARED_KEY].cpu())
            # Add class projection to cache
            if MODULE_KEYS.CLS_PROJ_KEY in loss_output.extra_metrics['extra_outputs']:
                self.epoch_cache[mode][MODULE_KEYS.CLS_PROJ_KEY].append(loss_output.extra_metrics['extra_outputs'][MODULE_KEYS.CLS_PROJ_KEY].cpu())

    def _process_epoch_cache(self, mode: str) -> dict[str, np.ndarray]:
        """Process and clear the epoch cache, returning concatenated arrays."""
        cache = self.epoch_cache.get(mode, None)
        if not cache:
            return {}
        # Concatenate all cached tensors
        processed = {
            MODULE_KEYS.Z_KEY: torch.cat(cache[MODULE_KEYS.Z_KEY], dim=0).numpy(),
            REGISTRY_KEYS.LABELS_KEY: torch.cat(cache[REGISTRY_KEYS.LABELS_KEY], dim=0).numpy().squeeze(),
            REGISTRY_KEYS.BATCH_KEY: torch.cat(cache[REGISTRY_KEYS.BATCH_KEY], dim=0).numpy().squeeze(),
        }
        # Add cached shared latent space to output
        if MODULE_KEYS.Z_SHARED_KEY in cache:
            processed[MODULE_KEYS.Z_SHARED_KEY] = torch.cat(cache.get(MODULE_KEYS.Z_SHARED_KEY), dim=0).numpy()
        # Add mean cached class projections to outuput
        if MODULE_KEYS.CLS_PROJ_KEY in cache:
            processed[MODULE_KEYS.CLS_PROJ_KEY] = torch.stack(cache.get(MODULE_KEYS.CLS_PROJ_KEY), dim=0).mean(dim=0).numpy()

        if cache.get(PREDICTION_KEYS.PREDICTION_KEY, []):
            processed[PREDICTION_KEYS.PREDICTION_KEY] = torch.cat(cache[PREDICTION_KEYS.PREDICTION_KEY], dim=0).numpy().squeeze()
            processed[MODULE_KEYS.CLS_LOGITS_KEY] = torch.cat(cache[MODULE_KEYS.CLS_LOGITS_KEY], dim=0).numpy().squeeze()
        
        # Add to historical cache
        for k, v in processed.items():
            self.history_cache[mode][k].append(v)
            # Keep only last N epochs
            if len(self.history_cache[mode][k]) > self.cache_n_epochs:
                self.history_cache[mode][k].pop(0)
        # Add mode to output data
        processed[REGISTRY_KEYS.SPLIT_KEY] = np.repeat(mode, processed[REGISTRY_KEYS.LABELS_KEY].shape[0])
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

    def _compute_weight(
        self,
        n_epochs_warmup: int | None,
        n_steps_warmup: int | None,
        n_epochs_stall: int | None = None,
        max_weight: float | None = 0.0,
        min_weight: float | None = 0.0,
        post_min_weight: float | None = None,
        schedule: Literal["linear", "sigmoid", "exponential"] = "sigmoid",
        anneal_k: float = 10.0,
        hard_stall: bool = True,
        invert: bool = False,
        **kwargs
    ) -> float:
        # Pull current epoch
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
        """KL weight for forward process. Resets every time a new loss objective activates."""
        # Get extra kl args from 
        kwargs = self.kl_kwargs
        # Determine reset params
        reset_at = self.reset_kl_at
        n_epochs_stall = 0
        n_epochs_warmup = self.n_epochs_warmup
        # Handle stall schedules
        if reset_at.shape[0] > 1 and self.use_local_stage_warmup:
            # Check in which state we currently are
            idx = (self.current_epoch > reset_at).sum() - 1
            # Get next stage starting epoch
            n_epochs_stall = reset_at[idx]
            
            # KL goes from min to max within each stage
            n_epochs_warmup = reset_at[idx+1] if idx+1 < len(reset_at) else self.n_epochs_warmup
            min_weight = kwargs['min_weight'] if n_epochs_stall < 1 else kwargs['max_weight'] * self.multistage_kl_frac
            n_epochs_warmup = n_epochs_warmup - n_epochs_stall                
            # Update min weight
            kwargs['min_weight'] = min_weight

        # Update warmup and stall epochs
        kwargs['n_epochs_warmup'] = n_epochs_warmup
        kwargs['n_steps_warmup'] = None
        kwargs['n_epochs_stall'] = n_epochs_stall
        
        # Calculate final KL weight
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
            klw if type(self).__name__ == "JaxTrainingPlan" else torch.tensor(klw).to(self.device)
        )

    @property
    def weights(self) -> dict[str, torch.Tensor]:
        """Returns a dictionary of all current weights."""
        return {k: self._get_schedule_weight(k) for k in self.anneal_schedules}
    
    def _plot_kl_distribution_over_latents(self, kl_tensor: torch.Tensor, metric: str) -> None:
        """Plot distribution of KL divergence over latent dimensions."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert to numpy for plotting
        kl_values = kl_tensor.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot density of KL values
        sns.kdeplot(data=kl_values.flatten(), ax=ax, fill=True, warn_singular=False)

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
            and self.full_log_every_n_epoch is not None 
            and self.current_epoch % self.full_log_every_n_epoch == 0
        ):
            self._plot_kl_distribution_over_latents(kl_tensor=kl_per_latent, metric=MODULE_KEYS.KL_Z_PER_LATENT_KEY)
        # Initialize default classification metrics
        metrics_dict = {
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY: np.inf,
            METRIC_KEYS.ACCURACY_KEY: 0.0,
            METRIC_KEYS.F1_SCORE_KEY: 0.0,
            f'clip_{METRIC_KEYS.ACCURACY_KEY}': 0.0,
            f'clip_{METRIC_KEYS.F1_SCORE_KEY}': 0.0,
        }
        # Get number of classes
        n_classes = self.n_classes
        # Get true class labels
        true_labels = loss_output.true_labels.squeeze(-1)
        # Add classification-based metrics if they exist
        if loss_output.classification_loss is not None:
            classification_loss = loss_output.classification_loss
            logits = loss_output.logits
            # Take argmax of logits to get highest predicted label
            predicted_labels = torch.argmax(logits, dim=-1).squeeze(-1)

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
        # Add clip classification metrics
        clip_cls_pred = loss_output.extra_metrics['extra_outputs'].pop(PREDICTION_KEYS.PREDICTION_KEY, None)
        if clip_cls_pred is not None:
            # Check if any classes are outside of training classes, i.e
            unknown_mask = (clip_cls_pred >= n_classes)
            if unknown_mask.any():
                # Assign unknown to classes that are outside of the training classes
                clip_cls_pred = clip_cls_pred.masked_fill(unknown_mask, n_classes)
                n_classes += 1
            # Assign unknown to classes that are outside of the training classes
            clip_cls_pred = clip_cls_pred.masked_fill((clip_cls_pred >= n_classes), n_classes)
            # Calculate accuracy over predicted and true labels
            accuracy = tmf.classification.multiclass_accuracy(
                clip_cls_pred,
                true_labels,
                n_classes,
                average=self.average
            )
            # Calculate F1 score over predicted and true labels
            f1 = tmf.classification.multiclass_f1_score(
                clip_cls_pred,
                true_labels,
                n_classes,
                average=self.average,
            )
            # Update metrics
            metrics_dict[f'clip_{METRIC_KEYS.ACCURACY_KEY}'] = accuracy
            metrics_dict[f'clip_{METRIC_KEYS.F1_SCORE_KEY}'] = f1
        # Add metrics to extra metrics for internal logging
        loss_output.extra_metrics.update(metrics_dict)
        # Remove all extra outputs from metrics for super logging
        loss_output.extra_metrics.pop('extra_outputs')
        # Compute and log all metrics
        super().compute_and_log_metrics(loss_output, metrics, mode)

    def _calc_cls_sim(self, cls_emb: torch.Tensor | None) -> torch.Tensor | None:
        if cls_emb is None:
            return None
        cls_emb = F.normalize(cls_emb, dim=-1)
        return cls_emb @ cls_emb.T
    
    def _log_temperatures(self, batch_size: int | None = None):
        # Log model temperatures
        ctx_temp = getattr(self.module.aligner, 'ctx_temperature', None)
        cls_temp = getattr(self.module.aligner, 'cls_temperature', None)
        joint_temperature = getattr(self.module.aligner, 'joint_temperature', None)
        sigma = getattr(self.module.aligner, 'noise_sigma', None)

        if ctx_temp:
            self.log(
                "T_ctx_logit",
                ctx_temp,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )
        if cls_temp:
            self.log(
                "T_cls_logit",
                cls_temp,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )
        if joint_temperature:
            self.log(
                "T_joint_logit",
                joint_temperature,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )
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
        self.loss_kwargs.update(self.weights)
        # Setup kwargs
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)
        
        # Add split-specific kwargs
        input_kwargs['use_posterior_mean'] = self.use_posterior_mean in [split, 'both']
        
        # TODO: move to model, Add external gene embedding to batch
        if self.gene_emb is not None:
            # Add gene embedding matrix to batch
            batch[REGISTRY_KEYS.GENE_EMB_KEY] = self.gene_emb
        # Perform full forward pass of model
        return self.forward(batch, loss_kwargs=input_kwargs)
        
    def training_step(self, batch, batch_idx):
        """Training step for supervised training."""
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

    def validation_step(self, batch, batch_idx):
        """Validation step for supervised training."""
        # Perform full forward pass of model
        inference_outputs, _, loss_output = self.step(split='val', batch=batch, batch_idx=batch_idx)
        loss = loss_output.loss
        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        # Cache the data
        with torch.no_grad():
            self._cache_step_data('val', batch, inference_outputs, loss_output)
        # Log metrics
        self.compute_and_log_metrics(loss_output, self.val_metrics, "validation")

    def test_step(self, batch, batch_idx):
        """Test step for supervised training."""
        # Perform full forward pass of model
        inference_outputs, _, loss_output = self.step(split='test', batch=batch, batch_idx=batch_idx)
        loss = loss_output.loss
        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        # Cache the data
        with torch.no_grad():
            self._cache_step_data('test', batch, inference_outputs, loss_output)
        # Log metrics
        self.compute_and_log_metrics(loss_output, self.test_metrics, "test")

    def on_train_epoch_start(self):        
        # Freeze module encoder at a certain epoch if option is enabled
        if self.freeze_encoder_epoch is not None and self.current_epoch >= self.freeze_encoder_epoch and not getattr(self.module.z_encoder, 'frozen', False):
            self.module.freeze_module('z_encoder')
        if self.freeze_decoder_epoch is not None and self.current_epoch >= self.freeze_decoder_epoch and not getattr(self.module.decoder, 'frozen', False):
            self.module.freeze_module('decoder')

    def _plt_random_ctx_predictions(self, random_predictions: torch.Tensor):
        if random_predictions is None:
            return
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Detach and move to cpu
        random_predictions = random_predictions.detach().cpu()

        # Get number of observed and total labels from module
        n = self.module.n_ctx
        n_obs = self.module.n_batch
        # Plot
        fig = plt.figure(dpi=120, figsize=(8,6))
        ax = sns.histplot(random_predictions)
        plt.axvline(n_obs, linestyle='--', color='red')
        plt.xlim((0,n))
        ax.set_xlabel("")
        ax.set_ylabel("Number of classes in Batch")
        ax.set_title("Distribution of Number of Classes per Batch")
        self.logger.experiment.add_figure("pseudo_argmax_ctx", fig, self.current_epoch)
        plt.close(fig)

    def _plt_random_cls_predictions(self, random_predictions: torch.Tensor):
        if random_predictions is None:
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Detach and move to cpu
        random_predictions = random_predictions.detach().cpu()

        # Get number of classes etc
        n = self.module.n_cls
        n_obs = self.module.n_labels
        # Plot
        fig = plt.figure(dpi=120, figsize=(8,6))
        ax = sns.histplot(random_predictions)
        plt.axvline(n_obs, linestyle='--', color='red')
        plt.xlim((0,n))
        ax.set_xlabel("")
        ax.set_ylabel("Number of classes in Batch")
        ax.set_title("Distribution of Number of Classes per Batch")
        self.logger.experiment.add_figure("pseudo_argmax_cls", fig, self.current_epoch)
        plt.close(fig)

    def on_train_epoch_end(self):
        # Get random prediction buffer from module and reset buffer
        random_ctx_predictions = self.module.get_ctx_buffer(reset=True)
        random_cls_predictions = self.module.get_cls_buffer(reset=True)
        # Plot random prediction distributions
        if self.log_random_predictions:
            # Plot the distributions
            self._plt_random_ctx_predictions(random_ctx_predictions)
            self._plt_random_cls_predictions(random_cls_predictions)
        # Plot class distribution in batches
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

    def _full_log_on_epoch_end(self):
        """Plot umap of mode on epoch end"""
        # Get modes to do a full log on
        modes = self.log_full
        if isinstance(modes, str):
            modes = [modes]
        # Do a full log for given modes
        if len(self.log_full) == 0:
            return
        # Get cached steps from all modes
        data = {}
        _modes = []
        for mode in modes:
            # Get cached mode data
            mode_data = self._process_epoch_cache(mode)
            if mode_data is None or len(mode_data.keys()) == 0:
                continue
            # Calculate performance metrics over entire set
            self._log_full_metrics(mode, mode_data)
            # Save used mode
            _modes.extend([mode])
            # Combine modes only if we plot
            if len(data) == 0 or not self.plot_umap:
                data = mode_data
                continue
            # Concatenate existing data
            for k, v in data.items():
                new_v = mode_data.get(k)
                if new_v is not None:
                    data[k] = np.concatenate((v, new_v), axis=0)
            
        # If data is not empty and we want to plot
        if len(_modes) > 0 and self.plot_umap:
            # Plot UMAP for all splits TODO: fix full name label
            full_name = '-'.join(_modes) if len(_modes) > 1 else _modes[0]
            self._plt_umap(full_name, data)

    # Calculate accuracy & f1 for entire validation set, not just per batch
    def on_validation_epoch_end(self):
        """Calculate metrics and plot charts for full data split(s)."""
        if self.full_log_every_n_epoch > 0 and self.current_epoch % self.full_log_every_n_epoch == 0:
            self._full_log_on_epoch_end()
                    
    def _log_full_metrics(self, mode: str, mode_data: dict[str, np.ndarray]) -> None:
        # Skip logging if no classification is available yet
        if PREDICTION_KEYS.PREDICTION_KEY not in mode_data:
            return
        true_labels = torch.tensor(mode_data.get(REGISTRY_KEYS.LABELS_KEY))
        # Remove control labels from true labels for metrics
        if self.module.ctrl_class_idx is not None:
            no_ctrl_mask = (true_labels != self.module.ctrl_class_idx)
            true_labels = true_labels[no_ctrl_mask]
        predicted_labels = torch.tensor(mode_data.get(PREDICTION_KEYS.PREDICTION_KEY)).squeeze(-1)
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
        f1 = tmf.classification.multiclass_f1_score(
            predicted_labels,
            true_labels,
            n_classes,
            average=self.average,
        )
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

    def _plt_umap(self, mode: str, mode_data: dict[str, np.ndarray]):
        """Plot UMAP of latent space (with proxies) into TensorBoard."""
        import os
        import umap
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Get shared latent embedding and annotation data, fall back to z if not in cached data
        emb_key = self.plot_umap_key if self.plot_umap_key in mode_data else MODULE_KEYS.Z_KEY
        embeddings = mode_data[emb_key]
        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        covs = mode_data[REGISTRY_KEYS.BATCH_KEY]
        modes = pd.Categorical(mode_data[REGISTRY_KEYS.SPLIT_KEY])

        # --- Subset: remove control cells
        if self.module.ctrl_class_idx is not None:
            mask = labels != self.module.ctrl_class_idx
            embeddings, labels, covs, modes = embeddings[mask], labels[mask], covs[mask], modes[mask]
        # Add batch annotation
        if self.batch_labels is not None:
            covs = self.batch_labels[covs]
        # --- Add class proxies if they exist (no alignment yet)
        if MODULE_KEYS.CLS_PROJ_KEY in mode_data:
            cls_proxies = mode_data[MODULE_KEYS.CLS_PROJ_KEY]
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

        # --- cached UMAP transform
        cache = self.cache.setdefault("umap_cache", {})
        transformer_key = f"{mode}_umap_transformer"
        ref_key = f"{mode}_umap_reference"

        if transformer_key not in cache:
            reducer = umap.UMAP(n_components=2)
            emb_2d = reducer.fit_transform(embeddings_all)
            cache[transformer_key] = reducer
            cache[ref_key] = embeddings_all
        else:
            ref = cache[ref_key]
            if embeddings_all.shape == ref.shape:
                rel_diff = np.mean(np.abs(embeddings_all - ref)) / (np.mean(np.abs(ref)) + 1e-6)
                if rel_diff > 0.1:
                    reducer = umap.UMAP(n_components=2)
                    emb_2d = reducer.fit_transform(embeddings_all)
                    cache[transformer_key] = reducer
                    cache[ref_key] = embeddings_all
                else:
                    reducer = cache[transformer_key]
                    emb_2d = reducer.transform(embeddings_all)
            else:
                reducer = umap.UMAP(n_components=2)
                emb_2d = reducer.fit_transform(embeddings_all)
                cache[transformer_key] = reducer
                cache[ref_key] = embeddings_all

        # Add umap coordinates to dataframe
        df["UMAP1"], df["UMAP2"] = emb_2d[:, 0], emb_2d[:, 1]

        # --- plotting helper
        def _scatter_base(ax, data, hue, title, legend=True, plot_proxy=True, proxy_color="black", proxy_legend=True):
            sns.scatterplot(
                data=data[~data.is_proxy],
                x="UMAP1", y="UMAP2", hue=hue,
                s=6, alpha=0.6, ax=ax, legend=legend
            )
            # overlay proxies as crosses
            if plot_proxy:
                sns.scatterplot(
                    data=data[data.is_proxy],
                    x="UMAP1", y="UMAP2", hue=hue,
                    s=40, marker="X", ax=ax, legend=proxy_legend,
                    edgecolor=proxy_color, linewidth=0.5
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
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)

        # 1. by modes (splits)
        _scatter_base(axes[0, 0], df, "mode", f"Splits @ Epoch {self.current_epoch}")

        # 2. by class (overlay proxies)
        _scatter_base(axes[0, 1], df, "label", f"Classes @ Epoch {self.current_epoch}", legend=False, plot_proxy=False, proxy_legend=False)

        # 3. by observed vs unseen
        df_obs = df.copy()
        df_obs["obs_label"] = np.where(df_obs.is_observed, "Observed proxy", "Unseen proxy")
        _scatter_base(axes[1, 0], df_obs, "obs_label", f"Observed vs Unseen @ Epoch {self.current_epoch}")

        # 4. by covariates (contexts)
        _scatter_base(axes[1, 1], df, "cov", f"Contexts @ Epoch {self.current_epoch}", plot_proxy=False, proxy_legend=False)

        plt.tight_layout()
        self.logger.experiment.add_figure(f"{mode}_{emb_key}_umap", fig, self.current_epoch)
        plt.close(fig)

        # Save data to file
        if self.save_cache:
            umap_dir = os.path.join(self.logger.log_dir, 'data')
            os.makedirs(umap_dir, exist_ok=True)
            csv_path = os.path.join(umap_dir, f"{mode}_umap_data_epoch{self.current_epoch}.csv")
            df.to_csv(csv_path, index=False)

