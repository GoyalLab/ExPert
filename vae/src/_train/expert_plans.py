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
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation", "validation_classification_loss"
        ] = "elbo_validation",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        average: str = "macro",
        top_k: int = 1,
        auto_temperature_args: dict | None = None,
        use_posterior_mean: Literal["train", "val", "both"] | None = "val",
        log_class_distribution: bool = True,
        freeze_modules: dict[str, float] | None = None,
        freeze_encoder_epoch: int | None = None,
        freeze_global_encoder_epoch: int | None = None,
        freeze_decoder_epoch: int | None = None,
        soft_freeze_lr: float | None = None,
        gene_emb: torch.Tensor | None = None,
        anneal_schedules: dict[str, dict[str | float]] | None = None,
        kl_stage_offset: int = 0,
        batch_labels: np.ndarray | None = None,
        cov_labels: np.ndarray | None = None,
        # Full log options
        log_full: list[str] | None = ['val'],
        full_log_every_n_epoch: int = 1,
        # Plotting options
        plot_kl_distribution: bool = False,
        log_random_predictions: bool = False,            # Monitor zero-shot capability
        plot_umap: bool = False,
        plot_umap_key: str | list[str] = 'z_shared',
        plot_f1_dist: bool = True,
        plot_cm: bool = True,
        plot_clip_geom: bool = True,
        plot_cat_emb: bool = True,
        plot_every_n_epochs: int = 10,
        # Caching options
        cache_n_epochs: int | None = 1,
        n_train_cache: int = 10_000,
        save_cache: bool = False,
        use_cosine_restart: bool = False,
        **loss_kwargs,
    ):
        # Set custom learning rates for zlocal encoder if auto temperature is enabled
        if auto_temperature_args is not None:
            # Set default learning rate of local encoder to overall learning rate
            auto_conf = {
                'local_encoder': lr, 'local_mu': lr, 'local_var': lr
            }
            if freeze_modules is None:
                freeze_modules = auto_conf
            else:
                freeze_modules.update(auto_conf)
        self.optimizer_cls = optimizer_cls
        self.use_cosine_restart = use_cosine_restart
        # Get custom optimizer for soft freezing if argument is provided
        optimizer, optimizer_fn = self._optimizer_creator_fn_custom(module_lrs=freeze_modules)
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
        self.kl_stage_offset = kl_stage_offset
        self.kl_reset_epochs = self._get_stall_epochs()
        self.reset_kl_at = np.array(sorted(self.kl_reset_epochs.values()))
        self.kl_kwargs = self.anneal_schedules.get('kl_weight')
        # CLS params
        self.n_classes = n_classes
        self.use_posterior_mean = use_posterior_mean if use_posterior_mean is not None else "none"
        self.gene_emb = to_tensor(gene_emb)
   
        # Contrastive params
        self.log_class_distribution = log_class_distribution
        self.freeze_encoder_epoch = freeze_encoder_epoch
        self.freeze_global_encoder_epoch = freeze_global_encoder_epoch
        self.freeze_decoder_epoch = freeze_decoder_epoch
        self.soft_freeze_lr = soft_freeze_lr
        if auto_temperature_args is not None:
            from src._train.controller import ClipTemperatureController
            self.auto_temp_controller = ClipTemperatureController(**auto_temperature_args)
            # Add empty schedule key to be filled by controller
            self.anneal_schedules['T_align'] = None
        else:
            self.auto_temp_controller = None
            
        # Misc
        self.average = average                                                      # How to reduce the classification metrics
        self.top_k = top_k                                                          # How many top predictions to consider for metrics
        # Plotting options
        self.plot_umap = plot_umap
        self.plot_umap_key = plot_umap_key if isinstance(plot_umap_key, list) else [plot_umap_key]
        self.plot_f1_dist = plot_f1_dist
        self.plot_cm = plot_cm
        self.plot_clip_geom = plot_clip_geom
        self.plot_kl_distribution = plot_kl_distribution
        self.plot_cat_emb = plot_cat_emb
        self.log_full = log_full
        self.full_log_every_n_epoch = full_log_every_n_epoch
        self.plot_every_n_epochs = plot_every_n_epochs
        self.batch_labels = batch_labels
        self.cov_labels = cov_labels
        self.log_random_predictions = log_random_predictions
        # Save classes that have been seen during training
        self.train_class_counts = []
        self.class_batch_counts = defaultdict(list)
        self.observed_classes = set()
        # ----- Cache -----
        self.cache = {}
        self.n_train_cache = n_train_cache
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
        
    def _cos_lr_scheduler(self, opt: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=getattr(self, "cosine_T0", 15),
            T_mult=getattr(self, "cosine_Tmult", 2),
            eta_min=getattr(self, "cosine_eta_min", 5e-6),
        )

    def _optimizer_creator_fn_custom(
        self,
        module_lrs: dict[str, float] | None = None,
    ):

        logging.info(f'Creating custom optimizer with module LRs: {module_lrs}')

        def _creator(params):

            params_list = list(params)

            if not hasattr(self, "module"):
                return self.optimizer_cls(
                    params_list,
                    lr=self.lr,
                    eps=self.eps,
                    weight_decay=self.weight_decay,
                )

            param_to_name = {}
            for module_name, module in self.module.named_modules():
                for p in module.parameters(recurse=False):
                    param_to_name[p] = module_name

            lr_to_params = {}

            for p in params_list:

                name = param_to_name.get(p, "")

                matched_lr = None
                if module_lrs:
                    for module_key, lr in module_lrs.items():
                        if module_key in name:
                            matched_lr = float(lr)
                            break

                if matched_lr is None:
                    matched_lr = self.lr

                lr_to_params.setdefault(matched_lr, []).append(p)

            param_groups = []
            for lr, params_group in lr_to_params.items():
                param_groups.append({"params": params_group, "lr": lr})

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
                verbose=True,
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
    
    def _scale_lr(self, group_name: str, factor: float, error: bool = True):
        optimizer = self.optimizers()

        for pg in optimizer.param_groups:
            if pg.get("name", None) == group_name:
                old_lr = pg["lr"]
                pg["lr"] = old_lr * factor
                self.log(
                    f"lr/{group_name}",
                    pg["lr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
                return old_lr, pg["lr"]
        if error:
            raise ValueError(f"Optimizer param group '{group_name}' not found.")

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
        if self.full_log_every_n_epoch <= 0 or self.current_epoch % self.full_log_every_n_epoch != 0:
            return
        # Stop caching training steps after limit is reached
        if mode == "train" and len(self.epoch_cache[mode][MODULE_KEYS.Z_KEY]) > self.n_train_cache:
            return
        # Create cache
        cache = self.epoch_cache[mode]
        # Core fields
        cache[MODULE_KEYS.Z_KEY].append(inference_outputs[MODULE_KEYS.Z_KEY].detach().cpu())
        cache[REGISTRY_KEYS.LABELS_KEY].append(inference_outputs[MODULE_KEYS.LABEL_KEY].detach().cpu())
        cache[REGISTRY_KEYS.BATCH_KEY].append(inference_outputs[MODULE_KEYS.BATCH_INDEX_KEY].detach().cpu())
        # Logits
        if loss_output.logits is not None:
            cache[PREDICTION_KEYS.PREDICTION_KEY].append(
                torch.argmax(loss_output.logits, dim=-1).detach().cpu()
            )
        # Handle extra outputs
        extra_outputs = loss_output.extra_metrics.get("extra_outputs", {})

        # Pop logits and store as sparse (B, n_labels) with only in-batch classes filled
        logits = extra_outputs.pop(MODULE_KEYS.CLS_LOGITS_KEY, None)
        if logits is not None:
            labels = inference_outputs[MODULE_KEYS.LABEL_KEY].detach().flatten()
            n_labels = self.module.n_labels
            full_logits = torch.zeros(logits.size(0), n_labels, device=logits.device)
            # Map in-batch unique classes to columns
            unique_cls = labels.unique()
            # Logits may span all n_labels (full) or only unique classes (subset)
            if logits.size(1) == n_labels:
                full_logits = logits
            else:
                full_logits[:, unique_cls] = logits[:, :unique_cls.size(0)]
            cache[MODULE_KEYS.CLS_LOGITS_KEY].append(full_logits.detach().cpu())

        # Automatically cache everything tensor-like
        for k, v in extra_outputs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                cache[k].append(v.detach().cpu())
                
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
            "t_s",
        }
        # Collect each cache item
        for k, tensors in cache.items():
            if len(tensors) == 0:
                continue
            # Mean aggregation
            if k in mean_keys:
                processed[k] = torch.stack(tensors, dim=0).mean(dim=0).numpy()
            # Concatenation
            else:
                processed[k] = torch.cat(tensors, dim=0).numpy()

            # squeeze labels/batch/preds
            if k in {
                REGISTRY_KEYS.LABELS_KEY,
                REGISTRY_KEYS.BATCH_KEY,
                PREDICTION_KEYS.PREDICTION_KEY,
            }:
                processed[k] = processed[k].squeeze()

        # Add to historical cache
        for k, v in processed.items():
            self.history_cache[mode][k].append(v)
            if len(self.history_cache[mode][k]) > self.cache_n_epochs:
                self.history_cache[mode][k].pop(0)
        # Add split info
        processed[REGISTRY_KEYS.SPLIT_KEY] = np.repeat(
            mode, processed[REGISTRY_KEYS.LABELS_KEY].shape[0]
        )
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
        kl_epoch_offset = 0
        if reset_at.shape[0] > 1 and self.kl_stage_offset > 0:
            n_stages_passed = max((self.current_epoch > reset_at).sum() - 1, 0)
            kl_epoch_offset = n_stages_passed * self.kl_stage_offset
        
        # Shift effective epoch back
        kwargs['epoch'] = max(self.current_epoch - kl_epoch_offset, 10)
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
        # Get alignment temperature from controller if active
        if key == 'T_align' and self.auto_temp_controller is not None:
            return self._get_alignment_temp()
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
        
    def _update_alignment_temp(self, val_f1: float):
        # Only change temperature based on validation
        if self.training or self.auto_temp_controller is None:
            return
        # Update T and get it from controller
        updated = self.auto_temp_controller.update(
            current_epoch=self.current_epoch,
            val_f1=val_f1,
        )
        # Update learning rate
        if updated:
            self._scale_lr('local_encoder', factor=0.5)
            self._scale_lr('local_mu', factor=0.5)
            self._scale_lr('local_var', factor=0.5)
    
    def _get_alignment_temp(self):
        if self.auto_temp_controller is None:
            return None
        return self.auto_temp_controller.T

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
        
    def _log_extra_output_histograms(self, mode: str, extra_outputs: dict):
        """
        Log TensorBoard histograms for all non-scalar values in extra_outputs.

        Values are removed from the dictionary using pop() to avoid downstream issues.
        """
        import numpy as np
        import torch
        # Skip if no exra outputs are given
        if extra_outputs is None:
            return
        # Get all extra keys
        keys = list(extra_outputs.keys())  # avoid dict size change during iteration
        # Log each key
        for k in keys:
            v = extra_outputs.pop(k, None)
            if v is None:
                continue
            # Convert tensors to numpy
            if isinstance(v, torch.Tensor):
                if v.ndim > 1:
                    continue
                v = v.detach().cpu().numpy()
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
        # Only log kl distribtion every n epochs on validation
        kl_per_latent = loss_output.extra_metrics.pop(MODULE_KEYS.KL_Z_PER_LATENT_KEY)
        if (
            self.plot_kl_distribution
            and mode != 'train'
            and self.full_log_every_n_epoch is not None 
            and self.current_epoch % self.full_log_every_n_epoch == 0
        ):
            self._plot_kl_distribution_over_latents(kl_tensor=kl_per_latent, metric=MODULE_KEYS.KL_Z_PER_LATENT_KEY)
        
        # Get clip weights if they exist
        responder_prob = loss_output.extra_metrics['extra_outputs'].get('responder_prob', None)
        
        # Initialize default classification metrics
        metrics_dict = {
            METRIC_KEYS.CLASSIFICATION_LOSS_KEY: np.inf,
            METRIC_KEYS.ACCURACY_KEY: 0.0,
            METRIC_KEYS.F1_SCORE_KEY: 0.0,
            f'clip_{METRIC_KEYS.ACCURACY_KEY}': 0.0,
            f'clip_{METRIC_KEYS.F1_SCORE_KEY}': 0.0,
        }
        # Log context prediction if available
        ctx_pred = loss_output.extra_metrics['extra_outputs'].get('ctx_pred', None)
        ctx_labels = loss_output.extra_metrics['extra_outputs'].get('ctx_labels', None)
        if ctx_pred is not None:
            # Calculate F1 score over predicted and true labels
            ctx_f1 = tmf.classification.multiclass_f1_score(
                ctx_pred,
                ctx_labels,
                self.module.n_batch,
                average=self.average,
            )
            metrics_dict[f'ctx_{METRIC_KEYS.F1_SCORE_KEY}'] = ctx_f1
        # Log adversial context prediction if available
        adv_ctx_pred = loss_output.extra_metrics['extra_outputs'].get('adv_ctx_pred', None)
        if adv_ctx_pred is not None:
            adv_ctx_labels = loss_output.extra_metrics['extra_outputs'].get('adv_ctx_labels')
            # Calculate F1 score over predicted and true labels
            adv_ctx_f1 = tmf.classification.multiclass_f1_score(
                adv_ctx_pred,
                adv_ctx_labels,
                self.module.n_batch,
                average=self.average,
            )
            metrics_dict[f'adv_ctx_{METRIC_KEYS.F1_SCORE_KEY}'] = adv_ctx_f1

        # Get true class labels
        true_labels = loss_output.true_labels.squeeze(-1)
        # Add clip classification metrics
        clip_cls_pred = loss_output.extra_metrics['extra_outputs'].get(PREDICTION_KEYS.PREDICTION_KEY, None)
        if clip_cls_pred is not None:
            # Get in-batch labels
            unique = list(torch.unique(true_labels))
            n_cls = len(unique)
            n_total_cls = self.n_classes
            base_clip_cls_pred = clip_cls_pred
            # Check if any classes are outside of training classes, i.e null classes or zero-shot candidates
            unknown_mask = (1-(clip_cls_pred == torch.unique(true_labels).reshape(-1, 1)).sum(0)).bool()
            # Check null predictions
            null_mask = (clip_cls_pred >= self.n_classes)
            if unknown_mask.any():
                # Assign anything outside of training to placeholder and add to list of predictions
                clip_cls_pred = clip_cls_pred.masked_fill(unknown_mask, n_cls)
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
                [mapping[int(v)] for v in clip_cls_pred],
                device=clip_cls_pred.device
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
                base_clip_cls_pred,
                true_labels,
                num_classes=n_total_cls,
                average=self.average,
            )

            # Log fraction of predicted responders
            null_frac = responder_prob.mean() if responder_prob is not None else 0.0
            metrics_dict[f'responder_fraction'] = null_frac
            # Log clip f1 within not-null predicted cells (if there are enough responders in batch)
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
                metrics_dict[f'clip_{METRIC_KEYS.F1_SCORE_KEY}_responder'] = f1_responder
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
                metrics_dict[f'clip_{METRIC_KEYS.F1_SCORE_KEY}_non_responder'] = f1_non_responder
                
            # Update metrics
            metrics_dict[f'clip_{METRIC_KEYS.ACCURACY_KEY}_local'] = accuracy_local
            metrics_dict[f'clip_{METRIC_KEYS.F1_SCORE_KEY}_local'] = f1_local
            metrics_dict[f'clip_{METRIC_KEYS.F1_SCORE_KEY}'] = f1
        
        # Plot extra histograms
        self._log_extra_output_histograms(mode, loss_output.extra_metrics['extra_outputs'])
        # Log batch composition histograms
        self._log_extra_output_histograms(mode, {MODULE_KEYS.BATCH_INDEX_KEY: ctx_labels, MODULE_KEYS.LABEL_KEY: true_labels})
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
        # Log hierarchical temperatures
        if self.module.use_learnable_hierarchy_temperatures:
            t_gene = 1.0 / self.module.log_t_gene.exp().clamp(min=1e-6)
            self.log('t_gene', t_gene, on_epoch=True, batch_size=batch_size, prog_bar=False)
            t_module = 1.0 / self.module.log_t_module.exp().clamp(min=1e-6)
            self.log('t_module', t_module, on_epoch=True, batch_size=batch_size, prog_bar=False)
            t_pathway = 1.0 / self.module.log_t_pathway.exp().clamp(min=1e-6)
            self.log('t_pathway', t_pathway, on_epoch=True, batch_size=batch_size, prog_bar=False)

    def step(self, split, batch, batch_idx):
        """Step for supervised training"""
        # Update loss kwargs with schedules weights
        self.loss_kwargs.update(self.weights)
        # Setup kwargs
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)
        
        # Add split-specific kwargs
        input_kwargs['use_posterior_mean'] = self.use_posterior_mean in [split, 'both']

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
        # Monitor batch composition
        self._epoch_labels = []
        self._epoch_datasets = []     
        # Re-sample epoch dataset if toggled
        if hasattr(self.trainer.train_dataloader, 'set_epoch'):
            self.trainer.train_dataloader.set_epoch(self.current_epoch)
        # Freeze module encoder at a certain epoch if option is enabled
        if self.freeze_encoder_epoch is not None and self.current_epoch >= self.freeze_encoder_epoch and not getattr(self.module.z_encoder, 'frozen', False):
            self.module.freeze_module('z_encoder', soft_lr=self.soft_freeze_lr, optim=self.optimizers())
        if self.freeze_global_encoder_epoch is not None and self.current_epoch >= self.freeze_global_encoder_epoch and not getattr(self.module.z_encoder, 'frozen', False):
            self.module.freeze_module('z_encoder.global_encoder', soft_lr=self.soft_freeze_lr, optim=self.optimizers())
        if self.freeze_decoder_epoch is not None and self.current_epoch >= self.freeze_decoder_epoch and not getattr(self.module.decoder, 'frozen', False):
            self.module.freeze_module('decoder', soft_lr=self.soft_freeze_lr, optim=self.optimizers())
        # Compute class embeddings once per epoch
        if self.module.cls_emb.encoder is not None and self._get_schedule_weight('cls_align_weight') > 0:
            self.module.reset_cached_cls_emb()

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

    def _plt_cat_emb(self):
        """Plot categorical embedding weights in decoder."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Get embedding weights and plot them
        cat_bias_emb = self.module.decoder.cat_bias.weight.detach().cpu().numpy()
        colnames = 'd' + pd.Series(np.arange(cat_bias_emb.shape[-1]), dtype=str)
        cat_bias_emb = pd.DataFrame(cat_bias_emb, index=self.cov_labels, columns=colnames)
        # Plot embeddings
        g = sns.clustermap(
            cat_bias_emb, 
            row_cluster=True,
            col_cluster=True, 
            figsize=(8,6),
        )
        self.logger.experiment.add_figure("cat_cov_emb", g.figure, self.current_epoch)
        plt.close(g.figure)

    def on_train_epoch_end(self):
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
        # Plot heatmap of categorical covariate embedding in decoder
        if self.plot_cat_emb and hasattr(self.module.decoder, 'cat_bias'):
            # Plot and save in tensorboard logger
            self._plt_cat_emb()

    def _full_log_on_epoch_end(self):
        """Calculate metrics synchronously, then offload heavy plotting to a background thread."""
        from concurrent.futures import ThreadPoolExecutor
        # Get modes to do a full log on
        modes = self.log_full
        if isinstance(modes, str):
            modes = [modes]
        if len(self.log_full) == 0:
            return
        # Get cached steps from all modes
        data = {}
        _modes = []
        should_plot = self.plot_every_n_epochs > 0 and self.current_epoch % self.plot_every_n_epochs == 0
        for mode in modes:
            mode_data = self._process_epoch_cache(mode)
            if mode_data is None or len(mode_data.keys()) == 0:
                continue
            # Metrics must stay synchronous — they call self.log()
            self._log_full_metrics(mode, mode_data)
            _modes.append(mode)
            # Combine modes only if we plot
            if len(data) == 0 or not self.plot_umap:
                data = mode_data
                continue
            for k, v in data.items():
                new_v = mode_data.get(k)
                if new_v is not None:
                    data[k] = np.concatenate((v, new_v), axis=0)

        # Offload all plotting to a background thread so training continues immediately
        if should_plot and _modes:
            # Snapshot everything the plotting functions need (all numpy/CPU at this point)
            plot_data = {k: v.copy() if hasattr(v, 'copy') else v for k, v in data.items()}
            plot_modes = list(_modes)
            epoch = self.current_epoch

            def _plot_async():
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                try:
                    for mode in plot_modes:
                        self._compute_disentanglement_metrics(mode, plot_data)
                        if self.plot_f1_dist:
                            self._plt_f1_per_ctx_boxplots(plot_data)
                        if self.plot_cm:
                            self._plt_confusion(mode, plot_data)
                        if self.plot_clip_geom:
                            self._plt_clip_geometry(mode, plot_data)
                            self._plt_eff_weight_corr(mode, plot_data)
                    if self.plot_umap:
                        full_name = '-'.join(plot_modes) if len(plot_modes) > 1 else plot_modes[0]
                        for z_key in self.plot_umap_key:
                            if z_key in plot_data:
                                self._plt_umap(full_name, plot_data, current_epoch=epoch, z_key=z_key)
                except Exception as e:
                    logging.warning(f'[Plot] Background plotting failed at epoch {epoch}: {e}')
                finally:
                    plt.close('all')

            # Wait for any previous plot job to finish before starting a new one
            if hasattr(self, '_plot_future') and self._plot_future is not None:
                self._plot_future.result()
            if not hasattr(self, '_plot_executor'):
                from concurrent.futures import ThreadPoolExecutor
                self._plot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='plot')
            self._plot_future = self._plot_executor.submit(_plot_async)

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
        # Update automatic alignment temperature
        if mode == 'val':
            self._update_alignment_temp(f1)
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

    def _compute_disentanglement_metrics(self, mode: str, mode_data: dict):
        """
        Compute and log dataset disentanglement metrics on latent space.
        Logs scalars + a dataset similarity heatmap.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        z = mode_data.get(MODULE_KEYS.Z_SHARED_KEY)
        if z is None:
            z = mode_data.get(MODULE_KEYS.Z_KEY)
        if z is None:
            return
        
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        
        ds_labels = mode_data[REGISTRY_KEYS.BATCH_KEY]
        if isinstance(ds_labels, torch.Tensor):
            ds_labels = ds_labels.cpu().numpy()
        ds_labels = ds_labels.flatten()
        
        cls_labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        if isinstance(cls_labels, torch.Tensor):
            cls_labels = cls_labels.cpu().numpy()
        cls_labels = cls_labels.flatten()
        
        # Subsample for speed
        max_samples = 5000
        if len(z) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(z), max_samples, replace=False)
            z = z[idx]
            ds_labels = ds_labels[idx]
            cls_labels = cls_labels[idx]
        
        # ---- Scalar metrics ----
        
        # Dataset silhouette (lower = better mixing)
        if len(np.unique(ds_labels)) >= 2:
            sil_ds = silhouette_score(z, ds_labels, sample_size=min(2000, len(z)), random_state=42)
            self.logger.experiment.add_scalar(f"disentangle/{mode}_silhouette_dataset", sil_ds, self.current_epoch)
        
        # Class silhouette (higher = better separation)
        if len(np.unique(cls_labels)) >= 2:
            sil_cls = silhouette_score(z, cls_labels, sample_size=min(2000, len(z)), random_state=42)
            self.logger.experiment.add_scalar(f"disentangle/{mode}_silhouette_class", sil_cls, self.current_epoch)
        
        # Invariance score
        self.logger.experiment.add_scalar(
            f"disentangle/{mode}_invariance_score", sil_cls - sil_ds, self.current_epoch
        )
        
        # KNN dataset prediction (lower = better invariance)
        if len(np.unique(ds_labels)) >= 2:
            knn_ds = KNeighborsClassifier(n_neighbors=min(15, len(z) // 4))
            ds_acc = cross_val_score(knn_ds, z, ds_labels, cv=3, scoring='accuracy').mean()
            self.logger.experiment.add_scalar(f"disentangle/{mode}_knn_dataset_acc", ds_acc, self.current_epoch)
        
        # Within-class dataset mixing
        mixing_scores = []
        for cls in np.unique(cls_labels):
            cls_mask = cls_labels == cls
            cls_z = z[cls_mask]
            cls_ds = ds_labels[cls_mask]
            if len(cls_z) >= 10 and len(np.unique(cls_ds)) >= 2:
                mixing_scores.append(silhouette_score(cls_z, cls_ds, random_state=42))
        if mixing_scores:
            self.logger.experiment.add_scalar(
                f"disentangle/{mode}_within_class_ds_silhouette", np.mean(mixing_scores), self.current_epoch
            )
        
        # ---- Dataset centroid similarity heatmap ----
        unique_ds = np.unique(ds_labels)
        if len(unique_ds) < 2:
            return
        
        centroids = np.stack([z[ds_labels == ds].mean(axis=0) for ds in unique_ds])
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        sim_matrix = centroids_norm @ centroids_norm.T
        
        # Map to labels if available
        if self.batch_labels is not None:
            ds_names = [self.batch_labels[int(ds)] if int(ds) < len(self.batch_labels) else str(ds) 
                        for ds in unique_ds]
        else:
            ds_names = [str(ds) for ds in unique_ds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: dataset centroid similarity
        sns.heatmap(
            sim_matrix,
            xticklabels=ds_names,
            yticklabels=ds_names,
            annot=True, fmt='.2f',
            cmap='RdYlBu_r',
            vmin=-1, vmax=1,
            ax=ax1,
        )
        ax1.set_title(f'{mode} | Dataset centroid cosine similarity\n(uniform = good disentanglement)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Right: per-class dataset mixing scores
        if mixing_scores:
            cls_names = []
            cls_mixing = []
            cls_ds_counts = []
            for cls in np.unique(cls_labels):
                cls_mask = cls_labels == cls
                cls_z = z[cls_mask]
                cls_ds = ds_labels[cls_mask]
                if len(cls_z) >= 10 and len(np.unique(cls_ds)) >= 2:
                    sil = silhouette_score(cls_z, cls_ds, random_state=42)
                    cls_names.append(str(cls))
                    cls_mixing.append(sil)
                    cls_ds_counts.append(len(np.unique(cls_ds)))
            
            if cls_names:
                sort_idx = np.argsort(cls_mixing)[::-1]
                colors = ['#e74c3c' if m > 0.3 else '#f39c12' if m > 0.1 else '#2ecc71' 
                        for m in np.array(cls_mixing)[sort_idx]]
                
                ax2.barh(
                    range(min(30, len(cls_names))),
                    np.array(cls_mixing)[sort_idx[:30]],
                    color=colors[:30],
                    alpha=0.8,
                )
                ax2.set_yticks(range(min(30, len(cls_names))))
                ax2.set_yticklabels(np.array(cls_names)[sort_idx[:30]], fontsize=7)
                ax2.set_xlabel('Dataset silhouette within class\n(lower = better mixing)')
                ax2.set_title(f'{mode} | Per-class dataset mixing\n(red = dataset-dominated, green = well-mixed)')
                ax2.axvline(0, color='gray', linewidth=0.5)
        
        plt.tight_layout()
        self.logger.experiment.add_figure(f"disentangle/{mode}_overview", fig, self.current_epoch)
        plt.close(fig)
    
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
        """
        Plot correlation between class effects and CLIP weights.

        x = class effects
        y = weights

        One subplot per context.
        Logged to TensorBoard: clip/eff_weight_corr_{mode}
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import math

        # Get efficiency and predicted weights
        eff = mode_data.get(REGISTRY_KEYS.CLS_EFF_KEY, None)
        weights = mode_data.get("clip/weights", None)
        # Exit if data is not available
        if eff is None or weights is None:
            return
        # Get data contexts
        contexts = mode_data.get(REGISTRY_KEYS.BATCH_KEY, None)
        # Transform to arrays if need
        eff = np.asarray(eff)
        weights = np.asarray(weights)

        # flatten if needed
        eff = eff.reshape(-1)
        weights = weights.reshape(-1)

        if contexts is None:
            contexts = np.zeros_like(eff)

        # Get unique contexts to color for
        contexts = np.asarray(contexts).reshape(-1)
        # Optional mapping
        if self.batch_labels is not None:
            contexts = self.batch_labels[contexts]
        unique_ctx = np.unique(contexts)

        n_ctx = len(unique_ctx)
        n_cols = min(3, n_ctx)
        n_rows = math.ceil(n_ctx / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)
        # Scatter for each dataset
        for ax, ctx in zip(axes, unique_ctx):
            mask = contexts == ctx

            x = eff[mask]
            y = weights[mask]

            if len(x) == 0:
                ax.axis("off")
                continue

            r = np.corrcoef(x, y)[0, 1]

            ax.scatter(x, y, s=6, alpha=0.5)
            ax.set_title(f"{ctx} \nR={r:.2f}")
            ax.set_xlabel("real")
            ax.set_ylabel("predicted")

        for ax in axes[len(unique_ctx):]:
            ax.axis("off")

        fig.tight_layout()

        if hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_figure(
                f"clip/eff_weight_corr_{mode}",
                fig,
                global_step=self.current_epoch
            )

        plt.close(fig)
            
    def _plt_clip_geometry(self, mode: str, mode_data: dict[str, np.ndarray]):
        """
        Plot CLIP geometry sanity scatter:
        x = hardest negative similarity
        y = positive similarity

        One subplot per condition.
        Logged to TensorBoard: clip/geom_{mode}
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
        weights = mode_data.get("clip/weights", None)
        distances = mode_data.get("clip/dists", None)

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
        if distances is not None:
            df["distance"] = np.asarray(distances)

        # Optional mapping
        if self.batch_labels is not None:
            df["context"] = self.batch_labels[df["context"]]

        # ----- define conditions to plot -----
        conditions = [c for c in ["context", "label", "weight", "distance"] if c in df.columns]

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

        fig.suptitle(f"{mode} | CLIP geometry sanity", fontsize=14)
        plt.tight_layout()

        self.logger.experiment.add_figure(
            tag=f"clip/geom_{mode}",
            figure=fig,
            global_step=self.current_epoch,
        )

        plt.close(fig)

    def _plt_f1_per_ctx_boxplots(self, mode_data: dict[str, np.ndarray]):
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

        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        contexts = mode_data[REGISTRY_KEYS.BATCH_KEY]
        splits = mode_data[REGISTRY_KEYS.SPLIT_KEY]

        df = pd.DataFrame({
            "pred": preds,
            "label": labels,
            "context": contexts,
            "split": splits,
        })

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
            clip_n_cls = self.n_classes
            unknown_mask = (preds_t >= self.n_classes)
            if unknown_mask.any():
                # Assign unknown to classes that are outside of the training classes
                preds_t = preds_t.masked_fill(unknown_mask, self.n_classes)
                clip_n_cls += 1

            f1_per_class = multiclass_f1_score(
                preds_t,
                labels_t,
                num_classes=clip_n_cls,
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

            ax.set_title(f"{split} | per-context F1 (colored by class support)")
            ax.set_xlabel("Context")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(ctx_order)))
            ax.set_xticklabels(ctx_order, rotation=45, ha="right")

            plt.tight_layout()
            self.logger.experiment.add_figure(
                tag=f"f1_per_class_boxplot/{split}",
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
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Get shared latent embedding and annotation data, fall back to z if not in cached data
        embeddings = mode_data[z_key]
        labels = mode_data[REGISTRY_KEYS.LABELS_KEY]
        covs = mode_data[REGISTRY_KEYS.BATCH_KEY]
        modes = pd.Categorical(mode_data[REGISTRY_KEYS.SPLIT_KEY])
        
        # Check if model has pathway and module information
        has_pathways = getattr(self.module, 'cls2pw') is not None
        has_modules = getattr(self.module, 'cls2module') is not None

        # --- Subset: remove control cells
        if getattr(self.module, 'ctrl_class_idx') is not None:
            mask = labels != self.module.ctrl_class_idx
            embeddings, labels, covs, modes = embeddings[mask], labels[mask], covs[mask], modes[mask]
        # Add batch annotation
        if self.batch_labels is not None:
            cov_idx = self.batch_labels
            covs = cov_idx[covs]
        else:
            cov_idx = sorted(np.unique(covs))
        # --- Add class proxies if they exist
        if MODULE_KEYS.CLS_PROJ_KEY in mode_data and z_key != MODULE_KEYS.Z_KEY:
            cls_proxies = mode_data[MODULE_KEYS.CLS_PROJ_KEY]
            # Get proxies (excluding null)
            cls_proxies = cls_proxies[:self.module.n_cls]
            # Aggregate into smaller shapes if needed
            if cls_proxies.ndim == 3:
                cls_proxies = cls_proxies.mean(1)
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

        # --- cached UMAP transform
        cache = self.cache.setdefault("umap_cache", {})
        transformer_key = f"{z_key}_{mode}_umap_transformer"
        ref_key = f"{z_key}_{mode}_umap_reference"

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
        # Add covariate ordering
        df["cov"] = pd.Categorical(df["cov"], cov_idx, ordered=True)
        
        # Create pathway-level UMAP if given
        if has_pathways:
            pw_z = mode_data['z_s']
            pw_prox = mode_data['t_s']
            # Stack data
            pw_emb = np.concatenate((pw_z, pw_prox), axis=0)
            # Calculate UMAP over latent dimensions
            if pw_emb.shape[1] > 2:
                pw_emb = umap.UMAP(n_components=2).fit_transform(pw_emb)
            # Add pathway umap to plotting data frame
            n_proxies = pw_prox.shape[0]

            # proxy label ids
            pw_labels = self.module.cls2pw[labels].cpu().numpy().astype(str)
            proxy_labels = np.arange(n_proxies)
            proxy_covs = np.repeat("proxy", n_proxies)
            proxy_modes = np.repeat("proxy", n_proxies)
            proxy_is_obs = np.array(
                [True] * self.module.n_observed_pw + [False] * (n_proxies - self.module.n_observed_pw)
            )[:n_proxies]

            # --- combine into a unified DataFrame
            df_pw = pd.DataFrame({
                "UMAP_input_idx": np.arange(len(pw_labels) + n_proxies),
                "label_id": np.concatenate((pw_labels, proxy_labels)),
                "cov": np.concatenate((covs, proxy_covs)),
                "mode": np.concatenate((modes, proxy_modes)),
                "is_proxy": np.concatenate((np.zeros(len(pw_labels), dtype=bool), np.ones(n_proxies, dtype=bool))),
                "is_observed": np.concatenate((np.ones(len(pw_labels), dtype=bool), proxy_is_obs))
            })
            # Add UMAP projection to plotting df
            df_pw["UMAP1"], df_pw["UMAP2"] = pw_emb[:, 0], pw_emb[:, 1]

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
        N = 3 if has_modules or has_pathways else 2
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

        # 5. by pathways (if they exist)
        if has_pathways:
            _scatter_base(axes[2, 0], df_pw, "label_id", f"Pathways @ Epoch {current_epoch}", legend=False, proxy_legend=False)
        # 6. by modules (if they exist)
        if has_modules:
            df["module"] = self.module.cls2module[df.label_id].cpu().numpy().astype(str)
            _scatter_base(axes[2, 1], df, "module", f"Modules @ Epoch {current_epoch}", plot_proxy=False, proxy_legend=False, legend=False)
        plt.tight_layout()
        self.logger.experiment.add_figure(f"{mode}_{z_key}_umap", fig, current_epoch)
        plt.close(fig)

        # Save data to file
        if self.save_cache:
            umap_dir = os.path.join(self.logger.log_dir, 'data', 'umap')
            os.makedirs(umap_dir, exist_ok=True)
            csv_path = os.path.join(umap_dir, f"umap_data_epoch{current_epoch}.csv")
            df.to_csv(csv_path, index=False)

