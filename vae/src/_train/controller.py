import torch.nn as nn

import logging


class ClipTemperatureController:
    def __init__(
        self,
        T_init: float = 0.3,
        T_min: float = 0.05,
        T_step: float = 0.1,
        ema_decay: float = 0.8,
        slope_eps: float = 1e-3,
        patience: int = 1,
        min_epochs: int = 5,
    ):
        self.T = T_init
        self.T_min = T_min
        self.T_step = T_step

        self.ema_decay = ema_decay
        self.slope_eps = slope_eps
        self.patience = patience
        self.min_epochs = min_epochs

        self.f1_ema = None
        self.prev_f1_ema = None
        self.plateau_counter = 0
        self.switched_steps = 0

    def update(self, current_epoch: int, val_f1: float) -> float:
        """
        Call once per validation epoch.
        Returns updated temperature.
        """

        # --- EMA smoothing ---
        if self.f1_ema is None:
            self.f1_ema = val_f1
            self.prev_f1_ema = val_f1
            return False

        self.prev_f1_ema = self.f1_ema
        self.f1_ema = (
            self.ema_decay * self.f1_ema
            + (1 - self.ema_decay) * val_f1
        )

        # --- compute slope ---
        slope = self.f1_ema - self.prev_f1_ema

        # --- guard: minimum warmup ---
        if current_epoch < self.min_epochs:
            return False

        # --- plateau detection ---
        if slope < self.slope_eps:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0

        # --- temperature drop ---
        if self.plateau_counter >= self.patience and self.T > self.T_min:
            old_T = self.T
            self.T = max(self.T - self.T_step, self.T_min)
            self.plateau_counter = 0
            self.switched_steps += 1
            logging.info(
                f"[CLIP T] Plateau detected → T {old_T:.3f} → {self.T:.3f}"
            )
            return True
        return False


class ClipController(nn.Module):
    def __init__(
        self,
        # --- targets ---
        margin_target: float = 0.5,
        entropy_target: float = 1.5,
        entropy_min: float = 1.0,

        # --- clip weight ---
        clip_weight_init: float = 1.0,
        clip_weight_min: float = 0.1,
        clip_weight_max: float = 2.0,

        # --- temperature ---
        T_init: float = 0.5,
        T_min: float = 0.1,
        T_max: float = 0.8,

        # --- smoothing ---
        decay: float = 0.9,
        update_every_n_steps: int = 200,
        schedule_weight: bool = False
    ):
        super().__init__()
        # CLIP weight
        self.clip_weight = clip_weight_init
        self.clip_weight_min = clip_weight_min
        self.clip_weight_max = clip_weight_max

        # Temperature
        self._T = T_init
        self.T_min = T_min
        self.T_max = T_max

        # EMA stats
        self.margin_ema = None
        self.entropy_ema = None
        self.decay = decay

        # Targets
        self.margin_target = margin_target
        self.entropy_target = entropy_target
        self.entropy_min = entropy_min
        # Counter
        self.c = 0
        self.update_every_n_steps = update_every_n_steps
        self.schedule_weight = schedule_weight
        
    @property
    def T(self) -> float:
        if self.training:
            return self._T
        else:
            return self.T_min
        
    def _state(self) -> dict:
        return {
            "clip_weight": self.clip_weight,
            "clip/T": self.T,
            "clip/margin_ema": self.margin_ema,
            "clip/entropy_ema": self.entropy_ema,
        }
    
    @property
    def collapse(self):
        if self.entropy_ema is None:
            return False
        return self.entropy_ema < self.entropy_min
    
    @property
    def confident(self):
        if self.margin_ema is None:
            return False
        return self.margin_ema > self.margin_target
    
    @property
    def healthy_entropy(self):
        if self.entropy_ema is None:
            return False
        return self.entropy_ema > self.entropy_target

    def update(self, margin: float, entropy: float):
        # Update step counter
        self.c += 1
        # -----------------------------
        # EMA smoothing
        # -----------------------------
        if self.margin_ema is None:
            self.margin_ema = margin
            self.entropy_ema = entropy
        else:
            self.margin_ema = (
                self.decay * self.margin_ema + (1 - self.decay) * margin
            )
            self.entropy_ema = (
                self.decay * self.entropy_ema + (1 - self.decay) * entropy
            )

        # -----------------------------
        # Detect regimes
        # -----------------------------
        collapse = self.entropy_ema < self.entropy_min
        confident = self.margin_ema > self.margin_target
        healthy_entropy = self.entropy_ema > self.entropy_target

        # Only update every n steps
        if self.update_every_n_steps > 1:
            if self.c % self.update_every_n_steps != 0:
                return self._state()

        # -----------------------------
        # CLIP weight update
        # -----------------------------
        if self.schedule_weight:
            if collapse and confident:
                self.clip_weight *= 0.90
            elif collapse:
                self.clip_weight *= 0.925
            elif confident:
                self.clip_weight *= 0.975
            else:
                # Increase clip weight
                self.clip_weight *= 1.025
        # Clamp clip weight
        self.clip_weight = float(
            max(self.clip_weight_min,
                min(self.clip_weight, self.clip_weight_max))
        )

        # -----------------------------
        # Temperature update
        # -----------------------------
        if healthy_entropy:
            # Safe to sharpen slowly
            self._T *= 0.975

        self._T = float(max(self.T_min, min(self._T, self.T_max)))
        # Return controlled weights
        return {
            "clip_weight": self.clip_weight,
            "clip/T": self.T,
            "clip/margin_ema": self.margin_ema,
            "clip/entropy_ema": self.entropy_ema,
        }
