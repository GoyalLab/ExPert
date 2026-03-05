import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import LOSS_KEYS

    
def margin(logits, targets):
    B, C = logits.shape
    pos = logits[torch.arange(B), targets]  # (B,)

    neg = logits.clone()
    neg[torch.arange(B), targets] = -1e9    # mask positive safely
    neg_max = neg.max(dim=1).values          # hardest negative

    margin = (pos - neg_max).mean()
    return margin

def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return entropy.mean()

def compute_pseudobulks(
    z: torch.Tensor,
    y: torch.Tensor,
    min_cells: int = 3,
):
    """
    Returns:
        z_pseudo: (C_batch, d)
        y_pseudo: (C_batch,)
    """
    z_pseudo = []
    y_pseudo = []

    for c in torch.unique(y):
        mask = y == c
        if mask.sum() >= min_cells:
            z_pseudo.append(z[mask].mean(dim=0))
            y_pseudo.append(c)

    if not z_pseudo:
        return None, None

    return torch.stack(z_pseudo), torch.tensor(y_pseudo, device=z.device)

def pseudobulk_pull_loss(z, y, z_pseudo, y_pseudo):
    proto_map = {int(c): z_pseudo[i] for i, c in enumerate(y_pseudo)}
    target = torch.stack([proto_map[int(c)] for c in y])
    return F.mse_loss(z, target, reduction="none")
    
def sample_unseen_proxies(
    emb_norm: torch.Tensor,
    seen_indices: torch.Tensor,
    n_unseen: int,
):
    """
    emb_norm: (C_all, d)
    seen_indices: (U,) indices used in current batch
    """
    if n_unseen <= 0:
        return None

    all_idx = torch.arange(emb_norm.size(0), device=emb_norm.device)
    mask = torch.ones_like(all_idx, dtype=torch.bool)
    mask[seen_indices] = False

    unseen_idx = all_idx[mask]
    if unseen_idx.numel() == 0:
        return None

    n_eff = int(min(n_unseen, unseen_idx.numel()))
    sampled = unseen_idx[torch.randperm(unseen_idx.numel())[:n_eff]]
    return emb_norm[sampled]

def augment_with_unseen(
    proxies: torch.Tensor,
    unseen: torch.Tensor | None,
):
    """
    proxies: (U, d)
    unseen: (K, d) or None
    """
    if unseen is None:
        return proxies

    return torch.cat([proxies, unseen], dim=0)

def logit_margins(logits: torch.Tensor, labels: torch.Tensor):
    # Get positives
    pos = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    # Check top 2 negatives
    top2 = torch.topk(logits, k=2, dim=-1).values
    neg = torch.where(top2[:, 0] == pos, top2[:, 1], top2[:, 0])
    # Compute margins from positives to closest negatives
    return pos - neg

def per_class_margin_equalizer(
    margins: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    # Compute per-class mean margins
    unique_classes = targets.unique()
    class_means = []

    for c in unique_classes:
        mask = targets == c
        if mask.sum() > 0:
            class_means.append(margins[mask].mean())

    if len(class_means) <= 1:
        return torch.tensor(0.0, device=margins.device)

    class_means = torch.stack(class_means)

    # Penalize variance across class means
    return class_means.var(unbiased=False)

def infer_responder_weights(
    logits: torch.Tensor,
    y: torch.Tensor,
    temp: float = 1.5,
    min_w: float = 0.15,
    max_w: float = 1.0,
    detach: bool = True,
):
    """
    Fast responder weighting from CLIP margins.

    Args:
        logits: (B, C)
        y:      (B,)
        temp:   sigmoid softness (higher = flatter)
        min_w:  floor so samples never die
        detach: stop gradients through weights

    Returns:
        weights: (B,)
    """

    device = logits.device
    B = logits.size(0)

    # ----------------------
    # positive similarity
    # ----------------------
    pos = logits[torch.arange(B, device=device), y]

    # ----------------------
    # hardest negative
    # ----------------------
    neg = logits.clone()
    neg[torch.arange(B, device=device), y] = -torch.inf
    hardest = neg.max(dim=1).values

    # ----------------------
    # margin
    # ----------------------
    margin = pos - hardest

    # ----------------------
    # normalize by batch std (VERY IMPORTANT)
    # prevents early collapse / exploding weights
    # ----------------------
    std = margin.std().clamp_min(1e-3)
    margin = margin / std

    # ----------------------
    # Softmax weighting
    # ----------------------
    weights = torch.softmax(margin / temp, dim=0) * len(margin)

    # keep samples alive
    weights = weights.clamp(min_w, max_w)

    if detach:
        weights = weights.detach()

    return weights


class ClipLoss(nn.Module):

    def __init__(
        self,
        use_reverse: bool = False,
        unique_proxies: bool = True,
        reduction: str = 'mean',
        infer_weights: bool = False,
        normalize: bool = True,
        center_logits: bool = False,
        memory_size: int = 0,
        use_global: bool = True,
        n_global: int = 256,
        label_smoothing: float = 0.0,
        noise_std: float = 0.01,
        null_frac_target: float = 0.2,
        null_threshold: float = 0.5,
        alpha: float = 0.7,
        global_weight: float = 0.3,
        contrastive_weight: float = 0.3,
        eps: float = 1e-6,
        supervision_dropout_rate: float = 0.0,
        cls_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        # Store settings here
        self.memory_size = memory_size
        self.use_global = use_global
        self.n_global = n_global
        self.label_smoothing = label_smoothing
        self.unique_proxies = unique_proxies
        self.reduction = reduction
        self.infer_weights = infer_weights
        self.noise_std = noise_std
        self.normalize = normalize
        self.center_logits = center_logits
        self.alpha = alpha
        self.use_reverse = use_reverse and alpha != 1.0
        self.global_weight = global_weight
        self.contrastive_weight = contrastive_weight
        # Null class settings
        self.null_frac_target = null_frac_target
        self.null_threshold = null_threshold
        self.eps = eps
        self.supervision_dropout_rate = supervision_dropout_rate
        # Class weights based on support
        self.cls_weights = cls_weights
        
        # Set batch memory
        if self.memory_size > 0:
            self.use_memory = True
            # memory buffers
            self.register_buffer("mem_z", torch.empty(0))
            self.register_buffer("mem_y", torch.empty(0, dtype=torch.long))
            # Enable unique proxies when using memory
            unique_proxies = True
        else:
            self.use_memory = False

    @torch.no_grad()
    def _update_memory(self, z, y):
        """Update latent space and label memory

        Args:
            z (torch.Tensor): aligned latent space (rna)
            y (torch.Tensor): class labels
        """
        # Init or update memory blocks
        if self.mem_z.numel() == 0:
            self.mem_z = z.detach()
            self.mem_y = y.detach()
        else:
            self.mem_z = torch.cat([self.mem_z, z.detach()], dim=0)
            self.mem_y = torch.cat([self.mem_y, y.detach()], dim=0)

        # Truncate overflowing memory
        if self.mem_z.size(0) > self.memory_size:
            excess = self.mem_z.size(0) - self.memory_size
            self.mem_z = self.mem_z[excess:]
            self.mem_y = self.mem_y[excess:]
            
    @property
    def memory_buffer_size(self):
        if self.use_memory:
            return self.mem_z.size(0)
        return 0
            
    def null_fraction_loss(
        self,
        logits: torch.Tensor,
        null_idx: torch.Tensor,
        T: float = 0.1
    ):
        probs = torch.softmax(logits, dim=-1)
        p_null = probs[:, null_idx].sum(-1)
        # soft indicator of "null-like"
        soft_is_null = torch.sigmoid((p_null - self.null_threshold)/T)
        frac = soft_is_null.mean()
        return ((frac - self.null_frac_target) / (self.null_frac_target + self.eps)) ** 2

    def cross_context_supcon_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        contexts: torch.Tensor,
        gate: torch.Tensor | None = None,
        T_i: torch.Tensor | None = None,
        temperature: float = 0.2,
    ):
        """
        Supervised contrastive loss forcing SAME LABEL across DIFFERENT CONTEXTS together.
        Gated by responder probability and modulated by per-sample temperature.
        """
        z = F.normalize(z, dim=-1)
        B = z.size(0)
        device = z.device

        # Per-pair temperature: use geometric mean of both samples' T_i
        if T_i is not None:
            T_i = T_i.squeeze(-1)  # (B,)
            pair_T = torch.sqrt(T_i[:, None] * T_i[None, :])  # (B, B)
        else:
            pair_T = temperature
        
        # Get pair-wise similarity
        sim = z @ z.T / pair_T

        # Remove self similarity
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, -1e9)

        # Positives = same label AND different context
        pos_mask = (
            (labels[:, None] == labels[None, :]) &
            (contexts[:, None] != contexts[None, :])
        )
        # SupCon loss
        log_denom = torch.logsumexp(sim, dim=1)
        sim_pos = sim.masked_fill(~pos_mask, -1e9)
        log_pos = torch.logsumexp(sim_pos, dim=1)

        valid = pos_mask.any(dim=1)
        # Check if there are any positives in the batch
        if valid.any():
            loss = -(log_pos[valid] - log_denom[valid])
            # Gate by responder probability
            if gate is not None:
                gate_valid = gate[valid]
                loss = (loss * gate_valid).sum() / gate_valid.sum().clamp(min=1)
            else:
                loss = loss.mean()
            return loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    # --------------------------------------------------
    # HARD NEGATIVE CE
    # --------------------------------------------------
    def _hard_negative_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        k: int,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Cross-entropy with optional hard negative mining.

        Args:
            logits          : (B, C) similarity scores
            targets         : (B,) class indices
            k               : number of hard negatives (0 = full softmax)
            label_smoothing : optional label smoothing

        Returns:
            loss : (B,) per-sample loss
        """

        B, C = logits.shape
        device = logits.device

        # ---- Full softmax (standard CE) ----
        if k is None or k == 0:
            # ---- Cross entropy ----
            return F.cross_entropy(
                logits,
                targets,
                reduction="none",
                label_smoothing=self.label_smoothing,
                weight=weights,
            )

        # ---- Hard negative mining ----
        # Mask positives
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[torch.arange(B, device=device), targets] = True

        # Get negative scores
        neg_scores = logits.masked_fill(mask, -1e9)

        # Choose k hardest negatives
        k_eff = int(min(k, C - 1))
        hard_negs = neg_scores.topk(k_eff, dim=-1).values

        # Get positive logits
        pos = logits[torch.arange(B, device=device), targets].unsqueeze(1)

        # Combine positive + negatives
        logits_mined = torch.cat([pos, hard_negs], dim=1)

        # Target is always index 0 (the positive)
        t = torch.zeros(B, dtype=torch.long, device=device)

        # ---- Cross entropy ----
        return F.cross_entropy(
            logits_mined.clamp(-20, 20),
            t,
            reduction="none",
            label_smoothing=self.label_smoothing,
            weight=weights,
        )
        
    def _select_proxies(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        emb: torch.Tensor,
    ):
        """
        Returns:
            z_norm: (B, d)
            proxies: (U, d)
            targets: (B,)
            inv: (B,) mapping sample -> proxy index (only if unique_proxies)
        """
        
        # Use unique class proxies for clip only
        if self.unique_proxies:
            unique_classes, inv = torch.unique(y, return_inverse=True)
            proxies = emb[unique_classes]
            targets = inv
        else:
            proxies = emb
            targets = y
            inv = None
            unique_classes = y
        # Assign class weights to batch
        if self.cls_weights is not None:
            if self.unique_proxies:
                weights = self.cls_weights[unique_classes]
            else:
                weights = self.cls_weights
        else:
            weights = None

        return z, proxies, targets, inv, weights
    
    def _logits_z2c(self, z, proxies, T: float):
        # Normalize both spaces before dot product
        if self.normalize:
            z = F.normalize(z, dim=-1)
            proxies = F.normalize(proxies, dim=-1)
        # Supports single or multi-proxy
        if proxies.ndim == 3:
            sim = torch.einsum("bd,cmd->bcm", z, proxies)
            # Center logits on batch mean
            s = torch.logsumexp(sim / T, dim=-1)
        else:
            s = (z @ proxies.T) / T
        # Center logits on mean over all predictions
        if self.center_logits:
            s = s - s.mean(dim=-1, keepdim=True)
        return s
    
    def _clip_reverse_loss(
        self,
        z: torch.Tensor,          # (B, D)
        proxies: torch.Tensor,    # (C,D) or (C,M,D)
        labels: torch.Tensor,     # (B,) class id per sample
        T: float,
    ):
        """
        Reverse CLIP loss supporting multi-prototype class embeddings.

        Each prototype pulls all samples of its class.
        Stable multi-positive InfoNCE.
        """

        device = z.device

        # ----------------------
        # flatten proxies if multi
        # ----------------------
        if proxies.ndim == 3:
            C, M, D = proxies.shape
            proxies_flat = proxies.reshape(C * M, D)

            # map each prototype → class id
            proto_classes = (
                torch.arange(C, device=device)
                .repeat_interleave(M)
            )
        else:
            proxies_flat = proxies
            proto_classes = torch.arange(proxies.size(0), device=device)

        # Normalize before similarity
        z = F.normalize(z, dim=-1)
        proxies_flat = F.normalize(proxies_flat, dim=-1)
        # ----------------------
        # logits: (num_proto, B)
        # ----------------------
        logits = (proxies_flat @ z.T) / T

        # ----------------------
        # build positive mask
        # ----------------------
        # proto_class == sample_label
        pos_mask = proto_classes[:, None] == labels[None, :]

        # ----------------------
        # multi-positive InfoNCE
        # ----------------------
        log_all = torch.logsumexp(logits, dim=1)
        log_pos = torch.logsumexp(
            logits.masked_fill(~pos_mask, -1e9),
            dim=1,
        )
        # Final contrastive loss
        loss = log_all - log_pos

        # remove prototypes with zero positives
        valid = pos_mask.any(dim=1)

        if valid.any():
            return loss[valid].mean()
        else:
            return torch.zeros((), device=device, requires_grad=True)
        
    def _reduce_loss(
        self,
        loss: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # pass-through for non-tensor
        if loss is None or not isinstance(loss, torch.Tensor):
            return loss

        # -------------------------------------------------
        # Apply weights if provided
        # -------------------------------------------------
        if weights is not None:
            # reshape weights to broadcast over loss dims
            w = weights.view(-1, *([1] * (loss.ndim - 1)))
            loss = loss * w

        # -------------------------------------------------
        # Reduction logic
        # -------------------------------------------------
        if self.reduction == "mean":

            if weights is None:
                return loss.mean()
            else:
                # divide by effective weight sum
                denom = weights.sum().clamp_min(1e-12)
                return loss.sum() / denom

        elif self.reduction == "sum":
            return loss.sum()

        elif self.reduction == "batchmean":
            # assumes first dim = batch
            loss_sum = loss.sum(dim=tuple(range(1, loss.ndim)))

            if weights is None:
                return loss_sum.mean()
            else:
                denom = weights.sum().clamp_min(1e-12)
                return loss_sum.sum() / denom
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
        
    def global_loss(
        self,
        z_norm: torch.Tensor,
        y: torch.Tensor,
        emb: torch.Tensor,
        T: float,
    ):
        C = emb.size(0)
        neg_pool = torch.arange(C, device=emb.device)

        # always include positives
        unique_y = y.unique()

        # sample extra negatives
        target = int(min(self.n_global, C))
        n_extra = int(max(target - unique_y.numel(), 0))

        neg_candidates = neg_pool[~torch.isin(neg_pool, unique_y)]

        if neg_candidates.numel() > 0 and n_extra > 0:
            extra = neg_candidates[
                torch.randperm(neg_candidates.numel(), device=emb.device)[:n_extra]
            ]
            global_idx = torch.cat([unique_y, extra])
        else:
            global_idx = unique_y
        # Subset embedding to sampled indices
        global_emb = emb[global_idx]

        # remap targets
        remap = torch.full((C,), -1, device=emb.device)
        remap[global_idx] = torch.arange(global_idx.size(0), device=emb.device)
        targets_global = remap[y]
        # Get full logits
        logits_full = self._logits_z2c(z_norm, global_emb, T)
        return F.cross_entropy(
            logits_full, 
            targets_global, 
            reduction='none',
            weight=self.cls_weights
        )
    
    def intra_class_spread(self, z: torch.Tensor, y: torch.Tensor):
        """Enforce intra class spread to prevent collapse."""
        sim = z @ z.T
        same = y[:,None] == y[None,:]
        return sim[same].mean()

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        ctx: torch.Tensor,
        emb: torch.Tensor,
        T: float = 0.1,
        k: int = 0,
        weights: torch.Tensor | None = None,
        T_i: torch.Tensor | None = None,
        class_repel: float = 0.0,
    ):

        # Flatten class labels
        y = y.flatten()
        # store original supervised batch
        z_batch = z
        # Add some noise to the latent space during training
        if self.training and self.noise_std > 0:
            z_batch = z_batch + torch.randn_like(z_batch) * self.noise_std
            #proxies = proxies + torch.randn_like(proxies) * self.noise_std
            #proxies = F.normalize(proxies, dim=-1)
        # Add memory negatives
        if self.training and self.use_memory and self.mem_z.numel() > 0:
            mem_z = F.normalize(self.mem_z.to(z.device), dim=-1)
            mem_y = self.mem_y.to(z.device)
            # append ONLY to embeddings used for similarity
            z_all = torch.cat([z_batch, mem_z], dim=0)
            y_all = torch.cat([y, mem_y], dim=0)

            # weights must match too
            if weights is not None:
                pad = torch.ones(mem_y.size(0), device=z.device)
                weights = torch.cat([weights, pad], dim=0)
        else:
            z_all = z_batch
            y_all = y
        
        # ---------- select proxies ----------
        z_norm, proxies, targets, inv, support_weights = self._select_proxies(
            z_all, y_all, emb,
        )
        
        # ---------- supervised contrastive loss -----
        if self.contrastive_weight > 0:
            loss_supcon = self.cross_context_supcon_loss(
                z_batch, y, ctx, weights, T_i, temperature=T
            )
        else:
            loss_supcon = 0.0

        # -------------------------
        # compute logits vs class embeddings
        # -------------------------
        T_i = T_i if T_i is not None else T
        logits = self._logits_z2c(z_norm, proxies, T_i)
        loss_z2c = self._hard_negative_ce(
            logits, targets, k, weights=support_weights
        )
        # Add full embedding logits
        if self.use_global:
            loss_z2c_full = self.global_loss(z_norm, y, emb, T_i)
        else:
            loss_z2c_full = 0.0
        
        # Weight logits based on margin to others (certainty)
        with torch.no_grad():
            margin_vals = logit_margins(logits, targets)
            margin_std = margin_vals.std()

        # Calculate responder weights
        if self.infer_weights:
            inf_weights = infer_responder_weights(
                logits=logits,
                y=targets,
            )
        else:
            # Set all weights to 1
            inf_weights = torch.ones(logits.size(0), device=targets.device)
        
        # Weight loss weights per cell
        if self.training:
            # Apply cell weights
            if weights is None:
                weights = inf_weights
            #else:
            #    weights = weights * inf_weights
        else:
            # Never use weights in validation
            weights = None

        # ---------- reverse ----------
        if self.use_reverse and inv is not None:
            rev_T = T_i.mean() if T_i is not None else T
            loss_c2z = self._clip_reverse_loss(
                z_norm,     # batch + memory
                proxies,   # full class embeddings
                targets,     # batch + memory
                rev_T,          # Use base T
            )
        else:
            loss_c2z = 0.0
            
        # Log margin between pos and neg
        m = margin(logits, targets).detach().item()
        # Get logit entropy
        e = entropy(logits).detach().item()
        
        # Apply supervision dropout
        if self.training and self.supervision_dropout_rate > 0:
            sup_drop = torch.bernoulli(
                torch.full((loss_z2c.size(0),), 1-self.supervision_dropout_rate, device=loss_z2c.device),
            )
            loss_z2c = loss_z2c * sup_drop
            loss_c2z = loss_c2z * sup_drop
            
        # Apply intra-class repel to prevent collapse and overfitting
        if class_repel > 0:
            loss_cls_repel = self.intra_class_spread(z_norm, y)
        else:
            loss_cls_repel = 0.0
        
        # ---------- reduce -----------
        loss_z2c = self._reduce_loss(loss_z2c, weights=weights)
        loss_z2c_full = self._reduce_loss(loss_z2c_full, weights=weights)
        loss_c2z = self._reduce_loss(loss_c2z)
        loss_cls_repel = self._reduce_loss(loss_cls_repel)

        # ---------- combine ----------
        loss = loss_z2c
        if self.use_global:
            loss = (1-self.global_weight) * loss + self.global_weight * loss_z2c_full
        
        # Add class repel loss
        loss = loss + class_repel * loss_cls_repel

        # Add reverse loss with alpha on forward
        if self.use_reverse:
            loss = self.alpha * loss + (1-self.alpha) * loss_c2z
        # Add contrastive loss
        if self.contrastive_weight > 0:
            loss = loss + self.contrastive_weight * loss_supcon
        # Update memory (after loss)
        if self.training and self.use_memory:
            self._update_memory(z.detach(), y.detach())
        # Collect loss results
        return {
            LOSS_KEYS.LOSS: loss,
            "clip/loss_z2c": loss_z2c,
            "clip/loss_c2z": loss_c2z,
            "clip/loss_z2c_full": loss_z2c_full,
            "clip/loss_cls_repel": loss_cls_repel,
            "clip/loss_supcon": loss_supcon,
            "clip/margin_item": m,
            "clip/entropy": e,
            "clip/margin": margin_vals,
            "clip/margin_mean": margin_vals.mean(),
            "clip/margin_std": margin_std,
            "clip/memory_size": self.memory_buffer_size,
        }
