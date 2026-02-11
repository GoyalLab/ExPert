import torch
import torch.nn.functional as F

from src.utils.constants import LOSS_KEYS


def select_proxies(
    z: torch.Tensor,
    y: torch.Tensor,
    emb: torch.Tensor,
    unique_proxies: bool,
    use_random_proxies: bool = False,
):
    """
    Returns:
        z_norm: (B, d)
        proxies: (U, d)
        targets: (B,)
        inv: (B,) mapping sample -> proxy index (only if unique_proxies)
    """
    device = z.device
    z_norm = F.normalize(z, dim=-1)
    emb_norm = F.normalize(emb, dim=-1)

    if unique_proxies:
        unique_classes, inv = torch.unique(y, return_inverse=True)
        proxies = emb_norm[unique_classes]
        targets = inv
    else:
        proxies = emb_norm[y]
        targets = torch.arange(z.size(0), device=device)
        inv = None

    if use_random_proxies:
        C = emb.size(0)
        proxies = emb_norm[torch.randperm(C, device=device)[:proxies.size(0)]]

    proxies = F.normalize(proxies, dim=-1)
    return z_norm, proxies, targets, inv

def clip_logits_z2c(z, proxies, T: float, center: bool = True, norm: bool = True):
    # Normalize both spaces before dot product
    if norm:
        z = F.normalize(z, dim=-1)
        proxies = F.normalize(proxies, dim=-1)
    # Supports single or multi-proxy
    if proxies.ndim == 3:
        sim = torch.einsum("bd,cmd->bcm", z, proxies)
        s = torch.logsumexp(sim / T, dim=-1)
    else:
        s = (z @ proxies.T) / T
    # Center logits on batch mean
    if center:
        return s - s.mean(dim=-1, keepdim=True)
    else:
        return s
    
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

def hard_negative_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    label_smoothing: float = 0.0,
):
    B, C = logits.shape
    device = logits.device
    
    # Use entire class space if k == 0
    if k is None or k == 0:
        k_eff = C - 1
        logits_mined = logits
        t = targets
    else:
        # Mask positives
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[torch.arange(B), targets] = True
        # Get all scores for text negatives
        neg_scores = logits.masked_fill(mask, -1e9)
        # Choose k hard negatives
        k_eff = int(min(k, C - 1))
        # Get top k
        hard_negs = neg_scores.topk(k_eff, dim=-1).values
        # Get positive logits
        pos = logits[torch.arange(B), targets].unsqueeze(1)
        # Combine
        logits_mined = torch.cat([pos, hard_negs], dim=1)
        t = torch.zeros(B, dtype=torch.long, device=device)
    # Return cross-entrpoy loss
    return F.cross_entropy(
        logits_mined.clamp(-20, 20),
        t,
        reduction="none",
        label_smoothing=label_smoothing,
    )

def clip_reverse_loss(
    z: torch.Tensor,
    proxies: torch.Tensor,
    inv: torch.Tensor,
    T: float,
    k: int,
    label_smoothing: float,
):
    U = proxies.size(0)
    loss_list = []

    logits = (proxies @ z.T) / T  # (U, B)

    for u in range(U):
        mask = inv == u
        num_pos = mask.sum()
        if num_pos == 0:
            continue

        pos = logits[u, mask]
        neg = logits[u, ~mask]

        if k and neg.numel() > 0:
            k_eff = int(min(k, neg.numel()))
            hard_neg = neg.topk(k_eff).values
            logits_mined = torch.cat(
                [pos.unsqueeze(1),
                hard_neg.unsqueeze(0).expand(num_pos, -1)],
                dim=1,
            )
            loss = F.cross_entropy(
                logits_mined.clamp(-20, 20),
                torch.zeros(num_pos, device=z.device, dtype=torch.long),
                reduction="none",
                label_smoothing=label_smoothing,
            )
            loss_list.append(loss)

    if not loss_list:
        return torch.zeros(1, device=z.device, requires_grad=True)

    return torch.cat(loss_list)

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

def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    # Check if loss is a tensor
    if loss is None or not isinstance(loss, torch.Tensor):
        return loss
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'batchmean':
        return loss.sum(-1).mean()
    else:
        raise ValueError(f'Invalid reduction: {reduction}')
    
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

def loss(
    z: torch.Tensor,
    y: torch.Tensor,
    emb: torch.Tensor,
    T: float = 0.1,
    k: int = 0,
    unique_proxies: bool = True,
    use_reverse: bool = False,
    label_smoothing: float = 0.0,
    pseudobulk_weight: float = 0.0,
    pull_weight: float = 0.1,
    reduction: str = 'mean',
    training: bool = True,
    n_unseen: int = 0,
    weight_tau: float = 1.0,
):
    # Flatten class labels
    y = y.flatten()

    # ---------- select proxies ----------
    z_norm, proxies, targets, inv = select_proxies(
        z, y, emb,
        unique_proxies=unique_proxies
    )
    # Normalize embedding
    emb_norm = F.normalize(emb, dim=-1)

    # Indices of seen classes (needed for unseen sampling)
    seen_indices = torch.unique(y) if unique_proxies else y
    # Sample unseen indices from embedding
    unseen = None
    if training and n_unseen > 0:
        unseen = sample_unseen_proxies(
            emb_norm=emb_norm,
            seen_indices=seen_indices,
            n_unseen=n_unseen,
        )

    # ---------- latent pseudobulk ----------
    if training and pseudobulk_weight > 0:
        z_pseudo, y_pseudo = compute_pseudobulks(z_norm, y)
        if z_pseudo is not None:
            # Align pseudobulks to text
            _, pb_proxies, pb_targets, _ = select_proxies(
                z_pseudo, y_pseudo, emb_norm, unique_proxies=True
            )
            # Add unseen
            pb_proxies = augment_with_unseen(pb_proxies, unseen)
            logits_pb = clip_logits_z2c(z_pseudo, pb_proxies, T)
            loss_pb = hard_negative_ce(
                logits_pb, pb_targets, k, label_smoothing
            )

            # Pull cells toward pseudobulk
            loss_pull = pseudobulk_pull_loss(
                z_norm, y, z_pseudo, y_pseudo
            )
        else:
            loss_pb = 0.0
            loss_pull = 0.0
    else:
        loss_pb = 0.0
        loss_pull = 0.0

    # ---------- main z → class ----------
    augmented_proxies = augment_with_unseen(proxies, unseen)
    logits = clip_logits_z2c(z_norm, augmented_proxies, T)
    loss_z2c = hard_negative_ce(
        logits, targets, k, label_smoothing
    )
    
    # Weight logits based on margin to others (certainty)
    if training and weight_tau > 0:
        with torch.no_grad():
            pos = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            top2 = torch.topk(logits, k=2, dim=-1).values
            neg = torch.where(top2[:, 0] == pos, top2[:, 1], top2[:, 0])
            margin_vals = pos - neg
            weights = torch.sigmoid(margin_vals * weight_tau)
        # Update cell loss based on margin to others
        loss_z2c = loss_z2c * weights

    # ---------- reverse ----------
    if use_reverse and inv is not None:
        loss_c2z = clip_reverse_loss(
            z_norm, proxies, inv, T, k, label_smoothing
        )
    else:
        loss_c2z = 0.0
        
    # Log margin between pos and neg
    m = margin(logits, targets).detach().item()
    # Get logit entropy
    e = entropy(logits).detach().item()
    
    # ---------- reduce -----------
    loss_z2c = _reduce_loss(loss_z2c, reduction)
    loss_pb = _reduce_loss(loss_pb, reduction)
    loss_pull = _reduce_loss(loss_pull, reduction)
    loss_c2z = _reduce_loss(loss_c2z, reduction)

    # ---------- combine ----------
    loss = (
        loss_z2c
        + pseudobulk_weight * loss_pb
        + pull_weight * loss_pull
    )

    if use_reverse:
        loss = 0.5 * (loss + loss_c2z)

    return {
        LOSS_KEYS.LOSS: loss,
        "clip/loss_z2c": loss_z2c,
        "clip/loss_pb": loss_pb,
        "clip/loss_pull": loss_pull,
        "clip/loss_c2z": loss_c2z,
        "clip/margin": m,
        "clip/entropy": e,
    }