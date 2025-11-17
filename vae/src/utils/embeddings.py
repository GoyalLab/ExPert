import torch
import torch.nn.functional as F


def replace_with_unseen_labels(
        labels: torch.Tensor, 
        n_seen: torch.Tensor, 
        n_total: torch.Tensor, 
        p: float = 0.1
    ) -> torch.Tensor:
    """Randomly replace label indices with unseen indices."""
    mask = torch.rand_like(labels.float()) < p
    unseen_idx = torch.randint(n_seen, n_total, labels.shape, device=labels.device)
    return torch.where(mask, unseen_idx, labels)[0]

def manifold_regularization(
        ext_emb: torch.Tensor, 
        proj_emb: torch.Tensor, 
        n_sample: int = 128
    ) -> torch.Tensor:
    """Randomly sample a subset to approximate pairwise geometry."""
    idx = torch.randperm(ext_emb.size(0), device=ext_emb.device)[:n_sample]
    e_ext = F.normalize(ext_emb[idx], dim=-1)
    e_proj = F.normalize(proj_emb[idx], dim=-1)
    sim_ext = e_ext @ e_ext.T
    sim_proj = e_proj @ e_proj.T
    return F.mse_loss(sim_ext, sim_proj)
