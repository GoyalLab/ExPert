import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS


def soft_rank(x, tau=1.0):
    diff = x.unsqueeze(-1) - x.unsqueeze(-2)
    P = torch.sigmoid(diff / tau)
    return P.sum(-1) / x.size(-1)


class BatchAugmentation(nn.Module):

    def __init__(
        self,
        n_augmentations=3,
        dropout=0.2,
        downsample_min=0.8,
        downsample_max=1.2,
        mixup_p=0.3,
        mixup_alpha=0.4,
        gene_scale_p=0.2,
        incl_x=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.downsample_min = downsample_min
        self.downsample_max = downsample_max
        self.mixup_p = mixup_p
        self.mixup_alpha = mixup_alpha
        self.gene_scale_p = gene_scale_p
        self.n_augmentations = n_augmentations
        self.incl_x = incl_x

    # ---------------------------------------------------

    def _cross_dataset_mixup(self, x, y, ctx):
        """
        Same-class cross-dataset mixup.
        
        Returns:
            x_new   : mixed expression
            ctx_new : randomly selected context from parents
        """
        B = x.size(0)
        device = x.device

        x_new = x.clone()
        ctx_new = ctx.clone()

        # flatten ctx if shape (B,1)
        ctx_flat = ctx.view(B, -1) if ctx.dim() > 1 else ctx

        # choose which samples to mix
        mask = torch.rand(B, device=device) < self.mixup_p
        idx = torch.where(mask)[0]

        for i in idx:

            # candidates: same class, different dataset
            candidates = torch.where((y == y[i]) & (ctx_flat != ctx_flat[i]))[0]

            if len(candidates) == 0:
                continue

            # pick partner
            j = candidates[torch.randint(len(candidates), (1,), device=device)]

            # mix coefficient
            lam = torch.distributions.Beta(
                self.mixup_alpha, self.mixup_alpha
            ).sample().to(device)

            # mix expression
            x_new[i] = lam * x[i] + (1 - lam) * x[j]

            # randomly choose context from parents
            if torch.rand(1, device=device) < 0.5:
                ctx_new[i] = ctx[i]
            else:
                ctx_new[i] = ctx[j]

        return x_new, ctx_new

    # ---------------------------------------------------

    def _gene_scaling(self, x):
        """Randomly rescale subsets of genes"""
        B, D = x.shape
        device = x.device

        mask = torch.rand(D, device=device) < self.gene_scale_p
        if mask.sum() == 0:
            return x

        scale = torch.rand(mask.sum(), device=device) * 0.4 + 0.8
        x = x.clone()
        x[:, mask] *= scale

        return x

    # ---------------------------------------------------

    def forward(self, batch):

        x = batch[MODULE_KEYS.X_KEY].float()
        ctx = batch[MODULE_KEYS.BATCH_INDEX_KEY]
        y = batch[MODULE_KEYS.LABEL_KEY]

        cont_covs = batch[MODULE_KEYS.CONT_COVS_KEY]
        cat_covs = batch[MODULE_KEYS.CAT_COVS_KEY]

        B, D = x.shape
        device = x.device

        augmented = []

        if self.incl_x:
            augmented.append(x)

        # ---------------------------
        # Augmentations
        # ---------------------------
        ctxs = []
        for _ in range(self.n_augmentations):

            aug = x.clone()

            # 1) SAME-CLASS CROSS-DATASET MIXUP
            aug, mix_ctx = self._cross_dataset_mixup(aug, y, ctx)

            # 2) gene dropout
            if self.dropout > 0:
                mask = torch.rand_like(aug) > self.dropout
                aug *= mask

            # 3) smooth library scaling
            scale = torch.rand(B, 1, device=device) * (
                self.downsample_max - self.downsample_min
            ) + self.downsample_min
            aug *= scale

            # 4) gene scaling noise
            aug = self._gene_scaling(aug)

            # 5) small gaussian noise (very small!)
            aug += torch.randn_like(aug) * 0.01

            augmented.append(aug.clamp(min=0))
            ctxs.append(mix_ctx)

        # stack
        augmented_counts = torch.cat(augmented, dim=0)
        augmented_ctxs = torch.cat(ctxs, dim=0).flatten()

        repeat = len(augmented)

        def rep(t):
            return t.repeat(repeat, 1).flatten() if t is not None else None

        return {
            MODULE_KEYS.X_KEY: augmented_counts,
            MODULE_KEYS.BATCH_INDEX_KEY: augmented_ctxs,
            MODULE_KEYS.LABEL_KEY: rep(y),
            MODULE_KEYS.CONT_COVS_KEY: rep(cont_covs),
            MODULE_KEYS.CAT_COVS_KEY: rep(cat_covs),
        }
