import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import MODULE_KEYS, REGISTRY_KEYS



class BatchAugmentation(nn.Module):
    """Advanced count-aware augmentation for scRNA-seq"""
    
    def __init__(
        self, 
        n_augmentations: int = 3,
        dropout: float = 0.3,
        downsample_min: float = 0.7,
        downsample_max: float = 1.0,
        min_cells_per_class: int = 5,
        incl_x: bool = False,
    ):
        """
        Args:
            dropout: gene dropout probability
            downsample_min: minimum library downsampling size
            downsample_max: maximum library downsampling size
            incl_x: add unaugmented X to batch
        """
        super().__init__()
        self.dropout = dropout
        self.downsample_min = downsample_min
        self.downsample_max = downsample_max
        self.n_augmentations = n_augmentations
        self.incl_x = incl_x
        self.min_cells_per_class = min_cells_per_class
    
    def forward(
            self, 
            batch: dict[str, torch.Tensor], 
        ):
        """
        Args:
            x: (B, D) - raw counts (long tensor)
            batch_index: (B,) or (B, 1) - batch indices
            label: (B,) or (B, 1) - labels
        
        Returns:
            tuple: (augmented_counts, augmented_batch_index, augmented_label)
                - augmented_counts: (B * n_augmentations, D)
                - augmented_batch_index: (B * n_augmentations,) or (B * n_augmentations, 1)
                - augmented_label: (B * n_augmentations,) or (B * n_augmentations, 1)
        """
        # Extract original data from batch
        x: torch.Tensor = batch[MODULE_KEYS.X_KEY]
        batch_index: torch.Tensor = batch[MODULE_KEYS.BATCH_INDEX_KEY]
        label: torch.Tensor = batch[MODULE_KEYS.LABEL_KEY]
        cont_covs: torch.Tensor = batch[MODULE_KEYS.CONT_COVS_KEY]
        cat_covs: torch.Tensor = batch[MODULE_KEYS.CAT_COVS_KEY]
        
        # Get batch shapes & device
        B, D = x.shape
        device = x.device
        # Collect augmentations
        augmented = []
        
        # Ensure input is long
        x = x.long()
        # Set repeat factor
        repeat_factor = self.n_augmentations - 1
        # Add x itself
        if self.incl_x:
            augmented.append(x)
            repeat_factor += 1
            
        # ----------------------------------
        # 0. Class-mean (pseudobulk) augmentation
        # ----------------------------------
        class_means = []
        class_labels = []
        class_batches = []
        class_cont_covs = []
        class_cat_covs = []

        for c in label.unique():
            mask = (label == c).flatten()
            if mask.sum() >= self.min_cells_per_class:
                class_means.append(x[mask].float().mean(dim=0))
                class_labels.append(c)
                # Use the first batch index / covariates as representative
                class_batches.append(batch_index[mask][0])
                if cont_covs is not None:
                    class_cont_covs.append(cont_covs[mask][0])
                if cat_covs is not None:
                    class_cat_covs.append(cat_covs[mask][0])

        if len(class_means) > 0:
            class_means = torch.stack(class_means).round().clamp(min=0).long()
            augmented.append(class_means)

            class_labels = torch.stack(class_labels) if label.dim() > 0 else torch.tensor(class_labels, device=device)
            class_batches = torch.stack(class_batches)

            augmented_label_means = class_labels
            augmented_batch_means = class_batches

            augmented_cont_covs_means = (
                torch.stack(class_cont_covs) if cont_covs is not None else None
            )
            augmented_cat_covs_means = (
                torch.stack(class_cat_covs) if cat_covs is not None else None
            )
        else:
            augmented_label_means = None
            augmented_batch_means = None
            augmented_cont_covs_means = None
            augmented_cat_covs_means = None
        
        # 1. Technical dropout
        if self.n_augmentations > 1:
            dropout_mask = torch.bernoulli(
                torch.full((B, D), 1 - self.dropout, device=device)
            )
            aug1 = (x * dropout_mask).long()
            augmented.append(aug1)
        
        # 2. Library size downsampling (vectorized)
        if self.n_augmentations > 2:
            downsample_rates = torch.rand(B, 1, device=device) * (
                self.downsample_max - self.downsample_min
            ) + self.downsample_min
            
            # Vectorized binomial sampling
            aug2 = torch.binomial(
                x.float(), 
                downsample_rates.expand_as(x)
            ).long()
            augmented.append(aug2)
        
        # 3. Poisson noise
        # Add small constant to avoid issues with zero counts
        if self.n_augmentations > 3:
            lambda_param = x.float() + 0.1
            aug3 = torch.poisson(lambda_param).long()
            augmented.append(aug3)
        
        # Optional 4. Negative binomial (more realistic than Poisson)
        if self.n_augmentations > 4:
            # NB(r, p) where mean = r(1-p)/p
            # For mean = lambda, r = 10, p = 10/(10+lambda)
            r = 10.0
            mean = x.float() + 0.1
            p = r / (r + mean)
            
            # Sample from NegativeBinomial
            aug4 = torch.distributions.NegativeBinomial(
                total_count=r, 
                probs=p
            ).sample().long()
            augmented.append(aug4)
        
        # Stack augmented counts
        augmented_counts = torch.cat(augmented, dim=0)  
        # Ensure non-negative integers
        augmented_counts = torch.clamp(augmented_counts, min=0).long()
        
        # Prepare labels / batch indices
        labels_list = []
        batches_list = []
        cont_covs_list = []
        cat_covs_list = []

        # Repeat per-cell metadata
        if repeat_factor > 0:
            if label.dim() == 1:
                labels_list.append(label.repeat(repeat_factor))
            else:
                labels_list.append(label.repeat(repeat_factor, 1))

            if batch_index.dim() == 1:
                batches_list.append(batch_index.repeat(repeat_factor))
            else:
                batches_list.append(batch_index.repeat(repeat_factor, 1))

            if cont_covs is not None:
                cont_covs_list.append(cont_covs.repeat(repeat_factor, 1))
            if cat_covs is not None:
                cat_covs_list.append(cat_covs.repeat(repeat_factor, 1))
        
        # Add mean labels
        if augmented_label_means is not None:
            labels_list.append(augmented_label_means)
            batches_list.append(augmented_batch_means)
            if cont_covs is not None:
                cont_covs_list.append(augmented_cont_covs_means)
            if cat_covs is not None:
                cat_covs_list.append(augmented_cat_covs_means)
        # Collapse label lists
        augmented_label = torch.cat(labels_list, dim=0)
        augmented_batch_index = torch.cat(batches_list, dim=0)
        augmented_cont_covs = torch.cat(cont_covs_list, dim=0) if cont_covs is not None else None
        augmented_cat_covs = torch.cat(cat_covs_list, dim=0) if cat_covs is not None else None
        
        # Return augmented batch
        return {
            MODULE_KEYS.X_KEY: augmented_counts,
            MODULE_KEYS.BATCH_INDEX_KEY: augmented_batch_index,
            MODULE_KEYS.LABEL_KEY: augmented_label,
            MODULE_KEYS.CONT_COVS_KEY: augmented_cont_covs,
            MODULE_KEYS.CAT_COVS_KEY: augmented_cat_covs
        }
