import torch
import torch.nn as nn



class BatchAugmentation(nn.Module):
    """Advanced count-aware augmentation for scRNA-seq"""
    
    def __init__(self, 
                 n_augmentations: int = 4,
                 dropout: float = 0.1,
                 downsample_range: tuple = (0.7, 1.0),
                 swap_prob: float = 0.05,
                 incl_x: bool = True):
        """
        Args:
            dropout: gene dropout probability
            downsample_range: library size downsampling range
            swap_prob: probability of swapping counts between similar genes
        """
        super().__init__()
        self.dropout = dropout
        self.downsample_min, self.downsample_max = downsample_range
        self.swap_prob = swap_prob
        self.n_augmentations = n_augmentations
        self.incl_x = incl_x
    
    def forward(
            self, 
            x: torch.Tensor, 
            batch_index: torch.Tensor,
            label: torch.Tensor,
            cont_covs: torch.Tensor | None = None,
            cat_covs: torch.Tensor | None = None,
            **kwargs
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
        B, D = x.shape
        device = x.device
        augmented = []
        
        # Ensure input is long
        x = x.long()
        
        # Add x itself
        if self.incl_x:
            augmented.append(x)
        
        # 1. Technical dropout
        dropout_mask = torch.bernoulli(
            torch.full((B, D), 1 - self.dropout, device=device)
        )
        aug1 = (x * dropout_mask).long()
        augmented.append(aug1)
        
        # 2. Library size downsampling (vectorized)
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
        lambda_param = x.float() + 0.5
        aug3 = torch.poisson(lambda_param).long()
        augmented.append(aug3)
        
        # Optional 4. Negative binomial (more realistic than Poisson)
        if self.n_augmentations > 3:
            # NB(r, p) where mean = r(1-p)/p
            # For mean = lambda, r = 10, p = 10/(10+lambda)
            r = 10.0
            mean = x.float() + 0.5
            p = r / (r + mean)
            
            # Sample from NegativeBinomial
            aug4 = torch.distributions.NegativeBinomial(
                total_count=r, 
                probs=p
            ).sample().long()
            augmented.append(aug4)
        
        # Stack augmented counts
        augmented_counts = torch.cat(augmented[:self.n_augmentations], dim=0)
        
        # Ensure non-negative integers
        augmented_counts = torch.clamp(augmented_counts, min=0).long()
        
        # Repeat batch_index and label n_augmentations times
        # Handle both (B,) and (B, 1) shapes
        if batch_index.dim() == 1:
            augmented_batch_index = batch_index.repeat(self.n_augmentations)
        else:
            augmented_batch_index = batch_index.repeat(self.n_augmentations, 1)
        
        if label.dim() == 1:
            augmented_label = label.repeat(self.n_augmentations)
        else:
            augmented_label = label.repeat(self.n_augmentations, 1)
        
        if cont_covs is None:
            augmented_cont_covs = None
        elif cont_covs.dim() == 1:
            augmented_cont_covs = cont_covs.repeat(self.n_augmentations)
        else:
            augmented_cont_covs = cont_covs.repeat(self.n_augmentations, 1)
        
        if cat_covs is None:
            augmented_cat_covs = None
        elif cat_covs.dim() == 1:
            augmented_cat_covs = cat_covs.repeat(self.n_augmentations)
        else:
            augmented_cat_covs = cat_covs.repeat(self.n_augmentations, 1)
        
        # Return augmented batch
        return augmented_counts, augmented_batch_index, augmented_label, augmented_cont_covs, augmented_cat_covs
