import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Optional
import anndata as ad
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import AnnDataModule

from pytorch_lightning.callbacks import ModelCheckpoint


class Encoder(nn.Module):
    """VAE Encoder Network"""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    """VAE Decoder Network"""
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class Classifier(nn.Module):
    """Classification head for latent space"""
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        # layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

class VAE(pl.LightningModule):
    """Semi-supervised VAE with detailed class-wise metrics"""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 32,
        classifier_hidden_dims: List[int] = [64],
        learning_rate: float = 1e-3,
        beta: float = 1.0,  # KL divergence weight
        alpha: float = 1.0,  # Classification loss weight,
        scheduler = None,
        monitor_loss = None,
        dropout_rate: float = 0.1,
        dynamic_alpha: bool = False     # Dynically adjust alpha to first learn latent space before classifying
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim,
            dropout_rate=dropout_rate
        )
        
        self.classifier = Classifier(
            latent_dim=latent_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Initialize metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        
        # Initialize per-class accuracy metrics
        self.train_class_acc = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.val_class_acc = MulticlassAccuracy(num_classes=num_classes, average=None)
        
        # Initialize confusion matrix
        self.train_confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        
        self.beta = beta
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.dynamic_alpha = dynamic_alpha
        self.scheduler = scheduler
        self.monitor_loss = monitor_loss
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        classification = self.classifier(z)
        return reconstruction, classification, mu, log_var
    
    def _get_reconstruction_loss(self, x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss"""
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        return recon_loss
    
    def _get_kl_divergence_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL divergence loss"""
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss
    
    def _get_classification_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Classification loss"""
        return F.binary_cross_entropy_with_logits(pred, target)

    def _convert_onehot_to_indices(self, onehot: torch.Tensor) -> torch.Tensor:
        """Convert one-hot encoded labels to class indices"""
        return torch.argmax(onehot, dim=1)
    
    def _log_class_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                          class_acc_metric, confusion_metric, prefix: str):
        """Log class-wise metrics"""
        # Convert target from one-hot to indices
        target_indices = self._convert_onehot_to_indices(target)
        
        # Get probabilities from logits
        pred_probs = torch.softmax(pred, dim=1)
        
        # Update and log per-class accuracies
        class_accuracies = class_acc_metric(pred_probs, target_indices)
        for i, acc in enumerate(class_accuracies):
            self.log(f'{prefix}_class_{i}_acc', acc, prog_bar=False)
            
        # Update confusion matrix
        confusion_matrix = confusion_metric(pred_probs, target_indices)
        
        # Calculate and log precision and recall for each class
        confusion_matrix = confusion_matrix.cpu().numpy()
        true_positives = np.diag(confusion_matrix)
        false_positives = np.sum(confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(confusion_matrix, axis=1) - true_positives
        
        # Precision and recall for each class
        epsilon = 1e-7  # To avoid division by zero
        for i in range(self.num_classes):
            precision = true_positives[i] / (true_positives[i] + false_positives[i] + epsilon)
            recall = true_positives[i] / (true_positives[i] + false_negatives[i] + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            self.log(f'{prefix}_class_{i}_precision', precision, prog_bar=False)
            self.log(f'{prefix}_class_{i}_recall', recall, prog_bar=False)
            self.log(f'{prefix}_class_{i}_f1', f1, prog_bar=False)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        reconstruction, classification, mu, log_var = self(x)
        
        # Calculate losses
        recon_loss = self._get_reconstruction_loss(x, reconstruction)
        kl_loss = self._get_kl_divergence_loss(mu, log_var)
        class_loss = self._get_classification_loss(classification, y)
        
        # Convert predictions to probabilities with softmax
        pred_probs = torch.softmax(classification, dim=1)
        # Convert target from one-hot to indices
        target_indices = self._convert_onehot_to_indices(y)
        # Calculate accuracy
        accuracy = self.train_acc(pred_probs, target_indices)
        
        # Log class-wise metrics
        self._log_class_metrics(classification, y, self.train_class_acc, 
                              self.train_confmat, 'train')
        
        # Adjust alpha dynically if option is enabled and reconstruction is already good
        if self.dynamic_alpha and recon_loss < 0.1:
            # switch from reconstructing to classifying
            alpha = 10
        else:
            alpha = self.alpha
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + alpha * class_loss
        
        # Log overall metrics
        self.log_dict({
            'train_loss': total_loss,
            'train_recon_loss': recon_loss,
            'train_kl_loss': kl_loss,
            'train_class_loss': class_loss,
            'train_accuracy': accuracy
        }, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        reconstruction, classification, mu, log_var = self(x)
        
        # Calculate losses
        recon_loss = self._get_reconstruction_loss(x, reconstruction)
        kl_loss = self._get_kl_divergence_loss(mu, log_var)
        class_loss = self._get_classification_loss(classification, y)
        
        # Convert predictions to probabilities with softmax
        pred_probs = torch.softmax(classification, dim=1)
        # Convert target from one-hot to indices
        target_indices = self._convert_onehot_to_indices(y)
        # Overall accuracy
        accuracy = self.val_acc(pred_probs, target_indices)
        
        # Log class-wise metrics
        self._log_class_metrics(classification, y, self.val_class_acc, 
                              self.val_confmat, 'val')
        
        # Adjust alpha dynically if option is enabled
        if self.dynamic_alpha and recon_loss < 0.1:
            # switch from reconstructing to classifying
            alpha = 10
        else:
            alpha = self.alpha
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + alpha * class_loss
        
        # Log overall metrics
        self.log_dict({
            'val_loss': total_loss,
            'val_recon_loss': recon_loss,
            'val_kl_loss': kl_loss,
            'val_class_loss': class_loss,
            'val_accuracy': accuracy
        }, prog_bar=True)
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Log confusion matrix at the end of each validation epoch"""
        confusion_matrix = self.val_confmat.compute()
        self.val_confmat.reset()
        
        # Log confusion matrix as a heatmap if using tensorboard
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(confusion_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
            plt.title('Validation Confusion Matrix')
            plt.close()
            
            # Log figure to tensorboard
            self.logger.experiment.add_figure(
                'validation_confusion_matrix',
                fig,
                global_step=self.current_epoch
            )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # set up scheduler if provided
        if self.scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler(optimizer),
                    "monitor": self.monitor_loss,
                },
            }
        else:
            return optimizer
        
    def get_latent_representation(self, 
                                dataloader: DataLoader,
                                return_labels: bool = True,
                                sample: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract latent space representations for the entire dataset.
        
        Args:
            dataloader: DataLoader containing the data
            return_labels: Whether to return corresponding labels
            sample: If True, return sampled latent vectors; if False, return mean (mu)
        
        Returns:
            latents: Array of shape (n_cells, latent_dim)
            labels: Array of shape (n_cells,) if return_labels=True
        """
        self.eval()
        latents = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if return_labels:
                    x, y = batch
                    labels.append(y)
                else:
                    x = batch
                
                # Move to same device as model
                x = x.to(self.device)
                
                # Get latent representations
                mu, log_var = self.encoder(x)
                
                if sample:
                    # Sample from the latent space
                    z = self.reparameterize(mu, log_var)
                    latents.append(z.cpu().numpy())
                else:
                    # Use mean of latent space
                    latents.append(mu.cpu().numpy())
        
        latents = np.vstack(latents)
        
        if return_labels:
            labels = np.concatenate(labels)
            return latents, labels
        return latents

    def get_latent_statistics(self, 
                            dataloader: DataLoader,
                            return_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Extract both mean and variance of latent space representations.
        
        Returns:
            mu: Mean of latent space (n_cells, latent_dim)
            log_var: Log variance of latent space (n_cells, latent_dim)
            labels: Labels if return_labels=True (n_cells,)
        """
        self.eval()
        mus = []
        log_vars = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if return_labels:
                    x, y = batch
                    labels.append(y)
                else:
                    x = batch
                
                x = x.to(self.device)
                mu, log_var = self.encoder(x)
                
                mus.append(mu.cpu().numpy())
                log_vars.append(log_var.cpu().numpy())
        
        mus = np.vstack(mus)
        log_vars = np.vstack(log_vars)
        
        if return_labels:
            labels = np.concatenate(labels)
            return mus, log_vars, labels
        return mus, log_vars

def save_model(model: pl.LightningModule, 
               data_module: pl.LightningDataModule,
               save_dir: str = 'saved_models',
               model_name: str = 'vae_model'):
    """
    Save the trained model and associated metadata
    
    Args:
        model: Trained VAE model
        data_module: DataModule containing label encoder and other metadata
        save_dir: Directory to save the model
        model_name: Name of the model file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, f"{model_name}.ckpt")
    trainer = pl.Trainer()
    trainer.save_checkpoint(model_path)
    
    # Save label encoder and other metadata
    metadata = {
        'label_encoder': data_module.label_encoder,
        'label_decoder': data_module.label_decoder,
        'num_classes': data_module.num_classes,
        'input_dim': model.encoder.encoder[0].in_features,
        'latent_dim': model.encoder.fc_mu.out_features,
        'hidden_dims': [model.encoder.encoder[0].out_features, 
                       model.encoder.encoder[4].out_features],
    }
    
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.pt")
    torch.save(metadata, metadata_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

def load_model(save_dir: str = 'saved_models',
               model_name: str = 'vae_model',
               map_location: str = 'cpu') -> Tuple[VAE, dict]:
    """
    Load a saved model and its metadata
    
    Args:
        save_dir: Directory where model is saved
        model_name: Name of the model file
        map_location: Device to load the model to ('cpu' or 'cuda')
        
    Returns:
        model: Loaded VAE model
        metadata: Dictionary containing model metadata
    """
    # Load metadata
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.pt")
    metadata = torch.load(metadata_path)
    
    # Initialize model with saved parameters
    model = VAE(
        input_dim=metadata['input_dim'],
        num_classes=metadata['num_classes'],
        hidden_dims=metadata['hidden_dims'],
        latent_dim=metadata['latent_dim']
    )
    
    # Load model state
    model_path = os.path.join(save_dir, f"{model_name}.ckpt")
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict['state_dict'])
    
    print(f"Model loaded from: {model_path}")
    print(f"Metadata loaded from: {metadata_path}")
    
    return model, metadata

def save_best_model_during_training(save_dir: str = 'saved_models',
                                  model_name: str = 'vae_model',
                                  monitor: str = 'val_loss',
                                  mode: str = 'min') -> ModelCheckpoint:
    """
    Create a callback to save the best model during training
    
    Args:
        save_dir: Directory to save checkpoints
        model_name: Base name for checkpoint files
        monitor: Metric to monitor
        mode: 'min' or 'max' for the monitored metric
        
    Returns:
        ModelCheckpoint callback
    """
    return ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{model_name}_best",
        monitor=monitor,
        mode=mode,
        save_last=True,
        save_weights_only=False,
        verbose=True
    )

def extract_and_plot_latent_space(model: VAE, 
                                data_module: AnnDataModule,
                                save_to_adata: bool = True,
                                plot: bool = True) -> Optional[ad.AnnData]:
    """
    Extract latent representations and create visualizations.
    Optionally save results back to AnnData object.
    """
    import scanpy as sc
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    # Get latent representations
    latents, labels = model.get_latent_representation(
        data_module.train_dataloader(),
        return_labels=True,
        sample=False  # Use mean of latent space
    )
    
    # Convert numeric labels back to original names
    label_names = [data_module.get_label_name(label) for label in labels]
    
    if save_to_adata:
        # Create new AnnData object with latent representations
        adata_latent = ad.AnnData(latents)
        adata_latent.obs[data_module.label_col] = label_names
        
        # Run PCA and UMAP
        sc.pp.pca(adata_latent)
        sc.pp.neighbors(adata_latent)
        sc.tl.umap(adata_latent)
        
        if plot:
            # Create visualization plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # PCA plot
            sc.pl.pca(adata_latent, color=data_module.label_col, ax=ax1, show=False)
            ax1.set_title('PCA of Latent Space')
            
            # UMAP plot
            sc.pl.umap(adata_latent, color=data_module.label_col, ax=ax2, show=False)
            ax2.set_title('UMAP of Latent Space')
            
            plt.tight_layout()
            plt.show()
            
            # Plot latent space statistics
            mus, log_vars, _ = model.get_latent_statistics(
                data_module.train_dataloader(),
                return_labels=True
            )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot distribution of means
            sns.boxplot(data=pd.DataFrame(mus), ax=ax1)
            ax1.set_title('Distribution of Latent Means')
            ax1.set_xlabel('Latent Dimension')
            ax1.set_ylabel('Value')
            
            # Plot distribution of variances
            sns.boxplot(data=pd.DataFrame(np.exp(log_vars)), ax=ax2)
            ax2.set_title('Distribution of Latent Variances')
            ax2.set_xlabel('Latent Dimension')
            ax2.set_ylabel('Value')
            
            plt.tight_layout()
            plt.show()
        
        return adata_latent
    
    return None
