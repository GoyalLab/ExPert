from src.vae import VAE, Encoder, Decoder, Classifier
from src.modules.zinb import ZINBDecoder, NoisyEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
import scanpy as sc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional
import pytorch_lightning as pl
from typing import List
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score


class MultiPropertyVAE(pl.LightningModule):
    
    """VAE for multiple cell properties with single encoder and detailed metrics"""
    def __init__(
        self,
        input_dim: int,
        cell_type_classes: int,
        pert_type_classes: int,
        pert_classes: int,
        n_datasets: int = 0,                        # If 0, won't use dataset information
        n_batches: int = 0,                         # If 0, do not use batch information
        hidden_dims: List[int] = [512, 256],
        encoder_latent_dim: int = 256,
        cell_type_latent_dim: int = 16,
        pert_type_latent_dim: int = 8,
        pert_latent_dim: int = 8,
        shared_latent_dim: int = 32,
        classifier_hidden_dims: List[int] = [64],
        learning_rate: float = 1e-3,
        beta: float = 1.0,  # KL divergence weight
        alpha: float = 1.0,  # Classification loss weight
        cell_type_weight: float = 1.0,
        perturbation_type_weight: float = 1.0,
        perturbation_weight: float = 1.0,
        scheduler=None,
        monitor_loss=None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.save_hyperparameters()

        # add dataset information if it is provided
        self.use_datasets = False
        if n_datasets > 1:
            self.use_datasets = True
            input_dim += n_datasets
        # add batch information if it is provided
        self.use_batches = False
        if n_batches > 1:
            self.use_batches = True
            input_dim += n_batches
        
        # Initialize shared encoder
        self.encoder = NoisyEncoder(
            input_dim=input_dim,
            latent_dim=encoder_latent_dim
        )
        
        # Separate projection layers for each latent space
        self.fc_cell_mu = nn.Linear(encoder_latent_dim, cell_type_latent_dim)
        self.fc_cell_var = nn.Linear(encoder_latent_dim, cell_type_latent_dim)
        
        self.fc_pert_type_mu = nn.Linear(encoder_latent_dim, pert_type_latent_dim)
        self.fc_pert_type_var = nn.Linear(encoder_latent_dim, pert_type_latent_dim)
        
        self.fc_pert_mu = nn.Linear(encoder_latent_dim, pert_latent_dim)
        self.fc_pert_var = nn.Linear(encoder_latent_dim, pert_latent_dim)
        
        self.fc_shared_mu = nn.Linear(encoder_latent_dim, shared_latent_dim)
        self.fc_shared_var = nn.Linear(encoder_latent_dim, shared_latent_dim)
        
        # Total latent dimension is sum of all subspaces
        total_latent_dim = (cell_type_latent_dim + pert_type_latent_dim + 
                           pert_latent_dim + shared_latent_dim)
        
        # Initialize decoder
        self.decoder = ZINBDecoder(
            latent_dim=total_latent_dim,
            output_dim=input_dim
        )
        
        # Initialize separate classifiers
        self.cell_type_classifier = Classifier(
            latent_dim=cell_type_latent_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=cell_type_classes,
            dropout_rate=dropout_rate
        )
        
        self.pert_type_classifier = Classifier(
            latent_dim=pert_type_latent_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=pert_type_classes,
            dropout_rate=dropout_rate
        )
        
        self.pert_classifier = Classifier(
            latent_dim=pert_latent_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=pert_classes,
            dropout_rate=dropout_rate
        )
        
        # Initialize metrics for each property
        use_cell_type = cell_type_classes > 1
        if use_cell_type:
            self.train_acc_cell = MulticlassAccuracy(num_classes=cell_type_classes, average='micro')
            self.val_acc_cell = MulticlassAccuracy(num_classes=cell_type_classes, average='micro')
        
        use_pert_type = pert_type_classes > 1
        if use_pert_type:
            self.train_acc_pert_type = MulticlassAccuracy(num_classes=pert_type_classes, average='micro')
            self.val_acc_pert_type = MulticlassAccuracy(num_classes=pert_type_classes, average='micro')
        
        use_pert = pert_classes > 1
        if use_pert:
            self.train_acc_pert = MulticlassAccuracy(num_classes=pert_classes, average='micro')
            self.val_acc_pert = MulticlassAccuracy(num_classes=pert_classes, average='micro')

        self.val_acc = MulticlassAccuracy(num_classes=pert_classes, average=None)

        # Initialize confusion matrix
        self.train_confmat = MulticlassConfusionMatrix(num_classes=pert_classes)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=pert_classes)
        
        # Save dimensions and parameters
        self.use_cell_type = use_cell_type
        self.use_pert_type = use_pert_type
        self.use_pert = use_pert
        self.cell_type_latent_dim = cell_type_latent_dim
        self.pert_type_latent_dim = pert_type_latent_dim
        self.pert_latent_dim = pert_latent_dim
        self.shared_latent_dim = shared_latent_dim
        self.beta = beta
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.monitor_loss = monitor_loss
        self.pert_classes = pert_classes
        self.cell_type_weight = cell_type_weight
        self.perturbation_type_weight = perturbation_type_weight
        self.perturbation_weight = perturbation_weight

    def compute_metrics(self, y_true, y_pred):
        """
        Compute precision, recall, and AUC for multi-class classification.
        y_true: true labels
        y_pred: predicted labels
        """
        y_true, y_pred = y_true.cpu(), y_pred.cpu()
        # Precision (macro average for multi-class)

        precision = precision_score(y_true, y_pred, average='micro', labels=range(self.pert_classes))
        
        # Recall (macro average for multi-class)
        recall = recall_score(y_true, y_pred, average='micro', labels=range(self.pert_classes))
        
        # AUC (One-vs-rest, macro average for multi-class)
        # We use predict_proba to get the class probabilities (not hard predictions)
        # Assumes that y_pred is the predicted probabilities, not class labels
        auc = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')

        return precision, recall, auc
        
    def encode(self, x: torch.Tensor):
        # Shared encoding
        h = self.encoder(x)[0]  # Get only the output, ignore log_var from encoder
        
        # Project to separate latent spaces
        cell_mu = self.fc_cell_mu(h)
        cell_log_var = self.fc_cell_var(h)
        
        pert_type_mu = self.fc_pert_type_mu(h)
        pert_type_log_var = self.fc_pert_type_var(h)
        
        pert_mu = self.fc_pert_mu(h)
        pert_log_var = self.fc_pert_var(h)
        
        shared_mu = self.fc_shared_mu(h)
        shared_log_var = self.fc_shared_var(h)
        
        return (cell_mu, cell_log_var, 
                pert_type_mu, pert_type_log_var,
                pert_mu, pert_log_var,
                shared_mu, shared_log_var)
    
    def forward(self, x: torch.Tensor):
        # Encode
        (cell_mu, cell_log_var, 
         pert_type_mu, pert_type_log_var,
         pert_mu, pert_log_var,
         shared_mu, shared_log_var) = self.encode(x)
        
        # Reparameterize each subspace
        z_cell = self.reparameterize(cell_mu, cell_log_var)
        z_pert_type = self.reparameterize(pert_type_mu, pert_type_log_var)
        z_pert = self.reparameterize(pert_mu, pert_log_var)
        z_shared = self.reparameterize(shared_mu, shared_log_var)
        
        # Concatenate all latent vectors
        z = torch.cat([z_cell, z_pert_type, z_pert, z_shared], dim=1)
        
        # Decode
        reconstruction = self.decoder(z)
        
        # Classify
        cell_type_pred = self.cell_type_classifier(z_cell)
        pert_type_pred = self.pert_type_classifier(z_pert_type)
        pert_pred = self.pert_classifier(z_pert)
        
        return (reconstruction, 
                cell_type_pred, pert_type_pred, pert_pred,
                cell_mu, cell_log_var, 
                pert_type_mu, pert_type_log_var,
                pert_mu, pert_log_var,
                shared_mu, shared_log_var)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x = batch['X']
        y = batch['Y']
        y_cell = y.get('cell_type')
        y_pert_type = y.get('pert_type')
        y_pert = y.get('pert')
        dataset = y.get('dataset')

        if self.use_datasets:
            x = torch.cat([x, dataset], dim=-1)
        
        (reconstruction, cell_type_pred, pert_type_pred, pert_pred,
         cell_mu, cell_log_var, pert_type_mu, pert_type_log_var,
         pert_mu, pert_log_var, shared_mu, shared_log_var) = self(x)
        
        # Calculate losses
        recon_loss = self._get_reconstruction_loss(x, reconstruction)
        
        # KL losses for each subspace
        kl_loss_cell = self._get_kl_divergence_loss(cell_mu, cell_log_var)
        kl_loss_pert_type = self._get_kl_divergence_loss(pert_type_mu, pert_type_log_var)
        kl_loss_pert = self._get_kl_divergence_loss(pert_mu, pert_log_var)
        kl_loss_shared = self._get_kl_divergence_loss(shared_mu, shared_log_var)
        kl_loss = kl_loss_cell + kl_loss_pert_type + kl_loss_pert + kl_loss_shared
        
        # Classification losses
        cell_type_loss = self._get_classification_loss(cell_type_pred, y_cell)
        pert_type_loss = self._get_classification_loss(pert_type_pred, y_pert_type)
        pert_loss = self._get_classification_loss(pert_pred, y_pert)
        class_loss = self.cell_type_weight * cell_type_loss + self.perturbation_type_weight * pert_type_loss + self.perturbation_weight * pert_loss
        
        # Calculate accuracies
        cell_acc = self.train_acc_cell(torch.softmax(cell_type_pred, dim=1), self._convert_onehot_to_indices(y_cell)) if self.use_cell_type else 1.0
        pert_type_acc = self.train_acc_pert_type(torch.softmax(pert_type_pred, dim=1), self._convert_onehot_to_indices(y_pert_type)) if self.use_pert_type else 1.0
        pert_acc = self.train_acc_pert(torch.softmax(pert_pred, dim=1), self._convert_onehot_to_indices(y_pert)) if self.use_pert else 1.0
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.alpha * class_loss

        # Log class-wise metrics
        self._log_class_metrics(pert_pred, y_pert, self.val_acc, 
                              self.val_confmat, 'train')
        
        # Log metrics
        self.log_dict({
            'train_loss': total_loss,
            'train_recon_loss': recon_loss,
            'train_kl_loss': kl_loss,
            'train_kl_loss_pert': kl_loss_pert,
            'train_cell_type_loss': cell_type_loss,
            'train_pert_type_loss': pert_type_loss,
            'train_pert_loss': pert_loss,
            'train_cell_type_acc': cell_acc,
            'train_pert_type_acc': pert_type_acc,
            'train_pert_acc': pert_acc,
            'train_class_loss': class_loss
        }, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        x = batch['X']
        y = batch['Y']
        y_cell = y.get('cell_type')
        y_pert_type = y.get('pert_type')
        y_pert = y.get('pert')
        dataset = y.get('dataset')

        if self.use_datasets:
            x = torch.cat([x, dataset], dim=-1)
        
        (reconstruction, cell_type_pred, pert_type_pred, pert_pred,
         cell_mu, cell_log_var, pert_type_mu, pert_type_log_var,
         pert_mu, pert_log_var, shared_mu, shared_log_var) = self(x)
        
        # Calculate losses (similar to training step)
        recon_loss = self._get_reconstruction_loss(x, reconstruction)
        
        kl_loss_cell = self._get_kl_divergence_loss(cell_mu, cell_log_var)
        kl_loss_pert_type = self._get_kl_divergence_loss(pert_type_mu, pert_type_log_var)
        kl_loss_pert = self._get_kl_divergence_loss(pert_mu, pert_log_var)
        kl_loss_shared = self._get_kl_divergence_loss(shared_mu, shared_log_var)
        kl_loss = kl_loss_cell + kl_loss_pert_type + kl_loss_pert + kl_loss_shared
        
        cell_type_loss = self._get_classification_loss(cell_type_pred, y_cell)
        pert_type_loss = self._get_classification_loss(pert_type_pred, y_pert_type)
        pert_loss = self._get_classification_loss(pert_pred, y_pert)
        class_loss = self.cell_type_weight * cell_type_loss + self.perturbation_type_weight * pert_type_loss + self.perturbation_weight * pert_loss
        
        # Calculate accuracies
        cell_acc = self.val_acc_cell(torch.softmax(cell_type_pred, dim=1), self._convert_onehot_to_indices(y_cell)) if self.use_cell_type else 0
        pert_type_acc = self.val_acc_pert_type(torch.softmax(pert_type_pred, dim=1), self._convert_onehot_to_indices(y_pert_type)) if self.use_pert_type else 0
        pert_acc = self.val_acc_pert(torch.softmax(pert_pred, dim=1), self._convert_onehot_to_indices(y_pert)) if self.use_pert else 0
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.alpha * class_loss

        # Log class-wise metrics
        self._log_class_metrics(pert_pred, y_pert, self.val_acc, 
                              self.val_confmat, 'val')
        
        # Log metrics
        self.log_dict({
            'val_loss': total_loss,
            'val_recon_loss': recon_loss,
            'val_kl_loss': kl_loss,
            'val_kl_loss_pert': kl_loss_pert,
            'val_cell_type_loss': cell_type_loss,
            'val_pert_type_loss': pert_type_loss,
            'val_pert_loss': pert_loss,
            'val_cell_type_acc': cell_acc,
            'val_pert_type_acc': pert_type_acc,
            'val_pert_acc': pert_acc,
            'val_class_loss': class_loss
        }, prog_bar=True)
        
        return total_loss
    
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
        p, r, f = [], [], []
        for i in range(self.pert_classes):
            precision = true_positives[i] / (true_positives[i] + false_positives[i] + epsilon)
            recall = true_positives[i] / (true_positives[i] + false_negatives[i] + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            p.append(precision)
            r.append(recall)
            f.append(f1)
        self.log(f'{prefix}_precision', np.mean(p), prog_bar=True)
        self.log(f'{prefix}_recall', np.mean(r), prog_bar=True)
        self.log(f'{prefix}_f1', np.mean(f1), prog_bar=True)
    
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
        labels = {
            'cell_type': [],
            'perturbation_type': [],
            'perturbation': [],
            'dataset': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                if return_labels:
                    x, y = batch['X'], batch['Y']
                    labels['cell_type'].append(y['cell_type'])
                    labels['perturbation_type'].append(y['pert_type'])
                    labels['perturbation'].append(y['pert'])
                    labels['dataset'].append(y.get('dataset'))
                else:
                    x = batch['X']
                if self.use_datasets:
                    x = torch.cat([x, batch['Y']['dataset']], dim=-1)
                
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
            # labels = np.concatenate(labels)
            return latents, labels
        return latents


    def plot_latent_space(self, latents, labels, dataloader, label_col):
        decoder = dataloader.dataset.label_decoder
        # convert one-hot to label indices
        labels = torch.argmax(torch.Tensor(labels), dim=1)
        # Convert numeric labels back to original names
        label_names = [decoder.get(int(label)) for label in labels]
        
        # Create AnnData object
        adata_latent = ad.AnnData(latents)
        adata_latent.obs[label_col] = label_names
        adata_latent.obs[['perturbation', 'cell_type', 'method']] = adata_latent.obs['exact_perturbation'].str.split(';', expand=True)
        # Compute UMAP
        sc.pp.neighbors(adata_latent, use_rep='X')
        sc.tl.umap(adata_latent)
        return adata_latent

    # Reuse existing helper methods, TODO: construct super class if we use multiple approaches ._.
    reparameterize = VAE.reparameterize
    _get_reconstruction_loss = VAE._get_reconstruction_loss
    _get_kl_divergence_loss = VAE._get_kl_divergence_loss
    _get_classification_loss = VAE._get_classification_loss
    _convert_onehot_to_indices = VAE._convert_onehot_to_indices
    configure_optimizers = VAE.configure_optimizers
