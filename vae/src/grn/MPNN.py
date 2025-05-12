import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torchmetrics.classification import MulticlassAccuracy


class MPNNClassifier(pl.LightningModule):
    """Message Passing Neural Network with edge features"""
    def __init__(self,
                 num_node_features: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Assuming scalar edge weights
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Message passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
        # Global pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Message passing
        for conv in self.convs:
            x_res = x  # Residual connection
            x = conv(x, edge_index)
            x = x + x_res  # Add residual
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        
        # Multiple pooling and concatenation
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_max, x_mean, x_sum], dim=1)
        
        # Final classification
        x = self.pool(x)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        
        # Reshape target and apply label smoothing
        batch_size = out.shape[0]
        num_classes = out.shape[1]
        target = batch.y.view(batch_size, num_classes)
        
        # Apply mixup augmentation during training
        if self.training:
            lam = np.random.beta(0.8, 0.8)
            idx = torch.randperm(batch_size)
            mixed_target = lam * target + (1 - lam) * target[idx]
            loss = F.binary_cross_entropy_with_logits(out, mixed_target)
        else:
            loss = F.binary_cross_entropy_with_logits(out, target)
        
        # Calculate multiple metrics
        pred_probs = torch.sigmoid(out)
        acc = self.train_acc(pred_probs, target.argmax(dim=1))
        auroc = self.train_auroc(pred_probs, target.argmax(dim=1))
        f1 = self.train_f1(pred_probs, target.argmax(dim=1))
        
        # Log all metrics
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        self.log('train_auroc', auroc, prog_bar=True, on_epoch=True)
        self.log('train_f1', f1, prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        
        batch_size = out.shape[0]
        num_classes = out.shape[1]
        target = batch.y.view(batch_size, num_classes)
        
        
        loss = F.binary_cross_entropy_with_logits(out, target.float())
        
        # Calculate multiple metrics
        pred_probs = torch.sigmoid(out)
        acc = self.val_acc(pred_probs, target.argmax(dim=1))
        auroc = self.val_auroc(pred_probs, target.argmax(dim=1))
        f1 = self.val_f1(pred_probs, target.argmax(dim=1))
        
        # Log all metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.log('val_auroc', auroc, prog_bar=True, on_epoch=True)
        self.log('val_f1', f1, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.01)
        return optimizer
