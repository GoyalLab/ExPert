import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torchmetrics.classification import MulticlassAccuracy

class GraphTransformerClassifier(pl.LightningModule):
    def __init__(self, 
                 num_node_features: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 pooling: str = 'concat'):
        super().__init__()
        self.save_hyperparameters()
        
        # Edge embedding layer
        self.edge_embedding = nn.Linear(1, hidden_dim * heads)
        
        # Multiple transformer layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(TransformerConv(
            num_node_features, 
            hidden_dim, 
            heads=heads, 
            dropout=dropout,
            edge_dim=hidden_dim * heads
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                hidden_dim * heads,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=hidden_dim * heads
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
            
        # Last layer
        self.convs.append(TransformerConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            edge_dim=hidden_dim * heads
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling strategy
        self.pooling = pooling
        pool_dim = hidden_dim * 3 if pooling == 'concat' else hidden_dim
        
        # MLP with sigmoid activation for multi-label output
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Process edge attributes
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = self.edge_embedding(edge_attr)
        
        # Initial embeddings 
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        
        # Multiple pooling strategies
        if self.pooling == 'concat':
            x_max = global_max_pool(x, batch)
            x_mean = global_mean_pool(x, batch)
            x_add = global_add_pool(x, batch)
            x = torch.cat([x_max, x_mean, x_add], dim=1)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
            
        # Final classification
        x = self.mlp(x)
        return x
    
    def training_step(self, batch, batch_idx):
        # Get model predictions
        out = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        
        # Reshape target to match output shape
        batch_size = out.shape[0]
        num_classes = out.shape[1]
        target = batch.y.view(batch_size, num_classes)
    
        # Calculate accuracy using predicted class
        pred_probs = torch.sigmoid(out)  # Use sigmoid for binary predictions

        # Binary cross entropy for multi-label classification
        loss = F.binary_cross_entropy_with_logits(out, target.float())
        
        # Calculate accuracy using predicted class
        pred_probs = torch.sigmoid(out)  # Use sigmoid for binary predictions
        acc = self.train_acc(pred_probs, target.argmax(dim=1))
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get model predictions
        out = self(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        
        # Reshape target to match output shape
        batch_size = out.shape[0]
        num_classes = out.shape[1]
        target = batch.y.view(batch_size, num_classes)

        # Calculate accuracy using predicted class
        pred_probs = torch.sigmoid(out)  # Use sigmoid for binary predictions
        # Binary cross entropy for multi-label classification
        loss = F.binary_cross_entropy_with_logits(out, target.float())
        # Determine accuracy
        acc = self.val_acc(pred_probs, target.argmax(dim=1))
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=self.hparams.learning_rate / 100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }