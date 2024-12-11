from json import load
import logging
import os
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy
from graph_tool import load_graph
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from regex import D
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import graph_tool.all as gt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Any, List, Optional, Tuple
from src.utils import scale


class WeightedGATConv(GATConv):
    """Modified GAT layer to handle edge weights"""
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None:
            # Normalize edge weights
            edge_weights = edge_attr.view(-1)
            edge_weights = F.sigmoid(edge_weights)  # Ensure weights are in [0,1]
            return super().forward(x, edge_index, edge_weights)
        return super().forward(x, edge_index)

class GRNClassifierModule(pl.LightningModule):
    def __init__(self, 
                 num_node_features: int, 
                 num_classes: int,
                 hidden_dim: int = 64,
                 num_heads: int = 8,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # GAT layers with proper input/output dimensions
        self.gat1 = WeightedGATConv(
            in_channels=num_node_features, 
            out_channels=hidden_dim, 
            heads=num_heads, 
            dropout=dropout_rate,
            add_self_loops=True  # Added self-loops for better message passing
        )
        
        # Adjusted input dimension to account for concatenated heads
        self.gat2 = WeightedGATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim // 2,
            heads=1,
            dropout=dropout_rate,
            add_self_loops=True
        )
        
        self.fc1 = torch.nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc2 = torch.nn.Linear(hidden_dim // 4, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.learning_rate = learning_rate

        # Initialize metrics with proper average parameter
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        
    def forward(self, x, edge_index, edge_attr, batch):
        # First GAT layer
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view((out.shape[0], out.shape[1]))
        pred_probs = torch.softmax(out, dim=1)
        target_idx = torch.argmax(y, dim=1)

        loss = F.binary_cross_entropy_with_logits(out, y)  
        acc = self.train_accuracy(pred_probs, target_idx)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view((out.shape[0], out.shape[1]))
        
        pred_probs = torch.softmax(out, dim=1)
        target_idx = torch.argmax(y, dim=1)

        loss = F.binary_cross_entropy_with_logits(out, y)  
        acc = self.train_accuracy(pred_probs, target_idx)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


class GRNDataset:
    features = ['in_degree', 'out_degree', 'pagerank', 'betweenness', 'clustering', 'eigenvalue', 'katz']

    def __init__(self, grns: List[gt.Graph], 
                 labels: List[str],
                 expression_data: Optional[List[np.ndarray]] = None,
                 weights_key: str = 'importance',
                 scale_features: bool = True):
        """
        Initialize GRN dataset with graph-tool Graph objects
        
        Parameters:
        grns: List of graph-tool Graph objects
        labels: List of string labels
        expression_data: Optional gene expression data
        """
        self.grns = grns
        self.expression_data = expression_data
        self.weights_key = weights_key
        self.scale_features = scale_features
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Store label mapping
        self.label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_) # type: ignore
        ))

        self.geom_dataset = self.to_pytorch_geometric()
        
    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Convert string labels to one-hot encoded tensors"""
        encoded = self.label_encoder.transform(labels)
        return F.one_hot(torch.tensor(encoded), num_classes=self.num_classes)
    
    def decode_labels(self, encoded_labels: torch.Tensor) -> List[str]:
        """Convert one-hot encoded tensors back to string labels"""
        indices = torch.argmax(encoded_labels, dim=1).numpy()
        return self.label_encoder.inverse_transform(indices)

    def __len__(self):
        return len(self.geom_dataset)
        
    def extract_features(self, g: gt.Graph) -> np.ndarray:
        """Extract network features using graph-tool with edge weights"""
        weights = g.edge_properties.get(self.weights_key, None)
        # Weighted degree calculations
        in_degrees = g.get_in_degrees(g.get_vertices(), eweight=weights)
        out_degrees = g.get_out_degrees(g.get_vertices(), eweight=weights)
        
        # Weighted PageRank
        pagerank = gt.pagerank(g, weight=weights)
        pr_values = pagerank
        
        # Weighted betweenness centrality
        vertex_betweenness, _ = gt.betweenness(g, weight=weights)
        bt_values = vertex_betweenness.get_array()
        
        # Weighted clustering coefficients (using geometric mean of weights)
        clustering = gt.local_clustering(g, weight=weights)
        cl_values = clustering.get_array()
        
        # Calculate weighted eigenvalue centrality
        _, eig = gt.eigenvector(g, weight=weights)
        eig_values = eig.get_array()
        
        # Calculate weighted katz centrality
        katz = gt.katz(g, weight=weights)
        katz_values = katz.get_array()

        feature_list = [
            in_degrees,
            out_degrees,
            pr_values,
            bt_values,
            cl_values,
            eig_values,
            katz_values
        ]
        # scale each feature
        if self.scale_features:
            feature_list = [scale(f) for f in feature_list]
        
        # Combine features
        features = np.column_stack(feature_list)
        
        return features

    def to_onehot(self, label: int) -> torch.Tensor:
        """Convert integer label to one-hot encoded tensor"""
        onehot = torch.zeros(self.num_classes)
        onehot[label] = 1.0
        return onehot
    
    def to_pytorch_geometric(self) -> List[Data]:
        """Convert graph-tool Graphs to PyTorch Geometric format"""
        dataset = []
        logging.info(f'Computing metrics for GRNs')
        for idx, g in enumerate(tqdm(self.grns, desc="Processing GRNs", unit="grn", position=0, leave=True)):
            # Get edge indices
            edge_index = np.array([[int(e.source()), int(e.target())] for e in g.edges()])
            edge_index = edge_index.T  # Convert to PyTorch Geometric format
            
            # Get edge weights if they exist
            weights = g.edge_properties.get(self.weights_key, None)
            if weights is not None:
                edge_attr = np.array([weights[e] for e in g.edges()])
                edge_attr = torch.tensor(edge_attr, dtype=torch.float).reshape(-1, 1)
            else:
                edge_attr = torch.ones(edge_index.shape[1], 1)
            
            # Extract features
            features = self.extract_features(g)
            
            # Add expression data if available
            if self.expression_data is not None:
                exp_features = self.expression_data[idx]
                features = np.hstack([features, exp_features])
            
            # Convert to PyTorch tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = torch.tensor(features, dtype=torch.float)
            
            label = self.encoded_labels[idx]
            # Create one-hot encoded label
            y = F.one_hot(
                torch.tensor(label), 
                num_classes=self.num_classes
            ).float()

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, label=label)
            dataset.append(data)
        # set self to geometric dataset
        return dataset
    
    def __getitem__(self, idx: int) -> Data:
        return self.geom_dataset[idx]

class GRNDataModule(pl.LightningDataModule):
    def __init__(self, dataset: List[Data], batch_size: int = 16,
                 train_val_split: float = 0.8, num_workers: int = 1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.split = train_val_split
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        labels = [d.label for d in self.dataset]
        # Second split: train, validation
        train_idx, val_idx = train_test_split(
            range(len(self.dataset)),
            train_size=self.split,
            random_state=42,
            stratify=labels
        )
        
        self.train_dataset = [self.dataset[i] for i in train_idx]
        self.val_dataset = [self.dataset[i] for i in val_idx]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

def train_grn_classifier(grns: List[gt.Graph], 
                        labels: List[str],
                        expression_data: Optional[List[np.ndarray]] = None,
                        max_epochs: int = 100,
                        batch_size: int = 32,
                        hparams: Optional[dict[str, Any]] = None):
    """Main training function for graph-tool Graphs"""
    
    # Create dataset
    logging.info(f'Creating dataset with {len(grns)} GRNs')
    dataset = GRNDataset(grns, labels, expression_data)
    logging.info(f'Computing metrics for GRNs')
    geometric_dataset = dataset.to_pytorch_geometric()
    
    # Create data module
    data_module = GRNDataModule(geometric_dataset, batch_size=batch_size)
    
    # Calculate number of features
    num_features = geometric_dataset[0].x.shape[1]
    num_classes = len(np.unique(labels))
    
    # Create model
    model = GRNClassifierModule(num_features, num_classes, **hparams)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(monitor='val_loss')
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator='auto',
        devices=1
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    return model, trainer

def load_grns(data_dir: str, remove_batch: int = 2) -> Tuple[List[str], List[gt.Graph]]:
    ls = []
    gs = []    
    for file in os.listdir(data_dir):
        if file.endswith('gt.gz'):
            g = load_graph(os.path.join(data_dir, file))
            fn = os.path.basename(file.rstrip('gt.gz'))
            if remove_batch:
                fn = ';'.join(fn.split('_')[:-remove_batch])
            ls.append(fn)
            gs.append(g)
    if len(gs) == 0:
        raise OSError(f'{data_dir} contains no .gt.gz files to load')
    return ls, gs


def get_class_accuracies(model, test_loader, label_mapping, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model = model.to(device)
    
    # Initialize counters for each class
    class_correct = {class_name: 0 for class_name in label_mapping.keys()}
    class_total = {class_name: 0 for class_name in label_mapping.keys()}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred_probs = torch.softmax(outputs, dim=1)
            
            # Count correct predictions for each class
            for pred, target in zip(pred_probs, batch.y):
                print(target)
                class_idx = target.argmax().item()
                class_name = list(label_mapping.keys())[class_idx]
                
                class_total[class_name] += 1
                if torch.equal(pred, target):
                    class_correct[class_name] += 1
    
    # Calculate accuracy for each class
    class_accuracies = {
        class_name: class_correct[class_name] / class_total[class_name] if class_total[class_name] > 0 else 0
        for class_name in label_mapping.keys()
    }
    
    # Print results
    print("\nClass-wise Accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.4f} ({class_correct[class_name]}/{class_total[class_name]})")
    
    return class_accuracies
