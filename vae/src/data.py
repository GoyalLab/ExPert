import warnings
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import anndata as ad
import scipy.sparse as sp
from typing import Optional, Tuple, Dict
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl


class BatchData():
    X: torch.Tensor
    Y: Dict[str, torch.Tensor]

    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y


class AnnDataset(Dataset):
    """Custom Dataset for loading AnnData objects with one-hot encoded labels"""
    def __init__(self, 
                 adata: ad.AnnData,
                 obs_label_col: Optional[str] = None,
                 cell_indices: Optional[np.ndarray] = None,
                 transform=None):
        """
        Args:
            adata: AnnData object
            obs_label_col: Column name in adata.obs containing labels
            cell_indices: List of cell indices to use (if None, use all cells)
            transform: Optional transform to be applied to data
        """
        self.adata = adata
        self.transform = transform
        self.obs_label_col = obs_label_col
        
        # Handle cell indices
        if cell_indices is not None:
            self.indices = cell_indices
        else:
            self.indices = np.arange(adata.n_obs)
            
        print(f"Dataset initialized with {len(self.indices)} cells")
            
        # Handle labels
        if obs_label_col is not None:
            self.labels = adata.obs[obs_label_col].values[self.indices]
            # Convert labels to numeric if they're categorical
            if self.labels.dtype == 'object' or self.labels.dtype.name == 'category':
                self.label_encoder = {label: idx for idx, label in enumerate(np.unique(self.labels))}
                self.label_decoder = {idx: label for idx, label in enumerate(np.unique(self.labels))}
                self.num_classes = len(self.label_encoder)
                self.labels = np.array([self.label_encoder[label] for label in self.labels])
                print(f"Labels encoded: {self.label_encoder}")
                print(f"Number of classes: {self.num_classes}")
        else:
            self.labels = None
            self.num_classes = None

    def __len__(self):
        return len(self.indices)

    def to_onehot(self, label: int) -> torch.Tensor:
        """Convert integer label to one-hot encoded tensor"""
        onehot = torch.zeros(self.num_classes)
        onehot[label] = 1.0
        return onehot

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the cell index
        cell_idx = self.indices[idx]
        
        # Get expression data for the cell
        expression = self.adata.X[cell_idx].toarray() if sp.issparse(self.adata.X) else self.adata.X[cell_idx]
        
        # Convert to float32 tensor
        expression = torch.FloatTensor(expression).squeeze()
        
        # Apply transform if specified
        if self.transform:
            expression = self.transform(expression)
            
        # Return with one-hot encoded labels if labels exist
        if self.labels is not None:
            label = self.labels[idx]
            return expression, self.to_onehot(label)
        return expression
    

def create_data_loader(adata,
                      batch_size=128,
                      obs_label_col=None,
                      cell_indices=None,
                      transform=None,
                      shuffle=False,
                      num_workers=4):
    """
    Creates a PyTorch DataLoader for an AnnData object
    
    Args:
        adata: AnnData object
        batch_size: Number of samples per batch
        obs_label_col: Column name in adata.obs containing labels
        cell_indices: List of cell indices to use (if None, use all cells)
        transform: Optional transform to be applied to data
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader object
    """
    dataset = AnnDataset(
        adata=adata,
        obs_label_col=obs_label_col,
        cell_indices=cell_indices,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

class AnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: ad.AnnData,
        label_col: str,
        batch_size: int = 128,
        train_val_split: float = 0.8,
        num_workers: int = 1,
        cell_indices: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.adata = adata
        self.label_col = label_col
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.cell_indices = cell_indices
        
        # Create label encoder
        self.label_encoder = {
            label: idx for idx, label in enumerate(adata.obs[label_col].unique())
        }
        self.num_classes = len(self.label_encoder)
        
    def setup(self, stage: Optional[str] = None):
        if self.cell_indices is None:
            self.cell_indices = np.arange(self.adata.n_obs)
            
        # Split indices into train and validation
        np.random.shuffle(self.cell_indices)
        split_idx = int(len(self.cell_indices) * self.train_val_split)
        self.train_indices = self.cell_indices[:split_idx]
        self.val_indices = self.cell_indices[split_idx:]
        
    def train_dataloader(self) -> DataLoader:
        return create_data_loader(
            self.adata,
            batch_size=self.batch_size,
            cell_indices=self.train_indices,
            num_workers=self.num_workers,
            obs_label_col=self.label_col
        )
    
    def val_dataloader(self) -> DataLoader:
        return create_data_loader(
            self.adata,
            batch_size=self.batch_size,
            cell_indices=self.val_indices,
            num_workers=self.num_workers,
            obs_label_col=self.label_col
        )
    

class MultiLabelAnnDataset(Dataset):
    """Custom Dataset for loading AnnData objects with multiple one-hot encoded labels"""
    
    def __init__(self, 
                 adata: ad.AnnData,
                 cell_type_col: str,
                 pert_type_col: str,
                 pert_col: str,
                 dataset_col: Optional[str] = 'dataset',
                 batch_col: Optional[str] = None,
                 cell_indices: Optional[np.ndarray] = None,
                 transform=None):
        """
        Args:
            adata: AnnData object
            cell_type_col: Column name for cell types
            pert_type_col: Column name for perturbation types
            pert_col: Column name for specific perturbations
            dataset_col: Column name for dataset index if multiple datasets have been combined
            batch_col: Column name for batch index if dataset(s) used multiple batches
            cell_indices: List of cell indices to use
            transform: Optional transform to be applied to data
        """
        self.adata = adata
        self.transform = transform
        
        # Handle cell indices
        if cell_indices is not None:
            self.indices = cell_indices
        else:
            self.indices = np.arange(adata.n_obs)
            
        print(f"Dataset initialized with {len(self.indices)} cells")

        # Create label encoders for each property
        self.label_cols = {
            'cell_type': cell_type_col,
            'pert_type': pert_type_col,
            'pert': pert_col
        }

        # check if the dataset column exists
        use_datasets = False
        if dataset_col:
            if dataset_col not in self.adata.obs.columns:
                warnings.warn(f'Could not find dataset column: {dataset_col} in adata.obs: {self.adata.obs.columns}\n ignoring parameter "dataset_col"')
            else:
                use_datasets = True
                self.label_cols['dataset'] = dataset_col
        self.use_datasets = use_datasets
            
        self.label_encoders = {}
        self.label_decoders = {}
        self.num_classes = {}
        self.encoded_labels = {}
        
        # Process each type of label
        for key, col in self.label_cols.items():
            labels = adata.obs[col].values[self.indices]
            
            # Convert labels to numeric if they're categorical
            if labels.dtype == 'object' or labels.dtype.name == 'category':
                label_encoder = {label: idx for idx, label in enumerate(np.unique(labels))}
                self.label_encoders[key] = label_encoder
                self.label_decoders[key] = {idx: label for label, idx in label_encoder.items()}
                self.num_classes[key] = len(label_encoder)
                self.encoded_labels[key] = np.array([label_encoder[label] for label in labels])
                print(f"{key} classes: {self.num_classes[key]}")
                print(f"{key} labels: {label_encoder}")

    def __len__(self):
        return len(self.indices)

    def to_onehot(self, label: int, num_classes: int) -> torch.Tensor:
        """Convert integer label to one-hot encoded tensor"""
        onehot = torch.zeros(num_classes)
        onehot[label] = 1.0
        return onehot

    def __getitem__(self, idx: int) -> BatchData:
        # Get the cell index
        cell_idx = self.indices[idx]
        
        # Get expression data for the cell
        expression = self.adata.X[cell_idx].toarray() if sp.issparse(self.adata.X) else self.adata.X[cell_idx]
        expression = torch.FloatTensor(expression).squeeze()
        
        # Apply transform if specified
        if self.transform:
            expression = self.transform(expression)
        
        # Get one-hot encoded labels for each property
        oh_map = {k: self.to_onehot(ls[idx], self.num_classes[k]) for k, ls in self.encoded_labels.items()}

        return BatchData(expression, oh_map)
        

class MultiLabelAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: ad.AnnData,
        cell_type_col: str,
        pert_type_col: str,
        pert_col: str,
        batch_size: int = 128,
        train_val_split: float = 0.8,
        num_workers: int = 1,
        cell_indices: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.adata = adata
        self.cell_type_col = cell_type_col
        self.pert_type_col = pert_type_col
        self.pert_col = pert_col
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.cell_indices = cell_indices
        
        # Get number of classes for each property
        self.cell_type_classes = len(adata.obs[cell_type_col].unique())
        self.pert_type_classes = len(adata.obs[pert_type_col].unique())
        self.pert_classes = len(adata.obs[pert_col].unique())
        
        # Create temporary dataset to get label encoders
        temp_dataset = MultiLabelAnnDataset(
            adata=adata,
            cell_type_col=cell_type_col,
            pert_type_col=pert_type_col,
            pert_col=pert_col
        )
        self.label_encoders = temp_dataset.label_encoders
        self.label_decoders = temp_dataset.label_decoders
        
    def setup(self, stage: Optional[str] = None):
        if self.cell_indices is None:
            self.cell_indices = np.arange(self.adata.n_obs)
        # set labels to combination of perturbations
        labels = self.adata.obs[self.cell_type_col].str.cat(self.adata.obs[self.pert_type_col], sep=';').str.cat(self.adata.obs[self.pert_col], sep=';')
        # Second split: train, validation
        train_idx, val_idx = train_test_split(
            range(self.adata.n_obs),
            train_size=self.train_val_split,
            random_state=42,
            stratify=labels
        )
        
        self.train_indices = train_idx
        self.val_indices = val_idx
        
    def train_dataloader(self) -> DataLoader:
        train_dataset = MultiLabelAnnDataset(
            adata=self.adata,
            cell_type_col=self.cell_type_col,
            pert_type_col=self.pert_type_col,
            pert_col=self.pert_col,
            cell_indices=self.train_indices
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        val_dataset = MultiLabelAnnDataset(
            adata=self.adata,
            cell_type_col=self.cell_type_col,
            pert_type_col=self.pert_type_col,
            pert_col=self.pert_col,
            cell_indices=self.val_indices
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    def get_label_name(self, encoded_label: int, label_type: str) -> str:
        """Convert encoded label back to original name"""
        return self.label_decoders[label_type][encoded_label]
    
