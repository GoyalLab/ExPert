import os
import logging
import anndata as ad
import scanpy as sc

import numpy as np
import pandas as pd
import json
import itertools
from typing import List, Dict, Any
import scipy.sparse as sp
from tqdm import tqdm

from src.vae import VAE

import torch
import torch.nn.functional as F
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from typing import NamedTuple


class FCParams(NamedTuple):
    N_LATENT: int = 10
    N_HIDDEN: int = 128
    N_LAYERS: int = 1
    DROPOUT: float = 0.1
    
def mean_pairwise_cosine(z):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    B = z.size(0)
    return (sim.sum() - B) / (B * (B - 1))

def zscore(x: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def batchmean(x: torch.Tensor) -> torch.Tensor:
    return x.sum(-1) / x.shape[0]

def pearson(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8, dim: int = -1):
    """
    Compute Pearson correlation
    x, y: tensors of shape (B, D)
    Returns: (B,) correlations
    """
    assert x.shape == y.shape, "x and y must have the same shape"

    xm = x - x.mean(dim=dim, keepdim=True)  # (B, D)
    ym = y - y.mean(dim=dim, keepdim=True)  # (B, D)

    cov = (xm * ym).sum(dim=dim)            # (B,)
    std = torch.sqrt((xm**2).sum(dim=dim) * (ym**2).sum(dim=dim)).clamp_min(eps)  # (B,)

    return cov / std

def cross_entropy(
            logits: torch.Tensor, 
            targets: torch.Tensor, 
            smoothing: float = 0.05
        ) -> torch.Tensor:
        """
        Cross-entropy with label smoothing.
        logits: (B, num_classes)
        targets: (B,) integer class indices
        """
        # Return default cross entropy
        if not smoothing > 0:
            return F.cross_entropy(logits, targets, reduction='none')
        # Return label smoothing cross entropy
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

        return torch.sum(-true_dist * log_probs, dim=-1)

def add_pp_layer(adata: ad.AnnData) -> None:
    if 'log1p' in adata.layers:
        logging.info('Detected log1p layer, deleting')
        del adata.layers['log1p']
    adata.layers['log1p'] = sc.pp.normalize_total(                                  # normalize for total number of cells
        adata, target_sum=None, inplace=False
    )['X']
    sc.pp.log1p(adata, layer='log1p')                                               # normalize to log1p


def get_zi_rate(adata: ad.AnnData, verbose: bool = True) -> float:
    zi_r = 1 - (adata.X.nnz / (adata.X.shape[0]*adata.X.shape[1]))
    return zi_r

class Cache:
    def __init__(self, n: int):
        self.data = []
        self.n = n
    
    def append(self, x: torch.Tensor):
        if len(self.data) > self.n:
            self.data = [x.detach()]
        else:
            self.data.append(x.detach())

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient sign
        return grad_output.neg() * ctx.lambda_, None
    
def grad_reverse(x: torch.Tensor, lamba_=1.0):
    return GradientReversalFn.apply(x, lamba_)


def run_hyperparameter_search(
    data_module: pl.LightningDataModule,
    model_params: dict,
    search_params: dict,
    base_dir: str = 'hyperparameter_search',
    max_epochs: int = 100,
    patience: int = 10,
    gpu: bool = True
) -> pd.DataFrame:
    """
    Run hyperparameter search and save results.
    
    Args:
        data_module: DataModule containing the data
        model_params: Fixed model parameters
        search_params: Dictionary of parameters to search over
        base_dir: Directory to save results
        max_epochs: Maximum number of epochs per trial
        patience: Early stopping patience
        gpu: Whether to use GPU
    
    Returns:
        DataFrame containing results
    """
    # Create base directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save search configuration
    config = {
        'model_params': model_params,
        'search_params': search_params,
        'max_epochs': max_epochs,
        'patience': patience
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Generate all combinations of hyperparameters
    param_names = list(search_params.keys())
    param_values = list(search_params.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    # Run trials
    for i, values in enumerate(combinations):
        params = dict(zip(param_names, values))
        print(f"\nTrial {i+1}/{len(combinations)}")
        print("Parameters:", params)
        
        # Create trial directory
        trial_dir = os.path.join(save_dir, f"trial_{i+1}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Initialize model with current parameters
        model = VAE(
            **model_params,
            **params
        )
        
        # Initialize callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min'
            ),
            ModelCheckpoint(
                dirpath=trial_dir,
                filename='best_model',
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if gpu and torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            logger=pl.loggers.TensorBoardLogger(
                save_dir=trial_dir,
                name='logs'
            )
        )
        
        # Train model
        try:
            trainer.fit(model, data_module)
            
            # Get best metrics
            best_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
            best_val_acc = trainer.callback_metrics.get('val_accuracy', 0.0)
            
            result = {
                'trial': i+1,
                'best_val_loss': best_val_loss.item(),
                'best_val_accuracy': best_val_acc.item(),
                'epochs_completed': trainer.current_epoch + 1,
                **params
            }
            
        except Exception as e:
            print(f"Error in trial {i+1}: {str(e)}")
            result = {
                'trial': i+1,
                'best_val_loss': float('inf'),
                'best_val_accuracy': 0.0,
                'epochs_completed': 0,
                'error': str(e),
                **params
            }
        
        results.append(result)
        
        # Save intermediate results
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    
    return pd.DataFrame(results)

# Function to analyze and visualize results
def analyze_hyperparameter_results(results_df: pd.DataFrame, save_dir: str = None):
    """
    Analyze and visualize hyperparameter search results.
    """
    import seaborn as sns
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot validation loss vs hyperparameters
    for i, param in enumerate(results_df.columns[4:]):
        if len(results_df[param].unique()) > 1:
            sns.scatterplot(
                data=results_df,
                x=param,
                y='best_val_loss',
                ax=axes[i//2, i%2]
            )
            axes[i//2, i%2].set_title(f'Validation Loss vs {param}')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'hyperparameter_analysis.pdf'))
    plt.show()
    
    # Print best configurations
    print("\nTop 5 configurations by validation loss:")
    print(results_df.sort_values('best_val_loss').head())
    
    return results_df.sort_values('best_val_loss').head(1).to_dict('records')[0]


class HyperparameterSearch:
    def __init__(
        self,
        data_module: pl.LightningDataModule,
        base_model_params: Dict[str, Any],
        search_space: Dict[str, List[Any]],
        max_epochs: int = 100,
        patience: int = 10,
        search_dir: str = 'hyperparameter_search',
        gpu: bool = True
    ):
        """
        Initialize hyperparameter search.
        
        Args:
            data_module: DataModule containing the data
            base_model_params: Fixed model parameters
            search_space: Dictionary of parameters to search over
            max_epochs: Maximum epochs per trial
            patience: Early stopping patience
            search_dir: Directory to save results
            gpu: Whether to use GPU
        """
        self.data_module = data_module
        self.base_model_params = base_model_params
        self.search_space = search_space
        self.max_epochs = max_epochs
        self.patience = patience
        self.gpu = gpu
        
        # Create directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.search_dir = os.path.join(search_dir, timestamp)
        os.makedirs(self.search_dir, exist_ok=True)
        
        # Save configuration
        self.save_config()
        
        # Initialize results tracking
        self.results = []
        
    def save_config(self):
        """Save search configuration"""
        config = {
            'base_model_params': self.base_model_params,
            'search_space': {k: [str(v) for v in vals] for k, vals in self.search_space.items()},
            'max_epochs': self.max_epochs,
            'patience': self.patience
        }
        
        with open(os.path.join(self.search_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def train_model(self, trial_params: Dict[str, Any], trial_num: int) -> Dict[str, Any]:
        """Train a single model with given parameters"""
        # Create trial directory
        trial_dir = os.path.join(self.search_dir, f'trial_{trial_num}')
        os.makedirs(trial_dir, exist_ok=True)
        
        # Initialize model with current parameters
        model_params = {**self.base_model_params, **trial_params}
        model = VAE(**model_params)
        
        # Initialize callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                mode='min'
            ),
            ModelCheckpoint(
                dirpath=trial_dir,
                filename='best_model',
                monitor='val_loss',
                mode='min',
                save_top_k=1
            )
        ]
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.gpu and torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            logger=TensorBoardLogger(
                save_dir=trial_dir,
                name='logs'
            ),
            enable_progress_bar=True
        )
        
        # Train model
        try:
            trainer.fit(model, self.data_module)
            
            # Get best metrics
            metrics = trainer.callback_metrics
            result = {
                'trial': trial_num,
                'best_val_loss': metrics['val_loss'].item(),
                'best_val_accuracy': metrics['val_accuracy'].item(),
                'epochs_completed': trainer.current_epoch + 1,
                'status': 'completed',
                **trial_params
            }
            
        except Exception as e:
            print(f"Error in trial {trial_num}: {str(e)}")
            result = {
                'trial': trial_num,
                'status': 'failed',
                'error': str(e),
                **trial_params
            }
        
        return result
    
    def run_search(self) -> pd.DataFrame:
        """Run full hyperparameter search"""
        combinations = self.generate_combinations()
        print(f"Starting hyperparameter search with {len(combinations)} combinations")
        
        for i, params in enumerate(combinations, 1):
            print(f"\nTrial {i}/{len(combinations)}")
            print("Parameters:", params)
            
            result = self.train_model(params, i)
            self.results.append(result)
            
            # Save intermediate results
            df_results = pd.DataFrame(self.results)
            df_results.to_csv(os.path.join(self.search_dir, 'results.csv'), index=False)
            
        return pd.DataFrame(self.results)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze search results and create visualizations"""
        df = pd.DataFrame(self.results)
        
        # Filter out failed trials
        df_completed = df[df['status'] == 'completed'].copy()
        
        if len(df_completed) == 0:
            print("No successful trials to analyze")
            return {}
        
        # Create visualizations
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Plot parameter relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, param in enumerate(self.search_space.keys()):
            if i < len(axes):
                sns.scatterplot(
                    data=df_completed,
                    x=param,
                    y='best_val_loss',
                    ax=axes[i]
                )
                axes[i].set_title(f'Validation Loss vs {param}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.search_dir, 'parameter_analysis.pdf'))
        plt.close()
        
        # Find best configuration
        best_idx = df_completed['best_val_loss'].idxmin()
        best_config = df_completed.loc[best_idx].to_dict()
        
        # Save best configuration
        with open(os.path.join(self.search_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=4)
        
        return best_config

def evaluate_model(model, test_loader, num_classes):
    """
    Evaluate the model's performance across all classes.
    
    Args:
        model: Trained GraphTransformerClassifier model
        test_loader: PyTorch Geometric DataLoader containing test data
        num_classes: Number of classes in the dataset
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to same device as model
            batch = batch.to(model.device)
            
            # Get model predictions
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            
            # Convert logits to probabilities and then to predicted classes
            pred_probs = torch.sigmoid(out)
            pred_classes = (pred_probs > 0.5).float()
            
            # Reshape target to match output shape
            batch_size = out.shape[0]
            target = batch.y.view(batch_size, num_classes)
            
            # Store predictions and labels
            all_preds.append(pred_classes.cpu())
            all_labels.append(target.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Generate classification report
    class_names = [f"Class_{i}" for i in range(num_classes)]
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Calculate confusion matrix
    conf_matrices = []
    for i in range(num_classes):
        conf_matrices.append(
            confusion_matrix(all_labels[:, i], all_preds[:, i])
        )
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, num_classes, figsize=(5*num_classes, 4))
    if num_classes == 1:
        axes = [axes]
    
    for i, (conf_matrix, ax) in enumerate(zip(conf_matrices, axes)):
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(f'Class {i} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    
    # Calculate per-class accuracy
    per_class_accuracy = {
        f"Class_{i}": report[f"Class_{i}"]['f1-score']
        for i in range(num_classes)
    }
    
    # Calculate overall metrics
    metrics = {
        'overall_accuracy': report['weighted avg']['f1-score'],
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrices': conf_matrices,
        'full_report': report,
        'figure': fig
    }
    
    return metrics


def plot_multilabel_confusion_matrix(model, dataloader, num_classes, figsize=(12, 8)):
    """
    Create a confusion matrix heatmap for all classes in a multi-label classification model.
    
    Args:
        model: Trained GraphTransformerClassifier model
        dataloader: PyTorch Geometric DataLoader containing test data
        num_classes: Number of classes in the dataset
        figsize: Tuple specifying figure size
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to same device as model
            batch = batch.to(model.device)
            
            # Get predictions
            outputs = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            pred_probs = torch.sigmoid(outputs)
            pred_labels = (pred_probs > 0.5).float()
            
            # Reshape targets to match predictions
            batch_size = outputs.shape[0]
            targets = batch.y.view(batch_size, num_classes)
            
            # Store predictions and labels
            all_preds.append(pred_labels.cpu())
            all_labels.append(targets.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate confusion matrices for all classes
    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    
    # Create a figure with subplots arranged in a grid
    n_rows = int(np.ceil(num_classes / 3))  # Up to 3 plots per row
    n_cols = min(3, num_classes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each confusion matrix
    for i in range(num_classes):
        if i < len(axes):
            sns.heatmap(conf_matrices[i], 
                       annot=True, 
                       fmt='d',
                       cmap='Blues',
                       ax=axes[i],
                       cbar=False,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            
            axes[i].set_title(f'Class {i} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
    
    # Remove empty subplots if any
    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(model, dataloader, num_classes, figsize=(10, 8)):
    """
    Create a single confusion matrix heatmap for all classes.
    
    Args:
        model: Trained GraphTransformerClassifier model
        dataloader: PyTorch Geometric DataLoader containing test data
        num_classes: Number of classes in the dataset
        figsize: Tuple specifying figure size
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to same device as model
            batch = batch.to(model.device)
            
            # Get predictions
            outputs = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            pred_probs = torch.sigmoid(outputs)
            
            # Get the index of the highest probability for each prediction
            pred_labels = torch.argmax(pred_probs, dim=1)
            
            # Get the index of the true label
            true_labels = torch.argmax(batch.y.view(pred_probs.shape), dim=1)
            
            # Store predictions and labels
            all_preds.append(pred_labels.cpu())
            all_labels.append(true_labels.cpu())
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create labels for each class
    labels = [f'Class {i}' for i in range(num_classes)]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cm, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Rotate axis labels if there are many classes
    if num_classes > 4:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Print accuracy metrics
    print("\nClassification Metrics:")
    print("-" * 40)
    
    # Calculate per-class metrics
    for i in range(num_classes):
        class_total = np.sum(all_labels == i)
        class_correct = np.sum((all_labels == i) & (all_preds == i))
        accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"Class {i} Accuracy: {accuracy:.4f}")
    
    # Calculate overall accuracy
    overall_accuracy = np.sum(all_labels == all_preds) / len(all_labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    
    return plt.gcf()

def scale(X):
    mean = np.mean(X)
    std = np.std(X)
    X_scaled = (X - mean) / std
    return X_scaled


def plot_confusion(y_true, y_pred, figsize=(10, 8)):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Get class labels (for multiclass classification, this will correspond to unique labels)
    class_labels = list(set(y_true) | (set(y_pred)))

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

    # Print the results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cm_percentage, annot=True, fmt='.0f', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_latent_space(model, indices):
    z = model.get_latent_representation(indices=indices)
    # Create AnnData object
    latent = ad.AnnData(z)
    latent.obs = model.adata.obs.iloc[indices,:]
    print('calulating neighbors')
    sc.pp.neighbors(latent, use_rep='X')
    print('calulating umap')
    sc.tl.umap(latent)
    sc.pl.umap(latent, color='dataset')
    sc.pl.umap(latent, color='celltype')
    sc.pl.umap(latent, color='exact_perturbation')


def umap(adata: ad.AnnData, color):
    logging.info('PCA')
    sc.pp.pca(adata)
    logging.info('Neighbors')
    sc.pp.neighbors(adata)
    logging.info('UMAP')
    sc.tl.umap(adata)
    logging.info('Plotting')
    sc.pl.umap(adata, color=color)

def get_mean_n_var(X) -> tuple[np.matrix, np.matrix]:
    m = X.mean(axis=0)
    xi = X - m
    v = np.mean(np.power(2, xi), axis=0)
    return m, v

def center_on_ctrl(
        adata: ad.AnnData,
        group_by: str | List[str],
        col: str = 'perturbation',
        ctrl_key: str = 'control',
        layer: str | None = 'centered',
        label: str = 'ctrl_diff'
    ) -> None:
    if not isinstance(adata.X, sp.csr_matrix):
        adata.X = sp.csr_matrix(adata.X)
    idc = []
    centered_ds = []
    for ds in tqdm(adata.obs[group_by].unique(), desc='Centering data', unit='dataset'):
        adata_ds = adata[adata.obs[group_by]==ds]
        idc.extend(adata_ds.obs.index.tolist())
        # Calculate control gene-wise mean and variance
        adata_ctrl_x = adata_ds[adata_ds.obs[col]==ctrl_key].X
        ctrl_mean, ctrl_var = get_mean_n_var(adata_ctrl_x)
        # Center perturbation data on control
        centered_X = (adata_ds.X - ctrl_mean) / ctrl_var
        centered_ds.append(centered_X)
    # Concatenate list of data
    centered_ds = sp.csr_matrix(np.concatenate(centered_ds, axis=0))
    # Sort adata by dataset
    adata = adata[idc,:].copy()
    # Add centered data as layer
    if layer is not None:
        adata.layers[layer] = centered_ds
    adata.obs[label] = np.abs(centered_ds.sum(axis=1))
