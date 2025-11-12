import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from typing import Literal

import umap

import matplotlib.pyplot as plt
import seaborn as sns

from scanpy.plotting import palettes

from src.utils.constants import REGISTRY_KEYS

import logging
log = logging.getLogger(__name__)


def get_pal(n: int, seed: int | None = None) -> list:
    """Return a list of n colors, using sensible defaults for small/medium sizes
    and falling back to evenly spaced hues for large n.
    """
    if seed is not None:
        np.random.seed(seed)
    # Small: use tab10 (up to 10)
    if n <= 10:
        return sns.color_palette('tab10', n)
    # Medium: use tab20 (up to 20)
    if n <= 20:
        return sns.color_palette('tab20', n)
    # Up to scanpy's default 102-color palette
    default_len = len(palettes.default_102)
    if n <= default_len:
        return palettes.default_102[:n]

    # Large: generate evenly spaced hues (HLS) for arbitrary n
    return sns.color_palette('hls', n)

def calc_umap(adata: ad.AnnData, rep: str = 'X_pca', slot_key: str | None = None, force: bool = True, return_umap: bool = False) -> None | umap.UMAP:
    # Create default slot name if none is given, else use the specified key
    slot_key = f'{rep}_umap' if slot_key is None else slot_key
    # Cache umap generation
    if slot_key in adata.obsm and not force:
        return
    # Setup umap instance, with sc.tl.umap defaults, apart from random state (slows it down a lot)
    _umap = umap.UMAP(n_components=2, min_dist=0.5)
    # Set embeddings to use
    if rep not in adata.obsm:
        raise ValueError(f'Could not find {rep} slot in adata.obsm.')
    # Save embeddings to adata
    adata.obsm[slot_key] = _umap.fit_transform(adata.obsm[rep])
    return _umap if return_umap else None

def plot_umap(
        adata: ad.AnnData, 
        slot: str, 
        hue: str, 
        output_file: str | None = None,
        title: str = '',
        show_spines: bool = True,
        **scatter_kwargs,
    ) -> None:
    # Check if adata.obsm slot exists
    if slot not in adata.obsm:
        raise ValueError(f'Could not find {slot} slot in adata.obsm.')
    # Get slot umap data
    data = adata.obsm[slot]
    # Add column labels and construct plotting dataframe
    cols = 'UMAP' + pd.Series(np.arange(data.shape[1])+1).astype(str)
    embedding = pd.DataFrame(data, columns=cols)
    # Transfer hue label from adata.obs
    embedding[hue] = adata.obs[hue].values
    # Add palette if less than 103 labels are given
    n_hue = embedding[hue].nunique()
    # Enable palette with legend
    if n_hue < 103:
        pal = get_pal(n_hue)
        scatter_kwargs['palette'] = pal
        scatter_kwargs['legend'] = True
    # Disable legend for too many labels
    else:
        scatter_kwargs['legend'] = True
    # Create plot
    plt.figure(dpi=300)
    default_scatter_kwargs = {'s': 4, 'alpha': 0.5}
    default_scatter_kwargs.update(scatter_kwargs)
    ax = sns.scatterplot(
        embedding, x='UMAP1', y='UMAP2', 
        hue=hue,
        **default_scatter_kwargs
    )
    # Toggle spines
    if not show_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)
    # Set axes labels
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title(title)
    # Push legend outside of the plot if we want to show it
    if scatter_kwargs.get('legend', False):
        max_labels_per_col = 20
        ncol = max(int(n_hue / max_labels_per_col), 1)
        plt.legend(
            title=hue,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0, 
            markerscale=2,
            ncol=ncol
        )
    # Just show the plot if no output file is provided
    if output_file is None:
        plt.show()
    # Save the plot to the given file path
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # Close the plot in any case
    plt.close()

def plot_umaps(
        adata: ad.AnnData,
        slot: str = 'X_umap',
        hue: list[str] | str = 'cls_label',
        output_dir: str | None = None,
        **kwargs
    ) -> None:
    """Function to plot multiple umaps."""
    # Ensure hue is always a list
    hue = [hue] if not isinstance(hue, list) else hue
    for h in hue:
        o = os.path.join(output_dir, f'umap_{h}.png') if output_dir is not None else None
        plot_umap(
            adata=adata, 
            slot=slot, 
            hue=h, 
            output_file=o
        )

def calc_umap_scanpy(adata: ad.AnnData, rep: str = 'X') -> None:
    if rep == 'X' and adata.X.shape[1] > 512:
        # Falling back to pca
        rep = 'X_pca'
        sc.pp.pca(adata, n_comps=50)
    log.info(f'Calculating latent neighbors')
    sc.pp.neighbors(adata, use_rep=rep)
    log.info(f'Calculating latent umap')
    sc.tl.umap(adata)

def plot_umap_scanpy(adata: ad.AnnData, hue: str, split: str, plt_dir: str) -> None:
    sc.pl.umap(adata, color=hue, return_fig=True, show=False)
    plt.savefig(os.path.join(plt_dir, f'{split}_umap_{hue}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion(
        y_true: np.ndarray | list, 
        y_pred: np.ndarray | list, 
        figsize: tuple[int, int] = (10, 8), 
        hm_kwargs: dict = {'annot': False}, 
        verbose: bool = False, 
        plt_file: str | None = None
    ) -> None:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Get class labels (for multiclass classification, this will correspond to unique labels)
    class_labels = np.unique(y_true)
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Print the results
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cm_percentage, xticklabels=class_labels, yticklabels=class_labels, **hm_kwargs)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if plt_file is None:
        plt.show()
    else:
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_support_corr(report: pd.DataFrame, o: str, hue: str | None = None):
    if report['f1-score'].min() == report['f1-score'].max():
        msg = f'Got uniform values for f1-score'
        if hue and hue in report.columns:
            msg += f', cannot plot kde for mode {report[hue].unique()[0]}.'
        log.info(msg)
        return

    plt.figure(figsize=(6, 5))

    # KDE plot
    sns.kdeplot(
        data=report,
        x='log_count',
        y='f1-score',
        hue=hue,
        fill=True,
        common_norm=False,
        alpha=0.4
    )

    # Scatter plot
    sns.scatterplot(
        data=report,
        x='log_count',
        y='f1-score',
        hue=hue,
        edgecolor='black',
        alpha=0.7,
        s=40,
        legend=(hue is not None)  # only show legend if hue is used
    )

    plt.xlabel('Class support (log)')
    plt.ylabel('Macro f1-score')
    plt.tight_layout()
    plt.savefig(o, dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_n_performance(
        top_n_predictions: pd.DataFrame,
        output_file: str | None,
        hue: str = 'split',
        cls_label: str = 'label',
        metric: Literal['f1-score', 'accuracy', 'precision', 'recall'] = 'f1-score',
        show_random: bool = True,
        title: str | None = None,
        top_n: int = 10,
        mean_split: Literal['train', 'val', 'test'] | None = 'test',
        **kwargs
    ) -> None:
    # Replace title with number of classes
    n_classes = top_n_predictions.label.nunique()
    title = f'Top N predictions (N={n_classes})'
    # Subset data to only show up to top_n predictions
    data = top_n_predictions[top_n_predictions.top_n <= top_n].copy()
    # Show random or not
    if show_random:
        # Replace all values of hue with random if 'mode' is random
        data.loc[data['mode']=='random',hue] = 'random'
    # Calculate mean values for a specified split if given
    if mean_split is not None and mean_split in top_n_predictions[hue].unique():
        top_split = data[data[hue]==mean_split]
        means = top_split.groupby('top_n')[metric].mean()
    # Don't plot means
    else:
        means = None
    # Create figure
    plt.figure(dpi=120, figsize=(10,6))
    # Ensure top_n is numeric
    data['top_n'] = pd.to_numeric(data['top_n'])
    # Plot distributions for every data split and top predictions
    ax = sns.boxenplot(
        data,
        x='top_n',
        y=metric,
        hue=hue,
        **kwargs
    )
    # Get handles and current labels
    handles, labels = ax.get_legend_handles_labels()
    # Compute unique counts per hue
    counts = data.groupby([hue])[cls_label].nunique()
    # Build new labels with (N=...)
    new_labels = [f"{lab} (N={counts.get(lab, 0)})" for lab in labels]
    # Put legend outside of main plot and add counts
    ax.legend(
        handles,
        new_labels,
        title=hue,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )

    # Rename axis labels
    plt.xlabel('Top N predictions')
    plt.ylabel(metric.capitalize())
    plt.title(f'{title} (w. {mean_split} mean)', pad=20)
    # Set y axis limits to 0-1
    plt.ylim((0, 1))
    # Display means on top of boxes if not none
    for i, top_n_value in enumerate(sorted(data['top_n'].unique())):
        ax.text(i, 1.0, f'{means[top_n_value]:.2f}', 
                horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    # Save plot to file if given
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    # Close the plot
    plt.close()