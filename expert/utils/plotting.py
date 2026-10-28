# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from tqdm import tqdm

from typing import Literal, TYPE_CHECKING
from pandas.api.types import is_numeric_dtype

# NOTE: do NOT import umap at module load. umap-learn's __init__ pulls in
# umap.parametric_umap → TensorFlow (~25s) whenever TF is installed, which
# dominated `import expert` time. umap is only needed by calc_umap, so it's
# imported lazily there.
if TYPE_CHECKING:
    import umap

import matplotlib.pyplot as plt
import seaborn as sns

from scanpy.plotting import palettes

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

def calc_umap(adata: ad.AnnData, rep: str = 'X_pca', slot_key: str | None = None, force: bool = True, return_umap: bool = False) -> "None | umap.UMAP":
    import umap  # lazy: avoids the TensorFlow pull at module import time
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

def _prepare_umap_df(adata: ad.AnnData, slot: str, hue: str) -> pd.DataFrame:
    """Construct plotting dataframe."""
    if slot not in adata.obsm:
        raise ValueError(f"Could not find {slot} slot in adata.obsm.")

    data = adata.obsm[slot]

    cols = "UMAP" + pd.Series(np.arange(data.shape[1]) + 1).astype(str)
    embedding = pd.DataFrame(data, columns=cols)

    if hue in adata.obs:
        embedding[hue] = adata.obs[hue].values
    elif hue in adata.var:
        embedding[hue] = adata.var[hue].values
    else:
        raise ValueError(f"{hue} not found in adata.obs or adata.var")

    return embedding


def _plot_umap_categorical(
    embedding: pd.DataFrame,
    hue: str,
    title: str,
    output_file: str | None,
    show_spines: bool,
    **scatter_kwargs,
):

    n_hue = embedding[hue].nunique()

    if n_hue < 103:
        scatter_kwargs["palette"] = get_pal(n_hue)
        scatter_kwargs["legend"] = True
    else:
        scatter_kwargs["legend"] = True

    fig, ax = plt.subplots(dpi=300)

    default_scatter_kwargs = {"s": 4, "alpha": 0.5}
    default_scatter_kwargs.update(scatter_kwargs)

    sns.scatterplot(
        data=embedding,
        x="UMAP1",
        y="UMAP2",
        hue=hue,
        ax=ax,
        **default_scatter_kwargs,
    )

    if not show_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)

    if scatter_kwargs.get("legend", False):
        max_labels_per_col = 20
        ncol = max(int(n_hue / max_labels_per_col), 1)

        ax.legend(
            title=hue,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            markerscale=2,
            ncol=ncol,
        )

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close()


def _plot_umap_continuous(
    embedding: pd.DataFrame,
    hue: str,
    title: str,
    output_file: str | None,
    show_spines: bool,
    cmap="viridis",
    **scatter_kwargs,
):

    fig, ax = plt.subplots(dpi=300)

    default_scatter_kwargs = {"s": 4, "alpha": 0.5}
    default_scatter_kwargs.update(scatter_kwargs)

    sc = ax.scatter(
        embedding["UMAP1"],
        embedding["UMAP2"],
        c=embedding[hue],
        cmap=cmap,
        **default_scatter_kwargs,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(hue)

    if not show_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.close()


def plot_umap(
    adata: ad.AnnData,
    slot: str,
    hue: str,
    output_file: str | None = None,
    title: str = "",
    show_spines: bool = True,
    **scatter_kwargs,
):
    """
    Plot UMAP embedding with automatic handling of categorical vs continuous hues.
    """

    embedding = _prepare_umap_df(adata, slot, hue)

    if is_numeric_dtype(embedding[hue]):
        _plot_umap_continuous(
            embedding,
            hue,
            title,
            output_file,
            show_spines,
            **scatter_kwargs,
        )
    else:
        _plot_umap_categorical(
            embedding,
            hue,
            title,
            output_file,
            show_spines,
            **scatter_kwargs,
        )


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
        N: int | None = None,
        top_n: int = 10,
        mean_split: Literal['train', 'val', 'test'] | None = 'test',
        **kwargs
    ) -> None:
    # Replace title with number of classes
    n_classes = top_n_predictions.label.nunique()
    title = f'Top N predictions (#observed={n_classes})'
    if N is not None:
        title += f', #available classes: {N}'
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


def plot_top_n_performance_per_dataset(
        top_n_predictions: pd.DataFrame,
        context_key: str = 'dataset',
        split_key: str = 'split',
        output_file: str | None = None,
        metric: Literal['f1-score', 'accuracy', 'precision', 'recall'] = 'f1-score',
        show_random: bool = True,
        top_n: int = 10,
        top_n_step: int = 1,
        figsize_per_row: tuple[float, float] = (14, 3.5),
        **kwargs,
    ) -> None:
    """
    Plot top-N performance per dataset.
    Rows = top_n values, x = dataset, y = metric, hue = split.
    Point size = class support within context.
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    data = top_n_predictions[top_n_predictions.top_n <= top_n].copy()
    data['top_n'] = pd.to_numeric(data['top_n'])

    # Include random in split hue
    if show_random:
        data.loc[data['mode'] == 'random', split_key] = 'random'
    else:
        data = data[data['mode'] != 'random']

    # Subsample top_n values by step
    top_n_values = sorted(data['top_n'].unique())
    top_n_values = [t for t in top_n_values if (t - 1) % top_n_step == 0 or t == max(top_n_values)]
    data = data[data['top_n'].isin(top_n_values)]

    n_rows = len(top_n_values)
    datasets = sorted(data[context_key].unique())
    ds_to_x = {ds: i for i, ds in enumerate(datasets)}
    splits = sorted(data[split_key].unique())
    split_palette = sns.color_palette('tab10', n_colors=len(splits))
    split_to_color = dict(zip(splits, split_palette))

    # Support-based sizing
    if 'support' in data.columns:
        global_support = data['support'].values
        s_min, s_max = 15, 120
    else:
        global_support = None

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        squeeze=False,
        sharex=True,
    )

    for row, tn in enumerate(top_n_values):
        ax = axes[row, 0]
        tn_data = data[data['top_n'] == tn]

        # Boxplot per split
        sns.boxplot(
            data=tn_data,
            x=context_key,
            y=metric,
            hue=split_key,
            order=datasets,
            hue_order=splits,
            ax=ax,
            showfliers=False,
            boxprops=dict(alpha=0.3),
            width=0.7,
            **kwargs,
        )

        # Scatter overlay with support-based sizing
        jitter_width = 0.7 / len(splits)
        for s_idx, split in enumerate(splits):
            split_data = tn_data[tn_data[split_key] == split]
            if len(split_data) == 0:
                continue

            x_pos = split_data[context_key].map(ds_to_x).values
            offset = (s_idx - len(splits) / 2 + 0.5) * jitter_width
            x_jittered = x_pos + offset + np.random.uniform(-jitter_width * 0.3, jitter_width * 0.3, size=len(x_pos))

            if global_support is not None and 'support' in split_data.columns:
                support = split_data['support'].values
                sizes = np.sqrt(support)
                sizes = (sizes - np.sqrt(global_support.min())) / (np.sqrt(global_support.max()) - np.sqrt(global_support.min()) + 1e-6)
                sizes = s_min + sizes * (s_max - s_min)
            else:
                sizes = 30

            ax.scatter(
                x_jittered,
                split_data[metric].values,
                c=[split_to_color[split]],
                s=sizes,
                alpha=0.6,
                linewidths=0,
                zorder=5,
            )

        # Per-split mean annotation
        split_means = tn_data.groupby(split_key)[metric].mean()
        mean_str = ' | '.join([f'{s}: {split_means.get(s, 0):.2f}' for s in splits if s != 'random'])
        ax.set_title(f'Top {tn}  —  {mean_str}', fontsize=10, fontweight='bold')

        ax.set_ylabel(metric.capitalize(), fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.get_legend().remove()

        if row < n_rows - 1:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('')
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)

    # Shared legend at top
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=split_to_color[s], alpha=0.7, label=s) for s in splits]
    if global_support is not None:
        legend_elements.append(plt.scatter([], [], s=s_min, c='gray', alpha=0.6, label=f'support={int(global_support.min())}'))
        legend_elements.append(plt.scatter([], [], s=s_max, c='gray', alpha=0.6, label=f'support={int(global_support.max())}'))
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=len(legend_elements),
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
def plot_umap_with_proxies(
    adata,
    latent_key: str,
    proxy_key: str | None = None,
    perturbation_col: str = "perturbation",
    batch_key: str | None = None,
    idx_to_label: np.ndarray | None = None,
    arrows_to_proxy: bool = False,
    arrow_frac: float = 0.02,
    palette: str = "tab20",
    title: str = "",
    output_file: str | None = None,
):
    import umap
    # --------------------------------------------------
    # Add latent space
    # --------------------------------------------------
    latent = adata.obsm[latent_key]
    # Normalize latent
    latent = latent / (np.linalg.norm(latent, axis=1, keepdims=True) + 1e-12)
    N = latent.shape[0]
    # Collect perturbation values
    perturbations = adata.obs[perturbation_col].astype(str).values
    # Collect covariates
    covs = (
        adata.obs[batch_key].astype(str).values
        if batch_key is not None
        else np.repeat("cell", N)
    )

    # --------------------------------------------------
    # Add proxies
    # --------------------------------------------------
    proxies = None
    proxy_labels = None
    # Add class proxies if they exist
    if proxy_key and proxy_key in adata.uns:
        # Get class proxies from model adata and normalize
        proxies = adata.uns[proxy_key]
        # Normalize proxies
        proxies = proxies / (np.linalg.norm(proxies, axis=1, keepdims=True) + 1e-12)

        if proxies.ndim == 3:
            proxies = proxies.mean(1)

        # TRAINED PREFIX ONLY -- see expert_plans.py. The untrained tail is random-init and
        # only adds a blob that distorts the UMAP fit.
        if idx_to_label is not None:
            proxies = proxies[:len(idx_to_label)]
        n_proxies = proxies.shape[0]
        proxy_labels = np.arange(n_proxies).astype(int)
        # Annotate proxies
        if idx_to_label is not None:
            proxy_labels = idx_to_label[proxy_labels]
        else:
            proxy_labels = proxy_labels.astype(str)
        observed = set(perturbations)

        proxy_is_obs = np.array([p in observed for p in proxy_labels])

        embeddings_all = np.concatenate([latent, proxies], axis=0)

        df = pd.DataFrame({
            "label": np.concatenate([perturbations, proxy_labels]),
            "cov": np.concatenate([covs, np.repeat("proxy", n_proxies)]),
            "is_proxy": np.concatenate([np.zeros(N, bool), np.ones(n_proxies, bool)]),
            "is_observed": np.concatenate([np.ones(N, bool), proxy_is_obs]),
        })

    else:
        # No proxies, just use latent space for plotting
        embeddings_all = latent
        df = pd.DataFrame({
            "label": perturbations,
            "cov": covs,
            "is_proxy": np.zeros(N, bool),
            "is_observed": np.ones(N, bool),
        })

    # --------------------------------------------------
    # UMAP
    # --------------------------------------------------
    reducer = umap.UMAP(n_components=2)
    emb_2d = reducer.fit_transform(embeddings_all)

    df["UMAP1"] = emb_2d[:, 0]
    df["UMAP2"] = emb_2d[:, 1]

    # --------------------------------------------------
    # observed vs unseen label
    # --------------------------------------------------
    df["obs_proxy"] = np.where(
        df.is_proxy,
        np.where(df.is_observed, "Observed proxy", "Unseen proxy"),
        "Cells",
    )

    # --------------------------------------------------
    # scatter helper
    # --------------------------------------------------
    def _scatter(ax, data, hue, title, plot_proxy=True, annot=False):
        unique_labels = sorted(data[hue].unique())
        pal = sns.color_palette(palette, len(unique_labels))
        color_map = dict(zip(unique_labels, pal))

        sns.scatterplot(
            data=data[~data.is_proxy],
            x="UMAP1",
            y="UMAP2",
            hue=hue,
            palette=color_map,
            s=6,
            alpha=0.6,
            ax=ax,
            legend=False,
        )

        if plot_proxy and data.is_proxy.any():

            proxy_df = data[data.is_proxy]

            sns.scatterplot(
                data=proxy_df,
                x="UMAP1",
                y="UMAP2",
                hue=hue,
                palette=color_map,
                marker="X",
                s=34,          # smaller: at s=80 the crosses swamp the cells underneath
                edgecolor="black",
                linewidth=0.4,
                ax=ax,
                legend=False,
            )

            # label proxies if not too many
            if annot and proxy_df.shape[0] <= 200:
                for _, row in proxy_df.iterrows():
                    ax.text(
                        row["UMAP1"] + 0.02,
                        row["UMAP2"] + 0.02,
                        row[hue],
                        fontsize=7,
                        color=color_map[row[hue]],
                        weight="bold",
                    )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # --------------------------------------------------
    # figure layout
    # --------------------------------------------------
    ncols = 3 if batch_key else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), dpi=150)

    axes = np.array(axes).flatten()

    _scatter(
        axes[0],
        df,
        "label",
        f"{title} Perturbations",
        annot=True
    )
    _scatter(
        axes[1],
        df,
        "obs_proxy",
        f"{title} Observed vs Unseen proxies",
    )
    if batch_key:
        _scatter(
            axes[2],
            df,
            "cov",
            f"{title} Contexts",
            plot_proxy=False,
        )

    # --------------------------------------------------
    # optional arrows to nearest proxy
    # --------------------------------------------------
    if arrows_to_proxy and proxies is not None:
        from sklearn.metrics import pairwise_distances
        cell_z = latent
        proxy_z = proxies

        D = pairwise_distances(cell_z, proxy_z)
        nearest = np.argmin(D, axis=1)

        cell_umap = emb_2d[:N]
        proxy_umap = emb_2d[N:]

        rng = np.random.default_rng(0)
        idx = rng.choice(N, int(N * arrow_frac), replace=False)

        for i in idx:
            j = nearest[i]
            axes[0].annotate(
                "",
                xy=proxy_umap[j],
                xytext=cell_umap[i],
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    alpha=0.3,
                    linewidth=0.5,
                ),
            )

    plt.tight_layout()
    # Save plot to file
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
    # Show plot
    else:
        plt.show()
    # Close plot
    plt.close(fig)
    return df, fig


def create_umap_gif(
    csv_dir: str,
    output_path: str | None = None,
    hue: str = 'cov',
    duration: int = 400,
    loop: int = 0,
    dpi: int = 150,
    s: float = 4,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (8, 6),
    proxy_marker: str = 'X',
    proxy_s: float = 60,
    highlight_observed: bool = True,
    max_labels: int = 50,
    verbose: bool = True,
) -> str:
    """Create a GIF animation from UMAP CSV snapshots across epochs.

    Caches rendered frame PNGs in ``<csv_dir>/.frame_cache/`` so that
    subsequent calls only render new epochs.  If the global axis limits or
    colour palette change (e.g. a new label appears), the cache is
    automatically invalidated and all frames are re-rendered.

    Parameters
    ----------
    csv_dir
        Directory containing files named ``umap_data_epoch{N}.csv``.
    output_path
        Path for the output ``.gif``. Defaults to ``<csv_dir>/umap_animation.gif``.
    hue
        Column to color points by.
    duration
        Milliseconds per frame.
    loop
        Number of times the GIF loops (0 = infinite).
    dpi
        Resolution of each frame.
    s
        Scatter point size for cells.
    alpha
        Scatter point alpha for cells.
    figsize
        Figure size per frame.
    proxy_marker
        Marker style for proxy (class embedding) points.
    proxy_s
        Marker size for proxy points.
    highlight_observed
        If True, dim unobserved classes.
    max_labels
        Maximum number of unique labels to render as text annotations.
    verbose
        Show progress bar.

    Returns
    -------
    Path to the saved GIF.
    """
    import re
    import io
    import json
    import hashlib
    import shutil
    from PIL import Image

    if output_path is None:
        output_path = os.path.join(csv_dir, f'umap_{hue}.gif')

    cache_dir = os.path.join(csv_dir, '.frame_cache', hue)

    # Discover and sort CSV files by epoch number
    pattern = re.compile(r'umap_data_epoch(\d+)\.csv$')
    csv_files = []
    for f in os.listdir(csv_dir):
        m = pattern.match(f)
        if m:
            csv_files.append((int(m.group(1)), os.path.join(csv_dir, f)))
    csv_files.sort(key=lambda x: x[0])

    if not csv_files:
        raise FileNotFoundError(f"No umap_data_epoch*.csv files found in {csv_dir}")

    # Build a consistent palette across all epochs
    all_labels = set()
    for _, fp in csv_files:
        df = pd.read_csv(fp, usecols=[hue], low_memory=False)
        all_labels.update(df[hue].unique())
    all_labels = sorted(all_labels, key=str)
    palette = dict(zip(all_labels, get_pal(len(all_labels))))

    # Compute global axis limits across all epochs
    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    for _, fp in csv_files:
        df = pd.read_csv(fp, usecols=['UMAP1', 'UMAP2'], low_memory=False)
        x_min = min(x_min, df['UMAP1'].min())
        x_max = max(x_max, df['UMAP1'].max())
        y_min = min(y_min, df['UMAP2'].min())
        y_max = max(y_max, df['UMAP2'].max())
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    # Build a metadata fingerprint to detect when cached frames are stale
    meta = json.dumps({
        'labels': all_labels,
        'xlim': [float(x_min - pad_x), float(x_max + pad_x)],
        'ylim': [float(y_min - pad_y), float(y_max + pad_y)],
    }, sort_keys=True)
    meta_hash = hashlib.md5(meta.encode()).hexdigest()

    # Check if existing cache is still valid
    meta_path = os.path.join(cache_dir, '_meta.json')
    cache_valid = False
    if os.path.isdir(cache_dir) and os.path.isfile(meta_path):
        try:
            with open(meta_path) as fh:
                old_meta = json.load(fh)
            cache_valid = old_meta.get('hash') == meta_hash
        except (json.JSONDecodeError, KeyError):
            pass
    if not cache_valid and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    with open(meta_path, 'w') as fh:
        json.dump({'hash': meta_hash}, fh)

    # Render each epoch as a frame, skipping cached ones
    frames = []
    file_iter = tqdm(csv_files) if verbose else csv_files
    for epoch, fp in file_iter:
        frame_path = os.path.join(cache_dir, f'frame_{epoch:05d}.png')

        if os.path.isfile(frame_path):
            frames.append(Image.open(frame_path).copy())
            continue

        df = pd.read_csv(fp, low_memory=False)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Split into cells and proxies
        is_proxy = df['is_proxy'].astype(bool)
        cells = df[~is_proxy]
        proxies = df[is_proxy]

        # Dim unobserved cells if requested
        if highlight_observed and 'is_observed' in df.columns:
            observed = cells[cells['is_observed'].astype(bool)]
            unobserved = cells[~cells['is_observed'].astype(bool)]
            if len(unobserved) > 0:
                ax.scatter(
                    unobserved['UMAP1'], unobserved['UMAP2'],
                    c='lightgrey', s=s * 0.5, alpha=alpha * 0.3,
                    linewidths=0, zorder=1,
                )
            plot_cells = observed
        else:
            plot_cells = cells

        # Plot cells
        colors = [palette.get(lab, 'grey') for lab in plot_cells[hue]]
        ax.scatter(
            plot_cells['UMAP1'], plot_cells['UMAP2'],
            c=colors, s=s, alpha=alpha,
            linewidths=0, zorder=2,
        )

        # Plot proxies
        if len(proxies) > 0:
            proxy_colors = [palette.get(lab, 'grey') for lab in proxies[hue]]
            ax.scatter(
                proxies['UMAP1'], proxies['UMAP2'],
                c=proxy_colors, s=proxy_s, alpha=0.9,
                marker=proxy_marker, edgecolors='black', linewidths=0.5,
                zorder=3,
            )

        # Add text labels at group centroids if few enough categories
        n_unique = df[hue].nunique()
        if n_unique <= max_labels:
            for lab, grp in plot_cells.groupby(hue):
                cx, cy = grp['UMAP1'].mean(), grp['UMAP2'].mean()
                ax.text(
                    cx, cy, str(lab),
                    fontsize=max(4, 8 - n_unique // 10),
                    ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.15',
                        facecolor='white', edgecolor='none', alpha=0.7,
                    ),
                    zorder=4,
                )

        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')

        # Render to PIL Image and cache the PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frame_img = Image.open(buf).copy()
        buf.close()
        frame_img.save(frame_path)
        frames.append(frame_img)

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    if verbose:
        log.info(f"Saved UMAP animation ({len(frames)} frames) to {output_path}")
    return output_path


def plot_efficiency_comparison(
    obs: pd.DataFrame,
    ref_col: str = 'eff',
    pred_col: str = 'eff_hat',
    split_col: str = 'split',
    splits: list[str] | None = None,
    dataset_col: str | None = None,
    datasets: list[str] | None = None,
    plt_file: str | None = None,
    bins: int = 60,
    kde_max_n: int = 50_000,
    figsize: tuple[float, float] | None = None,
    return_fig: bool = False,
) -> pd.DataFrame:
    """Compare predicted vs reference efficiency per split.

    Computes Spearman / Pearson / R² / Wasserstein per split and renders
    a two-row figure: a ridge-style density overlay (predicted filled,
    reference outlined) on top, and hexbin scatters with correlation
    metrics in titles below. When ``dataset_col`` is provided, the
    hexbin section becomes a (datasets x splits) grid; the ridge plot
    stays per-split regardless.

    Parameters
    ----------
    obs
        Frame containing ``ref_col``, ``pred_col``, and ``split_col``.
        Typically ``adata.obs``.
    ref_col, pred_col, split_col
        Column names for reference efficiency, predicted efficiency, and
        the split assignment (e.g. 'train' / 'val' / 'test' / 'ood_test').
    splits
        Subset / order of splits to plot. Defaults to sorted unique values.
    dataset_col
        Optional column to further split hexbins by (e.g. 'dataset').
        When set, the hexbin section becomes a grid of
        (datasets x splits) and per-cell metrics are computed.
    datasets
        Subset / order of datasets when ``dataset_col`` is set.
        Defaults to sorted unique values.
    plt_file
        If given, save figure to this path instead of showing.
    bins
        Hexbin gridsize.
    kde_max_n
        Subsample cap per split for KDE fitting (keeps the ridge plot
        fast on multi-million-cell adatas; correlation metrics use all
        cells regardless).

    Returns
    -------
    pd.DataFrame
        When ``dataset_col`` is None: indexed by split.
        When ``dataset_col`` is set: MultiIndex (split, dataset).
        Columns: ``n / spearman / pearson / r2 / wasserstein``.
    """
    from scipy.stats import spearmanr, pearsonr, wasserstein_distance, gaussian_kde
    from sklearn.metrics import r2_score

    cols = [ref_col, pred_col, split_col]
    if dataset_col is not None:
        cols = cols + [dataset_col]
    df = obs[cols].dropna().copy()
    if splits is None:
        splits = sorted(df[split_col].unique().tolist())
    if dataset_col is not None and datasets is None:
        datasets = sorted(df[dataset_col].unique().tolist())

    def _metrics_for(sub: pd.DataFrame) -> dict:
        if len(sub) < 2:
            return None
        ref = sub[ref_col].to_numpy()
        pred = sub[pred_col].to_numpy()
        try:
            rho, _ = spearmanr(ref, pred)
        except Exception:
            rho = np.nan
        try:
            r, _ = pearsonr(ref, pred)
        except Exception:
            r = np.nan
        try:
            r2 = r2_score(ref, pred)
        except Exception:
            r2 = np.nan
        try:
            w = wasserstein_distance(ref, pred)
        except Exception:
            w = np.nan
        return {'n': len(sub), 'spearman': rho, 'pearson': r, 'r2': r2, 'wasserstein': w}

    # Per-split metrics — always computed (used for ridge Wasserstein annotations).
    split_rows = []
    for s in splits:
        m = _metrics_for(df[df[split_col] == s])
        if m is not None:
            split_rows.append({'split': s, **m})
    split_metrics = (
        pd.DataFrame(split_rows).set_index('split')
        if split_rows else pd.DataFrame(columns=['n', 'spearman', 'pearson', 'r2', 'wasserstein'])
    )

    # Per-(split, dataset) metrics when dataset_col given.
    grid_metrics: pd.DataFrame | None = None
    if dataset_col is not None:
        grid_rows = []
        for s in splits:
            for d in datasets:
                sub = df[(df[split_col] == s) & (df[dataset_col] == d)]
                m = _metrics_for(sub)
                if m is not None:
                    grid_rows.append({'split': s, 'dataset': d, **m})
        grid_metrics = (
            pd.DataFrame(grid_rows).set_index(['split', 'dataset'])
            if grid_rows else pd.DataFrame(columns=['n', 'spearman', 'pearson', 'r2', 'wasserstein'])
        )

    n_splits = max(len(splits), 1)
    n_rows_hex = len(datasets) if dataset_col is not None else 1
    if figsize is None:
        ridge_h = max(1.0, 0.6 * n_splits)
        hex_h = 3.5 * n_rows_hex
        figsize = (4 * n_splits, ridge_h + hex_h)
    fig = plt.figure(figsize=figsize)
    ridge_h = max(1.0, 0.6 * n_splits)
    hex_h = 3.5 * n_rows_hex
    gs = fig.add_gridspec(
        1 + n_rows_hex, n_splits,
        height_ratios=[ridge_h] + [3.5] * n_rows_hex,
    )

    # Common axis range — clipped to robust quantiles so tail outliers don't
    # compress the ridge / scatter into a sliver.
    q_lo, q_hi = df[[ref_col, pred_col]].stack().quantile([0.001, 0.999]).values
    x_grid = np.linspace(q_lo, q_hi, 200)
    pal = sns.color_palette('mako', n_splits)

    # --- Ridge plot (always per-split) ---
    ax_ridge = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(0)
    ridge_scale = 0.9
    for i, s in enumerate(splits):
        sub = df[df[split_col] == s]
        if len(sub) < 2:
            continue
        if len(sub) > kde_max_n:
            sub = sub.iloc[rng.choice(len(sub), kde_max_n, replace=False)]
        try:
            kde_ref = gaussian_kde(sub[ref_col].to_numpy())
            kde_pred = gaussian_kde(sub[pred_col].to_numpy())
        except Exception:
            continue
        y_ref = kde_ref(x_grid)
        y_pred = kde_pred(x_grid)
        peak = max(y_ref.max(), y_pred.max())
        if peak <= 0:
            continue
        y_ref = y_ref / peak * ridge_scale
        y_pred = y_pred / peak * ridge_scale
        ax_ridge.fill_between(x_grid, i, i + y_pred, color=pal[i], alpha=0.55, lw=0)
        ax_ridge.plot(x_grid, i + y_ref, color='black', lw=1.2)
        if s in split_metrics.index:
            w = split_metrics.loc[s, 'wasserstein']
            ax_ridge.text(
                q_hi, i + 0.45, f'  W={w:.3f}',
                va='center', ha='left', fontsize=10,
            )
    ax_ridge.set_yticks(np.arange(n_splits) + 0.4)
    ax_ridge.set_yticklabels(splits)
    ax_ridge.set_xlim(q_lo, q_hi + 0.18 * (q_hi - q_lo))
    ax_ridge.set_xlabel('efficiency')
    ax_ridge.set_title(f'{pred_col} (filled) vs {ref_col} (line) density per split')
    sns.despine(ax=ax_ridge)

    # --- Hexbin scatters ---
    def _draw_hex(ax, sub: pd.DataFrame, title_prefix: str, m: dict | None, show_y: bool):
        if len(sub) < 2 or m is None:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title_prefix)
            return
        ax.hexbin(
            sub[ref_col].to_numpy(), sub[pred_col].to_numpy(),
            gridsize=bins, mincnt=1, bins='log', cmap='magma',
            extent=(q_lo, q_hi, q_lo, q_hi),
        )
        ax.plot([q_lo, q_hi], [q_lo, q_hi], color='cyan', ls='--', lw=1, alpha=0.6)
        ax.set_xlim(q_lo, q_hi)
        ax.set_ylim(q_lo, q_hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'reference {ref_col}')
        if show_y:
            ax.set_ylabel(f'predicted {pred_col}')
        ax.set_title(
            f"{title_prefix}  ρ={m['spearman']:.3f}  r={m['pearson']:.3f}  "
            f"R²={m['r2']:.2f}",
            fontsize=9,
        )

    if dataset_col is None:
        for i, s in enumerate(splits):
            ax = fig.add_subplot(gs[1, i])
            sub = df[df[split_col] == s]
            m = split_metrics.loc[s].to_dict() if s in split_metrics.index else None
            _draw_hex(ax, sub, s, m, show_y=(i == 0))
    else:
        for j, d in enumerate(datasets):
            for i, s in enumerate(splits):
                ax = fig.add_subplot(gs[1 + j, i])
                sub = df[(df[split_col] == s) & (df[dataset_col] == d)]
                key = (s, d)
                m = grid_metrics.loc[key].to_dict() if (
                    grid_metrics is not None and key in grid_metrics.index
                ) else None
                title = f"{s} / {d}"
                _draw_hex(ax, sub, title, m, show_y=(i == 0))

    fig.tight_layout()
    if plt_file:
        fig.savefig(plt_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
    elif return_fig:
        # Caller takes ownership of closing the figure (e.g. TensorBoard
        # logging). Skip plt.show() so non-interactive backends don't drop it.
        pass
    else:
        plt.show()
    metrics = grid_metrics if dataset_col is not None else split_metrics
    if return_fig:
        return fig, metrics
    return metrics


#: Okabe-Ito 8-color colorblind-friendly palette (de facto publication standard).
#: Skip black (0) and yellow (4) for default cell coloring — both are hard to read on white.
OKABE_ITO = [
    '#000000',  # 0  black
    '#E69F00',  # 1  orange
    '#56B4E9',  # 2  sky blue
    '#009E73',  # 3  bluish green
    '#F0E442',  # 4  yellow
    '#0072B2',  # 5  blue
    '#D55E00',  # 6  vermillion
    '#CC79A7',  # 7  reddish purple
]


def plot_perturbation_enrichment(
    adata: ad.AnnData,
    condition_col: str,
    soft_predictions_key: str = 'soft_predictions',
    top_k: int = 50,
    cluster: bool = True,
    cmap: str | None = None,
    z_clip: float = 3.0,
    color_norm: Literal['linear', 'log', 'col_minmax', 'col_zscore'] = 'linear',
    figsize: tuple[float, float] | None = None,
    library: str | list[str] | None = None,
    organism: str = 'Human',
    score: Literal['z_score', 'mean', 'frequency', 'mean_freq', 'mean_or_freq'] = 'z_score',
    top_per_cell: int = 1,
    select: Literal['union', 'max'] = 'union',
    output_file: str | None = None,
) -> pd.DataFrame:
    """Per-condition perturbation enrichment heatmap (Level 2 — soft profiles).

    For a queried adata, computes the mean soft-prediction profile per
    condition group, z-scores each perturbation across groups, and
    renders a heatmap of the top-K most variable perturbations. Useful
    for asking "which predicted perturbations are enriched in which
    experimental condition?" — far less noisy than argmax-based
    enrichment because it uses the full softmax mass.

    Parameters
    ----------
    adata
        Query adata that has been through ``model.predict(inplace=True)``
        (or ``model.evaluate()``). Must contain
        ``adata.obsm[soft_predictions_key]``.
    condition_col
        Column in ``adata.obs`` defining the groups (e.g. ``'cell_type'``,
        ``'treatment'``, ``'time_point'``). Required.
    top_k
        Number of perturbations to show, ranked by max absolute
        z-score across groups (most variable per-group profiles).
    cluster
        If True, use ``sns.clustermap`` (rows + columns hierarchically
        clustered). Otherwise a plain ``sns.heatmap`` with original order.
    z_clip
        Symmetric color limit for the z-score heatmap. Ignored when
        ``score != 'z_score'``.
    library
        Optional gseapy/Enrichr library name (or list of names) to
        annotate the top perturbations with pathway memberships, e.g.
        ``'MSigDB_Hallmark_2020'`` or ``['MSigDB_Hallmark_2020',
        'Reactome_Pathways_2024']``. When provided, the returned
        DataFrame's ``attrs['pathways']`` carries a long-form
        ``(perturbation, pathway, library)`` mapping for downstream
        target-prioritization. No-op when ``None``.
    organism
        Passed through to ``get_library`` when ``library`` is given.
    score
        How to score perturbations per group:

        - ``'z_score'`` (default) — z-score the per-group mean soft
          profile across groups. Highlights *differential* drivers but
          requires ≥2 groups.
        - ``'mean'`` — per-group mean soft probability. No control
          group needed; works for a single group. Identifies
          consistently-predicted perts regardless of cross-group
          contrast.
        - ``'frequency'`` — per-group fraction of cells where the
          perturbation falls in the top-``top_per_cell`` predictions.
          Argmax-style "vote counting"; robust to soft-score
          calibration. Works for a single group.
    top_per_cell
        Only used when ``score='frequency'``: number of top predictions
        to count per cell. Default 1 = pure argmax. Use 5 or 10 to
        relax the count to top-K membership.
    select
        How the displayed perturbation set is chosen from the
        per-condition score matrix:

        - ``'union'`` (default) — take each condition's *own* top-``top_k``
          and union them. Every condition is ranked against itself, so a
          high-magnitude condition cannot crowd out another condition's
          characteristic perturbations. The displayed set holds between
          ``top_k`` and ``n_groups × top_k`` perturbations.
        - ``'max'`` — legacy behaviour: the ``top_k`` perturbations with
          the single highest cross-condition value. Biases the set toward
          whichever condition sits on the largest overall magnitude.
    cmap
        Override colormap. Defaults: ``'RdBu_r'`` (diverging) for
        ``z_score`` or ``col_zscore``, ``'viridis'`` (sequential)
        otherwise.
    color_norm
        How values get mapped to colors (parallel to
        ``plot_pathway_enrichment``'s ``color_norm``). Note the matrix
        here is rows=groups, cols=perturbations, so "per-feature"
        normalization runs along axis 0 — hence the ``col_*`` names:

        - ``'linear'`` (default) — direct mapping, clipped by ``z_clip``.
        - ``'log'`` — ``LogNorm``. Useful when mean / frequency values
          span orders of magnitude.
        - ``'col_minmax'`` — per-perturbation min-max rescaled to
          ``[0, 1]`` across groups. Each pert's color spans its
          per-group min→max regardless of absolute level. Maximizes
          within-pert contrast.
        - ``'col_zscore'`` — per-perturbation z-score across groups.
          Signed, diverging-friendly. Best for differential views
          when frequency / mean values sit near the noise floor.

    Returns
    -------
    pd.DataFrame
        Z-score matrix: rows = condition groups, cols = top-K perturbations.
        Same data the heatmap shows; useful for follow-up tables.

        When ``library`` is supplied, ``df.attrs['pathways']`` holds a
        long-form DataFrame with columns
        ``[perturbation, pathway, library]`` linking every top
        perturbation to the pathways it belongs to.
    """
    # --- Validate inputs ---
    if condition_col not in adata.obs:
        raise KeyError(f'condition_col {condition_col!r} not in adata.obs.')
    if soft_predictions_key not in adata.obsm:
        raise KeyError(
            f'{soft_predictions_key!r} not in adata.obsm — adata does not look '
            f'like the output of model.predict() / model.evaluate(). '
            f'Available obsm keys: {list(adata.obsm.keys())}'
        )
    # Sanity check: does adata also carry ExPert latent keys? Indicates a
    # proper queried adata. Not required, just an info-level signal.
    expected_latent_keys = [k for k in adata.obsm.keys() if 'latent_z' in k]
    if not expected_latent_keys:
        log.info(
            f'[Enrichment] Note: no ExPert latent (X_latent_z*) key found in '
            f'obsm. soft_predictions_key={soft_predictions_key!r} present, '
            f'so proceeding — but adata may not be a fully queried ExPert output.'
        )

    # --- Load softmax + groups ---
    soft = adata.obsm[soft_predictions_key]
    if hasattr(soft, 'values') and hasattr(soft, 'columns'):
        # DataFrame in obsm — preserves perturbation column names.
        pert_names = np.asarray(soft.columns).astype(str)
        soft_arr = np.asarray(soft.values)
    else:
        soft_arr = np.asarray(soft)
        pert_names = np.arange(soft_arr.shape[1]).astype(str)
    n_cells, n_perts = soft_arr.shape

    groups = adata.obs[condition_col].astype(str).values
    unique_groups = pd.unique(groups)
    if score == 'z_score' and len(unique_groups) < 2:
        raise ValueError(
            f'score={score!r} needs ≥2 groups in {condition_col!r}; '
            f"found {len(unique_groups)}: {list(unique_groups)}. "
            f"Pass score='mean' or 'frequency' for single-group use."
        )

    n_groups = len(unique_groups)
    # All scoring modes build a (n_groups, n_perts) matrix; the
    # semantics differ but downstream top-K / heatmap code is shared.
    # mean_freq needs BOTH the mean profile AND the frequency profile,
    # so compute both unconditionally when that mode is requested.
    needs_mean = score in ('z_score', 'mean', 'mean_freq', 'mean_or_freq')
    needs_freq = score in ('frequency', 'mean_freq', 'mean_or_freq')
    profiles_mean = np.zeros((n_groups, n_perts), dtype=np.float32) if needs_mean else None
    profiles_freq = np.zeros((n_groups, n_perts), dtype=np.float32) if needs_freq else None
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    for i, g in enumerate(unique_groups):
        mask = groups == g
        n_in = int(mask.sum())
        group_sizes[i] = n_in
        if n_in == 0:
            continue
        sub = soft_arr[mask]
        if needs_mean:
            profiles_mean[i] = sub.astype(np.float32).mean(axis=0)
        if needs_freq:
            tpc = max(1, min(int(top_per_cell), n_perts))
            if tpc == 1:
                top_idx_cell = sub.argmax(axis=1)
            else:
                top_idx_cell = np.argpartition(sub, -tpc, axis=1)[:, -tpc:]
            counts = np.bincount(np.asarray(top_idx_cell).ravel(),
                                 minlength=n_perts)
            profiles_freq[i] = counts.astype(np.float32) / float(n_in)

    # --- Score matrix per mode ---
    if score == 'z_score':
        mean_per_pert = profiles_mean.mean(axis=0)
        std_per_pert = profiles_mean.std(axis=0)
        std_per_pert = np.where(std_per_pert > 1e-9, std_per_pert, 1.0)
        score_mat = (profiles_mean - mean_per_pert) / std_per_pert
        rank_metric = np.abs(score_mat).max(axis=0)
    elif score == 'mean':
        score_mat = profiles_mean
        rank_metric = score_mat.max(axis=0)
    elif score == 'frequency':
        score_mat = profiles_freq
        rank_metric = score_mat.max(axis=0)
    elif score == 'mean_freq':
        # Geometric mean of within-group percentile ranks. For each
        # (group, pert): take its rank within the group under mean and
        # under frequency, normalize to percentile in [0,1], take
        # sqrt(p_mean * p_freq). Rewards perts that are jointly high in
        # both modes; penalizes perts strong in only one. Scale-invariant
        # (handles the mean ~1e-3 vs freq ~1e-2 magnitude mismatch).
        from scipy.stats import rankdata
        score_mat = np.zeros_like(profiles_mean)
        denom = float(max(n_perts, 1))
        for i in range(n_groups):
            r_m = rankdata(profiles_mean[i], method='average') / denom
            r_f = rankdata(profiles_freq[i], method='average') / denom
            score_mat[i] = np.sqrt(r_m * r_f).astype(np.float32)
        rank_metric = score_mat.max(axis=0)
    elif score == 'mean_or_freq':
        # Probabilistic OR of within-group percentile ranks:
        #   score = 1 - (1 - P_mean)·(1 - P_freq)
        # Discovery-oriented complement to mean_freq: perts strong in
        # *either* mode survive into the top-K while joint-strong perts
        # still rank highest. Lets the splicing axis (mean-driven) and
        # the WNT/bile-acid axis (frequency-driven) both come through
        # in a single ranking. Result is bounded in [0, 1].
        from scipy.stats import rankdata
        score_mat = np.zeros_like(profiles_mean)
        denom = float(max(n_perts, 1))
        for i in range(n_groups):
            r_m = rankdata(profiles_mean[i], method='average') / denom
            r_f = rankdata(profiles_freq[i], method='average') / denom
            score_mat[i] = (1.0 - (1.0 - r_m) * (1.0 - r_f)).astype(np.float32)
        rank_metric = score_mat.max(axis=0)
    else:
        raise ValueError(
            f"Unknown score={score!r}. Use 'z_score', 'mean', "
            f"'frequency', 'mean_freq', or 'mean_or_freq'."
        )

    # --- Top-K most discriminative / most frequent perts ---
    # select='union' (default): each condition contributes its OWN top-k, then
    #   union — so a high-magnitude condition can't crowd out another condition's
    #   characteristic perturbations (the union holds k..n_groups*k perts).
    # select='max': legacy — the k perts with the single highest cross-condition
    #   value (rank_metric), which biases the set toward the loudest condition.
    k = min(top_k, n_perts)
    if select == 'union':
        sel: set[int] = set()
        for i in range(n_groups):
            sel.update(int(j) for j in np.argsort(score_mat[i])[-k:])
        # order the union by the cross-condition metric for stable, readable display
        top_idx = np.array(sorted(sel, key=lambda j: rank_metric[j], reverse=True))
    elif select == 'max':
        top_idx = np.argsort(rank_metric)[-k:][::-1]
    else:
        raise ValueError(f"Unknown select={select!r}. Use 'union' or 'max'.")
    score_top = score_mat[:, top_idx]
    top_pert_names = pert_names[top_idx]

    df = pd.DataFrame(
        score_top, index=unique_groups, columns=top_pert_names,
    )
    df.index.name = condition_col

    # --- Optional pathway membership annotation ---
    # Fetched once per library via the same gseapy/Enrichr fallback used
    # by plot_pathway_enrichment. Stored as a long-form DataFrame so the
    # caller can groupby('perturbation') to get a wide table or merge it
    # back onto a top-pert ranking for target prioritization.
    if library is not None:
        from expert.utils.performance import get_library
        libs = [library] if isinstance(library, str) else list(library)
        top_pert_upper_to_orig = {p.upper(): p for p in top_pert_names}
        path_rows = []
        for lib_name in libs:
            log.info(f'[Enrichment] Annotating top perts with {lib_name} ({organism}).')
            lib = get_library(library=lib_name, organism=organism)
            if lib is None:
                log.warning(f'[Enrichment] Could not load {lib_name}; skipping.')
                continue
            for pw, members in lib.items():
                for m in members:
                    orig = top_pert_upper_to_orig.get(str(m).upper())
                    if orig is not None:
                        path_rows.append({
                            'perturbation': orig,
                            'pathway': pw,
                            'library': lib_name,
                        })
        df.attrs['pathways'] = pd.DataFrame(
            path_rows,
            columns=['perturbation', 'pathway', 'library'],
        )

    # --- Plot ---
    if figsize is None:
        figsize = (max(10.0, 0.25 * k), max(4.0, 0.4 * len(unique_groups)))

    # Per-mode default cbar label.
    base_cbar = {
        'z_score':   'z-score (group profile)',
        'mean':      'mean soft-prediction (per group)',
        'frequency': (f'fraction of cells with pert in top-{top_per_cell}'
                      if top_per_cell > 1 else 'fraction of cells (argmax)'),
        'mean_freq': '√(percentile_mean · percentile_freq)',
        'mean_or_freq': '1 − (1−P_mean)·(1−P_freq)',
    }[score]

    # `df` (rows=groups, cols=perts) is the data we hand seaborn. For
    # row-style transforms we plot a transformed copy and update the
    # cbar label to make it explicit. linear-modes leave df alone.
    plot_df = df
    cbar_extra = ''

    if color_norm == 'log':
        from matplotlib.colors import LogNorm
        hm_cmap = cmap if cmap is not None else 'viridis'
        pos = df.values[df.values > 0]
        if pos.size == 0:
            hm_kwargs = dict(cmap=hm_cmap, vmin=0.0, vmax=1.0)
        else:
            vmin_log = float(pos.min())
            # Robust upper bound — 98th percentile of positive values
            # so a single dominant pert can't peg the colormap. Caller
            # can still override via z_clip.
            if z_clip is not None:
                vmax_log = float(z_clip)
            else:
                vmax_log = float(np.percentile(pos, 98))
                if vmax_log <= vmin_log:
                    vmax_log = float(pos.max())
            plot_df = df.where(df > 0, vmin_log * 0.5)
            hm_kwargs = dict(cmap=hm_cmap, norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
            cbar_extra = ' (log scale)'
    elif color_norm == 'col_minmax':
        hm_cmap = cmap if cmap is not None else 'viridis'
        mn = df.min(axis=0)
        mx = df.max(axis=0)
        rng = (mx - mn).where(lambda v: v.abs() > 1e-12, 1.0)
        plot_df = (df - mn) / rng
        hm_kwargs = dict(cmap=hm_cmap, vmin=0.0, vmax=1.0)
        cbar_extra = ' (per-pert 0–1)'
    elif color_norm == 'col_zscore':
        hm_cmap = cmap if cmap is not None else 'RdBu_r'
        mu = df.mean(axis=0)
        sd = df.std(axis=0).where(lambda v: v.abs() > 1e-12, 1.0)
        plot_df = (df - mu) / sd
        # With small n_groups the theoretical |z|max ≈ √(n-1)/n × ...;
        # clamp at z_clip if user provided one, else √2 (3-group case).
        clip_val = float(z_clip) if z_clip is not None else float(np.sqrt(2.0))
        hm_kwargs = dict(cmap=hm_cmap, center=0, vmin=-clip_val, vmax=clip_val)
        cbar_extra = ' (per-pert z)'
    elif score == 'z_score':
        # Linear, diverging — original z-score path.
        hm_cmap = cmap if cmap is not None else 'RdBu_r'
        hm_kwargs = dict(cmap=hm_cmap, center=0, vmin=-z_clip, vmax=z_clip)
    elif score == 'mean':
        # Linear, non-negative — original mean path.
        hm_cmap = cmap if cmap is not None else 'viridis'
        hm_kwargs = dict(cmap=hm_cmap, vmin=0.0)
    elif score == 'frequency':
        hm_cmap = cmap if cmap is not None else 'viridis'
        hm_kwargs = dict(
            cmap=hm_cmap, vmin=0.0,
            vmax=min(1.0, float(score_top.max()) or 1.0),
        )
    else:  # 'mean_freq' or 'mean_or_freq' — both in [0, 1]
        hm_cmap = cmap if cmap is not None else 'viridis'
        hm_kwargs = dict(cmap=hm_cmap, vmin=0.0, vmax=1.0)

    cbar_label = base_cbar + cbar_extra

    if cluster:
        # clustermap creates its own figure; pass figsize directly.
        cg = sns.clustermap(
            plot_df,
            figsize=figsize,
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': cbar_label},
            **hm_kwargs,
        )
        cg.ax_heatmap.set_xlabel('perturbation')
        cg.ax_heatmap.set_ylabel(condition_col)
        # Rotate x labels for readability.
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
        plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
        fig = cg.figure
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            plot_df, ax=ax,
            xticklabels=True, yticklabels=True,
            cbar_kws={'label': cbar_label},
            **hm_kwargs,
        )
        ax.set_xlabel('perturbation')
        ax.set_ylabel(condition_col)
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle(
        f'Top-{k} perturbations by group enrichment '
        f'({len(unique_groups)} groups, {n_cells:,} cells; score={score})',
        y=1.02 if cluster else 0.98,
        fontsize=11,
    )

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return df


def plot_pathway_enrichment(
    adata: ad.AnnData,
    condition_col: str,
    library: str = 'MSigDB_Hallmark_2020',
    organism: str = 'Human',
    soft_predictions_key: str = 'soft_predictions',
    top_k_pathways: int = 30,
    min_pathway_size: int = 2,
    score: Literal['mean_z', 'wilcoxon', 'mean', 'frequency', 'mean_freq', 'mean_or_freq'] = 'mean',
    top_per_cell: int = 1,
    cluster: bool = True,
    transpose: bool = False,
    show_dendrograms_row: bool = True,
    show_dendrograms_col: bool = True,
    cmap: str | None = None,
    z_clip: float | None = None,
    color_norm: Literal['linear', 'log', 'row_minmax', 'row_zscore'] = 'linear',
    size_min: float = 60.0,
    size_max: float = 800.0,
    pvalue_cap: float = 10.0,
    top_k_metric: Literal['max_abs', 'std', 'range', 'vs_baseline'] = 'max_abs',
    baseline_group: str | None = None,
    figsize: tuple[float, float] | None = None,
    title_fontsize: float = 12.0,
    label_fontsize: float = 12.0,
    dpi: float = 150,
    output_file: str | None = None,
) -> pd.DataFrame:
    """Per-condition pathway enrichment dot-plot.

    Aggregates per-perturbation z-scores (same computation as
    ``plot_perturbation_enrichment``) into pathway-level scores using
    a gseapy library (KEGG / Reactome / GO / MSigDB / etc.) and
    renders a publication-quality dot plot. Each dot's **color**
    encodes the per-group enrichment score and its **size** encodes
    Wilcoxon -log10(p) — so visually striking points are *both* large
    and saturated, signaling pathways that are simultaneously
    significant and strongly enriched in a group.

    Parameters
    ----------
    adata, condition_col, soft_predictions_key
        Same as ``plot_perturbation_enrichment``.
    library, organism
        gseapy / Enrichr library name and organism.
    top_k_pathways
        Number of pathways to display, ranked by max absolute group
        score (most discriminative pathways).
    min_pathway_size
        Drop pathways with fewer than this many of *your* perturbation
        labels — small overlaps give unstable scores.
    top_k_metric
        How to rank pathways when picking the top-K for display.
        Independent of ``score`` mode:

        - ``'max_abs'`` (default) — largest |score| in any group.
          Picks pathways with strong absolute prominence; in
          ``frequency`` / ``mean`` modes this can surface pathways
          that are equally high across all groups.
        - ``'std'`` — std across groups. Differential ranking —
          picks pathways that vary between conditions regardless of
          direction.
        - ``'range'`` — max − min across groups. Like ``std`` but
          extreme-sensitive.
        - ``'vs_baseline'`` — max over non-baseline groups − baseline
          value. Requires ``baseline_group``. Biases top-K toward
          pathways that are *high in the disease groups but absent in
          the baseline*. The natural choice for "show me disease-
          specific pathways the model picks up."
    baseline_group
        Name of the control / baseline condition for
        ``top_k_metric='vs_baseline'`` (e.g. ``'Normal'``). Must
        match one of the values in ``adata.obs[condition_col]``.
    score
        How to aggregate per-group per-pert signal into a pathway
        score. The four modes parallel
        ``plot_perturbation_enrichment`` so you can ask the same
        biological question at the pathway level:

        - ``'mean_z'`` — mean *z-score* across pathway members.
          Differential mode; requires ≥2 groups.
        - ``'wilcoxon'`` — signed -log10(p-value) of the Wilcoxon
          rank-sum test of pathway members vs non-members on the
          z-scored profile. Also differential.
        - ``'mean'`` — mean per-group soft probability across
          pathway members. *Absolute* / proximity mode — works for
          a single group; doesn't need a control. Answers "which
          pathways do these cells *look like* a perturbation of?"
          rather than "which pathways differ from controls?".
        - ``'frequency'`` — for each cell, count whether any pathway
          member is in the cell's top-``top_per_cell`` predictions,
          then take the per-group fraction averaged over members.
          Argmax-style proximity. Also single-group capable.

        Marker SIZE is always derived from the Wilcoxon p-value of
        pathway-member vs non-member values *under the chosen
        score mode*, so a "significantly enriched" dot is the
        cohort-vs-rest test computed on whichever signal is being
        coloured.
    top_per_cell
        Only used when ``score='frequency'``: number of top
        predictions per cell to count. Default 1 = pure argmax.
    cluster
        Reorder groups (rows) and pathways (cols) by hierarchical
        clustering of the score matrix.
    transpose
        Rotate the plot so pathways are on the y-axis and groups on
        the x-axis. The returned DataFrame is transposed to match.
        ``show_dendrograms_row``/``show_dendrograms_col`` still refer
        to the displayed left/top dendrograms; the underlying linkage
        used for each is swapped automatically.
    show_dendrograms
        Draw small dendrograms above/left of the dot panel when
        ``cluster=True``. Ignored when ``cluster=False``.
    z_clip
        Color limit, mode-aware. When ``None`` (default), sensible
        per-mode defaults are picked:

        - ``mean_z``    → ±2.0 (typical 3-group |z| range)
        - ``wilcoxon``  → ±5.0 (signed -log10p often goes higher)
        - ``mean`` / ``frequency`` → auto-scale to the 98th percentile
          of the displayed matrix (robust to one saturated outlier)

        Pass an explicit float to override. For diverging modes the
        clip is symmetric around 0; for non-negative modes it caps the
        upper end of the sequential cmap.
    cmap
        Override colormap. Defaults: ``'RdBu_r'`` (diverging) for
        z-scored modes, ``'viridis'`` (sequential) for ``mean`` /
        ``frequency``. ``'row_zscore'`` color_norm switches the
        default to ``'RdBu_r'`` regardless of ``score`` mode.
    color_norm
        How raw pathway scores get mapped to colors:

        - ``'linear'`` (default) — direct linear mapping, clipped to
          ``z_clip`` (or auto-scaled). Best for z-score modes.
        - ``'log'`` — ``LogNorm`` (matplotlib). Handles dynamic
          range when frequency / mean values span orders of magnitude.
          Drops non-positive values from the color scale; pair with a
          sequential cmap.
        - ``'row_minmax'`` — per-pathway min-max rescaled to ``[0,1]``
          *across groups*. Each pathway's color spectrum runs from its
          minimum-group to its maximum-group regardless of absolute
          magnitude. Maximizes within-pathway contrast; sacrifices
          cross-pathway comparability.
        - ``'row_zscore'`` — per-pathway z-score across groups. Signed,
          diverging-cmap-friendly. "Which group does this pathway
          assign most to vs. the others?". Best for differential views
          where the absolute frequency is noise-floor-low.
    size_min, size_max
        Marker area (matplotlib points²) for the smallest vs. largest
        -log10(p) in the *displayed* matrix.
    pvalue_cap
        Cap -log10(p) at this value before scaling to size — avoids
        one extremely-significant point swamping the legend.

    Returns
    -------
    pd.DataFrame
        Pathway score matrix: rows=groups (in plot order),
        cols=top-K pathways (in plot order). Suitable for table
        dumps. Additional outputs attached via ``df.attrs``:

        - ``df.attrs['pvalues']`` — Wilcoxon p-values, same shape/index.
        - ``df.attrs['members']`` — long-form DataFrame mapping every
          displayed pathway to its member perturbations. Columns:
          ``[<condition_col>, pathway, pathway_label, perturbation,
          pert_z, pathway_score, pathway_pvalue]``. Rows are sorted
          by descending |pert_z| within each (group, pathway) block,
          so the strongest driver perturbations come first.
    """
    from scipy.stats import mannwhitneyu
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

    # --- Reuse the per-pert z-score computation from level 2 ---
    if condition_col not in adata.obs:
        raise KeyError(f'condition_col {condition_col!r} not in adata.obs.')
    if soft_predictions_key not in adata.obsm:
        raise KeyError(f'{soft_predictions_key!r} not in adata.obsm.')

    soft = adata.obsm[soft_predictions_key]
    if hasattr(soft, 'columns') and hasattr(soft, 'values'):
        pert_names = np.asarray(soft.columns).astype(str)
        soft_arr = np.asarray(soft.values)
    else:
        soft_arr = np.asarray(soft)
        pert_names = np.arange(soft_arr.shape[1]).astype(str)

    groups = adata.obs[condition_col].astype(str).values
    unique_groups = pd.unique(groups)
    n_groups = len(unique_groups)
    n_perts_full = soft_arr.shape[1]
    diverging_modes = ('mean_z', 'wilcoxon')
    if score in diverging_modes and n_groups < 2:
        raise ValueError(
            f'score={score!r} needs ≥2 groups in {condition_col!r}; '
            f'found {n_groups}: {list(unique_groups)}. Pass '
            f"score='mean' or 'frequency' for single-group use."
        )

    # Per-group mean soft profile (used by mean_z, mean; basis for z).
    profiles = np.zeros((n_groups, n_perts_full), dtype=np.float32)
    for i, g in enumerate(unique_groups):
        mask = groups == g
        if not mask.any():
            continue
        profiles[i] = soft_arr[mask].astype(np.float32).mean(axis=0)

    # Per-pert per-group score matrix, chosen by `score` mode. Same role
    # `z` plays in the differential modes; downstream code (pathway
    # aggregation, Wilcoxon p-values for marker sizing, top-K
    # selection, clustering) all operate on `pert_score_matrix` uniformly.
    def _compute_freq_mat() -> np.ndarray:
        tpc = max(1, min(int(top_per_cell), n_perts_full))
        out = np.zeros((n_groups, n_perts_full), dtype=np.float32)
        for i, g in enumerate(unique_groups):
            mask = groups == g
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            sub = soft_arr[mask]
            if tpc == 1:
                top_idx_cell = sub.argmax(axis=1)
            else:
                top_idx_cell = np.argpartition(sub, -tpc, axis=1)[:, -tpc:]
            counts = np.bincount(np.asarray(top_idx_cell).ravel(),
                                 minlength=n_perts_full)
            out[i] = counts.astype(np.float32) / float(n_in)
        return out

    if score in diverging_modes:
        mu = profiles.mean(axis=0)
        sd = profiles.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        pert_score_matrix = (profiles - mu) / sd
    elif score == 'mean':
        pert_score_matrix = profiles
    elif score == 'frequency':
        pert_score_matrix = _compute_freq_mat()
    elif score == 'mean_freq':
        # Geometric mean of within-group percentile ranks under mean
        # and frequency. Captures the joint biology — perts strong in
        # both broad commitment (mean) and decisive picks (frequency).
        # Scale-invariant + bounded in [0, 1]. See
        # plot_perturbation_enrichment for the same logic at the
        # pert-display level.
        from scipy.stats import rankdata
        freq_mat = _compute_freq_mat()
        denom = float(max(n_perts_full, 1))
        pert_score_matrix = np.zeros_like(profiles)
        for i in range(n_groups):
            r_m = rankdata(profiles[i], method='average') / denom
            r_f = rankdata(freq_mat[i], method='average') / denom
            pert_score_matrix[i] = np.sqrt(r_m * r_f).astype(np.float32)
    elif score == 'mean_or_freq':
        # Probabilistic OR of within-group percentile ranks — union
        # complement to ``mean_freq``. Perts strong in *either* mode
        # survive into the top-K while joint-strong perts still rank
        # highest. Use when discovery (catching unique signal from
        # both modes) matters more than consensus.
        from scipy.stats import rankdata
        freq_mat = _compute_freq_mat()
        denom = float(max(n_perts_full, 1))
        pert_score_matrix = np.zeros_like(profiles)
        for i in range(n_groups):
            r_m = rankdata(profiles[i], method='average') / denom
            r_f = rankdata(freq_mat[i], method='average') / denom
            pert_score_matrix[i] = (1.0 - (1.0 - r_m) * (1.0 - r_f)).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown score: {score!r}. Use one of "
            f"'mean_z' / 'wilcoxon' / 'mean' / 'frequency' / "
            f"'mean_freq' / 'mean_or_freq'."
        )

    # --- Load pathway library and build pert -> pathway map ---
    from expert.utils.performance import get_library
    log.info(f'[Pathways] Loading {library} ({organism}) via gseapy.')
    lib = get_library(library=library, organism=organism)
    if lib is None:
        if 'reactome' in library.lower() and library != 'Reactome_Pathways_2024':
            log.warning(
                f"[Pathways] {library!r} failed; falling back to "
                "'Reactome_Pathways_2024'."
            )
            lib = get_library(library='Reactome_Pathways_2024', organism=organism)
            if lib is not None:
                library = 'Reactome_Pathways_2024'
        if lib is None:
            raise RuntimeError(
                f'Failed to load gseapy library {library!r} for {organism!r}. '
                'See printed alternatives above and pass one of them as '
                '`library=...`. Common working names: '
                "'MSigDB_Hallmark_2020', 'KEGG_2021_Human', "
                "'Reactome_Pathways_2024', 'WikiPathway_2024_Human', "
                "'GO_Biological_Process_2023'."
            )

    pert_set = {p.upper() for p in pert_names}
    pert_upper = np.asarray([p.upper() for p in pert_names])
    pert_to_idx = {p: i for i, p in enumerate(pert_upper)}

    pathway_members: dict[str, np.ndarray] = {}
    for pw, members in lib.items():
        idx = [pert_to_idx[m.upper()] for m in members if m.upper() in pert_set]
        if len(idx) >= min_pathway_size:
            pathway_members[pw] = np.asarray(idx, dtype=np.int64)
    if not pathway_members:
        raise RuntimeError(
            f'No pathways with ≥{min_pathway_size} members found in {library!r}. '
            'Lower min_pathway_size or pick a more specific library.'
        )
    log.info(
        f'[Pathways] {len(pathway_members)} pathways pass min_size={min_pathway_size} '
        f'(out of {len(lib)} in {library}).'
    )

    # --- Per-pathway color score AND p-value (Wilcoxon for marker size) ---
    # Performance: for large libraries (Reactome/GO ~2-3k pathways), the
    # original implementation ran mannwhitneyu for *every* pathway × group
    # — 7-9k slow scipy calls, plus np.setdiff1d per pathway. The
    # restructured version below:
    #   (a) computes pathway scores cheaply for all pathways in all
    #       non-wilcoxon modes,
    #   (b) defers Wilcoxon to ONLY the top-K pathways once the selection
    #       is known — typically 30 instead of 2500 (~80× speedup),
    #   (c) replaces np.setdiff1d with a boolean mask,
    #   (d) uses ``method='asymptotic'`` for faster mannwhitneyu when
    #       Wilcoxon is needed for all pathways (wilcoxon score mode).
    pathways = list(pathway_members.keys())
    n_pw = len(pathways)
    n_perts_pw = pert_score_matrix.shape[1]
    pw_score = np.zeros((n_groups, n_pw), dtype=np.float32)
    pw_pval = np.ones((n_groups, n_pw), dtype=np.float64)

    def _wilcoxon_for_pathway(j: int):
        """Fill ``pw_pval[:, j]`` (and ``pw_score`` for wilcoxon mode)
        for the single pathway at column ``j``. Boolean-mask membership
        is built once per pathway and reused across groups."""
        members = pathway_members[pathways[j]]
        mask = np.zeros(n_perts_pw, dtype=bool)
        mask[members] = True
        for i in range(n_groups):
            in_v = pert_score_matrix[i, mask]
            out_v = pert_score_matrix[i, ~mask]
            if in_v.size == 0 or out_v.size == 0:
                continue
            try:
                _, pval = mannwhitneyu(
                    in_v, out_v, alternative='two-sided',
                    method='asymptotic',
                )
            except ValueError:
                pval = 1.0
            pw_pval[i, j] = pval
            if score == 'wilcoxon':
                signed = np.sign(in_v.mean() - out_v.mean())
                pw_score[i, j] = signed * (-np.log10(max(pval, 1e-300)))

    if score in ('mean_z', 'mean', 'frequency', 'mean_freq', 'mean_or_freq'):
        # Pathway score = mean of pert scores for members — cheap,
        # vectorized per pathway. p-values deferred to top-K only.
        for j, pw in enumerate(pathways):
            members = pathway_members[pw]
            pw_score[:, j] = pert_score_matrix[:, members].mean(axis=1)
    elif score == 'wilcoxon':
        # p-values ARE the score — must compute for all pathways now so
        # we can rank by them. Asymptotic Wilcoxon is much faster than
        # the default 'auto' which uses the exact test for small samples.
        for j in range(n_pw):
            _wilcoxon_for_pathway(j)

    # --- Top-K pathways under chosen ranking metric ---
    # max_abs (default): largest |score| in any group — picks pathways
    #   that are strong in at least one group (absolute prominence).
    # std: largest std across groups — picks differential pathways
    #   regardless of which group is highest.
    # range: max − min across groups — like std but extreme-sensitive.
    # vs_baseline: max over non-baseline groups − baseline value, sorted
    #   descending. Biases toward "high in disease groups, absent in
    #   the baseline" — useful for filtering out pathways that are
    #   similarly high in the control.
    if top_k_metric == 'max_abs':
        rank_metric = np.abs(pw_score).max(axis=0)
    elif top_k_metric == 'std':
        rank_metric = pw_score.std(axis=0)
    elif top_k_metric == 'range':
        rank_metric = pw_score.max(axis=0) - pw_score.min(axis=0)
    elif top_k_metric == 'vs_baseline':
        if baseline_group is None:
            raise ValueError(
                "top_k_metric='vs_baseline' requires baseline_group "
                "(e.g. baseline_group='Normal')."
            )
        groups_list = list(unique_groups)
        if baseline_group not in groups_list:
            raise ValueError(
                f"baseline_group={baseline_group!r} not in groups "
                f"{groups_list}."
            )
        b_idx = groups_list.index(baseline_group)
        other_mask = np.arange(n_groups) != b_idx
        rank_metric = pw_score[other_mask].max(axis=0) - pw_score[b_idx]
    else:
        raise ValueError(
            f'Unknown top_k_metric: {top_k_metric!r}. Use one of '
            "'max_abs', 'std', 'range', 'vs_baseline'."
        )
    k = min(top_k_pathways, n_pw)
    top_idx = np.argsort(rank_metric)[-k:][::-1]
    # In non-wilcoxon score modes we deferred the Wilcoxon test above to
    # save 95%+ of the work; compute it now ONLY for the selected top-K
    # pathways so the marker-size legend stays meaningful.
    if score != 'wilcoxon':
        for j_top in top_idx:
            _wilcoxon_for_pathway(int(j_top))
    top_score = pw_score[:, top_idx]
    top_pval = pw_pval[:, top_idx]
    top_pathways = [pathways[i] for i in top_idx]
    sizes_n = np.asarray([len(pathway_members[p]) for p in top_pathways])

    # --- Hierarchical clustering ---
    row_link = col_link = None
    row_order = np.arange(n_groups)
    col_order = np.arange(k)
    if cluster:
        if k > 1:
            col_link = linkage(top_score.T, method='average', metric='euclidean')
            col_order = leaves_list(col_link)
        if n_groups > 1:
            row_link = linkage(top_score, method='average', metric='euclidean')
            row_order = leaves_list(row_link)

    score_mat = top_score[row_order][:, col_order]
    pval_mat = top_pval[row_order][:, col_order]
    group_labels = [str(unique_groups[i]) for i in row_order]
    pathway_labels_full = [top_pathways[j] for j in col_order]
    pathway_n = sizes_n[col_order]

    def _short(name, n=55):
        return name if len(name) <= n else name[:n - 1] + '…'
    pathway_labels = [f'{_short(p)} ({n})' for p, n in zip(pathway_labels_full, pathway_n)]

    df = pd.DataFrame(score_mat, index=group_labels, columns=pathway_labels)
    df.index.name = condition_col
    df.attrs['pvalues'] = pd.DataFrame(pval_mat, index=group_labels, columns=pathway_labels)

    # --- Long-form pathway ↔ perturbation membership table ---
    # Restricted to the displayed top-K pathways. One row per
    # (group, pathway, perturbation) triple, with the perturbation's
    # z-score for that group plus the rolled-up pathway score & p-value
    # repeated on every row of the block. Sorted so the highest-|z|
    # drivers within each (group, pathway) come first — handy for
    # answering "which perts are pushing this hit?".
    member_rows = []
    for j_plot, pw_full in enumerate(pathway_labels_full):
        members_idx = pathway_members[pw_full]
        pw_short = pathway_labels[j_plot]
        for grp_pos, grp in enumerate(group_labels):
            g_orig = row_order[grp_pos]
            pw_score_val = float(score_mat[grp_pos, j_plot])
            pw_pval_val = float(pval_mat[grp_pos, j_plot])
            for m_idx in members_idx:
                member_rows.append({
                    condition_col: grp,
                    'pathway': pw_full,
                    'pathway_label': pw_short,
                    'perturbation': pert_names[m_idx],
                    # `pert_score` holds whatever per-pert signal the
                    # chosen score mode produced (z-score, mean prob,
                    # or argmax frequency). Kept as `pert_z` for
                    # backward-compat with callers that already
                    # consume that column name.
                    'pert_z': float(pert_score_matrix[g_orig, m_idx]),
                    'pathway_score': pw_score_val,
                    'pathway_pvalue': pw_pval_val,
                })
    members_df = pd.DataFrame(member_rows)
    if not members_df.empty:
        # For diverging modes |pert_z| ranks "most differential"; for
        # mean/frequency the raw value already represents proximity so
        # absolute-value sort is equivalent to descending-value sort.
        members_df = members_df.sort_values(
            by=[condition_col, 'pathway', 'pert_z'],
            ascending=[True, True, False],
            key=lambda s: s.abs() if s.name == 'pert_z' else s,
        ).reset_index(drop=True)
    df.attrs['members'] = members_df

    # --- Orient axes (handle transpose) ---
    # show_dendrograms_row/col refer to displayed left/top dendrograms;
    # the underlying linkage used is swapped when transposed.
    if transpose:
        plot_mat = score_mat.T  # rows=pathways, cols=groups
        plot_pval = pval_mat.T
        x_labels = group_labels
        y_labels = pathway_labels
        x_link = row_link  # groups now on x-axis → top dendrogram is row_link
        y_link = col_link  # pathways now on y-axis → left dendrogram is col_link
        xlabel_text = condition_col
        ylabel_text = 'pathway (n perts in pathway)'
        df = df.T
        df.index.name = 'pathway'
        df.attrs['pvalues'] = df.attrs['pvalues'].T
    else:
        plot_mat = score_mat
        plot_pval = pval_mat
        x_labels = pathway_labels
        y_labels = group_labels
        x_link = col_link
        y_link = row_link
        xlabel_text = 'pathway (n perts in pathway)'
        ylabel_text = condition_col
    n_x = len(x_labels)
    n_y = len(y_labels)

    # --- Marker sizes from -log10(p), capped ---
    nlp = -np.log10(np.clip(plot_pval, 1e-300, 1.0))
    nlp_capped = np.minimum(nlp, pvalue_cap)
    if nlp_capped.max() > 0:
        size_norm = nlp_capped / max(nlp_capped.max(), 1e-9)
    else:
        size_norm = np.zeros_like(nlp_capped)
    sizes_pts = size_min + size_norm * (size_max - size_min)

    # --- Layout: optional dendrograms via gridspec ---
    if figsize is None:
        figsize = (max(8.0, 0.45 * n_x + 4), max(3.5, 0.45 * n_y + 2.5))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    show_top = cluster and show_dendrograms_col and x_link is not None
    show_left = cluster and show_dendrograms_row and y_link is not None

    # gridspec rows: [optional top dendro] [main]
    #          cols: [optional left dendro] [main] [colorbar+legend]
    h_top = 0.6 if show_top else 0.05
    w_left = 0.5 if show_left else 0.05
    # hspace/wspace are in axes-fraction units; bump them up so the y-tick
    # labels (which sit in the gap between subplots) don't bleed into the
    # adjacent dendrogram, and so the top dendrogram has a visible gap
    # above the topmost dot row.
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[h_top, 5.0],
        width_ratios=[w_left, 8.0, 2.4],
        wspace=0.18 if show_left else 0.06,
        hspace=0.12 if show_top else 0.04,
    )
    ax_main = fig.add_subplot(gs[1, 1])
    # Don't use sharex/sharey — we draw dendrogram lines manually in the
    # main axis's coordinate system so alignment is exact.
    ax_top = fig.add_subplot(gs[0, 1]) if show_top else None
    ax_left = fig.add_subplot(gs[1, 0]) if show_left else None
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis('off')

    # --- Main scatter ---
    # Two knobs control color: `score` (which signal is computed) and
    # `color_norm` (how it's mapped to color). They're independent.
    #
    # `color_norm` transforms `plot_mat` in-place to a "color matrix"
    # that scatter sees, plus selects the right cmap/normalize object.
    diverging = score in diverging_modes
    # The "pathway axis" for per-row normalization depends on transpose:
    #   transpose=True : plot_mat is (n_pathways, n_groups) → groups axis=1
    #   transpose=False: plot_mat is (n_groups, n_pathways) → groups axis=0
    groups_axis = 1 if transpose else 0
    norm_obj = None
    color_mat = plot_mat
    cbar_extra_label = ''

    if color_norm == 'log':
        # LogNorm requires strictly positive values; clip non-positive
        # to a small floor so they map to the bottom of the cmap.
        from matplotlib.colors import LogNorm
        scatter_cmap = cmap if cmap is not None else 'viridis'
        pos_vals = plot_mat[plot_mat > 0]
        if pos_vals.size == 0:
            # Degenerate: fall back to linear.
            scatter_cmap = cmap if cmap is not None else 'viridis'
            color_kwargs = dict(cmap=scatter_cmap, vmin=0.0, vmax=1.0)
        else:
            vmin_log = float(pos_vals.min())
            # Robust upper bound — 98th percentile of positive values
            # so a single dominant pathway can't peg the scale. User
            # can override via z_clip; the 98th-percentile floor stays
            # if z_clip > data percentile.
            if z_clip is not None:
                vmax_log = float(z_clip)
            else:
                vmax_log = float(np.percentile(pos_vals, 98))
                # Guard against degenerate percentile==vmin_log.
                if vmax_log <= vmin_log:
                    vmax_log = float(pos_vals.max())
            color_mat = np.where(plot_mat > 0, plot_mat, vmin_log * 0.5)
            norm_obj = LogNorm(vmin=vmin_log, vmax=vmax_log)
            color_kwargs = dict(cmap=scatter_cmap, norm=norm_obj)
            cbar_extra_label = ' (log scale)'
    elif color_norm == 'row_minmax':
        scatter_cmap = cmap if cmap is not None else 'viridis'
        mn = plot_mat.min(axis=groups_axis, keepdims=True)
        mx = plot_mat.max(axis=groups_axis, keepdims=True)
        rng = mx - mn
        rng = np.where(rng > 1e-12, rng, 1.0)
        color_mat = (plot_mat - mn) / rng
        color_kwargs = dict(cmap=scatter_cmap, vmin=0.0, vmax=1.0)
        cbar_extra_label = ' (per-pathway 0–1)'
    elif color_norm == 'row_zscore':
        # Per-pathway z across groups. Signed → diverging cmap default.
        scatter_cmap = cmap if cmap is not None else 'RdBu_r'
        mu = plot_mat.mean(axis=groups_axis, keepdims=True)
        sd = plot_mat.std(axis=groups_axis, keepdims=True)
        sd = np.where(sd > 1e-12, sd, 1.0)
        color_mat = (plot_mat - mu) / sd
        # With n_groups=3 the theoretical max |z| is √2; clamp to that.
        clip_val = float(z_clip) if z_clip is not None else float(np.sqrt(2.0))
        color_kwargs = dict(cmap=scatter_cmap, vmin=-clip_val, vmax=clip_val)
        cbar_extra_label = ' (per-pathway z)'
    elif diverging:
        # Linear, diverging — original z-score / wilcoxon path.
        scatter_cmap = cmap if cmap is not None else 'RdBu_r'
        if z_clip is None:
            vmax_used = 5.0 if score == 'wilcoxon' else 2.0
        else:
            vmax_used = float(z_clip)
        color_kwargs = dict(cmap=scatter_cmap, vmin=-vmax_used, vmax=vmax_used)
    else:
        # Linear, non-negative — original mean / frequency path.
        scatter_cmap = cmap if cmap is not None else 'viridis'
        if z_clip is None:
            finite = plot_mat[np.isfinite(plot_mat)]
            upper = float(np.percentile(finite, 98)) if finite.size else 1.0
            if upper <= 0:
                upper = float(finite.max() if finite.size else 1.0) or 1.0
        else:
            upper = float(z_clip)
        color_kwargs = dict(cmap=scatter_cmap, vmin=0.0, vmax=upper)
    xs, ys = np.meshgrid(np.arange(n_x), np.arange(n_y))
    sc_obj = ax_main.scatter(
        xs.ravel(), ys.ravel(),
        s=sizes_pts.ravel(),
        c=color_mat.ravel(),
        edgecolors='black', linewidths=0.4,
        alpha=0.95,
        **color_kwargs,
    )
    ax_main.set_xticks(np.arange(n_x))
    ax_main.set_xticklabels(x_labels, rotation=45, ha='right',
                            fontsize=label_fontsize)
    ax_main.set_yticks(np.arange(n_y))
    ax_main.set_yticklabels(y_labels, fontsize=label_fontsize)
    ax_main.set_xlim(-0.5, n_x - 0.5)
    ax_main.set_ylim(-0.5, n_y - 0.5)
    ax_main.set_xlabel(xlabel_text, fontsize=label_fontsize + 1)
    ax_main.invert_yaxis()
    # When the left dendrogram is present, follow the seaborn-clustermap
    # convention: move the row labels to the right so they don't collide
    # with the dendrogram lines on the left. Drop the rotated y-axis title
    # since it would otherwise sit between the labels and the colorbar.
    if show_left:
        ax_main.yaxis.tick_right()
        ax_main.set_ylabel('')
    else:
        ax_main.set_ylabel(ylabel_text, fontsize=label_fontsize + 1)
    ax_main.tick_params(axis='both', which='both', length=0)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.set_axisbelow(True)
    ax_main.grid(True, color='lightgrey', linestyle=':', linewidth=0.6, alpha=0.7)

    # --- Dendrograms drawn manually in the scatter's coordinate system ---
    # scipy's dendrogram puts leaves at icoord positions 5, 15, ..., 10*k - 5;
    # we rescale so leaf i lands at main-axis position i. This guarantees
    # every dendrogram line aligns with its column/row in the dot panel.
    def _scaled_dendro(link, ax, orientation):
        ddata = dendrogram(link, no_plot=True, color_threshold=0,
                           above_threshold_color='black')
        # icoord = leaf positions in [5, 15, ...]; rescale to [0, n-1].
        for icoord, dcoord in zip(ddata['icoord'], ddata['dcoord']):
            xs_d = [(x - 5) / 10.0 for x in icoord]
            ys_d = list(dcoord)
            if orientation == 'top':
                ax.plot(xs_d, ys_d, color='black', linewidth=0.7)
            else:  # 'left'
                ax.plot(ys_d, xs_d, color='black', linewidth=0.7)

    if show_top:
        _scaled_dendro(x_link, ax_top, 'top')
        ax_top.set_xlim(ax_main.get_xlim())
        # Pad y-min slightly negative so leaves visibly clear the panel edge.
        y0, y1 = ax_top.get_ylim()
        ax_top.set_ylim(-(y1 - y0) * 0.05, y1)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_visible(False)
    if show_left:
        _scaled_dendro(y_link, ax_left, 'left')
        ax_left.set_ylim(ax_main.get_ylim())
        ax_left.invert_xaxis()
        # Pad on the right side (closest to main panel) so leaf endpoints
        # don't touch the y-tick labels / panel border.
        x0, x1 = ax_left.get_xlim()
        # After invert_xaxis, x0 > x1 (max → 0). The right edge is x1=0;
        # extend it slightly negative so leaves stop short of the panel.
        ax_left.set_xlim(x0, x1 - (x0 - x1) * 0.05)
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        for spine in ax_left.spines.values():
            spine.set_visible(False)

    # --- Colorbar + size legend stacked in the legend column ---
    # inset_axes (in axes-fraction coords) tracks the gridspec through
    # tight_layout, unlike fig.add_axes with frozen pre-layout coords.
    cbar_label = {
        'mean_z':    'mean z-score',
        'wilcoxon':  'signed -log10(p)',
        'mean':      'mean soft-prob (pathway)',
        'frequency': (f'fraction in top-{top_per_cell}'
                      if top_per_cell > 1 else 'fraction (argmax)'),
        'mean_freq': '√(percentile_mean · percentile_freq) (pathway)',
        'mean_or_freq': '1 − (1−P_mean)·(1−P_freq) (pathway)',
    }[score] + cbar_extra_label
    # Pin the cbar (and size legend below it) flush to the right side of the
    # legend column so they sit on the right border of the figure regardless
    # of whether y-tick labels were moved to the right by show_left.
    cbar_left = 0.78
    # Reserve the legend column as two stacked, non-overlapping bands:
    #   cbar:        y ∈ [0.62, 0.97]  (top band, height 0.35)
    #   size legend: anchored at top y=0.50, grows downward (bottom band)
    # The 0.12 dead-zone in the middle absorbs the cbar label and any
    # extra height from larger font sizes — the bands cannot collide.
    cbar_ax = ax_legend.inset_axes([cbar_left, 0.62, 0.18, 0.35])
    cb = fig.colorbar(sc_obj, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=label_fontsize - 1)

    # Size legend: pick representative -log10(p) values from displayed range.
    nlp_max = float(nlp_capped.max()) if nlp_capped.size else 0.0
    if nlp_max <= 0:
        legend_levels = np.array([1.0])
    else:
        anchors = np.linspace(0.0, nlp_max, 4)
        legend_levels = np.unique(np.round(anchors, 1))
        legend_levels = legend_levels[legend_levels > 0]
        if legend_levels.size == 0:
            legend_levels = np.array([nlp_max])

    legend_handles = []
    for lvl in legend_levels:
        s_lvl = size_min + (min(lvl, pvalue_cap) / max(nlp_max, 1e-9)) * (size_max - size_min)
        # Empty-data scatter on ax_legend so the proxy lives in the legend axis.
        h = ax_legend.scatter(
            [], [], s=s_lvl, c='lightgrey', edgecolors='black', linewidths=0.4,
            label=f'{lvl:.1f}',
        )
        legend_handles.append(h)
    # Anchor at TOP (y=0.50) with loc='upper left' so the legend grows
    # downward into the bottom band; its top is hard-pinned and cannot
    # creep up into the colorbar even with bigger fonts or markers.
    leg = ax_legend.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(cbar_left, 0.50),
        frameon=False,
        title='-log10(p)',
        title_fontsize=label_fontsize,
        fontsize=label_fontsize - 1,
        labelspacing=1.4,
        handletextpad=1.0,
        borderpad=0.5,
    )
    leg._legend_box.align = 'left'

    fig.suptitle(
        f'Top-{k} pathways by group enrichment — {library} ({score})',
        fontsize=title_fontsize, y=0.995,
    )
    # tight_layout doesn't play with the colorbar inset_axes — use explicit
    # margins to leave room for rotated x-tick labels and the suptitle.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.30)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return df


def plot_deg_robustness_decay(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = (
        'pearson', 'top_k_jaccard',
        'pearson_pathway', 'jaccard_pathway',
    ),
    x: Literal['n_masked', 'n_top'] = 'n_masked',
    palette: str | list | dict | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = 'DEG-masking robustness vs random-mask baseline',
    output_file: str | None = None,
):
    """Faceted line plot of robustness-metric decay as more genes are masked.

    Designed for the output of ``evaluate_deg_robustness``. One panel
    per metric; one line per ``(group, mask_type)``:

    - **Solid** + circle markers = DEG mask
    - **Dashed** + square markers, half-alpha = random mask

    The visible gap between solid and dashed lines at the same x is the
    DEG-specific effect (the differential vs random-mask baseline). Flat
    lines that coincide = the model isn't DEG-dependent at that mask
    size. Solid lines dropping toward 0 or going negative while dashed
    stays near 1 = DEG-dependent.

    Parameters
    ----------
    df
        Output of ``evaluate_deg_robustness`` (long-form with
        ``mask_type ∈ {'deg', 'random'}``).
    metrics
        Which metric columns to facet over. Each must exist in ``df``.
    x
        ``'n_masked'`` (default; actual gene count) or ``'n_top'``
        (the input parameter — same per iteration).
    palette
        Group color override (str/list/dict). Defaults to ``get_pal``.
    title
        Figure-level title.
    output_file
        Save path; when ``None``, ``plt.show()`` is called.
    """
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError(
            f'No usable metric columns in df. Available: {list(df.columns)}'
        )
    if x not in df.columns:
        raise KeyError(f'x={x!r} not in df.columns: {list(df.columns)}')

    n_panels = len(metrics)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = (n_panels + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (5.5 * n_cols, 3.6 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    groups = list(pd.unique(df['group'].astype(str)))
    if isinstance(palette, dict):
        color_map = {g: palette.get(g, 'grey') for g in groups}
    elif isinstance(palette, list):
        color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
    elif isinstance(palette, str):
        cols = sns.color_palette(palette, len(groups))
        color_map = {g: cols[i] for i, g in enumerate(groups)}
    else:
        cols = get_pal(len(groups))
        color_map = {g: cols[i] for i, g in enumerate(groups)}

    mask_styles = {
        'deg':    dict(linestyle='-',  alpha=1.0,  marker='o', linewidth=2.0),
        'random': dict(linestyle='--', alpha=0.55, marker='s', linewidth=1.4),
    }
    mask_types_present = [m for m in ('deg', 'random')
                          if m in df['mask_type'].unique()]

    for ax, metric in zip(axes, metrics):
        for g in groups:
            for mt in mask_types_present:
                sub = (df[(df['group'].astype(str) == g)
                          & (df['mask_type'] == mt)]
                       .sort_values(x))
                if sub.empty or sub[metric].isna().all():
                    continue
                ax.plot(
                    sub[x].values, sub[metric].values,
                    color=color_map[g],
                    label=f'{g} ({mt})',
                    **mask_styles[mt],
                )
        ax.set_xlabel(x)
        ax.set_ylabel(metric)
        ax.set_title(metric.replace('_', ' '))
        # Reference lines: y=1 (perfect retention) and y=0 (rank
        # independence; below 0 means rank inversion under masking).
        ax.axhline(1.0, color='lightgrey', linewidth=0.6, linestyle=':')
        if 'pearson' in metric.lower():
            ax.axhline(0.0, color='lightgrey', linewidth=0.6, linestyle=':')
        ax.grid(True, color='lightgrey', linestyle=':', linewidth=0.4, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused panels.
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Shared figure-level legend (dedup labels).
    handles, labels = axes[0].get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        fig.legend(
            uniq_h, uniq_l,
            loc='lower center',
            ncol=min(len(uniq_h), 6),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig


def _plot_set_deg_venn(
    top_items: pd.DataFrame,
    deg_set: set,
    item_col: str,
    reference_label: str = 'DEG',
    condition_subset: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title_template: str = '{condition}',
    palette: list | None = None,
    label_fontsize: float = 11.0,
    output_file: str | None = None,
    case_insensitive: bool = True,
    stats_df: pd.DataFrame | None = None,
    show_stats: bool = True,
    axes_override=None,
):
    """Per-condition Venn panels of (mode top-K) vs reference set.

    Generic backend for ``plot_pert_deg_venn`` and
    ``plot_pathway_deg_venn``. Builds 2- or 3-set Venn per condition.
    """
    try:
        from matplotlib_venn import venn2, venn3
    except ImportError:
        raise ImportError(
            'Venn plotting requires matplotlib-venn. '
            'Install with: pip install matplotlib-venn'
        )

    if 'mode' not in top_items.columns or 'condition' not in top_items.columns:
        raise ValueError(
            f"top_items must have 'mode' and 'condition' columns; got {list(top_items.columns)}."
        )
    if item_col not in top_items.columns:
        raise ValueError(f'item_col={item_col!r} not in top_items columns.')

    modes = list(pd.unique(top_items['mode']))
    if len(modes) not in (1, 2):
        raise ValueError(
            f'Venn supports 1 or 2 modes (giving 2 or 3 sets with '
            f'{reference_label}); got {len(modes)}: {modes}. Subset the '
            f'modes upstream, or render an UpSet plot separately.'
        )

    conditions = list(pd.unique(top_items['condition']))
    if condition_subset is not None:
        conditions = [c for c in conditions if c in condition_subset]
    if not conditions:
        raise ValueError('No conditions to render after subsetting.')

    def _norm(x):
        return str(x).upper() if case_insensitive else str(x)
    ref_norm = {_norm(x) for x in deg_set}

    n = len(conditions)
    if axes_override is not None:
        # External axes (e.g. from plot_pathway_deg_venn with pert_panel=True);
        # caller manages the figure lifecycle.
        axes = np.asarray(axes_override).ravel()
        fig = None
        if len(axes) < n:
            raise ValueError(
                f'axes_override has {len(axes)} axes but {n} conditions to render.'
            )
    else:
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        if figsize is None:
            figsize = (4.5 * ncols, 4.2 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.ravel()

    if palette is None:
        palette = list(sns.color_palette('tab10', max(3, len(modes) + 1)))

    for ax, cond in zip(axes, conditions):
        sub = top_items[top_items['condition'] == cond]
        mode_sets = {
            m: {_norm(v) for v in sub.loc[sub['mode'] == m, item_col]}
            for m in modes
        }
        if len(modes) == 1:
            v = venn2(
                [mode_sets[modes[0]], ref_norm],
                set_labels=(modes[0], reference_label),
                set_colors=tuple(palette[:2]),
                alpha=0.55,
                ax=ax,
            )
        else:
            v = venn3(
                [mode_sets[modes[0]], mode_sets[modes[1]], ref_norm],
                set_labels=(modes[0], modes[1], reference_label),
                set_colors=tuple(palette[:3]),
                alpha=0.55,
                ax=ax,
            )
        for label in v.set_labels or []:
            if label is not None:
                label.set_fontsize(label_fontsize)
        for label in v.subset_labels or []:
            if label is not None:
                label.set_fontsize(label_fontsize - 1)
        ax.set_title(
            title_template.format(condition=cond, n_modes=len(modes)),
            fontsize=label_fontsize + 1,
        )

        # Per-mode chance-level annotation under the panel.
        # Looks up matching rows in `stats_df` (output of
        # compare_pert_deg_overlap / compare_pathway_deg_overlap) and
        # renders a one-line summary per mode: ``mode: obs (exp=X, p=Y)``.
        # The annotation makes each Venn self-documenting — reviewer
        # sees both the observed overlap AND the random chance level.
        if show_stats and stats_df is not None and 'pvalue' in stats_df.columns:
            sub_stats = stats_df[stats_df['condition'] == cond]
            if not sub_stats.empty:
                # If the stats came from pvalue_method='pert_sampling',
                # the row stats are pert-level (not pathway-level). Detect
                # by the marker column the merge adds, and label the
                # subtitle so users know what the numbers mean.
                is_pert_level = (
                    'pathway_n_overlap' in stats_df.columns
                    or stats_df.attrs.get('pvalue_method') == 'pert_sampling'
                )
                lines = []
                for m in modes:
                    row = sub_stats[sub_stats['mode'] == m]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    obs = int(r.get('n_overlap', 0))
                    exp_v = r.get('expected_overlap', float('nan'))
                    p = r.get('pvalue', float('nan'))
                    fold = r.get('fold_enrichment', float('nan'))
                    pieces = [f'{m}: obs={obs}']
                    if pd.notna(exp_v):
                        pieces.append(f'exp≈{exp_v:.1f}')
                    if pd.notna(fold):
                        pieces.append(f'fold={fold:.2f}')
                    if pd.notna(p):
                        # Use 2 sig figs; ``< 1e-3`` for tiny p's.
                        p_str = f'p={p:.2g}' if p >= 1e-3 else f'p<1e-3'
                        pieces.append(p_str)
                    lines.append('  '.join(pieces))
                if lines:
                    if is_pert_level:
                        lines.insert(0, '(pert-level enrichment vs DEG-pathway members)')
                    ax.text(
                        0.5, -0.08, '\n'.join(lines),
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=label_fontsize - 1.5,
                        family='monospace',
                    )

    # Only do figure-level operations when we own the figure (i.e., we
    # created it). When `axes_override` was supplied, the caller manages
    # the figure and we just draw into the axes.
    if fig is not None:
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    return fig


def plot_pert_deg_venn(
    gene_cmp: pd.DataFrame,
    condition_subset: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title_template: str = '{condition}',
    palette: list | None = None,
    label_fontsize: float = 11.0,
    show_stats: bool = True,
    output_file: str | None = None,
):
    """Per-condition Venn diagrams of top-K perturbation sets vs DEGs.

    Direct visualization of the gene-level overlap quantified by
    ``compare_pert_deg_overlap``. One panel per condition; intersects
    each mode's top-K perturbation set with the classical DEG list.

    Supports 1 or 2 modes (giving 2 or 3 sets total with DEG).

    Parameters
    ----------
    gene_cmp
        Output of ``compare_pert_deg_overlap``. Must carry
        ``attrs['top_perts']`` and ``attrs['deg_genes']``.

    See ``_plot_set_deg_venn`` for the other parameters.
    """
    top_perts = gene_cmp.attrs.get('top_perts')
    deg_genes = gene_cmp.attrs.get('deg_genes')
    if top_perts is None or deg_genes is None:
        raise ValueError(
            'gene_cmp must come from compare_pert_deg_overlap; missing '
            "attrs['top_perts'] / attrs['deg_genes']."
        )
    return _plot_set_deg_venn(
        top_items=top_perts,
        deg_set=deg_genes,
        item_col='perturbation',
        reference_label='DEG',
        condition_subset=condition_subset,
        figsize=figsize,
        title_template=title_template,
        palette=palette,
        label_fontsize=label_fontsize,
        output_file=output_file,
        stats_df=gene_cmp,
        show_stats=show_stats,
    )


def plot_pathway_deg_venn(
    pathway_cmp: pd.DataFrame,
    condition_subset: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title_template: str = '{condition}',
    palette: list | None = None,
    label_fontsize: float = 11.0,
    show_stats: bool = True,
    pert_panel: bool = False,
    output_file: str | None = None,
):
    """Per-condition Venn diagrams of top-K pathway sets vs DEG-pathways.

    Pathway-level analogue of ``plot_pert_deg_venn``. The expected
    visual story (paired with the gene-level Venn):

    - **Gene-level Venn**: small overlap between modes and DEG genes
      → the model's predicted *gene names* aren't mostly DEGs
      themselves; it nominates different upstream effectors.
    - **Pathway-level Venn**: substantial overlap between modes and
      DEG-pathways → both methods converge on the same downstream
      *biology*, just via different gene-level handles.

    That contrast is the cleanest visual statement of "different
    effectors, same pathway biology" — i.e. the model's value-add is
    distributional, not single-gene.

    Parameters
    ----------
    pathway_cmp
        Output of ``compare_pathway_deg_overlap``. Must carry
        ``attrs['top_pathways']`` and ``attrs['deg_pathways']``.

    See ``_plot_set_deg_venn`` for the other parameters.
    """
    top_pathways = pathway_cmp.attrs.get('top_pathways')
    deg_pathways = pathway_cmp.attrs.get('deg_pathways')
    if top_pathways is None or deg_pathways is None:
        raise ValueError(
            'pathway_cmp must come from compare_pathway_deg_overlap; '
            "missing attrs['top_pathways'] / attrs['deg_pathways']."
        )

    if not pert_panel:
        return _plot_set_deg_venn(
            top_items=top_pathways,
            deg_set=deg_pathways,
            item_col='pathway',
            reference_label='DEG_pathway',
            condition_subset=condition_subset,
            figsize=figsize,
            title_template=title_template,
            palette=palette,
            label_fontsize=label_fontsize,
            output_file=output_file,
            stats_df=pathway_cmp,
            show_stats=show_stats,
        )

    # --- Stacked layout: pathway Venn on top, pert Venn on bottom ---
    top_perts = pathway_cmp.attrs.get('top_perts')
    deg_pw_members = pathway_cmp.attrs.get('deg_pw_members')
    if top_perts is None or deg_pw_members is None:
        raise ValueError(
            "pert_panel=True requires pathway_cmp.attrs['top_perts'] and "
            "['deg_pw_members'], which are only populated when "
            "compare_pathway_deg_overlap was called with "
            "pvalue_method='pert_sampling' and pert_dfs supplied."
        )

    # Build the figure ourselves so both panels share a single layout.
    # 2 rows × n_conditions columns; top row = pathway Venn, bottom = pert Venn.
    conditions_all = list(pd.unique(top_pathways['condition']))
    if condition_subset is not None:
        conditions = [c for c in conditions_all if c in condition_subset]
    else:
        conditions = conditions_all
    n = len(conditions)
    ncols = min(n, 3)
    nrows_per = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.5 * ncols, 4.2 * nrows_per * 2)
    fig, axes = plt.subplots(
        2 * nrows_per, ncols, figsize=figsize, squeeze=False,
    )

    # Top half = pathway Venn axes; bottom half = pert Venn axes.
    pathway_axes = axes[:nrows_per].ravel()
    pert_axes = axes[nrows_per:].ravel()

    _plot_set_deg_venn(
        top_items=top_pathways,
        deg_set=deg_pathways,
        item_col='pathway',
        reference_label='DEG_pathway',
        condition_subset=conditions,
        title_template='[pathways] ' + title_template,
        palette=palette,
        label_fontsize=label_fontsize,
        stats_df=pathway_cmp,
        show_stats=show_stats,
        axes_override=pathway_axes,
    )

    _plot_set_deg_venn(
        top_items=top_perts,
        deg_set=deg_pw_members,
        item_col='perturbation',
        reference_label='DEG_pw_member',
        condition_subset=conditions,
        title_template='[perts] ' + title_template,
        palette=palette,
        label_fontsize=label_fontsize,
        # The same stats apply to the pert-level overlap (pert-sampling
        # test was computed on these perts), so show them under the
        # pert panel for direct readability.
        stats_df=pathway_cmp,
        show_stats=show_stats,
        axes_override=pert_axes,
    )

    # Hide any extra unused axes (when n_conditions < ncols × nrows_per).
    for ax in pathway_axes[n:]:
        ax.set_visible(False)
    for ax in pert_axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_pathway_null_distribution(
    null_result: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    panel_size: tuple[float, float] = (4.0, 3.2),
    bins: int = 20,
    obs_color: str = '#d62728',
    null_color: str = '#1f77b4',
    title_template: str = '{condition}',
    dpi: int = 200,
    output_file: str | None = None,
):
    """Histogram per condition: null pathway-DEG overlap with the observed value marked.

    Companion plot to ``pathway_overlap_permutation_test`` — drops cleanly
    into a poster panel next to the Venn so reviewers can see *both* the
    observed overlap and the null distribution that justifies it.

    Parameters
    ----------
    null_result
        Output of ``pathway_overlap_permutation_test``. Must carry
        ``attrs['null_overlaps']`` (n_perm × n_groups) and
        ``attrs['conditions']``.
    """
    null_overlaps = null_result.attrs.get('null_overlaps')
    conditions = null_result.attrs.get('conditions')
    if null_overlaps is None or conditions is None:
        raise ValueError(
            'null_result must come from pathway_overlap_permutation_test; '
            "missing attrs['null_overlaps'] / attrs['conditions']."
        )
    n_groups = len(conditions)
    ncols = min(n_groups, 3)
    nrows = (n_groups + ncols - 1) // ncols
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.ravel()

    for ax, cond, i in zip(axes, conditions, range(n_groups)):
        ax.hist(null_overlaps[:, i], bins=bins, color=null_color,
                alpha=0.7, edgecolor='white')
        row = null_result[null_result['condition'] == cond].iloc[0]
        obs = row['observed_overlap']
        p_emp = row['p_emp']
        ax.axvline(obs, color=obs_color, linewidth=2.0,
                   label=f'obs = {obs}\np_emp = {p_emp:.3g}')
        ax.set_xlabel(f'top-{int(row["top_k"])} ∩ DEG_pathway')
        ax.set_ylabel('null count')
        ax.set_title(title_template.format(condition=cond))
        ax.legend(loc='upper right', frameon=False, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes[n_groups:]:
        ax.set_visible(False)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_embedding_eval_summary(
    root_dir: str,
    output_file: str | None = None,
    label_fn=None,
    figsize: tuple[float, float] = (18, 11),
    dpi: int = 120,
):
    """Read all ``eval_summary.json`` / ``bio_eval_summary.json`` files under
    ``root_dir`` and produce a side-by-side comparison figure across encoders.

    Each encoder is identified by the directory containing its JSON files.
    ``label_fn(json_path)`` can return a custom short label; the default is
    the path relative to ``root_dir`` (joined with ``/``).

    Layout (2×3 gridspec; T-sweep spans the bottom-right two cells now
    that the on-figure summary table is gone — see CSV below):
      (0,0) bio precision@k bars per encoder
      (0,1) bio lift@k vs random bars per encoder
      (0,2) anisotropy: raw mean cos vs centered std cos per encoder
      (1,0) absolute dimensionality: eff_dim_95 + total_dim (log y)
      (1,1-2) T-sweep: mass_true / top5 / top20 vs T per encoder

    A single encoder color legend sits below the grid, centered. Within-panel
    distinctions (raw/centered bars, linestyle for top-k) get inline mini-legends.

    The summary table that used to live in the (1,2) slot is now written as
    ``<output_file basename>.csv`` next to the PNG — same per-encoder stats,
    plus precision/lift at every k and eff-dim utilization, in a format
    that's easier to read into a notebook for follow-up analysis.
    """
    import json
    import glob
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    eval_paths = sorted(glob.glob(os.path.join(root_dir, '**/eval_summary.json'), recursive=True))
    bio_paths = sorted(glob.glob(os.path.join(root_dir, '**/bio_eval_summary.json'), recursive=True))

    def _default_label(p: str) -> str:
        rel = os.path.relpath(os.path.dirname(p), root_dir)
        return rel if rel != '.' else os.path.basename(root_dir)
    label = label_fn or _default_label

    summaries: dict[str, dict] = {}
    for p in eval_paths:
        summaries.setdefault(label(p), {})['eval'] = json.load(open(p))
    for p in bio_paths:
        summaries.setdefault(label(p), {})['bio'] = json.load(open(p))
    if not summaries:
        raise FileNotFoundError(
            f"No eval_summary.json or bio_eval_summary.json under {root_dir}"
        )

    # Stable order: by p@10 lift if bio is available, else alphabetical
    def _sort_key(item):
        _, d = item
        b = d.get('bio')
        return -b['lift_at_k']['10'] if b else float('inf')
    ordered = sorted(summaries.items(), key=_sort_key)
    encoders = [name for name, _ in ordered]

    # Viridis palette over [0.05, 0.95] to avoid the very-dark / very-bright ends
    palette = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(encoders), 1)))
    color_map = dict(zip(encoders, palette))

    # gridspec instead of plt.subplots: T-sweep (line plot) gets the
    # bottom-middle + bottom-right cells now that the table is gone.
    # Extra horizontal real estate makes the T-axis far more readable
    # when there are many encoders.
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3)
    ax_prec = fig.add_subplot(gs[0, 0])
    ax_lift = fig.add_subplot(gs[0, 1])
    ax_aniso = fig.add_subplot(gs[0, 2])
    ax_dim = fig.add_subplot(gs[1, 0])
    ax_tsweep = fig.add_subplot(gs[1, 1:])

    # ---- bio precision@k and lift@k ----
    ks = [5, 10, 50, 100]
    x = np.arange(len(ks))
    width = 0.8 / max(len(encoders), 1)
    for i, (name, d) in enumerate(ordered):
        bio = d.get('bio')
        if bio is None:
            continue
        prec = [bio['precision_at_k'][str(k)] for k in ks]
        lift = [bio['lift_at_k'][str(k)] for k in ks]
        offset = (i - (len(encoders) - 1) / 2) * width
        ax_prec.bar(x + offset, prec, width, color=color_map[name])
        ax_lift.bar(x + offset, lift, width, color=color_map[name])
    any_bio = next((d['bio'] for _, d in ordered if d.get('bio')), None)
    if any_bio is not None:
        ax_prec.axhline(any_bio['random_baseline'], ls='--', c='gray', lw=0.8)
        ax_prec.text(len(ks) - 0.5, any_bio['random_baseline'],
                     f" random ({any_bio['random_baseline']:.3f})",
                     fontsize=7, color='gray', va='bottom', ha='right')
        ax_lift.axhline(1.0, ls='--', c='gray', lw=0.8)
    ax_prec.set_xticks(x); ax_prec.set_xticklabels([f'k={k}' for k in ks])
    ax_lift.set_xticks(x); ax_lift.set_xticklabels([f'k={k}' for k in ks])
    ax_prec.set_ylabel('precision@k'); ax_prec.set_title('Bio neighborhood — precision@k')
    ax_lift.set_ylabel('lift over random'); ax_lift.set_title('Bio neighborhood — lift@k')

    # ---- anisotropy (raw mean vs centered std) ----
    raw_means = []; centered_stds = []; names_with_eval = []
    for name, d in ordered:
        ev = d.get('eval')
        if ev is None:
            continue
        names_with_eval.append(name)
        raw_means.append(ev['anisotropy']['raw_mean_cos'])
        centered_stds.append(ev['anisotropy']['centered_std_cos'])
    xa = np.arange(len(names_with_eval))
    # Encoder color for the bars, raw/centered distinguished by hatch
    bar_colors = [color_map[n] for n in names_with_eval]
    ax_aniso.bar(xa - 0.2, raw_means, 0.4, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_aniso.bar(xa + 0.2, centered_stds, 0.4, color=bar_colors,
                 edgecolor='black', linewidth=0.5, hatch='///')
    ax_aniso.axhline(0, c='gray', lw=0.5)
    ax_aniso.set_xticks(xa)
    ax_aniso.set_xticklabels(names_with_eval, rotation=30, ha='right', fontsize=7)
    ax_aniso.set_title('Anisotropy (solid=raw mean cos, hatched=centered std cos)')
    aniso_handles = [
        Patch(facecolor='lightgray', edgecolor='black', label='raw mean cos (anisotropy)'),
        Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='centered std cos (structure)'),
    ]
    ax_aniso.legend(handles=aniso_handles, fontsize=7, frameon=False, loc='upper right')

    # ---- absolute dimensionality: eff_dim_95 vs total_dim, log scale ----
    eff_abs = [summaries[n]['eval']['effective_dim']['95pct_variance_dim'] for n in names_with_eval]
    tot_dim = [summaries[n]['eval']['dim'] for n in names_with_eval]
    ax_dim.bar(xa - 0.2, eff_abs, 0.4, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_dim.bar(xa + 0.2, tot_dim, 0.4, color=bar_colors,
               edgecolor='black', linewidth=0.5, hatch='///')
    # Annotate each pair with the utilization fraction
    for i, (e, t) in enumerate(zip(eff_abs, tot_dim)):
        frac = e / t if t else 0
        ax_dim.text(i, max(e, t) * 1.05, f"{frac:.0%}", ha='center', fontsize=7)
    ax_dim.set_yscale('log')
    ax_dim.set_xticks(xa)
    ax_dim.set_xticklabels(names_with_eff := names_with_eval, rotation=30, ha='right', fontsize=7)
    ax_dim.set_ylabel('dims (log scale)')
    ax_dim.set_title('Effective vs total dim (% label = eff / total)')
    dim_handles = [
        Patch(facecolor='lightgray', edgecolor='black', label='95% variance dim'),
        Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='total dim'),
    ]
    ax_dim.legend(handles=dim_handles, fontsize=7, frameon=False, loc='upper right')

    # ---- T-sweep: mass_true (solid) + top5 (dashed) + top20 (dotted) ----
    topk_styles = [('mass_true', '-', 1.6, 'top-1 (true)'),
                   ('top5_mass', '--', 1.2, 'top-5'),
                   ('top20_mass', ':', 1.0, 'top-20')]
    for name in names_with_eval:
        ev = summaries[name]['eval']
        sweep = ev['T_sweep']
        Ts = sorted(float(k.split('=')[1]) for k in sweep)
        for key, ls, lw, _ in topk_styles:
            ys = [sweep[f'T={t:.2f}'][key] for t in Ts]
            ax_tsweep.plot(Ts, ys, ls=ls, lw=lw, color=color_map[name],
                           marker='o' if ls == '-' else None, markersize=3)
    ax_tsweep.axhspan(0.3, 0.5, alpha=0.08, color='green')
    ax_tsweep.text(ax_tsweep.get_xlim()[1], 0.4, ' target band',
                   fontsize=7, color='green', va='center', ha='right')
    ax_tsweep.set_xlabel('soft_target_T')
    ax_tsweep.set_ylabel('cumulative softmax mass (centered)')
    ax_tsweep.set_title('T-sweep: mass at top-k vs temperature')
    style_handles = [Line2D([], [], ls=ls, lw=lw, color='black', label=lbl)
                     for _, ls, lw, lbl in topk_styles]
    ax_tsweep.legend(handles=style_handles, fontsize=7, frameon=False, loc='upper right')

    # ---- summary table — saved as CSV next to the PNG (richer than the
    #      old in-figure version: every precision/lift k, eff-dim
    #      utilization, raw anisotropy stats, all in raw numbers for
    #      downstream analysis).
    summary_rows = []
    for name in encoders:
        ev = summaries[name].get('eval') or {}
        bio = summaries[name].get('bio') or {}
        row = {'encoder': name}
        if ev:
            row['n_classes'] = ev.get('n_classes')
            row['dim'] = ev.get('dim')
            eff_dim_95 = ev.get('effective_dim', {}).get('95pct_variance_dim')
            row['eff_dim_95'] = eff_dim_95
            if eff_dim_95 is not None and ev.get('dim'):
                row['dim_utilization'] = round(eff_dim_95 / ev['dim'], 4)
            row['anisotropy_raw_mean_cos'] = round(ev.get('anisotropy', {}).get('raw_mean_cos', float('nan')), 4)
            row['anisotropy_centered_std_cos'] = round(ev.get('anisotropy', {}).get('centered_std_cos', float('nan')), 4)
            row['discriminability_T0.05'] = round(ev.get('discriminability_T0.05', float('nan')), 4)
            row['recommended_T'] = round(ev.get('recommended_T', float('nan')), 4)
        if bio:
            row['bio_n_annotated'] = bio.get('n_annotated_genes')
            row['bio_random_baseline'] = round(bio.get('random_baseline', float('nan')), 4)
            for k in (5, 10, 50, 100):
                row[f'precision@{k}'] = round(bio.get('precision_at_k', {}).get(str(k), float('nan')), 4)
                row[f'lift@{k}'] = round(bio.get('lift_at_k', {}).get(str(k), float('nan')), 4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ---- single shared encoder legend, centered below the grid ----
    enc_handles = [Patch(facecolor=color_map[n], edgecolor='black', linewidth=0.5, label=n)
                   for n in encoders]
    fig.legend(handles=enc_handles, ncol=min(len(encoders), 6),
               loc='lower center', bbox_to_anchor=(0.5, 0.0),
               fontsize=9, frameon=False, title='encoder', title_fontsize=10)

    # Leave room for the bottom legend
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if output_file:
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        log.info(f"Saved embedding eval summary plot to {output_file}")
        csv_path = os.path.splitext(output_file)[0] + '.csv'
        summary_df.to_csv(csv_path, index=False)
        log.info(f"Saved embedding eval summary table to {csv_path}")
    return fig



def plot_zero_shot_performance(
        per_class_df: pd.DataFrame,
        metric: str = 'f1',
        bins: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 1.0),
        seen_label: str = 'seen',
        output_file: str | None = None,
        show_strip: bool = True,
        annotate_n: bool = True,
        ax: 'plt.Axes | None' = None,
        figsize: tuple[float, float] = (9, 5),
        title: str | None = None,
        palette: str | list | dict | None = None,
        y_range: tuple[float, float] | None = None,
        baseline: float | dict | None = None,
        baseline_label: str = 'random baseline',
    ) -> 'plt.Axes':
    """Per-class metric distribution across zero-shot similarity bins.

    Plots a boxen per group:
      - ``seen``: trained classes (``is_zero_shot == False``)
      - one bin per similarity range for zero-shot classes

    Choice of ``metric`` controls what the boxen y-axis shows. Useful
    options when paired with ``get_prediction_output_report``:

    - ``'f1'`` — strict label-match F1 from sklearn. Brutal at large
      candidate counts (18K classes ⇒ near-zero for most cells).
    - ``'mean_cosine_raw'`` — *recommended* for "did the model predict
      something semantically similar?" Soft cosine between predicted and
      true class anchors in **raw** LLM embedding space. Model-free,
      semantically grounded, immune to adapter distortion. Use this when
      the goal is "credit close-but-wrong predictions". Range: roughly
      [-0.2, 1.0] depending on the embedding distribution.
    - ``'mean_cosine_proj'`` — same in the model's **projection** space.
      Useful as a *diagnostic* alongside raw: a big gap (raw high, proj
      low) means the adapter has warped the geometry that gave the raw
      score its meaning.
    - ``'precision'`` / ``'recall'`` — standard sklearn variants.

    Use ``y_range`` to override the default range (auto for soft cosine,
    [0, 1] for f1/precision/recall).

    The high-similarity bins ("zero-shot ≈ trained sibling") should track
    seen performance; the low-similarity bin is the actual generalization
    test. The visual gap between them is the most useful diagnostic — if all
    bins land at the same value, the model isn't getting any leverage from
    proximity to trained classes. If only the high-similarity bin works,
    "zero-shot" is mostly sibling-guessing and not real generalization.

    Parameters
    ----------
    per_class_df
        Output of ``zero_shot_classification_report``'s per-class frame, or
        ``summary.attrs['zero_shot_per_class']``. Must have the column
        named by ``metric`` (or its sklearn-style ``f1-score`` form),
        ``is_zero_shot``, and ``similarity_to_trained``.
    metric
        Column name (or shorthand for sklearn metrics: 'f1' → 'f1-score').
    bins
        Similarity-bin edges for zero-shot classes.
    seen_label
        x-axis label for the trained-class group.
    show_strip
        Overlay a strip plot so individual classes are visible.
    annotate_n
        Suffix each x-tick label with the number of classes in that bin.
    y_range
        Override y-axis limits. Default: ``[0, 1]`` for f1/precision/recall,
        auto-fit for soft-cosine metrics (whose range can extend below 0).
    """
    df = per_class_df.copy()
    # Resolve metric → column name. Direct hit first (covers soft-cosine
    # columns like 'mean_cosine_raw' or the per-cell 'cosine'); fall
    # back to sklearn-style suffix.
    if metric in df.columns:
        metric_col = metric
    elif f'{metric}-score' in df.columns:
        metric_col = f'{metric}-score'
    elif metric == 'cosine' and 'cosine' in df.columns:
        metric_col = 'cosine'
    else:
        raise KeyError(
            f"metric column {metric!r} not in input df "
            f"(have: {list(df.columns)[:10]}…)."
        )
    if 'is_zero_shot' not in df.columns:
        raise KeyError("per_class_df missing 'is_zero_shot' column.")
    if 'similarity_to_trained' not in df.columns:
        raise KeyError(
            "per_class_df missing 'similarity_to_trained' — pass the frame "
            "from zero_shot_classification_report with cls_similarities=…"
        )

    # Build a categorical bin column. Seen first, then zero-shot bins in
    # ascending similarity order so x-axis reads "trained → similar → far".
    bin_edges = np.asarray(bins, dtype=float)
    bin_labels = [f'[{lo:.2f}-{hi:.2f}{")" if hi < bin_edges[-1] else "]"}'
                  for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    def _assign(row):
        if not row['is_zero_shot']:
            return seen_label
        s = row['similarity_to_trained']
        if not np.isfinite(s):
            return None
        # Right-inclusive last bin; half-open elsewhere
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            top = (i == len(bin_edges) - 2)
            if (s >= lo) and (s < hi or (top and np.isclose(s, hi))):
                return bin_labels[i]
        return None

    df['_bucket'] = df.apply(_assign, axis=1)
    df = df.dropna(subset=['_bucket', metric_col])
    if df.empty:
        raise ValueError('No rows left to plot after bucket / metric NaN filtering.')

    order = [seen_label] + bin_labels
    order = [b for b in order if b in set(df['_bucket'])]
    counts = df.groupby('_bucket', observed=True)[metric_col].count()

    # Resolve palette: seen in a neutral color, zero-shot bins in gradient
    if palette is None:
        seen_color = '#8c8c8c'
        n_zs = len(bin_labels)
        zs_cmap = sns.color_palette('viridis', n_colors=max(n_zs, 1))
        palette = {seen_label: seen_color}
        for lbl, col in zip(bin_labels, zs_cmap):
            palette[lbl] = col
    elif isinstance(palette, str):
        palette = _resolve_palette(palette, order)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
    if show_strip:
        # Strip goes FIRST (behind the boxen) so the boxen reads as the
        # primary summary and the dots are visible context. For per-cell
        # input the data can be huge (>10k rows per bin) and the strip
        # would become a solid band — subsample so points stay legible.
        strip_data = df
        strip_cap = 500
        if strip_data.groupby('_bucket').size().max() > strip_cap:
            strip_data = strip_data.groupby('_bucket', group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), strip_cap), random_state=0)
            )
        sns.stripplot(
            data=strip_data, x='_bucket', y=metric_col, order=order, ax=ax,
            color='black', size=2.5, alpha=0.3, jitter=0.25, zorder=1,
        )
    sns.boxenplot(
        data=df, x='_bucket', y=metric_col, order=order, ax=ax,
        palette=palette, showfliers=False,
    )
    # Boxen on top of the strip — translucent fills so background dots
    # remain visible. Seaborn assigns one Patch per percentile band per
    # box; iterating ax.patches catches them all without depending on
    # private API.
    for patch in ax.patches:
        r, g, b, _ = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.75))
        patch.set_zorder(2)

    # Reference: median of seen, so eye can quickly compare bins
    if seen_label in counts.index:
        seen_median = df.loc[df['_bucket'] == seen_label, metric_col].median()
        ax.axhline(seen_median, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(
            len(order) - 0.5, seen_median, f' seen median = {seen_median:.2f}',
            color='red', va='bottom', ha='right', fontsize=8,
        )
    # Random baseline: where would the metric land if predictions were
    # uncorrelated with the truth? Important sanity check for soft-cosine
    # metrics — LLM embeddings can be anisotropic (mean random-pair cosine
    # at 0.2-0.5 even with no semantics), so "model achieves 0.65" needs
    # a baseline to interpret. Accepts either a scalar or the dict
    # returned by ``random_cosine_baseline`` (uses ``mean`` field).
    if baseline is not None:
        baseline_val = baseline['mean'] if isinstance(baseline, dict) else float(baseline)
        ax.axhline(baseline_val, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(
            len(order) - 0.5, baseline_val, f' {baseline_label} = {baseline_val:.2f}',
            color='gray', va='bottom', ha='right', fontsize=8,
        )
        # If a baseline dict gave us a std, shade ±1σ as a "noise band"
        # around the baseline — anything inside it is plausibly chance.
        if isinstance(baseline, dict) and baseline.get('std') is not None and baseline.get('std', 0) > 0:
            sd = float(baseline['std'])
            ax.axhspan(baseline_val - sd, baseline_val + sd, color='gray', alpha=0.08)

    if annotate_n:
        new_labels = [f'{b}\n(N={counts.get(b, 0)})' for b in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(new_labels)
    ax.set_xlabel('similarity bin (zero-shot) / trained')
    ax.set_ylabel(metric_col)
    ax.set_title(title or f'Per-class {metric_col} by zero-shot similarity')
    # F1/precision/recall live in [0, 1] — pin so cross-run plots are
    # directly comparable. Soft cosine can range into negatives (and rarely
    # above 1), so auto-fit there. Caller can always override via y_range.
    if y_range is not None:
        ax.set_ylim(*y_range)
    elif metric_col in ('f1-score', 'precision', 'recall', 'f1'):
        ax.set_ylim(0, 1)
    # else: matplotlib auto-fit picks up the soft-cosine range.

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return ax


def plot_zero_shot_recall_at_k(
        per_class_df: pd.DataFrame,
        ks: tuple[int, ...] = (5, 50, 100),
        bins: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 1.0),
        seen_label: str = 'seen',
        output_file: str | None = None,
        ax: 'plt.Axes | None' = None,
        figsize: tuple[float, float] = (11, 5),
        title: str | None = None,
        palette: str | list | dict | None = None,
        show_strip: bool = True,
        n_strip: int = 1000,
        annotate_n: bool = True,
        column_prefix: str = 'recall@',
        n_candidates: int | None = None,
        lift_log: bool = False,
    ) -> 'plt.Axes':
    """Grouped boxen of top-K recall across zero-shot similarity bins.

    Same per-bin x-axis structure as ``plot_zero_shot_performance`` but
    shows multiple K's side-by-side via hue. Each box = distribution of
    per-perturbation recall@K values for classes that fall into that
    bin. Lets you read off, at a glance, how K-resolution trades off
    against zero-shot novelty: tight gene-level recall (K=5) collapses
    fastest as you move toward more-novel bins, while broader top-K
    survives further.

    Parameters
    ----------
    per_class_df
        DataFrame with one row per class. Must include the recall
        columns named ``{column_prefix}{K}`` for each K in ``ks``, plus
        ``is_zero_shot`` and ``similarity_to_trained`` for binning.
        Typically: the ``report`` returned by
        ``model.get_prediction_output_report(...)`` (rows have
        recall@5, recall@50, recall@100).
    ks
        K values to display as grouped boxen.
    bins, seen_label
        Same semantics as ``plot_zero_shot_performance``.
    show_strip
        Overlay individual class points per (bin, K). Subsampled to 500
        per group.
    n_candidates
        Total number of candidate classes the model is choosing among (e.g.
        18614 for full-vocab predict with use_full_cls_emb=True). When set,
        a second panel is drawn next to the recall@K boxen showing
        ``recall@K × n_candidates / K`` — lift over the random baseline
        of picking K classes uniformly from n_candidates. At 18K candidates,
        recall@5 of 0.05 looks unimpressive on the raw scale but is a 180×
        lift over chance; the lift panel makes this readable.
    lift_log
        Plot the lift panel on a log y-axis. Helpful when the lift spread
        spans 2+ orders of magnitude (e.g. far-bin lifts of ~5× vs near-bin
        outliers at 500×). Zero-recall classes are clipped at the lower
        y-bound; the random=1× reference is drawn either way.
    """
    df = per_class_df.copy()
    cols = [f'{column_prefix}{k}' for k in ks]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing recall columns: {missing}. Available: "
            f"{[c for c in df.columns if c.startswith(column_prefix)]}"
        )
    if 'is_zero_shot' not in df.columns or 'similarity_to_trained' not in df.columns:
        raise KeyError(
            "per_class_df missing 'is_zero_shot' / 'similarity_to_trained' "
            "(needed for bin assignment)."
        )

    # Bin assignment — same logic as plot_zero_shot_performance.
    bin_edges = np.asarray(bins, dtype=float)
    bin_labels = [f'[{lo:.2f}-{hi:.2f}{")" if hi < bin_edges[-1] else "]"}'
                  for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    def _assign(row):
        if not row['is_zero_shot']:
            return seen_label
        s = row['similarity_to_trained']
        if not np.isfinite(s):
            return None
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            top = (i == len(bin_edges) - 2)
            if (s >= lo) and (s < hi or (top and np.isclose(s, hi))):
                return bin_labels[i]
        return None

    df['_bucket'] = df.apply(_assign, axis=1)
    df = df.dropna(subset=['_bucket'] + cols, how='all')

    # Melt to long: one row per (class, K) so boxenplot's hue works.
    melt = df[['_bucket'] + cols].melt(
        id_vars='_bucket', value_vars=cols, var_name='K', value_name='recall',
    ).dropna(subset=['recall'])
    # Clean up K labels: 'recall@5' → '5'
    melt['K'] = melt['K'].str.replace(column_prefix, '', regex=False)
    k_order = [str(k) for k in ks]

    bucket_order = [seen_label] + bin_labels
    bucket_order = [b for b in bucket_order if b in set(melt['_bucket'])]

    # Palette: one color per K. Default = viridis along the K axis so
    # tighter K (gene-level) is darker.
    if palette is None:
        palette = dict(zip(k_order, sns.color_palette('viridis', n_colors=len(k_order))))

    # Optional lift column. random_baseline_recall@K = K / n_candidates, so
    # lift = observed_recall / random_baseline = observed * n_candidates / K.
    if n_candidates is not None:
        k_int = melt['K'].astype(int)
        melt['lift'] = melt['recall'] * (float(n_candidates) / k_int)

    # Build axes: single panel by default, side-by-side when n_candidates
    # is given. ``ax`` is honored only for the single-panel case — a 2-panel
    # layout owns its own subplots.
    if n_candidates is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=120,
                                 gridspec_kw={'wspace': 0.25})
        ax_recall, ax_lift = axes
    else:
        if ax is None:
            fig, ax_recall = plt.subplots(figsize=figsize, dpi=120)
        else:
            ax_recall = ax
        ax_lift = None

    def _draw(target_ax, y_col, ylabel, ylim, hline_y, hline_label):
        """Draw the boxen + background strip + per-K seen-median refs onto
        ``target_ax`` for ``y_col``. Factored out so the recall and lift
        panels stay in sync visually."""
        if show_strip:
            strip_data = melt.groupby(['_bucket', 'K'], group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), n_strip), random_state=0)
            )
            sns.stripplot(
                data=strip_data, x='_bucket', y=y_col, hue='K',
                order=bucket_order, hue_order=k_order,
                dodge=True, ax=target_ax, color='black',
                size=2, alpha=0.3, jitter=0.2, legend=False, zorder=1,
            )
        sns.boxenplot(
            data=melt, x='_bucket', y=y_col, hue='K', order=bucket_order,
            hue_order=k_order, ax=target_ax, palette=palette, showfliers=False,
        )
        for patch in target_ax.patches:
            r, g, b, _ = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.75))
            patch.set_zorder(2)
        # Per-K seen-median dashed reference
        if seen_label in set(melt['_bucket']):
            for k_str, color in palette.items():
                seen_vals = melt.loc[
                    (melt['_bucket'] == seen_label) & (melt['K'] == k_str),
                    y_col,
                ]
                if len(seen_vals) == 0:
                    continue
                target_ax.axhline(float(seen_vals.median()),
                                  color=color, linestyle='--', linewidth=1, alpha=0.5)
        # Global reference: y=1 for the lift panel (chance), nothing extra for recall
        if hline_y is not None:
            target_ax.axhline(hline_y, color='gray', linestyle=':', linewidth=1)
            target_ax.text(
                len(bucket_order) - 0.5, hline_y, f' {hline_label}',
                color='gray', va='bottom', ha='right', fontsize=8,
            )
        target_ax.set_ylabel(ylabel)
        target_ax.set_xlabel('similarity bin (zero-shot) / trained')
        if ylim is not None:
            target_ax.set_ylim(*ylim)
        target_ax.legend(title='K', loc='upper right', frameon=True, fontsize=9)
        if annotate_n:
            counts_per_bucket = df.groupby('_bucket', observed=True).size()
            new_labels = [
                f'{b}\n(N={counts_per_bucket.get(b, 0)} cls)' for b in bucket_order
            ]
            target_ax.set_xticks(range(len(bucket_order)))
            target_ax.set_xticklabels(new_labels)

    _draw(ax_recall, 'recall',
          ylabel='per-class recall@K',
          ylim=(0, 1),
          hline_y=None, hline_label='')
    ax_recall.set_title('recall@K')
    if ax_lift is not None:
        # Auto y-limit: a touch above the 99th-percentile lift so the
        # box bodies stay readable even when a few outliers sit very high.
        y_max = float(np.nanpercentile(melt['lift'], 99.5))
        if lift_log:
            # On log scale, anchor the floor a hair below 1× (the random
            # baseline) so chance-level performance is right at the bottom
            # of the panel and the dotted reference reads cleanly. Anything
            # at 0 lift gets clipped, which is correct — log can't show 0.
            ylim = (0.5, max(y_max * 1.5, 10.0))
        else:
            ylim = (0, max(y_max * 1.1, 5.0))
        _draw(ax_lift, 'lift',
              ylabel=f'lift over random (1.0 = chance, n_cand={n_candidates})',
              ylim=ylim,
              hline_y=1.0, hline_label='random (1.0×)')
        if lift_log:
            ax_lift.set_yscale('log')
        ax_lift.set_title(f'recall@K × ({n_candidates}/K)  — lift over random')

    fig = ax_recall.figure
    fig.suptitle(title or 'Per-class top-K recall by zero-shot similarity', y=1.02)
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return ax_recall if ax_lift is None else (ax_recall, ax_lift)


def _resolve_palette(
    palette,
    groups: list[str],
) -> dict[str, str]:
    """Resolve various palette spec types to a {group: color} dict.

    Accepts:
      - None: Okabe-Ito (skip black/yellow) for ≤6 groups, falls back to
        seaborn 'colorblind' palette for >6.
      - str: name of a matplotlib/seaborn palette (e.g. 'viridis', 'tab10').
      - list/tuple of color strings: assigned in order.
      - dict {group: color}: used as-is, with missing groups filled.
    """
    n = len(groups)
    if isinstance(palette, dict):
        # Fill any missing keys from the colorblind default.
        backup = _resolve_palette(None, [g for g in groups if g not in palette])
        return {g: palette[g] if g in palette else backup[g] for g in groups}
    if isinstance(palette, (list, tuple)):
        cycle = list(palette)
        return {g: cycle[i % len(cycle)] for i, g in enumerate(groups)}
    if isinstance(palette, str):
        return dict(zip(groups, sns.color_palette(palette, n_colors=n)))
    # Default: Okabe-Ito with sensible reordering for typical N
    if n <= 6:
        # Drop indices 0 (black) and 4 (yellow); keep order otherwise.
        order = [2, 1, 5, 6, 3, 7]   # sky, orange, blue, vermillion, green, purple
        cycle = [OKABE_ITO[i] for i in order]
    else:
        cycle = list(sns.color_palette('colorblind', n_colors=n).as_hex())
    return {g: cycle[i % len(cycle)] for i, g in enumerate(groups)}
