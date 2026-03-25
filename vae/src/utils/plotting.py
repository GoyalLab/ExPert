import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from typing import Literal
from pandas.api.types import is_numeric_dtype

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
                s=80,
                edgecolor="black",
                linewidth=0.6,
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

def plot_confusion_full(
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        figsize: tuple[int, int] = (10, 8),
        hm_kwargs: dict = {'annot': False},
        verbose: bool = False,
        plt_file: str | None = None,
        ref_labels: np.ndarray | list | None = None
    ) -> None:

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    all_labels = np.unique(np.concatenate([true_labels, pred_labels]))

    # ---------------------------------------------------
    # ORDERING LOGIC
    # ---------------------------------------------------
    if ref_labels is not None:
        ref_labels = np.asarray(ref_labels)

        # Use ref_labels order, but keep only those appearing in data
        ordered_from_ref = [lab for lab in ref_labels if lab in all_labels]

        # Add any prediction-only labels NOT in ref_labels at the end
        remaining = [lab for lab in all_labels if lab not in ordered_from_ref]

        ordered_pred = np.array(ordered_from_ref + remaining)
        ordered_true = np.array([lab for lab in ordered_pred if lab in true_labels])

    else:
        # ---------- DEFAULT BEHAVIOR ----------
        intersect = np.intersect1d(true_labels, pred_labels)
        true_only = np.setdiff1d(true_labels, intersect)
        pred_only = np.setdiff1d(pred_labels, intersect)

        ordered_true = np.concatenate([intersect, true_only])
        ordered_pred = np.concatenate([intersect, pred_only])

    # ---------------------------------------------------
    # Compute confusion matrix with ordered_pred as columns
    # ---------------------------------------------------
    cm_full = confusion_matrix(y_true, y_pred, labels=ordered_pred)

    # Keep rows that correspond to ordered_true
    row_mask = np.isin(ordered_pred, ordered_true)
    cm = cm_full[row_mask, :]
    row_labels = ordered_pred[row_mask]

    # ---------------------------------------------------
    # Normalize rows
    # ---------------------------------------------------
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percentage = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    ) * 100

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    if verbose:
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1:        {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    # ---------------------------------------------------
    # Plot heatmap
    # ---------------------------------------------------
    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        cm_percentage,
        xticklabels=ordered_pred,
        yticklabels=row_labels,
        square=True,
        cbar=True,
        cbar_kws={"orientation": "horizontal", "pad": 0.15},
        **hm_kwargs
    )

    ax.set_aspect("equal")

    # ---------------------------------------------------
    # Draw green rectangles for diagonals
    # ---------------------------------------------------
    for i, true_lab in enumerate(row_labels):
        if true_lab in ordered_pred:
            j = np.where(ordered_pred == true_lab)[0][0]
            ax.add_patch(
                Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5
                )
            )

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    if plt_file:
        plt.savefig(plt_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
