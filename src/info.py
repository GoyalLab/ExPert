import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Iterable, Any
import matplotlib.pyplot as plt
import seaborn as sns

from src.statics import OBS_KEYS
from src.preprocess import get_adata_meta


def _overlap_hm(
        groups: Iterable[str], 
        unique_values: Iterable[Any],
        title: str = 'Overlap Heatmap', 
        return_matrix: bool = False,
        plt_dir: str | None = None
    ) -> np.ndarray | None:
    # Compute pairwise overlaps
    overlap_matrix = np.zeros((len(groups), len(groups)))
    total_overlap_matrix = np.zeros((len(groups), len(groups)))

    for i in np.arange(len(groups)):
        for j in np.arange(len(groups)):
            both = len(set(unique_values[i]).intersection(unique_values[j]))
            n_g1 = len(set(unique_values[i]))
            perc = np.round(both/n_g1*100, 1)
            overlap_matrix[i, j] = perc
            total_overlap_matrix[i, j] = both

    # Create a heatmap
    fig = plt.figure(dpi=300, figsize=(14,10))
    ax = sns.heatmap(overlap_matrix.T, xticklabels=groups, yticklabels=groups, fmt='g', annot=True)
    cbar = ax.collections[0].colorbar  # Get the colorbar object
    cbar.set_label('Overlap percentage', rotation=270, labelpad=20, fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(fontsize=14)
    if plt_dir is None:
        plt.show()
    else:
        o = os.path.join(plt_dir, f'{title}_overlap_hm.svg')
        plt.savefig(o, bbox_inches='tight')
    if return_matrix:
        return overlap_matrix

def collect_meta_from_files(input_files: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    meta = []
    vars = dict()
    for file in input_files:
        if file.endswith('.h5ad'):
            logging.info(f'Reading meta data of {file}')
            adata = sc.read(file, backed='r')
            m, v = get_adata_meta(adata)
            ds_name = os.path.basename(file).split('.h5ad')[0]
            m['dataset'] = ds_name
            vars[ds_name] = v
            m.index = m.index.astype(str) + '_' + m['dataset'].astype(str)
            meta.append(m)
            adata.file.close()
    meta = pd.concat(meta, axis=0)
    return meta, vars

def _plot_cpp(meta: pd.DataFrame, plt_dir: str | None = None, by: str = 'dataset', remove_ctrl: bool = True):
    # check number of cells/perturbation
    df = meta.groupby([by, 'perturbation']).size().reset_index()
    df.columns = [by, 'perturbation', 'n_cells']
    df = df[df.n_cells > 0]
    df['n_cells_log'] = np.log(df.n_cells)
    if remove_ctrl:
        df = df[df.perturbation != 'control'].copy()

    # order datasets by median log count
    order = df.groupby(by)['n_cells_log'].median().sort_values(ascending=False).index
    df[by] = pd.Categorical(df[by], categories=order, ordered=True)

    # total cells per dataset
    total_counts = df.groupby(by)['n_cells'].sum().reindex(order)

    # total perturbations per dataset (context)
    total_perturbations = (
        df.groupby(by)['perturbation']
        .nunique()
        .reindex(order)
    )

    # --- plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(13, len(order) * 0.4),
        dpi=300,
        sharey=True,
        gridspec_kw={'width_ratios': [3, 1, 1]}
    )

    # --- left: boxenplot of per-perturbation log counts ---
    sns.boxenplot(
        data=df,
        x='n_cells_log',
        y=by,
        hue=by,
        palette='tab10',
        dodge=False,
        legend=False,
        ax=ax1
    )
    ax1.set_xlabel('Number of cells per perturbation (log-scaled)')
    ax1.set_ylabel('')
    ax1.set_title('Distribution')

    # --- right: barplot of total counts per dataset (log scale) ---
    colors = sns.color_palette('tab10', n_colors=len(order))
    bars = sns.barplot(
        x=total_counts.values,
        y=total_counts.index,
        palette=colors,
        ax=ax2
    )

    # log scale
    ax2.set_xscale('log')
    ax2.set_xlabel('Total cells (log scale)')
    ax2.set_title('#Cells')

    # add numeric labels to each bar
    for bar, value in zip(bars.patches, total_counts.values):
        ax2.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f'{int(value):,}',  # formatted with thousands separator
            va='center',
            ha='left',
            fontsize=8
        )
        
    bars2 = sns.barplot(
        x=total_perturbations.values,
        y=total_perturbations.index,
        palette=colors,
        ax=ax3
    )

    ax3.set_xlabel('Total perturbations')
    ax3.set_title('#Perturbations')

    # add numeric labels
    for bar, value in zip(bars2.patches, total_perturbations.values):
        ax3.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f'{int(value)}',
            va='center',
            ha='left',
            fontsize=8
        )

    # adjust layout / aesthetics
    sns.despine(ax=ax1, left=True)
    sns.despine(ax=ax2, left=True)
    sns.despine(ax=ax3, left=True)
    plt.tight_layout()
    plt.show()
    if plt_dir is None:
        plt.show()
    else:
        o = os.path.join(plt_dir, f'cells_per_perturbation_{by}.svg')
        plt.savefig(o, bbox_inches='tight')

def _plt_n_cells_per_dataset(meta: pd.DataFrame, plt_dir: str | None = None):
    df = pd.DataFrame(meta.dataset.value_counts()).reset_index()
    fig = plt.figure(dpi=300)
    ax = sns.barplot(df, x='count', y='dataset', orient='y', hue='count', order=df.dataset.values, legend=False)
    for c in ax.containers:
        ax.bar_label(c, fontsize=10)
    plt.title('Number of cells per experiment')
    plt.xlabel('Number of cells')
    plt.text(s=f'Total number of cells: {meta.shape[0]}', x=1e6, y=10)
    plt.ylabel('')
    if plt_dir is None:
        plt.show()
    else:
        o = os.path.join(plt_dir, 'cells_per_dataset.svg')
        plt.savefig(o, bbox_inches='tight')

def _plt_n_cells_per_dataset(meta: pd.DataFrame, plt_dir: str | None = None):
    df = meta.groupby('dataset').perturbation.nunique().reset_index().sort_values('perturbation', ascending=False)
    fig = plt.figure(dpi=300)
    ax = sns.barplot(df, x='perturbation', y='dataset', orient='y', hue='perturbation', order=df.dataset.values, legend=False)
    for c in ax.containers:
        ax.bar_label(c, fontsize=10)
    plt.title('Number of perturbations per dataset (excl. gene combinations)')
    plt.xlabel('Number of unique perturbations')
    plt.ylabel('')
    if plt_dir is None:
        plt.show()
    else:
        o = os.path.join(plt_dir, 'perturbations_per_dataset.svg')
        plt.savefig(o, bbox_inches='tight')

def _plt_upset(meta: pd.DataFrame, plt_dir: str | None = None, min_set_size: int = 100):
    try:
        from upsetplot import UpSet, from_contents
    except ImportError:
        logging.warning(f'Install "upsetplot" to plot an UpSet plot. Skipping the plot.')
        return None
    # Define input sets
    sets = meta.groupby('dataset', observed=True)['perturbation'].apply(lambda x: set(x.unique()))
    data = from_contents(sets)
    # Plot
    plt.figure(dpi=300)
    us = UpSet(data, subset_size="count", sort_by="cardinality", min_subset_size=min_set_size, show_counts=True)
    us.plot()
    if plt_dir is None:
        plt.show()
    else:
        o = os.path.join(plt_dir, 'perturbation_intersection_upset.svg')
        plt.savefig(o, bbox_inches='tight')

def _plt_feature_upset(var_dict: dict[str, pd.DataFrame], plt_dir: str | None = None, min_set_size: int = 100) -> None:
    """Plot UpSet plot for feature overlap between datasets."""
    try:
        from upsetplot import UpSet, from_contents
    except ImportError:
        logging.warning(f'Install "upsetplot" to plot an UpSet plot. Skipping the plot.')
        return None
    # Define input sets
    sets = {k: set(df.index) for k, df in var_dict.items()}
    data = from_contents(sets)
    # Plot
    plt.figure(dpi=300)
    us = UpSet(data, subset_size="count", sort_by="cardinality", min_subset_size=min_set_size, show_counts=True)
    us.plot()
    if plt_dir is None:
        plt.show()
    else:
        o = os.path.join(plt_dir, 'feature_intersection_upset.svg')
        plt.savefig(o, bbox_inches='tight')

def _plt_meta_features(obs: pd.DataFrame, vars: dict[str, pd.DataFrame], plt_dir: str | None = None):
    logging.info('Plotting feature overlap.')
    uv = [set(v.index) for v in vars.values()]
    _overlap_hm(groups=obs.dataset.unique(), unique_values=uv, title='features', plt_dir=plt_dir)
    _plt_feature_upset(vars, plt_dir=plt_dir)
    logging.info('Plotting perturbation overlap.')
    uv = obs.groupby('dataset')['perturbation'].unique()
    _overlap_hm(groups=uv.index, unique_values=uv, title='perc_perturbation', plt_dir=plt_dir)
    logging.info('Plotting cells per perturbation distribution.')
    _plot_cpp(obs, plt_dir=plt_dir, by=OBS_KEYS.DATASET_KEY)
    _plot_cpp(obs, plt_dir=plt_dir, by=OBS_KEYS.CONTEXT_KEY)
    logging.info('Plotting intersection between perturbations in datasets.')
    _plt_upset(obs, plt_dir=plt_dir)

def _calc_pool_datasets(
        meta: pd.DataFrame, 
        min_set_size: int = 1000, 
        min_size: int = 2,
        normalize: bool = False,
        key: str = OBS_KEYS.CONTEXT_KEY,
    ) -> list[str]:
    from itertools import combinations

    # Define all sets of unique perturbation labels
    sets = meta.groupby(key, observed=True)[OBS_KEYS.PERTURBATION_KEY].apply(lambda x: set(x.unique()))
    all_sets = list(sets.keys())
    # Calculate perturbation label intersection between all combinations of datasets
    combos, set_sizes, n_sets, intersections = [], [], [], []
    for r in range(min_size, len(all_sets) + 1):
        for combo in combinations(all_sets, r):
            inter = set.intersection(*(sets[name] for name in combo))
            n = len(inter)
            # Only keep sets that have a minimal size
            if n < min_set_size:
                continue
            intersections.append(inter)
            combos.append(combo)
            n_sets.append(len(combo))
            set_sizes.append(n)
    intersections = np.array(intersections, dtype=object)
    combos = np.array(combos, dtype=object)
    set_sizes = np.array(set_sizes)
    n_sets = np.array(n_sets)
    # Choose best combination by normalized harmonic mean of number of datasets and number of perturbations
    if normalize:
        n_sets = n_sets / n_sets.max()
        set_sizes = set_sizes / set_sizes.max()
    hm = 2 * n_sets * set_sizes / (n_sets + set_sizes)
    # Choose best pool values
    best_idx = np.argmax(hm)
    chosen_ss, chosen_n = set_sizes[best_idx], n_sets[best_idx]
    pool_contexts = combos[best_idx]
    perturbation_pool = intersections[best_idx]
    logging.info(f'Found optimal perturbation pool between {chosen_n} contexts with {chosen_ss} shared perturbations.')
    logging.info(f'Chosen contexts: {list(pool_contexts)}')
    logging.info(f'Pool dim: {len(perturbation_pool)} gene-perturbations')
    return sorted(list(perturbation_pool))

def _calc_feature_pool(vars: dict[str, pd.DataFrame]) -> None:
    """Calculate overlap between indices of all datasets."""
    feature_pool = set.intersection(*[set(v.index) for v in vars.values()])
    logging.info(f'Found {len(feature_pool)} shared features across all datasets.')
    return feature_pool

def create_meta_summary(
        input_files: list[str],
        perturbation_pool_file: str,
        feature_pool_file: str,
        dataset_sheet: pd.DataFrame,
        min_cells_per_class: int = 50,
        plt_dir: str | None = None,
        plot: bool = True,
    ):
    # Collect meta data for all available datasets
    obs, vars_dict = collect_meta_from_files(input_files)
    
    # Add dataset meta information to observations
    obs = obs.merge(dataset_sheet, left_on=OBS_KEYS.DATASET_KEY, right_index=True, how='left')
    # Filter obs by minimum number of cells per perturbation within a context
    cpp = obs.groupby(OBS_KEYS.CONTEXT_KEY)[OBS_KEYS.PERTURBATION_KEY].value_counts()
    # Filter each context-specific perturbation for minimum number of cells
    valid_classes = cpp[cpp >= min_cells_per_class].reset_index()
    # Get valid combinations of dataset name and perturbation
    valid_combinations = (valid_classes[OBS_KEYS.CONTEXT_KEY].astype(str) + '-' + valid_classes[OBS_KEYS.PERTURBATION_KEY].astype(str)).unique()
    # Filter all cells for valid combinations
    all_combinations = obs[OBS_KEYS.CONTEXT_KEY].astype(str) + '-' + obs[OBS_KEYS.PERTURBATION_KEY].astype(str)
    obs_mask = all_combinations.isin(valid_combinations)
    obs = obs[obs_mask].copy()
    
    # Get union of all perturbations in the datasets
    perturbation_union = pd.DataFrame(obs[OBS_KEYS.PERTURBATION_KEY].unique(), columns=[OBS_KEYS.PERTURBATION_KEY])
    union_file = os.path.join(os.path.dirname(perturbation_pool_file), 'all_perturbations.csv')
    perturbation_union.to_csv(union_file)

    # Calculate optimal perturbation pool
    perturbation_pool = _calc_pool_datasets(obs)
    # Save pool to output file
    pp = pd.DataFrame(perturbation_pool, columns=[OBS_KEYS.POOL_PERTURBATION_KEY])
    pp.to_csv(perturbation_pool_file)

    # Calculate feature overlap and save to file
    feature_pool = _calc_feature_pool(vars_dict)
    fp = pd.DataFrame(feature_pool, columns=[OBS_KEYS.POOL_FEATURE_KEY])
    fp.to_csv(feature_pool_file)

    # Choose to plot or not
    if plot:
        # Create plot directory if given
        if plt_dir is not None:
            os.makedirs(plt_dir, exist_ok=True)
        # Plot all basic meta related information about datasets
        _plt_meta_features(obs, vars_dict, plt_dir=plt_dir)
    