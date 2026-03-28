import logging

import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path
import numpy as np
from typing import Iterable


def _filter(d: ad.AnnData, dataset_name: str, hvg_pool: Iterable[str], zero_pad: bool = True) -> ad.AnnData:
    hvg_pool = list(hvg_pool)
    var_set = set(d.var.index)
    pool_in_data = [g for g in hvg_pool if g in var_set]

    logging.info(f'Found {len(pool_in_data)} pool genes in {dataset_name}')

    if not zero_pad:
        return d[:, pool_in_data].copy()

    n_missing = len(hvg_pool) - len(pool_in_data)
    if n_missing:
        logging.info(f'{n_missing} hvgs missing; padding with zeros')

    # Extract expression data for matching genes only
    X_sub = d[:, pool_in_data].X
    if sp.issparse(X_sub):
        X_sub = X_sub.tocsc()
    else:
        X_sub = sp.csc_matrix(X_sub)

    n_existing = len(pool_in_data)
    n_pool = len(hvg_pool)

    if n_missing == 0:
        X_final = X_sub.tocsr()
    else:
        # Use a sparse permutation matrix to place existing columns at pool positions
        # This avoids creating intermediate AnnData objects and concat/reorder copies
        pool_idx = {g: i for i, g in enumerate(hvg_pool)}
        target_cols = np.array([pool_idx[g] for g in pool_in_data])
        P = sp.csc_matrix(
            (np.ones(n_existing, dtype=X_sub.dtype), (np.arange(n_existing), target_cols)),
            shape=(n_existing, n_pool)
        )
        X_final = (X_sub @ P).tocsr()

    return ad.AnnData(
        X=X_final,
        obs=d.obs.copy(),
        var=pd.DataFrame(index=pd.Index(hvg_pool))
    )

def prepare_merge(
    input_pth: str,
    pool_genes: Iterable[str],
    out: str,
    zero_pad: bool = True,
    scores_file: str | None = None,
    **kwargs
) -> pd.DataFrame:
    # Check input file extension
    if not input_pth.endswith('.h5ad'):
        raise FileNotFoundError(f'File has to be .h5ad format, got {input_pth}')

    logging.info(f'Preparing file: {input_pth}')
    name = Path(input_pth).stem
    pool_genes = list(pool_genes)

    # Read backed to avoid loading full X into memory
    adata = ad.read_h5ad(input_pth, backed='r')
    obs = adata.obs.copy()
    var_names = adata.var_names.tolist()

    # Determine which pool genes exist in this dataset
    var_set = set(var_names)
    pool_in_data = [g for g in pool_genes if g in var_set]
    n_missing = len(pool_genes) - len(pool_in_data)
    logging.info(f'Found {len(pool_in_data)} pool genes in {name}')
    if n_missing:
        logging.info(f'{n_missing} hvgs missing; padding with zeros')

    # Read only the needed columns from disk into memory
    col_indices = [var_names.index(g) for g in pool_in_data]
    X_sub = adata.X[:, col_indices]
    if not sp.issparse(X_sub):
        X_sub = sp.csc_matrix(X_sub)
    else:
        X_sub = X_sub.tocsc()

    # Close backed file — no longer needed
    adata.file.close()
    del adata

    # Reorder/pad to match pool gene ordering
    if zero_pad and n_missing > 0:
        pool_idx = {g: i for i, g in enumerate(pool_genes)}
        target_cols = np.array([pool_idx[g] for g in pool_in_data])
        n_existing = len(pool_in_data)
        P = sp.csc_matrix(
            (np.ones(n_existing, dtype=X_sub.dtype), (np.arange(n_existing), target_cols)),
            shape=(n_existing, len(pool_genes))
        )
        X_final = (X_sub @ P).tocsr()
        var_index = pool_genes
    elif zero_pad:
        # No missing genes — reorder to match pool order
        pool_idx = {g: i for i, g in enumerate(pool_genes)}
        order = np.argsort([pool_idx[g] for g in pool_in_data])
        X_final = X_sub[:, order].tocsr()
        var_index = pool_genes
    else:
        X_final = X_sub.tocsr()
        var_index = pool_in_data

    del X_sub

    # add per-cell efficiency scores if not None and is csv
    if scores_file is not None and str(scores_file).endswith('.csv'):
        efficiency_scores_df = pd.read_csv(scores_file, index_col=0)
        idx = obs.index.intersection(efficiency_scores_df.index)
        assert idx.shape[0] == obs.shape[0], "Score indices and adata barcodes don't match."
        scores = efficiency_scores_df.loc[idx]
        obs = pd.concat([obs, scores], axis=1)
        logging.info(f'Added efficicency scores from: {scores_file}')

    # make cell barcodes unique to dataset
    obs.index = obs.index + ';' + name
    obs['dataset'] = name

    # Build output AnnData
    adata = ad.AnnData(
        X=X_final,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(var_index))
    )

    # Filter for a minimum number of cells per perturbation
    mcpp = kwargs.get('min_cells_per_perturbation', 0)
    if mcpp > 0:
        p_col = kwargs.get('perturbation_col', 'perturbation')
        cpp = adata.obs[p_col].value_counts()
        filtered_perturbations = cpp[cpp >= mcpp].index
        filter_mask = adata.obs[p_col].isin(filtered_perturbations)
        if filter_mask.sum() < adata.shape[0]:
            adata._inplace_subset_obs(filter_mask)
            logging.info(f'Filtering for a minimum of {mcpp} cells per perturbation. N perturbations: {filtered_perturbations.shape[0]}/{cpp.shape[0]}')

    # save prepared adata
    adata.write_h5ad(out)
    return adata.obs
