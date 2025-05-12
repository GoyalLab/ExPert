import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from typing import Literal
import logging
from scipy.stats import median_abs_deviation


def get_abs_dev(M, nmads: int = 2) -> tuple:
    mad = median_abs_deviation(M)
    m = np.median(M)
    neg = m - nmads * mad
    pos = m + nmads * mad
    return neg, pos


# credit to https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    n, p = get_abs_dev(M, nmads)
    outlier = (M < n) | (p < M)
    return outlier

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def quality_control_filter(
        adata: ad.AnnData,
        percent_threshold: int = 20, 
        nmads: int = 5, 
        mt_nmads: int = 3, 
        mt_per: int = 10,
        inplace: bool = False,
        return_mask: bool = True
    ) -> None | pd.Series:
    # ensure unique variable names
    adata.var_names_make_unique()
    # mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    # ribosomal genes
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    # hemoglobin genes
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')
    # calculate qc metrics
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt', 'ribo', 'hb'],
        inplace=True, percent_top=[percent_threshold],
        log1p=True
    )
    # determine outliers
    adata.obs['outlier'] = (
            is_outlier(adata, 'log1p_total_counts', nmads)
            | is_outlier(adata, 'log1p_n_genes_by_counts', nmads)
            | is_outlier(adata, f'pct_counts_in_top_{percent_threshold}_genes', nmads)
    )
    # determine mitochondrial outliers
    adata.obs['mt_outlier'] = is_outlier(adata, 'pct_counts_mt', mt_nmads) | (
            adata.obs['pct_counts_mt'] > mt_per
    )
    # remove outliers
    mask = (~adata.obs.outlier) & (~adata.obs.mt_outlier)
    adata.obs['qc_mask'] = mask
    if inplace:
        adata._inplace_subset_obs(mask)
    if return_mask:
        return mask
    