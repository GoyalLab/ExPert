from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import pandas as pd
import logging
import anndata as ad


# credit to https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def is_outlier(
        adata: ad.AnnData, 
        metric: str, 
        nmads: int
    ) -> pd.Series:
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def quality_control_filter(
        adata: ad.AnnData,                  # input anndata object
        percent_threshold: int = 50,        # top x percent of counts to base calculations an
        nmads: int = 5,                     # Number of median absolute deviations
        mt_nmads: int = 3,                  # Number of median absolute deviations for mitochondrial counts
        mt_per: int = 20                    # Maximum percent of mitochondrial counts in cell (is higher for cancer cells)
    ) -> None:
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
    # calculate 
    adata.obs['mt_outlier'] = is_outlier(adata, 'pct_counts_mt', mt_nmads) | (
            adata.obs['pct_counts_mt'] > mt_per
    )
    # remove outliers
    logging.info(f'Total number of cells: {adata.n_obs}')
    adata._inplace_subset_obs((~adata.obs.outlier) & (~adata.obs.mt_outlier))
    logging.info(f'Number of cells after filtering for low quality cells: {adata.n_obs}')

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def preprocess_dataset(
        adata: ad.AnnData, 
        cancer: bool,
        name: str = 'Unknown', 
        qc: bool = True, 
        norm: bool = True, 
        log: bool = True, 
        scale: bool = True, 
        hvg: bool = True,
        n_hvg: int = 2000, 
        subset: bool = False, 
        min_genes: int = 5000
    ) -> ad.AnnData:
    # apply quality control measures
    if qc:
        logging.info(f'Quality control for dataset {name}')
        mt_percent = 25 if cancer else 12           # set threshold higher for cancer
        logging.info(f'Cancer: {cancer}, mt_per: {mt_percent}')
        quality_control_filter(adata, mt_per=mt_percent)
    if hvg:
        # Calculate highly variable genes
        if adata.n_vars <= min_genes:
            logging.info(f'Selecting all genes for dataset {name}, because of low gene number: {adata.n_vars}')
            adata.var['highly_variable'] = True
        else:
            logging.info(f'Determining highly variable genes for dataset {name}')
            # Use seurat_v3 for raw counts
            logging.info(f'Using flavor "seurat_v3" when determining hvgs for raw counts')
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=subset, flavor='seurat_v3')
    else:
        # set all genes to highly variable for downstream selection
        adata.var['highly_variable'] = True
    # normalize data
    if norm:
        logging.info(f'Normalizing dataset {name}')
        sc.pp.normalize_total(adata)
    # apply log transformation
    if log:
        logging.info(f'log1p normalizing dataset {name}')
        sc.pp.log1p(adata)
    # center and scale the data
    if scale:
        logging.info(f'Scaling and centering {name}')
        sc.pp.scale(adata)
    logging.info(f'Found {np.sum(adata.var.highly_variable)} highly variable genes out of {adata.n_vars} total genes')
    return adata
