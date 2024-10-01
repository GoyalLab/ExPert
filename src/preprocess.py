from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import logging


# credit to https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def quality_control_filter(adata, percent_threshold=20, nmads=5, mt_nmads=3, mt_per=8):
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
            | is_outlier(adata, 'pct_counts_in_top_20_genes', nmads)
    )
    # determine mitochondrial outliers
    adata.obs['mt_outlier'] = is_outlier(adata, 'pct_counts_mt', mt_nmads) | (
            adata.obs['pct_counts_mt'] > mt_per
    )
    # remove outliers
    logging.info(f'Total number of cells: {adata.n_obs}')
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)]
    logging.info(f'Number of cells after filtering of low quality cells: {adata.n_obs}')
    return adata


# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def prepare_dataset(adata, name='Unknown', qc=True, norm=True, log=True, scale=True, n_hvg=2000, subset=False):
    # apply quality control measures
    if qc:
        logging.info(f'Quality control for dataset {name}')
        adata = quality_control_filter(adata)
    if norm:
        logging.info(f'Normalizing dataset {name}')
        sc.pp.normalize_total(adata)
    # apply log transformation
    if log:
        logging.info(f'log1p normalizing dataset {name}')
        sc.pp.log1p(adata)

    logging.info(f'Determining highly variable genes for dataset {name}')
    if isinstance(n_hvg, float):
        if n_hvg > 1:
            raise ValueError('Percentage param "n_hvg" must be <= 1, or total number of genes')
        # take percentage of total genes in dataset instead of fixed number
        perc = n_hvg
        n_hvg = int(adata.n_vars * n_hvg)
        logging.info(f'Number of highly variable genes to use: {n_hvg} ({perc}* {adata.n_vars})')
    # Calculate highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=subset)
    if scale:
        logging.info(f'Scaling and centering {name}')
        sc.pp.scale(adata)
    logging.info(f'Found {np.sum(adata.var.highly_variable)} highly variable genes out of {adata.n_vars} total genes')
    return adata
