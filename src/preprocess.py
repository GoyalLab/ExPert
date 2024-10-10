from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import logging
from scipy import sparse
import scipy.sparse as sp


# credit to https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
# mt_nmads: Increased NMADS for mitochondrial counts
# mt_per: Maximum percentage of mitochondrial counts is increased to up to 20% to account for variablity in cancer cell lines
def quality_control_filter(adata, percent_threshold=20, nmads=5, mt_nmads=5, mt_per=20):
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
    logging.info(f'Number of cells after filtering for low quality cells: {adata.n_obs}')
    return adata


def _memory_check(ds, max_mem=30):
    # check size allocated by object after log-transform, convert to float16 if necessary
    num_elements = ds.X.shape[0] * ds.X.shape[1] if not sparse.issparse(ds.X) else ds.X.nnz
    mem_float32 = num_elements * np.dtype('float32').itemsize
    # Set maximum memory threshold
    memory_threshold = max_mem * 1024 ** 3

    # Decide on dtype based on memory requirements
    if mem_float32 <= memory_threshold:
        target_dtype = np.float32
    else:
        target_dtype = np.float16

    logging.info(f"Converted ds.X to {target_dtype} based on memory requirements. (max. Mem: {max_mem}GB)")
    # Convert ds.X to the decided dtype and make sure it's sparse
    if not sparse.issparse(ds.X):
        ds.X = sparse.csr_matrix(ds.X, dtype=target_dtype)
    else:
        ds.X = ds.X.astype(target_dtype)


# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def preprocess_dataset(adata, name='Unknown', qc=True, norm=True, log=True, scale=True, n_hvg=2000, subset=False, min_genes=10000):
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

    if isinstance(n_hvg, float):
        if n_hvg > 1:
            raise ValueError('Percentage param "n_hvg" must be <= 1, or total number of genes')
        # take percentage of total genes in dataset instead of fixed number
        perc = n_hvg
        n_hvg = int(adata.n_vars * n_hvg)
        logging.info(f'Number of highly variable genes to use: {n_hvg} ({perc}* {adata.n_vars})')
    # Calculate highly variable genes
    if adata.n_vars <= min_genes:
        logging.info(f'Selecting all genes for dataset {name}, because of low gene number: {adata.n_vars}')
        adata.var['highly_variable'] = True
    else:
        logging.info(f'Determining highly variable genes for dataset {name}')
        if not log:
            # Use seurat_v3 for raw counts
            logging.info(f'Using flavor "seurat_v3" when determining hvgs for raw counts')
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=subset, flavor='seurat_v3')
        else:
            # Use default options for normalized counts
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=subset)
    if scale:
        logging.info(f'Scaling and centering {name}')
        sc.pp.scale(adata)
    logging.info(f'Found {np.sum(adata.var.highly_variable)} highly variable genes out of {adata.n_vars} total genes')
    return adata
