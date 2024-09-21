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
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    # calculate qc metrics
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"],
        inplace=True, percent_top=[percent_threshold],
        log1p=True
    )
    # determine outliers
    adata.obs["outlier"] = (
            is_outlier(adata, "log1p_total_counts", nmads)
            | is_outlier(adata, "log1p_n_genes_by_counts", nmads)
            | is_outlier(adata, "pct_counts_in_top_20_genes", nmads)
    )
    # determine mitochondrial outliers
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", mt_nmads) | (
            adata.obs["pct_counts_mt"] > mt_per
    )
    # remove outliers
    logging.info(f"Total number of cells: {adata.n_obs}")
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)]
    logging.info(f"Number of cells after filtering of low quality cells: {adata.n_obs}")
    return adata


# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def normalize(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)


def hvg_filter(adata):
    sc.pp.highly_variable_genes(adata)  # calculate highly variable genes
    return adata[:, adata.var.highly_variable]


def prepare_dataset(adata, name='Unknown', qc=True, hvg=True):
    if qc:
        logging.info(f"Quality control for dataset {name}")
        adata = quality_control_filter(adata)                                   # QC
    logging.info(f"Normalizing dataset {name}")
    normalize(adata)                                                            # normalize
    if hvg:
        logging.info(f"Reducing dataset {name} to highly variable genes")
        hvg_filter(adata)                                                       # HVG filter
    return adata
