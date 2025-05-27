from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from src.statics import P_COLS, CTRL_KEYS


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

def subset_ctrl_cells(
        adata: ad.AnnData, 
        n_ctrl: int = 10_000, 
        perturbation_col: str = 'perturbation',
        ctrl_key: str = 'control',
        seed: int = 42
    ) -> None:
    # Randomly pick N control cells out of pool of cells with control label
    ctrl_mask = adata.obs[perturbation_col]==ctrl_key
    ctrl_obs = adata.obs[ctrl_mask]
    # Keep all cells if number of control cells is smaller than given number
    if ctrl_obs.shape[0] < n_ctrl:
        logging.info(f'Number of control cells ({ctrl_obs.shape[0]}) < {n_ctrl}, keeping all cells.')
    else:
        logging.info(f'Randomly selecting {n_ctrl} control cells out of pool of {ctrl_obs.shape[0]} cells')
        # Sample control cells
        ctrl_idc = ctrl_obs.sample(n_ctrl, replace=False, random_state=seed).index
        # Include all perturbed cells
        p_idc = adata.obs[~ctrl_mask].index
        # Build final list of indices to filter for
        idc = ctrl_idc.tolist()
        idc.extend(p_idc)
        # Filter dataset
        adata._inplace_subset_obs(idc)

def single_perturbation_mask(meta: pd.DataFrame, p_col: str = 'perturbation'):
    # remove multiple perturbations
    all_perturbations = meta[p_col].str.split('_', expand=True)         # Expand perturbation label
    if all_perturbations.shape[1] < 2:
        return None
    col = all_perturbations.iloc[:, 1]                                  # Focus on second label
    mask = (
        col.isna() |                                    # Keep all empty second perturbations
        col.str.match(r'^\d+$|^pDS|^pBA')               # Keep all numbers and plasmids
    )
    return mask


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
        n_ctrl: int | None = 10_000,
        single_perturbations_only: bool = True,
        p_col: str = 'perturbation',
        ctrl_key: str = 'control',
        min_genes: int = 5_000,
        seed: int = 42
    ) -> ad.AnnData:
    # Check perturbation column
    if p_col not in adata.obs.columns:
        logging.info(f'"{p_col}" not found in adata.obs, looking for alternatives.')
        candidate_p_cols = adata.obs.columns.intersection(set(P_COLS))
        if len(candidate_p_cols) == 0:
            raise ValueError(f'Could not find an alternative perturbation column in adata. Looked for {P_COLS}')
        # Fall back to first hit
        adata.obs[p_col] = adata.obs[candidate_p_cols[0]]
        logging.info(f'Falling back to "{p_col}" as perturbation column')
    # Check control key
    if np.sum(adata.obs[p_col]==ctrl_key) == 0:
        logging.info(f'{ctrl_key} not found in "{p_col}", looking for alternatives.')
        all_ps = set(adata.obs[p_col].unique())
        candidate_ctrl_keys = all_ps.intersection(set(CTRL_KEYS))
        if len(candidate_ctrl_keys) == 0:
            raise ValueError(f'Could not find an alternative control key in "{p_col}". Looked for {CTRL_KEYS}')
        ctrl_key_hit = list(candidate_ctrl_keys)[0]
        adata.obs[p_col] = adata.obs[p_col].replace(ctrl_key_hit, ctrl_key)
        logging.info(f'Falling back to "{ctrl_key_hit}" as control key')
    # Filter for single perturbations only
    if single_perturbations_only:
        logging.info(f'Filtering column "{p_col}" for single perturbations only')
        sp_mask = single_perturbation_mask(adata.obs, p_col=p_col)
        if sp_mask is not None:
            adata._inplace_subset_obs(sp_mask)
    # Ensure adata.X is always in csr format
    if not isinstance(adata.X, sp.csr_matrix):
        adata.X = sp.csr_matrix(adata.X)
    # apply quality control measures
    if qc:
        logging.info(f'Quality control for dataset {name}')
        mt_percent = 25 if cancer else 12           # set threshold higher for cancer
        logging.info(f'Cancer: {cancer}, mt_per: {mt_percent}')
        quality_control_filter(adata, mt_per=mt_percent)
    if n_ctrl is not None:
        subset_ctrl_cells(adata, n_ctrl=n_ctrl, seed=seed, perturbation_col=p_col, ctrl_key=ctrl_key)
    if hvg:
        # Calculate highly variable genes
        if adata.n_vars <= min_genes:
            logging.info(f'Selecting all genes for dataset "{name}", because of low gene number: {adata.n_vars}')
            adata.var['highly_variable'] = True
        else:
            logging.info(f'Determining highly variable genes for dataset "{name}"')
            # Use seurat_v3 for raw counts
            logging.info(f'Using flavor "seurat_v3" when determining hvgs for raw counts')
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=subset, flavor='seurat_v3')
    else:
        # set all genes to highly variable for downstream selection
        adata.var['highly_variable'] = True
    # normalize data
    if norm:
        logging.info(f'Normalizing dataset "{name}"')
        sc.pp.normalize_total(adata)
    # apply log transformation
    if log:
        logging.info(f'log1p normalizing dataset "{name}"')
        sc.pp.log1p(adata)
    # center and scale the data
    if scale:
        logging.info(f'Scaling and centering "{name}"')
        sc.pp.scale(adata)
    logging.info(f'Found {np.sum(adata.var.highly_variable)} highly variable genes out of {adata.n_vars} total genes')
    return adata
