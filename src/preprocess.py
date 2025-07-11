from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from typing import Iterable
from src.statics import P_COLS, CTRL_KEYS, GENE_SYMBOL_KEYS, SETTINGS, OBS_KEYS


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
    # keep perturbation version or plasmid name
    if all_perturbations.shape[1] < 2:
        return None
    col = all_perturbations.iloc[:, 1]                                  # Focus on second label
    mask = (
        col.isna() |                                    # Keep all empty second perturbations
        col.str.match(r'^\d+$|^pDS|^pBA')               # Keep all numbers and plasmids
    )
    return mask

def clean_perturbation_labels(p: pd.Series, keep_versions: bool = False) -> pd.Categorical:
    # Remove guide prefixes
    clean_p = p.str.replace(r'^ES.sg\d*.', '', regex=True)
    # Remove versions
    if not keep_versions:
        clean_p = clean_p.str.split('_').str[0]
    return pd.Categorical(clean_p.values)

def ens_to_symbol(adata: ad.AnnData) -> ad.AnnData:
    # Look for possible gene symbol columns
    gscl = adata.var.columns.intersection(set(GENE_SYMBOL_KEYS)).values
    if len(gscl) == 0:
        raise ValueError(f'Could not find a column that describes gene symbol mappings in adata.var, looked for {GENE_SYMBOL_KEYS}')
    # Choose first hit if multiple
    gsh = list(gscl)[0]
    # Convert index
    adata.var.reset_index(names='ensembl_id', inplace=True)
    adata.var.set_index(gsh, inplace=True)
    # Check for duplicate index conflicts
    if adata.var_names.nunique() != adata.shape[0]:
        logging.info(f'Found duplicate indices for ensembl to symbol mapping, highest number of conflicts: {adata.var_names.value_counts().max()}')
        # Fix conflicts by choosing the gene with the higher harmonic mean of mean expression and normalized variance out of pool
        if len(set(['means', 'variances_norm']).intersection(adata.var.columns)) == 2:
            adata.var['hm_var'] = (2 * adata.var.means * adata.var.variances_norm) / (adata.var.means + adata.var.variances_norm)
        else:
            adata.var['hm_var'] = np.arange(adata.n_vars)
        idx = adata.var.reset_index().groupby(gsh, observed=True).hm_var.idxmax().values
        adata = adata[:,idx]
    return adata

def _find_match_idx(target: Iterable[str], ref: Iterable[str]) -> int:
    t = np.array(target)
    r = np.array(ref)
    target_idc, _ = np.where(t.reshape(-1, 1)==r)
    if target_idc.shape[0] == 0:
        raise ValueError(f'Could not find match between target: {t} and reference: {r}.')
    return target_idc[0]

def check_p_col(adata: ad.AnnData, p_col: str = 'perturbation') -> None:
    if p_col not in adata.obs.columns:
        logging.info(f'"{p_col}" not found in adata.obs, looking for alternatives.')
        p_col_idx = _find_match_idx(adata.obs.columns.str.lower(), ref=P_COLS)
        p_col_hit = adata.obs.columns[p_col_idx]
        adata.obs[p_col] = adata.obs[p_col_hit]
        logging.info(f'Found "{p_col_hit}" as perturbation column')

def check_ctrl_col(adata: ad.AnnData, p_col: str = 'perturbation', ctrl_key: str = 'control') -> None:
    if np.sum(adata.obs[p_col]==ctrl_key) == 0:
        logging.info(f'{ctrl_key} not found in "{p_col}", looking for alternatives.')
        all_ps = adata.obs[p_col].unique()
        ctrl_idx = _find_match_idx(pd.Series(all_ps).str.lower(), ref=CTRL_KEYS)
        ctrl_key_hit = all_ps[ctrl_idx]
        adata.obs[p_col] = pd.Series(adata.obs[p_col].str.replace(ctrl_key_hit, ctrl_key))
        logging.info(f'Found "{ctrl_key_hit}" as control key')

def get_adata_meta(adata: ad.AnnData, p_col: str = 'perturbation', ctrl_key: str = 'control', single_perturbations_only: bool = True):
    check_p_col(adata, p_col=p_col)
    # Check control key
    check_ctrl_col(adata, p_col=p_col, ctrl_key=ctrl_key)
    # Filter for single perturbations only
    obs = adata.obs.copy()
    if single_perturbations_only:
        logging.info(f'Filtering column "{p_col}" for single perturbations only')
        sp_mask = single_perturbation_mask(obs, p_col=p_col)
        if sp_mask is not None:
            obs = obs[sp_mask].copy()
    obs[p_col] = clean_perturbation_labels(obs[p_col], keep_versions=False)
    # Extract .var
    var = adata.var.copy()
    if adata.var_names.str.lower().str.startswith('ens').all():
        logging.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        var = ens_to_symbol(adata).var.copy()
    return obs, var

def _filter_perturbation_pool(adata: ad.AnnData, perturbation_pool_file: str | None, p_col: str = OBS_KEYS.PERTURBATION_KEY) -> None:
    if perturbation_pool_file is None:
        return None
    # Load pool and filter for hits in adata
    perturbation_pool = pd.read_csv(perturbation_pool_file, index_col=0)[OBS_KEYS.POOL_PERTURBATION_KEY]
    mask = adata.obs[p_col].isin(perturbation_pool)
    adata._inplace_subset_obs(mask)
    logging.info(f'Found {adata.obs[p_col].nunique()} perturbations from pool ({perturbation_pool.shape[0]}) in adata.')

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def preprocess_dataset(
        adata: ad.AnnData, 
        cancer: bool,
        perturbation_pool_file: str | None = None,
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
        p_col: str = OBS_KEYS.PERTURBATION_KEY,
        ctrl_key: str = OBS_KEYS.CTRL_KEY,
        min_genes: int = 5_000,
        keep_versions: bool = False,
        seed: int = 42
    ) -> ad.AnnData:
    # Ensure adata.X is in csr format
    if not isinstance(adata.X, sp.csr_matrix):
        logging.info('Converting adata.X to CSR matrix.')
        adata.X = sp.csr_matrix(adata.X)
    # Check perturbation column
    check_p_col(adata, p_col=p_col)
    # Check control key
    check_ctrl_col(adata, p_col=p_col, ctrl_key=ctrl_key)
    # Clean up perturbation label
    adata.obs[p_col] = clean_perturbation_labels(adata.obs[p_col], keep_versions=keep_versions)
    # Filter for single perturbations only
    if single_perturbations_only:
        logging.info(f'Filtering column "{p_col}" for single perturbations only')
        sp_mask = single_perturbation_mask(adata.obs, p_col=p_col)
        if sp_mask is not None:
            adata._inplace_subset_obs(sp_mask)
    # Filter perturbations for perturbation pool if given
    _filter_perturbation_pool(adata, perturbation_pool_file=perturbation_pool_file, p_col=p_col)
    if n_ctrl is not None:
        subset_ctrl_cells(adata, n_ctrl=n_ctrl, seed=seed, perturbation_col=p_col, ctrl_key=ctrl_key)
    # apply quality control measures
    if qc:
        logging.info(f'Quality control for dataset {name}')
        mt_percent = SETTINGS.MT_PERCENT_CANCER if cancer else SETTINGS.MT_PERCENT_NORMAL           # set threshold higher for cancer
        logging.info(f'Cancer: {cancer}, mt_per: {mt_percent}')
        quality_control_filter(adata, mt_per=mt_percent)
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
    # check if .var indices are gene symbols or ensembl ids
    if adata.var_names.str.lower().str.startswith('ens').all():
        logging.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        adata = ens_to_symbol(adata).copy()
    return adata
