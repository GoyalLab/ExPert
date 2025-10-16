from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from typing import Iterable
from sklearn.covariance import EmpiricalCovariance

from src.statics import P_COLS, CTRL_KEYS, GENE_SYMBOL_KEYS, SETTINGS, OBS_KEYS, MISC_LABELS_KEYS


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

def _preprocess(
        adata: ad.AnnData,
        norm: bool = True,
        log1p: bool = True,
        pca: bool = True,
        neighbors: bool = True,
        umap: bool = False,
        verbose: bool = True,
        return_adata: bool = False
    ) -> None:
    if not sp.issparse(adata.X):
        logging.info('Converting .X to CSR.')
        adata.X = sp.csr_matrix(adata.X)
    if norm and verbose:
        logging.info('Normalizing adata.')
        adata.layers['norm'] = sc.pp.normalize_total(adata, inplace=False)['X']
    if log1p and verbose:
        logging.info('Log1p transforming adata.')
        adata.layers['norm'] = sc.pp.log1p(adata.layers['norm'])
    if pca and verbose:
        logging.info('Calculating PCA.')
        sc.pp.pca(adata, layer='norm')
    if neighbors and verbose:
        logging.info('Calculating neighbors.')
        sc.pp.neighbors(adata)
    if umap and verbose:
        logging.info('Calculating UMAP.')
        sc.tl.umap(adata)
    if return_adata:
        return adata

def _neighborhood_purity_sparse(
    labels: np.ndarray,
    conn: sp.csr_matrix,
    threshold: float = 0.5,
    ignore_class: str = 'control',
) -> tuple[np.ndarray, np.ndarray]:
    if not sp.issparse(conn):
        conn = sp.csr_matrix(conn)

    n = len(labels)
    frac_same = np.zeros(n, dtype=np.float32)

    # Convert to CSR for fast row access
    conn = conn.tocsr()

    # Compute denominator: total neighborhood weight
    total_conn = np.array(conn.sum(axis=1)).flatten()

    # Build mapping of label → indices
    label_to_idx = {}
    for i, lbl in enumerate(labels):
        label_to_idx.setdefault(lbl, []).append(i)

    # Iterate only over classes (not individual cells!)
    for lbl, idx in label_to_idx.items():
        if lbl == ignore_class:
            continue
        idx = np.array(idx)
        # Subset connectivity rows for this class
        sub_conn = conn[idx]
        # Restrict columns to same class
        col_mask = np.zeros(n, dtype=bool)
        col_mask[label_to_idx[lbl]] = True
        sub_conn_same = sub_conn[:, col_mask]

        # Sum same-class connectivity per cell
        same_sum = np.array(sub_conn_same.sum(axis=1)).flatten()
        frac_same[idx] = same_sum / (total_conn[idx] + 1e-12)  # avoid /0

    # Compute mask
    mask = frac_same >= threshold

    # Ignore class always kept
    if ignore_class is not None and ignore_class in label_to_idx:
        mask[label_to_idx[ignore_class]] = True

    kept = mask.sum()
    total = (~(labels == ignore_class)).sum()
    logging.info(f'Keeping {kept} / {total} cells ({100 * kept / total:.1f}%)')

    return mask, frac_same

def neighborhood_purity(
    adata: ad.AnnData, 
    label_key: str = 'perturbation', 
    threshold: float = 0.1,
    ignore_class: str | None = 'control', 
    new_label_key: str = 'label',
    unknown_key: str = 'unknown',
) -> None:
    if 'connectivities' not in adata.obsp:
        # Pre-process adata
        _preprocess(adata)

    conn = adata.obsp['connectivities']
    # Ensure connectivities are sparse
    if not sp.issparse(conn):
        conn = sp.csr_matrix(conn)
    # Get list of all perturbation labels in adata
    labels = np.array(adata.obs[label_key])
    mask, frac_same = _neighborhood_purity_sparse(labels=labels, conn=conn, threshold=threshold, ignore_class=ignore_class)
    # Save mask to adata
    adata.obs['full_frac_mask'] = mask
    # Store for inspection
    adata.obs['same_class_frac'] = frac_same
    adata.obs['keep_cell'] = mask
    # Add new label
    adata.obs[new_label_key] = adata.obs[label_key].tolist()
    adata.obs.loc[~adata.obs.keep_cell,new_label_key] = unknown_key

def filter_by_distance_to_control(
    adata: ad.AnnData,
    condition_col: str = 'perturbation', 
    ctrl_key: str = 'control',
    method: str = 'euclidean', 
    cutoff: float = 25.0,
    return_dist: bool = False,
    normalize: bool = True
) -> None | np.ndarray:
    """
    Keep samples that deviate significantly from control mean in latent/feature space.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    control_mask : boolean array
        Mask of control samples (True = control).
    method : str
        "euclidean" or "mahalanobis".
    cutoff : float
        - 25: keep samples above 25th percentile of control distances
    """
    # Pre-process adata
    if normalize:
        logging.info(f'Normalizing a copy of adata for z-score filtering.')
        _adata = adata.copy()
        sc.pp.normalize_total(_adata, target_sum=1e6)
        sc.pp.log1p(_adata)
    else:
        _adata = adata
    # Extract gene expression data
    X = _adata.X.toarray() if hasattr(_adata.X, "toarray") else _adata.X
    
    # --- Control statistics ---
    control_mask = _adata.obs[condition_col]==ctrl_key
    Xc = X[control_mask]
    mu = Xc.mean(axis=0)

    if method == "euclidean":
        dists = np.linalg.norm(X - mu, axis=1)
    elif method == "mahalanobis":
        cov = EmpiricalCovariance().fit(Xc)
        dists = cov.mahalanobis(X - mu)  # squared Mahalanobis
        dists = np.sqrt(dists)

    else:
        raise ValueError("method must be 'euclidean' or 'mahalanobis'")

    # --- Define cutoff ---
    thr = np.percentile(dists[control_mask], cutoff)

    # --- Keep samples beyond cutoff ---
    keep_mask = dists > thr
    mask = keep_mask | control_mask
    if return_dist:
        return dists
    else:
        # Subset adata for mask
        adata._inplace_subset_obs(mask)

def filter_min_number_of_cells_per_class(
        adata: ad.AnnData, 
        condition_col: str = 'perturbation', 
        min_cells: int = 50
    ) -> None:
    # Calculate number of cells per perturbation
    cpp = adata.obs[condition_col].value_counts()
    # Filter for perturbations with at least min_cells cells
    valid_cls = cpp[cpp>=min_cells].index
    # Subset adata for valid classes while also including control cells
    mask = adata.obs['perturbation'].isin(valid_cls)
    adata._inplace_subset_obs(mask)

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

def _remove_invalid_labels(adata: ad.AnnData, p_col: str) -> None:
    # Collect all unique labels
    labels = adata.obs[p_col].unique()
    # Check for invalid labels
    invalid_labels = labels[labels.isna() | labels.isin(MISC_LABELS_KEYS)]
    if invalid_labels.shape[0] > 0:
        invalid_mask = adata.obs[p_col].isin(invalid_labels)
        logging.info(f'Removing {invalid_mask.sum()} cells with invalid labels: {invalid_labels.astype(str)}')
        # Remove invalid labels
        adata._inplace_subset_obs(~invalid_mask)

def _filter_feature_pool(adata: ad.AnnData, feature_pool_file: str) -> None:
    """Filter adata features according to pre-calculated pool of features."""
    feature_pool = pd.read_csv(feature_pool_file, index_col=0)[OBS_KEYS.POOL_FEATURE_KEY]
    feature_mask = adata.var.index.isin(feature_pool)
    logging.info(f'Subsetting to features in pre-calculated pool: {feature_mask.sum()}/{adata.n_vars}')
    adata._inplace_subset_var(feature_mask)

def read_adata(
        adata_p: str,
        p_col: str = 'perturbation',
        ctrl_key: str = 'control',
        keep_versions: bool = False,
        remove_invalid_labels: bool = True,
        single_perturbations_only: bool = True
    ) -> ad.AnnData:
    logging.info(f'Loading dataset from {adata_p}')
    adata = sc.read(adata_p)
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
    # Remove unknown or NaN labels
    if remove_invalid_labels:
        _remove_invalid_labels(adata, p_col=p_col)
    # Filter for single perturbations only
    if single_perturbations_only:
        logging.info(f'Filtering column "{p_col}" for single perturbations only')
        sp_mask = single_perturbation_mask(adata.obs, p_col=p_col)
        if sp_mask is not None:
            adata._inplace_subset_obs(sp_mask)
    return adata

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def preprocess_dataset(
        adata: ad.AnnData, 
        cancer: bool,
        perturbation_pool_file: str,
        feature_pool_file: str,
        name: str = 'Unknown', 
        qc: bool = True, 
        norm: bool = False, 
        log: bool = False, 
        scale: bool = False, 
        hvg: bool = False,
        n_hvg: int = 2000, 
        subset: bool = False, 
        n_ctrl: int | None = 10_000,
        single_perturbations_only: bool = True,
        use_perturbation_pool: bool = False,
        use_feature_pool: bool = True,
        z_score_filter: bool = False,
        control_neighbor_threshold: float = 0.1,
        min_cells_per_class: int | None = 50,
        p_col: str = OBS_KEYS.PERTURBATION_KEY,
        ctrl_key: str = OBS_KEYS.CTRL_KEY,
        min_genes: int = 5_000,
        keep_versions: bool = False,
        seed: int = 42,
        remove_invalid_labels: bool = True,
    ) -> ad.AnnData:
    logging.info(f'Preprocessing dataset with shape: {adata.shape} (cells x genes)')
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
    # Remove unknown or NaN labels
    if remove_invalid_labels:
        _remove_invalid_labels(adata, p_col=p_col)
    # Filter for single perturbations only
    if single_perturbations_only:
        logging.info(f'Filtering column "{p_col}" for single perturbations only')
        sp_mask = single_perturbation_mask(adata.obs, p_col=p_col)
        if sp_mask is not None:
            adata._inplace_subset_obs(sp_mask)
    # Filter perturbations for perturbation pool if given
    if use_perturbation_pool:
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
    # Filter for feature pool if option is given
    if use_feature_pool:
        logging.info(f'Using feature pool to filtering for shared features.')
        _filter_feature_pool(adata, feature_pool_file=feature_pool_file)
    # Remove outliers of perturbation classes that are too close to control cells
    if control_neighbor_threshold > 0:
        logging.info(f'Filtering cells based on minimum amount of class neighbors.')
        nlk = 'label'
        neighborhood_purity(adata, label_key=p_col, ignore_class=ctrl_key, new_label_key=nlk, threshold=control_neighbor_threshold)
        adata._inplace_subset_obs(adata.obs['keep_cell'])
    # Filter cells for control z-score distance
    if z_score_filter:
        logging.info(f'Filtering cells based on control z-score.')
        filter_by_distance_to_control(adata, condition_col=p_col, ctrl_key=ctrl_key, normalize=not norm)
    # Filter perturbations based on minimum number of support
    if min_cells_per_class is not None and min_cells_per_class > 0:
        logging.info(f'Filtering for at least {min_cells_per_class} cells per perturbation.')
        filter_min_number_of_cells_per_class(adata, condition_col=p_col, min_cells=min_cells_per_class)
    logging.info(f'Pre-processed adata shape: {adata.shape}')
    return adata
