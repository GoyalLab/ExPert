import gc
import os
from scipy.stats import median_abs_deviation
import scanpy as sc
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from typing import Iterable, Optional, List, Literal
from sklearn.covariance import EmpiricalCovariance

from src.statics import P_COLS, CTRL_KEYS, GENE_SYMBOL_KEYS, SETTINGS, OBS_KEYS, MISC_LABELS_KEYS
import src.prepare as prep

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
def _compute_qc_mask(
        adata: ad.AnnData,
        percent_threshold: int = 50,
        nmads: int = 5,
        mt_nmads: int = 3,
        mt_per: int = 20,
    ) -> np.ndarray:
    """
    Compute QC keep-mask without modifying adata.  Works on backed *or*
    in-memory AnnData — scanpy's calculate_qc_metrics reads backed h5ad
    files via HDF5 without loading the full matrix into RAM.

    When var_names look like Ensembl IDs, MT/ribo/hb genes are detected
    through the gene-symbol column in adata.var instead.
    """
    # Prefer gene symbols over Ensembl IDs for MT/ribo/hb detection
    vn = adata.var_names.to_series()
    if vn.str.lower().str.startswith('ens').mean() > 0.9:
        gscl = adata.var.columns.intersection(set(GENE_SYMBOL_KEYS)).values
        if len(gscl) > 0:
            vn = adata.var[gscl[0]].astype(str)
    adata.var['mt']   = vn.str.startswith('MT-').values
    adata.var['ribo'] = vn.str.startswith(('RPS', 'RPL')).values
    adata.var['hb']   = vn.str.contains('^HB[^(P)]').values
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt', 'ribo', 'hb'],
        inplace=True, percent_top=[percent_threshold], log1p=True,
    )
    outlier = (
        is_outlier(adata, 'log1p_total_counts', nmads)
        | is_outlier(adata, 'log1p_n_genes_by_counts', nmads)
        | is_outlier(adata, f'pct_counts_in_top_{percent_threshold}_genes', nmads)
    )
    mt_outlier = (
        is_outlier(adata, 'pct_counts_mt', mt_nmads)
        | (adata.obs['pct_counts_mt'] > mt_per)
    )
    keep = (~outlier) & (~mt_outlier)
    logging.info(f'QC: keeping {keep.sum()}/{len(keep)} cells')
    return keep.values


def quality_control_filter(
        adata: ad.AnnData,
        percent_threshold: int = 50,
        nmads: int = 5,
        mt_nmads: int = 3,
        mt_per: int = 20,
    ) -> None:
    keep = _compute_qc_mask(adata, percent_threshold=percent_threshold,
                             nmads=nmads, mt_nmads=mt_nmads, mt_per=mt_per)
    logging.info(f'Total number of cells: {adata.n_obs}')
    adata._inplace_subset_obs(keep)
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

def clean_perturbation_labels(p: pd.Series, keep_versions: bool = False) -> pd.Series:
    # Remove guide prefixes
    clean_p = p.str.replace(r'^ES.sg\d*.', '', regex=True)
    # Remove versions
    if not keep_versions:
        clean_p = clean_p.str.split('_').str[0]
    return pd.Series(clean_p.values)

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
    # Check for duplicate gene symbols after conversion
    if adata.var_names.nunique() != adata.n_vars:
        logging.info(f'Found duplicate indices for ensembl to symbol mapping, highest number of conflicts: {adata.var_names.value_counts().max()}')
        # Fix conflicts by choosing the gene with the higher harmonic mean of mean expression and normalized variance out of pool
        if len(set(['means', 'variances_norm']).intersection(adata.var.columns)) == 2:
            adata.var['hm_var'] = (2 * adata.var.means * adata.var.variances_norm) / (adata.var.means + adata.var.variances_norm + 1e-9)
        else:
            adata.var['hm_var'] = np.arange(adata.n_vars)
        idx = adata.var.reset_index().groupby(gsh, observed=True).hm_var.idxmax().values
        # Subset adata
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
    obs[p_col] = clean_perturbation_labels(obs[p_col], keep_versions=False).values
    # Extract .var
    var = adata.var.copy()
    if adata.var_names.str.lower().str.startswith('ens').sum() / adata.n_vars > 0.9:
        logging.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        var = ens_to_symbol(adata).var.copy()
    return obs, var

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
    adata.obs[p_col] = clean_perturbation_labels(adata.obs[p_col], keep_versions=keep_versions).values
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

def _ensure_csr(adata: ad.AnnData) -> ad.AnnData:
    """
    Ensure adata.X is CSR. If the adata is backed and X is dense, reload the
    full object, convert X to CSR, overwrite the source file, and re-open
    backed. If not backed, convert X to CSR in-place.
    """
    # Check if adata.X is sparse
    try:
        from anndata._core.sparse_dataset import _CSRDataset, _CSCDataset
        is_sparse = isinstance(adata.X, (_CSRDataset, _CSCDataset)) or sp.issparse(adata.X)
    except ImportError:
        is_sparse = sp.issparse(adata.X)
    # Return unchanged if it is regardless of backed status
    if is_sparse:
        return adata
    # If not backed but dense, just change it inplace
    if not adata.isbacked:
        logging.info(f'Converting dense X ({type(adata.X).__name__}) to CSR in-place.')
        adata.X = sp.csr_matrix(adata.X)
        return adata
    # If backed but dense, load it into memory, change inplace, and update on disk
    file_path = str(adata.filename)
    logging.info(
        f'Backed X is dense ({type(adata.X).__name__}, shape {adata.X.shape}). '
        f'Converting to CSR and rewriting {file_path} for faster backed access.'
    )
    # Close backed connection
    adata.file.close()
    # Read full dataset and update .X to CSR matrix
    full = sc.read(file_path)
    full.X = sp.csr_matrix(full.X)
    full.write(file_path)
    logging.info('Rewrote file with CSR X. Re-opening in backed mode.')
    del full
    gc.collect()

    return sc.read(file_path, backed='r')


def read_subset(
    adata_p: str,
    p_col: str = 'perturbation',
    ctrl_key: str = 'control',
    perturbation_pool_file: str | None = None,
    use_perturbation_pool: bool = True,
    n_ctrl: int = 10_000,
    min_cells_per_perturbation: int = 50,
    ensure_csr: bool = True,
    feature_pool_file: str | None = None,
    use_feature_pool: bool = False,
    remove_invalid_labels: bool = False,
    single_perturbations_only: bool = False,
    keep_versions: bool = False,
    qc: bool = False,
    cancer: bool = False,
    seed: int = 42,
) -> ad.AnnData:
    """
    Load a minimal subset of an h5ad into memory using a single backed read.

    All cell-level filters (perturbation pool, control sampling, label
    cleaning, QC) are evaluated while the file is still backed so that
    ``to_memory()`` loads only the final set of cells x genes.  This avoids
    the repeated in-memory copies that ``_inplace_subset_obs`` causes on
    large datasets.
    """
    # Set random seed
    np.random.seed(seed)
    try:
        # Ensure adata is backed as CSR matrix for major speedups
        adata = _ensure_csr(sc.read(adata_p, backed='r'))
        logging.info(f'Dataset shape: {adata.shape} (cells x genes)')

        # --- Identify perturbation column ---
        p_idx = _find_match_idx(adata.obs.columns, P_COLS)
        p_key = adata.obs.columns[p_idx]

        # --- Clean labels once (obs only, no X) ---
        labels = clean_perturbation_labels(adata.obs[p_key], keep_versions=keep_versions)

        # --- Perturbation pool filter (obs only) ---
        if use_perturbation_pool and perturbation_pool_file is not None:
            gene_targets = pd.read_csv(perturbation_pool_file, index_col=0)[OBS_KEYS.POOL_PERTURBATION_KEY]
            gene_targets = gene_targets[~gene_targets.isin(CTRL_KEYS)]
            mask = labels.isin(gene_targets)
            logging.info(f'Subsetting to perturbation pool, found {len(labels[mask].unique())}/{len(labels.unique())} of total perturbations in dataset.')
        else:
            mask = np.ones(adata.n_obs, dtype=bool)

        # --- Control sampling (obs only) ---
        sample_control = n_ctrl is not None and n_ctrl > 0
        ctrl_mask = labels.str.lower().isin(CTRL_KEYS)
        if ctrl_mask.sum() == 0:
            raise ValueError(f'No matching control key found in adata.obs["{p_key}"], looked for "{CTRL_KEYS}"')
        if sample_control:
            if n_ctrl > ctrl_mask.sum():
                logging.info(f'Adding all {ctrl_mask.sum()} control cells')
                mask |= ctrl_mask
            else:
                logging.info(f'Sampling {n_ctrl} control cells from {ctrl_mask.sum()}')
                ctrl_indices = np.where(ctrl_mask)[0]
                sampled = np.random.choice(ctrl_indices, size=n_ctrl, replace=False)
                sampled_mask = np.zeros(adata.n_obs, dtype=bool)
                sampled_mask[sampled] = True
                mask |= sampled_mask

        # --- Optional obs-only filters (no X access) ---
        if remove_invalid_labels:
            logging.info(f'Removing invalid labels.')
            invalid = labels.isna() | labels.isin(MISC_LABELS_KEYS)
            mask &= ~invalid.values

        if single_perturbations_only:
            logging.info(f'Filtering for single perturbations.')
            sp_mask = single_perturbation_mask(labels.rename(p_key).to_frame(), p_col=p_key)
            if sp_mask is not None:
                mask &= np.asarray(sp_mask)

        # --- QC filter ---
        # calculate_qc_metrics doesn't support backed HDF5 Dataset objects,
        # so load the candidate cells temporarily, compute the mask, then free.
        # The two loads are sequential so peak RAM is always 1x, never 2x.
        if qc:
            logging.info(f'Applying QC on {mask.sum()} candidate cells.')
            mt_per = SETTINGS.MT_PERCENT_CANCER if cancer else SETTINGS.MT_PERCENT_NORMAL
            # Load masked adata 
            candidate_positions = np.sort(np.where(mask)[0])
            qc_adata = adata[candidate_positions].to_memory()
            qc_keep = _compute_qc_mask(qc_adata, mt_per=mt_per)
            # Delete loaded adata out of memory
            del qc_adata
            gc.collect()
            # Map back to full indices
            full_qc_mask = np.zeros(adata.n_obs, dtype=bool)
            full_qc_mask[candidate_positions[qc_keep]] = True
            mask = full_qc_mask
        # Filter perturbations for minimum amount of cells (after qc).
        # Use `labels` (already cleaned) and boolean indexing — p_col doesn't
        # exist on the backed adata yet, it's only added post-load.
        if min_cells_per_perturbation is not None:
            logging.info(f'Filtering perturbations for a minimum of {min_cells_per_perturbation} cells.')
            cpp = labels[mask].value_counts()
            valid_perturbations = cpp[cpp >= min_cells_per_perturbation].index
            mask &= labels.isin(valid_perturbations).values

        logging.info(f'Backed filters: {mask.sum()} / {adata.n_obs} cells retained')

        # --- Var (gene) filter (obs metadata only) ---
        # Also detect ensembl IDs here so we can reuse gsh post-load for a
        # zero-copy index rename instead of calling ens_to_symbol on loaded data.
        var_mask = None
        gsh = None  # gene-symbol column name; set when var_names are ensembl IDs
        if adata.var_names.str.lower().str.startswith('ens').mean() > 0.9:
            gscl = adata.var.columns.intersection(set(GENE_SYMBOL_KEYS)).values
            if len(gscl) > 0:
                gsh = gscl[0]
                symbols = adata.var[gsh].astype(str)
                if {'means', 'variances_norm'}.issubset(adata.var.columns):
                    hm = (2 * adata.var['means'] * adata.var['variances_norm']) / (
                        adata.var['means'] + adata.var['variances_norm'] + 1e-9)
                else:
                    hm = pd.Series(np.arange(adata.n_vars), index=adata.var.index)
                keep_pos = (symbols.to_frame()
                            .assign(_hm=hm.values)
                            .reset_index()
                            .groupby(gsh, observed=True)['_hm']
                            .idxmax().values)
                var_mask = np.zeros(adata.n_vars, dtype=bool)
                var_mask[keep_pos] = True
                logging.info(f'Ensembl dedup: {var_mask.sum()} / {adata.n_vars} genes retained')
        # Optional futher filtering using pre-calculated feature pools
        if use_feature_pool and feature_pool_file is not None:
            feature_pool = pd.read_csv(feature_pool_file, index_col=0)[OBS_KEYS.POOL_FEATURE_KEY]
            if gsh is not None:
                fp_mask = adata.var[gsh].isin(feature_pool)
            else:
                fp_mask = adata.var_names.isin(feature_pool)
            var_mask = fp_mask if var_mask is None else (var_mask & fp_mask)
            logging.info(f'Feature pool filter: {var_mask.sum()} / {adata.n_vars} genes retained')
        
        # --- Single to_memory() with sorted obs indices ---
        # AnnData backed mode doesn't support fancy indexing on both axes at
        # once ("Only one indexing vector allowed"), so we slice obs only here
        # and apply the var mask after loading via _inplace_subset_var.
        # Peak RAM is (1 + selected_gene_fraction) x size — well under 2x.
        mask = np.sort(np.where(mask)[0])
        n_vars_out = var_mask.sum() if var_mask is not None else adata.n_vars
        logging.info(f'Loading {len(mask)} cells x {adata.n_vars} genes into memory '
                     f'(will trim to {n_vars_out} genes post-load)')
        subset = adata[mask].to_memory()
        if var_mask is not None and var_mask.sum() > 0:
            subset._inplace_subset_var(var_mask)

    except Exception as e:
        raise e
    finally:
        # Close file connection
        logging.info(f'Closing file connection.')
        adata.file.close()

    # Post-load fixups
    subset.obs[p_col] = labels[mask].values
    if sample_control:
        ctrl_mask_post = subset.obs[p_col].str.lower().isin(CTRL_KEYS)
        if ctrl_mask_post.sum() > 0:
            ps = np.array(subset.obs[p_col], dtype=str)
            ps[ctrl_mask_post] = ctrl_key
            subset.obs[p_col] = pd.Categorical(ps)

    # Rename ensembl IDs --> gene symbols using the column we already found
    if gsh is not None:
        logging.info(f'Renaming var index from ensembl IDs to gene symbols via "{gsh}"')
        subset.var.reset_index(names='ensembl_id', inplace=True)
        subset.var.set_index(gsh, inplace=True)

    if ensure_csr and not isinstance(subset.X, sp.csr_matrix):
        logging.info('Converting adata.X to CSR matrix.')
        subset.X = sp.csr_matrix(subset.X)
    return subset

# inspired by https://www.sc-best-practices.org/preprocessing_visualization/normalization.html
def preprocess_dataset(
        dataset_file: str, 
        cancer: bool,
        perturbation_pool_file: str,
        feature_pool_file: str,
        name: str = 'Unknown', 
        qc: bool = True, 
        norm: bool = False, 
        log: bool = False, 
        scale: bool = False, 
        hvg: bool = True,
        n_hvg: int = 3000, 
        subset: bool = False, 
        n_ctrl: int | None = 10_000,
        min_cells_per_perturbation: int | None = 50,
        single_perturbations_only: bool = True,
        use_perturbation_pool: bool = True,
        use_feature_pool: bool = False,
        z_score_filter: bool = False,
        control_neighbor_threshold: float = 0.0,
        p_col: str = OBS_KEYS.PERTURBATION_KEY,
        ctrl_key: str = OBS_KEYS.CTRL_KEY,
        min_genes: int = 5_000,
        keep_versions: bool = False,
        remove_invalid_labels: bool = True,
        **kwargs
    ) -> ad.AnnData:
    # All cell-level filters (labels, single-pert, QC) and the gene filter are
    # pushed into read_subset so they run in backed mode.  to_memory() is called
    # exactly once on the minimal (cells x genes) slice — no in-memory copies.
    logging.info(f'Reading: {dataset_file}')
    adata = read_subset(
        adata_p=dataset_file,
        p_col=p_col,
        ctrl_key=ctrl_key,
        perturbation_pool_file=perturbation_pool_file,
        use_perturbation_pool=use_perturbation_pool,
        n_ctrl=n_ctrl,
        min_cells_per_perturbation=min_cells_per_perturbation,
        feature_pool_file=feature_pool_file if use_feature_pool else None,
        use_feature_pool=use_feature_pool,
        remove_invalid_labels=remove_invalid_labels,
        single_perturbations_only=single_perturbations_only,
        keep_versions=keep_versions,
        qc=qc,
        cancer=cancer,
    )
    if hvg:
        # Calculate highly variable genes on raw counts (seurat_v3 requires this).
        if adata.n_vars <= min_genes:
            logging.info(f'Selecting all genes for dataset "{name}", because of low gene number: {adata.n_vars}')
            adata.var['highly_variable'] = True
        else:
            logging.info(f'Determining highly variable genes for dataset "{name}"')
            logging.info(f'Using flavor "seurat_v3" when determining hvgs for raw counts')
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=subset, flavor='seurat_v3')
    else:
        adata.var['highly_variable'] = True

    logging.info(f'Found {np.sum(adata.var.highly_variable)} highly variable genes out of {adata.n_vars} total genes')

    # Release memory from filtered-out genes before the expensive dense ops below
    gc.collect()

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
    logging.info(f'Pre-processed adata shape: {adata.shape}')
    logging.info(f'\t -> # perturbations: {adata.obs[p_col].nunique()}')
    logging.info(f'\t -> # cpp (median): {adata.obs[p_col].value_counts().median()}')
    return adata

# Per-dataset adaptive threshold
def compute_responder_threshold(scores, dataset_ids, min_threshold=1.5, quantile=0.5):
    """Threshold as median of non-zero scores per dataset."""
    thresholds = {}
    for ds in np.unique(dataset_ids):
        ds_scores = np.abs(scores[dataset_ids == ds])
        nonzero = ds_scores[ds_scores > 0.5]  # ignore near-zero
        if len(nonzero) > 0:
            thresholds[ds] = max(np.quantile(nonzero, quantile), min_threshold)
        else:
            thresholds[ds] = min_threshold
    return thresholds

def scale_scores(
        adata: ad.AnnData, 
        score_col: str = 'mixscale_score', 
        dataset_col: str = 'dataset', 
        q: float = 0.99,
        max_scaling: bool = False,
    ):
    # Get max values for each cell
    df = adata.obs.copy()
    df['score'] = adata.obs[score_col].abs()
    max_per_ds = df.groupby(dataset_col).score.quantile(q)[adata.obs[dataset_col]].values
    score = df.score
    score = max_per_ds * np.tanh(score / max_per_ds)
    if max_scaling:
        return score / max_per_ds * max_per_ds.min()
    else:
        return score

def select_perturbations(
    adata,
    n: int,
    perturbation_col: str = "perturbation",
    context_col: str = "context",
    efficiency_col: Optional[str] = None,
    min_contexts: int = 4,
    min_cells: int = 100,
    min_abs_eff: float = 0.0,
    min_eff_variance: Optional[float] = None,
    random_state: Optional[int] = None,
    return_stats: bool = False,
) -> List[str]:
    """
    Select N perturbations based on cross-context support and signal richness.

    Parameters
    ----------
    adata : AnnData
    n : int
        Number of perturbations to select
    perturbation_col : str
    context_col : str
    efficiency_col : Optional[str]
        Column with cell-level efficiency scores
    min_contexts : int
        Minimum number of distinct contexts per perturbation
    min_cells : int
        Minimum total number of cells per perturbation
    min_eff_variance : Optional[float]
        Minimum variance in efficiency score (if provided)
    random_state : Optional[int]

    Returns
    -------
    List[str]
        Selected perturbation names
    """

    df = adata.obs[[perturbation_col, context_col]].copy()
    # Add efficiency column and filter
    if efficiency_col is not None:
        df["efficiency"] = np.log1p(adata.obs[efficiency_col].abs())
        # Filter for min efficiency
        df = df[df.efficiency > min_abs_eff].copy()

    # ---- Aggregate stats per perturbation ----
    stats = df.groupby(perturbation_col).agg(
        n_cells=(perturbation_col, "count"),
        n_contexts=(context_col, "nunique"),
    )

    # Add efficiency stats if provided
    if efficiency_col is not None:
        eff_stats = df.groupby(perturbation_col)["efficiency"].agg(
            eff_mean="mean",
            eff_var="var",
        )
        stats = stats.join(eff_stats)

    # ---- Filtering ----
    stats = stats[stats["n_contexts"] >= min_contexts]
    stats = stats[stats["n_cells"] >= min_cells]

    if efficiency_col is not None and min_eff_variance is not None:
        stats = stats[stats["eff_var"] >= min_eff_variance]

    if len(stats) == 0:
        raise ValueError("No perturbations satisfy the selection criteria.")

    # ---- Ranking ----
    # Primary: number of contexts
    # Secondary: number of cells
    sort_cols = ["n_contexts", "n_cells"]

    if efficiency_col is not None:
        sort_cols.append("eff_var")

    stats = stats.sort_values(sort_cols, ascending=False)

    # ---- Select top N ----
    selected = stats.index.tolist()

    if len(selected) < n:
        print(f"Warning: only {len(selected)} perturbations available after filtering.")
        return selected

    # Optional random sampling within top 2N for diversity
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        top_pool = selected[: min(len(selected), 2 * n)]
        selected = rng.choice(top_pool, size=n, replace=False).tolist()
    else:
        selected = selected[:n]
    if return_stats:
        return selected, stats
    else:
        return selected
    
def build_perturbation_groups(class_embeddings, class_names, threshold=0.8, method='embedding'):
    """Group perturbations by similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    if method == 'embedding':
        emb = class_embeddings
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sim = emb_norm @ emb_norm.T
    elif method == 'expression':
        # Average expression profile per perturbation
        sim = cosine_similarity(class_embeddings)
    
    # Cluster similar perturbations via connected components
    from scipy.sparse.csgraph import connected_components
    adjacency = (sim > threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    _, component_labels = connected_components(adjacency, directed=False)
    
    groups = {}
    for name, group_id in zip(class_names, component_labels):
        groups[name] = f'group_{group_id}'
    
    return groups

def sample_merged_dataset(
    obs: pd.DataFrame,
    perturbation_col: str = "perturbation",
    context_col: str = "dataset",
    sampling_method: Literal['subsample', 'maxsample'] = 'maxsample',
    cells_per_perturbation: int = 100,
    min_contexts: int = 3,
    efficiency_col: str | None = None,
    **kwargs
):
    if sampling_method == 'subsample':
        return sample_dataset_idx(
            obs,
            perturbation_col=perturbation_col,
            context_col=context_col,
            cells_per_perturbation=cells_per_perturbation,
            min_contexts=min_contexts,
            efficiency_col=efficiency_col,
            **kwargs
        )
    elif sampling_method == 'maxsample':
        return sample_max_dataset_idx(
            obs,
            perturbation_col=perturbation_col,
            context_col=context_col,
            cells_per_perturbation=cells_per_perturbation,
            min_contexts=min_contexts,
            efficiency_col=efficiency_col,
            **kwargs
        )
    else:
        raise ValueError(f'Unrecognized sampling_method: {sampling_method}')

def sample_dataset_idx(
    obs: pd.DataFrame,
    perturbation_col: str = "perturbation",
    context_col: str = "dataset",
    cells_per_perturbation: int = 100,
    min_contexts: int = 3,
    seed: int = 42,
    ignore_key: str = "control",
    efficiency_col: str | None = None,
    min_eff: float = 0.0,
    strict: bool = True,
):
    # Init random generator
    rng = np.random.default_rng(seed)

    # ----------------------------
    # Filtering
    # ----------------------------
    mask = np.ones(len(obs), dtype=bool)
    # Remove control cells
    if ignore_key is not None:
        mask &= obs[perturbation_col] != ignore_key
    # Check if efficiency col is present and can be used
    use_eff = efficiency_col is not None and efficiency_col in obs
    if use_eff:
        # Use absolute values
        obs[efficiency_col] = obs[efficiency_col].abs()
        # Remove cells below efficiency cutoff
        mask &= obs[efficiency_col] > min_eff

    # Subset to overall available cells
    obs = obs[mask]

    # ----------------------------
    # Stats
    # ----------------------------
    stats = obs.groupby(perturbation_col).agg(
        n_cells=(perturbation_col, "count"),
        n_contexts=(context_col, "nunique"),
    )
    # Filter for min contexts per perturbation
    if min_contexts < 0:
        min_contexts = obs[context_col].nunique()
    # Draw n * min contexts cells per perturbation
    min_cells_total = min_contexts * cells_per_perturbation
    # Collect stats per group
    stats = stats[
        (stats.n_cells >= min_cells_total) &
        (stats.n_contexts >= min_contexts)
    ]
    # Subset to perturbation that pass the filtering
    valid_perts = stats.index
    obs = obs[obs[perturbation_col].isin(valid_perts)]
    # Collect adata indices for subsetting
    sampled_indices = []

    # ----------------------------
    # Sample cells per perturbation x dataset group
    # ----------------------------
    for _, dfp in obs.groupby(perturbation_col):
        # Collect perturbation indices
        pert_indices = []
        # Gather all contexts per perturbation
        contexts = dfp[context_col].unique()
        k = len(contexts)
        # Skip perturbation if not enough contexts are present
        if k < min_contexts:
            continue
        # Determine how many cells to sample per group
        base = min_cells_total // k
        remainder = min_cells_total % k
        # Loop over each available context and draw equal cells
        for ctx in contexts:
            # Subset to context perturbation
            d: pd.DataFrame = dfp[dfp[context_col] == ctx]
            # Update remainder counts
            n = base
            if remainder > 0:
                n += 1
                remainder -= 1
            # Determine number of available cells
            n = min(len(d), n)
            # Randomly draw cells
            if not use_eff:
                sampled_idx = rng.choice(d.index, size=n, replace=False)
            # Draw cells with highest efficiency
            else:
                sampled_idx = d.sort_values(efficiency_col, ascending=False).index[:n]
            # Add indices per perturbation
            pert_indices.extend(
                sampled_idx
            )

        # Fill remainder if needed and not strict batching
        if not strict and len(pert_indices) < min_cells_total:
            # Check how many cells are missing
            remaining = dfp.index.difference(pert_indices)
            # Draw extra cells
            extra = rng.choice(
                remaining,
                size=min_cells_total - len(pert_indices),
                replace=False
            )
            # Add to drawn indices
            pert_indices.extend(extra)

        sampled_indices.extend(pert_indices)
    # Sort sampled indices to speed up backed up selection
    sampled_indices = np.sort(np.array(sampled_indices))

    return sampled_indices

def sample_max_dataset_idx(
    src_obs: pd.DataFrame,
    perturbation_col: str = "perturbation",
    context_col: str = "dataset",
    cells_per_perturbation: int = 10,
    min_contexts: int = 2,
    cap_quantile: float = 0.95,
    cap_max: int | None = None,
    seed: int = 42,
    ignore_key: str = "control",
    efficiency_col: str | None = None,
    min_eff: float = 0.0,
    norm_by_context_support: bool = True,
    sample_by_eff: bool = False,
    n_classes: int | None = None,
):
    """
    Light subsampling: keep all data except extreme outlier groups.
    Only caps dataset x perturbation groups above the cap_quantile threshold.
    """
    rng = np.random.default_rng(seed)

    # ----------------------------
    # Filtering
    # ----------------------------
    mask = np.ones(len(src_obs), dtype=bool)
    if ignore_key is not None:
        mask &= src_obs[perturbation_col] != ignore_key
    # Filter for minimum efficiency cells
    use_eff = efficiency_col is not None and efficiency_col in src_obs.columns
    # Subset to valid cells
    obs = src_obs[mask].copy()
    
    # Calculate overall context support
    if norm_by_context_support:
        cpds = obs[context_col].value_counts()
        ctx_weights = cpds.mean() / cpds

    # ----------------------------
    # Group stats
    # ----------------------------
    # Determine group sizes
    group_sizes = obs.groupby([perturbation_col, context_col]).size()
    # Filter each group by mean efficiency over 0
    if use_eff:
        obs[efficiency_col] = obs[efficiency_col].abs()
        mean_eff_per_group = obs.groupby([perturbation_col, context_col])[efficiency_col].mean()
        valid_mean_eff_groups = mean_eff_per_group[mean_eff_per_group > min_eff].index
        group_sizes = group_sizes[valid_mean_eff_groups]
    # Filter: each group must have minimum cells
    valid_groups = group_sizes[group_sizes >= cells_per_perturbation]
    # Filter valid groups for minimum amount of contexts per perturbation
    ctxpp = valid_groups.reset_index().groupby(perturbation_col)[context_col].nunique()
    valid_perturbations = ctxpp[ctxpp >= min_contexts].index.values
    if n_classes is not None and n_classes > 0:
        logging.info(f'Subsampling to {n_classes} classes.')
        valid_perturbations = rng.choice(valid_perturbations, n_classes, replace=False)
    valid_group_idx = valid_groups[valid_perturbations].index

    # Compute cap from quantile of valid group sizes
    cap = int(valid_groups.quantile(cap_quantile))
    if cap_max is not None:
        cap = min(cap, cap_max)

    # ----------------------------
    # Sample from all groups
    # ----------------------------
    sampled_indices = []

    for (pert, ctx), group_df in obs.groupby([perturbation_col, context_col]):
        # Skip if perturbation doesn't meet filters
        if (pert, ctx) not in valid_group_idx:
            continue
        # Get index pool for group
        idx = group_df.index.tolist()
        # Get context specific cap
        if norm_by_context_support:
            ctx_cap = int(cap * ctx_weights[ctx])
        # Use overall cap
        else:
            ctx_cap = cap
        
        # Only subsample if above cap
        if len(idx) > ctx_cap:
            if use_eff and sample_by_eff:
                idx = group_df.nlargest(ctx_cap, efficiency_col).index.tolist()
            else:
                idx = rng.choice(idx, size=ctx_cap, replace=False).tolist()
        
        sampled_indices.extend(idx)
    # Collect all sampled indices
    sampled_indices = np.array(sampled_indices)
    # Sort indices for faster indexing of backed adata objects
    sampled_indices = np.sort(sampled_indices)
    
    # Log stats
    n_total = len(obs)
    n_sampled = len(sampled_indices)
    n_capped = (group_sizes > cap).sum()
    logging.info(f"Subsampling: {n_sampled}/{n_total} cells kept "
          f"({n_sampled/n_total:.1%}), {n_capped} groups capped at {cap}")

    return sampled_indices
    
def process(
    adata_p: str, 
    gene_targets: list[str], 
    n_hvg: int = 3000,
    p_col: str = 'perturbation', 
    n_ctrl: int = 10_000, 
    ctrl_key: str = 'control',
    cpm: bool = False,
    log1p: bool = False,
    scale: bool = False,
):
    if not os.path.exists(adata_p):
        logging.info(f'File not found: {adata_p}')
        return None
    logging.info(f'Reading: {adata_p}')
    # Read adata in backed mode
    adata = sc.read(adata_p, backed='r')
    logging.info(f'Shape: {adata.shape}')
    # Check adata type
    logging.info(f'X: {type(adata.X)}')
    
    # Find correct key
    p_idx = _find_match_idx(adata.obs.columns, P_COLS)
    p_key = adata.obs.columns[p_idx]
    # Subset adata to target perturbations
    mask = adata.obs[p_key].isin(gene_targets)
    # Randomly sample n control cells if given
    if n_ctrl > 0:
        ctrl_mask = adata.obs[p_key].str.lower().isin(CTRL_KEYS)
        # Add all control cells if there are less than given
        if n_ctrl > ctrl_mask.sum():
            logging.info(f'Adding all {ctrl_mask.sum()} control cells')
            mask |= ctrl_mask
        # Sample N control cells
        else:
            logging.info(f'Sampling {n_ctrl} control cells from {ctrl_mask.sum()}')
            ctrl_indices = np.where(ctrl_mask)[0]
            sampled_ctrl = np.random.choice(ctrl_indices, size=n_ctrl, replace=False)
            sampled_mask = np.zeros(adata.n_obs, dtype=bool)
            sampled_mask[sampled_ctrl] = True
            mask |= sampled_mask

    # Stop if no cells have been selected
    logging.info(f'Found {mask.sum()} cells matching the target genes.')
    if mask.sum() == 0:
        return None
    # Subset by sorted index for faster slicing
    idx = sorted(np.where(mask)[0])
    # Subset adata view
    subset = adata[idx]
    # Load only subset into memory
    subset = subset.to_memory()
    # Rename perturbation column
    subset.obs[p_col] = subset.obs[p_key].values
    # Rename control column
    if n_ctrl > 0:
        ctrl_mask = subset.obs[p_col].str.lower().isin(CTRL_KEYS)
        if ctrl_mask.sum() > 0:
            ps = np.array(subset.obs[p_col], dtype=str)
            ps[ctrl_mask] = ctrl_key
            subset.obs[p_col] = pd.Categorical(ps)
    # Close adata file connection TODO: wrap in try
    adata.file.close()
    # Convert to csr if neccessary
    if not sp.issparse(subset.X):
        logging.info('Converting .X to csr.')
        subset.X = sp.csr_matrix(subset.X)
    # Subset adata for highly variable genes, use v3 for raw data
    sc.pp.highly_variable_genes(subset, n_top_genes=n_hvg, subset=False, flavor='seurat_v3', check_values=True)
    # Ensure var names are symbols
    if subset.var_names.str.lower().str.startswith('ens').sum() / subset.n_vars > 0.9:
        logging.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        subset = ens_to_symbol(subset).copy()
    # Apply pre-processing to adata
    norms = []
    if cpm:
        norms.append('cpm')
        sc.pp.normalize_total(subset, target_sum=1e6)
    if log1p:
        norms.append('log1p')
        sc.pp.log1p(subset)
    if scale:
        norms.append('scale')
        sc.pp.scale(subset)
    if len(norms) > 0:
        logging.info(f'Applied normalizations: {norms}')
    return subset

def prepare(adata, ds: str, pool_genes: list[str], scores_file: str | None = None):
    # add per-cell efficicency scores if not None and is csv
    if scores_file is not None and str(scores_file).endswith('.csv'):
        # Read named series as dataframe
        efficiency_scores_df = pd.read_csv(scores_file, index_col=0)
        # Get valid indices
        idx = adata.obs_names.intersection(efficiency_scores_df.index)
        # Add scores per cell if barcodes match up
        assert idx.shape[0] == adata.shape[0], "Score indices and adata barcodes don't match."
        # Subset to valid indices if needed
        scores = efficiency_scores_df.loc[idx]
        # Update adata.obs with new column
        adata.obs = pd.concat([adata.obs, scores], axis=1)
    # Make cell names unique per dataset
    adata.obs_names = adata.obs_names + ';' + ds
    # Add dataset marker to adata
    adata.obs['dataset'] = ds
    # Zero-pad missing genes from adata
    adata = prep._filter(adata, ds, pool_genes)
    # Convert adata.X to csr
    if not isinstance(adata.X, sp.csr_matrix):
        logging.info('Converting to CSR matrix.')
        adata.X = sp.csr_matrix(adata.X.compute())
    return adata


def subset_adata(adata, labels_key='perturbation', dataset_key='dataset', max_per_group=500):
    """Only downsample large groups, never upsample."""
    idx = []
    for _, group_df in adata.obs.groupby([labels_key, dataset_key]):
        group_idx = group_df.index.tolist()
        if len(group_idx) > max_per_group:
            idx.extend(np.random.choice(group_idx, max_per_group, replace=False))
        else:
            idx.extend(group_idx)
    return adata[idx].copy()

def compute_dataset_similarity_matrix(adata, dataset_key, labels_key, n_pcs=50):
    import scanpy as sc
    """Pairwise dataset similarity per label, averaged."""
    if 'X_pca' not in adata.obsm:
        logging.info(f'Calculating PCA with {n_pcs} components.')
        sc.pp.pca(adata, n_comps=n_pcs)

    # Collect dataset information
    datasets = list(adata.obs[dataset_key].unique())
    ds_to_idx = {ds: i for i, ds in enumerate(datasets)}
    n_ds = len(datasets)
    
    # Accumulate per-label similarities
    sim_sum = np.zeros((n_ds, n_ds))
    sim_count = np.zeros((n_ds, n_ds))
    
    for label, group in adata.obs.groupby(labels_key):
        centroids = {}
        for ds in group[dataset_key].unique():
            mask = group[dataset_key] == ds
            if mask.sum() >= 5:
                centroids[ds] = adata[group[mask].index].obsm['X_pca'].mean(axis=0)
        
        ds_list = list(centroids.keys())
        if len(ds_list) < 2:
            continue
        
        for a in ds_list:
            for b in ds_list:
                if a == b:
                    continue
                c1, c2 = centroids[a], centroids[b]
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                i, j = ds_to_idx[a], ds_to_idx[b]
                sim_sum[i, j] += sim
                sim_count[i, j] += 1
    
    # Average similarity
    sim_matrix = np.divide(sim_sum, sim_count, where=sim_count > 0, out=np.zeros_like(sim_sum))
    
    # Convert to contrastive weight: dissimilar pairs get higher weight
    # They provide more learning signal
    weight_matrix = 1.0 - sim_matrix
    np.fill_diagonal(weight_matrix, 0)
    # Normalize so mean weight = 1
    mask = weight_matrix > 0
    weight_matrix[mask] /= weight_matrix[mask].mean()
    
    return weight_matrix, datasets

def dataset_anchors(
    adata, 
    batch_col: str = 'dataset', 
    dataset_to_idx_key: str = 'dataset_to_mask_idx',
):
    if batch_col not in adata.obs:
        raise ValueError(f'Missing category: {batch_col}')
    X = adata.X

    batch_labels = adata.obs[batch_col].values
    n_genes = X.shape[1]
    # Get datasets from unique values
    if dataset_to_idx_key not in adata.uns:
        datasets = adata.obs[batch_col].unique()
    # Get precalculated dataset from index mapping
    else:
        datasets = adata.uns[dataset_to_idx_key].keys()
    n_datasets = len(datasets)

    mean_matrix = np.zeros((n_datasets, n_genes), dtype=np.float32)
    std_matrix = np.zeros((n_datasets, n_genes), dtype=np.float32)

    for i, ds in enumerate(datasets):
        mask = batch_labels == ds
        ds_data = X[mask]

        if sp.issparse(ds_data):
            # Sparse mean is fast
            m = np.asarray(ds_data.mean(axis=0)).flatten()
            # Var = E[x^2] - E[x]^2, all sparse-friendly
            sq = ds_data.copy()
            sq.data **= 2
            mean_sq = np.asarray(sq.mean(axis=0)).flatten()
            var = mean_sq - m ** 2
            var = np.maximum(var, 0)  # numerical safety
            std_matrix[i] = np.sqrt(var)
        else:
            m = ds_data.mean(axis=0)
            std_matrix[i] = ds_data.std(axis=0)

        mean_matrix[i] = m
    return mean_matrix, std_matrix
