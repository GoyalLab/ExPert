"""
Mixscale-style perturbation efficiency scoring.

Per-cell efficiency scores are computed via scalar projection of each cell's
expression profile onto the perturbation direction vector, then z-score
normalized relative to non-targeting control cells.

Reference: Papalexi et al., Nature Cell Biology 2025
           https://github.com/satijalab/Mixscale
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from anndata import AnnData
import scanpy as sc
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Module-level worker state (shared via initializer — avoids re-pickling the
# large expression matrix for every task)
# ---------------------------------------------------------------------------

_WORKER_EXPR: Optional[np.ndarray] = None


def _is_raw_counts(adata: AnnData, n_sample: int = 200) -> bool:
    """Check whether adata.X looks like raw (integer) counts."""
    X = adata.X
    if sp.issparse(X):
        data = X[:n_sample].data if X.shape[0] >= n_sample else X.data
    else:
        data = X[:n_sample].ravel()
    if len(data) == 0:
        return False
    return np.allclose(data, np.round(data)) and np.all(data >= 0)
_WORKER_CTRL_IDX: Optional[np.ndarray] = None


def _init_worker(expr: np.ndarray, ctrl_idx: np.ndarray) -> None:
    """Initializer for worker processes — stores shared data as module globals."""
    global _WORKER_EXPR, _WORKER_CTRL_IDX
    _WORKER_EXPR = expr
    _WORKER_CTRL_IDX = ctrl_idx


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _local_ctrl_adjustment(
    expr: np.ndarray,
    ctrl_idx: np.ndarray,
    embed: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    """
    Subtract local NT neighborhood mean from every cell's expression (in-place).

    For each cell (perturbed and control alike), find its n_neighbors nearest
    NT cells in embedding space and subtract their mean expression.  This
    removes technical / cell-type variation so the residual reflects true
    perturbation signal.

    Parameters
    ----------
    expr : (n_cells, n_genes) — modified in-place and returned
    ctrl_idx : indices of NT control cells
    embed : (n_cells, n_dims)  embedding used for neighbor search (e.g. PCA)
    n_neighbors : number of NT neighbors per cell

    Returns
    -------
    The same expr array, adjusted in-place.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=1)
    nn.fit(embed[ctrl_idx])

    # Only query non-control cells — control cells don't need adjustment
    pert_cell_idx = np.setdiff1d(np.arange(len(embed)), ctrl_idx)
    _, indices = nn.kneighbors(embed[pert_cell_idx])  # (n_pert, k) — local ctrl indices

    # Compute local means in batches to avoid materialising (n_pert, k, n_genes)
    ctrl_expr = expr[ctrl_idx]  # view into ctrl rows, reused across batches
    batch_size = 512
    for start in range(0, len(pert_cell_idx), batch_size):
        end = min(start + batch_size, len(pert_cell_idx))
        batch_global = pert_cell_idx[start:end]
        batch_nn_local = indices[start:end]             # (b, k) — local ctrl indices
        expr[batch_global] -= ctrl_expr[batch_nn_local].mean(axis=1)  # (b, n_genes)

    return expr


def _get_expr_sparse(adata: AnnData, layer: Optional[str]) -> sp.csr_matrix:
    """Return expression matrix as a CSR sparse matrix in float32."""
    X = adata.layers[layer] if layer is not None else adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X.copy()


def _wilcoxon_ranksum(a: np.ndarray, b: np.ndarray):
    """
    Vectorized two-sample Wilcoxon rank-sum test across columns.

    a : (n1, n_genes), b : (n2, n_genes)
    Returns (z_stats, p_vals) each of shape (n_genes,).
    Uses normal approximation with tie correction.
    """
    from scipy.stats import rankdata

    n1 = a.shape[0]
    n2 = b.shape[0]
    n = n1 + n2

    combined = np.vstack([a, b])                                      # (n, n_genes)
    ranks = np.apply_along_axis(rankdata, 0, combined)                # tie-corrected

    R1 = ranks[:n1].sum(axis=0)
    U1 = R1 - n1 * (n1 + 1) / 2.0

    mu_U = n1 * n2 / 2.0
    # Tie correction term: sum(t^3 - t) per gene where t = tie group size
    tie_corr = np.zeros(ranks.shape[1])
    for j in range(ranks.shape[1]):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        tie_corr[j] = np.sum(counts ** 3 - counts)

    sigma_U = np.sqrt(n1 * n2 / (n * (n - 1)) * ((n ** 3 - n) / 12.0 - tie_corr / 12.0))
    sigma_U[sigma_U == 0] = 1.0

    z = (U1 - mu_U) / sigma_U
    p_vals = 2 * stats.norm.sf(np.abs(z))
    return z, p_vals


def _select_de_genes(
    expr: np.ndarray,
    perturbed_idx: np.ndarray,
    ctrl_idx: np.ndarray,
    n_top_genes: int = 100,
    lfc_threshold: float = 0.25,
    pval_cutoff: float = 0.05,
    min_de_genes: int = 5,
    min_pct: float = 0.05,
) -> Optional[np.ndarray]:
    """
    Select top DE genes between perturbed and NT cells via Wilcoxon rank-sum
    test + LFC filter, matching Mixscale / Seurat FindMarkers defaults.
    Returns gene index array, or None if fewer than min_de_genes pass.
    """
    pert_expr = expr[perturbed_idx]
    ctrl_expr = expr[ctrl_idx]

    lfc = pert_expr.mean(axis=0) - ctrl_expr.mean(axis=0)

    # min_pct: gene must be detected (> 0) in >= min_pct fraction of either group
    pct_mask = ((pert_expr > 0).mean(axis=0) >= min_pct) | \
               ((ctrl_expr > 0).mean(axis=0) >= min_pct)
    lfc_gene_idx = np.where((np.abs(lfc) >= lfc_threshold) & pct_mask)[0]

    if len(lfc_gene_idx) < min_de_genes:
        return None

    z_stats, p_vals = _wilcoxon_ranksum(pert_expr[:, lfc_gene_idx], ctrl_expr[:, lfc_gene_idx])
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    z_stats = np.nan_to_num(z_stats, nan=0.0)

    p_adj = np.minimum(p_vals * len(lfc_gene_idx), 1.0)
    sig_mask = p_adj < pval_cutoff

    if sig_mask.sum() < min_de_genes:
        return None

    sig_idx = lfc_gene_idx[sig_mask]
    order = np.argsort(-np.abs(z_stats[sig_mask]))
    return sig_idx[order[:n_top_genes]]


def _scalar_projection(
    expr: np.ndarray,
    perturbed_idx: np.ndarray,
    ctrl_idx: np.ndarray,
) -> np.ndarray:
    """
    Compute per-cell scalar projection scores.

        vec = mean(perturbed) - mean(ctrl)           (n_de_genes,)
        score_i = (expr_i · vec) / (vec · vec)       scalar per cell

    Returns scores for all cells in expr.
    """
    vec = expr[perturbed_idx].mean(axis=0) - expr[ctrl_idx].mean(axis=0)
    vec_sq_sum = np.dot(vec, vec)

    if vec_sq_sum == 0:
        return np.zeros(len(expr))

    return expr.dot(vec) / vec_sq_sum


def _scalar_projection_loo(
    expr: np.ndarray,
    perturbed_idx: np.ndarray,
    ctrl_idx: np.ndarray,
) -> np.ndarray:
    """
    Leave-one-out scalar projection scores.

    For each DE gene k, recompute the score excluding gene k.
    Returns a (n_cells, n_de_genes) matrix where column k gives the score
    computed without gene k.  Used to avoid circularity in weighted DEG testing.
    """
    vec = expr[perturbed_idx].mean(axis=0) - expr[ctrl_idx].mean(axis=0)
    vec_sq = vec * vec

    pvec_mat = expr * vec[np.newaxis, :]             # (n_cells, n_genes)
    total_num = pvec_mat.sum(axis=1, keepdims=True)  # (n_cells, 1)
    total_den = vec_sq.sum()

    loo_num = total_num - pvec_mat                   # (n_cells, n_genes)
    loo_den = total_den - vec_sq[np.newaxis, :]      # (1, n_genes)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(loo_den != 0, loo_num / loo_den, 0.0)


def _wmvreg_test(
    expr_sub: np.ndarray,
    loo_z: np.ndarray,
    local_pert: np.ndarray,
    local_ctrl: np.ndarray,
    pval_cutoff: float,
) -> np.ndarray:
    """
    Weighted multivariate regression DEG test using LOO scores.

    For each DE gene k, regresses its expression on the LOO score computed
    *without* gene k (to avoid circularity).  Control cells are included with
    a score of 0.  Returns a boolean mask over DE genes that pass after
    Bonferroni correction.

    Model: expr_k ~ loo_score_(-k) + intercept  (OLS, all cells)
    """
    n_de = expr_sub.shape[1]
    n_pert = len(local_pert)
    n_cells = len(local_pert) + len(local_ctrl)

    p_vals = np.ones(n_de)

    for k in range(n_de):
        # Perturbed cells use LOO score for gene k; controls score = 0
        x = np.zeros(n_cells)
        x[:n_pert] = loo_z[:, k]

        y = expr_sub[:, k]

        X = np.column_stack([x, np.ones(n_cells)])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        residuals = y - X @ coeffs
        sse = residuals @ residuals
        if sse == 0:
            continue

        se2 = sse / (n_cells - 2)
        # Var(beta_0) = se2 * (X'X)^-1[0,0]
        xTx = X.T @ X
        det = xTx[0, 0] * xTx[1, 1] - xTx[0, 1] ** 2
        if det == 0:
            continue
        var_b0 = se2 * xTx[1, 1] / det
        se_b0 = np.sqrt(var_b0)
        if se_b0 == 0:
            continue

        t_stat = coeffs[0] / se_b0
        p_vals[k] = 2 * stats.t.sf(np.abs(t_stat), df=n_cells - 2)

    p_adj = np.minimum(p_vals * n_de, 1.0)
    return p_adj < pval_cutoff


# ---------------------------------------------------------------------------
# Per-perturbation worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _score_one(
    pert: str,
    pert_idx: np.ndarray,
    n_top_genes: int,
    lfc_threshold: float,
    pval_cutoff: float,
    min_de_genes: int,
    min_pct: float,
    loo: bool,
    var_names: np.ndarray,
) -> tuple:
    """
    Score all cells for a single perturbation.
    Uses the module-level _WORKER_EXPR / _WORKER_CTRL_IDX set by the initializer.

    Returns
    -------
    (pert, pert_idx, z_scores, loo_df_or_None, de_gene_names_or_None)
    """
    expr_all = _WORKER_EXPR
    ctrl_idx = _WORKER_CTRL_IDX

    de_idx = _select_de_genes(
        expr=expr_all,
        perturbed_idx=pert_idx,
        ctrl_idx=ctrl_idx,
        n_top_genes=n_top_genes,
        lfc_threshold=lfc_threshold,
        pval_cutoff=pval_cutoff,
        min_de_genes=min_de_genes,
        min_pct=min_pct,
    )

    if de_idx is None:
        return (pert, pert_idx, None, None, None)

    all_idx = np.concatenate([pert_idx, ctrl_idx])
    expr_sub = expr_all[np.ix_(all_idx, de_idx)]

    n_pert = len(pert_idx)
    local_pert = np.arange(n_pert)
    local_ctrl = np.arange(n_pert, len(all_idx))

    loo_df = None
    final_de_names = var_names[de_idx]
    
    # Apply leave-one-out bias correction
    if loo:
        loo_raw = _scalar_projection_loo(expr_sub, local_pert, local_ctrl)
        loo_ctrl_vals = loo_raw[local_ctrl]
        loo_mean = loo_ctrl_vals.mean(axis=0)
        loo_std = loo_ctrl_vals.std(axis=0)
        loo_std[loo_std == 0] = 1.0
        loo_z = (loo_raw[local_pert] - loo_mean) / loo_std

        # wmvReg: re-test each initial DEG using its LOO score → corrected DEG set
        sig_mask = _wmvreg_test(expr_sub, loo_z, local_pert, local_ctrl, pval_cutoff)
        final_de_names = var_names[de_idx[sig_mask]]

        loo_df = pd.DataFrame(loo_z[:, sig_mask], columns=final_de_names)

    # Efficiency scores always project onto the initial DEG set (matching Mixscale)
    raw = _scalar_projection(expr_sub, local_pert, local_ctrl)
    mean_nt = raw[local_ctrl].mean()
    std_nt = raw[local_ctrl].std()
    if std_nt == 0:
        std_nt = 1.0
    z_scores = (raw[local_pert] - mean_nt) / std_nt

    return (pert, pert_idx, z_scores, loo_df, final_de_names)


def _scale_scores(scores: np.ndarray, q: float = 0.99):
    """Scale efficiency scores within this dataset to smooth extreme outliers."""
    scores = np.abs(scores)
    M = scores.max()
    return M * np.tanh(scores / M)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_perturbation_scores(
    adata: AnnData,
    perturbation_key: str = "perturbation",
    ctrl_key: str = "control",
    layer: Optional[str] = None,
    preprocess: bool = True,
    n_top_genes: int = 100,
    lfc_threshold: float = 0.25,
    pval_cutoff: float = 0.05,
    min_de_genes: int = 5,
    min_pct: float = 0.05,
    use_rep: Optional[str] = "X_pca",
    n_neighbors: int = 20,
    n_hvg: Optional[int] = 3000,
    n_pcs: int = 40,
    loo: bool = True,
    n_jobs: int = 4,
    output_key: str = "efficiency",
    output_loo_key: str = "efficiency_loo",
    output_deg_key: str = "efficiency_degs",
    scale_scores: bool = True,
    inplace: bool = False,
    copy: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute Mixscale per-cell perturbation efficiency scores.

    Each cell receives a continuous score reflecting how strongly its expression
    profile aligns with the perturbation direction for its assigned gRNA/target.
    Scores are z-score normalized relative to non-targeting control cells, so
    a score of 0 matches the average control and higher values indicate stronger
    perturbation effect.  Cells whose perturbation yields fewer than
    `min_de_genes` are left as NaN.

    Parameters
    ----------
    adata
        AnnData with log-normalised expression in `adata.X` or a named layer.
    perturbation_key
        Column in `adata.obs` containing perturbation labels.
    ctrl_key
        Value in `perturbation_key` that identifies non-targeting controls.
    layer
        Layer to use for expression (default: `adata.X`).
    preprocess
        If True, normalize (CPM), log1p, then z-score each gene before
        projection (matches Mixscale's scale.data slot).
    n_top_genes
        Maximum number of DE genes used to build the perturbation vector.
    lfc_threshold
        Minimum absolute log-fold change for DE gene selection.
    pval_cutoff
        Bonferroni-adjusted p-value cutoff for DE gene selection.
    min_de_genes
        Minimum DE genes required; perturbations below this are scored NaN.
    use_rep
        Key in `adata.obsm` used for KNN-based local control adjustment
        (e.g. ``'X_pca'``).  For each cell, its n_neighbors nearest NT cells
        are found in this space and their mean expression is subtracted before
        scoring, removing technical/cell-type variation.  Set to None to skip.
        If the key is absent from ``adata.obsm`` and ``preprocess=True``,
        PCA is computed internally and stored under this key.
    n_neighbors
        Number of NT neighbors used in local control adjustment.
    n_hvg
        Number of highly variable genes (by variance) used when computing PCA
        internally.  None uses all genes.  Has no effect when a pre-computed
        embedding is found in ``adata.obsm[use_rep]``.
    n_pcs
        Number of principal components to compute when PCA is run internally.
    loo
        If True, also compute leave-one-out scores (n_cells x n_de_genes per
        perturbation) and store them in `adata.uns[output_loo_key]` as a dict
        keyed by perturbation label.
    n_jobs
        Number of worker processes for parallel perturbation scoring.
    output_key
        Column name written to `adata.obs` for the z-scored efficiency score.
    output_loo_key
        Key written to `adata.uns` for the LOO score dict (only when loo=True).
    output_deg_key
        Key written to `adata.uns` mapping each perturbation to its DE gene list.
    inplace
        If True, write results into `adata` and return None.
        If False, return a DataFrame with efficiency scores indexed by cell barcode.
    verbose
        If True, show a tqdm progress bar.
    """
    # Avoid overwriting original adata
    if not inplace and copy:
        adata = adata.copy()
    logging.info('Extracting gene expression layer.')
    is_raw = _is_raw_counts(adata)
    logging.info(f'Raw counts detected: {is_raw}')
    labels = adata.obs[perturbation_key].values
    ctrl_idx = np.where(labels == ctrl_key)[0]

    # Subset to highly variable genes
    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        logging.info(f"Using {hvg_mask.sum()} pre-computed HVGs.")
        adata._inplace_subset_var(hvg_mask)
    elif n_hvg is not None and n_hvg < adata.n_vars:
        if is_raw:
            logging.info(f"Calculating {n_hvg} HVGs with seurat_v3.")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', subset=True, check_values=True)
        else:
            logging.info(f"Calculating {n_hvg} HVGs with default flavor.")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    # Apply further preprocessing to adata.X if option is toggled
    if preprocess and is_raw:
        logging.info('Preprocessing raw gene expression layer.')
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
    # Get adata.X
    X = adata.layers[layer] if layer is not None else adata.X
    # Convert to dense if not already
    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    # PCA + KNN control adjustment (computed on z-scored data when preprocessed)
    if use_rep is not None:
        if use_rep not in adata.obsm:
            logging.info('Computing PCA for KNN control adjustment.')
            sc.pp.pca(adata, n_comps=n_pcs)
        # Get PCA embedding from adata
        embed = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        # Do control cell KNN adjustments
        logging.info('Applying KNN control adjustment.')
        X = _local_ctrl_adjustment(X, ctrl_idx, embed, n_neighbors)
    # Get obs and var names
    var_names = np.asarray(adata.var_names)
    obs_names = adata.obs_names
    # Calculate scores for every non-control perturbation
    perturbations = [p for p in np.unique(labels) if p != ctrl_key]
    # Set empty scores and initialize dicts
    scores = pd.Series(np.nan, index=obs_names, name=output_key)
    loo_dict: dict = {}
    deg_dict: dict = {}
    # Execute in parallel
    futures_meta = {}  # future --> pert

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_worker,
        initargs=(X, ctrl_idx),
    ) as pool:
        # Create futures
        for pert in perturbations:
            pert_idx = np.where(labels == pert)[0]
            if len(pert_idx) == 0:
                continue
            future = pool.submit(
                _score_one,
                pert, pert_idx,
                n_top_genes, lfc_threshold, pval_cutoff, min_de_genes, min_pct,
                loo, var_names,
            )
            futures_meta[future] = pert

        iterator = as_completed(futures_meta)
        if verbose:
            iterator = tqdm(iterator, total=len(futures_meta), desc="Scoring perturbations")
        # Collect futures
        for future in iterator:
            pert, pert_idx, z_scores, loo_df, de_genes = future.result()

            if z_scores is None:
                continue

            scores.iloc[pert_idx] = z_scores
            deg_dict[pert] = de_genes.tolist()

            if loo and loo_df is not None:
                loo_df.index = obs_names[pert_idx]
                loo_dict[pert] = loo_df
    # Set NaNs to 0s
    scores.fillna(0, inplace=True)

    # Save values to adata object
    if inplace:
        logging.info(f'Setting adata.obs["{output_key}"].')
        adata.obs[output_key] = scores.values
        adata.uns[output_deg_key] = deg_dict
        if loo:
            adata.uns[output_loo_key] = loo_dict
        return None
    logging.info(f'Returning scores series.')
    # Scale scores to prevent extreme values if toggled
    if scale_scores:
        scores = _scale_scores(scores)
    # Return scores series to preserve barcodes
    return scores
