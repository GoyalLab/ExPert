import numpy as np
import pandas as pd
import logging
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm
from src.utils.constants import REGISTRY_KEYS


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def scale_1d_array(
        x: np.ndarray, 
        zero_center: bool = True, 
        max_value: float | None = None, 
        abs: bool = True,
        log: bool = True,
        use_sigmoid: bool = True,
        check_scale: bool = True,
    ) -> np.ndarray:
    x = x.astype(float)
    # Check if x is already bounded [0,1]
    if check_scale and x.min() >= 0 and x.max() <= 1:
        return x
    if abs:
        x = np.abs(x)
    if log:
        x = np.log10(x)
    if zero_center:
        mean = np.mean(x)
        x -= mean
    std = np.std(x)
    if std != 0:
        x /= std
    else:
        x[:] = 0.0  # Avoid division by zero
    if max_value is not None:
        x = np.clip(x, -max_value, max_value)
    if use_sigmoid:
        x = sigmoid(x)
    return x

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
