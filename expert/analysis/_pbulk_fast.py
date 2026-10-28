# SPDX-License-Identifier: MIT
# ExPert: Copyright (c) 2024-2026 Leon Schwartz, Northwestern University, Goyal Lab.
"""Control-centred per-perturbation pseudobulk in ONE pass.

The obvious implementation -- loop datasets x perturbations, slice the backed h5ad, densify -- is
what made the UMAP job sit for 15 minutes without reaching its first print: 4,400 random-access
fancy-index reads against a backed CSR file, each densified separately.

Both preprocessing steps are sparsity-preserving, which is what makes the one-pass version legal:
CPM is a per-row scale, and log1p(0) = 0. So normalise in place, then take group means as a single
sparse matmul with a group-indicator matrix. Seconds instead of tens of minutes.
"""
import numpy as np
import scipy.sparse as sp


def _norm_log1p(X):
    """CPM + log1p, sparsity preserved (row scaling, and log1p fixes 0)."""
    X = sp.csr_matrix(X, copy=True).astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    inv = np.zeros_like(lib)
    nz = lib > 0
    inv[nz] = 1e4 / lib[nz]
    X = sp.diags(inv) @ X
    X.data = np.log1p(X.data)
    return X


def _group_mean(X, codes, n_groups):
    """Mean of X rows within each group code, as one sparse matmul."""
    ok = codes >= 0
    rows, cols = codes[ok], np.flatnonzero(ok)
    cnt = np.bincount(rows, minlength=n_groups).astype(np.float64)
    w = np.zeros_like(cnt)
    w[cnt > 0] = 1.0 / cnt[cnt > 0]
    G = sp.csr_matrix((w[rows], (rows, cols)), shape=(n_groups, X.shape[0]))
    return np.asarray((G @ X).todense() if sp.issparse(G @ X) else (G @ X)), cnt


def response_matrix(ada, ctrl, label_key='perturbation', batch_key='dataset', min_cells=15):
    """(perturbations, genes) control-centred response, averaged over datasets.

    Returns (perts, R). Per-dataset centring happens BEFORE the cross-dataset average -- that is
    what makes these responses rather than baselines.
    """
    lab = ada.obs[label_key].astype(str).to_numpy()
    ds = ada.obs[batch_key].astype(str).to_numpy()
    cds = ctrl.obs[batch_key].astype(str).to_numpy()

    dsets = np.unique(ds)
    d_ix = {d: i for i, d in enumerate(dsets)}
    Xc = _norm_log1p(ctrl.X[:] if not sp.issparse(ctrl.X) else ctrl.X)
    cm, ccnt = _group_mean(Xc, np.array([d_ix.get(d, -1) for d in cds]), len(dsets))
    del Xc

    X = _norm_log1p(ada.X[:] if not sp.issparse(ada.X) else ada.X)
    perts = np.unique(lab)
    p_ix = {p: i for i, p in enumerate(perts)}
    # one group per (dataset, perturbation)
    codes = np.array([d_ix[d] * len(perts) + p_ix[p] for d, p in zip(ds, lab)])
    PB, cnt = _group_mean(X, codes, len(dsets) * len(perts))
    del X

    PB = PB.reshape(len(dsets), len(perts), -1)
    cnt = cnt.reshape(len(dsets), len(perts))
    have_ctrl = ccnt >= min_cells
    acc = np.zeros((len(perts), PB.shape[-1]))
    num = np.zeros(len(perts))
    for di in range(len(dsets)):
        if not have_ctrl[di]:
            continue
        keep = cnt[di] >= min_cells
        if not keep.any():
            continue
        acc[keep] += PB[di, keep] - cm[di]
        num[keep] += 1
    ok = num > 0
    R = np.zeros_like(acc)
    R[ok] = acc[ok] / num[ok, None]
    return perts[ok], R[ok]
