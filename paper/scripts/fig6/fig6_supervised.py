#!/usr/bin/env python
# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Fig. 6 supervised companion to the label-free clustering (panel E): how well do the donors' profiles predict
PMD vs non-PMD on held-out donors, ExPert latent vs expression?

Per cell type (donors with >= --min-cells cells):
    expert       each donor's centroid of the per-cell ExPert latent (fig6_pmd.py encode, z_cells.npy)
    expression   each donor's mean log1p(CP10k) over the cell type's --n-hvg highly variable genes (label-free HVGs)
Classifier: nearest centroid in cosine space, features centred on the TRAINING donors' mean; score of a held-out
donor = cos(x, PMD centroid) - cos(x, non-PMD centroid).
    split  every split with 2 of the 4 PMD donors and 2 or 3 of the 5 non-PMD donors in training (120): AUC on the
           held-out donors per split -> mean, share of perfectly separated splits
    loo    leave-one-donor-out: every donor scored once -> one AUC over all donors; exact null = the same LOO AUC for
           every relabelling with the same number of PMD donors

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_supervised.py --query <h5ad> --z-dir <pmd_mtmt> --out <csv>
"""
from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_supervised')
log.setLevel(logging.INFO)

PMD = ['PT43', 'PT46', 'PT54', 'PT67']
NON = ['PT38', 'PT50', 'PT55', 'PT41', 'PT42']


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def scores(X, y, train, test):
    """Nearest-centroid (cosine) margin for the test rows, centred on the training mean."""
    mu = X[train].mean(0)
    Xc = unit(X - mu)
    cp = unit(Xc[train][y[train]].mean(0))
    cn = unit(Xc[train][~y[train]].mean(0))
    return Xc[test] @ cp - Xc[test] @ cn


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def evaluate(X, donors):
    y = np.array([d in PMD for d in donors])
    ip, inn = np.flatnonzero(y), np.flatnonzero(~y)
    res = []
    for tp in itertools.combinations(ip, len(ip) // 2):
        for k in (len(inn) // 2, len(inn) - len(inn) // 2):
            for tn in itertools.combinations(inn, k):
                train = np.zeros(len(y), bool)
                train[list(tp) + list(tn)] = True
                res.append(auc(scores(X, y, train, ~train), y[~train]))
    res = np.array(res)

    def loo(lab):
        s = np.array([scores(X, lab, np.arange(len(lab)) != i, np.array([i]))[0] for i in range(len(lab))])
        return auc(s, lab), s
    a_loo, s_loo = loo(y)
    null = []
    for c in itertools.combinations(range(len(y)), int(y.sum())):
        lab = np.zeros(len(y), bool)
        lab[list(c)] = True
        null.append(loo(lab)[0])
    null = np.array(null)
    return dict(split_auc=float(np.nanmean(res)), split_auc_sd=float(np.nanstd(res)), n_splits=int(len(res)),
                split_perfect=float(np.nanmean(res == 1.0)), loo_auc=float(a_loo),
                loo_p=float(np.mean(null >= a_loo - 1e-12)), n_perm=int(len(null)),
                loo_scores=' '.join(f'{d}:{v:+.3f}' for d, v in zip(donors, s_loo)))


def main(a):
    import anndata as ad
    import scanpy as sc
    z = np.load(Path(a.z_dir) / 'z_cells.npy').astype(np.float64)
    zo = pd.read_csv(Path(a.z_dir) / 'obs_cells.csv.gz')
    q = ad.read_h5ad(a.query)
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(len(v), bool)
    for p in a.mask.split(','):
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    q = q[:, ~drop].copy()
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    dn = q.obs[a.donor_col].astype(str).to_numpy()
    rows = []
    for t in sorted(set(ct)):
        donors = [d for d in PMD + NON if ((ct == t) & (dn == d)).sum() >= a.min_cells]
        if sum(d in PMD for d in donors) < 2 or sum(d not in PMD for d in donors) < 2:
            continue
        mz = (zo.celltype == t).to_numpy()
        Xz = np.stack([z[mz & (zo.donor == d).to_numpy()].mean(0) for d in donors])
        x = q[(ct == t) & np.isin(dn, donors)].copy()
        d_ = dn[(ct == t) & np.isin(dn, donors)]
        x = x[:, np.asarray((x.X > 0).mean(0)).ravel() >= a.min_frac].copy()
        sc.pp.normalize_total(x, target_sum=1e4)
        sc.pp.log1p(x)
        sc.pp.highly_variable_genes(x, n_top_genes=min(a.n_hvg, x.n_vars))
        H = x.X[:, x.var.highly_variable.to_numpy()]
        Xe = np.stack([np.asarray(H[d_ == d].mean(0)).ravel() for d in donors])
        for feat, X in (('expert', Xz), ('expression', Xe)):
            r = evaluate(X, donors)
            rows.append(dict(celltype=t, features=feat, n_donors=len(donors),
                             n_pmd=sum(d in PMD for d in donors), **r))
            log.info('%s %-10s split AUC %.2f +- %.2f (perfect %.0f%%) | LOO AUC %.2f p %.3f', t, feat, r['split_auc'],
                     r['split_auc_sd'], 100 * r['split_perfect'], r['loo_auc'], r['loo_p'])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--query', required=True, help='PBMC h5ad (raw counts)')
    ap.add_argument('--z-dir', required=True, help='fig6_pmd.py encode dir (z_cells.npy, obs_cells.csv.gz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mask', default='MT-*,MT1*,MT2A,MT3,MT4', help='genes dropped (the ExPert input mask)')
    ap.add_argument('--celltype-col', default='singler_labels_main')
    ap.add_argument('--donor-col', default='patient_id')
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--min-frac', type=float, default=0.01)
    ap.add_argument('--n-hvg', type=int, default=2000)
    main(ap.parse_args())
