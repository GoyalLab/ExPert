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

"""Fig. 5 (degron): which clones should be recoverable at all? Data-only clone covariates.

Per clone with >= --min-cells nuclei in both arms (same nuclei filter as the model run):

    response_z     the clone's transcriptional response to depletion, beyond the shift every clone
                   shares: |mean(auxin) - mean(DMSO) - shared| over library-normalised log1p
                   expression, z-scored against SIZE-MATCHED PSEUDO-CLONES (n_auxin and n_DMSO nuclei
                   drawn from the whole pool, a mix of all clones). NOT a within-clone label shuffle:
                   that also shuffles away the DMSO/auxin library shift, so its null sits at |shared|
                   and every real clone scores negative
    response_rep   split-half REPRODUCIBILITY of that response: each arm's nuclei split in two at random,
                   Pearson r between the two half-clone (delta - shared) vectors, mean over --n-perm
                   splits. Size-aware without a null model: a real response replicates, noise does not
    target_expr    the degraded gene's own mean log1p expression in that clone's DMSO nuclei
                   (a degron on a barely expressed protein has little to remove)
    n_dmso, n_iaa  labelled nuclei per arm

Writes degron_covariates.csv (joined to the fig4_degron clones.csv when --clones is given).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main(a):
    import anndata as ad
    import scipy.sparse as sp
    rng = np.random.default_rng(a.seed)
    q = ad.read_h5ad(a.query)
    if a.max_pct_mt is not None:
        q = q[(q.obs[a.mt_col].astype(float) <= a.max_pct_mt).to_numpy()].copy()
    X = sp.csr_matrix(q.X, dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    X.data = np.log1p(X.data)
    if a.n_hvg:
        mu = np.asarray(X.mean(0)).ravel()
        sq = np.asarray(X.multiply(X).mean(0)).ravel()
        keep = np.argsort(-(sq - mu ** 2))[:a.n_hvg]
    else:
        keep = np.arange(X.shape[1])
    genes = np.asarray(q.var_names).astype(str)
    cond = q.obs[a.condition_col].astype(str).to_numpy()
    clone = q.obs[a.clone_col].astype(str).to_numpy()
    ctrl, trt = cond == a.control_condition, cond == a.treated_condition
    clones = [g for g in sorted(set(clone) - {a.unassigned})
              if ((clone == g) & ctrl).sum() >= a.min_cells and ((clone == g) & trt).sum() >= a.min_cells]
    Xk = X[:, keep].toarray()
    delta = {g: Xk[(clone == g) & trt].mean(0) - Xk[(clone == g) & ctrl].mean(0) for g in clones}
    shared = np.mean([delta[g] for g in clones], axis=0)
    gpos = {g: i for i, g in enumerate(genes)}
    rows = []
    pool_t, pool_c = np.flatnonzero(trt), np.flatnonzero(ctrl)
    for g in clones:
        idx = np.flatnonzero(clone == g)
        lab = trt[idx]
        obs = np.linalg.norm(delta[g] - shared)
        null = np.array([np.linalg.norm(Xk[rng.choice(pool_t, lab.sum(), replace=False)].mean(0)
                                        - Xk[rng.choice(pool_c, (~lab).sum(), replace=False)].mean(0) - shared)
                         for _ in range(a.n_perm)])
        it, ic = idx[lab], idx[~lab]
        rep = []
        for _ in range(a.n_perm):
            pt, pc = rng.permutation(it), rng.permutation(ic)
            h = [Xk[pt[k::2]].mean(0) - Xk[pc[k::2]].mean(0) - shared for k in (0, 1)]
            rep.append(np.corrcoef(h[0], h[1])[0, 1])
        te = (float(np.asarray(X[idx[~lab]][:, gpos[g]].mean())) if g in gpos else np.nan)
        rows.append(dict(clone=g, n_dmso=int((~lab).sum()), n_iaa=int(lab.sum()), response_norm=float(obs),
                         response_z=float((obs - null.mean()) / null.std()),
                         response_rep=float(np.nanmean(rep)), target_expr_dmso=te))
    out = pd.DataFrame(rows)
    if a.clones:
        c = pd.read_csv(a.clones)
        out = c.merge(out.drop(columns=['n_dmso', 'n_iaa']), on='clone', how='left')
    out.to_csv(a.out, index=False)
    from scipy.stats import spearmanr
    for x in ('response_rep', 'response_z', 'target_expr_dmso', 'n_iaa'):
        if 'cos_z_did' in out and x in out:
            r = spearmanr(out[x], out['cos_z_did'], nan_policy='omit')
            print(f'[cov] {x:18s} vs own-gene z: rho {r.statistic:+.2f} p {r.pvalue:.3f}', flush=True)
    print(out.sort_values('response_z', ascending=False).to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--query', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clones', default=None, help='fig4_degron.py clones.csv to join')
    ap.add_argument('--condition-col', required=True)
    ap.add_argument('--control-condition', required=True)
    ap.add_argument('--treated-condition', required=True)
    ap.add_argument('--clone-col', required=True)
    ap.add_argument('--unassigned', default='unassigned')
    ap.add_argument('--max-pct-mt', type=float, default=None)
    ap.add_argument('--mt-col', default='pct_counts_mt')
    ap.add_argument('--n-hvg', type=int, default=3000, help='0 = all genes')
    ap.add_argument('--min-cells', type=int, default=5)
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
