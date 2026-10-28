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

"""Supplementary Fig. S1D data: how much single-cell expression variance is explained by the perturbation?

Single cells from the training DB on the 3,096 input-panel genes, log1p(counts per 10^4 panel counts).
Per dataset: up to --n-perts selected perturbations with >= --cells-per-pert cells, --cells-per-pert cells
each, plus --n-ctrl control cells (database_ctrl.h5ad; controls form their own group).

Sequential partition of the total sum of squares (summed over genes):
  library       linear regression on log panel depth (the DB stores panel genes only, so this is depth over
                the 3,096 panel genes, not the full transcriptome)
  perturbation  one-way ANOVA on the library residuals, bias-adjusted for the number of groups
                (SS_between - (k-1) MS_within), so many small groups do not inflate it
  dataset       (pooled analyses only) entered BEFORE perturbation: a perturbation label is spread unevenly over
                datasets, so entered first it would absorb dataset differences and overstate the perturbation
  cell cycle    regression on S and G2/M scores (mean z-score of the Seurat cell-cycle genes in the panel)
  mitochondrial regression on the fraction of panel counts from MT- genes
  ribosomal     regression on the fraction of panel counts from RPL / RPS genes
  residual      split into SAMPLING NOISE, estimated by count splitting (each cell's counts split binomially
                into two halves; the variance the halves do not share is sampling noise; an upper bound
                under the log transform), and UNEXPLAINED (the rest)

Pooled across datasets (equal cells per dataset), twice: raw, and after subtracting each dataset's control
mean (the per-dataset control centring ExPert applies to its input). Perturbation identity is the gene label
shared across datasets. Writes figS1d_variance_explained.csv (scope, dataset, component, fraction).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def logcpm(X):
    X = sp.csr_matrix(X, dtype=np.float32)
    d = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(d, 1)) @ X
    X.data = np.log1p(X.data)
    return X.toarray(), np.log(np.maximum(d, 1))


def anova_ss(X, labels):
    """Total, between, within SS summed over genes (X already centred per gene or not -- uses group means)."""
    codes, uniq = pd.factorize(labels)
    k, n = len(uniq), len(codes)
    I = sp.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(k, n))
    cnt = np.asarray(I.sum(1)).ravel()
    mu = X.mean(0)
    gm = np.asarray(I @ X) / cnt[:, None]
    ss_t = float(((X - mu) ** 2).sum())
    ss_b = float((cnt[:, None] * (gm - mu) ** 2).sum())
    ss_w = ss_t - ss_b
    adj = max(ss_b - (k - 1) * ss_w / max(n - k, 1), 0.0)
    return ss_t, adj, gm, codes


# Tirosh et al. 2016 / Seurat cc.genes.updated.2019; scored on the subset present in the input panel
S_GENES = ('MCM5 PCNA TYMS FEN1 MCM7 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 CENPU HELLS RFC2 POLR1B NASP '
           'RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM '
           'CASP8AP2 USP1 CLSPN POLA1 CHAF1B MRPL36 E2F8').split()
G2M_GENES = ('HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 PIMREG SMC4 '
             'CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C '
             'KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 '
             'G2E3 GAS2L3 CBX5 CENPA').split()


def regress_out(R, C):
    """Remove the least-squares fit of the (centred) covariates C from R; return (R', SS explained)."""
    C = C - C.mean(0)
    B, *_ = np.linalg.lstsq(C, R, rcond=None)
    F = C @ B
    return R - F, float((F ** 2).sum())


def partition(X, steps):
    """Sequential variance partition of X (cells x genes). steps: [(name, kind, data)] with kind 'cov'
    (continuous covariates, n x k) or 'group' (labels; bias-adjusted one-way ANOVA)."""
    R = X - X.mean(0)
    ss_tot = float((R ** 2).sum())
    out = {}
    for name, kind, data in steps:
        if kind == 'cov':
            R, ss = regress_out(R, np.asarray(data, np.float64).reshape(len(R), -1))
            out[name] = ss / ss_tot
        else:
            _, adj, gm, codes = anova_ss(R, data)
            out[name] = adj / ss_tot
            R = R - (gm[codes] - R.mean(0))
    out['residual'] = max(1.0 - sum(out.values()), 0.0)
    return out, ss_tot


def noise_fraction(A, B, X):
    """Technical sampling noise by count splitting: counts split binomially into halves A and B; the
    variance the halves share (sum over genes of cov(A, B)) estimates the signal, so 1 - shared / total is
    the sampling-noise share of X's variance (an upper bound under the log transform)."""
    Ac, Bc, Xc = A - A.mean(0), B - B.mean(0), X - X.mean(0)
    shared = float((Ac * Bc).sum())
    return min(max(1.0 - shared / float((Xc ** 2).sum()), 0.0), 1.0)


def main(a):
    import h5py
    sys.path.insert(0, a.repo)
    from val_input_gene_gap import read_rows
    rng = np.random.default_rng(a.seed)
    sel = set(pd.read_parquet(a.perts).query('selected').perturbation.astype(str))

    def cat(o, k):
        g = o[k]
        if isinstance(g, h5py.Group):                 # categorical codes: never 19.5M Python strings
            return pd.Categorical.from_codes(g['codes'][:], np.asarray(g['categories'].asstr()[:]))
        return pd.Categorical(g.asstr()[:] if g.dtype.kind in 'OS' else g[:])

    with h5py.File(a.db, 'r') as h:
        obs = pd.DataFrame({'dataset': cat(h['obs'], 'dataset'), 'perturbation': cat(h['obs'], 'perturbation')})
        pick = []
        gidx = obs.groupby(['dataset', 'perturbation'], observed=True).indices   # (ds, pert) -> rows, once
        by_ds = {}
        for (ds, p), idx in gidx.items():
            if len(idx) >= a.cells_per_pert and str(p) in sel:
                by_ds.setdefault(ds, []).append(idx)
        for ds in sorted(by_ds):
            groups = by_ds[ds]
            for j in rng.choice(len(groups), min(len(groups), a.n_perts), replace=False):
                pick.append(rng.choice(groups[j], a.cells_per_pert, replace=False))
        rows = np.sort(np.concatenate(pick))
        print(f'[db] {len(rows):,} perturbed cells sampled', flush=True)
        X = read_rows(h, rows)
        o = obs.loc[rows].astype(str).reset_index(drop=True)
    with h5py.File(a.ctrl_db, 'r') as h:
        cds = h['obs']['dataset'].asstr()[:]
        crow = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(cds == ds), min(int((cds == ds).sum()), a.n_ctrl), replace=False)
            for ds in o.dataset.unique()]))
        Xc = read_rows(h, crow)
    o = pd.concat([o, pd.DataFrame({'dataset': cds[crow], 'perturbation': 'control'})], ignore_index=True)
    C = sp.vstack([X, Xc]).tocsr()
    del X, Xc
    with h5py.File(a.db, 'r') as h:
        genes = np.asarray(h['var']['_index'].asstr()[:])
    E, lib = logcpm(C)
    # count splitting (binomial thinning) for the sampling-noise floor
    A = C.copy(); A.data = rng.binomial(C.data.astype(np.int64), 0.5).astype(np.float32)
    Bm = C.copy(); Bm.data = (C.data - A.data).astype(np.float32)
    EA, _ = logcpm(A); EB, _ = logcpm(Bm)
    del A, Bm
    # biological covariates: cell-cycle scores, mitochondrial and ribosomal fraction
    tot = np.asarray(C.sum(1)).ravel().astype(np.float64)
    frac = lambda m: np.asarray(C[:, np.flatnonzero(m)].sum(1)).ravel() / np.maximum(tot, 1)
    mt = frac(np.char.startswith(genes.astype(str), 'MT-'))
    rp = frac(np.char.startswith(genes.astype(str), 'RPL') | np.char.startswith(genes.astype(str), 'RPS'))
    sidx = np.flatnonzero(np.isin(genes, S_GENES)); gidx = np.flatnonzero(np.isin(genes, G2M_GENES))
    print(f'[cov] cell-cycle genes in panel: S {len(sidx)}/{len(S_GENES)}, G2M {len(gidx)}/{len(G2M_GENES)}',
          flush=True)
    del C
    print(f'[expr] {E.shape[0]:,} cells x {E.shape[1]:,} genes', flush=True)

    def cc_scores(Y):
        Z = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
        return np.c_[Z[:, sidx].mean(1), Z[:, gidx].mean(1)]

    def steps(m, pooled):
        st = [('library', 'cov', lib[m])]
        if pooled:
            st.append(('dataset', 'group', o.dataset.to_numpy()[m]))
        st += [('perturbation', 'group', o.perturbation.to_numpy()[m]),
               ('cell cycle', 'cov', cc_scores(E[m])),
               ('mitochondrial', 'cov', mt[m]),
               ('ribosomal', 'cov', rp[m])]
        return st

    def record(scope, ds, r, nf):
        # split the residual into sampling noise (capped at the residual) and unexplained
        r = dict(r); res = r.pop('residual'); nz = min(nf, res)
        r['sampling noise'] = nz; r['unexplained'] = res - nz
        return [dict(scope=scope, dataset=ds, component=k, fraction=v) for k, v in r.items()]

    res = []
    for ds in sorted(o.dataset.unique()):
        m = (o.dataset == ds).to_numpy()
        r, _ = partition(E[m], steps(m, False))
        rows = record('dataset', ds, r, noise_fraction(EA[m], EB[m], E[m]))
        res += rows
        print(f'  {ds}: ' + '  '.join(f"{x['component']} {x['fraction']:.3f}" for x in rows), flush=True)
    allm = np.ones(len(o), bool)
    ctrl = (o.perturbation == 'control').to_numpy()

    def centred(Y):                                    # per-dataset control centring, as ExPert's input
        Y = Y.copy()
        for ds in o.dataset.unique():
            m = (o.dataset == ds).to_numpy()
            Y[m] -= Y[m & ctrl].mean(0)
        return Y

    for scope, f in (('pooled, raw', lambda Y: Y), ('pooled, control-centred', centred)):
        Y = f(E)
        r, _ = partition(Y, steps(allm, True))
        rows = record(scope, 'all', r, noise_fraction(f(EA), f(EB), Y))
        res += rows
        print(f'  {scope}: ' + '  '.join(f"{x['component']} {x['fraction']:.3f}" for x in rows), flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(out / 'figS1d_variance_explained.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', required=True, help='database_perturbed.h5ad')
    ap.add_argument('--ctrl-db', required=True, help='database_ctrl.h5ad')
    ap.add_argument('--perts', required=True, help='fig1c_perts.parquet (selected flag)')
    ap.add_argument('--repo', required=True, help='dir holding val_input_gene_gap.py (read_rows)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-perts', type=int, default=300)
    ap.add_argument('--cells-per-pert', type=int, default=20)
    ap.add_argument('--n-ctrl', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
