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

"""Supplementary Fig. S1C data: how similar is the response to the SAME perturbation across datasets?

Pseudobulk profiles from the store ('rep' variant: each perturbation's cells split at random into two
replicates), restricted to the 3,096 input-panel genes: log-CPM, minus the dataset's control profile
(ExPert-data src.decode.normalize / delta_vs_control, exactly as the gene-panel analyses).

  diagonal      split-half reliability: Pearson r between a perturbation's two replicates in that dataset
  off-diagonal  Pearson r between the (replicate-averaged) responses to the same perturbation in two
                datasets, over genes measured in both, for every perturbation the two datasets share
  null          the same with the perturbation pairing shuffled

Summary per pair = median over perturbations. Writes figS1c_response_similarity.csv (long: dataset_i,
dataset_j, n_shared, median_r, null_median_r, median_r_reproducible) where `reproducible` restricts to
perturbations whose split-half r >= --rep-min in BOTH datasets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rowcorr(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A * A).sum(1) * (B * B).sum(1))
    return np.where(den > 0, (A * B).sum(1) / np.maximum(den, 1e-12), np.nan)


def main(a):
    sys.path.insert(0, a.data_repo)
    from src.decode import delta_vs_control, load_pseudobulk, normalize
    import h5py
    store = Path(a.store)
    with h5py.File(a.db, 'r') as f:
        panel = set(np.asarray(f['var']['_index'].asstr()[:]))
    datasets = sorted(p.stem for p in (store / 'pseudobulk').glob('*.h5ad'))
    hv = pd.read_parquet(store / 'stats' / f'{datasets[0]}.hvg.parquet', columns=['ensembl_id', 'symbol'])
    ens = hv.ensembl_id[hv.symbol.isin(panel)].astype(str).to_numpy()
    print(f'{len(datasets)} datasets; {len(ens):,} panel genes (of {len(panel):,} symbols)', flush=True)

    prof, meas, rel = {}, {}, {}
    for ds in datasets:
        X, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad', genes=ens, variant='rep')
        meas[ds] = (X[~obs.is_ctrl.to_numpy()] > 0).any(0)
        P = delta_vs_control(normalize(X, obs), obs); del X
        pert = (~obs.is_ctrl.to_numpy()) & (obs.k.to_numpy() > 1)
        o = obs[pert].reset_index(drop=True); Pp = P[pert].astype(np.float64)
        lab = o.label.astype(str).to_numpy(); grp = o.group.astype(int).to_numpy()
        w = o.n_cells.to_numpy(np.float64)
        labs = np.unique(lab)
        idx = {l: i for i, l in enumerate(labs)}
        li = np.array([idx[l] for l in lab])
        # replicate A / B and the n_cells-weighted mean per perturbation
        A = np.zeros((len(labs), Pp.shape[1])); B = np.zeros_like(A)
        hasA = np.zeros(len(labs), bool); hasB = np.zeros(len(labs), bool)
        g0 = grp == grp.min()
        A[li[g0]] = Pp[g0]; hasA[li[g0]] = True
        B[li[~g0]] = Pp[~g0]; hasB[li[~g0]] = True
        M = np.zeros_like(A); W = np.zeros(len(labs))
        np.add.at(M, li, Pp * w[:, None]); np.add.at(W, li, w)
        M /= np.maximum(W, 1e-12)[:, None]
        both = hasA & hasB
        r = np.full(len(labs), np.nan)
        m = meas[ds]
        r[both] = rowcorr(A[both][:, m], B[both][:, m])
        prof[ds] = pd.DataFrame(M, index=labs)
        rel[ds] = pd.Series(r, index=labs)
        print(f'  {ds}: {len(labs):,} perturbations, split-half median r {np.nanmedian(r):.3f}', flush=True)
        del P, Pp, A, B, M

    rng = np.random.default_rng(0)
    rows = []
    for i, di in enumerate(datasets):
        rows.append(dict(dataset_i=di, dataset_j=di, n_shared=int(rel[di].notna().sum()),
                         median_r=float(rel[di].median()), null_median_r=np.nan,
                         median_r_reproducible=float(rel[di][rel[di] >= a.rep_min].median())))
        for dj in datasets[i + 1:]:
            sh = prof[di].index.intersection(prof[dj].index)
            g = meas[di] & meas[dj]
            if len(sh) < a.min_shared or g.sum() < 50:
                rows.append(dict(dataset_i=di, dataset_j=dj, n_shared=len(sh), median_r=np.nan,
                                 null_median_r=np.nan, median_r_reproducible=np.nan))
                continue
            Xi = prof[di].loc[sh].to_numpy()[:, g]; Xj = prof[dj].loc[sh].to_numpy()[:, g]
            r = rowcorr(Xi, Xj)
            rn = rowcorr(Xi, Xj[rng.permutation(len(sh))])
            ok = ((rel[di].reindex(sh) >= a.rep_min) & (rel[dj].reindex(sh) >= a.rep_min)).to_numpy()
            rows.append(dict(dataset_i=di, dataset_j=dj, n_shared=len(sh), median_r=float(np.nanmedian(r)),
                             null_median_r=float(np.nanmedian(rn)),
                             median_r_reproducible=float(np.nanmedian(r[ok])) if ok.sum() >= 10 else np.nan,
                             n_reproducible=int(ok.sum())))
        print(f'  pairs with {di} done', flush=True)
    T = pd.DataFrame(rows)
    # symmetric long table
    T2 = T[T.dataset_i != T.dataset_j].rename(columns={'dataset_i': 'dataset_j', 'dataset_j': 'dataset_i'})
    T = pd.concat([T, T2], ignore_index=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / 'figS1c_response_similarity.csv', index=False)
    off = T[T.dataset_i != T.dataset_j]
    print(f'\noff-diagonal median r: median {off.median_r.median():.3f} (null {off.null_median_r.median():.3f}); '
          f'diagonal split-half: median {T[T.dataset_i == T.dataset_j].median_r.median():.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True)
    ap.add_argument('--db', required=True, help='training database_perturbed.h5ad (panel genes)')
    ap.add_argument('--data-repo', required=True, help='ExPert-data repo root (src.decode)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-shared', type=int, default=20)
    ap.add_argument('--rep-min', type=float, default=0.2)
    main(ap.parse_args())
