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

"""Fig. 6 donor-set contrast (the Fig. 5 'rrep' design on the mitochondrial cohort): unbiased pathway hits that
replicate across independent donor sets.

Conditions: PMD (PT43/46/54/67), TBCK (PT41/42), undiagnosed (PT38/50/55). A SPLIT puts the donors into two disjoint
sets, each with 2 PMD, 1 TBCK and 1-2 undiagnosed donors; every such split is enumerated (36), so no pairing is chosen
by hand. Within a set and cell type, each condition's centroid (equal-weight mean of its donors' z centroids) is
contrasted with the mean of the set's condition centroids, scored by cosine against the bank, and every gene set
(Reactome 10-500 in the bank + the collaborator modules) gets the Fig. 5 set z (mean percentile vs random sets).
Threshold = the --alpha two-sided quantile of the NON-PMD conditions' set z (TBCK and undiagnosed play the role of
Fig. 5's Normal rows). A set is REPLICATED in a split when both halves pass the threshold with the same sign; the
replication RATE is over the 36 splits, reported for PMD and for the non-PMD conditions (= the false-positive rate).

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_donorset.py --data-dir <pmd_mtmt> --bank-npz ... --out ...
"""
from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig6_pmd import load_sets, unit

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_donorset')
log.setLevel(logging.INFO)

COND = {'PT43': 'PMD', 'PT46': 'PMD', 'PT54': 'PMD', 'PT67': 'PMD', 'PT41': 'TBCK', 'PT42': 'TBCK',
        'PT38': 'undiagnosed', 'PT50': 'undiagnosed', 'PT55': 'undiagnosed'}


def splits():
    """Every unordered split into two sets of (2 PMD, 1 TBCK, 1-2 undiagnosed)."""
    pmd = [d for d, c in COND.items() if c == 'PMD']
    tb = [d for d, c in COND.items() if c == 'TBCK']
    un = [d for d, c in COND.items() if c == 'undiagnosed']
    out = []
    for p in itertools.combinations(pmd, 2):
        if pmd[0] not in p:                                   # unordered: set A always holds the first PMD donor
            continue
        for t in tb:
            for k in (1, 2):
                for u in itertools.combinations(un, k):
                    A = list(p) + [t] + list(u)
                    out.append((A, [d for d in COND if d not in A]))
    return out


def set_z_rows(S, M, sizes):
    """Fig. 5 set z for every row of S (loss tail = high cosine)."""
    from scipy.stats import rankdata
    N = S.shape[1]
    pct = (rankdata(S, axis=1) - 1) / (N - 1)
    mu = np.asarray(M.T @ pct.T).T / sizes
    sd = np.sqrt(pct.var(1, keepdims=True) / sizes * (N - sizes) / (N - 1))
    return (mu - pct.mean(1, keepdims=True)) / sd


def main(a):
    src = Path(a.data_dir)
    Z = np.load(src / 'z_cells.npy').astype(np.float64)
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    b = np.load(a.bank_npz, allow_pickle=True)
    SPL = splits()
    log.info('%d splits', len(SPL))
    rows = []
    for bank in ('trained', 'full'):
        keep = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
        labels = [str(l) for l in b['labels'][keep]]
        Pn = unit(b['P'][keep].astype(np.float64))
        sets = load_sets(a, labels)
        pos = {l: i for i, l in enumerate(labels)}
        keys = [k for k, v in sets.items() if len(v) >= 5]
        ii = np.concatenate([[pos[g] for g in sets[k]] for k in keys])
        jj = np.concatenate([[j] * len(sets[k]) for j, k in enumerate(keys)])
        M = sp.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(labels), len(keys)))
        sizes = np.asarray(M.sum(0)).ravel()
        for t in a.celltypes.split(','):
            mt = (obs.celltype == t).to_numpy()
            cen = {d: Z[mt & (obs.donor == d).to_numpy()].mean(0) for d in COND
                   if (mt & (obs.donor == d).to_numpy()).sum() >= a.min_cells}
            Q, meta = [], []
            for s, halves in enumerate(SPL):
                for h, members in enumerate(halves):
                    ms = [d for d in members if d in cen]
                    conds = sorted({COND[d] for d in ms})
                    if len(conds) < 3:                        # a half needs all three conditions to contrast
                        continue
                    C = {c: np.mean([cen[d] for d in ms if COND[d] == c], 0) for c in conds}
                    mu = np.mean(list(C.values()), 0)
                    for c in conds:
                        Q.append(C[c] - mu)
                        meta.append((s, h, c))
            if not Q:
                continue
            Zs = set_z_rows(unit(np.stack(Q)) @ Pn.T, M, sizes)
            for (s, h, c), z in zip(meta, Zs):
                rows.append(pd.DataFrame({'bank': bank, 'celltype': t, 'split': s, 'half': h, 'condition': c,
                                          'family': [k[0] for k in keys], 'set': [k[1] for k in keys], 'z': z}))
            log.info('[%s] %s: %d contrasts', bank, t, len(Q))
    T = pd.concat(rows, ignore_index=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / 'donorset_z.csv.gz', index=False)
    # threshold from the non-PMD conditions (both tails), per bank
    rep = []
    for bank, x in T.groupby('bank'):
        null = x.z[x.condition != 'PMD'].abs()
        thr = float(np.quantile(null, 1 - a.alpha))
        w = x.pivot_table(index=['celltype', 'condition', 'family', 'set', 'split'], columns='half', values='z')
        w = w.dropna()
        both_loss = (w[0] > thr) & (w[1] > thr)
        both_gain = (w[0] < -thr) & (w[1] < -thr)
        g = pd.DataFrame({'loss': both_loss, 'gain': both_gain, 'z_mean': (w[0] + w[1]) / 2}).groupby(
            level=['celltype', 'condition', 'family', 'set'])
        r = pd.DataFrame({'rep_loss': g.loss.mean(), 'rep_gain': g.gain.mean(), 'z_mean': g.z_mean.mean(),
                          'n_splits': g.size()}).reset_index()
        r.insert(0, 'bank', bank)
        r['threshold'] = thr
        rep.append(r)
    R = pd.concat(rep, ignore_index=True)
    R.to_csv(out / 'donorset_replication.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 300, 'display.max_colwidth', 60)
    for bank, x in R.groupby('bank'):
        log.info('[%s] threshold |z| %.2f', bank, x.threshold.iloc[0])
        for t, y in x.groupby('celltype'):
            for tail in ('rep_loss', 'rep_gain'):
                top = y[(y.condition == 'PMD') & (y[tail] > 0)].nlargest(12, [tail, 'z_mean'] if tail == 'rep_loss'
                                                                        else [tail])
                fp = y[y.condition != 'PMD'][tail]
                log.info('%s %s %s: %d PMD sets replicated in >= half the splits (non-PMD: %d) | %s', bank, t, tail,
                         int((y[(y.condition == 'PMD')][tail] >= 0.5).sum()), int((fp >= 0.5).sum()),
                         '; '.join(f'{s[:34]} {v:.2f}' for s, v in zip(top.set, top[tail])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='fig6_pmd encode dir (z_cells.npy, obs_cells.csv.gz)')
    ap.add_argument('--bank-npz', required=True)
    ap.add_argument('--modules', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--celltypes', default='Monocytes,Dendritic cells,B cells,NK cells,CD4+ T cells,Nonclassical T cells')
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--alpha', type=float, default=0.01, help='non-PMD conditions exceed the threshold this often')
    main(ap.parse_args())
