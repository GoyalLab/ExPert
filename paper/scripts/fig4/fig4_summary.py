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

"""Fig. 5b: target recovery per modality, comparable across screens.

--unit query (default): one entry per query (drug or degron clone), scored by the best rank over
its annotated targets (any target is a hit). --unit pair: one entry per (query, target gene); the
null for a pair is the other pairs' target genes ranked in the same query, excluding that query's
own targets (so a promiscuous drug cannot score its own other targets as 'null'). With --trained-only, target sets keep only trained genes and every ranking
is over the trained prototypes alone (zero-shot rows cannot compete).

R[i, j] = best rank of query j's target set in query i's ranking. The diagonal is observed; the
off-diagonal is the target-permutation null (another query's targets scored in this query's
ranking), which keeps each screen's target-sharing structure -- the reason absolute recall is not
comparable across screens (sci-Plex3 has ~25 HDAC inhibitors, so a shuffled set often still hits).

    recall@K fold   mean(R_ii <= K) / mean_{i != j}(R_ij <= K); exact binomial CI on the hit count
                    divided by the null rate (the null rests on n(n-1) pairs and is precise)
    rank test       per query: log10 R_ii vs mean_{j != i} log10 R_ij (that query's own expected rank
                    under the null). One-sided paired t-test (lower = better); Wilcoxon alongside
    medians         median observed rank, median per-query null rank

Writes the summary table (--out) and per-query ranks (--out-queries).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig4_tahoe_score as fs


def ranks(S):
    rk = np.empty(S.shape, np.int32)
    np.put_along_axis(rk, np.argsort(-S, axis=1), np.arange(1, S.shape[1] + 1)[None, :], axis=1)
    return rk


def rank_matrix(S, sets):
    rk = ranks(S)
    return np.stack([rk[:, s].min(1) for s in sets], axis=1), np.ones((len(sets), len(sets)), bool)


def pair_matrix(S, sets):
    """Entries = (query, target) pairs. R[p, j] = rank of pair j's gene in pair p's query; the
    valid null cells exclude every pair whose gene is one of p's own query's targets."""
    rk = ranks(S)
    q = np.concatenate([[i] * len(s) for i, s in enumerate(sets)])
    g = np.concatenate(sets)
    R = rk[q][:, g]
    own = np.array([set(sets[i]) for i in q], dtype=object)
    valid = np.array([[gj not in own[p] for gj in g] for p in range(len(q))])
    return R, valid


def main(a):
    from scipy.stats import beta, ttest_rel, wilcoxon
    rng = np.random.default_rng(a.seed)
    bank = np.load(a.bank, allow_pickle=True)
    n_trained = int(bank['n_labels'])
    rows, qrows = [], []
    for name, scores, truth, key_col, tgt_col, cls_col in a.dataset:
        z = np.load(scores, allow_pickle=True)
        labels = z['labels'].astype(str)
        S = z['S'].astype(np.float32)
        if a.trained_only:
            labels, S = labels[:n_trained], S[:, :n_trained]
        pos = {g: i for i, g in enumerate(labels)}
        keys = [str(k).strip() for k in z['keys'].astype(str)]
        t = pd.read_csv(truth)
        t['_key'] = t[key_col].astype(str).str.strip()
        t = t.drop_duplicates('_key').set_index('_key')
        keep, sets = [], []
        for i, k in enumerate(keys):
            if k not in t.index or (cls_col != '-' and str(t.loc[k, cls_col]) == 'nonlof'):
                continue
            s = np.array([pos[g] for g in fs.split(t.loc[k, tgt_col]) if g in pos], int)
            if len(s):
                keep.append(i); sets.append(s)
        R, valid = (pair_matrix if a.unit == 'pair' else rank_matrix)(S[keep], sets)
        R = R.astype(np.float64)
        n, C = len(R), S.shape[1]
        off = ~np.eye(n, dtype=bool) & valid
        obs = np.diag(R)
        null_q = np.array([np.mean(np.log10(R[i, off[i]])) for i in range(n)])
        tt = ttest_rel(np.log10(obs), null_q, alternative='less')
        wx = wilcoxon(np.log10(obs), null_q, alternative='less')
        base = dict(dataset=name, unit=a.unit, n=n, n_queries=len(keep), n_candidates=C,
                    trained_only=bool(a.trained_only),
                    median_rank=float(np.median(obs)), median_null_rank=float(np.median(10 ** null_q)),
                    t_stat=float(tt.statistic), p_ttest=float(tt.pvalue), p_wilcoxon=float(wx.pvalue),
                    mean_log10_lift=float(np.mean(null_q - np.log10(obs))))
        for k in a.ks:
            hit = obs <= k
            nul = (R[off] <= k).mean()
            x = int(hit.sum())
            lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
            hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
            # permutation p: reassign entries' truths at random among VALID cells of each row
            vi = [np.flatnonzero(off[i]) for i in range(n)]
            perm = np.array([np.mean([R[i, rng.choice(vi[i])] <= k for i in range(n)]) for _ in range(a.n_perm)])
            rows.append(dict(base, k=k, recall=float(hit.mean()), null=float(nul),
                             fold=float(hit.mean() / nul) if nul > 0 else np.nan,
                             fold_lo=float(lo / nul) if nul > 0 else np.nan,
                             fold_hi=float(hi / nul) if nul > 0 else np.nan,
                             n_hits=x, expected_hits=float(nul * n),
                             p_perm=float(((perm >= hit.mean()).sum() + 1) / (a.n_perm + 1))))
            print(f'[{name}] @{k:<4d} n={n:3d} recall {hit.mean():.3f} null {nul:.4f} fold {rows[-1]["fold"]:.2f} '
                  f'[{rows[-1]["fold_lo"]:.2f}, {rows[-1]["fold_hi"]:.2f}] p_perm {rows[-1]["p_perm"]:.4f}', flush=True)
        print(f'[{name}] median rank {base["median_rank"]:.0f} vs null {base["median_null_rank"]:.0f} of {C}; '
              f'paired t {tt.statistic:.2f} p {tt.pvalue:.2e}; wilcoxon p {wx.pvalue:.2e}', flush=True)
        qn = ([keys[i] for i in keep] if a.unit == 'query' else
              [f'{keys[keep[i]]}|{labels[gg]}' for i, s in enumerate(sets) for gg in s])
        qrows.append(pd.DataFrame({'dataset': name, 'query': qn, 'rank': obs,
                                   'null_rank_geomean': 10 ** null_q,
                                   'null_rank_median': [float(np.median(R[i, off[i]])) for i in range(n)],
                                   'log10_lift': null_q - np.log10(obs)}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    pd.concat(qrows).to_csv(a.out_queries, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', nargs=6, action='append', required=True,
                    metavar=('NAME', 'SCORES_NPZ', 'TRUTH_CSV', 'KEY_COL', 'TARGET_COL', 'CLASS_COL'),
                    help="CLASS_COL '-' = no class filter")
    ap.add_argument('--bank', required=True, help='bank.npz (n_labels = size of the trained block)')
    ap.add_argument('--trained-only', action='store_true')
    ap.add_argument('--unit', choices=('query', 'pair'), default='query')
    ap.add_argument('--out', required=True)
    ap.add_argument('--out-queries', required=True)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 10, 50])
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
