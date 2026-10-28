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

"""Fig. 3a "all seen perturbations" row: the TRAINED perturbations scored exactly like the zero-shot
rows, so the two summaries share one axis.

    cap     a fold's id_test / ood_test split capped at --cap cells per (dataset, perturbation), the
            zero-shot queries' cap (obs read with h5py; seeded)
    score   one fold model: queries (cache_queries.py output for the capped ID / OOD split) ranked
            against the SAME candidate table as the zero-shot score -- that fold's full 19,895-row
            vocabulary from its zero-shot cache, the 9,874 measured untrained rows replaced by the
            model's shipped fill (prior_missing_T). Pooled per perturbation over datasets (mean query,
            mean weight), weight = within-dataset percentile of the group's mean efficiency, x chance =
            top-10 / (10 / n_rows). 20 row-permutation nulls. Writes one JSON per fold.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def _u(M):
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def _obs(adata, cols):
    import h5py
    from anndata.io import read_elem
    with h5py.File(adata, 'r') as fh:
        return {c: np.asarray(read_elem(fh['obs'][c])) for c in cols}


def cmd_cap(a):
    o = _obs(a.adata, ['dataset', 'perturbation'])
    rng = np.random.default_rng(a.seed)
    for split in a.splits:
        with gzip.open(Path(a.fold_dir) / f'{split}.txt.gz', 'rt') as fh:
            idx = np.array([int(l) for l in fh if l.strip()], dtype=np.int64)
        key = np.char.add(np.char.add(o['dataset'][idx].astype(str), '|'),
                          o['perturbation'][idx].astype(str))
        # vectorised: a random rank within each group, keep ranks < cap (one argsort, no per-group scan)
        codes = np.unique(key, return_inverse=True)[1]
        r = rng.random(len(idx))
        srt = np.lexsort((r, codes))               # NOT `o`: that name holds the obs columns
        first = np.r_[0, np.flatnonzero(np.diff(codes[srt])) + 1]
        rank = np.arange(len(srt)) - np.repeat(first, np.diff(np.r_[first, len(srt)]))
        keep = np.sort(idx[srt[rank < a.cap]])
        out = Path(a.out_dir) / f'{split}_cap{a.cap}.txt.gz'
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, 'wt') as fh:
            fh.write('\n'.join(map(str, keep)) + '\n')
        print(f'[cap] {split}: {len(idx):,} -> {len(keep):,} cells, {codes.max() + 1:,} groups -> {out}',
              flush=True)


def cmd_score(a):
    import torch
    zc = np.load(a.zs_cache, allow_pickle=True)
    A = zc['atoms'].astype(np.float32); ntr = int(zc['n_trained'])
    cand = np.unique(np.asarray(zc['truth'], np.int64))            # the 9,874 measured untrained rows
    sd = torch.load(a.model_pt, map_location='cpu', weights_only=False)['model_state_dict']
    T = sd[next(k for k in sd if k.endswith('prior_missing_T'))].float().numpy()
    A = A.copy(); A[cand] = T[cand]                                 # the zero-shot score's table
    D = _u(A).astype(np.float32)
    ch10 = 10.0 / len(D)
    obs = _obs(a.adata, ['dataset', 'perturbation', 'efficiency'])
    rec = {'model_pt': a.model_pt, 'n_rows': int(len(D)), 'n_trained': ntr, 'chance10': ch10}
    rng = np.random.default_rng(0)
    for split, cache in (('ID', a.id_cache), ('OOD', a.ood_cache)):
        c = np.load(cache, allow_pickle=True)
        Q, truth = c['Q'].astype(np.float32), np.asarray(c['truth'], np.int64)
        keys = np.asarray(c['keys']).astype(str)
        assert truth.max() < ntr, 'seen-perturbation truth must index the trained block'
        # efficiency weight per query: mean over the group's cells in the split, within-dataset percentile
        with gzip.open(Path(a.fold_dir) / a.splits[split], 'rt') as fh:
            idx = np.array([int(l) for l in fh if l.strip()], dtype=np.int64)
        # keys for the split's cells only (not all 19.5M)
        ek = np.char.add(np.char.add(obs['dataset'][idx].astype(str), '|'), obs['perturbation'][idx].astype(str))
        ev = obs['efficiency'][idx].astype(float)
        import pandas as pd
        g = pd.Series(ev).groupby(ek).agg(lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan)
        raw = g.reindex(keys).to_numpy()
        qds = np.array([k.split('|', 1)[0] for k in keys])
        w = np.zeros(len(keys))
        for dd in np.unique(qds):
            m = (qds == dd) & np.isfinite(raw)
            if m.sum() > 1:
                o = np.argsort(np.argsort(np.clip(raw[m], 0, None), kind='stable'), kind='stable')
                w[m] = o / (m.sum() - 1)
        # pool per perturbation over datasets: mean query, mean weight (the zero-shot pooled rows)
        tu, inv = np.unique(truth, return_inverse=True)
        Qp = np.zeros((len(tu), Q.shape[1]), np.float64); np.add.at(Qp, inv, Q)
        Qp = _u(Qp).astype(np.float32)
        wp = np.bincount(inv, w) / np.bincount(inv)

        def xs(Dm):
            S = Qp @ Dm.T
            r = (S > S[np.arange(len(tu)), tu][:, None]).sum(1)
            h = (r < 10).astype(float)
            return float((wp * h).sum() / max(wp.sum(), 1e-12) / ch10), float(h.mean())
        wx, top10 = xs(D)
        null = []
        for _ in range(a.reps):
            P = D.copy(); P[:] = D[rng.permutation(len(D))]
            null.append(xs(P)[0])
        # per-QUERY top-1 (unpooled) for the F1 view: query = (dataset, perturbation)
        Qn = _u(Q)
        top1 = np.concatenate([np.argmax(Qn[i:i + 4096] @ D.T, axis=1) for i in range(0, len(Qn), 4096)])
        np.savez_compressed(Path(a.out).with_suffix(f'.{split}.npz'), keys=keys, truth=truth, top1=top1, w=w)
        rec[split] = {'w_zs_x': wx, 'top10': top10, 'n_unique': int(len(tu)), 'n_query': int(len(keys)),
                      'random_mean': float(np.mean(null)), 'shuffled': null}
        print(f'[{split}] {len(keys):,} queries, {len(tu):,} perturbations: {wx:.2f}x chance '
              f'(top10 {top10:.4f}); random {np.mean(null):.2f}x', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('cap')
    c.add_argument('--adata', required=True); c.add_argument('--fold-dir', required=True)
    c.add_argument('--splits', nargs='+', required=True); c.add_argument('--cap', type=int, required=True)
    c.add_argument('--seed', type=int, default=0); c.add_argument('--out-dir', required=True)
    s = sp.add_parser('score')
    s.add_argument('--model-pt', required=True); s.add_argument('--zs-cache', required=True)
    s.add_argument('--adata', required=True); s.add_argument('--fold-dir', required=True)
    s.add_argument('--id-cache', required=True); s.add_argument('--ood-cache', required=True)
    s.add_argument('--id-split', required=True); s.add_argument('--ood-split', required=True)
    s.add_argument('--reps', type=int, default=20); s.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'score':
        a.splits = {'ID': a.id_split, 'OOD': a.ood_split}
    {'cap': cmd_cap, 'score': cmd_score}[a.cmd](a)
