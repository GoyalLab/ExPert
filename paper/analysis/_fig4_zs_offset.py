# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Is the zero-shot null caused by the f(missing) tail's shared offset / low-rank collapse?

The tail occupies ~75 effective dims against the trained atoms' ~446, with ||mean(unit)|| 0.199 vs
0.021. If that shared structure is what buries the true gene, projecting the tail's own top principal
directions out should lift it. Scored TAIL-ONLY (rank the true zero-shot gene among the 10,931
untrained rows, chance = k/10931) so the trained block cannot carry the result -- the free-ride trap
from the first pass.

The removed directions come from the tail's own SVD: unsupervised, no label information, so this is a
legitimate readout change rather than a fit to the answer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import fig4_common as fc

unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encode-root', required=True)
    ap.add_argument('--pair-ranks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--screens', required=True, help='screen:fp_subdir:tail:modality, comma separated')
    ap.add_argument('--ranks', default='0,1,2,4,8,16,32,64', help='# shared directions to remove')
    a = ap.parse_args()
    pairs_all = pd.read_csv(a.pair_ranks)
    RK = [int(x) for x in a.ranks.split(',')]

    rows = []
    for spec in a.screens.split(','):
        s, sub, tail, mod = spec.split(':')
        e = Path(a.encode_root) / s
        obs = pd.read_csv(e / 'obs.csv', keep_default_na=False)
        b = np.load(e / 'bank.npz', allow_pickle=True)
        P, L, nl = b['P'].astype(np.float64), b['labels'].astype(str), int(b['n_labels'])
        ctrl = int(np.flatnonzero(obs.condition == fc.CONTROL)[0])
        desc = tail == 'loss'
        Z = np.load(e / 'z_mtmask.npy').astype(np.float64)

        zs_lab = L[nl:]
        zpos = {g: i for i, g in enumerate(zs_lab)}
        pr = pairs_all[pairs_all.screen == sub]
        gmap = {r.double: (r.gene_a, r.gene_b) for r in pr.itertuples()}
        q = obs.index[obs.order >= 1].to_numpy()
        tg = [t.split(';') for t in obs.targets[q]]
        order = obs.order[q].to_numpy()
        by_pair = {frozenset(t): i for i, t, o in zip(q, tg, order) if o == 2 and len(t) == 2}

        # every (double row, zero-shot gene) pair: the gene must be IN the tail to be rankable there
        qi, gi, gn = [], [], []
        for d, (x, y) in gmap.items():
            k = frozenset((x, y))
            if k not in by_pair:
                continue
            for g in (x, y):
                if g in zpos:
                    qi.append(by_pair[k]); gi.append(zpos[g]); gn.append(g)
        if not qi:
            print(f'[{s}] no zero-shot genes among the doubles -- skipped', flush=True)
            continue
        qi, gi = np.array(qi), np.array(gi)
        D = unit(Z[qi] - Z[ctrl])
        T = P[nl:]
        N = T.shape[0]
        # tail principal directions (centred): the shared structure to strip
        Tn = unit(T)
        mu = Tn.mean(0)
        _, _, Vt = np.linalg.svd(Tn - mu, full_matrices=False)

        for r in RK:
            if r == 0:
                Te, De = Tn, D
            else:
                B = np.vstack([unit(mu[None, :]), Vt[:max(r - 1, 0)]]) if r >= 1 else Vt[:0]
                B = np.linalg.qr(B.T)[0].T                       # orthonormalise mean + top PCs
                Te = unit(Tn - (Tn @ B.T) @ B)
                De = unit(D - (D @ B.T) @ B)
            S = De @ Te.T
            o = np.argsort(-S, 1) if desc else np.argsort(S, 1)
            R = np.empty_like(o)
            np.put_along_axis(R, o, np.arange(1, N + 1)[None, :].repeat(len(o), 0), axis=1)
            rk = R[np.arange(len(gi)), gi].astype(float)
            for g, v in zip(gn, rk):
                rows.append(dict(screen=s, modality=mod, rank_removed=r, gene=g, rank=v, n_cand=N))
            print(f'[{s}] r={r:3d}  median rank {np.median(rk):7.0f}/{N}  pct {100*np.median(rk)/N:5.1f}%  '
                  f'top100 {(rk<=100).mean():.3f}  n={len(rk)}', flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)

    print('\n########## TAIL-ONLY rank of the true zero-shot gene, per GENE (not per double) ##########')
    for mod, g in R.groupby('modality'):
        print(f'\n--- {mod} ---')
        pg = g.groupby(['rank_removed', 'gene'])['rank'].median().reset_index()
        N = int(g.n_cand.iloc[0])
        out = []
        for r, gg in pg.groupby('rank_removed'):
            v = gg['rank'].to_numpy()
            out.append(dict(rank_removed=r, n_genes=len(v), median_pct=round(100 * np.median(v) / N, 1),
                            top100=int((v <= 100).sum()), top500=int((v <= 500).sum()),
                            top1000=int((v <= 1000).sum()),
                            fold_top500=round(((v <= 500).mean()) / (500 / N), 2)))
        print(pd.DataFrame(out).to_string(index=False))


if __name__ == '__main__':
    main()
