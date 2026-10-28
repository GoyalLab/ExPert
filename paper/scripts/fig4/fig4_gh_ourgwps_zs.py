# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Does enabling ExPert's zero-shot bank rows buy any doubles recovery the dictionary cannot reach?

The 8,347-column fingerprint dictionary is a strict subset of ExPert's 8,964 trained perturbations, so
zero-shot rows add NOTHING to the shared candidate list -- the fig4l/fig4m head-to-head cannot move.
What they can do is cover doubles the dictionary drops. This script factorises the two effects by
crossing candidate pools with query sets:

  pools    dict    the 8,347 shared columns          (what fig4l/fig4m rank over)
           trained the full 8,964 trained block      (pool grows, still no f(text) rows)
           full    all 19,895 bank rows              (trained + zero-shot f(text))
  queries  shared  doubles with both genes in dict   -- the head-to-head set; measures DISTRACTOR COST
           extra   doubles droppped by dict but placeable in the bank -- measures NEW COVERAGE

Fingerprinting is structurally at zero on `extra` (no dictionary column exists for those genes), so any
non-trivial recall there is coverage no fingerprint arm can match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig4_common as fc

KS = [1, 2, 3, 5, 10, 20, 50, 100, 500]
unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def ranks_of(S, desc, ia, ib):
    """(rank_a, rank_b); both/any are their max/min -- see fig4_gh_ourgwps.ranks_of.

    The per-gene ranks are kept because `any` alone cannot tell which gene of a pair was recovered: a
    mixed double (one trained gene, one zero-shot gene) could score entirely off its trained partner,
    which would make any zero-shot claim vacuous.
    """
    o = np.argsort(-S, 1) if desc else np.argsort(S, 1)
    R = np.empty_like(o)
    np.put_along_axis(R, o, np.arange(1, o.shape[1] + 1)[None, :].repeat(o.shape[0], 0), axis=1)
    rr = np.arange(S.shape[0])
    return R[rr, ia].astype(float), R[rr, ib].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encode-root', required=True)
    ap.add_argument('--fp-root', required=True)
    ap.add_argument('--pair-ranks', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--screens', required=True, help='screen:fp_subdir:tail:modality, comma separated')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_all = pd.read_csv(a.pair_ranks)

    per, meta = [], {}
    for spec in a.screens.split(','):
        s, sub, tail, mod = spec.split(':')
        e = Path(a.encode_root) / s
        pip = pd.read_csv(Path(a.fp_root) / sub / 'pip.csv', index_col=0, nrows=1)
        obs = pd.read_csv(e / 'obs.csv', keep_default_na=False)
        b = np.load(e / 'bank.npz', allow_pickle=True)
        P, L, nl = b['P'].astype(np.float64), b['labels'].astype(str), int(b['n_labels'])
        ctrl = int(np.flatnonzero(obs.condition == fc.CONTROL)[0])
        desc = tail == 'loss'
        assert len(set(L)) == len(L), f'{s}: duplicate bank labels'
        bpos = {g: i for i, g in enumerate(L)}

        # ---- the three candidate pools, each an ordered gene list into bank rows -------------------
        trained, dict_cols = L[:nl], set(pip.columns)
        pools = {'dict': [g for g in L[:nl] if g in dict_cols],
                 'trained': list(trained),
                 'full': list(L)}
        in_dict = set(pools['dict'])

        # ---- matched doubles (same matching as fig4_gh_ourgwps) -------------------------------------
        pr = pairs_all[pairs_all.screen == sub]
        gmap = {r.double: (r.gene_a, r.gene_b) for r in pr.itertuples()}
        q = obs.index[obs.order >= 1].to_numpy()
        tg = [t.split(';') for t in obs.targets[q]]
        order = obs.order[q].to_numpy()
        by_pair = {frozenset(t): i for i, t, o in zip(q, tg, order) if o == 2 and len(t) == 2}
        panel = {tuple(t)[0] for t, o in zip(tg, order) if o == 1}

        rows, ga, gb = [], [], []
        for d, k in ((d, frozenset(gmap[d])) for d in pd.read_csv(
                Path(a.fp_root) / sub / 'pip.csv', index_col=0, usecols=[0]).index if d in gmap):
            if k not in by_pair:
                continue
            x, y = sorted(k)
            rows.append(by_pair[k]); ga.append(x); gb.append(y)
        if not rows:
            print(f'[{s}] no matched doubles -- skipped', flush=True)
            continue
        rows, ga, gb = np.array(rows), np.array(ga), np.array(gb)

        Z = np.load(e / 'z_mtmask.npy').astype(np.float64)
        D = unit(Z[rows] - Z[ctrl])

        # query sets: shared = both genes in the dictionary; extra = dropped by dict, in the bank
        shared = np.array([x in in_dict and y in in_dict for x, y in zip(ga, gb)])
        for pool, cands in pools.items():
            cpos = {g: j for j, g in enumerate(cands)}
            ok = np.array([x in cpos and y in cpos for x, y in zip(ga, gb)])
            if not ok.any():
                continue
            ia = np.array([cpos[g] for g in ga[ok]])
            ib = np.array([cpos[g] for g in gb[ok]])
            Pe = unit(P[np.array([bpos[g] for g in cands])])
            r_a, r_b = ranks_of(D[ok] @ Pe.T, desc, ia, ib)
            tset = set(trained)
            a_tr = np.array([x in tset for x in ga[ok]])
            b_tr = np.array([x in tset for x in gb[ok]])
            # rank of the zero-shot gene(s) alone, and of the trained gene(s) alone
            zs_r = np.where(a_tr, np.inf, r_a)
            zs_r = np.minimum(zs_r, np.where(b_tr, np.inf, r_b))
            tr_r = np.where(a_tr, r_a, np.inf)
            tr_r = np.minimum(tr_r, np.where(b_tr, r_b, np.inf))
            per.append(pd.DataFrame({
                'screen': s, 'modality': mod, 'pool': pool, 'n_cand': len(cands),
                'gene_a': ga[ok], 'gene_b': gb[ok], 'shared': shared[ok],
                'in_panel': [x in panel and y in panel for x, y in zip(ga[ok], gb[ok])],
                'a_trained': a_tr, 'b_trained': b_tr,
                'n_zs': (~a_tr).astype(int) + (~b_tr).astype(int),
                'zs_row': ~(a_tr & b_tr),
                'rank_a': r_a, 'rank_b': r_b,
                'rank_zs': zs_r, 'rank_tr': tr_r,
                'both': np.maximum(r_a, r_b), 'any': np.minimum(r_a, r_b)}))
        meta[s] = dict(modality=mod, n_doubles=len(rows), n_shared=int(shared.sum()),
                       n_extra=int((~shared).sum()), **{f'n_{k}': len(v) for k, v in pools.items()})
        print(f'[{s}] {len(rows)} doubles: {int(shared.sum())} shared / {int((~shared).sum())} extra; '
              f'pools ' + ' '.join(f'{k}={len(v)}' for k, v in pools.items()), flush=True)

    R = pd.concat(per, ignore_index=True)
    R.to_csv(out / 'zs_doubles_ranks.csv', index=False)
    (out / 'zs_meta.json').write_text(json.dumps(meta, indent=2))

    # ---- curves: recall@k and fold-over-random, per pool x query set x modality ----------------------
    # Pool size differs between regimes (8,347 vs 19,895), so RAW recall is not comparable across pools
    # while fold-over-random is: it is the only axis on which the trained and zero-shot regimes can be
    # read side by side. Random is exact for 2 distinct true genes in N candidates, k draws no-replace:
    #   any@k  = 1 - C(N-2,k)/C(N,k) = 1 - (N-k)(N-k-1)/(N(N-1))
    #   both@k =     C(k,2)/C(N,2)   = k(k-1)/(N(N-1))
    def rnd(met, k, N):
        if met == 'both':
            return k * (k - 1) / (N * (N - 1))
        return 1.0 - (N - k) * (N - k - 1) / (N * (N - 1))

    cur = []
    for (mod, pool, qs), g in R.assign(
            qset=np.where(R.shared, 'shared', 'extra')).groupby(['modality', 'pool', 'qset']):
        N = int(g.n_cand.iloc[0])
        for met in ('any', 'both', 'rank_zs', 'rank_tr'):
            # rank_zs / rank_tr are single-gene metrics, so chance is the 1-gene rate k/N
            sub = g[np.isfinite(g[met])] if met in ('rank_zs', 'rank_tr') else g
            if not len(sub):
                continue
            for k in KS:
                r = float((sub[met] <= k).mean())
                b = (k / N) if met in ('rank_zs', 'rank_tr') else rnd(met, k, N)
                cur.append(dict(modality=mod, pool=pool, qset=qs, metric=met, k=k, n=len(sub),
                                n_cand=N, recall=r, random=b, fold=r / b if b > 0 else np.nan,
                                n_hit=int((sub[met] <= k).sum())))
    C = pd.DataFrame(cur)
    C.to_csv(out / 'zs_doubles_curves.csv', index=False)

    for met in ('any', 'both', 'rank_zs', 'rank_tr'):
        for val in ('recall', 'fold'):
            print(f'\n########## {met}@k -- {val} ##########')
            q = C[(C.metric == met) & C.k.isin([1, 2, 5, 10, 50, 100])]
            print(q.pivot_table(index=['modality', 'qset', 'pool', 'n', 'n_cand'], columns='k',
                                values=val).round(3).to_string())



if __name__ == '__main__':
    main()
