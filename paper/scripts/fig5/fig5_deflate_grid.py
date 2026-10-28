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

"""Deflated readout settings on the ALD winner configuration (fig5_readout_chain.py): does any deflate setting beat
the cosine readout, and is the stop / head path sensible on per-cell-type latent-centred vectors?

Fixed: pseudobulk input (fig5_predict.py encode), trained bank, condition x cell type (and donor x cell type) groups.
Latent: ct_mean (the winner: cell type's unweighted mean over donor pseudobulks subtracted) and none (uncorrected).
Readout variants (expert.utils.readout primitives, the same ones predict() chains):
    cosine                          round 1 only
    d_h{head}_r{rounds}_{mode}_{merge}_{stop}
        head    slots held by round 1 (0, 5, 10 = predict() default, 25, 50)
        rounds  2, 3 (default), 5
        mode    subspace (default) | soft | atom
        merge   pm2 (default) | rrf | max
        stop    n     the depth-matched noise stop at the group's nucleus count (predict() default)
                neff  the same at the contrast's effective depth
                off   every round used
    round2      round 2 alone (what round 1 left unexplained)
Scored exactly like the chain (top-200 per group, abs_top and z within cell type; targets, shared with Normal, donor
replication) via fig5_readout_chain.stage_summarize.

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_deflate_grid.py --stage run|summarize ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_readout_chain as ch

VARIANTS = [dict(name='cosine')]
for h in (0, 5, 10, 25, 50):
    VARIANTS.append(dict(head=h, rounds=3, mode='subspace', merge='pm2', stop='n'))
for r in (2, 5):
    VARIANTS.append(dict(head=10, rounds=r, mode='subspace', merge='pm2', stop='n'))
for m in ('soft', 'atom'):
    VARIANTS.append(dict(head=10, rounds=3, mode=m, merge='pm2', stop='n'))
for mg in ('rrf', 'max'):
    VARIANTS.append(dict(head=0, rounds=3, mode='subspace', merge=mg, stop='n'))
for st in ('neff', 'off'):
    for h in (0, 10):
        VARIANTS.append(dict(head=h, rounds=3, mode='subspace', merge='pm2', stop=st))
VARIANTS.append(dict(name='round2', rounds=3, mode='subspace'))
for v in VARIANTS:
    if 'name' not in v:
        v['name'] = f"d_h{v['head']}_r{v['rounds']}_{v['mode']}_{v['merge']}_{v['stop']}"


def readouts(Q, P, n, neff, noise):
    """{variant name: transplanted score matrix}, rounds used per variant."""
    import expert.utils.readout as ro
    Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    Pn = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)
    S1 = Qn @ Pn.T
    Vs = ro.span_basis(P, 256)
    eff = lambda R: np.linalg.norm(R @ Vs.T, axis=1)
    rr_by_mode = {}
    out, used_log = {}, {}
    for v in VARIANTS:
        if v['name'] == 'cosine':
            out['cosine'] = S1
            continue
        mode = v['mode']
        if mode not in rr_by_mode:
            rr_by_mode[mode] = ro.deflate_rounds(Q, P, rounds=5, mode=mode, k=200, effect=eff)
        rr = rr_by_mode[mode]
        if v['name'] == 'round2':
            out['round2'] = ro.transplant_scores(S1, rr['top_loss'][:, 1], rr['top_gain'][:, 1])
            used_log['round2'] = np.full(len(Q), 2)
            continue
        R = v['rounds']
        rrR = {k: (x[:, :R] if isinstance(x, np.ndarray) and x.ndim >= 2 and x.shape[1] >= 5 else x)
               for k, x in rr.items()}
        E = rr['effect']
        if v['stop'] == 'off':
            used = np.full(len(Q), R)
        else:
            depth = n if v['stop'] == 'n' else neff
            thr = ro.noise_level(np.asarray(depth, np.float64), noise[2], noise[3])
            used = 1 + np.cumprod(E[:, 1:R] > thr[:, None], axis=1).sum(1)
        loss = ro.merge_rounds(rrR, used, 'loss', v['merge'], 200, head=v['head'] if v['merge'] == 'pm2' else 0)
        gain = ro.merge_rounds(rrR, used, 'gain', v['merge'], 200, head=v['head'] if v['merge'] == 'pm2' else 0)
        out[v['name']] = ro.transplant_scores(S1, loss, gain)
        used_log[v['name']] = used
    return out, used_log


def stage_run(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = [str(l) for l in b['labels'][tr]]
    P = b['P'][tr].astype(np.float64)
    N = len(labels)
    sets = ch.gene_sets(a, labels)
    names = list(sets)
    pos = {g: i for i, g in enumerate(labels)}
    M = np.zeros((len(names), N), np.float32)
    for i, k in enumerate(names):
        M[i, [pos[g] for g in sets[k] if g in pos]] = 1
    nset = M.sum(1)
    e = np.load(a.predict_npz, allow_pickle=True)
    obs = pd.read_csv(Path(a.predict_npz).parent / 'obs_groups.csv')
    obs = obs.assign(w=obs.n_cells.astype(float), condition=obs.donor.map(ch.cond_of))
    noise = e['noise'].astype(np.float64)
    recs, diag, rounds = [], [], []
    for lat in ('ct_mean', 'none'):
        Z = e['Z'].astype(np.float64).copy()
        inv_n = {}
        for c, ix in obs.groupby('celltype').indices.items():
            if lat == 'ct_mean':
                Z[ix] -= Z[ix].mean(0)
            inv_n[c] = (1.0 / obs.n_cells.to_numpy(float)[ix]).sum() / len(ix) ** 2   # var factor of the subtracted mean
        for level in ('condition', 'donor'):
            G, n = ch.group_latents(Z, obs, level, 30)
            keys = sorted(G)
            Q = np.stack([G[k] for k in keys])
            nn = np.array([n[k] for k in keys])
            neff = np.array([1.0 / (1.0 / n[k] + (inv_n[k.split('|')[1]] if lat == 'ct_mean' else 0.0)) for k in keys])
            S, used = readouts(Q, P, nn, neff, noise)
            for name, u in used.items():
                rounds += [dict(latent=lat, level=level, readout=name, group=k, rounds=int(x)) for k, x in zip(keys, u)]
            for rd, Sm in S.items():
                for c in ch.CT:
                    nk = [k for k in keys if k.endswith('|' + c) and ch.cond_of(k.split('|')[0]) == 'Normal']
                    for k in keys:
                        if k.endswith('|' + c) and nk and k not in nk:
                            i, j = keys.index(k), keys.index(nk[0])
                            diag.append(dict(arm='pb_ct_normal', bank='trained', latent=lat, level=level, readout=rd,
                                             group=k, r_vs_normal=float(np.corrcoef(Sm[i], Sm[j])[0, 1]),
                                             top200_jaccard=len(set(np.argsort(-Sm[i])[:200])
                                                                & set(np.argsort(-Sm[j])[:200])) / 200))
                for sch in ('abs_top', 'unionz_ct'):
                    for k, (ll, gl) in ch.lists_for(Sm, keys, sch, a.top).items():
                        for tail, lst in (('loss', ll), ('gain', gl)):
                            if not lst:
                                continue
                            v = np.zeros(N, np.float32)
                            v[list(lst)] = 1
                            h = M @ v
                            recs.append(pd.DataFrame(dict(arm='pb_ct_normal', bank='trained', latent=lat, level=level,
                                                          readout=rd, scheme=sch, group=k, tail=tail, set=names,
                                                          K=len(lst), hits=h.astype(np.int16),
                                                          expected=(nset * len(lst) / N).astype(np.float32),
                                                          p=hypergeom.sf(h - 1, N, nset, len(lst)))))
    df = pd.concat(recs, ignore_index=True)
    for c in ('arm', 'bank', 'latent', 'level', 'readout', 'scheme', 'group', 'tail', 'set'):
        df[c] = df[c].astype('category')
    df.to_parquet(out / 'tests__deflgrid.parquet', index=False)
    pd.DataFrame(diag).to_csv(out / 'diag__deflgrid.csv', index=False)
    R = pd.DataFrame(rounds)
    R.to_csv(out / 'rounds_used.csv', index=False)
    print(R.groupby(['latent', 'level', 'readout']).rounds.mean().unstack(['latent', 'level']).round(2).to_string())


def stage_summarize(a):
    ch.stage_summarize(a)
    R = pd.read_csv(Path(a.out_dir) / 'chain_scorecard.csv')
    cols = ['latent', 'readout', 'scheme', 'condition_hits', 'condition_opposite', 'condition_shared_with_normal',
            'net', 'donor_replicated', 'export_hep', 'splicing_hep_AH', 'r_vs_normal']
    pd.set_option('display.width', 250, 'display.max_rows', 200)
    print('\n== deflate grid (pb, trained, condition level)')
    print(R[cols].sort_values(['latent', 'scheme', 'net'], ascending=[True, True, False]).round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['run', 'summarize'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--predict-npz')
    ap.add_argument('--bank-npz')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--modules')
    ap.add_argument('--top', type=int, default=200)
    a = ap.parse_args()
    {'run': stage_run, 'summarize': stage_summarize}[a.stage](a)
