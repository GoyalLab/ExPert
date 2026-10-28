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

"""Fig. 3b data: does the prior-graph fill land where the learned prototype would have been?

An untrained perturbation gets its prototype from the shipped f(missing) fill
(`prior_missing_T`): per prior bank, its top-k prior neighbours among the trained atoms,
similarity-weighted mean of their UNIT atoms, banks fused. Whether that position is right cannot
be checked on a gene that was never trained -- so the same map is run on every TRAINED
perturbation as if it were unseen (leave-one-out: the atom is never its own neighbour) and the
placement is compared with the prototype the data actually learned.

    placement   cos(f_loo(c), a_c) and the retrieval rank of a_c among all trained atoms by cosine to
                f_loo(c) (1 = the placement is closer to its own prototype than to any other)
    null        the same map with the bank ROWS permuted among covered trained genes: identical graph
                topology and weights, wrong genes (--n-null seeds)
    oracle      the same neighbour mean over a_c's own top-k ATOM neighbours: the best any k-neighbour
                mean can do, so the prior is read against both ends
    radial      angle to the mean trained-atom direction (the model scores cosine, so this is the
                origin-anchored distance on the sphere) for trained atoms, their leave-one-out
                placements, and the shipped fills of untrained genes. A neighbour MEAN pulls toward
                the centroid, so if the fills' radial offset matches the trained placements', the
                'shell' in a UMAP is the averaging map, not an off-manifold position.

Atoms the prior term PULLED in training (w_hard > 0) were moved toward this very target, so they are
flagged and reported separately; only never-pulled atoms make a clean test.

The neighbour pool here is every trained atom; the shipped fill uses the evidence-selected fit set
(a subset), so this is the same map with a slightly larger pool.

Writes <out>/fig3b_atoms.parquet (one row per trained atom), fig3b_radial.parquet,
fig3b_summary.csv, fig3b_meta.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from expert.analysis.atom_umap import load_atoms
from expert.modules._prior_atoms import PriorAtomPredictor


def unit(M):
    return M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def banks_from_ckpt(sd: dict, names: list[str], n: int):
    """[(name, E[:n], valid[:n])] from the checkpoint's own prior buffers (text first)."""
    key = lambda suf: next((k for k in sd if k.endswith('cls_emb.' + suf)), None)
    out = [(names[0], sd[key('_atom_text')].float()[:n], None)]
    i = 0
    while key(f'_atom_prior_{i}') is not None:
        v = key(f'_atom_prior_valid_{i}')
        out.append((names[i + 1] if i + 1 < len(names) else f'prior{i}',
                    sd[key(f'_atom_prior_{i}')].float()[:n],
                    None if v is None else sd[v].bool()[:n]))
        i += 1
    return out


def place(A, banks, k, seed=0):
    """Leave-one-out graph placement of every trained atom; NaN rows where no bank covers it."""
    rows = torch.arange(A.shape[0])
    f = PriorAtomPredictor(map='graph', graph_k=k, graph_loo=True, cos_floor=-1.0, min_fit=1,
                           seed=seed)
    f.fit(A, banks, rows, rows)
    T = torch.full_like(A, float('nan'))
    T[f.mask] = unit(f.target[f.mask])
    return T, dict(f.chosen)


def shuffle_banks(banks, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for nm, E, valid in banks:
        E = E.clone()
        cov = torch.arange(E.shape[0]) if valid is None else torch.nonzero(valid).flatten()
        E[cov] = E[cov[torch.randperm(len(cov), generator=g)]]
        out.append((nm, E, valid))
    return out


def zs_rows(zs_cache, labels):
    """Vocabulary rows of the zero-shot perturbations that were MEASURED in a screen (the zs query
    cache's truth rows: 9,874 of the 10,931 filled untrained rows; 1,057 no screen measured). Fig. 3
    uses this measured set throughout (19,895 shipped = 8,964 trained + 10,931 untrained).
    Returns (rows, names).

    The cache was built on another model (EFFGATE3); its rows are only reusable if the vocabulary is
    in the same order, so the trained block is checked label by label first."""
    z = np.load(zs_cache, allow_pickle=True)
    rg = np.asarray(z['row_gene']).astype(str)
    n = len(labels)
    if int(z['n_trained']) != n or not np.array_equal(rg[:n], np.asarray(labels, dtype=str)):
        raise SystemExit(f'{zs_cache}: trained block does not match this model\'s labels '
                         f'({int(z["n_trained"])} vs {n}); cannot reuse its row order')
    rows = np.unique(np.asarray(z['truth'], np.int64))
    return rows, rg[rows]


def score(T, U, chunk=2048):
    """cos(T_c, a_c) and the rank of a_c among all atoms by cosine to T_c (NaN where T is NaN)."""
    cos = (T * U).sum(1)
    rank = torch.full((T.shape[0],), float('nan'))
    ok = torch.nonzero(~torch.isnan(cos)).flatten()
    for a in range(0, len(ok), chunk):
        i = ok[a:a + chunk]
        S = T[i] @ U.T
        rank[i] = 1 + (S > cos[i, None]).sum(1).float()
    return cos.numpy(), rank.numpy()


def oracle(U, k, chunk=2048):
    """k-neighbour mean over each atom's own nearest atoms (self excluded)."""
    out = torch.empty_like(U)
    for a in range(0, U.shape[0], chunk):
        S = U[a:a + chunk] @ U.T
        S[torch.arange(S.shape[0]), torch.arange(a, a + S.shape[0])] = -float('inf')
        v, idx = torch.topk(S, k, dim=1)
        w = v.clamp_min(0)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        out[a:a + chunk] = unit(torch.einsum('bk,bkd->bd', w, U[idx]))
    return out


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    labels, A_np, note = load_atoms(a.model_pt)
    labels = [str(x) for x in labels]
    n = len(labels)
    A = torch.from_numpy(A_np).float()
    U = unit(A)
    blob = torch.load(a.model_pt, map_location='cpu', weights_only=False)
    sd = blob['model_state_dict']
    banks = banks_from_ckpt(sd, a.bank_names.split(','), n)
    print(f'[atoms] {tuple(A.shape)} {note or ""}', flush=True)
    for nm, E, v in banks:
        print(f'[bank] {nm}: {tuple(E.shape)} covered {n if v is None else int(v.sum())}/{n}',
              flush=True)

    T, chosen = place(A, banks, a.k)
    cos, rank = score(T, U)
    print(f'[real] map {chosen}; cos median {np.nanmedian(cos):.3f}, rank median '
          f'{np.nanmedian(rank):.0f}, top10 {np.nanmean(rank <= 10):.3f}', flush=True)
    nulls = []
    for s in range(a.n_null):
        Tn, _ = place(A, shuffle_banks(banks, s), a.k)
        cn, rn = score(Tn, U)
        nulls.append((cn, rn, Tn))
        print(f'[null {s}] cos median {np.nanmedian(cn):.3f}, rank median {np.nanmedian(rn):.0f}, '
              f'top10 {np.nanmean(rn <= 10):.3f}', flush=True)
    To = oracle(U, a.k)
    co, ro = score(To, U)
    print(f'[oracle] cos median {np.nanmedian(co):.3f}, rank median {np.nanmedian(ro):.0f}',
          flush=True)

    per = pd.read_parquet(a.perts).set_index('perturbation')
    pulled = per.w_hard.reindex(labels).fillna(0).to_numpy() > 0
    df = pd.DataFrame({'perturbation': labels, 'pulled': pulled,
                       'n_train': per.n_train.reindex(labels).to_numpy(),
                       'evidence': per.evidence.reindex(labels).to_numpy(),
                       'cos_prior': cos, 'rank_prior': rank,
                       'cos_oracle': co, 'rank_oracle': ro})
    for s, (cn, rn, _) in enumerate(nulls):
        df[f'cos_null{s}'] = cn; df[f'rank_null{s}'] = rn
    df.to_parquet(out / 'fig3b_atoms.parquet', index=False)

    rows = []
    for subset, m in (('all', np.ones(n, bool)), ('never pulled', ~pulled), ('pulled', pulled)):
        for arm, c, r in [('prior', cos, rank), ('oracle', co, ro)] + \
                [(f'null{s}', cn, rn) for s, (cn, rn, _) in enumerate(nulls)]:
            ok = m & ~np.isnan(c)
            rows.append({'subset': subset, 'arm': arm, 'n': int(ok.sum()),
                         'cos_median': float(np.median(c[ok])), 'cos_mean': float(c[ok].mean()),
                         'rank_median': float(np.median(r[ok])),
                         'top1': float((r[ok] <= 1).mean()), 'top10': float((r[ok] <= 10).mean()),
                         'top100': float((r[ok] <= 100).mean())})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / 'fig3b_summary.csv', index=False)
    print(summ.to_string(index=False), flush=True)

    # radial: angle to the mean TRAINED direction, for every kind of point
    mdir = unit(U.mean(0, keepdim=True))[0]
    ang = lambda M: np.degrees(np.arccos(np.clip((unit(M) @ mdir).numpy(), -1, 1)))
    rad = [pd.DataFrame({'perturbation': labels, 'kind': 'trained prototype', 'angle_deg': ang(U),
                         'pulled': pulled})]
    ok = ~torch.isnan(T[:, 0])
    rad.append(pd.DataFrame({'perturbation': np.array(labels)[ok.numpy()],
                             'kind': 'trained, prior-placed (held out)',
                             'angle_deg': ang(T[ok]), 'pulled': pulled[ok.numpy()]}))
    Tn0 = nulls[0][2] if nulls else None
    if Tn0 is not None:
        okn = ~torch.isnan(Tn0[:, 0])
        rad.append(pd.DataFrame({'perturbation': np.array(labels)[okn.numpy()],
                                 'kind': 'trained, shuffled-prior placed',
                                 'angle_deg': ang(Tn0[okn]), 'pulled': pulled[okn.numpy()]}))
    k_fill = next((k for k in sd if k.endswith('prior_missing_T')), None)
    n_fill = 0
    if k_fill is not None:
        F = sd[k_fill].float()
        ad = blob['attr_dict']
        vocab = next((np.asarray(ad[k], dtype=str) for k in ('idx_to_label', 'vocab_labels_')
                      if k in ad and ad[k] is not None and len(ad[k]) == F.shape[0]), np.array([]))
        live = torch.nonzero(F[n:].norm(dim=1) > 0).flatten() + n
        names = (vocab[live.numpy()] if len(vocab) == F.shape[0]
                 else np.array([f'row{i}' for i in live.numpy()]))
        if a.zs_cache:
            rows, zn = zs_rows(a.zs_cache, labels)
            keep = np.isin(live.numpy(), rows)
            print(f'[fill] {int(keep.sum())}/{len(live)} filled rows were measured in a screen '
                  f'(zs cache: {len(rows)})', flush=True)
            live = live[torch.from_numpy(keep)]
            names = dict(zip(rows, zn)); names = np.array([names[i] for i in live.numpy()])
        n_fill = len(live)
        rad.append(pd.DataFrame({'perturbation': names, 'kind': 'untrained, shipped fill',
                                 'angle_deg': ang(F[live]), 'pulled': False}))
        print(f'[fill] {k_fill}: {n_fill} filled untrained rows', flush=True)
    else:
        print('[fill] no prior_missing_T in this checkpoint: radial plot has no untrained rows',
              flush=True)
    rad = pd.concat(rad, ignore_index=True)
    rad.to_parquet(out / 'fig3b_radial.parquet', index=False)
    print(rad.groupby('kind').angle_deg.describe().to_string(), flush=True)

    json.dump({'model_pt': a.model_pt, 'k': a.k, 'n_atoms': n, 'n_null': a.n_null,
               'banks': [b[0] for b in banks], 'map': chosen, 'n_pulled': int(pulled.sum()),
               'n_fill': n_fill, 'atom_note': note},
              open(out / 'fig3b_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-pt', required=True)
    ap.add_argument('--perts', required=True, help='fig1c_perts.parquet (w_hard, n_train, evidence)')
    ap.add_argument('--bank-names', required=True,
                    help='comma list: the text bank, then class_emb_extra_paths in order')
    ap.add_argument('--k', type=int, required=True, help='prior_atom_graph_k of the model')
    ap.add_argument('--n-null', type=int, default=3)
    ap.add_argument('--zs-cache', default=None,
                    help='zero-shot query cache: keep only fills of perturbations a screen measured')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
