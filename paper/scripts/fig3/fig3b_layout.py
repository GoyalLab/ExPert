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

"""Fig. 3b layouts: trained prototypes and the shipped fills of untrained genes in ONE geometry.

Two candidate views of the same 19,895 rows (trained atoms + non-zero `prior_missing_T` rows),
both fit JOINTLY -- the earlier UMAP fit on trained atoms and .transform()-ed the fills in, which
places out-of-sample points by a different rule than the points they are compared with.

    radial UMAP   radius = EXACT angle to the mean trained-prototype direction (the model scores
                  cosine, so this is the origin-anchored distance on the sphere); azimuth = angle
                  of the point around the centre of a joint cosine UMAP. A fill that is off-
                  manifold shows as a ring; one that is on it shares the trained points' radius
                  band at its neighbours' azimuth.
    circular tree Ward linkage on unit vectors (Euclidean on the sphere is monotone in cosine), leaf
                  order around the circle; the plot draws trained / filled as a ring under the
                  leaves, so fills forming their own clades would show as solid orange arcs.
    placement     (--tree placement) Ward over the TRAINED prototypes only; every zero-shot fill is
                  grafted beside its nearest trained prototype (cosine), most similar first. The ring
                  then shows where zero-shot genes land in the trained space. A joint tree cannot show
                  this: fills are neighbour means (shrunk, lower spread), so every joint linkage
                  tried -- Ward, average, complete, per-kind centred / standardised -- groups them
                  apart (ring switch rate 0.12-0.15 vs 0.32 here, random 0.50)
    mixing        per row, the share of its k cosine nearest neighbours (over all rows) that are
                  fills, against the share expected if kind were ignored by the geometry.

Writes <out>/fig3b_layout.parquet, fig3b_linkage.npz, fig3b_layout_meta.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from expert.analysis.atom_umap import load_atoms
import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels, A, note = load_atoms(a.model_pt)
    labels = [str(x) for x in labels]
    n = len(labels)
    blob = torch.load(a.model_pt, map_location='cpu', weights_only=False)
    sd = blob['model_state_dict']
    k_fill = next(k for k in sd if k.endswith('prior_missing_T'))
    F = sd[k_fill].float().numpy()
    live = np.flatnonzero(np.linalg.norm(F[n:], axis=1) > 0) + n
    ad = blob['attr_dict']
    vocab = next((np.asarray(ad[k], dtype=str) for k in ('idx_to_label', 'vocab_labels_')
                  if k in ad and ad[k] is not None and len(ad[k]) == F.shape[0]), None)
    fill_names = vocab[live] if vocab is not None else np.array([f'row{i}' for i in live])
    if a.zs_cache:
        from fig3_data import zs_rows
        rows, zn = zs_rows(a.zs_cache, labels)
        keep = np.isin(live, rows)
        print(f'[fill] {int(keep.sum())}/{len(live)} filled rows were measured in a screen', flush=True)
        live = live[keep]
        nm = dict(zip(rows, zn)); fill_names = np.array([nm[i] for i in live])
    X = np.concatenate([A, F[live]]).astype(np.float32)
    kind = ['trained'] * n + ['fill'] * len(live)
    names_loo = []
    if a.with_loo:
        # CONTROL: the same map applied to TRAINED genes held out (fig3_data.place). If these land
        # among the fills rather than among the prototypes, fills segregate because they are
        # neighbour means, not because they are misplaced
        from fig3_data import banks_from_ckpt, place
        T, _ = place(torch.from_numpy(A).float(),
                     banks_from_ckpt(sd, a.bank_names.split(','), n), a.graph_k)
        ok = ~torch.isnan(T[:, 0])
        X = np.concatenate([X, T[ok].numpy()]).astype(np.float32)
        kind += ['held-out placement'] * int(ok.sum())
        names_loo = np.array(labels)[ok.numpy()]
    U = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    kind = np.array(kind)
    print(f'[rows] {n} trained + {len(live)} fills + {len(names_loo)} held-out placements '
          f'= {len(U)} {note or ""}', flush=True)

    m = U[:n].mean(0); m /= np.linalg.norm(m)
    radius = np.degrees(np.arccos(np.clip(U @ m, -1, 1)))

    if a.no_umap:                                       # the tree-only layouts skip the slow part
        um = np.full((len(U), 2), np.nan)
    else:
        import umap
        um = umap.UMAP(n_neighbors=a.n_neighbors, min_dist=0.3, metric='cosine',
                       random_state=0).fit_transform(U)
    c = np.median(um, axis=0)
    azim = np.degrees(np.arctan2(um[:, 1] - c[1], um[:, 0] - c[0]))
    print('[umap] done', flush=True)

    # mixing: share of fills among each row's k cosine neighbours (self excluded)
    T = torch.from_numpy(U)
    fill = torch.from_numpy(kind == 'fill')
    kinds = sorted(set(kind)); kcode = torch.from_numpy(np.searchsorted(kinds, kind))
    nb = np.zeros((len(U), len(kinds)), np.float32)       # share of each kind among neighbours
    mix = np.empty(len(U), np.float32)
    for s in range(0, len(U), 4096):
        S = T[s:s + 4096] @ T.T
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -2
        idx = torch.topk(S, a.k, dim=1).indices
        mix[s:s + 4096] = fill[idx].float().mean(1).numpy()
        for j in range(len(kinds)):
            nb[s:s + 4096, j] = (kcode[idx] == j).float().mean(1).numpy()
    p_fill = float((kind == 'fill').mean())
    nbt = pd.DataFrame(nb, columns=kinds).groupby(kind).mean()
    print('[mixing] row = kind of the point, column = share of its neighbours of each kind; '
          f'expected (fully mixed): {dict(zip(kinds, np.round([np.mean(kind == k) for k in kinds], 3)))}\n'
          + nbt.round(3).to_string(), flush=True)
    print(f'[mixing] k={a.k}: fill share among neighbours of trained {mix[kind == "trained"].mean():.3f}, '
          f'of fills {mix[kind == "fill"].mean():.3f}; expected if mixed {p_fill:.3f}', flush=True)

    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    if a.tree == 'placement':
        tr = np.flatnonzero(kind == 'trained'); ot = np.flatnonzero(kind != 'trained')
        Z = linkage(U[tr].astype(np.float64), method='ward')
        anc, sim = np.empty(len(ot), int), np.empty(len(ot), np.float32)
        for s in range(0, len(ot), 4096):
            S = U[ot[s:s + 4096]] @ U[tr].T
            anc[s:s + 4096] = S.argmax(1); sim[s:s + 4096] = S.max(1)
        grp = pd.DataFrame({'row': ot, 'anc': anc, 'sim': sim}).sort_values(['anc', 'sim'],
                                                                             ascending=[True, False])
        by = grp.groupby('anc').row.apply(list).to_dict()
        order = []
        for t in leaves_list(Z):
            order.append(tr[t]); order.extend(by.get(t, []))
        order = np.asarray(order)
        anchor = np.full(len(U), -1); anchor[ot] = tr[anc]
        cl_tr = fcluster(Z, a.n_clusters, criterion='maxclust')
        cl = np.empty(len(U), int); cl[tr] = cl_tr; cl[ot] = cl_tr[anc]
    else:
        Z = linkage(U.astype(np.float64), method='ward')
        order = leaves_list(Z)
        anchor = np.full(len(U), -1)
        cl = fcluster(Z, a.n_clusters, criterion='maxclust')
    pos = np.empty(len(U), int); pos[order] = np.arange(len(U))
    k_ord = kind[order]
    switch = float((k_ord[1:] != k_ord[:-1]).mean())
    print(f'[tree] {a.tree}: ring switch rate {switch:.3f} (random '
          f'{1 - sum(np.mean(kind == k) ** 2 for k in set(kind)):.3f})', flush=True)
    print('[linkage] done', flush=True)

    per = pd.read_parquet(a.perts).set_index('perturbation')
    pw = per.w_hard.reindex(labels).fillna(0).to_numpy() > 0
    pulled = np.concatenate([pw, np.zeros(len(live), bool),
                             pd.Series(pw, index=labels).reindex(names_loo).to_numpy(bool)])
    cat = (pd.read_parquet(a.categories).set_index('perturbation').category
           if a.categories else None)
    names = np.concatenate([labels, fill_names, names_loo]).astype(str)
    df = pd.DataFrame({'perturbation': names, 'kind': kind, 'pulled': pulled,
                       'angle_deg': radius, 'azimuth_deg': azim,
                       'umap1': um[:, 0], 'umap2': um[:, 1],
                       'leaf_pos': pos, 'cluster': cl, 'fill_share_knn': mix,
                       'anchor_row': anchor})
    if cat is not None:
        df['category'] = cat.reindex(names).to_numpy()
    df.to_parquet(out / 'fig3b_layout.parquet', index=False)
    np.savez_compressed(out / 'fig3b_linkage.npz', Z=Z)
    cs = df.groupby('cluster').kind.apply(lambda s: (s == 'fill').mean())
    json.dump({'model_pt': a.model_pt, 'n_trained': n, 'n_fill': int(len(live)),
               'k': a.k, 'p_fill': p_fill, 'with_loo': bool(a.with_loo),
               'tree': a.tree, 'ring_switch_rate': switch,
               'neighbour_share': {k: nbt.loc[k].round(4).to_dict() for k in nbt.index},
               'expected_share': {k: float(np.mean(kind == k)) for k in kinds},
               'mix_trained': float(mix[kind == 'trained'].mean()),
               'mix_fill': float(mix[kind == 'fill'].mean()),
               'n_clusters': a.n_clusters,
               'cluster_fill_share_quantiles': cs.quantile([0, .1, .5, .9, 1]).round(3).tolist(),
               'n_clusters_ge90pct_fill': int((cs >= 0.9).sum()), 'atom_note': note},
              open(out / 'fig3b_layout_meta.json', 'w'), indent=1)
    print(f'[clusters] fill share per cluster q0/10/50/90/100: '
          f'{cs.quantile([0, .1, .5, .9, 1]).round(3).tolist()}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-pt', required=True)
    ap.add_argument('--perts', required=True, help='fig1c_perts.parquet (w_hard)')
    ap.add_argument('--categories', default=None,
                    help='optional parquet with perturbation,category (fig1d_layout.parquet)')
    ap.add_argument('--k', type=int, default=10, help='neighbours for the mixing statistic')
    ap.add_argument('--n-neighbors', type=int, default=30, help='UMAP n_neighbors')
    ap.add_argument('--n-clusters', type=int, default=60, help='tree cut for per-clade fill share')
    ap.add_argument('--with-loo', action='store_true',
                    help='add the held-out placements of trained genes as a third kind (control)')
    ap.add_argument('--bank-names', default='text', help='as fig3_data.py (needed with --with-loo)')
    ap.add_argument('--graph-k', type=int, default=25, help='as fig3_data.py --k')
    ap.add_argument('--zs-cache', default=None,
                    help='zero-shot query cache: keep only fills of perturbations a screen measured')
    ap.add_argument('--tree', default='ward', choices=('ward', 'placement'),
                    help="ward = joint Ward over all rows; placement = Ward over trained, fills grafted "
                         "beside their nearest trained prototype (linkage then covers trained rows only)")
    ap.add_argument('--no-umap', action='store_true', help='skip the UMAP (tree-only layout)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
