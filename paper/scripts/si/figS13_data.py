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

"""Supplementary Fig. S13 plotting data (zero-shot predictions in alcohol-associated liver disease: ESRP2 and the other
splicing regulators; related to Fig. 5). Fig. 5 scores the 8,964 trained prototypes only; here the same hepatocyte
profiles are scored against the production model's full vocabulary (trained + zero-shot rows, the cos^4 fill).

    profile   exactly fig5_final.profiles: per cell type, the donor x cell type latents (fig5_predict.py encode) minus
              their unweighted mean; per condition, their n_cells-weighted mean; cosine to every unit bank row.
              Per donor, the donor's centred latent. Score > 0 = the state resembles losing the gene.
    bank      --bank (fig5_predict.sbatch MODEL=v0.1.0_cos4 STEP=encode: written from the refit model's own
              prior_missing_T) and, for comparison, --old-bank (the cos fill the earlier full-bank outputs used)

One CSV per panel in --out-dir (paper/figures/si/figS13/data):

    figS13a_ranks.csv        per gene of --genes x condition (healthy / AC / AH hepatocytes) x fill: zero-shot or
                             trained, score, z over the bank, change from healthy in SDs of the healthy scores
                             (z_vs_healthy = (score - healthy score) / SD over all rows of the healthy scores), rank among all rows and among zero-shot rows, whether
                             in the top --top
    figS13aux_donors.csv       per gene x donor (hepatocytes): score and rank, shipped fill
    figS13aux_neighbours.csv   per zero-shot gene of --genes: its fill's --n-nbr nearest trained prototypes (cosine),
                             flagged if they are mRNA splicing / export members (--modules / --reactome-gmt, as Fig. 5)
    figS13aux_ora.csv.gz       Fig. 5E's enrichment (healthy / ALD / AC / AH hepatocytes) on the full bank, top --top-ora
    figS13aux_pathway_rows.csv         its rows, chosen with fig5_final's rule (ALD column, hit-overlap folding)
    figS13b_nodes.csv        per hub (--centres), the --graph-n most enriched of its --graph-pool nearest prototypes, with
    figS13b_edges.csv        their z per condition; edges = each node's --graph-k nearest within the drawn set
    figS13_meta.json         bank sizes, top-k, zero-shot rows in each condition's top-k (both fills)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_final as F  # noqa: E402

CONDS = [('Normal', 'healthy'), ('AC', 'AC'), ('AH', 'AH')]


def centred(predict_dir):
    e = np.load(Path(predict_dir) / 'predict_groups.npz', allow_pickle=True)
    obs = pd.read_csv(Path(predict_dir) / 'obs_groups.csv')
    obs['condition'] = obs.donor.map(F.cond_of)
    Z = e['Z'].astype(np.float64)
    for c in obs.celltype.unique():
        m = np.flatnonzero((obs.celltype == c).to_numpy())
        Z[m] -= Z[m].mean(0)
    return Z, obs


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    Z, obs = centred(a.predict_dir)
    w = obs.n_cells.to_numpy(np.float64)
    hep = obs.celltype == a.celltype
    cond = {}
    for cd, _ in CONDS:
        ix = np.flatnonzero((hep & (obs.condition == cd)).to_numpy())
        cond[cd] = np.average(Z[ix], axis=0, weights=w[ix])
    donors = {d: Z[i] for i, d in zip(np.flatnonzero(hep.to_numpy()), obs[hep].donor)}

    banks = {'cos4': np.load(a.bank, allow_pickle=True), 'cos': np.load(a.old_bank, allow_pickle=True)}
    L = [str(x) for x in banks['cos4']['labels']]
    assert L == [str(x) for x in banks['cos']['labels']], 'the two banks differ in row order'
    zs = ~banks['cos4']['trained'].astype(bool)
    pos = {g: i for i, g in enumerate(L)}
    # A's rows: the named groups (--group NAME=G1,G2,...), then the --n-direct nearest prototypes of --direct-of
    group = {}
    for spec in a.group:
        nm, gl = spec.split('=', 1)
        for g in gl.split(','):
            group.setdefault(g, nm)
    if a.direct_of:
        P0 = F.unit(banks['cos4']['P'].astype(np.float64))
        c = P0 @ P0[pos[a.direct_of]]
        c[pos[a.direct_of]] = -np.inf
        for j in np.argsort(-c)[:a.n_direct]:
            group.setdefault(L[j], f'{a.direct_of} neighbours')
    want = list(dict.fromkeys(list(group) + a.genes))
    genes = [g for g in want if g in pos]
    missing = sorted(set(want) - set(genes))

    rows, meta_top, sd_h = [], {}, {}
    for fill, b in banks.items():
        P = F.unit(b['P'].astype(np.float64))
        s_h = F.unit(cond['Normal']) @ P.T                  # healthy hepatocytes: the reference
        sd_h[fill] = float(s_h.std())
        for cd, cl in CONDS:
            s = F.unit(cond[cd]) @ P.T
            z = (s - s.mean()) / s.std()
            r_all = np.empty(len(s), int); r_all[np.argsort(-s, kind='stable')] = np.arange(1, len(s) + 1)
            r_zs = np.full(len(s), -1); zi = np.flatnonzero(zs)
            r_zs[zi[np.argsort(-s[zi], kind='stable')]] = np.arange(1, len(zi) + 1)
            top = r_all <= a.top
            meta_top[f'{fill}|{cl}'] = dict(zero_shot_in_top=int((top & zs).sum()), top=a.top)
            for g in genes:
                i = pos[g]
                rows.append(dict(gene=g, group=group.get(g, ''), condition=cl, fill=fill, zero_shot=bool(zs[i]), score=float(s[i]),
                                 z=float(z[i]), z_vs_healthy=float((s[i] - s_h[i]) / sd_h[fill]), rank_all=int(r_all[i]), rank_zero_shot=int(r_zs[i]) if zs[i] else None,
                                 in_top=bool(top[i])))
    A = pd.DataFrame(rows)
    A.to_csv(out / 'figS13a_ranks.csv', index=False)

    P = F.unit(banks['cos4']['P'].astype(np.float64))
    drows = []
    for d, v in donors.items():
        s = F.unit(v) @ P.T
        r = np.empty(len(s), int); r[np.argsort(-s, kind='stable')] = np.arange(1, len(s) + 1)
        for g in genes:
            drows.append(dict(gene=g, donor=d, condition=dict(CONDS)[F.cond_of(d)], zero_shot=bool(zs[pos[g]]),
                              score=float(s[pos[g]]), rank_all=int(r[pos[g]])))
    pd.DataFrame(drows).to_csv(out / 'figS13aux_donors.csv', index=False)

    gs = F.gene_sets(argparse.Namespace(modules=a.modules, reactome_gmt=a.reactome_gmt), L)
    spl, exp = set(gs['mRNA splicing']), set(gs['mRNA export (grouped)'])
    tr = np.flatnonzero(~zs)
    nrows = []
    for g in genes:
        if not zs[pos[g]]:
            continue
        c = P[tr] @ P[pos[g]]
        for k, j in enumerate(np.argsort(-c)[:a.n_nbr]):
            nb = L[tr[j]]
            nrows.append(dict(gene=g, rank=k + 1, neighbour=nb, cosine=float(c[j]), splicing=nb in spl, export=nb in exp))
    pd.DataFrame(nrows).to_csv(out / 'figS13aux_neighbours.csv', index=False)

    # ---- D: Fig. 5E on the full bank -- the same ORA, row rule and hit-overlap folding as fig5_final.stage_plot, with
    # the top-k raised to --top-ora because the candidate pool more than doubled (19,895 vs 8,964 rows)
    ct = a.celltype
    S = {f'{cd}|{ct}': F.unit(cond[cd]) @ P.T for cd, _ in CONDS}
    cols = {'healthy': [f'Normal|{ct}'], 'ALD': [f'AC|{ct}', f'AH|{ct}'], 'AC': [f'AC|{ct}'], 'AH': [f'AH|{ct}']}
    T = F.ora_table(S, cols, gs, L, a.top_ora)
    T.to_csv(out / 'figS13aux_ora.csv.gz', index=False)
    mem_m = {k: set(v) for k, v in gs.items() if k.startswith('Reactome: ') or k in dict(F.HIGHLIGHT).values()}
    Larr = np.asarray(L)
    U = {t: set().union(*[set(Larr[np.argsort(-sg * S[k], kind='stable')[:a.top_ora]]) for k in cols['ALD']])
         for t, sg in (('loss', 1), ('gain', -1))}
    bt = T[T.column == 'ALD'].loc[lambda x: x.groupby('set').p.idxmin()].set_index('set')['tail']
    hits = {k: v & U[bt.get(k, 'loss')] for k, v in mem_m.items()}
    d_rows, _, d_more = F.top_rows(T, 'ALD', mem_m, a.n_top, n_more_p=0.05, hits=hits, folded={}, order='fe')
    pd.DataFrame(d_rows, columns=['label', 'set', 'block']).to_csv(out / 'figS13aux_pathway_rows.csv', index=False)
    in5e = sorted(set(pd.read_csv(a.fig5e_source).set)) if a.fig5e_source else []
    # ---- E: prototype neighbourhoods of the centres (top predicted splicing hits), joined into one kNN graph
    zc = {}
    for cd, cl in CONDS:
        v = S[f'{cd}|{ct}']
        zc[cl] = (v - v.mean()) / v.std()
    zc['ALD'] = (zc['AC'] + zc['AH']) / 2
    # each hub's --graph-pool nearest prototypes; of those, the --graph-n most enriched toward predicted loss in
    # --graph-condition relative to healthy hepatocytes
    s_h = S[f'Normal|{ct}']
    sd = float(s_h.std())
    gc = {cl: cd for cd, cl in CONDS}[a.graph_condition]
    zh_g = (S[f'{gc}|{ct}'] - s_h) / sd
    sel = {}
    for cen in a.centres:
        c = P @ P[pos[cen]]
        c[pos[cen]] = -np.inf
        pool = np.argsort(-c)[:a.graph_pool]
        sel.setdefault(cen, set())
        for j in pool[np.argsort(-zh_g[pool], kind='stable')][:a.graph_n]:
            sel.setdefault(L[j], set()).add(cen)
    gnames = list(sel)
    gi = np.array([pos[g] for g in gnames])
    C = P[gi] @ P[gi].T
    np.fill_diagonal(C, -np.inf)
    ed = {tuple(sorted((i, int(j)))) for i in range(len(gi)) for j in np.argsort(-C[i])[:a.graph_k]}
    pd.DataFrame([dict(gene=g, centre=g in a.centres, neighbour_of=';'.join(sorted(sel[g])), zero_shot=bool(zs[pos[g]]),
                       regulator=g in genes, splicing=g in spl, **{f'z_{k}': float(v[pos[g]]) for k, v in zc.items()},
                       **{f'cos_{cl}': float(S[f'{cd}|{ct}'][pos[g]]) for cd, cl in CONDS},
                       **{f'zh_{cl}': float((S[f'{cd}|{ct}'][pos[g]] - s_h[pos[g]]) / sd) for cd, cl in CONDS})
                  for g in gnames]).to_csv(out / 'figS13b_nodes.csv', index=False)
    pd.DataFrame([dict(a=gnames[i], b=gnames[j], cosine=float(C[i, j])) for i, j in sorted(ed)]).to_csv(
        out / 'figS13b_edges.csv', index=False)

    meta = dict(bank=a.bank, old_bank=a.old_bank, n_rows=len(L), n_trained=int((~zs).sum()), n_zero_shot=int(zs.sum()),
                top=a.top, celltype=a.celltype, genes=genes, genes_not_in_bank=missing, top_zero_shot=meta_top,
                donors=sorted(donors), sd_healthy=sd_h, top_ora=a.top_ora, n_top=a.n_top, d_n_more=int(d_more),
                d_rows_also_in_5e=sorted(set(k for _, k, _ in d_rows) & set(in5e)),
                centres=a.centres, graph_n=a.graph_n, graph_k=a.graph_k, graph_condition=a.graph_condition,
                graph_pool=a.graph_pool,
                graph_nodes=len(gnames), graph_shared=sorted(g for g in gnames if len(sel[g]) > 1),
                groups={g: group[g] for g in genes if g in group})
    (out / 'figS13_meta.json').write_text(json.dumps(meta, indent=1))
    pd.set_option('display.width', 220)
    print(A.pivot_table(index=['gene', 'zero_shot'], columns=['fill', 'condition'], values='rank_all').to_string())
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--predict-dir', required=True, help='fig5_predict.py encode output (predict_groups.npz, obs_groups.csv)')
    ap.add_argument('--bank', required=True, help='shipped full bank (cos^4 fill), written by the same encode')
    ap.add_argument('--old-bank', required=True, help='full bank with the earlier cos fill (comparison only)')
    ap.add_argument('--genes', nargs='+', required=True, help='splicing regulators (graph flag, donors, neighbours)')
    ap.add_argument('--group', nargs='+', default=[], help="A's row groups, NAME=G1,G2,... in display order")
    ap.add_argument('--direct-of', default='', help="A: also show this gene's nearest prototypes (full bank)")
    ap.add_argument('--n-direct', type=int, default=4)
    ap.add_argument('--modules', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--celltype', default='Hepatocyte')
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--n-nbr', type=int, default=10)
    ap.add_argument('--top-ora', type=int, default=500, help='D: top-k of the full-bank 5E (5E uses 200 of 8,964)')
    ap.add_argument('--n-top', type=int, default=15, help='D: rows, as fig5_final --n-top')
    ap.add_argument('--fig5e-source', default='',
                    help='D: Fig. 5E source table, to report shared rows '
                         '(paper/figures/fig5/data/fig5e_pathways_source.csv)')
    ap.add_argument('--centres', nargs='+', required=True, help='E: centre prototypes (top predicted splicing hits)')
    ap.add_argument('--graph-n', type=int, default=10, help='B: most enriched (vs healthy) of the pool per hub')
    ap.add_argument('--graph-k', type=int, default=4, help='E: kNN edges per node within the drawn set')
    ap.add_argument('--graph-condition', default='AH', help='E: condition whose z colours the nodes')
    ap.add_argument('--graph-pool', type=int, default=50, help='B: nearest prototypes per hub considered')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
