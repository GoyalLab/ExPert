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

"""Table S10: every non-redundant pathway enriched (nominal p < 0.05) in ALD hepatocytes -- (A) among the top-200 ExPert
predictions (the rows of Fig. 5E plus its "+n more"), (B) among the top-200 down / up DE genes (Fig. 5F). Rows are
selected with fig5_final's own code (top_rows with the same hit-overlap collapse, viral / disease terms dropped), so
they are exactly the sets behind the figure; checked against Fig. 5E / 5F's source tables. BH q is over every gene set
x tail tested in the ALD column. (C) the curated gene sets: every set tested that is not a plain Reactome 2024 pathway
(the grouped mRNA-export set and its parts, mRNA splicing, the readout modules) and the literature rows of Fig. 5E with
the pathway each maps to, with definition, reference and member genes in the bank. Written to --out-dir (si/tables) as
Table_S10.xlsx plus Table_S10A.csv / Table_S10B.csv / Table_S10C.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_final as F

COLUMN = 'ALD'


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    out = np.empty_like(q)
    out[o] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1)
    return out


def part(T, mem, hits, top, label_of, check, healthy=False):
    """All non-redundant sets of `mem` in the ALD column (p < 0.05), with their stats, hits and folded sets."""
    fo = {}
    rows, tails, _ = F.top_rows(T, COLUMN, mem, n=10 ** 6, n_more_p=0.05, highlight=label_of is not None,
                                hits=hits, folded=fo, order='p')
    A = T[(T.column == COLUMN) & T.set.isin(mem)].copy()
    A['q'] = bh(A.p.values)
    best = A.loc[A.groupby('set').p.idxmin()].set_index('set')
    keep = [(lab, k) for lab, k, _ in rows if best.loc[k, 'p'] < 0.05]
    shown = set(check[check.column == COLUMN].set)
    assert shown <= {k for _, k in keep}, f'figure rows missing: {shown - {k for _, k in keep}}'
    H = T[T.column == 'healthy'].set_index(['set', 'tail']) if healthy else None
    out = []
    for lab, k in keep:
        b = best.loc[k]
        folded = sorted(s for s, into in fo.items() if into == k)
        r = {'pathway': lab, 'gene_set': k.replace('Reactome: ', ''),
             'source': 'Reactome 2024' if k.startswith('Reactome: ') else 'curated',
             'direction': b['tail'], 'set_size': len(mem[k]), 'hits': int(b.hits), 'expected': round(float(b.expected), 3),
             'fold_enrichment': round(float(b.hits / b.expected), 3) if b.expected > 0 else np.nan,
             'log2_fold_enrichment_plotted': round(float(np.log2((b.hits + 0.5) / (b.expected + 0.5))), 3),
             'p': float(b.p), 'q_BH': float(b.q), 'shown_in_figure': k in shown}
        if healthy:
            h = H.loc[(k, b['tail'])]
            r.update({'healthy_hits': int(h.hits), 'healthy_p': float(h.p)})
        r.update({'hit_genes': ';'.join(sorted(hits[k])), 'n_folded_sets': len(folded),
                  'folded_sets': ';'.join(s.replace('Reactome: ', '') for s in folded)})
        out.append(r)
    t = pd.DataFrame(out).sort_values('p').reset_index(drop=True)
    t.insert(0, 'rank', np.arange(1, len(t) + 1))
    return t


def curated(gs, labels, a):
    """C: the non-Reactome gene sets (and the literature rows), with how each is defined."""
    import fig5_data as D
    import fig5_rank as R
    from fig5_centring_audit import NPC
    pats = {}
    for line in open(a.modules):
        if line.startswith('#') or '\t' not in line:
            continue
        m, pat = line.rstrip('\n').split('\t')[:2]
        pats.setdefault(m, []).append(pat)
    parts = D.export_parts(labels, a.reactome_gmt, a.modules)
    defs = {'nuclear pore': f'nucleoporins ({len(NPC)} listed) and readout module nuclear_pore ({", ".join(pats["nuclear_pore"])})',
            'TREX / TREX-2 / adaptors': f'readout module rna_export_trex ({", ".join(pats["rna_export_trex"])})',
            'export receptors / Ran': ', '.join(D.EXPORT_RECEPTORS),
            'Reactome mature-mRNA transport': 'Reactome 2024: ' + '; '.join(D.EXPORT_REACTOME)}
    rows = [dict(gene_set='mRNA export (grouped)', shown_as='mRNA export', source='curated grouping',
                 definition='union of the four parts below (NUPR1/NUPR2, not nucleoporins, excluded)', reference='',
                 genes=parts['mRNA export (grouped)'])]
    rows += [dict(gene_set=f'mRNA export part: {k}', shown_as='', source='curated grouping', definition=v, reference='',
                  genes=parts[k]) for k, v in defs.items()]
    rows.append(dict(gene_set='mRNA splicing', shown_as='mRNA splicing', source='curated grouping',
                     definition='Reactome 2024: ' + '; '.join(R.SPLICE_REACTOME) + '; plus every SR protein (SRSF*)',
                     reference='', genes=gs['mRNA splicing']))
    for k in sorted(x for x in gs if x.startswith('module: ')):
        m = k.split(': ', 1)[1]
        rows.append(dict(gene_set=k, shown_as='', source='readout module (fixed before scoring)',
                         definition='genes matching: ' + ', '.join(pats[m]), reference='', genes=gs[k]))
    for lab, key in F.LIT:
        if f'lit: {lab}' not in gs:
            continue
        src = ('curated grouping' if key is None else 'readout module' if key.startswith('module: ')
               else 'Reactome 2024')
        rows.append(dict(gene_set=f'literature row: {lab}', shown_as=lab, source=src,
                         definition='as mRNA splicing above' if key is None else key.replace('module: ', 'module '),
                         reference=F.REF.get(lab, ''), genes=gs[f'lit: {lab}']))
    t = pd.DataFrame(rows)
    t.insert(4, 'n_genes_in_bank', t.genes.map(len))
    t['genes'] = t.genes.map(lambda g: ';'.join(sorted(g)))
    n_reactome = sum(k.startswith('Reactome: ') for k in gs)
    print(f'[S10C] {len(t)} curated sets / literature rows; {n_reactome} Reactome 2024 pathways tested '
          f'(10-500 genes in the {len(labels):,}-prototype bank) not listed', flush=True)
    return t


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = Path(a.fig5_data)
    T = pd.read_csv(d / 'fig5b_ora.csv.gz')
    TD = pd.read_csv(d / 'fig5b_ora_deg.csv.gz')
    pe = np.load(d / 'fig5_profiles.npz', allow_pickle=True)
    labels = [str(x) for x in pe['labels']]
    gs = F.gene_sets(SimpleNamespace(modules=a.modules, reactome_gmt=a.reactome_gmt), labels)
    hl = dict(F.HIGHLIGHT).values()
    mem_m = {k: set(v) for k, v in gs.items() if k.startswith('Reactome: ') or k in hl}
    mem_d = {k: set(str(v).split(';')) for k, v in TD.drop_duplicates('set').set_index('set').members.items()}
    # the genes that hit each set in its best ALD tail (AC + AH hepatocyte top-200 union), as fig5_final's hit_of
    lab_ = np.asarray(labels)
    Sc = {str(k): pe['cond'][i] for i, k in enumerate(pe['cond_keys'])}
    U_m = {t: set().union(*[set(lab_[np.argsort(-sg * Sc[k], kind='stable')[:a.top]])
                            for k in ('AC|Hepatocyte', 'AH|Hepatocyte')]) for t, sg in (('loss', 1), ('gain', -1))}
    deg = pd.read_csv(a.deg)
    H_ = deg[deg.group.isin(['AC|Hepatocyte', 'AH|Hepatocyte'])]
    U_d = {t: set().union(*[set(H_[H_.group == g].sort_values('score', ascending=asc).gene.head(a.top))
                            for g in ('AC|Hepatocyte', 'AH|Hepatocyte')]) for t, asc in (('loss', True), ('gain', False))}

    def hit_of(mem, U, X):
        bt = X[X.column == COLUMN].loc[lambda z: z.groupby('set').p.idxmin()].set_index('set')['tail']
        return {k: v & U[bt.get(k, 'loss')] for k, v in mem.items()}

    fig_dir = Path(a.fig5_figdata)
    sA = part(T, mem_m, hit_of(mem_m, U_m, T), a.top, True, pd.read_csv(fig_dir / 'fig5e_pathways_source.csv'),
              healthy=True)
    sB = part(TD, mem_d, hit_of(mem_d, U_d, TD), a.top, None, pd.read_csv(fig_dir / 'fig5f_pathways_deg_source.csv'))
    sA['direction'] = sA.direction.map({'loss': 'predicted loss', 'gain': 'predicted gain'})
    sB['direction'] = sB.direction.map({'loss': 'downregulated', 'gain': 'upregulated'})
    for nm, t in (('A', sA), ('B', sB)):
        print(f'[S10{nm}] {len(t)} non-redundant pathways (p < 0.05), {int(t.shown_in_figure.sum())} shown, '
              f'{int((t.q_BH < 0.05).sum())} at BH q < 0.05', flush=True)
    sC = curated(gs, labels, a)
    with pd.ExcelWriter(out / 'Table_S10.xlsx') as w:
        sA.to_excel(w, sheet_name='S10A ExPert pathways', index=False)
        sB.to_excel(w, sheet_name='S10B DE pathways', index=False)
        sC.to_excel(w, sheet_name='S10C curated gene sets', index=False)
    sC.to_csv(out / 'Table_S10C.csv', index=False)
    sA.to_csv(out / 'Table_S10A.csv', index=False)
    sB.to_csv(out / 'Table_S10B.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig5-data', required=True, help='Fig. 5 data/final (fig5b_ora*.csv.gz, fig5_profiles.npz)')
    ap.add_argument('--fig5-figdata', required=True, help="Fig. 5 data/ (fig5e / fig5f pathway source tables)")
    ap.add_argument('--deg', required=True, help='pool-level pseudobulk DE table (readout_deg_pbulk/deg.csv.gz)')
    ap.add_argument('--modules', required=True, help='readout_panels.tsv')
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
