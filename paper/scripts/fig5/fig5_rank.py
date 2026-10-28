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

"""Fig. 5b (simple): where does mRNA export rank among ALL pathways in one cell type's predictions?

Per condition (AC, AH): the union of its donors' top-k predicted perturbations (each donor vs the Normal donors of
the cell type; fig5_predict.py profiles, key --key), tested for over-representation of every Reactome pathway
(10-500 members in the bank), the readout modules and the grouped mRNA-export set; dysregulation = the smaller
p of the loss / gain tails. Pathways are ranked by that p; the grouped export set and the highlight rows are marked.

    python paper/scripts/fig5/fig5_rank.py --stage data|plot ... --src-dir ... (plot)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]


SPLICE_REACTOME = ['mRNA Splicing - Major Pathway', 'mRNA Splicing - Minor Pathway']


def gene_sets(a, L):
    """name -> genes: every Reactome pathway (10-500 in the bank), the readout modules (no ctrl_ panels), the grouped
    mRNA-export set and 'mRNA splicing' = Reactome major + minor spliceosome pathways + every SR protein (SRSF*)."""
    from fig5_data import read_gmt, export_parts
    sets = {f'Reactome: {k}': v for k, v in read_gmt(a.reactome_gmt, L).items()}
    mods = {}
    for line in open(a.modules):
        if line.startswith('#') or '\t' not in line:
            continue
        m, p = line.rstrip('\n').split('\t')[:2]
        mods.setdefault(m, []).append(p)
    for m, pats in mods.items():
        if m.startswith('ctrl_'):
            continue
        g = sorted({x for p in pats for x in L if (x.startswith(p[:-1]) if p.endswith('*') else x == p)})
        if len(g) >= 5:
            sets[f'module: {m}'] = g
    allr = read_gmt(a.reactome_gmt, L, lo=1, hi=10_000)
    sets['mRNA export (grouped)'] = export_parts(L, a.reactome_gmt, a.modules)['mRNA export (grouped)']
    sets['mRNA splicing'] = sorted(set().union(*[set(allr.get(n, [])) for n in SPLICE_REACTOME])
                                   | {g for g in L if g.startswith('SRSF')})
    return sets


def stage_data(a):
    z = np.load(a.profiles_npz, allow_pickle=True)
    L = [str(l) for l in z['labels']]
    pos = {g: i for i, g in enumerate(L)}
    sets = gene_sets(a, L)
    S = z[a.key]
    gk = [tuple(k.split('|')) for k in z[f'keys__{a.key}']]
    rows = []
    groups = (['AC', 'AH', 'Nor'] if a.level == 'condition'
              else sorted({d for d, t in gk if t == a.celltype}, key=lambda d: (d[:2] == 'No', d)))
    for c in groups:
        ix = [i for i, (d, t) in enumerate(gk) if t == a.celltype and (d.startswith(c) if a.level == 'condition'
                                                                         else d == c)]
        for tail, sg in (('loss', 1), ('gain', -1)):
            u = np.unique(np.concatenate([np.argsort(-sg * S[i], kind='stable')[:a.top] for i in ix]))
            us = set(u.tolist())
            for name, genes in sets.items():
                m = [pos[g] for g in genes if g in pos]
                h = len(us & set(m))
                rows.append(dict(condition=c, tail=tail, set=name, n_set=len(m), K=len(u), hits=h,
                                 expected=len(m) * len(u) / len(L), p=hypergeom.sf(h - 1, len(L), len(m), len(u))))
    t = pd.DataFrame(rows)
    w = t.loc[t.groupby(['condition', 'set']).p.idxmin()].copy()          # the more significant tail
    w['rank'] = w.groupby('condition').p.rank(method='min')
    w['n_ranked'] = w.groupby('condition').p.transform('size')
    w['log2_fe'] = np.log2((w.hits + 0.5) / (w.expected + 0.5))
    w.to_csv(Path(a.out), index=False)
    for c in w.condition.unique():
        x = w[w.condition == c].sort_values('p')
        print(f'\n== {c} {a.celltype}: top 12 of {len(x)}')
        print(x.head(12)[['rank', 'set', 'tail', 'hits', 'expected', 'p']].round(5).to_string(index=False))
        for hl in [a.export_row] + a.highlight:
            y = x[x.set == hl]
            if len(y):
                r = y.iloc[0]
                print(f'   {hl}: rank {int(r["rank"])}/{len(x)} {r["tail"]} {r.hits}/{r.expected:.1f} p={r.p:.2g}')


def stage_plot(a):
    """One row per condition: every pathway's rank (top % among all ranked pathways) as grey ticks; the grouped export
    set and the highlight rows marked and labelled directly (filled = loss tail, open = gain tail)."""
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    VS.apply()
    matplotlib.rcParams['pdf.fonttype'] = 42
    w = pd.read_csv(a.out)
    rows = [(a.export_row, 'mRNA export (grouped)', '#4a3aa7')]
    rows += [(h, s_, c_) for h, s_, c_ in zip(a.highlight, a.highlight_label, [SPLICE_C, '#c98a00'])]
    col_of = lambda g: VS.C['blue'] if g.startswith('AC') else VS.C['orange'] if g.startswith('AH') else '#8a8983'
    name_of = {'Nor': 'healthy\n(Nor1 vs Nor2)', 'Nor1': 'Nor1 vs Nor2', 'Nor2': 'Nor2 vs Nor1'}
    conds = [(g, name_of.get(g, g), col_of(g)) for g in w.condition.unique()]
    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(conds) + 1.2))
    src = []
    for k, (c, lab, col) in enumerate(conds):
        x = w[w.condition == c]
        n = len(x)
        pct = 100 * (x['rank'] - 1) / n
        sig = x.p < 0.05
        ax.scatter(pct[~sig], np.full((~sig).sum(), k), marker='|', s=90, color='#e3e3e3', lw=0.8, zorder=1)
        ax.scatter(pct[sig], np.full(sig.sum(), k), marker='|', s=90, color='#b5b5b5', lw=0.8, zorder=1)
        for j, (key, txt, hc) in enumerate(rows):
            r = x[x.set == key]
            if not len(r):
                continue
            r = r.iloc[0]
            pv = 100 * (r['rank'] - 1) / n
            loss = r['tail'] == 'loss'
            ax.scatter(pv, k, s=110, marker='o', facecolor=hc if loss else 'white', edgecolor=hc, lw=2, zorder=3)
            ax.annotate(f'#{int(r["rank"])}', xy=(pv, k), xytext=(0, 13 if j == 0 else -15), textcoords='offset points',
                        ha='center', va='center', color=hc, fontsize=VS.FONT_MIN, fontweight='bold')
            src.append(dict(condition=c, set=key, rank=int(r['rank']), n_ranked=n, top_pct=pv, tail=r['tail'],
                            hits=r.hits, expected=r.expected, p=r.p))
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([l for _, l, _ in conds])
    for tl, (_, _, col) in zip(ax.get_yticklabels(), conds):
        tl.set_color(col); tl.set_fontweight('bold')
    ax.set_ylim(len(conds) - 0.45, -0.6)
    ax.set_xlim(-2, 100)
    ax.set_xlabel('pathway rank (top %)')
    ax.set_title(f'{a.celltype.lower()} predictions', color=VS.INK, fontweight='bold')
    ax.grid(False)
    for sp_ in ('left', 'right', 'top'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(axis='y', length=0)
    # direct labels in the empty right half, between the AH and healthy rows (no legend box)
    for j, (key, txt, hc) in enumerate(rows):
        ax.text(1.02, 0.95 - 0.09 * j, txt, transform=ax.transAxes, ha='left', va='top', color=hc,
                fontweight='bold')
    ax.text(1.02, 0.95 - 0.09 * len(rows), 'dark ticks: other\npathways, p < 0.05', transform=ax.transAxes,
            ha='left', va='top', color=VS.INK2, fontsize=VS.FONT_MIN)
    ax.text(1.02, 0.95 - 0.09 * (len(rows) + 2), 'filled = loss\nopen = gain', transform=ax.transAxes,
            ha='left', va='top', color=VS.INK2, fontsize=VS.FONT_MIN)
    out = Path(a.fig_dir)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{a.name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{a.name}.svg')
    pd.DataFrame(src).to_csv(Path(a.src_dir) / f'{a.name}_source.csv', index=False)
    print(pd.DataFrame(src).round(4).to_string(index=False))


EXPORT_C, SPLICE_C, BOTH_C = '#4a3aa7', '#1baf7a', '#e0a100'


def stage_barcode(a):
    """Flattened rank plot: each donor's top-k predictions (ranks 1..k, left = strongest) as ticks, coloured by
    membership: mRNA export only / mRNA splicing only / both (e.g. SRSF proteins); everything else light grey."""
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    VS.apply()
    matplotlib.rcParams['pdf.fonttype'] = 42
    z = np.load(a.profiles_npz, allow_pickle=True)
    L = np.asarray(z['labels']).astype(str)
    sets = gene_sets(a, list(L))
    ex, sp_ = set(sets['mRNA export (grouped)']), set(sets['mRNA splicing'])
    S, gk = z[a.key], [tuple(k.split('|')) for k in z[f'keys__{a.key}']]
    donors = sorted({d for d, t in gk if t == a.celltype}, key=lambda d: (d[:2] == 'No', d))
    name_of = {'Nor1': 'Nor1 vs Nor2', 'Nor2': 'Nor2 vs Nor1'}
    col_of = lambda g: VS.C['blue'] if g.startswith('AC') else VS.C['orange'] if g.startswith('AH') else '#8a8983'
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(donors) + 1.4))
    src = []
    for k, d in enumerate(donors):
        i = gk.index((d, a.celltype))
        top = L[np.argsort(-S[i], kind='stable')[:a.top]]
        for r, g in enumerate(top, 1):
            cat = ('both' if g in ex and g in sp_ else 'export' if g in ex else 'splicing' if g in sp_ else 'other')
            col = {'both': BOTH_C, 'export': EXPORT_C, 'splicing': SPLICE_C, 'other': '#e6e6e6'}[cat]
            ax.plot([r, r], [k - 0.36, k + 0.36], color=col, lw=2.4 if cat != 'other' else 1.2,
                    zorder=3 if cat != 'other' else 1, solid_capstyle='butt')
            src.append(dict(donor=d, celltype=a.celltype, rank=r, gene=g, category=cat))
        n_ex = sum(g in ex for g in top)
        n_sp = sum(g in sp_ for g in top)
        ax.text(a.top + 4, k, f'{n_ex}', color=EXPORT_C, va='center', ha='left', fontweight='bold')
        ax.text(a.top + 16, k, f'{n_sp}', color=SPLICE_C, va='center', ha='left', fontweight='bold')
    ax.text(a.top + 4, -0.75, 'export', color=EXPORT_C, ha='left', va='bottom', fontsize=VS.FONT_MIN, rotation=45)
    ax.text(a.top + 16, -0.75, 'splicing', color=SPLICE_C, ha='left', va='bottom', fontsize=VS.FONT_MIN, rotation=45)
    ax.set_yticks(range(len(donors)))
    ax.set_yticklabels([name_of.get(d, d) for d in donors])
    for tl, d in zip(ax.get_yticklabels(), donors):
        tl.set_color(col_of(d)); tl.set_fontweight('bold')
    ax.set_ylim(len(donors) - 0.5, -0.6)
    ax.set_xlim(0, a.top + 1)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlabel('prediction rank')
    ax.set_title(f'{a.celltype.lower()} predictions', color=VS.INK, fontweight='bold')
    ax.grid(False)
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    y0 = len(donors) - 0.5 + 0.95
    for j, (txt, col) in enumerate((('mRNA export', EXPORT_C), ('mRNA splicing', SPLICE_C),
                                    ('both (e.g. SR proteins)', BOTH_C))):
        ax.text(1 + 62 * j, y0, txt, color=col, fontweight='bold', ha='left', va='top', clip_on=False)
    out = Path(a.fig_dir)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{a.name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{a.name}.svg')
    pd.DataFrame(src).to_csv(Path(a.src_dir) / f'{a.name}_source.csv', index=False)
    print(pd.DataFrame(src).query("category != 'other'").groupby(['donor', 'category']).gene
          .apply(lambda x: ' '.join(x)).to_string())


SHORT_LAB = {'FASTK Family Proteins Regulate Processing and Stability of Mitochondrial RNAs':
                 'mitochondrial RNA processing (FASTK)',
             'Transcriptional Regulation by TP53': 'TP53 transcriptional regulation'}
DOT_ROWS = [  # (label, set, block) -- highlighted rows first, then a few broad pathways for context
    ('mRNA export', 'mRNA export (grouped)', 'highlight'),
    ('mRNA splicing', 'mRNA splicing', 'highlight'),
    ('Pol II elongation', 'Reactome: RNA Polymerase II Transcription Elongation', 'context'),
    ('chromatin modification', 'Reactome: Chromatin Modifying Enzymes', 'context'),
    ('cytosolic translation', 'Reactome: Cap-dependent Translation Initiation', 'context'),
    ('mitochondrial translation', 'Reactome: Mitochondrial Translation', 'context'),
    ('SUMOylation', 'Reactome: SUMOylation', 'context'),
]


def stage_dots(a):
    """Per-donor dot plot for one cell type: rows = DOT_ROWS, columns = disease donors + the healthy donor-vs-donor
    comparison; colour = log2 enrichment of the stronger tail in the donor's top-k (loss violet / gain amber), dark
    outline = hypergeometric p < 0.05, size uniform."""
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    VS.apply()
    matplotlib.rcParams['pdf.fonttype'] = 42
    z = np.load(a.profiles_npz, allow_pickle=True)
    L = [str(l) for l in z['labels']]
    pos = {g: i for i, g in enumerate(L)}
    sets = gene_sets(a, L)
    sets['SR proteins'] = sorted(g for g in L if g.startswith('SRSF') or g in ('TRA2A', 'TRA2B'))
    allr = None
    for _, key, _ in DOT_ROWS:                       # broad Reactome sets may exceed the 500-member cap
        if key not in sets and key.startswith('Reactome: '):
            from fig5_data import read_gmt
            allr = allr or read_gmt(a.reactome_gmt, L, lo=1, hi=10_000)
            sets[key] = allr.get(key[len('Reactome: '):], [])
    S, gk = z[a.key], [tuple(k.split('|')) for k in z[f'keys__{a.key}']]
    donors = [d for d in a.donors if (d, a.celltype) in gk]
    # a column is one donor, or (--pool) ONE column whose top lists are the union of the donors' top-k per tail
    groups = {a.pool: donors} if a.pool else {d: [d] for d in donors}
    cols = list(groups)

    def col_tops(c):
        return [set(np.concatenate([np.argsort(-sg * S[gk.index((d, a.celltype))], kind='stable')[:a.top]
                                    for d in groups.get(c, [c])]).tolist()) for sg in (1, -1)]
    dot_rows = list(DOT_ROWS)
    if a.auto_top:
        # highlighted rows + the `auto_top` most enriched other pathways (smallest p over the shown donors, either
        # tail, >= 3 hits); viral / disease-named Reactome terms skipped; a set overlapping a chosen row by >= 50%
        # of the smaller set is redundant
        import fig1d_plot as f1
        bad = tuple(f1.BROAD) + ('hiv', 'hcmv', 'influenza', 'viral', 'virus', 'sars', 'hepatitis')
        msets = {k: {pos[g] for g in v if g in pos} for k, v in sets.items()}
        tops = {d: col_tops(d) for d in cols}
        chosen = [r for r in DOT_ROWS if r[2] == 'highlight']
        score = []
        for k, m in msets.items():
            if any(k == r[1] for r in chosen) or any(b in k.lower() for b in bad) or len(m) < 5:
                continue
            best = min((hypergeom.sf(len(u & m) - 1, len(L), len(m), len(u)), len(u & m)) for d in cols
                       for u in tops[d])
            if best[1] >= 3:
                score.append((best[0], k))
        for _, k in sorted(score):
            if all(len(msets[k] & msets[r[1]]) / max(1, min(len(msets[k]), len(msets[r[1]]))) < 0.5 for r in chosen):
                lab = k.replace('Reactome: ', '').replace('module: ', '').replace('_', ' ')
                lab = SHORT_LAB.get(lab, lab)
                chosen.append((f1.sentence_case(lab) if lab[:1].isupper() else lab, k, 'context'))
            if len(chosen) == sum(r[2] == 'highlight' for r in DOT_ROWS) + a.auto_top:
                break
        dot_rows = chosen
    rows = []
    for d in cols:
        ct = col_tops(d)
        for lab, key, blk in dot_rows:
            m = {pos[g] for g in sets[key] if g in pos}
            best = None
            for tail, u in zip(('loss', 'gain'), ct):
                h = len(u & m)
                e = len(m) * len(u) / len(L)
                p = hypergeom.sf(h - 1, len(L), len(m), len(u))
                if best is None or p < best['p']:
                    best = dict(donor=d, row=lab, set=key, block=blk, tail=tail, hits=h, expected=e, p=p,
                                n_set=len(m), genes=';'.join(sorted(L[j] for j in u & m)))
            rows.append(best)
    t = pd.DataFrame(rows)
    t['log2_fe'] = np.log2((t.hits + 0.5) / (t.expected + 0.5)).clip(lower=0)   # depletion is not the other tail
    t['signed'] = np.where(t['tail'] == 'loss', -t.log2_fe, t.log2_fe)
    print(t[['donor', 'row', 'tail', 'hits', 'expected', 'p', 'genes']].round(4).to_string(index=False))
    LOSS, GAIN = '#4a3aa7', '#c98a00'
    cmap = LinearSegmentedColormap.from_list('lossgain', [LOSS, '#d9d9d9', GAIN])
    clip = 3.0
    labs = [r[0] for r in dot_rows]
    # '+N': other tested pathways (Reactome 10-500 + modules, not shown) beyond healthy donor variation in at least
    # one shown donor: empirical p < .01 against every Nor1-vs-Nor2 / Nor2-vs-Nor1 test (all sets, both tails)
    shown = {key for _, key, _ in dot_rows}
    msets = {k: {pos[g] for g in v if g in pos} for k, v in sets.items()}

    def tail_ps(d):
        tops = col_tops(d)
        return {k: [hypergeom.sf(len(u & m) - 1, len(L), len(m), len(u)) for u in tops] for k, m in msets.items()}
    nor = [d for d in ('Nor1', 'Nor2') if (d, a.celltype) in gk]
    null = np.sort(np.concatenate([np.ravel(list(tail_ps(d).values())) for d in nor]))
    pe = lambda p: (1 + np.searchsorted(null, p, side='right')) / (1 + len(null))
    dis = [d for d in cols if not d.startswith('Nor')]     # the healthy column is the null, never counted
    dps = {d: tail_ps(d) for d in dis}
    n_more = sum(min(pe(min(dps[d][k])) for d in dis) < 0.01 for k in msets if k not in shown)
    print(f'+N: {n_more} other pathways beyond healthy variation (p_emp < .01) in >= 1 shown donor')
    xs = np.array([0.0 if d.startswith('A') else 0.6 for d in cols]) + np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(0.8 * len(cols) + 4.6, 0.5 * (len(labs) + 1) + 1.8))
    for k, d in enumerate(cols):
        x = t[t.donor == d].set_index('row').reindex(labs)
        sig = (x.p < 0.05).to_numpy()
        sc = ax.scatter(np.full(len(labs), xs[k]), np.arange(len(labs)), s=330, c=np.clip(x.signed, -clip, clip),
                        cmap=cmap, norm=TwoSlopeNorm(0, -clip, clip), edgecolor=np.where(sig, VS.INK, 'none'),
                        lw=1.0, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([a.col_labels[k] if a.col_labels else (d if d != 'Nor1' else 'Nor1 vs\nNor2')
                        for k, d in enumerate(cols)])
    col_of = lambda g: (VS.C['blue'] if g.startswith('AC') else VS.C['orange'] if g.startswith('AH')
                        else '#8a8983' if g.startswith('Nor') else VS.INK)
    ax.xaxis.tick_top()
    for tl, d in zip(ax.xaxis.get_majorticklabels(), cols):
        tl.set_color(col_of(d)); tl.set_fontweight('bold')
    ax.set_yticks(range(len(labs) + 1))
    ax.set_yticklabels(labs + [f'+{n_more:,} more pathways'])
    for tl, (_, _, blk) in zip(ax.get_yticklabels(), dot_rows):
        if blk == 'highlight':
            tl.set_fontweight('bold')
    tl = ax.get_yticklabels()[-1]
    tl.set_fontstyle('italic'); tl.set_color(VS.INK2)
    nh = sum(b == 'highlight' for *_, b in dot_rows)
    ax.axhline(nh - 0.5, color=VS.INK2, lw=0.8)
    ax.set_ylim(len(labs) + 0.5, -0.6)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for y in range(len(labs)):
        ax.axhline(y, color=VS.GRID, lw=0.8, zorder=0)
    ax.set_title(f'{a.celltype.lower()} predictions', color=VS.INK, fontweight='bold', pad=34)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04, aspect=25)
    cb.set_ticks([-clip, 0, clip]); cb.set_ticklabels([f'loss\n{clip:g}', '0', f'gain\n{clip:g}'])
    cb.set_label('log2 fold enrichment in the top 200 predictions')
    cb.outline.set_visible(False)
    ax.text(1.03, 0.0, 'outlined:\np < 0.05', transform=ax.transAxes, ha='left', va='bottom', color=VS.INK2,
            fontsize=VS.FONT_MIN)
    out = Path(a.fig_dir)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{a.name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{a.name}.svg')
    t.to_csv(Path(a.src_dir) / f'{a.name}_source.csv', index=False)


def stage_ald(a):
    """ALD vs healthy, one cell type. ALD = union of the disease donors' top-k (each donor vs the Normal donors),
    healthy = union of Nor1-vs-Nor2 and Nor2-vs-Nor1 (symmetric: its loss and gain lists are the same set, so it has
    no direction). Rows: mRNA export, mRNA splicing, then the `n_top` pathways that best separate ALD from healthy
    (ALD p_emp < .01 against every healthy test, healthy p > .05, ranked by -log10 p_ALD + log10 p_healthy;
    redundant sets -- member Jaccard >= .5 with a chosen row -- skipped). Colour = log2 enrichment in ALD's stronger
    tail; the healthy dot shows the SAME tail. Outline = p < 0.05."""
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import fig1d_plot as f1
    VS.apply()
    matplotlib.rcParams['pdf.fonttype'] = 42
    z = np.load(a.profiles_npz, allow_pickle=True)
    L = [str(l) for l in z['labels']]
    pos = {g: i for i, g in enumerate(L)}
    sets = gene_sets(a, L)
    M = {k: {pos[g] for g in v if g in pos} for k, v in sets.items()}
    S, gk = z[a.key], [tuple(k.split('|')) for k in z[f'keys__{a.key}']]

    def ora(ds):
        out = {}
        for tail, sg in (('loss', 1), ('gain', -1)):
            u = set(np.concatenate([np.argsort(-sg * S[gk.index((d, a.celltype))], kind='stable')[:a.top]
                                    for d in ds]).tolist())
            for k, m in M.items():
                h = len(u & m)
                out[(k, tail)] = dict(hits=h, expected=len(m) * len(u) / len(L),
                                      p=hypergeom.sf(h - 1, len(L), len(m), len(u)))
        return out
    A, H = ora(a.ald), ora([d for d in ('Nor1', 'Nor2') if (d, a.celltype) in gk])
    null = np.sort([v['p'] for v in H.values()])
    pe = lambda p: (1 + np.searchsorted(null, p, side='right')) / (1 + len(null))
    best = {k: min(('loss', 'gain'), key=lambda t: A[(k, t)]['p']) for k in M}
    fixed = ['mRNA export (grouped)', 'mRNA splicing']
    cand = []
    for k, t in best.items():
        if k in fixed:
            continue
        pa, ph = A[(k, t)]['p'], H[(k, t)]['p']
        if pe(pa) < 0.01 and ph > 0.05 and A[(k, t)]['hits'] >= 3:
            cand.append((np.log10(ph) - np.log10(pa), k))
    rows = list(fixed)
    for _, k in sorted(cand, reverse=True):
        if all(len(M[k] & M[r]) / max(1, min(len(M[k]), len(M[r]))) < 0.5 for r in rows):   # overlap coefficient
            rows.append(k)
        if len(rows) == len(fixed) + a.n_top:
            break
    n_more = sum(pe(A[(k, best[k])]['p']) < 0.01 for k in M if k not in rows)
    lab = lambda k: ('mRNA export' if k == fixed[0] else 'mRNA splicing' if k == fixed[1]
                     else f1.sentence_case(k.replace('Reactome: ', '').replace('module: ', '').replace('_', ' ')))
    recs = []
    for k in rows:
        t = best[k]
        for col, R in (('ALD', A), ('healthy', H)):
            r = R[(k, t)]
            # ENRICHMENT ONLY: fewer hits than expected (depletion) must not read as the opposite direction
            fe = max(np.log2((r['hits'] + 0.5) / (r['expected'] + 0.5)), 0.0)
            recs.append(dict(row=lab(k), set=k, column=col, tail=t, hits=r['hits'], expected=r['expected'], p=r['p'],
                             p_emp_vs_healthy=pe(r['p']) if col == 'ALD' else np.nan,
                             signed=-fe if t == 'loss' else fe))
    T = pd.DataFrame(recs)
    print(T.round(4).to_string(index=False))
    print(f'+N = {n_more} other pathways with ALD p_emp < .01 vs healthy')
    LOSS, GAIN = '#4a3aa7', '#c98a00'
    cmap = LinearSegmentedColormap.from_list('lossgain', [LOSS, '#d9d9d9', GAIN])
    clip = 3.0
    labs = [lab(k) for k in rows]
    cols = ['ALD', 'healthy']
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * (len(labs) + 1) + 1.8))
    for j, c in enumerate(cols):
        x = T[T.column == c].set_index('row').reindex(labs)
        sig = (x.p < 0.05).to_numpy()
        sc = ax.scatter(np.full(len(labs), j), np.arange(len(labs)), s=330, c=np.clip(x.signed, -clip, clip),
                        cmap=cmap, norm=TwoSlopeNorm(0, -clip, clip), edgecolor=np.where(sig, VS.INK, 'none'),
                        lw=1.0, zorder=3)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    for tl, c in zip(ax.xaxis.get_majorticklabels(), (VS.INK, '#8a8983')):
        tl.set_color(c); tl.set_fontweight('bold')
    ax.set_yticks(range(len(labs) + 1))
    ax.set_yticklabels(labs + [f'+{n_more:,} more pathways'])
    yl = ax.get_yticklabels()
    for tl in yl[:2]:
        tl.set_fontweight('bold')
    yl[-1].set_fontstyle('italic'); yl[-1].set_color(VS.INK2)
    ax.axhline(1.5, color=VS.INK2, lw=0.8)
    ax.set_ylim(len(labs) + 0.5, -0.6)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for y in range(len(labs)):
        ax.axhline(y, color=VS.GRID, lw=0.8, zorder=0)
    ax.set_title(f'{a.celltype.lower()} predictions', color=VS.INK, fontweight='bold', pad=34)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04, aspect=25)
    cb.set_ticks([-clip, 0, clip]); cb.set_ticklabels([f'loss\n{clip:g}', '0', f'gain\n{clip:g}'])
    cb.set_label('log2 fold enrichment in the top 200 predictions')
    cb.outline.set_visible(False)
    ax.text(1.04, 0.0, 'outlined:\np < 0.05', transform=ax.transAxes, ha='left', va='bottom', color=VS.INK2,
            fontsize=VS.FONT_MIN)
    out = Path(a.fig_dir)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{a.name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{a.name}.svg')
    T.to_csv(Path(a.src_dir) / f'{a.name}_source.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['data', 'plot', 'barcode', 'dots', 'ald'])
    ap.add_argument('--profiles-npz')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--modules')
    ap.add_argument('--key', default='group_deflate__ctrl', help='profile key: each donor vs the Normal donors')
    ap.add_argument('--celltype', default='Hepatocyte')
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--out', required=True, help='ranked table (csv)')
    ap.add_argument('--export-row', default='mRNA export (grouped)')
    ap.add_argument('--level', default='condition', choices=['condition', 'donor'])
    ap.add_argument('--ald', nargs='+', default=['AC1', 'AC2', 'AH1', 'AH2'], help='ald: disease donors pooled')
    ap.add_argument('--n-top', type=int, default=5, help='ald: contrast pathways shown below export / splicing')
    ap.add_argument('--pool', default='', help="dots: ONE column with this label = union of --donors' top-k")
    ap.add_argument('--auto-top', type=int, default=0, help='dots: replace the context rows by the N most enriched')
    ap.add_argument('--col-labels', nargs='*', default=None, help='dots: column labels (default: donor names)')
    ap.add_argument('--donors', nargs='+', default=['AC1', 'AC2', 'AH1', 'AH2', 'Nor1'], help='dots: columns')
    ap.add_argument('--highlight', nargs='*', default=['mRNA splicing'])
    ap.add_argument('--highlight-label', nargs='*', default=['mRNA splicing'])
    ap.add_argument('--fig-dir')
    ap.add_argument('--src-dir',
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root')
    ap.add_argument('--name', default='fig5b_pathway_rank')
    a = ap.parse_args()
    {'data': stage_data, 'plot': stage_plot, 'barcode': stage_barcode, 'dots': stage_dots, 'ald': stage_ald}[a.stage](a)
