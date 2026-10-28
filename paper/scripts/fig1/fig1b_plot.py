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

"""Fig. 1b: the training corpus. Reads the tables written by fig1b_data.py.

Columns are the CONTEXTS the pool rule counts: one per broad dataset key (publication + last
token of the dataset index), so the four Marson donors pool into one context per condition.
Datasets without a dataset index (Zhuang, Wong, Mali) do not count toward the cutoff; they are
shown as a separate block because their cells are in the training DB.

    header     stacked bar of training cells per context, one segment per dataset
    matrix     every perturbation seen in the store (rows) x context; blue = in the training DB
               (shade = log10 cells), grey = measured but excluded by the pool rule
    marginal   perturbations present in >= k counted contexts; the cutoff is marked
    evidence   histogram of n_c x d_c over the selected perturbations; red = prior-pulled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
# SVG text as real <text> objects (editable), not glyph outlines; PDF text as TrueType
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
COND_LABEL = {'Rest': 'Rest', 'Stim8hr': '8 h', 'Stim48hr': '48 h'}
COND_ORDER = ['Rest', 'Stim8hr', 'Stim48hr']


def context_columns(cells: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """dataset -> column. Counted contexts first (cell lines, then T-cell conditions), then the
    datasets that do not vote. Returns one row per DATASET with its column key and label."""
    ds = (cells[['dataset', 'group', 'axis', 'context', 'condition']].drop_duplicates('dataset')
          .set_index('dataset').loc[order].reset_index())
    ds['counted'] = ds.group.notna()
    ds['key'] = ds.group.where(ds.counted, ds.dataset)
    # the two K562 screens (genome-wide / essential) are one context: same line, same lab,
    # 2,031 of the 2,046 essential perturbations are also in genome-wide
    k562 = ds.counted & (ds.context == 'K562')
    ds.loc[k562, 'key'] = 'K562'
    ds['label'] = ds.context
    tc = ds.axis == 'donor'
    ds.loc[tc, 'label'] = ds.loc[tc, 'condition'].map(COND_LABEL)
    # a cell line split over two counted contexts (K562 genome-wide / essential) says which
    dup = ds[ds.counted & ~tc].groupby('context').key.transform('nunique') > 1
    idx = dup[dup].index
    ds.loc[idx, 'label'] = ds.loc[idx, 'context'] + ' ' + ds.loc[idx, 'key'].str.split('_').str[-1] \
        .map({'gwps': 'GW', 'essential': 'ess.'}).fillna(ds.loc[idx, 'key'].str.split('_').str[-1])
    ds['_blk'] = np.select([ds.counted & ~tc, ds.counted & tc], [0, 1], 2)
    ds['_c'] = ds.condition.map({c: i for i, c in enumerate(COND_ORDER)}).fillna(0)
    ds['_o'] = np.arange(len(ds))
    ds = ds.sort_values(['_blk', '_c', '_o'])
    keys = list(dict.fromkeys(ds.key))
    ds['col'] = ds.key.map({k: i for i, k in enumerate(keys)})
    return ds.drop(columns=['_c', '_o'])


def context_palette(n: int) -> list:
    """n categorical colours from scanpy's 102-colour palette, near-greys dropped."""
    from matplotlib.colors import rgb_to_hsv, to_rgb
    from scanpy.plotting.palettes import default_102
    def ok(h):
        r = np.array(to_rgb(h))
        _, sat, val = rgb_to_hsv(r)
        lum = 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2]     # too light = invisible on white
        return sat >= 0.35 and val >= 0.35 and lum <= 0.55
    return [h for h in default_102 if ok(h)][:n]


def hierarchy_inset(ax, colinfo: pd.DataFrame, dsc: pd.DataFrame, n_pert: dict,
                    n_sel: int, centre: list[tuple[str, str]], vs, ccol: dict) -> None:
    """Hierarchy Fig. 2 splits on, plus the corpus headline in the hole.

    inner ring  system (cell lines | primary CD4+ T), curved labels
    outer ring  one slice per DATASET (equal angle; T conditions split into their donors, the
                K562 context into its two screens). Bar depth = raw number of selected
                perturbations in that dataset, over a grey track = all n_sel.
    centre      headline numbers, one per line
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Wedge
    from matplotlib.textpath import TextToPath
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.24, 1.04)
    tc = colinfo.axis.to_numpy() == 'donor'
    leaves = []                                       # (col, dataset), matrix column order
    for i in range(len(colinfo)):
        ds = dsc[dsc.col == i].sort_values('dataset').dataset.tolist()
        if tc[i]:
            ds = sorted(ds, key=lambda d: d.split('_')[1])   # D1..D4
        leaves += [(i, d) for d in ds]
    n = len(leaves)
    step = 360.0 / n
    t_leaf = [j for j, (i, _) in enumerate(leaves) if tc[i]]
    j0 = max(t_leaf) + 1
    order = list(range(j0, n)) + list(range(0, j0))   # cell lines one arc, T one arc
    pos = {j: q for q, j in enumerate(order)}
    tq = [pos[j] for j in t_leaf]
    start = 270 + (np.mean(tq) + 0.5) * step          # T block centred at 6 o'clock
    ang = lambda q: start - q * step
    hue = {True: vs.INK2, False: vs.MUTED}           # systems neutral; contexts carry colour
    r_in, r_sys, r_leaf0, r_leaf1 = 0.64, 0.79, 0.82, 1.0

    def w(r0, r1, q0, q1, col, alpha=1.0, gap=1.0):
        ax.add_patch(Wedge((0, 0), r1, ang(q1) + gap / 2, ang(q0) - gap / 2, width=r1 - r0,
                           facecolor=col, alpha=alpha, edgecolor='none'))

    ax.apply_aspect()
    p_ = ax.get_position()
    dpp = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (p_.width * ax.figure.get_size_inches()[0] * 72)
    t2p = TextToPath()

    def curved(r, mid, text, color, weight='normal'):
        fp = FontProperties(size=vs.FONT_MIN, weight=weight)
        tw = lambda t: t2p.get_text_width_height_descent(t, fp, ismath=False)[0] * dpp
        top = np.sin(np.deg2rad(mid)) >= 0
        span = np.rad2deg(tw(text) / r)
        a0 = mid + span / 2 if top else mid - span / 2
        for c, ch in enumerate(text):
            if ch == ' ':
                continue
            off = np.rad2deg((tw(text[:c]) + tw(ch) / 2) / r)
            th = a0 - off if top else a0 + off
            ax.text(r * np.cos(np.deg2rad(th)), r * np.sin(np.deg2rad(th)), ch, ha='center',
                    va='center', rotation=th - 90 if top else th + 90, rotation_mode='anchor',
                    color=color, fontproperties=fp)

    q_t0, q_t1 = min(tq), max(tq) + 1
    w(r_in, r_sys, q_t0, q_t1, hue[True], gap=2.5)
    w(r_in, r_sys, q_t1, q_t0 + n, hue[False], gap=2.5)
    rm = (r_in + r_sys) / 2
    curved(rm, ang((q_t0 + q_t1) / 2), 'primary CD4⁺ T', vs.SURFACE)
    curved(rm, ang((q_t1 + q_t0 + n) / 2), 'cell lines', vs.SURFACE)
    for j, (i, d) in enumerate(leaves):
        q = pos[j]
        first = j == 0 or leaves[j - 1][0] != i
        g = 0.8 if not first else 3.0
        w(r_leaf0, r_leaf1, q, q + 1, vs.GRID, gap=g)
        f = n_pert.get(d, 0) / n_sel
        w(r_leaf0, r_leaf0 + (r_leaf1 - r_leaf0) * f, q, q + 1, ccol[i], 1.0, gap=g)
    for i in np.flatnonzero(tc):
        qs = [pos[j] for j, (ii, _) in enumerate(leaves) if ii == i]
        curved(1.10, ang(np.mean(qs) + 0.5), colinfo.label.iloc[i], vs.INK)
    # centre: headline, one fact per line, each line centred on its FULL width
    fb = FontProperties(size=vs.FONT_MIN, weight='bold')
    fr = FontProperties(size=vs.FONT_MIN)
    wd = lambda t, f: t2p.get_text_width_height_descent(t, f, ismath=False)[0] * dpp
    lh = vs.FONT_MIN * 1.25 * dpp
    y0 = lh * (len(centre) - 1) / 2
    for k, (big, small) in enumerate(centre):
        big_t = f'{big} ' if (small and big) else big
        x = -(wd(big_t, fb) + (wd(small, fr) if small else 0)) / 2
        ax.text(x, y0 - k * lh, big_t, ha='left', va='center', color=vs.INK, fontproperties=fb)
        if small:
            ax.text(x + wd(big_t, fb), y0 - k * lh, small, ha='left', va='center',
                    color=vs.INK2, fontproperties=fr)


def cluster_columns(M: np.ndarray, colinfo: pd.DataFrame) -> list[int]:
    """Column order from average-linkage clustering of perturbation OCCUPANCY (Jaccard on
    measured-or-selected presence over every row), with optimal leaf ordering.

    The CD4 T-cell conditions are clustered as ONE unit (presence = any of the three) and
    expanded back in Rest / 8 h / 48 h order, so the donor block is never split."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist
    P = ~np.isnan(M)
    tc = list(colinfo.index[colinfo.axis == 'donor'])
    units = [[i] for i in colinfo.index if i not in tc] + [tc]
    X = np.stack([P[:, u].any(1) for u in units])
    D = pdist(X, 'jaccard')
    Z = optimal_leaf_ordering(linkage(D, 'average'), D)
    return [i for j in leaves_list(Z) for i in units[j]]


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'                              # paper figures: plain white background
    plt.rcParams.update({'axes.grid': False, 'figure.facecolor': 'white',
                         'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    plot(Path(a.data_dir), a.out_dir, vs, name=a.name, col_order=a.col_order, row_order=a.row_order)


def plot(d, out_dir, vs, name='fig1c_training_corpus', col_order='cluster', row_order='cluster',
         col_order_out=None):
    """The whole panel from data dir d. col_order_out: where fig1c_col_order.json goes (default d,
    as the sbatch plot step does; the notebook points it at render/src)."""
    a = argparse.Namespace(out_dir=out_dir, name=name, col_order=col_order, row_order=row_order)
    d = Path(d)
    cells = pd.read_parquet(d / 'fig1c_cells.parquet')
    per = pd.read_parquet(d / 'fig1c_perts.parquet').set_index('perturbation')
    meta = json.load(open(d / 'fig1c_meta.json'))
    mcc, min_groups, n_groups = meta['min_cells_per_class'], meta['min_groups'], meta['n_groups']

    dsc = context_columns(cells, meta['dataset_order'])
    colinfo = dsc.drop_duplicates('col').set_index('col').sort_index()
    n_col = len(colinfo)
    n_cnt = int(colinfo.counted.sum())
    cells = cells.merge(dsc[['dataset', 'col', 'counted']].rename(columns={'counted': 'cnt'}),
                        on='dataset')
    # breadth over the MERGED counted contexts. The recipe counted the K562 screens twice, so
    # 71 selected perturbations reach >= min_groups only through that; they stay selected
    # (this is the trained DB) and sit just above the cutoff line.
    # The marginal shows the pool rule AS RUN: per.n_groups is breadth over the recipe's 10 voting
    # groups (the two K562 screens count separately), so the cutoff falls exactly at 60% and the
    # selected set is exactly the rows above it. The matrix columns still merge K562 for display.

    # ---- matrix: (perturbation, context) -----------------------------------------------------
    # measured = >= mcc cells in ANY dataset of the context, the same test the pool rule applies
    cells['meas'] = cells.store_cells >= mcc
    agg = cells.groupby(['perturbation', 'col'], as_index=False).agg(
        db_cells=('db_cells', 'sum'), meas=('meas', 'any'))
    agg = agg[agg.meas | (agg.db_cells > 0)]
    ncol_present = agg.groupby('perturbation').col.nunique()
    per = per.assign(n_cols=ncol_present.reindex(per.index).fillna(0).astype(int))
    per = per[per.n_cols > 0]
    per = per.assign(_sel=per.selected.astype(int)).sort_values(
        ['_sel', 'n_groups', 'n_cols', 'store_cells'], ascending=False)
    ns = meta['n_selected']
    assert per._sel.to_numpy()[:ns].all(), 'selected rows are not contiguous'
    rix = pd.Series(np.arange(len(per)), index=per.index)
    agg = agg[agg.perturbation.isin(rix.index)]
    M = np.full((len(per), n_col), np.nan)
    r, c = rix.loc[agg.perturbation].to_numpy(), agg.col.to_numpy()
    ind = agg.db_cells.to_numpy() > 0
    M[r[ind], c[ind]] = np.log10(agg.db_cells.to_numpy()[ind])
    ex = ~ind
    M[r[ex], c[ex]] = -1.0                           # measured, excluded
    lo_v, vmax = np.log10(mcc), np.nanpercentile(M[M > 0], 99)
    print(f'[colour] cells per perturbation x context: scale {10 ** lo_v:.0f} .. '
          f'{10 ** vmax:,.0f} (99th percentile, saturates above); max {10 ** M[M > 0].max():,.0f}')

    if a.col_order == 'cluster':
        perm = cluster_columns(M, colinfo)
        print('column order:', ', '.join(colinfo.label.iloc[perm]))
        # shared with fig1b_sunburst.py so both panels use one context order
        json.dump(list(colinfo.label.iloc[perm]),
                  open(Path(col_order_out or d) / 'fig1c_col_order.json', 'w'))
        new_of_old = {old: new for new, old in enumerate(perm)}
        M = M[:, perm]
        colinfo = colinfo.iloc[perm].reset_index(drop=True)
        colinfo.index.name = 'col'
        dsc['col'] = dsc.col.map(new_of_old)

    if a.row_order == 'cluster':
        # WITHIN each block, so the selection cutoff stays one horizontal line. Selected rows
        # cluster on log10 training cells per context; excluded rows have no training cells,
        # so they cluster on where they were measured.
        from scipy.cluster.hierarchy import leaves_list, linkage
        X_sel = np.where(M[:ns] > 0, M[:ns], 0.0)
        X_exc = (M[ns:] < 0).astype(float)
        perm_r = np.concatenate([leaves_list(linkage(X_sel, 'ward')),
                                 ns + leaves_list(linkage(X_exc, 'ward'))])
        M = M[perm_r]
        per = per.iloc[perm_r]

    # ---- figure ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 11.2))
    # row 0 (left) holds the matrix colour key; on the right, rows 0-2 hold the hierarchy inset
    gs = GridSpec(5, 2, figure=fig, width_ratios=[4.6, 2.0], wspace=0.05, top=0.985,
                  height_ratios=[0.75, 0.45, 1.8, 5.4, 1.0], hspace=0.08)
    ax_key = fig.add_subplot(gs[0, 0]); ax_key.axis('off')
    ax_band = fig.add_subplot(gs[1, 0])
    ax_hdr = fig.add_subplot(gs[2, 0])
    ax_m = fig.add_subplot(gs[3, 0])
    ax_mg = fig.add_subplot(gs[3, 1], sharey=ax_m)
    ccol = dict(zip(colinfo.index, context_palette(n_col)))   # one colour per context
    xl = (-0.5, n_col - 0.5)


    # header band: which contexts the cutoff counts; T-cell bracket
    ax_band.set_xlim(*xl); ax_band.set_ylim(0, 2); ax_band.axis('off')
    tcols = colinfo.index[colinfo.axis == 'donor']
    x0, x1 = tcols.min() - 0.4, tcols.max() + 0.4
    ax_band.plot([x0, x1], [0.4, 0.4], color=vs.MUTED, lw=1.5, solid_capstyle='butt')
    ax_band.text((x0 + x1) / 2, 0.55, 'CD4⁺ T (4 donors)', ha='center', va='bottom',
                 color=vs.INK2)

    # header: stacked training cells per context, one segment per dataset (single colour; the
    # 2 px surface gap between segments is what separates datasets)
    base = np.zeros(n_col)
    tr = cells.groupby('dataset').db_cells.sum()
    for _, row in dsc.iterrows():
        v = tr.get(row.dataset, 0) / 1e6
        ax_hdr.bar(row.col, v, bottom=base[row.col], width=0.78, color=ccol[row.col],
                   edgecolor=vs.SURFACE, linewidth=1.5)
        base[row.col] += v
    ax_hdr.set_xlim(*xl)
    ax_hdr.set_ylabel('cells (M)')
    ax_hdr.set_xticks([]); ax_hdr.spines['bottom'].set_visible(False)
    ax_hdr.yaxis.grid(True, color=vs.GRID); ax_hdr.set_axisbelow(True)

    # matrix: explicit RGB so absent / excluded / log-cells never fight over a "bad" colour
    cmap = vs.seq_cmap()
    to_rgb = lambda h: np.array(matplotlib.colors.to_rgb(h))
    img = np.tile(to_rgb(vs.SURFACE), (len(per), n_col, 1))
    img[M < 0] = to_rgb(vs.EXCLUDED)
    pos = M > 0
    img[pos] = cmap(np.clip((M[pos] - lo_v) / (vmax - lo_v), 0, 1))[:, :3]
    print(pd.DataFrame({'ctx': colinfo.label, 'sel_in_db': (M[:ns] > 0).mean(0).round(3),
                        'unsel_measured': (M[ns:] < 0).mean(0).round(3)}).to_string())
    # vector, not imshow (a bitmap in the svg): one rectangle per run of identical colour down
    # each column, which keeps thousands of rows to a few thousand shapes
    verts, fcs = [], []
    for j in range(n_col):
        col = img[:, j]
        brk = np.r_[0, np.flatnonzero((col[1:] != col[:-1]).any(1)) + 1, len(col)]
        for s, e in zip(brk[:-1], brk[1:]):
            verts.append([(j - 0.5, s), (j + 0.5, s), (j + 0.5, e), (j - 0.5, e)])
            fcs.append(col[s])
    ax_m.add_collection(PolyCollection(verts, facecolors=fcs, linewidths=0, antialiased=False, zorder=0))
    ax_m.set_xlim(*xl)
    ax_m.set_xticks(np.arange(n_col))
    ax_m.set_xticklabels(colinfo.label, rotation=90)
    ax_m.set_ylabel(f'perturbations ({len(per):,}'
                    + (', clustered)' if a.row_order == 'cluster' else ', sorted by breadth)'))
    ax_m.yaxis.set_major_formatter(plt.FuncFormatter(vs.thousands))
    for s_ in ('top', 'right'):
        ax_m.spines[s_].set_visible(False)
    for x in (tcols.min() - 0.5, tcols.max() + 0.5):
        ax_m.axvline(x, color=vs.SURFACE, lw=3)
    ax_m.axhline(ns, color=vs.INK, lw=1.2, ls=(0, (4, 3)))

    # colour key, in the strip above the header bars
    cax = ax_key.inset_axes([0.10, 0.30, 0.32, 0.20])
    im = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(lo_v, vmax))
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ticks = [t for t in (1, 2, 3, 4) if lo_v <= t <= vmax]
    cb.set_ticks(ticks); cb.set_ticklabels([f'10{str(t).translate(SUP)}' for t in ticks])   # Unicode, not mathtext
    cb.outline.set_visible(False)
    ax_key.text(0.09, 0.40, 'cells', transform=ax_key.transAxes, ha='right', va='center',
                color=vs.INK2)
    ax_key.add_patch(plt.Rectangle((0.52, 0.28), 0.035, 0.24, color=vs.EXCLUDED,
                                   transform=ax_key.transAxes))
    ax_key.text(0.57, 0.40, 'measured, not selected', transform=ax_key.transAxes,
                color=vs.INK2, va='center')

    # hierarchy inset: system -> condition -> donor, the axes Fig. 2 splits on
    # inset fills the top-right block, from the figure top down to the header bars' baseline
    ph, pm = ax_hdr.get_position(), ax_mg.get_position()
    ax_in = fig.add_axes([pm.x0 - 0.02, ph.y0, 0.995 - (pm.x0 - 0.02), 0.995 - ph.y0])
    n_pert = cells[cells.db_cells > 0].groupby('dataset').perturbation.nunique().to_dict()
    centre = [(f"{meta['n_datasets']}", 'datasets'), (f'{n_col}', 'contexts'),
              (f"{meta['n_cells_db'] / 1e6:.1f} M", 'cells'), (f'{ns:,}', ''),
              ('', 'perturbations')]
    hierarchy_inset(ax_in, colinfo, dsc, n_pert, ns, centre, vs, ccol)

    # marginal: perturbations present in >= x% of the pool-rule contexts (shares the row axis)
    n_vote = meta['n_groups']
    kg = np.arange(1, n_vote + 1)
    Ng = [(per.n_groups >= k).sum() for k in kg]
    xs = 100 * kg / n_vote
    ax_mg.step(xs, Ng, where='post', color=vs.C['blue'], lw=2)
    ax_mg.axhline(ns, color=vs.INK, lw=1.2, ls=(0, (4, 3)))
    ax_mg.plot([100 * min_groups / n_vote], [ns], 'o', ms=9, color=vs.C['blue'],
               mec=vs.SURFACE, mew=2)
    assert Ng[min_groups - 1] == ns, 'cutoff does not reproduce the selected set'
    ax_mg.set_xlim(0, 105)
    ax_mg.set_xticks([0, 20, 40, 60, 80, 100])
    ax_mg.set_xlabel('present in ≥ x% of contexts')
    ax_mg.tick_params(labelleft=False)
    ax_mg.xaxis.grid(True, color=vs.GRID); ax_mg.set_axisbelow(True)
    ax_m.set_ylim(len(per), 0)

    vs.save(fig, a.out_dir, a.name)
    print(f'rows {len(per):,} (selected {ns:,}); contexts {n_col} ({n_cnt} counted)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--name', default='fig1c_training_corpus')
    ap.add_argument('--n-label', type=int, default=4)
    ap.add_argument('--col-order', choices=['cluster', 'context'], default='cluster')
    ap.add_argument('--row-order', choices=['cluster', 'breadth'], default='cluster')
    ap.add_argument('--label-low', default='', help='comma list; default = lowest-evidence genes')
    main(ap.parse_args())
