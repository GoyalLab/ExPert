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

"""Supplementary Fig. S1 panels, one file per panel in --out-dir (si/figS1), rendered ONLY from --data-dir
(paper/figures/si/figS1/data: figS1_data.py, figS1c_response_data.py, figS1d_variance_data.py).
Datasets are ordered and coloured
as in Fig. 1c.

    figS1a_cells_per_perturbation   cells per perturbation (store), total cells and perturbations in / not in the DB
    figS1b_pull_cutoff              training support of the 8,964 selected perturbations; pulled tail in red
    figS1c_response_similarity      same-perturbation pseudobulk response correlation between datasets, over
                                    perturbations reproducible (split-half r >= 0.2) in both (diagonal: split-half)
    figS1d_variance_explained       single-cell variance explained by library depth, perturbation (and dataset)
    figS1e_effect_size              per-cell measured effect (raw signed Mixscale z) and the share of cells scored
    figS1f_gene_selection           perturbation separability (held-out replicate matching) by input-gene selection
                                    strategy and size; solid = a union of per-dataset sets, dashed = not a union
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

MIN_CELLS = 10          # per-class gate of the pool rule (recipe min_cells_per_class)
N_SHIP = 500            # input genes per dataset (shipped)


# Context colours picked from the Fig. 1c bar chart of the assembled Fig1.pdf (one colour per context; the
# four CD4+ T donors share their condition's colour), and the Fig. 1c column order.
CONTEXT_COLOR = {'TeloHAEC': '#dc3668', 'HCT116': '#b26cf7', 'HEK293T': '#55bca3', 'Rest': '#97c364',
                 '8 h': '#669039', '48 h': '#3e5d1e', 'K562': '#0000a6', 'KOLF2.1J': '#bf629c',
                 'hESC': '#882d43', 'HepG2': '#f87893', 'Jurkat': '#6d3c6c', 'RPE1': '#9d5349'}
# (dataset, context, label = context + condition), in Fig. 1c column order
DATASETS = [('Wong2026', 'TeloHAEC', 'TeloHAEC'),
            ('XAtlas2025_HCT116', 'HCT116', 'HCT116'),
            ('XAtlas2025_HEK293T', 'HEK293T', 'HEK293T')] + \
           [(f'Marson2025_D{d}_{c}', lab, f'CD4⁺ T {lab} D{d}')
            for c, lab in (('Rest', 'Rest'), ('Stim8hr', '8 h'), ('Stim48hr', '48 h')) for d in (1, 2, 3, 4)] + \
           [('ReplogleWeissman2022_K562_essential', 'K562', 'K562 essential'),
            ('ReplogleWeissman2022_K562_gwps', 'K562', 'K562 genome-wide'),
            ('Mali2026', 'KOLF2.1J', 'KOLF2.1J'),
            ('Zhuang2026', 'hESC', 'hESC'),
            ('ReplogleWeissman2025_HepG2', 'HepG2', 'HepG2'),
            ('ReplogleWeissman2025_Jurkat', 'Jurkat', 'Jurkat'),
            ('ReplogleWeissman2022_rpe1', 'RPE1', 'RPE1')]
LABEL = {ds: lab for ds, _, lab in DATASETS}


def short(ds: str) -> str:
    return LABEL.get(ds, ds)


def dataset_order(d: Path):
    """Datasets in Fig. 1c column order and their Fig. 1 context colour."""
    return [ds for ds, _, _ in DATASETS], {ds: CONTEXT_COLOR[c] for ds, c, _ in DATASETS}


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.grid': False})


def thousands(x, _):
    return f'{int(x):,}'


# --------------------------------------------------------------------------------------------- A
def panel_a(d, out, vs, order, col):
    pp = pd.read_csv(d / 'figS1a_cells_per_perturbation.csv')
    sm = pd.read_csv(d / 'figS1a_cells_per_perturbation_summary.csv').set_index('dataset').loc[order]
    y = np.arange(len(order))[::-1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 0.42 * len(order) + 1.6), sharey=True,
                                     gridspec_kw={'width_ratios': [2.2, 1, 1], 'wspace': 0.08})
    data = [np.log10(pp.loc[pp.dataset == ds, 'n_cells'].to_numpy()) for ds in order]
    bp = a1.boxplot(data, positions=y, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=vs.INK, lw=1.2), whiskerprops=dict(color=vs.INK2),
                    capprops=dict(color=vs.INK2))
    for b, ds in zip(bp['boxes'], order):
        b.set(facecolor=col[ds], edgecolor=vs.INK2, alpha=0.9)
    a1.axvline(np.log10(MIN_CELLS), ls='--', lw=1, color=vs.MUTED)
    a1.set_xticks([0, 1, 2, 3]); a1.set_xticklabels(['1', '10', '100', '1,000'])
    a1.set_xlim(-0.15, np.log10(5000))
    a1.set_xlabel('cells per perturbation')
    a1.set_yticks(y); a1.set_yticklabels([short(s) for s in order])
    for ax, inc, exc, lab, log in ((a2, 'cells_incl', 'cells_excl', 'perturbed cells', True),
                                   (a3, 'perts_incl', 'perts_excl', 'perturbations', False)):
        # on the log axis bars must START at the axis floor, not 0: log(0) = -inf is clipped on screen but
        # written into the SVG as huge coordinates, which breaks placement in Illustrator
        x0 = 5e4 if log else 0.0
        ax.barh(y, sm[inc] - x0, left=x0, color=[col[s] for s in order], height=0.7)
        ax.barh(y, sm[exc], left=sm[inc], color=vs.GRID, height=0.7)
        ax.set_xlabel(lab)
        if log:
            ax.set_xscale('log'); ax.set_xlim(left=x0)
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    a2.text(0.0, 1.01, 'in training DB', transform=a2.transAxes, ha='left', va='bottom', color=vs.INK2)
    a2.text(1.0, 1.01, 'not in DB', transform=a2.transAxes, ha='right', va='bottom', color=vs.MUTED)
    vs.save(fig, out, 'figS1a_cells_per_perturbation')


# --------------------------------------------------------------------------------------------- B
def panel_b(d, out, vs, order, col, n_label=4):
    f = pd.read_csv(d / 'figS1b_pull_cutoff.csv').sort_values('evidence')
    ev = f.evidence.to_numpy(); pulled = f.pulled.to_numpy()
    red, blue = vs.C['red'], vs.C['blue']
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = np.logspace(np.log10(ev.min()) - 0.02, np.log10(ev.max()) + 0.02, 60)
    ax.hist([ev[pulled], ev[~pulled]], bins=bins, stacked=True, color=[red, blue], edgecolor='white', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('training support per perturbation (training cells × datasets)')
    ax.set_ylabel('perturbations')
    ymax = ax.get_ylim()[1]
    for j, r in enumerate(f.head(n_label).itertuples()):
        ax.annotate(f'{r.perturbation}  ({int(r.n_train)} × {int(r.d_train)})', xy=(r.evidence, ymax * 0.02),
                    xytext=(ev.min(), ymax * (0.93 - 0.13 * j)), ha='left', va='center', fontstyle='italic',
                    color=vs.INK, arrowprops=dict(arrowstyle='-', color=vs.MUTED, lw=0.8))
    thr = ev[pulled].max()
    ax.axvline(thr, ls='--', lw=1, color=vs.MUTED)
    ax.text(thr * 0.8, ymax * 0.45, f'pulled\n{int(pulled.sum()):,} ({pulled.mean():.1%})', ha='right',
            va='center', color=red, fontweight='bold')
    ax.text(9e4, ymax * 0.9, f'not pulled\n{int((~pulled).sum()):,}', ha='left', va='top', color=blue,
            fontweight='bold')
    vs.save(fig, out, 'figS1b_pull_cutoff')


# --------------------------------------------------------------------------------------------- C
def panel_c(d, out, vs, order, col):
    t = pd.read_csv(d / 'figS1c_response_similarity.csv')
    # Marson ordered by DONOR (conditions within a donor), the axis the response similarity follows
    mo = sorted([x for x in order if x.startswith('Marson')], key=lambda x: (x.split('_')[1], order.index(x)))
    i0 = next(i for i, x in enumerate(order) if x.startswith('Marson'))
    rest = [x for x in order if not x.startswith('Marson')]
    order = rest[:i0] + mo + rest[i0:]
    M = t.pivot(index='dataset_i', columns='dataset_j', values='median_r_reproducible') \
        .reindex(index=order, columns=order)
    # symmetric: keep the lower triangle and the diagonal (split-half reliability)
    M = M.where(np.tril(np.ones(M.shape, bool)))
    n = len(order)
    fig, ax = plt.subplots(figsize=(0.42 * n + 3.2, 0.42 * n + 2.2))
    vmax = float(np.nanmax(M.to_numpy()))
    # pcolormesh, not imshow: the SVG must stay vector (vizstyle.assert_vector_svg)
    im = ax.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, M.to_numpy(), cmap='Reds',
                       vmin=0, vmax=vmax, edgecolors='none')
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5); ax.set_aspect('equal')
    for i in range(n):
        for j in range(n):
            v = M.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center',
                        fontsize=vs.FONT_MIN - 7, color='white' if abs(v) > 0.6 * vmax else vs.INK)
    ax.set_xticks(range(n)); ax.set_xticklabels([short(s) for s in order], rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels([short(s) for s in order])
    for tl, ds in zip(ax.get_xticklabels(), order):
        tl.set_color(col[ds])
    for tl, ds in zip(ax.get_yticklabels(), order):
        tl.set_color(col[ds])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('median r')
    for i in range(n):                                  # diagonal = split-half reliability
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=vs.INK, lw=0.8))
    for s_ in ax.spines.values():
        s_.set_visible(False)
    vs.save(fig, out, 'figS1c_response_similarity')


# --------------------------------------------------------------------------------------------- D
def panel_d(d, out, vs, order, col):
    t = pd.read_csv(d / 'figS1d_variance_explained.csv')
    # partition order (sequential), then the residual split into sampling noise and unexplained
    comps = ['library', 'dataset', 'perturbation', 'cell cycle', 'mitochondrial', 'ribosomal',
             'sampling noise', 'unexplained']
    ccol = {'library': vs.MUTED, 'dataset': vs.C['violet'], 'perturbation': vs.C['blue'],
            'cell cycle': vs.C['aqua'], 'mitochondrial': vs.C['yellow'], 'ribosomal': vs.C['magenta'],
            'sampling noise': '#c9c8c3', 'unexplained': '#efeeea'}
    rows = [('dataset', ds, short(ds)) for ds in order] + \
           [('pooled, raw', 'all', 'all datasets, raw'), ('pooled, control-centred', 'all', 'all datasets, centred')]
    W = t.pivot_table(index=['scope', 'dataset'], columns='component', values='fraction').reindex(
        [(s_, ds) for s_, ds, _ in rows]).reindex(columns=comps).fillna(0) * 100
    y = np.arange(len(rows))[::-1].astype(float)
    y[-2:] -= 0.6                                      # gap before the pooled bars
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 3.0))
    left = np.zeros(len(rows))
    for c in comps:
        ax.barh(y, W[c].to_numpy(), left=left, color=ccol[c], height=0.72)
        left += W[c].to_numpy()
    xr = 101.5
    for yy, v in zip(y, W['perturbation'].to_numpy()):                 # one aligned column
        ax.text(xr, yy, f'{v:.1f}%', va='center', color=ccol['perturbation'], fontsize=vs.FONT_MIN - 3)
    # labels: dataset inside its segment of the pooled raw bar (the only row that has it); everything else
    # on the TOP row -- wide segments inside, thin ones with leader lines to a label tier above the plot
    k = [r[0] for r in rows].index('pooled, raw')
    xd = W.iloc[k]['library'] + W.iloc[k]['dataset'] / 2
    ax.text(xd, y[k], 'dataset', ha='center', va='center', color='white', fontweight='bold')
    top = W.iloc[0]
    x0, thin = 0.0, []
    for c in comps:
        w = top[c]
        if c == 'dataset' or w <= 0:
            continue
        if w >= 15:
            ax.text(x0 + w / 2, y[0], c, ha='center', va='center', fontweight='bold', color=vs.INK2,
                    fontsize=vs.FONT_MIN - 2)
        else:
            thin.append((c, x0 + w / 2))
        x0 += w
    ytier = y[0] + 1.5
    for j, (c, xc) in enumerate(thin):
        ax.annotate(c, xy=(xc, y[0] + 0.36), xytext=(8 + 17 * j, ytier), ha='center', va='bottom',
                    color=ccol[c], fontweight='bold', fontsize=vs.FONT_MIN - 2,
                    arrowprops=dict(arrowstyle='-', color=ccol[c], lw=1))
    ax.text(xr, y[0] + 0.8, 'perturbation', ha='left', va='bottom', color=ccol['perturbation'], fontweight='bold')
    ax.set_ylim(y.min() - 0.8, ytier + 1.2)
    ax.set_yticks(y); ax.set_yticklabels([r[2] for r in rows])
    ax.set_xlim(0, 100); ax.set_xlabel('% of single-cell expression variance')
    vs.save(fig, out, 'figS1d_variance_explained')


# --------------------------------------------------------------------------------------------- E
def panel_e(d, out, vs, order, col):
    s = pd.read_csv(d / 'figS1e_effect_size.csv')
    sm = pd.read_csv(d / 'figS1e_effect_size_summary.csv').set_index('dataset').reindex(order)
    y = np.arange(len(order))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6), sharey=True,
                                 gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.08})
    data = [s.loc[s.dataset == ds, 'z'].to_numpy() for ds in order]
    bp = a1.boxplot(data, positions=y, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=vs.INK, lw=1.2), whiskerprops=dict(color=vs.INK2),
                    capprops=dict(color=vs.INK2), whis=(5, 95))
    for b, ds in zip(bp['boxes'], order):
        b.set(facecolor=col[ds], edgecolor=vs.INK2, alpha=0.9)
    a1.set_xscale('symlog', linthresh=1)
    a1.axvline(0, color=vs.MUTED, lw=1)
    a1.set_xlabel('measured per-cell effect (Mixscale z)')
    a1.set_yticks(y); a1.set_yticklabels([short(x) for x in order])
    a2.barh(y, sm.frac_scored.to_numpy() * 100, color=[col[x] for x in order], height=0.7)
    for yy, v in zip(y, sm.frac_scored.to_numpy() * 100):
        a2.text(v + 1, yy, f'{v:.0f}%', va='center', color=vs.INK2, fontsize=vs.FONT_MIN - 3)
    a2.set_xlim(0, 115); a2.set_xticks([0, 50, 100])
    a2.set_xlabel('% perturbed cells scored')
    vs.save(fig, out, 'figS1e_effect_size')


# --------------------------------------------------------------------------------------------- F
def panel_f(d, out, vs, order, col):
    g = pd.read_csv(d / 'figS1f_gene_selection.csv')
    blue, orange_free = vs.C['blue'], vs.C['violet']
    # solid = a union of per-dataset sets (deployable: one gene axis for every dataset);
    # dashed = not a union (needs to know which dataset a cell is from to mask, or is a control)
    lines = [('union', 'F-stat, union of per-dataset top-n (shipped)', blue, '-'),
             ('hvg_union', 'HVG, union of per-dataset top-n', orange_free, '-'),
             ('per_dataset', "F-stat, each dataset's own top-n only", blue, '--'),
             ('intersection', 'genes measured in every dataset', vs.C['aqua'], '--'),
             ('random', 'random (size-matched)', vs.MUTED, '--')]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ends = []
    for s, lab, c, ls in lines:
        q = g[g.strategy == s].sort_values('n')
        ax.plot(q.n, q.top1, color=c, ls=ls, lw=2.2, marker='o', ms=4)
        ends.append([q.top1.iloc[-1], lab, c, q.n.iloc[-1]])
    ends.sort(key=lambda e: e[0])
    ly = [e[0] for e in ends]
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + 0.03)
    for (y0, lab, c, x0), yl in zip(ends, ly):
        ax.annotate(lab, xy=(x0, y0), xytext=(x0 * 1.12, yl), color=c, va='center', fontsize=vs.FONT_MIN - 2)
    ship = g[(g.strategy == 'union') & (g.n == N_SHIP)].top1.iloc[0]
    ax.axvline(N_SHIP, color=vs.C['red'], ls='--', lw=1.5, zorder=0)
    ax.scatter([N_SHIP], [ship], s=110, facecolor='white', edgecolor=vs.C['red'], lw=2.2, zorder=5)
    ax.text(N_SHIP * 1.04, 0.02, f'shipped: {N_SHIP} per dataset\n(3,096 genes)', color=vs.C['red'], va='bottom')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xticks([100, 250, 500, 1000, 2000, 5000])
    ax.set_xlabel('genes selected per dataset')
    ax.set_ylabel('perturbation separability\n(held-out replicate matching, top-1)')
    ax.set_ylim(0, None)
    vs.save(fig, out, 'figS1f_gene_selection')


PANELS = {'a': panel_a, 'b': panel_b, 'c': panel_c, 'd': panel_d, 'e': panel_e, 'f': panel_f}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order, col = dataset_order(d)
    for k in a.panels:
        PANELS[k](d, out, vs, order, col)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
