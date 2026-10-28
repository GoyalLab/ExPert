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

"""Supplementary Fig. S13 panels (zero-shot predictions in ALD: ESRP2 and the other splicing regulators; related to
Fig. 5), one file per panel in --out-dir (si/figS13), rendered ONLY from --data-dir
(paper/figures/si/figS13/data, figS13_data.py).
Predicted loss in blue, predicted gain in green (the Fig. 5E scale); conditions in Fig. 5's colours.

    figS13a_regulators     change from healthy (z in SDs of the healthy scores) of each splicing regulator, healthy / AC / AH hepatocytes; zero-shot
                           rows first
    figS13b_neighbourhood  per hub (SRSF6, the top predicted splicing hit in AH; SRSF10, the SR protein nearest the
                           ESRP2 fill), the 10 most enriched of its 50 nearest prototypes, as one kNN graph; fill = AH vs healthy z (blue loss,
                           green gain), border black = trained, gray = zero-shot
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_final as F  # noqa: E402

CONDS = ['healthy', 'AC', 'AH']
CC = {'healthy': '#8a8983', 'AC': F.AC_C, 'AH': F.AH_C}
SHIPPED = 'cos4'
# Fig. 5E's scale: predicted loss blue, predicted gain green (z > 0 = loss, so loss sits at the top of the map)
LOSSGAIN = LinearSegmentedColormap.from_list('lossgain', [F.GAIN_C, '#ffffff', F.LOSS_C])


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def lossgain_arrows(cax, vs, x=-1.6):
    """as the Fig. 5E colour bar: an arrow out from the centre on each half, loss up (higher cosine), gain down"""
    for y0, y1, txt in ((0.56, 0.92, 'loss'), (0.44, 0.08, 'gain')):
        cax.annotate('', xy=(x, y1), xytext=(x, y0), xycoords='axes fraction', annotation_clip=False,
                     arrowprops=dict(arrowstyle='-|>', color=vs.INK2, lw=1.3, mutation_scale=12))
        cax.text(x - 0.6, (y0 + y1) / 2, txt, transform=cax.transAxes, rotation=90, ha='center', va='center',
                 color=vs.INK2, fontsize=vs.FONT_MIN - 2, clip_on=False)


# measured direction in severe alcoholic hepatitis reported by the cohort's paper (doi 10.1038/s41467-025-63251-2):
# RBP mRNA / protein down (ESRP2, RBM47; A1CF hepatocyte-specific) or up (CELF2, SRSF1)
PAPER_DIR = {'ESRP2': '↓', 'RBM47': '↓', 'A1CF': '↓', 'CELF2': '↑', 'SRSF1': '↑'}


def panel_a(d, out, vs, meta):
    t = pd.read_csv(d / 'figS13a_ranks.csv')
    t = t[(t.fill == SHIPPED) & t.group.notna() & (t.group != '')]
    groups = list(dict.fromkeys(t.group))
    genes = list(dict.fromkeys(t.gene))
    gof = dict(zip(t.gene, t.group))
    zs = dict(zip(t.gene, t.zero_shot))
    cols = [c for c in CONDS if c != 'healthy']        # healthy is the reference (0 by construction)
    M = np.array([[t[(t.gene == g) & (t.condition == c)].z_vs_healthy.iloc[0] for c in cols] for g in genes])
    lim = np.ceil(np.abs(M).max() * 2) / 2
    fig, ax = plt.subplots(figsize=(4.8, 0.42 * len(genes) + 1.8))
    vs.heatmap(ax, M, cmap=LOSSGAIN, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    for i, g in enumerate(genes):
        for j in range(len(cols)):
            ax.text(j, i, f'{M[i, j] + 0.0:.1f}'.replace('-0.0', '0.0').replace('-', '−'), ha='center', va='center',
                    color=vs.INK, fontsize=vs.FONT_MIN - 3)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels([f'{g} {PAPER_DIR[g]}' if g in PAPER_DIR else g for g in genes], fontstyle='italic')
    for tl, g in zip(ax.get_yticklabels(), genes):
        if zs[g]:
            tl.set_fontweight('bold')
    ax.tick_params(length=0)
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    tr = ax.get_yaxis_transform()
    for gr in groups:
        ix = [i for i, g in enumerate(genes) if gof[g] == gr]
        if ix[0] > 0:
            ax.axhline(ix[0] - 0.5, color=vs.INK, lw=1.5)
        ax.text(1.04, (ix[0] + ix[-1]) / 2, gr.replace(' (', '\n('), transform=tr, rotation=270, ha='left',
                va='center', color=vs.INK2, fontsize=vs.FONT_MIN - 4, linespacing=1.0)
    cb = fig.colorbar(ax.collections[0], ax=ax, fraction=0.06, pad=0.42)
    cb.outline.set_visible(False)
    cb.set_label('predicted loss vs healthy (z)', color=vs.INK2)
    lossgain_arrows(cb.ax, vs)
    vs.save(fig, out, 'figS13a_regulators')


def panel_neighbourhood(d, out, vs, meta):
    """neighbourhoods of the centre prototypes as one kNN graph; colour = predicted loss z, border = trained / zero-shot"""
    import networkx as nx
    from adjustText import adjust_text
    from matplotlib.lines import Line2D
    n = pd.read_csv(d / 'figS13b_nodes.csv')
    e = pd.read_csv(d / 'figS13b_edges.csv')
    zcol = f'zh_{meta["graph_condition"]}'
    g = nx.Graph()
    g.add_nodes_from(n.gene)
    for r in e.itertuples():
        g.add_edge(r.a, r.b, weight=max(r.cosine, 0.01))
    pos = nx.spring_layout(g, weight='weight', seed=0, iterations=600, k=1.6 / np.sqrt(len(g)))
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    for r in e.itertuples():
        (x0, y0), (x1, y1) = pos[r.a], pos[r.b]
        ax.plot([x0, x1], [y0, y1], color=vs.GRID, lw=0.5 + 2.5 * max(r.cosine, 0), zorder=1)
    lim = np.ceil(np.abs(n[zcol]).max() * 2) / 2
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    xy = np.array([pos[x] for x in n.gene])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=n[zcol], cmap=LOSSGAIN, norm=norm, zorder=3,
                    s=np.where(n.centre, 520, 180), edgecolors=np.where(n.zero_shot, vs.MUTED, vs.INK),
                    linewidths=np.where(n.centre, 2.6, 1.6))
    txt = [ax.text(x, y, gname, ha='center', va='center', fontsize=vs.FONT_MIN - (1 if c else 4), fontstyle='italic',
                   fontweight='bold' if (c or rg) else 'normal', color=vs.INK, zorder=5)
           for (x, y), gname, c, rg in zip(xy, n.gene, n.centre, n.regulator)]
    # the centres' own circles are large: pad their footprint so no label lands on them
    th = np.linspace(0, 2 * np.pi, 16)
    cx = [xy[i] + 0.035 * np.array([np.cos(t), np.sin(t)]) for i in np.flatnonzero(n.centre.to_numpy()) for t in th]
    px, py = np.r_[xy[:, 0], [c[0] for c in cx]], np.r_[xy[:, 1], [c[1] for c in cx]]
    adjust_text(txt, x=px, y=py, ax=ax, expand=(1.6, 2.0), force_text=(0.8, 1.2), force_static=(0.6, 1.0),
                iter_lim=800, arrowprops=dict(arrowstyle='-', color=vs.MUTED, lw=0.5))
    ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.07)
    cb.outline.set_visible(False)
    lossgain_arrows(cb.ax, vs)
    cb.set_label(f'predicted loss in {meta["graph_condition"]} vs healthy hepatocytes (z)', color=vs.INK2)
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=10, mfc='white', mec=vs.INK, mew=1.6, label='trained'),
                       Line2D([], [], ls='', marker='o', ms=10, mfc='white', mec=vs.MUTED, mew=1.6, label='zero-shot')],
              loc='upper left', frameon=False, fontsize=vs.FONT_MIN - 3)
    vs.save(fig, out, 'figS13b_neighbourhood')


PANELS = {'a': panel_a, 'b': panel_neighbourhood}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / 'figS13_meta.json').read_text())
    for k in a.panels:
        PANELS[k](d, out, vs, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
