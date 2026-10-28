#!/usr/bin/env python
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Supplementary Fig. S14 panels (PBMC overview; related to Fig. 6), one file per panel in --out-dir (si/figS14),
rendered ONLY from --data-dir (paper/figures/si/figS14/data, figS14_data.py). Fig. 6's colours:
PMD blue, non-PMD light gray,
undiagnosed gray; the OXPHOS-loss score on Fig. 6C's one-sided violet scale.

    figS14a_celltypes   UMAP of all PBMCs in the ExPert latent, by cell type
    figS14b_donors      the same UMAP by donor group, each donor named at its densest region
    figS14c_oxphos      the same UMAP by the uncentred OXPHOS-loss score
    figS14d_unconv_t    composition of the "unconventional T" cells (SingleR fine labels), per donor
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig6_pmd_plot as F6  # noqa: E402  (constants and colours only)

DONOR_ORDER = ['PT43', 'PT46', 'PT54', 'PT67', 'PT41', 'PT42', 'PT38', 'PT50', 'PT55']
# cell types without the vizstyle oranges (#eb6834, #eda100 ~ ExPert's reserved #E69F00)
CT_COL = {'Monocytes': '#2a78d6', 'Dendritic cells': '#e34948', 'B cells': '#1baf7a', 'NK cells': '#8c6d5a',
          'CD4+ T cells': '#e87ba4', 'CD8+ T cells': '#008300', 'Nonclassical T cells': '#4a3aa7'}


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    F6.VS = vs


def gcol(vs):
    return {'PMD': vs.C['blue'], 'non-PMD': '#cfcec8', 'undiagnosed': '#a9a8a2'}


def umap_ax(vs):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    return fig, ax


def finish(ax):
    F6.bare(ax)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')


def panel_a(d, out, vs, meta):
    from adjustText import adjust_text
    c = pd.read_csv(d / 'figS14abc_cells.csv').sample(frac=1, random_state=0)
    cts = [t for t in F6.CT if t in set(c.celltype)]
    cmap = {t: CT_COL[t] for t in cts}
    fig, ax = umap_ax(vs)
    ax.scatter(c.umap1_all, c.umap2_all, s=3, lw=0, c=c.celltype.map(cmap), alpha=0.6)
    texts = []
    for t in cts:
        s_ = c[c.celltype == t]
        H, xe, ye = np.histogram2d(s_.umap1_all, s_.umap2_all, bins=40)
        i, j = np.unravel_index(H.argmax(), H.shape)
        texts.append(ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, F6.CT_LAB[t], ha='center',
                             va='center', color=cmap[t], fontsize=vs.FONT_MIN + 1, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8)))
    adjust_text(texts, ax=ax, only_move={'text': 'y'}, force_text=(0.2, 1.0))
    finish(ax)
    vs.save(fig, out, 'figS14a_celltypes')


def panel_b(d, out, vs, meta):
    from adjustText import adjust_text
    c = pd.read_csv(d / 'figS14abc_cells.csv').sample(frac=1, random_state=0)
    gc = gcol(vs)
    fig, ax = umap_ax(vs)
    for g in ('undiagnosed', 'non-PMD', 'PMD'):              # PMD on top
        s_ = c[c.group == g]
        ax.scatter(s_.umap1_all, s_.umap2_all, s=3, lw=0, color=gc[g], alpha=0.6, label=g)
    texts = []
    for dn, s_ in c.groupby('donor'):
        H, xe, ye = np.histogram2d(s_.umap1_all, s_.umap2_all, bins=40)
        i, j = np.unravel_index(H.argmax(), H.shape)
        texts.append(ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, dn, ha='center', va='center',
                             fontsize=vs.FONT_MIN - 2, fontweight='bold' if dn in F6.PMD else 'normal',
                             color=vs.INK if dn in F6.PMD else vs.INK2,
                             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85)))
    adjust_text(texts, ax=ax, force_text=(0.4, 0.6), expand=(1.2, 1.4),
                arrowprops=dict(arrowstyle='-', color=vs.INK2, lw=0.7))
    finish(ax)
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=8, color=gc[g], label=g)
                       for g in ('PMD', 'non-PMD', 'undiagnosed')], loc='upper left', frameon=False,
              fontsize=vs.FONT_MIN - 3)
    vs.save(fig, out, 'figS14b_donors')


def panel_c(d, out, vs, meta):
    c = pd.read_csv(d / 'figS14abc_cells.csv').sample(frac=1, random_state=0)
    v = c.oxphos_raw_smooth.to_numpy()
    lo, hi = np.quantile(v, [0.02, 0.98])
    cmap = LinearSegmentedColormap.from_list('lossonly', [F6.MID, F6.LOSS])     # Fig. 6C: one-sided, no gain end
    fig, ax = umap_ax(vs)
    sc = ax.scatter(c.umap1_all, c.umap2_all, s=3, lw=0, c=np.clip(v, lo, hi), cmap=cmap, vmin=lo, vmax=hi)
    finish(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([lo, hi]); cb.set_ticklabels([f'{lo:.2f}', f'{hi:.2f}'])
    cb.set_label('OXPHOS loss score'); cb.outline.set_visible(False)
    vs.save(fig, out, 'figS14c_oxphos')


def panel_d(d, out, vs, meta):
    n = pd.read_csv(d / 'figS14d_counts.csv')
    cts = [t for t in F6.CT if t in set(n.celltype)]
    donors = [x for x in DONOR_ORDER if x in set(n.donor)]
    M = n.pivot(index='celltype', columns='donor', values='n_cells').reindex(index=cts, columns=donors).fillna(0)
    fig, ax = plt.subplots(figsize=(0.8 * len(donors) + 2.6, 0.55 * len(cts) + 1.8))
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cmap = LinearSegmentedColormap.from_list('n', ['#ffffff', '#a9a8a2'])
    norm = Normalize(0, np.log10(M.to_numpy().max() + 1) * 1.15)
    vs.heatmap(ax, np.log10(M.to_numpy() + 1), cmap=cmap, norm=norm)
    # colour = log10(cells + 1); the bar is labelled in cells
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
    ticks = [t for t in (0, 10, 100, 1000, 10000) if np.log10(t + 1) <= norm.vmax]
    cb.set_ticks([np.log10(t + 1) for t in ticks]); cb.set_ticklabels([f'{t:,}' for t in ticks])
    cb.set_label('cells'); cb.outline.set_visible(False)
    for i, t in enumerate(cts):
        for j, dn in enumerate(donors):
            k = int(M.loc[t, dn])
            low = k < meta['min_cells']
            ax.text(j, i, f'{k:,}', ha='center', va='center', fontsize=vs.FONT_MIN - 4,
                    color=vs.INK, fontweight='bold' if low else 'normal')
            if low:
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, fill=False, ec=vs.INK, lw=1.4, ls='--'))
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors, rotation=45, ha='right', rotation_mode='anchor')
    for tl, dn in zip(ax.get_xticklabels(), donors):
        tl.set_color(gcol(vs)[F6.group_of(dn)] if dn in F6.PMD else vs.INK2)
        if dn in F6.PMD:
            tl.set_fontweight('bold')
    ax.set_yticks(range(len(cts))); ax.set_yticklabels([F6.CT_LAB[t] for t in cts])
    ax.tick_params(length=0); ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    vs.save(fig, out, 'figS14d_counts')


def panel_e(d, out, vs, meta):
    u = pd.read_csv(d / 'figS14e_unconv_t.csv')
    donors = [x for x in DONOR_ORDER if x in set(u.donor)]
    labs = u.groupby('fine').n_cells.sum().sort_values(ascending=False).index.tolist()
    M = u.pivot(index='donor', columns='fine', values='n_cells').reindex(index=donors, columns=labs).fillna(0)
    cols = [vs.INK2, '#cfcec8', vs.MUTED, '#e8e7e2'][:len(labs)]
    fig, ax = plt.subplots(figsize=(0.7 * len(donors) + 2.4, 4.0))
    bottom = np.zeros(len(donors))
    for lab, col in zip(labs, cols):
        ax.bar(range(len(donors)), M[lab], bottom=bottom, color=col, width=0.7, label=lab, zorder=3)
        bottom += M[lab].to_numpy()
    ax.set_xticks(range(len(donors)))
    ax.set_xticklabels(donors, rotation=45, ha='right', rotation_mode='anchor')
    for tl, dn in zip(ax.get_xticklabels(), donors):
        if dn in F6.PMD:
            tl.set_fontweight('bold')
    ax.set_ylabel('unconventional T cells')
    ax.legend(frameon=False, fontsize=vs.FONT_MIN - 3, loc='upper left')
    ax.grid(axis='x', visible=False)
    vs.save(fig, out, 'figS14d_unconv_t')


PANELS = {'a': panel_a, 'b': panel_b, 'c': panel_c, 'd': panel_e}   # the counts heatmap (panel_d) was dropped


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / 'figS14_meta.json').read_text())
    for k in a.panels:
        PANELS[k](d, out, vs, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
