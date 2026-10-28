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

"""Supplementary Fig. S12 panels (ExPert scores of the bulk RNA-seq livers against the snRNA-seq hepatocyte scores;
related to Fig. 5), one file per panel in --out-dir (si/figS12), rendered ONLY from --data-dir
(paper/figures/si/figS12/data,
figS12_data.py). Conditions in Fig. 5's colours (AC pink, AH plum, healthy gray); predicted loss in ExPert's colour.

    figS12a_families     mean predicted-loss z of the mRNA export families, snRNA-seq vs bulk, per condition, with
                         the Spearman correlation of the two columns
    figS12b_enrichment   top-200 enrichment of mRNA export and mRNA splicing among the predicted losses, snRNA-seq vs
                         bulk, per condition (fold over expected; hits above each bar)
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

CONDS = [('AC', 'AC'), ('AH', 'AH'), ('Normal', 'healthy')]
CC = {'AC': F.AC_C, 'AH': F.AH_C, 'Normal': '#8a8983'}


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def panel_a(d, out, vs, meta):
    t = pd.read_csv(d / 'figS12a_families.csv')
    rho = pd.read_csv(d / 'figS12a_rho.csv').set_index('cond')
    fams = meta['families']
    cols = [(c, s) for c, _ in CONDS for s in ('z_sn', 'z_bulk')]
    M = np.array([[t[(t.family == f) & (t.cond == c)][s].iloc[0] for c, s in cols] for f in fams])
    lim = np.ceil(np.nanmax(np.abs(M)) * 2) / 2
    cmap = LinearSegmentedColormap.from_list('z', ['#6baed6', '#ffffff', F.EXPERT])
    fig, ax = plt.subplots(figsize=(0.9 * len(cols) + 3.2, 0.62 * len(fams) + 2.2))
    vs.heatmap(ax, M, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    for i in range(len(fams)):
        for j in range(len(cols)):
            ax.text(j, i, f'{M[i, j]:.2f}'.replace('-', '−'), ha='center', va='center', color=vs.INK,
                    fontsize=vs.FONT_MIN - 3)
    for x in (1.5, 3.5):
        ax.axvline(x, color=vs.SURFACE, lw=4)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(['snRNA' if s == 'z_sn' else 'bulk' for _, s in cols], rotation=45, ha='right',
                       rotation_mode='anchor')
    for k, (c, cl) in enumerate(CONDS):
        x0, x1, yb = 2 * k - 0.35, 2 * k + 1.35, -0.62                   # bracket over snRNA | bulk
        ax.plot([x0, x0, x1, x1], [yb + 0.08, yb, yb, yb + 0.08], color=vs.INK, lw=1.0, clip_on=False)
        ax.text(2 * k + 0.5, yb - 0.06, f'{cl}\nρ = {rho.loc[c, "rho_families"]:.2f}'.replace('-', '−'),
                ha='center', va='bottom', color=vs.INK, fontweight='bold', linespacing=1.3)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    cb = fig.colorbar(ax.collections[0], ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label('ExPert predicted loss (z)', color=vs.INK2)
    vs.save(fig, out, 'figS12a_families')


def panel_b(d, out, vs, meta):
    t = pd.read_csv(d / 'figS12b_enrichment.csv')
    sets = ['mRNA export', 'mRNA splicing']
    w = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
    for ax, gs in zip(axs, sets):
        for i, (c, cl) in enumerate(CONDS):
            for j, src in enumerate(('snRNA', 'bulk')):
                r = t[(t.cond == c) & (t.gene_set == gs) & (t.source == src)].iloc[0]
                x = i + (j - 0.5) * w
                ax.bar(x, r.fold, w * 0.9, color=CC[c], alpha=0.45 if src == 'snRNA' else 1.0, zorder=3,
                       edgecolor=CC[c], lw=1.0)
                ax.text(x, r.fold, f'{r.hits}', ha='center', va='bottom', fontsize=vs.FONT_MIN - 3,
                        color=vs.INK, fontweight='bold' if r.p < 0.05 else 'normal')
        ax.axhline(1, color=vs.INK2, lw=1, ls=':')
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([cl for _, cl in CONDS])
        ax.set_title(gs, loc='left')
        ax.grid(axis='x', visible=False)
    axs[0].set_ylabel(f'top-{meta["top"]} hits over expected')
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(fc=vs.MUTED, alpha=0.45, label='snRNA-seq hepatocytes'),
                           Patch(fc=vs.MUTED, label='bulk livers')], frameon=False, fontsize=vs.FONT_MIN - 3,
                  loc='upper right')
    fig.tight_layout()
    vs.save(fig, out, 'figS12b_enrichment')


PANELS = {'a': panel_a, 'b': panel_b}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / 'figS12_meta.json').read_text())
    for k in a.panels:
        PANELS[k](d, out, vs, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
