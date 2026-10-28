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

"""Supplementary Fig. S6 panels (zero-shot prototype geometry and the fill map, related to Fig. 3), one file per panel
in --out-dir (si/figS6), rendered ONLY from --data-dir (paper/figures/si/figS6/data,
figS6_data.py). Kinds are coloured as in Fig. 3C;
the shipped fill (graph, k = 25, shipped prior banks) in ExPert's colour.

    figS6a_angle               angle to the mean trained-prototype direction: trained prototypes, zero-shot fills and
                               trained genes re-placed by the fill map as if unseen
    figS6b_graph_k             cos^4-weighted graph fill at k = 25 / 50 / 100 vs the cos fill, 12 fold models
    figS6c_weighting           neighbour weighting (cos^4 shipped, cos^8, softmax) vs the cos fill, 12 fold models
    figS6d_fill_map            zero-shot (x chance) by fill map (MLP, linear, graph): shipped bank set (line) and every
                               co-expression-free bank set (points)
    figS6e_posthoc_vs_trained  top-10 hits of trained perturbations recovered over the post-hoc model when the zero-shot
                               fills join the candidates, gated pull (ExPert) and full pull, +x%
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
from matplotlib.lines import Line2D  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig2_common import EXPERT_COLOR  # noqa: E402

# Fig. 3C's kind colours (fig3_plot.KIND)
KIND = {'trained': ('trained prototypes', '#0b0b0b'),
        'fill': ('zero-shot fills', '#bdbcb6'),
        'held-out placement': ('trained genes, re-placed\nby the fill map', '#7d7c76')}


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.grid': False})


def panel_a(d, out, vs, meta):
    t = pd.read_csv(d / 'figS6a_angle.csv')
    bins = np.linspace(np.floor(t.angle_deg.quantile(0.001)), np.ceil(t.angle_deg.quantile(0.999)), 70)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    hs = []
    for k in ('trained', 'fill', 'held-out placement'):
        lab, c = KIND[k]
        x = t.angle_deg[t.kind == k]
        ax.hist(x, bins=bins, density=True, histtype='step', lw=2.2, color=c)
        ax.hist(x, bins=bins, density=True, color=c, alpha=0.18, lw=0)
        med = float(x.median())
        ax.axvline(med, color=c, lw=1.4, ls='--')
        hs.append(Line2D([], [], color=c, lw=2.2,
                         label=f'{lab.replace(chr(10), " ")}: median {med:.1f}° (n = {len(x):,})'))
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=vs.FONT_MIN - 1)
    ax.set_xlabel('angle to the mean trained-prototype direction (°)')
    ax.set_ylabel('density')
    vs.save(fig, out, 'figS6a_angle')


def _sets(ax, x, y, vs, rng):
    """every co-expression-free bank set as a light point, jittered along x"""
    ax.scatter(x + rng.uniform(-0.12, 0.12, len(x)), y, s=14, color=vs.MUTED, alpha=0.45, lw=0, zorder=2)


# the three measures the fill was chosen on: zero-shot ID / OOD and trained-query OOD (relative to the cos fill)
Y_CAP = 60          # % : the one fold with few OOD unseen perturbations reaches +130-170%; drawn at the cap, labelled
MEASURES = [('zs_ID', 'zero-shot, ID', 'o', 'full'), ('zs_OOD', 'zero-shot, OOD', 's', 'none'),
            ('trained_OOD', 'trained queries, OOD', '^', 'full')]


def _strips(d, out, vs, meta, csv, name, panel, xlabel):
    """per arm: per-fold relative change vs the rebuilt cos fill (dots) and the median over folds (bar)"""
    t = pd.read_csv(d / csv)
    net = pd.read_csv(d / csv.replace('.csv', '_summary.csv')).drop_duplicates('variant').set_index('variant').row_sum
    arms = list(dict.fromkeys(t.variant))
    labs = t.drop_duplicates('variant').set_index('variant').label
    shipped = meta['shipped'][panel]
    w = 0.26
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.9 * len(arms) + 1.6, 4.8))
    for i, v in enumerate(arms):
        for j, (m, lab, mk, fill) in enumerate(MEASURES):
            g = t[(t.variant == v) & (t.metric == m)]
            x = i + (j - 1) * w
            c = vs.INK if m != 'trained_OOD' else vs.MUTED
            yv = 100 * g.rel.to_numpy(); xj = x + rng.uniform(-0.05, 0.05, len(g))
            cap = yv > Y_CAP
            ax.scatter(xj[~cap], yv[~cap], s=16, marker=mk, color=c,
                       facecolor=c if fill == 'full' else 'white', lw=0.9, alpha=0.8, zorder=5)
            if cap.any():                 # one marker + one label per strip, values listed
                ax.scatter(xj[cap], np.full(cap.sum(), Y_CAP), s=22, marker=mk, color=c, facecolor='white', lw=0.9,
                           zorder=5, clip_on=False)
                ax.annotate(', '.join(f'+{yy:.0f}' for yy in sorted(yv[cap])), (xj[cap].max(), Y_CAP), xytext=(5, 0),
                            textcoords='offset points', ha='left', va='center', fontsize=vs.FONT_MIN - 3,
                            color=vs.INK2, annotation_clip=False)
            med = 100 * float(np.median(g.rel))
            ax.hlines(med, x - w * 0.42, x + w * 0.42, color=EXPERT_COLOR if v == shipped else c, lw=2.2, zorder=4)
    ax.axhline(0, color=vs.INK2, lw=1.2)
    ax.set_ylim(top=Y_CAP)
    # net change = sum of the three medians (the selection criterion), one value per arm above the panel
    tr = ax.get_xaxis_transform()
    for i, v in enumerate(arms):
        ax.text(i, 1.07, f'{100 * net[v]:+.1f}%'.replace('-', '−'), transform=tr, ha='center', va='bottom', fontsize=vs.FONT_MIN,
                color=EXPERT_COLOR if v == shipped else vs.INK, fontweight='bold' if v == shipped else 'normal')
    ax.text(-0.6, 1.07, 'net', transform=tr, ha='right', va='bottom', fontsize=vs.FONT_MIN - 2, color=vs.INK2)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([labs[v].replace(', k', '\nk') for v in arms])
    for tl, v in zip(ax.get_xticklabels(), arms):
        if v == shipped:
            tl.set_color(EXPERT_COLOR); tl.set_fontweight('bold')
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('change vs cos-weighted fill (%)')
    hs = [Line2D([], [], color=vs.INK if m != 'trained_OOD' else vs.MUTED, marker=mk, lw=0, ms=6,
                 mfc=(vs.INK if m != 'trained_OOD' else vs.MUTED) if fill == 'full' else 'white', label=lab)
          for m, lab, mk, fill in MEASURES]
    hs.append(Line2D([], [], color=vs.INK2, lw=2.2, label=f'median over {meta["n_folds"]} fold models'))
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
              fontsize=vs.FONT_MIN - 2)
    vs.save(fig, out, name)


def panel_b(d, out, vs, meta):
    _strips(d, out, vs, meta, 'figS6b_graph_k.csv', 'figS6b_graph_k', 'b', 'neighbour weighting and prior-graph neighbours')


def panel_c(d, out, vs, meta):
    _strips(d, out, vs, meta, 'figS6c_weighting.csv', 'figS6c_weighting', 'c', 'neighbour weighting (k = 25)')


# one representative per family: linear at the previously shipped ridge strength, graph at the shipped k
FAMILY = [('mlp', 'MLP'), ('ridge0.5', 'linear'), ('graph25', 'graph')]


def panel_d(d, out, vs, meta):
    t = pd.read_csv(d / 'figS6d_fill_map.csv')
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for i, (mp, lab) in enumerate(FAMILY):
        g = t[t['map'] == mp]
        _sets(ax, np.full(len(g), i, float), g.zero_shot_x_chance.to_numpy(), vs, rng)
        v = float(g[g.shipped_banks].zero_shot_x_chance.iloc[0])
        c = EXPERT_COLOR if lab == 'graph' else vs.INK
        ax.hlines(v, i - 0.24, i + 0.24, color=c, lw=3, zorder=4)
        ax.text(i + 0.27, v, f'{v:.1f}', va='center', ha='left', color=c, fontsize=vs.FONT_MIN - 1)
    ax.set_xticks(range(len(FAMILY))); ax.set_xticklabels([lab for _, lab in FAMILY])
    ax.set_xlim(-0.6, len(FAMILY) - 0.3)
    ax.set_ylabel('zero-shot top-10 (× chance)')
    ax.set_ylim(bottom=0)
    hs = [Line2D([], [], color=vs.INK, lw=3, label='shipped prior banks'),
          Line2D([], [], color=vs.MUTED, marker='o', lw=0, ms=5, alpha=0.6,
                 label=f'other prior-bank sets (n = {meta["n_bank_sets"] - 1})')]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=vs.FONT_MIN - 1)
    vs.save(fig, out, 'figS6d_fill_map')


def panel_e(d, out, vs, meta):
    """as the 'filled bank' row of Fig. 3's draft (fig3d_sim_vs_realised): top-10 hits over the post-hoc model"""
    t = pd.read_csv(d / 'figS6e_posthoc_vs_trained.csv')
    col = {'ExPert': EXPERT_COLOR, 'full pull': vs.INK}
    bh = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for j, r in enumerate(t.itertuples()):
        y = bh / 2 - j * bh
        ax.barh(y, r.recovered, height=bh * 0.92, color=col[r.model], zorder=3)
        ax.errorbar(r.recovered, y, xerr=[[r.recovered - r.ci_lo], [r.ci_hi - r.recovered]], fmt='none',
                    ecolor=vs.INK if r.model == 'ExPert' else vs.MUTED, lw=1.3, capsize=3, zorder=4)
        ax.text(r.ci_hi + 8, y, f'{r.model}  {r.recovered:+.0f} ({r.recovered_pct:+.0f}%)', va='center',
                color=col[r.model], fontweight='bold' if r.model == 'ExPert' else 'normal')
    ax.axvline(0, color=vs.INK2, lw=1.2)
    ax.set_yticks([0]); ax.set_yticklabels([f'zero-shot\nprototypes\n(+{int(t.n_rows_added.iloc[0]):,} rows)'])
    ax.set_ylim(-0.55, 0.55); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, float(t.ci_hi.max()) * 1.75)
    ax.set_xlabel('top-10 hits over post-hoc adjustment')
    vs.save(fig, out, 'figS6e_posthoc_vs_trained')


PANELS = {'a': panel_a, 'b': panel_b, 'c': panel_c, 'd': panel_d, 'e': panel_e}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / 'figS6_meta.json').read_text())
    for k in a.panels:
        PANELS[k](d, out, vs, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
