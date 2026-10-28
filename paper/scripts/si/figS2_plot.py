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

"""Supplementary Fig. S2 panels (the prior pull: gain by training support, and applied after vs during training;
related to Fig. 3), one file per panel in --out-dir (si/figS2), rendered ONLY from --data-dir
(paper/figures/si/figS2/data,
figS2_data.py). Arms are coloured as in (B) (formerly Fig. 3D, drawn by figS2b_pull_plot.py).

    figS2a_pull_set_size        offline sweep of the pulled-set size (best pull strength per size): net top-10 hits vs
                                no pull; open circle = the shipped size (formerly Supplementary Fig. S2A)
    (figS2b_pull_vs_base        pooled top-10 of gated pull, pull-all and no pull: figS2b_pull_plot.py)
    figS2c_deciles              net top-10 hits vs the no-pull model per decile of training support (1 = lowest),
                                pull-all vs gated pull; dotted line = median support
    figS2d_halves               the same summed over the lower and upper five deciles
    figS2e_posthoc_vs_trained   net top-10 flips vs the no-pull model on the gated prototypes' own queries, on all other
                                queries and in total: the post-hoc static blend over the full range of pull strength
                                (dashed, 95% bands) and the trained gated-pull model at its implied strength (points,
                                95% intervals; shaded, its interquartile range)
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

# S1G's arms (formerly Fig. 3D): the gated pull is ExPert's; equal pull on all prototypes in ink
ARMS = {'FULLOPEN27': 'equal pull, all prototypes', 'EFFGATE3': 'gated pull (ExPert)'}


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.grid': False})


def colours(vs):
    return {'FULLOPEN27': vs.INK, 'EFFGATE3': EXPERT_COLOR}


COLLATERAL = '#c9c8c3'          # gated pull on deciles with no gated prototype: collateral only


def panel_deciles(d, out, vs):
    t = pd.read_csv(d / 'figS2c_deciles.csv')
    meta = json.loads((d / 'figS2_meta.json').read_text())
    col = colours(vs)
    arms = [a_ for a_ in ARMS if a_ in set(t.arm)]
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    # the gate: the lowest-support queries, up to its edge on the decile axis
    edge = meta['gate_edge_decile']
    ax.axvspan(0.5, edge, color=EXPERT_COLOR, alpha=0.1, lw=0, zorder=0)
    for i, arm in enumerate(arms):
        g = t[t.arm == arm].sort_values('decile')
        x = g.decile.to_numpy() + (i - 0.5) * w
        c = [col[arm] if arm != 'EFFGATE3' or sh > 0 else COLLATERAL for sh in g.gated_share]
        ax.bar(x, g.net_top10, width=w * 0.92, color=c, zorder=3)
        ax.errorbar(x, g.net_top10, yerr=[g.net_top10 - g.ci_lo, g.ci_hi - g.net_top10], fmt='none',
                    ecolor=vs.INK2, lw=1.1, capsize=2.5, zorder=4)
    ax.axhline(0, color=vs.INK2, lw=1.2)
    ax.axvline(5.5, color=vs.MUTED, lw=1.2, ls=':')
    lo, hi = ax.get_ylim()
    top = hi + (hi - lo) * 0.3                     # headroom for the in-plot labels
    ax.set_ylim(lo, top)
    ax.text(5.6, top, 'median support', color=vs.MUTED, ha='left', va='top', fontsize=vs.FONT_MIN - 1)
    ax.text(0.55, top, f'gated: the {meta["n_gated"]:,}\nlowest-support\nprototypes', color=vs.INK2, ha='left',
            va='top', fontsize=vs.FONT_MIN - 1)
    dec = sorted(t.decile.unique())
    ax.set_xticks(dec)
    ax.set_xticklabels([f'{q}\nlowest' if q == dec[0] else f'{q}\nhighest' if q == dec[-1] else str(q) for q in dec])
    ax.set_xlim(0.5, dec[-1] + 0.5)
    ax.set_xlabel('decile of training support (training cells × training datasets)')
    ax.set_ylabel('net top-10 hits vs no pull')
    from matplotlib.patches import Patch
    hs = [Patch(color=col[a_], label=ARMS[a_]) for a_ in arms] + \
        [Patch(color=COLLATERAL, label='gated pull, deciles without gated prototypes (collateral)')]
    ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    vs.save(fig, out, 'figS2c_deciles')


def panel_halves(d, out, vs):
    t = pd.read_csv(d / 'figS2d_halves.csv')
    col = colours(vs)
    arms = [a_ for a_ in ARMS if a_ in set(t.arm)]
    halves = ['lower half', 'upper half']
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for i, arm in enumerate(arms):
        g = t[t.arm == arm].set_index('half').reindex(halves)
        x = np.arange(len(halves)) + (i - 0.5) * w
        ax.bar(x, g.net_top10, width=w * 0.92, color=col[arm], label=ARMS[arm], zorder=3)
        ax.errorbar(x, g.net_top10, yerr=[g.net_top10 - g.ci_lo, g.ci_hi - g.net_top10], fmt='none',
                    ecolor=vs.INK2, lw=1.2, capsize=3, zorder=4)
        for xx, v, hi, lo in zip(x, g.net_top10, g.ci_hi, g.ci_lo):
            ax.text(xx, (hi if v >= 0 else lo) + (4 if v >= 0 else -4), f'{v:+d}'.replace('-', '−'), ha='center',
                    va='bottom' if v >= 0 else 'top', color=vs.INK, fontsize=vs.FONT_MIN - 1)
    ax.axhline(0, color=vs.INK2, lw=1.2)
    ax.set_xticks(range(len(halves)))
    ax.set_xticklabels(['lower half\nof support', 'upper half\nof support'])
    ax.set_ylabel('net top-10 hits vs no pull')
    ax.margins(y=0.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False)
    vs.save(fig, out, 'figS2d_halves')


def panel_posthoc(d, out, vs):
    t = pd.read_csv(d / 'figS2e_posthoc_vs_trained.csv')
    meta = json.loads((d / 'figS2_meta.json').read_text())
    sim = t[t.kind == 'post-hoc'].sort_values('alpha')
    tr = t[t.kind == 'trained'].iloc[0]
    a0, (q1, q3) = meta['implied_alpha_median'], meta['implied_alpha_iqr']
    # as Fig. 3's draft dose curve: moved prototypes in ExPert's colour
    parts = (('s', 'gated prototypes\' own queries', EXPERT_COLOR),
             ('c', 'all other queries', vs.MUTED),
             ('t', 'total', vs.INK))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axvspan(q1, q3, color=vs.GRID, alpha=0.8, zorder=0)
    for k, lab, c in parts:
        ax.fill_between(sim.alpha, sim[k + 'lo'], sim[k + 'hi'], color=c, alpha=0.15, lw=0)
        ax.plot(sim.alpha, sim[k], color=c, lw=2.4, ls='--', marker='o', ms=4)
    off = {'s': -0.012, 'c': 0.0, 't': 0.012}
    for k, lab, c in parts:
        ax.errorbar(a0 + off[k], tr[k], yerr=[[tr[k] - tr[k + 'lo']], [tr[k + 'hi'] - tr[k]]],
                    fmt='o', ms=11, color=c, mec='white', mew=1.5, capsize=4, lw=2, zorder=5)
    ax.axhline(0, color=vs.INK2, lw=1.4)                   # no pull (BASE) is the zero line
    ax.set_xlim(0, 1.03)
    ax.set_xlabel(f'pull strength α on the {meta["n_gated"]:,} gated prototypes')
    ax.set_ylabel('net top-10 hits vs no pull')
    hs = [Line2D([], [], color=c, lw=2.4, label=lab) for _, lab, c in parts] + [
        Line2D([], [], color=vs.INK2, lw=2, ls='--', marker='o', ms=4, label='post-hoc adjustment'),
        Line2D([], [], color=vs.INK2, marker='o', lw=0, ms=10, label='trained model')]
    ax.legend(handles=hs, loc='lower left', frameon=False, ncol=2)
    vs.save(fig, out, 'figS2e_posthoc_vs_trained')


def panel_pull_size(d, out, vs):
    """A: net top-10 hits vs no pull at the best pull strength for each pulled-set size; the shipped size circled."""
    shipped = json.loads((d / 'figS2_meta.json').read_text())['pull_q']
    t = pd.read_csv(d / 'figS2a_pull_set_size.csv')
    best = t.loc[t.groupby('pull_q').total.idxmax()].sort_values('pull_q')
    x, y = best.pull_q.to_numpy() * 100, best.total.to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(0, color=vs.MUTED, lw=1)
    ax.plot(x, y, color=vs.C['blue'], lw=2.4, marker='o', ms=6)
    i = int(np.flatnonzero(np.isclose(best.pull_q, shipped))[0])
    ax.scatter([x[i]], [y[i]], s=110, facecolor='white', edgecolor=vs.C['red'], lw=2.2, zorder=5)
    ax.annotate(f'shipped: {shipped * 100:.1f}%\n({int(best.n_pulled.iloc[i]):,} perturbations)', xy=(x[i], y[i]),
                xytext=(x[i] + 6, y[i] + 6), color=vs.C['red'], va='bottom',
                arrowprops=dict(arrowstyle='-', color=vs.C['red'], lw=1))
    ax.set_xlabel('perturbations pulled (% with the lowest support)')
    ax.set_ylabel('net top-10 hits vs no pull (OOD)')
    ax.set_xlim(0, x.max() + 2)
    ax.set_ylim(min(0, y.min()) - 5, y.max() + 25)
    vs.save(fig, out, 'figS2a_pull_set_size')


PANELS = {'a': panel_pull_size, 'c': panel_deciles, 'd': panel_halves, 'e': panel_posthoc}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in a.panels:
        PANELS[k](d, out, vs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
