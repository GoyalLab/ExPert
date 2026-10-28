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

"""Supplementary Fig. S8 panels (drug-target recovery with the zero-shot prototypes added to the candidate bank, as fold
over random; related to Fig. 4), one file per panel in --out-dir (si/figS8), rendered ONLY from --data-dir
(paper/figures/si/figS8/data, figS8_data.py). The zero-shot-padded bank in ExPert's colour, the
trained bank in ink, the matched
panel muted; shipped (cos^4) fill throughout the main panels.

    figS8a_sciplex3_fold     sci-Plex3: fold over random vs top-k for the matched panel, trained and padded banks
    figS8b_tahoe_top10       Tahoe-100M: top-10 fold over the matched-random null, annotated target and complex
                             partner, trained vs padded bank, drugs recovered per bar
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

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig2_common import EXPERT_COLOR  # noqa: E402

SHIPPED = 'cos4'


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.grid': False})


def colours(vs):
    return {'full': EXPERT_COLOR, 'trained': vs.INK, 'matched': vs.MUTED}


def _k_axis(ax, ks):
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('top-k')


def panel_a(d, out, vs, meta):
    t = pd.read_csv(d / 'figS8a_sciplex3_fold.csv')
    t = t[t.fill == SHIPPED]
    col = colours(vs)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for j, bank in enumerate(('matched', 'trained', 'full')):
        g = t[t.bank == bank].sort_values('k')
        c = col[bank]
        x = g.k * 1.07 ** (j - 1)                    # dodge on the log axis so the intervals stay readable
        ax.errorbar(x, g.fold, yerr=[g.fold - g.ci_lo, g.ci_hi - g.fold], fmt='none', ecolor=c, lw=1.2, capsize=2.5,
                    alpha=0.8, zorder=2)
        ax.plot(x, g.fold, marker='o', ms=5, lw=2, color=c, zorder=3,
                label=f'{g.bank_label.iloc[0]} ({g.n_cand.iloc[0]:,} candidates, {g.n_cond.iloc[0]:,} conditions)')
    ax.axhline(1, color=vs.INK2, lw=1, ls=':')
    _k_axis(ax, sorted(t.k.unique()))
    ax.set_ylim(bottom=0)
    ax.set_ylabel('fold over random')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=vs.FONT_MIN - 2)
    vs.save(fig, out, 'figS8a_sciplex3_fold')


def panel_b(d, out, vs, meta):
    t = pd.read_csv(d / 'figS8b_tahoe_top10.csv')
    t = t[t.fill == SHIPPED]
    stats = list(dict.fromkeys(t.statistic))
    arms = [('trained', vs.INK, 'trained'), ('padded', EXPERT_COLOR, 'zero-shot padded')]
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    for i, s in enumerate(stats):
        for j, (arm, c, lab) in enumerate(arms):
            r = t[(t.statistic == s) & (t.arm == arm)].iloc[0]
            x = i + (j - 0.5) * w
            ax.bar(x, r.fold, w * 0.9, color=c, label=lab if i == 0 else None, zorder=3)
            ax.errorbar(x, r.fold, yerr=[[r.fold - r.ci_lo], [r.ci_hi - r.fold]], fmt='none',
                        ecolor=vs.MUTED if arm == 'trained' else vs.INK, lw=1.3, capsize=3, zorder=4)
            ax.text(x, 0.12, f'{r.n_recovered}\nof\n{r.n_drugs}', ha='center', va='bottom', linespacing=0.95,
                    fontsize=vs.FONT_MIN - 4, color='white', zorder=5)
    ax.axhline(1, color=vs.INK2, lw=1, ls=':')
    ax.set_xticks(range(len(stats))); ax.set_xticklabels(stats)
    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.set_ylabel(f'top-{meta["tahoe_top_k"]} fold over matched random')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=vs.FONT_MIN - 2)
    vs.save(fig, out, 'figS8b_tahoe_top10')


PANELS = {'a': panel_a, 'b': panel_b}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / 'figS8_meta.json').read_text())
    for k in a.panels:
        PANELS[k](d, out, vs, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
