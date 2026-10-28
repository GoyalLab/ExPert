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

"""Supplementary Fig. S5 (extended doubles recovery, related to Fig. 3). Reads ONLY --data-dir
(paper/figures/si/figS5/data,
figS5_data.py); panels drawn exactly as Fig. 3 H/I (fig3_doubles_plot.panel). One file per panel in --out-dir
(si/figS5); each panel's source table -> --data-dir/<name>_source.csv.

    figS5a_singles_gain             CRISPRa single perturbations: target recovered at top k of Fig. 3 H/I's full
                                    candidates
    figS5b_singles_gain_by_screen   (A) per screen, side by side
    figS5c_doubles_loss             CRISPRi / Cas13d: both genes recovered at top k, in-panel doubles, candidates =
                                    genes in the screen's own doubles
    figS5d_doubles_loss_by_screen   (C) per screen
    figS5e_doubles_gain             CRISPRa, the same
    figS5f_doubles_gain_by_screen   (E) per screen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig3_doubles_plot as P3  # noqa: E402
import fig4_pb_plot as fp  # noqa: E402

PANELS = {'loss': 'c', 'gain': 'e'}
SCREENS = {'loss': [('gse146194', 'CRISPRi'), ('carpoolseq', 'Cas13d (CaRPool-seq)')],
           'gain': [('norman2019', 'CRISPRa (Norman et al.)'), ('properseq', 'CRISPRa (ProPer-seq)')]}


def by_screen(C, meta, mod, subset, metric, out, name):
    """One summary panel's curves split by screen, side by side; each screen stops at its own candidate count."""
    screens = SCREENS[mod]
    fig, axes = P3.plt.subplots(1, len(screens), figsize=(5.2 * len(screens) + 1.6, 4.6), squeeze=False)
    rows = []
    for j, (ax, (key, title)) in enumerate(zip(axes[0], screens)):
        sub = C[(C.group == key) & (C.denom == subset) & (C.metric == metric)]
        sub = sub[sub.k <= meta[key]['n_cand']]
        if metric == 'both':
            sub = sub[sub.k >= 2]
        lab_rows = []
        for label, col, ls, lw in P3.ARMS:
            g = sub[sub.method == label].sort_values('k')
            if g.empty:
                continue
            ax.plot(g.k, g.both, ls, color=col, lw=lw, marker='o', ms=3.5, clip_on=False, zorder=3)
            lab_rows.append((P3.SHORT.get(label, label), float(g.k.iloc[-1]), float(g.both.iloc[-1]), col))
            rows.append(pd.DataFrame(dict(screen=key, modality=mod, subset=subset, metric=metric, method=label,
                                          k=g.k, recovered=g.both, n=g.n)))
        ax.set_xscale('log')
        ax.set_xticks(sorted({t for t in (1, 2, 5, 10, 50, 500) if sub.k.min() <= t <= sub.k.max()}
                             | {int(sub.k.max())}))
        ax.get_xaxis().set_major_formatter(P3.matplotlib.ticker.ScalarFormatter())
        ymax = float(sub.both.max())
        ax.set_ylim(-ymax * 0.045, min(ymax * 1.18, 1.05))         # a fraction: no headroom past 1
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= 1 + 1e-9])
        ax.set_xlabel(f"top k  (of {meta[key]['n_cand']:,} candidates)", fontsize=13)
        what = {'both': 'both genes recovered', 'single': 'target recovered'}[metric]
        ax.set_ylabel(f"{what}\n(n = {int(sub.n.iloc[0])})", fontsize=13)
        ax.set_title(title, fontsize=13, color=fp.VS.INK, pad=8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if j == len(screens) - 1:                     # arms are labelled once, on the rightmost screen
            P3.end_labels_floored(ax, lab_rows, gap=0.13, fs=11)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.17, wspace=0.32)
    fp.save(fig, out, name, src=pd.concat(rows), d=None)


def main(a):
    fp.setup(a.vizstyle_root)
    P3.plt, P3.matplotlib = fp.plt, fp.matplotlib
    d = Path(a.data_dir)
    fp.SRC_DIR = str(d)
    C = pd.read_csv(d / 'figS5_doubles_curves.csv')
    meta = json.load(open(d / 'figS5_meta.json'))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for mod, letter in PANELS.items():
        P3.panel(C, meta, mod, 'in_panel', 'both', a.out_dir, '', name=f'figS5{letter}_doubles_{mod}')
        by_screen(C, meta, mod, 'in_panel', 'both', a.out_dir,
                  f'figS5{chr(ord(letter) + 1)}_doubles_{mod}_by_screen')
    Cs = pd.read_csv(d / 'figS5_singles_curves.csv')
    smeta = json.load(open(d / 'figS5_singles_meta.json'))
    P3.panel(Cs, smeta, 'gain', 'scorable', 'single', a.out_dir, '', name='figS5a_singles_gain')
    by_screen(Cs, smeta, 'gain', 'scorable', 'single', a.out_dir, 'figS5b_singles_gain_by_screen')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True,
                    help='paper/figures/si/figS5/data (figS5_data.py output)')
    ap.add_argument('--out-dir', required=True, help='si/figS5')
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    main(ap.parse_args())
