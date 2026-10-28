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

"""Supplementary Fig. S3 (benchmark, related to Fig. 2). Reads ONLY --data-dir
(paper/figures/si/figS3/data, figS3_data.py);
styling is Fig. 2's (fig2_plot.py helpers). One file per panel in --out-dir (si/figS3).

    figS3a_ood_condition             OOD on held-out conditions, pseudobulk (cell lines and donors: Fig. 2C)
    figS3b_increase_over_next_best   per fold: ExPert / next-best method, top-10 recall, ID and OOD
    figS3c_target_mask               own target gene masked in the query, every ctx7 perturbation, pseudobulk
    figS3d_ood_by_shift_cell         Figure 2C on single cells (effect-size weighted), by held-out context
    figS3e_subsample_control         effect-size threshold vs a random sample of the same number of cells

Fingerprint's no-calls: Fig. S4.
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
from matplotlib.lines import Line2D  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig2_common as F  # noqa: E402
import fig2_plot as P2  # noqa: E402



def save(fig, out, name):
    P2.bold_expert(fig)
    P2.VS.save(fig, out, name)
    P2.assert_single_text_svg(Path(out) / f'{name}.svg')


MASK_ROWS = [('ExPert', 'ExPert'), ('fingerprint_perdataset', 'Fingerprint (ensemble)'),
             ('fingerprint_merged', 'Fingerprint (merged)'),
             ('baseline_centroid', 'Centroid (reads target)'),
             ('baseline_centroid_notarget', 'Centroid (target-masked)'),
             ('baseline_neg_target', 'Negative target')]


def target_mask(d, out, table='figS3c_target_mask.csv', name='figS3c_target_mask'):
    """Every ctx7 perturbation, one plain / own-target-masked pair per method (fingerprint: published
    scores where the mask cannot reach, the arm's scores on the target-input cells, the ID sample
    reweighted to the full split)."""
    PAL, VS = P2.PAL, P2.VS
    T = pd.read_csv(d / table)
    groups = [('all', 'o', 0.0, 'all perturbations')]
    rows = [(m, lab) for m, lab in MASK_ROWS if m in set(T.method)]
    col = {m: PAL.get(m, PAL.get('baseline_centroid_notarget', '0.4')) for m, _ in rows}
    mets = [('F1', 'F1'), ('R10', 'top-10 recall'), ('MRR', 'MRR')]
    fig, axes = plt.subplots(2, 3, figsize=(17, 0.85 * len(rows) * 2 + 3.5), sharey=True,
                             gridspec_kw=dict(wspace=0.15, hspace=0.25))
    y = np.arange(len(rows))
    for i, reg in enumerate(('id', 'ood')):
        for j, (mc, ml) in enumerate(mets):
            ax = axes[i, j]
            hi = 0
            for cells, mk, dy, _ in groups:
                v = T[(T.regime == reg) & (T.cells == cells)].pivot_table(index='method', columns='arm', values=mc)
                for yy, (m, _) in enumerate(rows):
                    if m not in v.index or not {'plain', 'masked'} <= set(v.columns):
                        continue
                    a, b = v.loc[m, 'plain'], v.loc[m, 'masked']
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    hi = max(hi, a, b)
                    yp = yy + dy
                    ax.plot([a, b], [yp, yp], color=col[m], lw=3.0 if m == 'ExPert' else 1.8, zorder=3)
                    ax.plot(a, yp, mk, ms=10, mfc='white', mec=col[m], mew=2, zorder=4)
                    ax.plot(b, yp, mk, ms=10, mfc=col[m], mec=VS.INK2, mew=0.8, zorder=5)
                    if a > 0:
                        ax.text(max(a, b), yp, f'  {b / a - 1:+.0%}', va='center', ha='left')
            ax.set_xlim(0, hi * 1.35); ax.grid(axis='y', visible=False)
            ax.set_title(f'{reg.upper()} {ml}')
    axes[0, 0].set_yticks(y); axes[0, 0].set_yticklabels([lab for _, lab in rows])
    axes[0, 0].set_ylim(len(rows) - 0.4, -0.6)
    h = [Line2D([], [], marker='o', ls='', ms=10, mfc='white', mec=VS.INK2, mew=2),
         Line2D([], [], marker='o', ls='', ms=10, mfc=VS.INK2, mec=VS.INK2)]
    fig.legend(h, ['plain (open)', 'own target masked (filled)'],
               loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=len(h))
    save(fig, out, name)


def headline(d, out, metric='balanced_weighted_recall@10', pool='all methods', name='figS3b_increase_over_next_best'):
    """Per fold: ExPert / the best OTHER method in that fold (figS3b_increase_over_next_best.csv), ID and OOD side by
    side; a bar from 0 to the median across folds -- the number quoted in the text. Points are coloured by
    which method was next-best in that fold."""
    r = pd.read_csv(d / 'figS3b_increase_over_next_best.csv')
    r = r[(r.pool == pool) & (r.metric == metric)]
    PAL, NAME, VS = P2.PAL, P2.NAME, P2.VS
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    rng = np.random.default_rng(0)
    regs = [g for g in ('id', 'ood') if g in set(r.regime)]
    for i, reg in enumerate(regs):
        g = r[r.regime == reg]
        x = i + rng.uniform(-0.16, 0.16, len(g))
        ax.scatter(x, g.ratio, s=70, c=[PAL.get(m, VS.MUTED) for m in g.next_method],
                   edgecolor=VS.INK2, lw=0.6, zorder=4)
        med = g.ratio.median()
        ax.bar(i, med, width=0.6, color=F.EXPERT_COLOR, alpha=0.35, edgecolor=F.EXPERT_COLOR, lw=1.6,
               zorder=1)                                   # bar from 0 to the median; fold points on top
        ax.text(i + 0.34, med, f'{med:.2f}\u00d7', va='center', ha='left', fontweight='bold',
                color=VS.INK, zorder=6)
    ax.axhline(1, color=VS.MUTED, lw=1.2, ls=':')
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([g.upper() for g in regs])
    ax.set_xlim(-0.6, len(regs) - 0.25)
    ax.set_ylim(0, r.ratio.max() * 1.12)
    ax.set_ylabel(f'ExPert / next-best method\n({P2.MET_LABEL.get(metric, metric)}, per fold)')
    ax.grid(axis='x', visible=False)
    used = [m for m in P2.ordered(r.next_method) if m in set(r.next_method)]
    ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=9, mfc=PAL.get(m, VS.MUTED), mec=VS.INK2,
                              label=NAME.get(m, m)) for m in used],
              title='next-best method', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    save(fig, out, name)


SUB_METHODS = ('ExPert', 'fingerprint_perdataset', 'baseline_centroid_notarget')


def subsample(d, out, name='figS3e_subsample_control'):
    """OOD, per metric: performance on the cells at or above each knockdown-effect-size threshold (solid,
    filled) vs a random sample of the same number of cells (dashed, open), for ExPert and the two strongest
    comparators; mean +- SEM over the matched dataset x fold views."""
    T = pd.read_csv(d / 'figS3e_subsample_control.csv')
    PAL, NAME, VS = P2.PAL, P2.NAME, P2.VS
    mets = [('F1', 'F1'), ('recall@10', 'top-10 recall'), ('MRR', 'MRR')]
    fig, axes = plt.subplots(1, len(mets), figsize=(4.6 * len(mets), 4.6), sharex=True, squeeze=False)
    for ax, (met, lab) in zip(axes[0], mets):
        for m in SUB_METHODS:
            for arm, ls, fc in (('effect size', '-', None), ('random', '--', 'white')):
                g = T[(T.method == m) & (T.arm == arm) & (T.metric == met)].sort_values('eff')
                if g.empty:
                    continue
                ex = m == 'ExPert'
                ax.plot(g.eff, g['mean'], ls=ls, color=PAL[m], lw=3 if ex else 1.8, marker='o', ms=8 if ex else 6,
                        mfc=fc or PAL[m], mec=PAL[m], mew=1.6, zorder=5 if ex else 3)
                ax.fill_between(g.eff, g['mean'] - g['sem'], g['mean'] + g['sem'], color=PAL[m], alpha=0.12, lw=0)
        ax.set_title(lab); ax.set_ylim(bottom=0); ax.set_xticks(sorted(T.eff.unique()))
        ax.set_xlabel('min. effect size')
    axes[0, 0].set_ylabel('OOD single cells')
    h = [Line2D([], [], color=PAL[m], lw=3 if m == 'ExPert' else 1.8, label=NAME.get(m, m)) for m in SUB_METHODS]
    h += [Line2D([], [], color=VS.INK2, ls='-', marker='o', label='cells ≥ threshold'),
          Line2D([], [], color=VS.INK2, ls='--', marker='o', mfc='white', label='random, same number of cells')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.0), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, name)


def main(a):
    d, out = Path(a.data_dir), Path(a.out_dir)
    P2.setup(d, a.vizstyle_root, palette='figS3_palette.csv')
    meta = json.loads((d / 'figS3_meta.json').read_text())
    metrics = [meta['f1'], F.R10, meta['mrr']]
    P2.panel_b(d, out, metrics, keep=('ood_condition',), name='figS3a_ood_condition',
               table='figS3a_ood_condition.csv')                                         # A
    headline(d, out)                                                                     # B
    target_mask(d, out)                                                                  # C
    P2.panel_b(d, out, metrics, keep=F.SHIFTS, name='figS3d_ood_by_shift_cell',
               table='figS3d_ood_by_shift_cell.csv')                                     # D
    subsample(d, out)                                                                    # E


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True,
                    help='paper/figures/si/figS3/data (figS3_data.py output)')
    ap.add_argument('--out-dir', required=True, help='canonical Figure S3 panels (si/figS3)')
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    main(ap.parse_args())
