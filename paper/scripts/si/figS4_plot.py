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

"""Supplementary Fig. S4 (Fingerprint's cell assignment, related to Fig. 2). Reads ONLY --data-dir
(paper/figures/si/figS4/data,
figS4_data.py); styling is Fig. 2's (fig2_plot.py helpers). One file per panel in --out-dir (si/figS4).

    figS4b_coverage          top-1 accuracy and recall@10 vs coverage (most confident cells kept), ID and OOD;
                             Fingerprint (ensemble) dotted past its call rate (dashed line)
    figS4c_called_vs_all     every method on all cells vs on the cells Fingerprint (ensemble) called
    figS4a_abstention        Fingerprint (ensemble): share of cells with no call, per dataset, ID and OOD
"""
from __future__ import annotations

import argparse
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig2_plot as P2  # noqa: E402

FP = 'fingerprint_perdataset'


def save(fig, out, name):
    P2.bold_expert(fig)
    P2.VS.save(fig, out, name)
    P2.assert_single_text_svg(Path(out) / f'{name}.svg')


def panel_a(d, out):
    C = pd.read_csv(d / 'figS4b_coverage.csv')
    meths = P2.ordered(C.method)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True,
                             gridspec_kw=dict(wspace=0.22, hspace=0.18))
    for i, reg in enumerate(('id', 'ood')):
        for j, (col, lab) in enumerate((('accuracy', 'top-1 accuracy'), ('recall10', 'top-10 recall'))):
            ax = axes[i, j]
            for m in reversed(meths):
                g = C[(C.method == m) & (C.regime == reg)].sort_values('coverage')
                if g.empty:
                    continue
                st = P2.style(m)
                cr_ = g.call_rate.iloc[0]
                if cr_ < 0.999:                       # past the call rate the cells are abstentions
                    inn, past = g[g.coverage <= cr_ + 1e-9], g[g.coverage >= cr_ - 0.05]
                    ax.plot(inn.coverage, inn[col], **st)
                    ax.plot(past.coverage, past[col], **{**st, 'ls': ':'})
                    ax.axvline(cr_, color=st['color'], lw=1, ls='--', alpha=0.7)
                else:
                    ax.plot(g.coverage, g[col], **st)
            ax.set_ylim(bottom=0); ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0%}'))
            ax.set_title(f'{reg.upper()} {lab}')
            if i == 1:
                ax.set_xlabel('coverage (most confident cells kept)')
    fig.tight_layout()
    P2.method_legend(fig, meths, ncol=4, y=0.0)
    save(fig, out, 'figS4b_coverage')


def panel_b(d, out):
    PAL, NAME, VS = P2.PAL, P2.NAME, P2.VS
    M = pd.read_csv(d / 'figS4c_called_vs_all.csv')
    mets = ['F1', 'recall@10', 'MRR']
    meths = P2.ordered(M.method)
    fig, axes = plt.subplots(2, 3, figsize=(16, 0.5 * len(meths) * 2 + 3), sharey=True,
                             gridspec_kw=dict(wspace=0.12, hspace=0.25))
    y = np.arange(len(meths))
    for i, reg in enumerate(('id', 'ood')):
        for j, met in enumerate(mets):
            ax = axes[i, j]
            t = M[M.regime == reg].pivot_table(index='method', columns='subset', values=met).reindex(meths)
            for yy, m in enumerate(meths):
                a, b = t.loc[m, 'all cells'], t.loc[m, 'fingerprint-called cells']
                ax.plot([a, b], [yy, yy], color=PAL[m], lw=3.6 if m == 'ExPert' else 2, zorder=3)
                ax.plot(a, yy, 'o', ms=10, mfc='white', mec=PAL[m], mew=2, zorder=4)
                ax.plot(b, yy, 'o', ms=10, mfc=PAL[m], mec=VS.INK2, mew=0.8, zorder=5)
            ax.set_xlim(left=0); ax.grid(axis='y', visible=False)
            ax.set_title(f'{reg.upper()} ' + {'recall@10': 'top-10 recall'}.get(met, met))
    axes[0, 0].set_yticks(y); axes[0, 0].set_yticklabels([NAME[m] for m in meths]); axes[0, 0].invert_yaxis()
    fig.legend([Line2D([], [], marker='o', ls='', ms=10, mfc='white', mec=VS.INK2, mew=2),
                Line2D([], [], marker='o', ls='', ms=10, mfc=VS.INK2, mec=VS.INK2)],
               ['all cells', 'cells Fingerprint (ensemble) called'],
               loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2)
    save(fig, out, 'figS4c_called_vs_all')


def panel_c(d, out):
    PAL, VS = P2.PAL, P2.VS
    A = pd.read_csv(d / 'figS4a_abstention.csv')
    ood = A[A.regime == 'ood'].set_index('dataset').abstain_cells.sort_values()
    idr = A[A.regime == 'id'].set_index('dataset').abstain_cells.reindex(ood.index)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ood) + 1.8))
    y = np.arange(len(ood))
    ax.barh(y, ood.values, color=PAL[FP], edgecolor=VS.INK2, lw=0.5, label='held out (OOD)')
    ax.plot(idr.values, y, 'o', ms=8, mfc='white', mec=VS.INK, mew=1.6, ls='', label='in training (ID)')
    ax.set_yticks(y); ax.set_yticklabels(ood.index); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, 1); ax.set_xlabel('cells with no fingerprint call')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.legend(loc='lower right')
    save(fig, out, 'figS4a_abstention')


PANELS = {'a': panel_a, 'b': panel_b, 'c': panel_c}


def main(a):
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P2.setup(d, a.vizstyle_root, palette='figS4_palette.csv')
    for k in a.panels:
        PANELS[k](d, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True,
                    help='paper/figures/si/figS4/data (figS4_data.py output)')
    ap.add_argument('--out-dir', required=True, help='si/figS4')
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
