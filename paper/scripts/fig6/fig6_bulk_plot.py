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

"""Fig. 6G/H from fig6_bulk.py output (--analysis-dir; canonical input data/mito_bulk/analysis_repro_collab).

    f2  Fig. 6G: PCA of the sorted-cell bulk per cell type, 95% ellipses per group (panel_pca_ellipses)
    g2  Fig. 6H: the disease PC's scores per group (panel_pc_scores; --pc-rule picks the PC)

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_bulk_plot.py --analysis-dir ... --out-dir ... --src-dir ... --vizstyle-root ...
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
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
from matplotlib.lines import Line2D  # noqa: E402

CTS = [('CD4', 'CD4 T'), ('CD8', 'CD8 T'), ('Mono', 'Mono')]
W = 7.0


def save(fig, out, name):
    VS.save(fig, out, name)


def colours():
    return {'mito': VS.C['blue'], 'control': '#a9a8a2'}


SRC = None  # source-table dir (--src-dir), set in main


def _ellipse(ax, P, col, q=5.991):
    """95% normal-theory ellipse of 2-D points (chi2, 2 dof)."""
    from matplotlib.patches import Ellipse
    if len(P) < 3:
        return
    C = np.cov(P.T)
    w, v = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    ax.add_patch(Ellipse(P.mean(0), 2 * np.sqrt(q * w[-1]), 2 * np.sqrt(q * w[0]), angle=ang, facecolor=col,
                         alpha=0.10, edgecolor=col, lw=1.2, zorder=0))


def _disease_pc(R, ct, rule='max'):
    """(stimulation PC, disease PC) among PC1 / PC2. rule 'max': each the PC carrying more of its covariate's r^2
    (can be the same PC); 'other' (the collaborators' 261022_pca_minimal.R): the stimulation PC as above, the disease
    PC = the other one."""
    x = R[(R.cell_type == ct) & R.pc.isin(['PC1', 'PC2'])]
    pick = lambda cov: x[x.covariate == cov].set_index('pc').r2.idxmax()
    s_ = pick('stim')
    return s_, ({'PC1': 'PC2', 'PC2': 'PC1'}[s_] if rule == 'other' else pick('group'))


def panel_pca_ellipses(A, out, name, star='P67'):
    """PC1-PC2 per cell type, 95% ellipse per group (PMD blue, control grey), stimulated filled / unstimulated
    hollow, donors named, `star` (the reclassified patient PT67, starred as everywhere else in Fig. 6) marked."""
    P = pd.read_csv(A / 'bulk_pca.csv')
    cc = colours()
    fig, axs = plt.subplots(1, len(CTS), figsize=(W + 2.0, (W + 2.0) / len(CTS) + 0.7))
    for ax, (ct, lab) in zip(axs, CTS):
        x = P[P.cell_type == ct]
        for g in ('control', 'mito'):
            _ellipse(ax, x[x.group == g][['PC1', 'PC2']].to_numpy(), cc[g])
        for r in x.itertuples():
            filled = r.stim == 'stim'
            ax.scatter(r.PC1, r.PC2, s=46, facecolor=cc[r.group] if filled else 'white', edgecolor=cc[r.group],
                       lw=1.4, zorder=3)
            ax.annotate(r.patient + (' ★' if r.patient == star else ''), (r.PC1, r.PC2), xytext=(3, 3),
                        textcoords='offset points', fontsize=VS.FONT_MIN - 5, color=cc[r.group], zorder=4,
                        fontweight='bold' if r.patient == star else 'normal')
        v = x.iloc[0]
        ax.set_xlabel(f'PC1 ({100 * v.var1:.0f}%)'); ax.set_ylabel(f'PC2 ({100 * v.var2:.0f}%)')
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.grid(False)
        ax.set_title(lab, loc='left')
    h = [Line2D([], [], ls='', marker='o', ms=8, color=cc['mito'], label='PMD'),
         Line2D([], [], ls='', marker='o', ms=8, color=cc['control'], label='control'),
         Line2D([], [], ls='', marker='o', ms=8, mfc='white', mec=VS.INK2, mew=1.4, label='unstimulated'),
         Line2D([], [], ls='', marker='o', ms=8, color=VS.INK2, label='stimulated')]
    fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.4)
    save(fig, out, name)
    P.to_csv(SRC / f'{name}_source.csv', index=False)


def panel_pc_scores(A, out, name, rule='max'):
    """Top row: the stimulation PC (each donor's unstimulated -> stimulated joined); bottom row: the disease PC
    (control vs PMD, group mean); r^2 of the covariate with that PC in the corner. PC = PC1 or PC2, whichever carries
    more of the covariate."""
    P = pd.read_csv(A / 'bulk_pca.csv')
    R = pd.read_csv(A / 'bulk_pc_r2.csv')
    cc = colours()
    fig, axs = plt.subplots(2, len(CTS), figsize=(W + 2.0, W * 0.75), gridspec_kw={'hspace': 0.45, 'wspace': 0.45})
    rng = np.random.default_rng(0)
    src = []
    for j, (ct, lab) in enumerate(CTS):
        x = P[P.cell_type == ct].copy()
        pc_s, pc_g = _disease_pc(R, ct, rule)
        # PC signs are arbitrary: orient stimulation up and PMD up (a sign flip changes no r^2)
        if x[x.stim == 'stim'][pc_s].mean() < x[x.stim == 'unstim'][pc_s].mean():
            x[pc_s] = -x[pc_s]
        if pc_g != pc_s and x[x.group == 'mito'][pc_g].mean() < x[x.group == 'control'][pc_g].mean():
            x[pc_g] = -x[pc_g]
        r2 = lambda cov, pc: R[(R.cell_type == ct) & (R.covariate == cov) & (R.pc == pc)].r2.iloc[0]
        var = lambda pc: x.iloc[0][f'var{pc[2:]}']
        ax = axs[0, j]
        for p_, y in x.groupby('patient'):
            y = y.set_index('stim')
            if {'unstim', 'stim'} <= set(y.index):
                ax.plot([0, 1], [y.loc['unstim', pc_s], y.loc['stim', pc_s]], color=VS.MUTED, lw=0.9, zorder=1)
        for r in x.itertuples():
            k = 0 if r.stim == 'unstim' else 1
            ax.scatter(k, getattr(r, pc_s), s=40, facecolor=cc[r.group] if k else 'white', edgecolor=cc[r.group],
                       lw=1.3, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['unstim', 'stim'])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(f'{pc_s} ({100 * var(pc_s):.0f}%)')
        ax.text(0.03, 0.97, f'r² = {r2("stim", pc_s):.2f}', transform=ax.transAxes, ha='left', va='top',
                fontsize=VS.FONT_MIN - 3, color=VS.INK2)
        ax.set_title(lab, loc='left')
        ax = axs[1, j]
        for k, g in enumerate(('control', 'mito')):
            y = x[x.group == g]
            xs = k + rng.uniform(-0.1, 0.1, len(y))
            for xx, r in zip(xs, y.itertuples()):
                ax.scatter(xx, getattr(r, pc_g), s=40, facecolor=cc[g] if r.stim == 'stim' else 'white',
                           edgecolor=cc[g], lw=1.3, zorder=3)
            ax.plot([k - 0.25, k + 0.25], [y[pc_g].mean()] * 2, color=cc[g], lw=2.2, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['control', 'PMD'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(f'{pc_g} ({100 * var(pc_g):.0f}%)')
        ax.text(0.03, 0.97, f'r² = {r2("group", pc_g):.2f}', transform=ax.transAxes, ha='left', va='top',
                fontsize=VS.FONT_MIN - 3, color=VS.INK2)
        for a_ in axs[:, j]:
            for s_ in ('top', 'right'):
                a_.spines[s_].set_visible(False)
            a_.grid(False)
        src.append(dict(cell_type=ct, stim_pc=pc_s, stim_r2=r2('stim', pc_s), disease_pc=pc_g,
                        disease_r2=r2('group', pc_g)))
    axs[0, 0].annotate('stimulation PC', xy=(-0.55, 0.5), xycoords='axes fraction', rotation=90, ha='center',
                       va='center', fontweight='bold', color=VS.INK)
    axs[1, 0].annotate('disease PC', xy=(-0.55, 0.5), xycoords='axes fraction', rotation=90, ha='center',
                       va='center', fontweight='bold', color=VS.INK)
    save(fig, out, name)
    pd.DataFrame(src).to_csv(SRC / f'{name}_source.csv', index=False)
    print(f'[{name}] ' + '; '.join(f'{r["cell_type"]}: stim {r["stim_pc"]} {r["stim_r2"]:.2f}, disease {r["disease_pc"]} '
                                   f'{r["disease_r2"]:.2f}' for r in src), flush=True)


def main(a):
    global VS
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    VS.apply()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    global SRC
    A, out = Path(a.analysis_dir), Path(a.out_dir)
    SRC = Path(a.src_dir); SRC.mkdir(parents=True, exist_ok=True)
    want = set(a.panels.split('+'))
    if 'f2' in want:
        panel_pca_ellipses(A, out, a.f_name)
    if 'g2' in want:
        panel_pc_scores(A, out, a.g_name, a.pc_rule)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analysis-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', default='f2+g2', help='f2 = Fig. 6G (PCA ellipses), g2 = Fig. 6H (PC scores). Canonical input: '
                    'analysis_repro_collab (see paper/slurm/fig6_bulk.sbatch)')
    ap.add_argument('--pc-rule', default='other', choices=['max', 'other'], help='g2: disease PC choice (see _disease_pc)')
    ap.add_argument('--f-name', default='fig6g_bulk_pca')
    ap.add_argument('--g-name', default='fig6h_bulk_pc_scores')
    return ap


if __name__ == '__main__':
    main(build_parser().parse_args())
