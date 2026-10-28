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

"""Fig. 6E: primary-mitochondrial-disease gene groups in each donor's monocyte profile (fig6_donor_mag.py table).

Tall dot plot. Rows: 'all' (the union of the group rows), the OXPHOS complexes CI-CV (--complexes mitocarta: MitoCarta3.0
subunits + assembly factors, as Fig. 6F; panel: only the PanelApp green genes), the PanelApp CoQ and PDH & TCA groups,
then each patient's suggested lesion gene on its own. Columns: the group-level test (PMD vs reference, as Fig. 6F), then
the four PMD patients, the non-PMD and the undiagnosed donors.
Score (all columns) = the magnitude-aware projection of the donor's shift from one common baseline (the reference
donors' cell-weighted centroid) on the set's mean prototype direction, in units of the reference donors' SD:
    donor column   z = (proj - mean of the reference donors) / their SD; size = one-sided prediction-interval p,
                   t(df = n_ref - 1) at z / sqrt(1 + 1 / n_ref)
    group column   (mean PMD - mean reference) / reference SD; size = exact one-sided donor-label permutation p
                   (fig6_pmd_plot._pmd_vs_nonpmd, scaled), outlined at p < 0.05 as in Fig. 6E
Box = the patient's suggested gene, and the group row holding it.

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_pmd_panel_plot.py --donor-mag ... --panel-dir ... --out-dir ...
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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig6_pmd_panel import CTRL, SUGGESTED  # noqa: E402
from fig6_pmd_plot import _pmd_vs_nonpmd, colors, loss_score_cmap, loss_score_cbar  # noqa: E402

PATIENTS = ['PT54', 'PT46', 'PT43', 'PT67']
OTHERS = [('non-PMD', ['PT41', 'PT42']), ('undiagnosed', ['PT38', 'PT50', 'PT55'])]   # right of the patients
COMPLEXES = ['CI', 'CII', 'CIII', 'CIV', 'CV']
PANEL_ROWS = ['CoQ', 'PDH & TCA']                               # PanelApp groups (fig6_pmd_panel)
P_BINS = [(1e-3, 'p < 0.001', 230), (1e-2, 'p < 0.01', 150), (5e-2, 'p < 0.05', 90), (1.01, 'n.s.', 30)]


def p_size(p):
    return next(s for t, _, s in P_BINS if p < t)


def donor_z(M):
    """{key: DataFrame(donor -> z, p)}: each donor vs the reference donors (CTRL) on the projection."""
    from scipy.stats import t as tdist
    out = {}
    for k, x in M.groupby('key'):
        v = x.set_index('donor').proj
        r = v[[d for d in CTRL if d in v.index]]
        z = (v - r.mean()) / r.std(ddof=1)
        out[k] = pd.DataFrame(dict(z=z, p=tdist.sf(z / np.sqrt(1 + 1 / len(r)), len(r) - 1)))
    return out


def main(a):
    global VS
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    VS.apply()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    import fig6_pmd_plot
    fig6_pmd_plot.VS = VS
    M = pd.read_csv(a.donor_mag)
    M = M[M.celltype == a.celltype].assign(key=lambda x: x.family + ':' + x.set)
    G = pd.read_csv(Path(a.panel_dir) / f'pmd_panel_genes_{a.bank}.csv').set_index('model_symbol').group
    genes = [SUGGESTED[p_] for p_ in PATIENTS]
    rows = [('all', f'all:{a.complexes}')] + [(c, f'{a.complexes}:{c}') for c in COMPLEXES] + \
        [(r, f'panel:{r}') for r in PANEL_ROWS] + [(g, f'gene:{g}') for g in genes]
    n = M.drop_duplicates('key').set_index('key').n_genes
    Mr = M[M.key.isin([k for _, k in rows])]
    DZ = donor_z(Mr)
    GT = _pmd_vs_nonpmd(Mr.rename(columns={'proj': 'z'}).drop(columns=['family']).rename(columns={'key': 'family'})
                        .assign(set=''), one_sided=True, scaled=True).set_index('family')
    # columns: group test, then (after a gap per group) PMD patients, non-PMD, undiagnosed donors
    cols, xs, x_ = [('test', 'PMD vs ref.')], [0.0], 0.0
    present = set(M.donor)                                      # in-vitro groups lack some donors
    for g_, ds in [('PMD', PATIENTS)] + OTHERS:
        ds = [d_ for d_ in ds if d_ in present]
        if not ds:
            continue
        x_ += 0.6
        for d_ in ds:
            x_ += 1
            cols.append((g_, d_)); xs.append(x_)
    xs = np.array(xs)
    ng = len(rows) - len(genes)                                  # group rows; the gene rows follow
    cmap, norm = loss_score_cmap(), TwoSlopeNorm(0, -a.clip, a.clip)
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    src = []
    for i, (lab, key) in enumerate(rows):
        for k, (g_, d_) in enumerate(cols):
            if g_ == 'test':
                if key not in GT.index:
                    continue
                v, p = GT.loc[key, 'effect'], GT.loc[key, 'p']
            else:
                if d_ not in DZ[key].index:
                    continue
                v, p = DZ[key].loc[d_, 'z'], DZ[key].loc[d_, 'p']
            src.append(dict(row=lab, key=key, n_genes=int(n[key]), column=d_, score=v, p=p))
            sig = p < 0.05                                            # outlined, as in Fig. 6F
            ax.scatter(xs[k], i, s=p_size(p), c=[np.clip(v, -a.clip, a.clip)], cmap=cmap, norm=norm, zorder=3,
                       lw=0.7 if sig else 0, edgecolor=VS.INK if sig else 'none')
            gene = SUGGESTED.get(d_)
            if gene and (lab == gene or lab == G.get(gene)):
                ax.scatter(xs[k], i, s=380, marker='s', facecolor='none', edgecolor=VS.INK, lw=1.0, zorder=4)
    labels = [f'{lab} ({int(n[key])})' for lab, key in rows[:ng]] + genes
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=VS.FONT_MIN - 3)
    for tl in ax.get_yticklabels()[ng:]:
        tl.set_fontstyle('italic')
    ax.set_ylim(len(rows) - 0.5, -1.0)
    for y in (0.5, len(COMPLEXES) + 0.5, ng - 0.5):
        ax.axhline(y, color=VS.INK2, lw=0.8)
    # donor names as in Fig. 6D: PMD bold ink, the rest regular grey; condition bars above (the D ring colours)
    ax.set_xticks(xs); ax.set_xticklabels([d_ for _, d_ in cols], rotation=90, fontsize=VS.FONT_MIN - 3)
    ax.xaxis.tick_top()
    for (g_, _), t_ in zip(cols, ax.xaxis.get_major_ticks()):  # top labels are label2
        for tl in (t_.label1, t_.label2):
            tl.set_color(VS.INK if g_ in ('PMD', 'test') else VS.INK2)
            tl.set_fontweight('bold' if g_ in ('PMD', 'test') else 'normal')
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    cc = colors()
    for g_ in ('PMD', 'non-PMD', 'undiagnosed'):
        xg = xs[[k for k, (gg, _) in enumerate(cols) if gg == g_]]
        ax.add_patch(plt.Rectangle((xg[0] - 0.42, -0.95), xg[-1] - xg[0] + 0.84, 0.18, color=cc[g_], clip_on=False))
    ax.axvline((xs[0] + xs[1]) / 2, color=VS.INK2, lw=0.8)      # group test | donors
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.grid(False)
    for i in range(len(rows)):
        ax.axhline(i, color=VS.GRID, lw=0.6, zorder=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.03, pad=0.04, aspect=12)
    loss_score_cbar(cb, a.clip, 'loss score (z vs reference donors)')
    h = [Line2D([], [], ls='', marker='o', ms=np.sqrt(s), mfc=VS.MUTED, mec=VS.INK if s > P_BINS[-1][2] else 'none',
                mew=0.7, label=l) for _, l, s in P_BINS]
    h += [plt.Rectangle((0, 0), 1, 1, color=cc[g_], label=g_) for g_ in ('PMD', 'non-PMD', 'undiagnosed')]
    h.append(Line2D([], [], ls='', marker='s', ms=np.sqrt(380), mfc='none', mec=VS.INK, mew=1.0,
                    label="patient's suggested\nlesion gene / its group"))
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, labelspacing=1.0,
               columnspacing=1.0, fontsize=VS.FONT_MIN - 4)   # below: keeps the panel slim
    out = Path(a.out_dir)
    VS.save(fig, out, a.name)
    Path(a.src_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(src).to_csv(Path(a.src_dir) / f'{a.name}_source.csv', index=False)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--donor-mag', required=True, help='fig6_donor_mag.py table (donor_mag_trained.csv.gz)')
    ap.add_argument('--panel-dir', required=True, help='fig6_pmd_panel.py out dir (gene -> group)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--complexes', default='panel', choices=['mitocarta', 'panel'],
                    help='CI-CV: MitoCarta3.0 structural (as Fig. 6F) or PanelApp green genes only')
    ap.add_argument('--bank', default='trained')
    ap.add_argument('--celltype', default='Monocytes')
    ap.add_argument('--clip', type=float, default=5.0)
    ap.add_argument('--width', type=float, default=6.0)
    ap.add_argument('--height', type=float, default=8.0)
    ap.add_argument('--name', default='fig6e_pmd_genes_panelapp')
    return ap


if __name__ == '__main__':
    main(build_parser().parse_args())
