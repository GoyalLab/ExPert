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

"""Fig 5i: hepatocyte poly(A) RNA-FISH, nuclear vs cytoplasmic foci per frame (healthy / AC / AH).

Input: the per-frame table (file, frame_id, group, nuclear_total_ratio = nuclear foci / all foci, 90th-percentile puncta
cutoff applied upstream). Every focus is nuclear or cytoplasmic, so cytoplasmic = 1 - nuclear and the single metric
carrying both is N/C = r / (1 - r). Box + jittered frames:
    fig5i_fish_nc        log2 N/C on a linear axis, healthy vs ALD (healthy / ALD / AC / AH: Supplementary Fig. S12F)
ALD = AC and AH frames pooled. Brackets: two-sided Mann-Whitney U for healthy-AC, AC-AH, healthy-AH and healthy-ALD,
BH-adjusted over those 4 (rank test: identical p for both versions).
Frames are the replicates here (donor of each frame not recorded) -- state that in the legend.

    python paper/scripts/fig5/fig5i_fish.py --frames .../nuc_puncta_frame_summary_master_raw.csv --fig-dir ... --src-dir ... --vizstyle-root ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ALD_C = '#b2182b'   # Fig 5-wide ALD colour (fig5_final.ALD_C); the box outline, dots keep AC / AH
ORDER = [('Healthy', 'healthy'), ('ALD', 'ALD'), ('AC', 'AC'), ('AH', 'AH')]   # ALD = AC + AH frames pooled
POS = [0, 1, 2.2, 3.2]                                                            # AC / AH set apart from ALD
PAIRS = [('Healthy', 'ALD'), ('AC', 'AH'), ('Healthy', 'AC'), ('Healthy', 'AH')]   # tested + bracket order


CC = {'healthy': '#8a8983', 'AC': '#e57ea8', 'AH': '#7a2f72'}   # fig5_final AC_C / AH_C; ALD: open box
YLAB = 'log2 nuclear : cytoplasmic\npoly(A) foci'
MAIN = (ORDER[:2], POS[:2], PAIRS[:1])                               # main: healthy vs ALD only
PANELS = [('log2nc', YLAB, 'fig5i_fish_nc', MAIN)]


def stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def prepare(frames):
    """frames table -> (x with N/C columns and the pooled ALD group, Mann-Whitney tests with BH, per-group summary)"""
    x = pd.read_csv(frames)
    x['cond'] = x.group
    x = pd.concat([x, x[x.group.isin(['AC', 'AH'])].assign(group='ALD')], ignore_index=True)
    x['nc'] = x.nuclear_total_ratio / (1 - x.nuclear_total_ratio)
    x['fold'] = x.nc / x[x.group == 'Healthy'].nc.median()
    x['nuc_fold'] = x.nuclear_total_ratio / x[x.group == 'Healthy'].nuclear_total_ratio.median()
    x['log2nc'] = np.log2(x.nc)                                         # main panel: log2(N/C) on a linear axis
    pairs = PAIRS
    raw_p = [mannwhitneyu(x[x.group == g1].nc, x[x.group == g2].nc, alternative='two-sided').pvalue for g1, g2 in pairs]
    tests = pd.DataFrame(pairs, columns=['group1', 'group2']).assign(p=raw_p, p_bh=bh(raw_p))
    summ = x.groupby('group').agg(n_frames=('nc', 'size'), nuclear_frac_median=('nuclear_total_ratio', 'median'),
                                  nc_median=('nc', 'median'), fold_median=('fold', 'median')).reindex([g for g, _ in ORDER])
    return x, tests, summ


def box_panel(VS, plt, x, tests, col, ylab, order, pos, pairs_, cc, rng):
    """one box + frame-dots panel with BH-adjusted Mann-Whitney brackets; returns the figure (caller saves)"""
    logy = col == 'fold'                                       # fold: log axis, powers of 2
    fig, ax = plt.subplots(figsize=(2.4 if len(order) == 2 else 3.9, 5.3))
    data = [x[x.group == g][col].to_numpy() for g, _ in order]
    dcol = [x[x.group == g].cond.map({'Healthy': cc['healthy'], 'AC': cc['AC'], 'AH': cc['AH']}).to_numpy()
            for g, _ in order]
    bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False, patch_artist=True, zorder=2,
                    medianprops=dict(color=VS.INK, lw=1.8), whiskerprops=dict(color=VS.INK2, lw=1.2),
                    capprops=dict(color=VS.INK2, lw=1.2), boxprops=dict(lw=1.2, edgecolor=VS.INK2))
    for patch, (_, ck) in zip(bp['boxes'], order):
        if ck == 'ALD':
            patch.set_facecolor('none'); patch.set_edgecolor(ALD_C); patch.set_linewidth(1.8)
        else:
            patch.set_facecolor(cc[ck]); patch.set_alpha(0.35)
    for i, v, c_ in zip(pos, data, dcol):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=14, c=c_, edgecolor=VS.SURFACE, lw=0.4, zorder=3)
    if logy:
        ax.set_yscale('log')
    if col == 'nuc_fold':
        ax.axhline(1, color=VS.INK2, ls='--', lw=1.0, zorder=1)
        tk = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    elif col == 'fold':
        ax.axhline(1, color=VS.INK2, ls='--', lw=1.0, zorder=1)
        tk = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    elif col == 'log2nc':
        tk = list(range(-1, 9))                                    # whole log2 steps, evenly spaced
    else:
        tk = [0, 50, 100, 150, 200]                                # raw N/C, linear
    lo, hi = x[col].min(), x[col].max()
    keep = [t for t in tk if (t == 0 and col == 'nc') or lo - 0.3 * abs(lo) - 0.5 <= t <= hi * 1.3 + 0.5]
    ax.set_yticks(keep)
    ax.set_yticklabels([(f'{t:.1f}' if col == 'nuc_fold' else f'{t:g}'.replace('-', '−')) for t in keep])
    ax.minorticks_off()
    # significance brackets above the data, stacked in log space
    T_ = np.log10 if logy else (lambda v: v)
    I_ = (lambda v: 10 ** v) if logy else (lambda v: v)
    top, step = T_(hi), 0.085 * (T_(hi) - T_(lo))
    order_br = pairs_
    idx = {g: i for i, (g, _) in zip(pos, order)}
    for k, (g1, g2) in enumerate(order_br):
        r_ = tests[((tests.group1 == g1) & (tests.group2 == g2)) | ((tests.group1 == g2) & (tests.group2 == g1))]
        y = I_(top + step * (k + 0.35))
        h = I_(top + step * (k + 0.35) + 0.2 * step)
        i1, i2 = idx[g1], idx[g2]
        ax.plot([i1, i1, i2, i2], [y, h, h, y], color=VS.INK, lw=1.0, clip_on=False)
        ax.text((i1 + i2) / 2, h, stars(r_.p_bh.iloc[0]), ha='center', va='bottom', color=VS.INK)
    ax.set_ylim(I_(T_(lo) - 0.05 * (top - T_(lo))), I_(top + step * (len(pairs_) + 0.4)))
    ax.set_xticks(pos); ax.set_xticklabels([ck for _, ck in order])
    ax.set_xlim(-0.6, pos[-1] + 0.6)
    ax.set_ylabel(ylab)
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    return fig


def render(VS, plt, x, tests, col, ylab, name, layout, out, rng, cc=CC):
    """draw one box panel and save it (pdf + svg + png)"""
    order, pos, pairs_ = layout
    fig = box_panel(VS, plt, x, tests, col, ylab, order, pos, pairs_, cc, rng)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{name}.svg')
    plt.close(fig)
    print('wrote', name)


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    VS.apply()
    VS.SURFACE = 'white'
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                                'savefig.facecolor': 'white'})
    out = Path(a.fig_dir)
    Path(a.src_dir).mkdir(parents=True, exist_ok=True)
    x, tests, summ = prepare(a.frames)
    tests.to_csv(Path(a.src_dir) / 'fig5i_fish_tests.csv', index=False)
    summ.to_csv(Path(a.src_dir) / 'fig5i_fish_summary.csv')
    x[x.group != 'ALD'].drop(columns='cond').to_csv(Path(a.src_dir) / 'fig5i_fish_source.csv', index=False)
    print(summ.round(3).to_string()); print(tests.to_string(index=False))
    rng = np.random.default_rng(0)
    for col, ylab, name, layout in PANELS:
        render(VS, plt, x, tests, col, ylab, name, layout, out, rng)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True, help='nuc_puncta_frame_summary_master_raw.csv')
    ap.add_argument('--fig-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    main(ap.parse_args())
