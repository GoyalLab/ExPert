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

"""Fig 5j: hepatocyte nuclear morphometry (DAPI, Otsu) per nucleus, healthy / ALD (AC + AH pooled).

Box + every nucleus as a dot (as 5i FISH), the per-sample medians as larger open diamonds. The replicate is the SAMPLE
(liver), not the nucleus: brackets = two-sided Mann-Whitney U on per-sample medians (healthy 3 vs ALD 6; AC vs AH 3 vs 3).
    fig5j_nuclei_diameter                   average diameter, healthy vs ALD (healthy / ALD / AC / AH: Supplementary
                                            Fig. S12G); fig5j_nuclei_tests.csv has the tests for every metric

    python paper/scripts/fig5/fig5j_nuclei.py --measurements .../NucleiMeasurements.csv --fig-dir ... --src-dir ... --vizstyle-root ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ALD_C, AC_C, AH_C, HEALTHY_C = '#b2182b', '#e57ea8', '#7a2f72', '#8a8983'   # fig5_final colours
METRICS = {'average_diameter': 'nuclear diameter (µm)', 'nucleus_area': 'nuclear area (µm²)',
           'major_diameter': 'major diameter (µm)', 'minor_diameter': 'minor diameter (µm)',
           'compactness_circularity': 'circularity'}


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


def sample_tests(x, metric, pairs):
    """two-sided Mann-Whitney U on per-sample medians; returns (per-sample medians, tests)"""
    med = x.groupby(['grp', 'sample_id', 'group'])[metric].median().reset_index()
    tests = []
    for g1, g2 in pairs:
        a_, b_ = med[med.grp == g1][metric], med[med.grp == g2][metric]
        tests.append(dict(metric=metric, group1=g1, group2=g2, n1=len(a_), n2=len(b_),
                          p=mannwhitneyu(a_, b_, alternative='two-sided').pvalue))
    return med, tests


def plot(VS, plt, x, metric, groups, pos, pairs, od, nm, pair_mode, medians=True):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.4 if len(groups) == 2 else 3.9, 5.3))
    col = {'Healthy': HEALTHY_C, 'AC': AC_C, 'AH': AH_C}
    med, tests = sample_tests(x, metric, pairs)
    for xp, g in zip(pos, groups):
        sub = x[x.grp == g]
        bp = ax.boxplot([sub[metric]], positions=[xp], widths=0.6, showfliers=False, patch_artist=True, zorder=2,
                        medianprops=dict(color=VS.INK, lw=1.8), whiskerprops=dict(color=VS.INK2, lw=1.2),
                        capprops=dict(color=VS.INK2, lw=1.2), boxprops=dict(lw=1.2, edgecolor=VS.INK2))
        b = bp['boxes'][0]
        if g == 'ALD':
            b.set_facecolor('none'); b.set_edgecolor(ALD_C); b.set_linewidth(1.8)
        else:
            b.set_facecolor(col[g]); b.set_alpha(0.35)
        dc = [col[c] for c in sub.group]                  # ALD dots keep their AC / AH colour (as 5i)
        ax.scatter(xp + rng.uniform(-0.17, 0.17, len(sub)), sub[metric], s=14, c=dc, edgecolor=VS.SURFACE, lw=0.4,
                   zorder=3)
        m_ = med[med.grp == g]
        if medians:
            ax.scatter(np.full(len(m_), xp + 0.38), m_[metric], s=46, marker='D', facecolor='none', lw=1.3,
                       zorder=4, edgecolor=[col[c] for c in m_.group])
    lo, hi = x[metric].min(), x[metric].max()
    step = 0.085 * (hi - lo)
    idx = dict(zip(groups, pos))
    for k, t in enumerate(tests):
        y = hi + step * (k + 0.35); h = y + 0.2 * step
        i1, i2 = idx[t['group1']], idx[t['group2']]
        ax.plot([i1, i1, i2, i2], [y, h, h, y], color=VS.INK, lw=1.0, clip_on=False)
        ax.text((i1 + i2) / 2, h, stars(t['p']), ha='center', va='bottom', color=VS.INK)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + step * (len(tests) + 0.6))
    ax.set_xticks(pos); ax.set_xticklabels(['healthy' if g == 'Healthy' else g for g in groups])
    ax.set_xlim(-0.6, pos[-1] + (0.75 if medians else 0.6))
    ax.set_ylabel(METRICS[metric])
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(od / f'{nm}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(od / f'{nm}.svg')
    plt.close(fig)
    print('wrote', nm)
    return tests


def load(measurements):
    """nucleus table (segmented, status OK) with grp = condition, plus AC + AH again as grp 'ALD'"""
    m = pd.read_csv(measurements)
    m = m[(m.row_type == 'Nucleus') & (m.status == 'OK')]
    return pd.concat([m.assign(grp=m.group), m[m.group.isin(['AC', 'AH'])].assign(grp='ALD')], ignore_index=True)


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
    x = load(a.measurements)
    x[x.grp != 'ALD'].to_csv(Path(a.src_dir) / 'fig5j_nuclei_source.csv', index=False)
    tests = plot(VS, plt, x, 'average_diameter', ['Healthy', 'ALD'], [0, 1], [('Healthy', 'ALD')], out,
                 'fig5j_nuclei_diameter', True, medians=False)
    for met in METRICS:
        tests += sample_tests(x, met, [('Healthy', 'ALD'), ('AC', 'AH')])[1]
    T = pd.DataFrame(tests).drop_duplicates(['metric', 'group1', 'group2'])
    T.to_csv(Path(a.src_dir) / 'fig5j_nuclei_tests.csv', index=False)
    print(T.round(4).to_string(index=False))
    print(x.groupby('grp').agg(samples=('sample_id', 'nunique'), nuclei=('sample_id', 'size'),
                               diam_median=('average_diameter', 'median')).round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--measurements', required=True, help='NuclearMorphometry/NucleiMeasurements.csv')
    ap.add_argument('--fig-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    main(ap.parse_args())
