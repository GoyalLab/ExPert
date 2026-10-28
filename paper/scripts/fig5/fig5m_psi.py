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

"""Fig 5m: RT-PCR exon PSI of the export / splicing genes probed after the ExPert predictions (gel quantification).

Reads the GraphPad Prism file (a zip): every 4-group data sheet (Healthy, AH, AC, NASH; 3 replicates each, values
stored set by set) is one exon. NASH is dropped (same groups as 6g); ALD = the AC + AH replicates pooled. One panel per
exon in a grid: PSI minus the healthy median (0 = healthy), background bar to each group's median,
replicate dots, mean +- SD. Brackets: healthy vs ALD (6 vs 3) and AC vs AH (3 vs 3),
two-sided Welch t-tests, uncorrected. Identical value tables across exons are reported (copy-paste check).

    python paper/scripts/fig5/fig5m_psi.py --prism .../Gel/Qty.prism --fig-dir ... --src-dir ... --vizstyle-root ...
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

ALD_C = '#b2182b'
CC = {'Healthy': '#8a8983', 'AC': '#e57ea8', 'AH': '#7a2f72'}   # fig5_final AC_C / AH_C


def read_prism(path):
    z = zipfile.ZipFile(path)
    rows = []
    for nm in z.namelist():
        if not (nm.startswith('data/sheets/') and nm.endswith('sheet.json')):
            continue
        j = json.loads(z.read(nm))
        if j.get('@class') != 'DataSheet' or j['title'].endswith('-3Grp'):
            continue
        sets = [json.loads(z.read(f'data/sets/{u}.json')).get('title') for u in j['table']['dataSets']]
        if sets != ['Healthy', 'AH', 'AC', 'NASH']:
            continue
        nrep = j['table'].get('replicatesCount', 3)
        vals = pd.read_csv(io.BytesIO(z.read(f"data/tables/{j['table']['uid']}/data.csv")), header=None).values.ravel()
        vals = vals[~pd.isna(vals)].astype(float)
        assert len(vals) == nrep * len(sets), (j['title'], len(vals))
        for k, g in enumerate(sets):
            rows += [dict(exon=j['title'], group=g, rep=r + 1, psi=v) for r, v in enumerate(vals[k * nrep:(k + 1) * nrep])]
    return pd.DataFrame(rows)


def prepare(prism, order):
    """Prism file -> (replicate PSI table without NASH, exons in panel order, Welch tests healthy-ALD and AC-AH)"""
    x = read_prism(prism)
    x = x[x.group != 'NASH']
    exons = list(dict.fromkeys(order)) + sorted(set(x.exon) - set(order))
    exons = [e for e in exons if e in set(x.exon)]
    # copy-paste check: identical replicate tables under two exon names
    sig = {e: tuple(np.round(x[x.exon == e].sort_values(['group', 'rep']).psi.to_numpy(), 6)) for e in exons}
    dup = [(e1, e2) for i, e1 in enumerate(exons) for e2 in exons[i + 1:] if sig[e1] == sig[e2]]
    for e1, e2 in dup:
        print(f'WARNING: {e1} and {e2} hold IDENTICAL values -- check the gel quantification')
    tests = []
    for e in exons:
        h = x[(x.exon == e) & (x.group == 'Healthy')].psi.to_numpy()
        ac, ah = (x[(x.exon == e) & (x.group == g)].psi.to_numpy() for g in ('AC', 'AH'))
        tests += [dict(exon=e, pair='healthy-ALD', g1='Healthy', g2='ALD',
                       p=ttest_ind(np.r_[ac, ah], h, equal_var=False).pvalue),
                  dict(exon=e, pair='AC-AH', g1='AC', g2='AH', p=ttest_ind(ac, ah, equal_var=False).pvalue)]
    return x, exons, pd.DataFrame(tests)


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.'


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
    cc = CC
    out = Path(a.fig_dir)
    Path(a.src_dir).mkdir(parents=True, exist_ok=True)
    x, exons, tests = prepare(a.prism, a.order)
    x.to_csv(Path(a.src_dir) / 'fig5m_psi_source.csv', index=False)
    tests.to_csv(Path(a.src_dir) / 'fig5m_psi_tests.csv', index=False)
    print(tests.pivot(index='exon', columns='pair', values='p').round(4).to_string())
    _grid(VS, plt, x, exons, tests, cc, a, ['Healthy', 'ALD'], [0, 1], 'fig5m_psi', out)    # healthy vs ALD


def _grid(VS, plt, x, exons, tests, cc, a, GROUPS, POS, name, out):
    pair = len(GROUPS) == 2
    nc = a.ncol
    nr = int(np.ceil(len(exons) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=((2.1 if pair else 2.9) * nc, 3.1 * nr), squeeze=False, gridspec_kw={'wspace': 0.45,
                                                                                               'hspace': 0.55})
    rng = np.random.default_rng(0)
    for ax, e in zip(axs.ravel(), exons):
        sub = x[x.exon == e]
        vals = {g: sub[sub.group == g].psi.to_numpy() for g in ('Healthy', 'AC', 'AH')}
        vals['ALD'] = np.r_[vals['AC'], vals['AH']]
        h0 = np.median(vals['Healthy'])                             # centre every exon on the healthy median
        vals = {g: v - h0 for g, v in vals.items()}
        cols = {'Healthy': [cc['Healthy']] * 3, 'AC': [cc['AC']] * 3, 'AH': [cc['AH']] * 3,
                'ALD': [ALD_C] * 6 if pair else [cc['AC']] * 3 + [cc['AH']] * 3}   # main: all ALD dots red
        hi = max(v.max() for v in vals.values()); lo = min(v.min() for v in vals.values())
        span = max(hi, 0) - min(lo, 0)
        for xp, g in zip(POS, GROUPS):
            v = vals[g]
            m, sd = v.mean(), v.std(ddof=1)
            ec = ALD_C if g == 'ALD' else VS.INK
            md = np.median(v)                                       # background bar: 0 (= healthy) -> median
            if g == 'ALD':
                ax.bar(xp, md, width=0.7, color='none', edgecolor=ALD_C, lw=1.4, zorder=1)
            else:
                ax.bar(xp, md, width=0.7, color=cols[g][0], alpha=0.25, lw=0, zorder=1)
            ax.plot([xp - 0.25, xp + 0.25], [m, m], color=ec, lw=2.0, zorder=3)
            ax.plot([xp, xp], [m - sd, m + sd], color=ec, lw=1.2, zorder=3)
            ax.scatter(xp + rng.uniform(-0.12, 0.12, len(v)), v, s=26, c=cols[g], edgecolor=VS.SURFACE, lw=0.5,
                       zorder=4)

        # brackets: healthy vs ALD and AC vs AH, side by side at one height
        for _, r_ in tests[(tests.exon == e) & tests.g2.isin(GROUPS)].iterrows():
            x1, x2 = POS[GROUPS.index(r_.g1)], POS[GROUPS.index(r_.g2)]
            y0 = max(hi, 0) + 0.08 * span; y1 = y0 + 0.04 * span
            ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], color=VS.INK, lw=0.9, clip_on=False)
            ax.text((x1 + x2) / 2, y1, stars(r_.p), ha='center', va='bottom', color=VS.INK,
                    fontsize=VS.FONT_MIN * 0.85)
        ax.set_ylim(min(lo, 0) - 0.08 * span, max(hi, 0) + 0.3 * span)
        ax.axhline(0, color=VS.INK2, lw=0.9, zorder=1)
        ax.set_xticks(POS); ax.set_xticklabels([g.lower() if g == 'Healthy' else g for g in GROUPS], rotation=45, ha='right', rotation_mode='anchor',
                           fontsize=VS.FONT_MIN * 0.85)
        ax.set_xlim(-0.5, POS[-1] + 0.5)
        gene, ex = e.split('_') if '_' in e else (e, '')
        ax.set_title(f'{gene}' + (f' exon {ex}' if ex else ''), fontstyle='italic', color=VS.INK, fontsize=VS.FONT_MIN)
        if list(axs.ravel()).index(ax) % nc == 0:                   # y label on the left column only
            ax.set_ylabel('ΔPSI vs healthy')
        ax.grid(axis='x', visible=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for ax in axs.ravel()[len(exons):]:
        ax.set_visible(False)
    VS.vectorize(fig)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', **kw)
    VS.assert_vector_svg(out / f'{name}.svg')
    plt.close(fig)
    print('wrote', name)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prism', required=True, help='Gel/Qty.prism')
    ap.add_argument('--fig-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--ncol', type=int, default=2)
    ap.add_argument('--order', nargs='+', default=['SRSF3_456', 'POLDIP3_87', 'XPO1_64', 'NUP42_58', 'NUP54_55',
                                                   'NUP214_33', 'UPF3A', 'AKAP8L_137'],
                    help='panel order (6i family order)')
    return ap


if __name__ == '__main__':
    main(parser().parse_args())
