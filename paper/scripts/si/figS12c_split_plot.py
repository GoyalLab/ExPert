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

"""Supplementary Fig. S12C-H panels (Fig. 5 split by condition: AC and AH instead of ALD; related to Fig. 5). Each panel is
drawn by Fig. 5's own function on Fig. 5's data (read only: --fig5-data, --deg, the FISH frames and nucleus
measurements), so rows, colours, tests and scales are those of the main panels; only the columns change. Panels go to
--out-dir (si/figS12) and every source table the functions write goes to <out-dir>/data.

    figS12c_umap              5B: nuclei and the top prototypes, healthy / AC / AH (the 5B UMAP, conditions apart)
    figS12d_pathways          5E: the 5E rows (ExPert hepatocyte predictions), healthy / AC / AH columns
    figS12e_pathways_deg      5F: the 5F rows (differentially expressed genes), AC / AH columns
    figS12f_fish              5I: nuclear : cytoplasmic poly(A) FISH foci, healthy / ALD / AC / AH
    figS12g_nuclei            5J: hepatocyte nuclear diameter, healthy / ALD / AC / AH (tests on per-sample medians)
    figS12h_export_families   5L: mRNA export families, healthy / AC / AH
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

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_final as F  # noqa: E402
import fig5i_fish as FISH  # noqa: E402
import fig5j_nuclei as NUC  # noqa: E402


def rows_5e_5f(a, d, T, TD, gs, plabels):
    """the 5E / 5F rows, selected exactly as fig5_final.stage_plot does (ALD column, hit-overlap folding)"""
    mem_m = {k: set(v) for k, v in gs.items() if k.startswith('Reactome: ') or k in dict(F.HIGHLIGHT).values()}
    mem_d = {k: set(str(v).split(';')) for k, v in TD.drop_duplicates('set').set_index('set').members.items()}
    lab_ = np.asarray(plabels)
    pe = np.load(d / 'fig5_profiles.npz', allow_pickle=True)
    Sc_ = {str(k): pe['cond'][i] for i, k in enumerate(pe['cond_keys'])}
    U_m = {t: set().union(*[set(lab_[np.argsort(-sg_ * Sc_[k], kind='stable')[:a.top]]) for k in
                            ('AC|Hepatocyte', 'AH|Hepatocyte')]) for t, sg_ in (('loss', 1), ('gain', -1))}
    deg_ = pd.read_csv(a.deg)
    H_ = deg_[deg_.group.isin(['AC|Hepatocyte', 'AH|Hepatocyte'])]
    U_d = {t: set().union(*[set(H_[H_.group == g].sort_values('score', ascending=asc).gene.head(a.top))
                            for g in ('AC|Hepatocyte', 'AH|Hepatocyte')]) for t, asc in (('loss', True), ('gain', False))}
    bt_ = lambda X: X[X.column == 'ALD'].loc[lambda z: z.groupby('set').p.idxmin()].set_index('set')['tail']

    def hit_of(mem, U, X):
        b_ = bt_(X)
        return {k: v & U[b_.get(k, 'loss')] for k, v in mem.items()}
    r_m, _, mo_m = F.top_rows(T, 'ALD', mem_m, a.n_top, n_more_p=0.05, hits=hit_of(mem_m, U_m, T), folded={},
                              order='fe')
    r_d, _, mo_d = F.top_rows(TD, 'ALD', mem_d, a.n_top, n_more_p=0.05, highlight=False,
                              hits=hit_of(mem_d, U_d, TD), folded={}, order='fe')
    return (r_m, mo_m), (r_d, mo_d)


def setup(a, VS=None):
    """style + Fig. 5's tables and gene sets; sets F.SRC = --src-dir. Returns the state draw() needs."""
    from types import SimpleNamespace
    if VS is None:
        sys.path.insert(0, str(Path(a.vizstyle_root)))
        from src import vizstyle as VS
    VS.apply()
    VS.SURFACE = 'white'
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                                'savefig.facecolor': 'white'})
    d, out = Path(a.fig5_data), Path(a.out_dir)
    src = Path(a.src_dir); src.mkdir(parents=True, exist_ok=True)
    F.SRC = src
    cc = {'Normal': '#a9a8a2', 'healthy': '#8a8983', 'AC': F.AC_C, 'AH': F.AH_C, 'ALD': F.ALD_C}
    T = pd.read_csv(d / 'fig5b_ora.csv.gz')
    TD = pd.read_csv(d / 'fig5b_ora_deg.csv.gz')
    assert {'AC', 'AH', 'healthy'} <= set(T.column) and {'AC', 'AH'} <= set(TD.column), 'condition columns missing'
    plabels = [str(x) for x in np.load(d / 'fig5_profiles.npz', allow_pickle=True)['labels']]
    gs = F.gene_sets(a, plabels)
    return SimpleNamespace(VS=VS, d=d, out=out, src=src, cc=cc, T=T, TD=TD, plabels=plabels, gs=gs)


def main(a):
    draw(a, setup(a), a.panels)


def draw(a, c, panels):
    """draw the requested panels (letters c-h) from the setup() state"""
    VS, d, out, src, cc, T, TD, plabels, gs = c.VS, c.d, c.out, c.src, c.cc, c.T, c.TD, c.plabels, c.gs
    want = set(panels)

    if 'c' in want:                                   # 5B UMAP keys: highlighted families + the literature rows
        rows, _, _ = F.best_tail_rows(T, 'ALD')
        keys = [k for _, k in F.HIGHLIGHT] + [k for _, k, blk in rows if blk == 'literature']
        F.umap_plot(VS, d, out, 'scoring', 'figS12c_umap', gs, keys, cc, a)
    if want & {'d', 'e'}:
        (r_m, mo_m), (r_d, mo_d) = rows_5e_5f(a, d, T, TD, gs, plabels)
        pd.DataFrame({'panel': ['D'] * len(r_m) + ['E'] * len(r_d),
                      'set': [k for _, k, _ in r_m] + [k for _, k, _ in r_d]}).to_csv(
            src / 'figS12de_rows.csv', index=False)
        if 'd' in want:
            F.column_dots(VS, [('healthy', T, 'healthy'), ('AC', T, 'AC'), ('AH', T, 'AH')], r_m, mo_m, out,
                          'figS12d_pathways', title=('ExPert', 'D', F.EXPERT))
        if 'e' in want:
            F.column_dots(VS, [('AC', TD, 'AC'), ('AH', TD, 'AH')], r_d, mo_d, out, 'figS12e_pathways_deg',
                          title=('DEG', 'o', VS.INK2))
    if 'f' in want:
        x, tests, summ = FISH.prepare(a.fish_frames)
        tests.to_csv(src / 'figS12f_fish_tests.csv', index=False)
        summ.to_csv(src / 'figS12f_fish_summary.csv')
        x[x.group != 'ALD'].drop(columns='cond').to_csv(src / 'figS12f_fish_source.csv', index=False)
        fig = FISH.box_panel(VS, plt, x, tests, 'log2nc', 'log2 nuclear : cytoplasmic\npoly(A) foci', FISH.ORDER,
                             FISH.POS, FISH.PAIRS, {'healthy': cc['healthy'], 'AC': cc['AC'], 'AH': cc['AH']},
                             np.random.default_rng(0))
        F.save(VS, fig, out, 'figS12f_fish')
    if 'g' in want:
        m = pd.read_csv(a.nuclei)
        m = m[(m.row_type == 'Nucleus') & (m.status == 'OK')]
        x = pd.concat([m.assign(grp=m.group), m[m.group.isin(['AC', 'AH'])].assign(grp='ALD')], ignore_index=True)
        x[x.grp != 'ALD'].to_csv(src / 'figS12g_nuclei_source.csv', index=False)
        t = NUC.plot(VS, plt, x, 'average_diameter', ['Healthy', 'ALD', 'AC', 'AH'], [0, 1, 2.2, 3.2],
                     [('Healthy', 'ALD'), ('AC', 'AH')], out, 'figS12g_nuclei', False)
        pd.DataFrame(t).to_csv(src / 'figS12g_nuclei_tests.csv', index=False)
    if 'h' in want:
        F.family_bars(VS, d, out, cc, a, gs, name='figS12h_export_families')


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig5-data', required=True,
                    help='paper/figures/fig5/data/final (fig5_final.py --stage data)')
    ap.add_argument('--deg', required=True, help='donor-pseudobulk DESeq2 table (as fig5_final.sbatch)')
    ap.add_argument('--modules', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--fish-frames', required=True, help='nuc_puncta_frame_summary_master_raw.csv')
    ap.add_argument('--nuclei', required=True, help='NucleiMeasurements.csv')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--top', type=int, default=200, help='as fig5_final')
    ap.add_argument('--n-top', type=int, default=15, help='as fig5_final (5E / 5F rows)')
    ap.add_argument('--label-size', type=float, default=9.5, help='as fig5_final (5B labels)')
    ap.add_argument('--panels', nargs='+', default=list('cdefgh'), choices=list('cdefgh'))
    return ap


if __name__ == '__main__':
    main(parser().parse_args())
