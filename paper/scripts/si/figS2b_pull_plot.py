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

"""Supplementary Fig. S2B (formerly Fig. 3D, then Supplementary Fig. S2B): pooled top-10 of the gated prior pull (ExPert), an equal pull on
every prototype and no pull, on held-out K562 queries, against run-to-run noise. Drawn by fig3_plot.panel_c_dumbbell
(same code and styling as the old 3D); this driver only sets up the style and routes the outputs to si/figS2.

    figS2b_pull_vs_base               -> --out-dir (si/figS2); source table -> --src-dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas  # noqa: E402,F401  before the bootstrap: paper/scripts/gene_panel_src/select.py shadows stdlib select
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig3_plot as F3  # noqa: E402


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    VS.apply()
    VS.SURFACE = 'white'
    F3.plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                            'svg.fonttype': 'none', 'pdf.fonttype': 42})
    from fig2_common import EXPERT_COLOR
    F3.VS = VS
    F3.COL.update({'expert': EXPERT_COLOR, 'oracle': VS.INK, 'null': VS.MUTED, 'fill': '#bdbcb6'})
    for p in (a.out_dir, a.src_dir):
        Path(p).mkdir(parents=True, exist_ok=True)
    F3.panel_c_dumbbell(a.out_dir, a.effgate_per_query, a.ranks_dir, a.src_dir, name='figS2b_pull_vs_base')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--effgate-per-query', required=True, help='effgate_decomp/per_query.tsv.gz')
    ap.add_argument('--ranks-dir', required=True, help='v017 readout ranks/<ARM>/<ARM>_ranks.csv')
    ap.add_argument('--out-dir', required=True, help='si/figS2')
    ap.add_argument('--src-dir', required=True, help='paper/figures/si/figS2/data')
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    main(ap.parse_args())
