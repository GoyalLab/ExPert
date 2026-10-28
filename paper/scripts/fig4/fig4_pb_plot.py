# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Shared style and save helpers for the combinatorial-screen panels (Fig. 3H/I: fig3_doubles_plot.py;
Supplementary Fig. S5: figS5_plot.py).

Every panel is its own file (svg + png at 300 dpi) with a source csv. Standing rules: white background, no
in-panel titles or notes, coloured classes labelled directly (no legends), 14 pt text, ExPert = Okabe-Ito orange.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

EXPERT = '#E69F00'          # fig2_common.EXPERT_COLOR
CENTROID = '#8c8c8c'        # training centroids (fig2 baseline grey)
COSINE = '#b8b8b8'
VS = None


def setup(root):
    global VS
    sys.path.insert(0, str(Path(root)))
    from src import vizstyle as _vs
    VS = _vs
    VS.apply()
    VS.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                         'savefig.dpi': 300, 'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'svg.fonttype': 'none'})


SRC_DIR = None


def save(fig, out, name, src=None, d=None):
    """figure -> out (svg + png 300 dpi); its source table -> SRC_DIR (default out)."""
    VS.save(fig, out, name)
    if src is not None:
        sd = Path(SRC_DIR or out); sd.mkdir(parents=True, exist_ok=True)
        src.to_csv(sd / f'{name}_source.csv', index=False)
    plt.close(fig)
