# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Helpers for the per-figure notebooks (paper/figures/<fig>/<fig>.ipynb).

A notebook runs from its figure folder and only ever uses paths relative to it:
    data/     the figure's plotting cache (a symlink; see paper/figures/README.md)
    inputs/   any other file the panels read (symlinks)
    render/   where the notebook writes its panels (svg + png) and the source tables they emit (render/src)
The style module is paper/style/src/vizstyle.py (vizstyle_root = paper/style).
"""
from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2]          # paper/
SCRIPTS = PAPER / 'scripts'
STYLE = PAPER / 'style'
sys.path += [str(p) for p in SCRIPTS.iterdir() if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]


def vizstyle():
    """The paper's style module (what every plot script imports as `from src import vizstyle`)."""
    sys.path.insert(0, str(STYLE))
    from src import vizstyle as vs
    return vs


def folders(render='render'):
    """(data, inputs, render, render/src) for the current figure folder; render dirs are created."""
    here = Path.cwd()
    out = here / render
    (out / 'src').mkdir(parents=True, exist_ok=True)
    return here / 'data', here / 'inputs', out, out / 'src'


def show(out, *names, width=520):
    """Display rendered panels inline (the PNG each plot script wrote)."""
    from IPython.display import Image, display
    for n in names:
        display(Image(filename=str(Path(out) / f'{n}.png'), width=width))
