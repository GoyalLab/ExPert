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

"""Supplementary Fig. S9 panels (prototype bridges between drug classes; related to Fig. 4), one file per panel in
--out-dir (si/figS9), rendered ONLY from --data-dir (paper/figures/si/figS9/data, figS9_data.py).
The drawing is Fig. 4's
(fig4_plot._bridge_strip / _bridge_graph, used as a library): learned prototypes in ExPert's colour, the best prior in
ink; a bridge a prior also finds is split between the two colours.

    figS9a_strip           the top bridges: traversal fold over the matched null, learned prototypes vs best prior
    figS9b_PDPK1           neighbourhood of PDPK1 between the mTOR and PI3K inhibitors' targets
    figS9c_RALGDS          neighbourhood of RALGDS between the MEK and FGFR inhibitors' targets
    figS9d_other_bridges   the same neighbourhood for every other bridge in A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig4_plot as F4  # noqa: E402


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    F4.VS = vs                                   # fig4_plot's drawing reads its module-level VS


def load(d):
    meta = json.loads((d / 'figS9_meta.json').read_text())
    t = pd.read_csv(d / 'figS9a_strip.csv').reset_index(drop=True)
    nodes, edges = pd.read_csv(d / 'figS9_graph_nodes.csv'), pd.read_csv(d / 'figS9_graph_edges.csv')
    backed = dict(zip(t.gene, t.prior_backed.astype(bool)))
    return meta, t, nodes, edges, backed


def centre(backed, gene):
    return (F4.EXPERT, F4.VS.INK2) if backed.get(gene, False) else (F4.EXPERT, F4.EXPERT)


def panel_a(d, out, vs):
    meta, t, _, _, backed = load(d)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    F4._bridge_strip(fig, ax, t, meta['main'], backed)
    vs.save(fig, out, 'figS9a_strip')


def panel_graph(d, out, vs, gene, name):
    meta, _, nodes, edges, backed = load(d)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    F4._bridge_graph(ax, nodes, edges, gene, meta, max_drugs=3, sub=False, centre=centre(backed, gene), header='top')
    vs.save(fig, out, name)


def panel_b(d, out, vs):
    panel_graph(d, out, vs, json.loads((d / 'figS9_meta.json').read_text())['main'][0], 'figS9b_PDPK1')


def panel_c(d, out, vs):
    panel_graph(d, out, vs, json.loads((d / 'figS9_meta.json').read_text())['main'][1], 'figS9c_RALGDS')


def panel_d(d, out, vs, ncol=2):
    meta, _, nodes, edges, _ = load(d)
    rest = [x['gene'] for x in meta['top'] if x['gene'] not in meta['main'] and x['gene'] in set(nodes.bridge)]
    nrow = int(np.ceil(len(rest) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(7.6, 3.1 * nrow), gridspec_kw={'hspace': 0.85, 'wspace': 0.3})
    for ax, gene in zip(np.atleast_1d(axs).ravel(), rest):
        F4._bridge_graph(ax, nodes, edges, gene, meta, max_drugs=2, compact=True)
    for ax in np.atleast_1d(axs).ravel()[len(rest):]:
        ax.set_axis_off()
    vs.save(fig, out, 'figS9d_other_bridges')


PANELS = {'a': panel_a, 'b': panel_b, 'c': panel_c, 'd': panel_d}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in a.panels:
        PANELS[k](d, out, vs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
