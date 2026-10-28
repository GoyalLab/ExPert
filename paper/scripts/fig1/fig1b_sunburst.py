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

"""Fig. 1b, compact: the training corpus as a sunburst. Reads the tables from fig1b_data.py.

Angle = cells in the training DB. Rings, inside out:

    system     cell lines | primary CD4+ T cells
    context    the pool-rule contexts (K562 screens merged, donors pooled per condition)
    dataset    the 22 datasets (the two K562 screens, the four donors)
    coverage   thin ring: share of the selected perturbations present in that dataset
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
# SVG text as real <text> objects (editable), not glyph outlines; PDF text as TrueType
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig1b_plot import context_columns  # noqa: E402

R = {'sys': (0.50, 0.64), 'ctx': (0.65, 0.85), 'ds': (0.86, 1.00), 'cov': (1.02, 1.20)}
GAP = 0.35   # degrees of surface between neighbouring wedges


def ds_label(row) -> str:
    if row.axis == 'donor':
        return str(row.donor)
    if row.context == 'K562':
        return 'GW' if row.dataset.endswith('gwps') else 'ess.'
    return row.context


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    vs.apply()
    plt.rcParams['axes.grid'] = False

    d = Path(a.data_dir)
    cells = pd.read_parquet(d / 'fig1c_cells.parquet')
    meta = json.load(open(d / 'fig1c_meta.json'))
    dsc = context_columns(cells, meta['dataset_order'])
    donor = cells.drop_duplicates('dataset').set_index('dataset').donor
    dsc['donor'] = dsc.dataset.map(donor)
    n_db = cells.groupby('dataset').db_cells.sum()
    cov = (cells[cells.db_cells > 0].groupby('dataset').perturbation.nunique()
           / meta['n_selected'])
    dsc['cells'] = dsc.dataset.map(n_db).astype(float)
    dsc['cov'] = dsc.dataset.map(cov).fillna(0.0)
    dsc['n_pert'] = (dsc['cov'] * meta['n_selected']).round().astype(int)
    dsc['system'] = np.where(dsc.axis == 'donor', 'primary CD4⁺ T cells', 'cell lines')
    # the heatmap's clustered context order, rotated so the T-cell block comes last -- the order
    # is circular here, and this keeps the cell lines one contiguous arc
    order = json.load(open(d / 'fig1c_col_order.json'))
    t = [i for i, l in enumerate(order) if l in set(dsc.loc[dsc.axis == 'donor', 'label'])]
    order = order[t[-1] + 1:] + order[:t[-1] + 1]
    dsc['col'] = dsc.label.map({l: i for i, l in enumerate(order)})
    dsc = dsc.sort_values(['col', 'donor'], na_position='first').reset_index(drop=True)
    tot = dsc.cells.sum()
    dsc['a1'] = 90 - 360 * dsc.cells.cumsum() / tot          # clockwise from 12 o'clock
    dsc['a0'] = dsc.a1 + 360 * dsc.cells / tot

    hue = {'cell lines': vs.C['blue'], 'primary CD4⁺ T cells': vs.C['orange']}
    alpha = {'sys': 1.0, 'ctx': 0.72, 'ds': 0.45}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.45)

    def wedge(ring, t0, t1, color, alpha_=1.0, frac=1.0):
        """frac < 1 draws a radial BAR: the wedge fills that fraction of the ring's depth."""
        r0, r1 = R[ring]
        r1 = r0 + (r1 - r0) * float(frac)
        ax.add_patch(Wedge((0, 0), r1, t1 + GAP / 2, t0 - GAP / 2, width=r1 - r0,
                           facecolor=color, alpha=alpha_, edgecolor='none'))

    # data units per typographic point, fixed once the equal-aspect box is laid out
    ax.apply_aspect()
    pos_ = ax.get_position()
    dpp = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (pos_.width * fig.get_size_inches()[0] * 72)
    t2p = TextToPath()

    def text_w(txt, fp):
        return t2p.get_text_width_height_descent(txt, fp, ismath=False)[0] * dpp

    def curved(ring, t0, t1, text, color=vs.INK, fs=vs.FONT_MIN, weight='normal', fill=0.88):
        """Text set along the arc, centred in the wedge and reading left-to-right in both halves
        (clockwise on top, counter-clockwise below). Returns False if it does not fit."""
        fp = FontProperties(size=fs, weight=weight)
        r = sum(R[ring]) / 2
        span = np.rad2deg(text_w(text, fp) / r)
        if span > fill * (t0 - t1):
            return False
        m = (t0 + t1) / 2
        top = np.sin(np.deg2rad(m)) >= 0
        start = m + span / 2 if top else m - span / 2
        for i, ch in enumerate(text):
            off = np.rad2deg((text_w(text[:i], fp) + text_w(ch, fp) / 2) / r) if ch != ' ' else None
            if off is None:
                continue
            ang = start - off if top else start + off
            th = np.deg2rad(ang)
            ax.text(r * np.cos(th), r * np.sin(th), ch, ha='center', va='center',
                    rotation=ang - 90 if top else ang + 90, rotation_mode='anchor',
                    color=color, fontsize=fs, fontweight=weight)
        return True

    outside = []   # (angle, text) for arcs too short to hold their label

    # ring 1: system; depth = share of the pool the system measures at all (union of its datasets)
    sys_of = dsc.set_index('dataset').system
    sc = cells[cells.db_cells > 0].assign(system=lambda x: x.dataset.map(sys_of))
    sys_cov = sc.groupby('system').perturbation.nunique() / meta['n_selected']
    print('system coverage of the pool:', sys_cov.round(3).to_dict())
    for sysname, g in dsc.groupby('system', sort=False):
        t0, t1 = g.a0.max(), g.a1.min()
        wedge('sys', t0, t1, vs.GRID)
        wedge('sys', t0, t1, hue[sysname], frac=sys_cov[sysname])
        curved('sys', t0, t1, sysname, vs.SURFACE, weight='bold')

    # ring 2: contexts
    for col, g in dsc.groupby('col', sort=True):
        t0, t1 = g.a0.max(), g.a1.min()
        row = g.iloc[0]
        wedge('ctx', t0, t1, hue[row.system], alpha['ctx'])
        # a single-dataset context is the same entity in the dataset ring, where the arc is
        # longer -- fall back there before sending the label outside
        # (always, for consistency: every single-dataset context reads at the same radius)
        placed = (curved('ds', t0, t1, row.label) if len(g) == 1
                  else curved('ctx', t0, t1, row.label))
        if not placed:
            outside.append(((t0 + t1) / 2, row.label))

    # ring 3: datasets -- only where a context holds more than one dataset
    multi = dsc.groupby('col').dataset.transform('size') > 1
    for i, row in dsc.iterrows():
        wedge('ds', row.a0, row.a1, hue[row.system], alpha['ds'])
        if multi[i]:
            if not curved('ds', row.a0, row.a1, ds_label(row)):
                outside.append(((row.a0 + row.a1) / 2, f'K562 {ds_label(row)}'))

    # ring 4: perturbations per dataset, linear; a full track = all selected perturbations
    for _, row in dsc.iterrows():
        wedge('cov', row.a0, row.a1, vs.GRID)
        wedge('cov', row.a0, row.a1, hue[row.system], frac=row.n_pert / meta['n_selected'])

    # outside labels, nudged apart so short neighbouring arcs do not collide
    outside.sort(key=lambda x: -x[0])
    pos, min_sep = [], 9.0
    for ang, _ in outside:
        pos.append(ang if not pos or pos[-1] - ang >= min_sep else pos[-1] - min_sep)
    for (ang, text), p in zip(outside, pos):
        r_in, r_out = R['cov'][1] + 0.02, 1.36
        x0, y0 = r_in * np.cos(np.deg2rad(ang)), r_in * np.sin(np.deg2rad(ang))
        x1, y1 = r_out * np.cos(np.deg2rad(p)), r_out * np.sin(np.deg2rad(p))
        ax.plot([x0, x1], [y0, y1], color=vs.MUTED, lw=1)
        ax.text(x1 + (0.03 if x1 >= 0 else -0.03), y1, text, ha='left' if x1 >= 0 else 'right',
                va='center', color=vs.INK)

    # centre: the headline numbers, widest caption in the middle row where the hole is widest
    lines = [(f"{meta['n_cells_db'] / 1e6:.1f} M", 'cells'),
             (f"{dsc.col.nunique()} · {meta['n_datasets']}", 'contexts · datasets'),
             (f"{meta['n_selected']:,}", 'perturbations')]
    for k, (big, small) in enumerate(lines):
        y = 0.25 - 0.25 * k
        ax.text(0, y + 0.04, big, ha='center', va='center', fontsize=vs.FONT_MIN + 5,
                fontweight='bold', color=vs.INK)
        ax.text(0, y - 0.065, small, ha='center', va='center', color=vs.INK2)

    # key for the two bar rings (grey track = 100%)
    ax.text(1.6, -1.47, f"outer bars: perturbations per dataset (full = {meta['n_selected']:,})\n"
            f"inner bars: share of the {meta['n_selected']:,} measured per system",
            ha='right', va='bottom', color=vs.INK2)

    vs.save(fig, a.out_dir, a.name)
    print(dsc[['dataset', 'label', 'cells', 'cov', 'n_pert']].assign(
        cells=lambda x: (x.cells / 1e6).round(2), cov=lambda x: x['cov'].round(3)).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--name', default='fig1b_training_corpus_sunburst')
    main(ap.parse_args())
