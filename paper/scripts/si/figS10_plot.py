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

"""Supplementary Fig. S10 panels (Proxy Anchor vs a softmax head: the gradient once the correct prototype wins), one file
per panel in --out-dir (si/figS10), rendered ONLY from --data-dir (paper/figures/si/figS10/data,
figS10_data.py). Proxy Anchor (the
shipped objective) in ExPert's colour, its two halves as lighter / broken orange lines; the cosine cross-entropy head in
ink.

    figS10_gradient    gradient on the prototypes and on the latent, relative to chance geometry, against the batch's
                      mean margin (cosine to the own prototype minus the best other; > 0 = own prototype ranks first)
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
from fig2_common import EXPERT_COLOR  # noqa: E402

LIGHT = '#f2c46d'
Y_MIN = 1e-6       # below this the softmax head's gradient is float noise (it reaches 1e-11)


def style(vs):
    vs.apply()
    vs.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'axes.grid': False})


def term_style(vs):
    return {'cosine CE': dict(color=vs.INK, ls='-', lw=2.2, label='softmax head (cosine cross-entropy)'),
            'Proxy Anchor': dict(color=EXPERT_COLOR, ls='-', lw=2.6, label='Proxy Anchor'),
            'Proxy Anchor, negative term': dict(color=EXPERT_COLOR, ls='--', lw=1.6, label='its negative term'),
            'Proxy Anchor, positive term': dict(color=LIGHT, ls=':', lw=2.0, label='its positive term')}


def panel_a(d, out, vs, meta):
    t = pd.read_csv(d / 'figS10a_sweep.csv')
    st = term_style(vs)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    for ax, col, title in ((axs[0], 'grad_atoms_rel', 'on the prototypes'), (axs[1], 'grad_z_rel', 'on the latent')):
        for term in meta['terms']:
            g = t[t.term == term].sort_values('margin_mean')
            ax.plot(g.margin_mean, g[col].clip(lower=Y_MIN), **st[term])
        ax.axhline(1, color=vs.INK2, lw=1, ls=':')
        ax.axvline(0, color=vs.INK2, lw=1, ls=':')
        ax.text(0.01, 0.02, 'own prototype\nranks first →', transform=ax.get_xaxis_transform(), ha='left',
                va='bottom', fontsize=vs.FONT_MIN - 3, color=vs.INK2)
        ax.set_yscale('log'); ax.set_ylim(Y_MIN, 4)
        ax.set_title(title, loc='left')
    axs[0].set_ylabel('gradient norm (× chance)')
    for ax in axs:
        ax.set_xlabel('mean margin of the batch')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=vs.FONT_MIN - 3)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.3, wspace=0.12)
    vs.save(fig, out, 'figS10_gradient')


PANELS = {'a': panel_a}


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / 'figS10_meta.json').read_text())
    for k in a.panels:
        PANELS[k](d, out, vs, meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', nargs='+', default=list(PANELS), choices=list(PANELS))
    main(ap.parse_args())
