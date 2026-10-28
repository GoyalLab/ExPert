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

"""Supplementary Fig. S11 panels (per-cell effect size predicted by the effect-size head), one file per panel in
--out-dir (si/figS11), rendered ONLY from --data-dir (paper/figures/si/figS11/data,
figS11_data.py). Hexbins as in the model's
TensorBoard comparison (log counts, magma, dashed identity line), measured effect size on x, predicted on y, both
z-scored within dataset; Pearson r and Spearman rho in each panel.

    figS11a_pooled          the four essential-gene screens pooled (left) and the other 18 pooled (right), each with
                            marginal histograms: measured on top, predicted on the right
    figS11b_per_dataset     one hexbin per dataset, sorted by the mean of Pearson r and Spearman rho (best first),
                            shared axes and colour scale; both in each title
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
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from figS1_plot import DATASETS, short, style  # noqa: E402  (dataset order + labels as in Fig. 1c / Fig. S1)
from figS11_data import ESSENTIAL  # noqa: E402

CMAP = 'magma'


def axis_label(kind):
    return f'{kind} effect size (z)'


def limits(c):
    lo, hi = c[['eff_z', 'eff_hat_z']].stack().quantile([0.001, 0.999]).to_numpy()
    return float(np.floor(lo)), float(np.ceil(hi))


def hexbin(ax, g, lim, norm, grid, vs):
    hb = ax.hexbin(g.eff_z, g.eff_hat_z, gridsize=grid, extent=(*lim, *lim), mincnt=1, cmap=CMAP, norm=norm,
                   linewidths=0)
    ax.plot(lim, lim, color=vs.INK2, ls='--', lw=1, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal', adjustable='box')
    t = [0, 5]
    ax.set_xticks(t); ax.set_yticks(t)
    return hb


def count_fmt(v, _pos=None):
    return f'{v / 1000:g}k' if v >= 1000 else f'{v:g}'


def fmt_r(r):
    return f'{r:.2f}'.replace('-', '\u2212')


def r_rho(r, rho):
    return f'r = {fmt_r(r)}, \u03c1 = {fmt_r(rho)}'


def r_label(ax, r, vs):
    ax.text(0.04, 0.96, f'r = {fmt_r(r)}', transform=ax.transAxes, ha='left', va='top', fontsize=vs.FONT_MIN,
            color=vs.INK)


def panel_a(c, T, out, vs):
    T = T.assign(combined=(T.r + T.rho) / 2)
    order = list(T.loc[[ds for ds, _, _ in DATASETS if ds in set(c.dataset)]].combined
                 .sort_values(ascending=False).index)
    lim = limits(c)
    ncol = 6
    nrow = int(np.ceil(len(order) / ncol))
    norm = LogNorm(1, max(np.histogram2d(g.eff_z, g.eff_hat_z, bins=40, range=[lim, lim])[0].max()
                          for _, g in c.groupby('dataset')))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.35 * nrow), sharex=True, sharey=True,
                             gridspec_kw={'wspace': 0.12, 'hspace': 0.45})
    axes = axes.ravel()
    for ax, ds in zip(axes, order):
        hb = hexbin(ax, c[c.dataset == ds], lim, norm, 40, vs)
        ax.set_title(f"{short(ds)}\n{r_rho(T.loc[ds, 'r'], T.loc[ds, 'rho'])}", fontsize=vs.FONT_MIN, color=vs.INK)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:len(order)]):
        if i >= len(order) - ncol:
            ax.tick_params(labelbottom=True)
    fig.supxlabel(axis_label('measured'), fontsize=vs.FONT_MIN, color=vs.INK2, y=0.04)
    fig.supylabel(axis_label('predicted'), fontsize=vs.FONT_MIN, color=vs.INK2, x=0.08)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.2])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('cells', fontsize=vs.FONT_MIN); cb.outline.set_visible(False)
    vs.save(fig, out, 'figS11b_per_dataset')


def panel_b(c, P, out, vs):
    """Essential screens and all screens pooled, side by side, each a hexbin with marginal histograms."""
    lim = limits(c)
    bins = np.linspace(*lim, 61)
    fig = plt.figure(figsize=(12.4, 6.2))
    outer = fig.add_gridspec(1, 2, wspace=0.32)
    hbs = []
    ess = c.dataset.isin(ESSENTIAL)
    for k, (key, g) in enumerate((('essential', c[ess]), ('non_essential', c[~ess]))):
        gs = outer[k].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        top = fig.add_subplot(gs[0, 0], sharex=ax)
        right = fig.add_subplot(gs[1, 1], sharey=ax)
        hbs.append(hexbin(ax, g, lim, LogNorm(1, None), 50, vs))
        ax.set_aspect('auto')
        ax.set_xlabel(axis_label('measured'))
        ax.set_ylabel(axis_label('predicted') if k == 0 else '')
        top.hist(g.eff_z, bins=bins, color=vs.INK, alpha=0.85)
        right.hist(g.eff_hat_z, bins=bins, color=vs.INK, alpha=0.85, orientation='horizontal')
        top.set_title(f"{'essential-gene screens' if key == 'essential' else 'non-essential screens'}\n"
                      f"{r_rho(float(P.loc[key, 'r']), float(P.loc[key, 'rho']))}", fontsize=vs.FONT_MIN, color=vs.INK)
        for h in (top, right):
            h.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            for sp_ in h.spines.values():
                sp_.set_visible(False)
            h.grid(False)
        top.spines['bottom'].set_visible(True); right.spines['left'].set_visible(True)
        # count axes: cells per bin, on the outer side of each marginal
        top.spines['left'].set_visible(True); right.spines['bottom'].set_visible(True)
        top.tick_params(left=True, labelleft=True, labelsize=vs.FONT_MIN - 4)
        right.tick_params(bottom=True, labelbottom=True, labelsize=vs.FONT_MIN - 4)
        top.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        right.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        top.yaxis.set_major_formatter(FuncFormatter(count_fmt))
        right.xaxis.set_major_formatter(FuncFormatter(count_fmt))
        top.set_ylabel('cells', fontsize=vs.FONT_MIN - 3)
        right.set_xlabel('cells', fontsize=vs.FONT_MIN - 3)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.25])
    cb = fig.colorbar(hbs[-1], cax=cax)
    cb.set_label('cells', fontsize=vs.FONT_MIN); cb.outline.set_visible(False)
    vs.save(fig, out, 'figS11a_pooled')


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    d, out = Path(a.data_dir), Path(a.out_dir)
    c = pd.read_parquet(d / 'figS11_cells.parquet')
    T = pd.read_csv(d / 'figS11_per_dataset.csv').set_index('dataset')
    P = pd.read_csv(d / 'figS11_pooled.csv', index_col=0)
    panel_a(c, T, out, vs)
    panel_b(c, P, out, vs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True)
    main(ap.parse_args())
