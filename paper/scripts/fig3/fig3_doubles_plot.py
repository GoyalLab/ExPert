# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Fig. 3 H/I (formerly Fig. 4 L/M): doubles recovery against the FULL K562 GWPS fingerprint
dictionary (8,347).

Every arm is frozen, none saw these screens, and all four rank the same 8,347 candidates -- which is
100% of the fingerprint dictionary and 93% of ExPert's trained table, so neither method is handicapped
by the shared list (verified: ExPert on 8,347 vs its own 8,964 is identical on every screen but
GSE146194, where seven ingest-dropped genes including the hub FDPS cost it 22 doubles).
Subset: in-panel doubles -- both genes also measured as singles in that screen.

Panels (any@k, in-panel doubles) -> --out-dir. The same comparison
on the candidates that occur in each screen's own doubles is Supplementary Fig. S5 (figS5_plot.py, reusing
panel() below).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig4_pb_plot as fp

# dark grey, distinct from Centroids (#8c8c8c) and Random (#b8b8b8) by value, not hue
FINGERPRINT = '#404040'
ARMS = [('ExPert', fp.EXPERT, '-', 3.0),
        ('Fingerprint (our GWPS)', FINGERPRINT, '-', 2.6),
        ('Centroids', fp.CENTROID, '-', 2.0),
        ('Random', fp.COSINE, ':', 1.6)]
SHORT = {'Fingerprint (our GWPS)': 'RNA fingerprinting\n(genome-wide K562)'}
MOD = {'loss': ('h', 'CRISPRi / Cas13d'), 'gain': ('i', 'CRISPRa')}


def end_labels_floored(ax, rows, gap=0.13, fs=11, floor=0.07):
    """fig4_pb_plot._end_labels, but the stack is lifted so nothing lands on the tick row.

    The original pushes labels DOWN to resolve collisions, so an arm sitting flat at zero (Centroids on
    the CRISPRa panel) gets a label at axes-fraction 0, right on top of the x tick labels. Here the
    positions are computed the same way, then the whole stack is shifted up by whatever it needs to clear
    `floor`.
    """
    x0, x1 = ax.get_xlim()
    log = ax.get_xscale() == 'log'
    ax.set_xlim(x0, x1 * 2.6 if log else x1 + (x1 - x0) * 0.55)
    xa = max(r[1] for r in rows)
    xt = xa * 1.15 if log else xa + (x1 - x0) * 0.05
    inv_ax, inv_dat = ax.transAxes.inverted(), ax.transData.inverted()
    to_f = lambda y: float(inv_ax.transform(ax.transData.transform((xa, y)))[1])
    to_y = lambda f: float(inv_dat.transform(ax.transAxes.transform((0.5, f)))[1])
    items = sorted(((to_f(y), lab, xp, y, c) for lab, xp, y, c in rows), reverse=True)
    prev, pos = None, []
    for f, lab, *_ in items:
        f = min(f, 0.97)
        st = gap * (1 + lab.count('\n'))
        if prev is not None and f > prev - st:
            f = prev - st
        pos.append(f); prev = f
    lift = max(0.0, floor - min(pos))
    pos = [min(f + lift, 0.99) for f in pos]
    for (_f, lab, xp, yp, c), f in zip(items, pos):
        y = to_y(f)
        ax.plot([xp, xt], [yp, y], color=fp.VS.MUTED, lw=0.6, zorder=4, clip_on=False)
        ax.text(xt * (1.03 if log else 1) + (0 if log else 0.01 * (x1 - x0)), y, lab, color=c,
                fontsize=fs, va='center', ha='left', zorder=6)


def panel(C, meta, mod, subset, metric, out, suffix, name=None):
    sub = C[(C.group == mod) & (C.denom == subset) & (C.metric == metric)]
    if sub.empty:
        return False
    # a small (matched) candidate set saturates every arm at 1 once k reaches it: stop at the
    # smallest screen's candidate count so the curves end where they still differ
    ncs = sorted({m['n_cand'] for m in meta.values() if m.get('modality', mod) == mod})
    sub = sub[sub.k <= ncs[0]]
    if metric == 'both':                     # two genes cannot both sit in the top 1: start at k = 2
        sub = sub[sub.k >= 2]
    letter, modality = MOD[mod]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    rows, lab_rows = [], []
    for label, col, ls, lw in ARMS:
        s = sub[sub.method == label].sort_values('k')
        if s.empty:
            continue
        ax.plot(s.k, s.both, ls, color=col, lw=lw, marker='o', ms=3.5, clip_on=False, zorder=3)
        lab_rows.append((SHORT.get(label, label), float(s.k.iloc[-1]), float(s.both.iloc[-1]), col))
        rows.append(pd.DataFrame(dict(modality=mod, subset=subset, metric=metric, method=label,
                                      k=s.k, recovered=s.both, n=s.n)))
    ax.set_xscale('log')
    ax.set_xticks(sorted({t for t in (1, 2, 5, 10, 50, 500) if sub.k.min() <= t <= sub.k.max()}
                         | {int(sub.k.max())}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # a hair of headroom BELOW zero: arms that sit flat at 0 would otherwise have their end label
    # land on the axis spine and collide with the x-label (Centroids on the CRISPRa panel)
    ymax = float(sub.both.max())
    ax.set_ylim(-ymax * 0.045, ymax * 1.18)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
    span = f'{ncs[0]:,}' if len(ncs) == 1 else f'{ncs[0]:,}-{ncs[-1]:,} per screen'
    ax.set_xlabel(f'top k  (of {span} candidate perturbations)', fontsize=13)
    what = {'any': 'either gene recovered', 'both': 'both genes recovered', 'single': 'target recovered'}[metric]
    unit_n = 'single perturbations' if metric == 'single' else 'doubles'
    ax.set_ylabel(f'{what}\n(n = {int(sub.n.iloc[0])} {unit_n})', fontsize=13)
    ax.set_title(modality, fontsize=13, color=fp.VS.INK, pad=8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    end_labels_floored(ax, lab_rows, gap=0.13, fs=11)
    fig.subplots_adjust(left=0.13, right=0.72, top=0.96, bottom=0.15)
    # main panel -> fig3<letter>_doubles_<mod>; every variant -> fig3<letter>_doubles_<mod><suffix>
    name = name or f'fig3{letter}_doubles_{mod}{suffix}'
    fp.save(fig, out, name, src=pd.concat(rows), d=None)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--source-dir', default=None)
    ap.add_argument('--vizstyle-root', required=True)
    a = ap.parse_args()
    fp.setup(a.vizstyle_root)
    fp.SRC_DIR = a.source_dir or a.out_dir
    globals()['plt'] = fp.plt
    globals()['matplotlib'] = fp.matplotlib
    d = Path(a.data_dir)
    C = pd.read_csv(d / 'ourgwps_doubles_curves.csv')
    meta = json.load(open(d / 'ourgwps_meta.json'))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for mod in ('loss', 'gain'):
        # any@k on in-panel doubles. both@k is pinned by whichever gene the response represents worse, so it
        # reads as a floor-level comparison and hides a consistent ~2x lead (both@k: Supplementary Fig. S5)
        if panel(C, meta, mod, 'in_panel', 'any', a.out_dir, ''):
            print(f'[fig3{MOD[mod][0]}] {MOD[mod][1]} any@k in-panel', flush=True)


if __name__ == '__main__':
    main()
