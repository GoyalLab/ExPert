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

"""Fig. 2: benchmark. Reads fig2_data.py output only; one file per panel in --out-dir (svg + png,
300 dpi). Supplementary Fig. S3: figS3_plot.py (own data, si/figS3).

    a  radar: F1 / MRR / recall@5 / recall@10 / recall@100, ID (top) and OOD (bottom), one shared
       ladder (all are fractions); per-perturbation, efficiency-weighted, from the per-cell cache
    b  OOD by distribution shift (unseen cell line / condition / donor), each panel on its own scale:
       one box per method, one point per (fold, held-out dataset)
    c  OOD by efficiency threshold; at the end of each line a small bar = the method's median Spearman
       rho between a cell's knockdown efficiency and how well it is scored
    d  compute vs OOD recall@10 (top) and eval compute per method (bottom; CPU core-h + GPU-h,
       linear, axis broken). ExPert is drawn as measured (CPU); a GPU-job estimate (encoding is ~7% of
       the job) is no cheaper and is kept only as a table (fig2d_expert_gpu_estimate.csv)
    e  scaling from 200 to 1,157 perturbations: eval compute-h and SLURM jobs per fold on their
       measured scale, then the combined-cost exponent; x over random (OOD) vs class-space size

Colours are the benchmark pipeline's performance-ranked palette (viridis for methods, greys for
baselines), except ExPert, which has its own colour (fig2_common.EXPERT_COLOR) so it reads at a
glance. White background, no panel shading. All text is 14 pt.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig2_common as F  # noqa: E402

FS = 14
PAL: dict[str, str] = {}
NAME: dict[str, str] = {}
ORDER: list[str] = []
MET_LABEL = {}
VS = None


def setup(d: Path, vizstyle_root: str, palette: str = 'fig2_palette.csv'):
    global VS
    sys.path.insert(0, str(Path(vizstyle_root)))
    from src import vizstyle as _VS
    VS = _VS
    VS.apply()
    plt.rcParams.update({k: FS for k in ('font.size', 'axes.titlesize', 'axes.labelsize',
                                          'xtick.labelsize', 'ytick.labelsize',
                                          'legend.fontsize', 'figure.titlesize')})
    # no background colour anywhere
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'})
    VS.SURFACE = 'white'
    p = pd.read_csv(d / palette).sort_values('order')
    PAL.update(zip(p.method, p.color)); NAME.update(zip(p.method, p.display))
    ORDER.extend(p.method)
    MET_LABEL.update(F.METRIC_LABEL)
    MET_LABEL.update(F.RADAR_LABEL)


def style(m, lw_main=1.6):
    st = dict(color=PAL.get(m, '0.5'), lw=lw_main, zorder=5)
    if m == 'ExPert':
        st.update(lw=3.6, zorder=20)
    elif 'baseline' in m:
        st.update(lw=1.3, ls='--', zorder=3)
    return st


def ordered(methods) -> list[str]:
    s = set(methods)
    return [m for m in ORDER if m in s]


def method_legend(fig, methods, ncol=4, y=0.0, **kw):
    h = [Line2D([], [], marker='o' if 'baseline' not in m else None, ms=8, **style(m))
         for m in methods]
    fig.legend(h, [NAME.get(m, m) for m in methods], loc='upper center', bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, handlelength=2.2, columnspacing=1.4, **kw)


def nice_max(v):
    """Round up to 1 / 2 / 2.5 / 3 / 4 / 5 x 10^k."""
    e = 10 ** np.floor(np.log10(v))
    for s in (1, 2, 2.5, 3, 4, 5, 10):
        if s * e >= v:
            return s * e
    return 10 * e


EXPERT_LABELS = ('ExPert', 'ExPert*', 'ExPert (GPU, est.)')


def bold_expert(fig):
    """Every text that names ExPert (tick labels, legend entries, annotations) in bold."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if t.get_text().strip() in EXPERT_LABELS:
            t.set_fontweight('bold')


PDF = set()                                          # base names that also get a .pdf (--pdf)


def assert_single_text_svg(path):
    """Every label must be ONE text object. Mathtext ($...$, also matplotlib's default log-axis labels)
    is written as a <tspan> with one x-position per glyph, which editors split into single letters."""
    import re
    bad = re.findall(r'<tspan[^>]*\sx="[-0-9.e]+ [-0-9.e]+[^"]*"[^>]*>([^<]*)<', Path(path).read_text())
    if bad:
        raise ValueError(f'{path}: per-glyph text (mathtext) in {bad[:3]}; use plain/unicode text instead')


def save(fig, out, name):
    bold_expert(fig)
    d = Path(out)
    if name in PDF:                                   # before VS.save, which closes the figure
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f'{name}.pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'  wrote {name}.pdf')
    VS.save(fig, d, name)
    assert_single_text_svg(d / f'{name}.svg')


# ---- a -----------------------------------------------------------------------------------------
class Axis:
    """One radar metric: value -> radius in [0, 1] (0 = chance end, 1 = best end) + its ladder."""

    def __init__(self, col, label, to_r, ticks):
        self.col, self.label, self.to_r, self.ticks = col, label, to_r, ticks


def linear_axis(col, vmax, step):
    ticks = [(v / vmax, f'{v:g}') for v in np.round(np.arange(step, vmax + 1e-9, step), 6)]
    return Axis(col, MET_LABEL.get(col, col), lambda v: v / vmax, ticks)


def lift_chance(col, C):
    """Analytic uniform-random value of a radar column over C classes."""
    if col.startswith('R') and col[1:].isdigit():
        return min(int(col[1:]) / C, 1.0)
    if col == 'MRR':
        return float(np.sum(1 / np.arange(1, C + 1)) / C)
    return 1 / C                                       # F1 / top-1


def lift_lin_axis(col, C, lmax, n=4):
    """Lift over random on a LINEAR radius: centre = 0x, edge = lmax; the 1x ring marks chance."""
    ch = lift_chance(col, C)
    step = nice_max(lmax / n)
    ticks = [(x / lmax, f'{x:g}\u00d7') for x in np.arange(step, lmax + 1e-9, step)]
    return Axis(col, MET_LABEL.get(col, col), lambda v: np.clip(np.asarray(v, float) / ch / lmax, 0, 1), ticks)


def radar(a, specs, out, name, shared):
    """ID metrics over the top half (left -> right), OOD mirrored beneath, so each metric's ID and
    OOD spokes sit on one vertical. shared: one ladder (rings + labels on the ID/OOD divide);
    else: per-spoke ladders (tick marks + labels on every spoke, no rings)."""
    n = len(specs)
    spokes = [(s, 'id') for s in specs] + [(s, 'ood') for s in reversed(specs)]
    N = len(spokes)
    step = 360 / N
    ang = np.radians([k * step for k in range(N)] + [0.0])
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.radians(90 + (n - 1) / 2 * step)); ax.set_theta_direction(-1)
    meths = ordered(a.method)
    for m in reversed(meths):                         # ExPert drawn last = on top
        r = [float(sp.to_r(a[(a.method == m) & (a.regime == reg)][sp.col].iloc[0])) for sp, reg in spokes]
        ax.plot(ang, r + [r[0]], **style(m), alpha=0.95)
    ax.set_ylim(0, 1.0)
    ax.spines['polar'].set_visible(False)
    ax.xaxis.grid(True, color='#b9b8b2', lw=1.2)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([f'{sp.label}\n({reg.upper()})' for sp, reg in spokes])
    box = dict(fc='white', ec='none', alpha=0.85, pad=0.4)
    if shared:
        ticks = specs[0].ticks
        ax.set_yticks([r for r, _ in ticks]); ax.set_yticklabels([])
        ax.yaxis.grid(True, color=VS.GRID, lw=0.9)
        ax.tick_params(axis='x', pad=14)
        for div in (np.radians(-step / 2), np.radians(n * step - step / 2)):   # the ID/OOD divide
            for r, lab in ticks:
                ax.text(div, r, lab, ha='center', va='center', color=VS.INK2, zorder=50, bbox=box)
    else:
        ax.set_yticks([]); ax.yaxis.grid(False)
        ax.tick_params(axis='x', pad=40)
        for t, (sp, _) in zip(ang[:-1], spokes):
            for r, lab in sp.ticks:
                ax.plot([t - 0.035 / max(r, 0.05), t + 0.035 / max(r, 0.05)], [r, r],
                        color='#9c9b95', lw=1.2, zorder=40)
                ax.text(t, r, lab, ha='center', va='center', color=VS.INK2, zorder=50, bbox=box)
    method_legend(fig, meths, ncol=3, y=0.04)
    save(fig, out, name)


def radar_pair(d, out, shared_name, spokes_name):
    """The two radars of one scoring level: one shared linear ladder (spokes F1 < MRR < recall@5 < @10 <
    @100), and the same values with every spoke on its OWN linear ladder (lift over random, 0x at the
    centre -- per spoke this is the value / its spoke maximum, so the shape is the raw-value shape)."""
    radar_shared(d, out, shared_name)
    radar_spokes(d, out, spokes_name)


def _radar_table(d):
    e = pd.read_csv(d / 'fig2a_radar_ext.csv')
    return e, [c for c in ('F1', 'MRR', 'R5', 'R10', 'R100') if c in e]


def radar_shared(d, out, name):
    """Radar on one shared linear ladder (main Fig. 2A from the pseudobulk cache)."""
    e, cols = _radar_table(d)
    vmax = np.ceil(float(e[cols].to_numpy().max()) / 0.1 + 0.25) * 0.1   # next 0.1 rung past the largest value
    step = 0.1 if vmax <= 0.7 + 1e-9 else 0.2                              # pseudobulk runs to 1.0: 0.1 rungs crowd
    radar(e, [linear_axis(c, vmax, step) for c in cols], out, name, shared=True)


def radar_spokes(d, out, name):
    """The same values, every spoke on its own linear ladder of lift over random (main Fig. 2B)."""
    e, cols = _radar_table(d)
    C = json.loads((d / 'fig2_meta.json').read_text())['n_classes']['ctx7']
    per = [lift_lin_axis(c, C, nice_max(float((e[c] / lift_chance(c, C)).max()) * 1.02)) for c in cols]
    radar(e, per, out, name, shared=False)


# ---- b -----------------------------------------------------------------------------------------
def boxes(ax, data, meths, width=0.7, jitter=True):
    """One coloured box per method, points over it."""
    rng = np.random.default_rng(0)
    for i, m in enumerate(meths):
        v = np.asarray(data.get(m, []), float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        ex = m == 'ExPert'
        bp = ax.boxplot(v, positions=[i], widths=width, patch_artist=True, showfliers=False,
                        medianprops=dict(color=VS.INK, lw=2.2 if ex else 1.6),
                        whiskerprops=dict(color=VS.INK2), capprops=dict(color=VS.INK2))
        for b in bp['boxes']:
            b.set(facecolor=PAL[m], alpha=0.85 if ex else 0.55, edgecolor=VS.INK if ex else VS.INK2,
                  lw=1.6 if ex else 1.0)
        x = i + (rng.uniform(-width / 3, width / 3, len(v)) if jitter else 0)
        ax.scatter(x, v, s=18 if ex else 14, color=PAL[m], edgecolor=VS.INK2, lw=0.4, zorder=4)


B_MAIN = ('ood_cell_line', 'ood_donor')          # main panel; unseen condition: Supplementary Fig. S3A


def panel_b(d, out, metrics, keep=B_MAIN, name='fig2c_ood_by_shift', table='fig2b_shift.csv'):
    """one box per method and shift, each context on its own scale"""
    b = pd.read_csv(d / table)
    meths = ordered(b.method)
    shifts = [s for s in keep if s in set(b['shift'])]
    b = b[b['shift'].isin(shifts)]
    fig, axes = plt.subplots(len(metrics), len(shifts), figsize=(6.2 * len(shifts), 3.6 * len(metrics)),
                             sharex=True, sharey=False, squeeze=False)
    for i, met in enumerate(metrics):
        for j, sh in enumerate(shifts):
            ax = axes[i, j]
            sub = b[b['shift'] == sh]
            boxes(ax, {m: g[met].to_numpy() for m, g in sub.groupby('method')}, meths)
            ax.grid(axis='x', visible=False)
            if i == 0:
                ax.set_title(F.SHIFT_LABEL[sh])
            if j == 0:
                ax.set_ylabel(MET_LABEL[met])
            ax.set_ylim(0, sub[met].max() * 1.05)
    for ax in axes[-1]:
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([NAME[m] for m in meths], rotation=55, ha='right', rotation_mode='anchor')
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    save(fig, out, name)


# ---- c -----------------------------------------------------------------------------------------
def spread(y, gap, lo, hi):
    """Label positions: y sorted, pushed apart to >= gap, kept inside [lo, hi]."""
    order = np.argsort(y)
    p = np.asarray(y, float)[order].copy()
    for k in range(1, len(p)):
        p[k] = max(p[k], p[k - 1] + gap)
    if len(p) and p[-1] > hi:
        p -= p[-1] - hi
        for k in range(len(p) - 2, -1, -1):
            p[k] = min(p[k], p[k + 1] - gap)
    p = np.maximum(p, lo)
    out = np.empty_like(p); out[order] = p
    return out


def panel_c(d, out):
    """OOD lines by efficiency threshold; at each line's end a small bar = that method's median
    Spearman rho (over OOD fold x dataset views) between a cell's knockdown efficiency and how well
    the cell is scored (top-1 correct / in top 10 / reciprocal rank, matching the row's metric)."""
    L = pd.read_csv(d / 'fig2c_lines.csv')
    R = pd.read_csv(d / 'fig2c_rho.csv') if (d / 'fig2c_rho.csv').exists() else None   # per-cell rho bars
    mets = list(F.SWEEP)
    meths = ordered(L.method)
    fig, axes = plt.subplots(len(mets), 1, figsize=(7.2, 4.0 * len(mets)), squeeze=False, sharex=True)
    xs = sorted(L.eff.unique())
    rmax = float(R.groupby(['metric', 'method']).rho.median().abs().max()) if R is not None else 0.0
    tick = 0.05 if rmax <= 0.15 else (0.1 if rmax <= 0.35 else 0.25)
    x0, Lmax = xs[-1] + 0.25, 1.4 / max(rmax, 1e-9)    # bar origin (rho = 0); the largest |rho| -> 1.4 x-units
    for i, met in enumerate(mets):
        ax = axes[i, 0]
        med = R[R.metric == met].groupby('method').rho.median() if R is not None else pd.Series(dtype=float)
        ends = {}
        for m in reversed(meths):
            g = L[(L.method == m) & (L.metric == met)].set_index('eff').reindex(xs)
            st = style(m)
            ax.plot(xs, g['mean'], marker='o', ms=9 if m == 'ExPert' else 5, **st)
            ax.fill_between(xs, g['mean'] - g['sem'], g['mean'] + g['sem'],
                            color=st['color'], alpha=0.15, lw=0, zorder=1)
            if np.isfinite(g['mean'].iloc[-1]):
                ends[m] = g['mean'].iloc[-1]
        top = max(ends.values()) * 1.12
        ax.set_ylim(0, top)
        lab = [m for m in meths if m in ends and m in med.index and np.isfinite(med[m])]
        gap = top * 0.066
        ly = spread([ends[m] for m in lab], gap=gap, lo=top * 0.01, hi=top * 0.99)
        for m, yl in zip(lab, ly):
            ex = m == 'ExPert'
            ax.plot([xs[-1], x0], [ends[m], yl], color=PAL[m], lw=0.9, zorder=2, clip_on=False)
            ax.barh(yl, med[m] * Lmax, left=x0, height=gap * 0.72, color=PAL[m], clip_on=False,
                    edgecolor=VS.INK if ex else 'none', lw=1.2 if ex else 0, zorder=6)
        ax.set_ylabel(F.SWEEP[met]); ax.set_xticks(xs)
        if R is None:
            ax.set_xlim(xs[0] - 0.15, xs[-1] + 0.15)
            continue
        ax.axvline(x0, color=VS.INK2, lw=0.8, zorder=5)
        ax.set_xlim(xs[0] - 0.15, x0 + rmax * Lmax + 0.08)

    # a scale for the bars: rho = 0.25 / 0.5 ticks under the last panel's bar origin
    ax = axes[-1, 0]
    for v in (np.arange(tick, rmax + 1e-9, tick) if R is not None else []):
        ax.plot([x0 + v * Lmax] * 2, [-0.035, -0.015], transform=ax.get_xaxis_transform(),
                color=VS.INK2, lw=1, clip_on=False)
        ax.text(x0 + v * Lmax, -0.04, f'{v:g}', transform=ax.get_xaxis_transform(), ha='center',
                va='top', color=VS.INK2)
    if R is not None:
        ax.text(x0, -0.04, '0', transform=ax.get_xaxis_transform(), ha='center', va='top', color=VS.INK2)
    axes[-1, 0].set_xticklabels([f'{x:g}' for x in xs])
    # one x label under each part of the shared x-axis: the threshold lines, and the rho bars
    tr = ax.get_xaxis_transform()
    ax.text((xs[0] + xs[-1]) / 2, -0.13, 'min. effect size', transform=tr, ha='center', va='top',
            color=VS.INK2, fontsize=plt.rcParams['axes.labelsize'])
    if R is not None:
        ax.text(x0 + rmax * Lmax / 2, -0.13, 'Spearman ρ', transform=tr, ha='center', va='top',
                color=VS.INK2, fontsize=plt.rcParams['axes.labelsize'])
    axes[0, 0].set_title('OOD, by effect-size threshold', loc='left', pad=12)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(hspace=0.16)
    method_legend(fig, meths, ncol=2, y=0.03)
    save(fig, out, 'fig2d_efficiency')


# ---- d -----------------------------------------------------------------------------------------
def _break_marks(a1, a2, vertical=True):
    d, kw = .012, dict(color=VS.INK2, clip_on=False, lw=1)
    if vertical:            # a1 top, a2 bottom
        a1.plot((-d, d), (-d * 2.4, d * 2.4), transform=a1.transAxes, **kw)
        a1.plot((1 - d, 1 + d), (-d * 2.4, d * 2.4), transform=a1.transAxes, **kw)
        a2.plot((-d, d), (1 - d, 1 + d), transform=a2.transAxes, **kw)
        a2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=a2.transAxes, **kw)
    else:                   # a1 left, a2 right
        a1.plot((1 - d / 2.4, 1 + d / 2.4), (-d, d), transform=a1.transAxes, **kw)
        a1.plot((1 - d / 2.4, 1 + d / 2.4), (1 - d, 1 + d), transform=a1.transAxes, **kw)
        a2.plot((-d, d), (-d, d), transform=a2.transAxes, **kw)
        a2.plot((-d, d), (1 - d, 1 + d), transform=a2.transAxes, **kw)


GPU_EST = 'ExPert_gpu_est'


def _cost(d):
    """The eval-cost table shared by E and F (sorted by compute, hidden methods dropped) + colour lookup."""
    c = pd.read_csv(d / 'fig2d_cost.csv')
    c = c[~c.method.isin(F.HIDE)].copy()
    # ExPert as a GPU job was estimated (fig2d_expert_gpu_estimate.csv: measured CPU job with only the
    # encoding swapped for GPU time) and came out no cheaper than the measured CPU job, so it is not drawn.
    c['compute_h'] = c['compute_h'].astype(float)
    c = c.sort_values('compute_h', ascending=False).reset_index(drop=True)
    NAME[GPU_EST] = 'ExPert (GPU, est.)'
    col = lambda m: PAL.get('ExPert' if m == GPU_EST else m, '0.5')
    NAME['baselines'] = 'Baselines (all)'
    PAL.setdefault('baselines', PAL.get('baseline_centroid_notarget', '0.5'))
    return c, col


def panel_d(d, out, metrics):
    """E (compute vs OOD recall@10) and F (eval compute per method)."""
    panel_e_cost(d, out, metrics)
    panel_f_compute(d, out)


def panel_e_cost(d, out, metrics):
    c, col = _cost(d)
    met = metrics[1]
    # E: compute vs OOD recall@10 (x linear, broken at the fingerprint ensemble)
    fig = plt.figure(figsize=(8.2, 7.8))
    top = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.08)

    # top: compute vs OOD recall@10 (x linear, broken at the fingerprint ensemble)
    s = c[c.method != GPU_EST].dropna(subset=[met])   # GPU estimate ~= measured at this scale: bars only
    vals = np.sort(s.compute_h.to_numpy())[::-1]
    mx, rest = vals[0], vals[1]
    axl = fig.add_subplot(top[0]); axr = fig.add_subplot(top[1], sharey=axl)
    for ax in (axl, axr):
        for _, r in s.iterrows():
            ex = r.method in ('ExPert', GPU_EST)
            hollow = bool(r.combined) or r.method == GPU_EST
            ax.scatter(r.compute_h, r[met], s=320 if ex else 120, marker='*' if r.method == GPU_EST else 'o',
                       facecolor='white' if hollow else col(r.method),
                       edgecolor=col(r.method) if hollow else VS.INK2,
                       lw=2.2 if hollow else 0.8, zorder=10 if ex else 4)
    axl.set_xlim(-rest * 0.03, rest * 1.35); axr.set_xlim(mx * 0.97, mx * 1.06)
    axl.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    axr.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='lower'))
    axl.spines['right'].set_visible(False); axr.spines['left'].set_visible(False)
    axr.tick_params(labelleft=False, left=False)
    for ax in (axl, axr):
        ax.xaxis.set_major_formatter(FuncFormatter(VS.thousands))
    _break_marks(axl, axr, vertical=False)
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None
    for ax, sel in ((axl, s.compute_h <= rest * 1.35), (axr, s.compute_h > rest * 1.35)):
        sub = s[sel]
        txt = [ax.text(r.compute_h, r[met], NAME.get(r.method, r.method) + ('*' if r.combined else ''),
                       fontweight='bold' if r.method in ('ExPert', GPU_EST) else 'normal',
                       va='center', ha='left')
               for _, r in sub.iterrows()]
        if adjust_text and txt:
            # the dots themselves are obstacles, so labels are pushed off them, not just off each other
            adjust_text(txt, x=sub.compute_h.astype(float).to_numpy(), y=sub[met].astype(float).to_numpy(),
                        ax=ax, expand=(1.25, 1.6), force_static=(0.25, 0.4), force_text=(0.3, 0.5),
                        arrowprops=dict(arrowstyle='-', color=VS.MUTED, lw=0.8))
    axl.set_ylabel(f'OOD {MET_LABEL[met]}'); axl.set_ylim(bottom=0)
    axl.set_xlabel('eval compute-hours (CPU core-h + GPU-h)', x=0.7)
    save(fig, out, 'fig2e_cost')


def panel_f_compute(d, out):
    c, col = _cost(d)
    # F: eval compute per method -- CPU solid, GPU hatched, both in the method's colour; * = train and
    # eval run as one job (their cost cannot be split, so it includes training)
    fig = plt.figure(figsize=(11, 6.6))
    bot = fig.add_gridspec(2, 1, height_ratios=[1, 2.4], hspace=0.06)
    meths = list(c.method)
    cpu, gpu = c.cpu_h.astype(float).to_numpy(), c.gpu_h.astype(float).to_numpy()
    tot = cpu + gpu
    lab = [NAME.get(m, m) + ('*' if cb else '') for m, cb in zip(meths, c.combined)]
    x = np.arange(len(meths))
    mx, rest = tot[0], tot[1]
    axt = fig.add_subplot(bot[0]); axb = fig.add_subplot(bot[1], sharex=axt)
    for ax in (axt, axb):
        ax.bar(x, cpu, color=[col(m) for m in meths], width=0.7, edgecolor=VS.INK2, lw=0.6)
        ax.bar(x, gpu, bottom=cpu, color='white', hatch='///', width=0.7,
               edgecolor=[col(m) for m in meths], lw=1.2)
        ax.grid(axis='x', visible=False)
    axt.set_ylim(mx * 0.985, mx * 1.03); axb.set_ylim(0, rest * 1.3)
    axt.yaxis.set_major_locator(MaxNLocator(nbins=2, prune='lower'))
    axb.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    axt.spines['bottom'].set_visible(False); axb.spines['top'].set_visible(False)
    axt.tick_params(labelbottom=False, bottom=False)
    for ax in (axt, axb):
        ax.yaxis.set_major_formatter(FuncFormatter(VS.thousands))
    _break_marks(axt, axb, vertical=True)
    for i, v in enumerate(tot):
        txt = f'{v:,.0f}' if v >= 10 else (f'{v:.1f}' if v >= 0.1 else f'{v:.2f}')
        (axt if v >= mx * 0.985 else axb).text(i, v, txt, ha='center', va='bottom',
                                                fontweight='bold' if meths[i] in ('ExPert', GPU_EST) else 'normal')
    axb.set_xticks(x); axb.set_xticklabels(lab, rotation=55, ha='right', rotation_mode='anchor')
    axb.set_ylabel('eval compute-hours\n(CPU core-h + GPU-h)', y=0.65)
    axb.legend(handles=[Patch(fc='0.6', ec=VS.INK2, label='CPU core-hours'),
                        Patch(fc='white', ec='0.4', hatch='///', label='GPU-hours'),
                        Line2D([], [], ls='', marker='$*$', ms=10, color=VS.INK2, label='train + eval')],
               loc='upper right')
    save(fig, out, 'fig2f_eval_compute')


# ---- e -----------------------------------------------------------------------------------------
def scaling_grid(R, S, specs, out, name, row_order=None):
    """One column of panels sharing the method rows. spec = (kind, key, title, xlabel):
    kind 'measured' -> two-point line per method on the measured scale (open = smallest class space,
    filled = largest), x N label at the right; kind 'exp' -> scaling-exponent bars of S[what == key]."""
    combined = set(R.loc[R.combined.astype(bool), 'method'])
    r_order = row_order or list(S[S.what == 'burden'].sort_values('exponent').method)
    lo_C, hi_C = R.C.min(), R.C.max()
    ex = S[S.what.isin([k for kind, k, *_ in specs if kind == 'exp'])].exponent
    xlim = (min(-0.3, np.nanmin(ex) - 0.3), max(1.4, np.nanmax(ex) + 0.55)) if len(ex) else None
    y = np.arange(len(r_order))
    fig, axes = plt.subplots(len(specs), 1, figsize=(9.5, len(specs) * (0.55 * len(r_order) + 1.7) + (0.6 if len(specs) == 1 else 0)),
                             sharey=True, squeeze=False, gridspec_kw=dict(hspace=0.42))
    num = FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}')
    for ax, (kind, key, title, xlab) in zip(axes[:, 0], specs):
        if kind == 'exp':
            e = S[S.what == key].set_index('method').reindex(r_order)
            ax.barh(y, e.exponent, color=[PAL[m] for m in r_order], edgecolor=VS.INK2, lw=0.5)
            for yy, m in enumerate(r_order):
                if np.isfinite(e.loc[m, 'exponent']):
                    ax.text(max(e.loc[m, 'exponent'], 0) + 0.04, yy, f'×{e.loc[m, "fold_change"]:.2g}',
                            va='center', fontweight='bold' if m == 'ExPert' else 'normal')
            ax.axvline(0, color=VS.INK2, lw=1); ax.axvline(1, color=VS.MUTED, lw=1.2, ls=':')
            ax.text(1.02, 0.995, 'linear', transform=ax.get_xaxis_transform(), ha='left', va='top',
                    color=VS.MUTED)
            ax.set_xlim(*xlim)
        else:
            v = R.pivot_table(index='method', columns='C', values=key).reindex(r_order)
            for yy, m in enumerate(r_order):
                a, b = v.loc[m, lo_C], v.loc[m, hi_C]
                c = PAL[m]
                if np.isfinite(a) and np.isfinite(b) and a > 0 and b / a >= 1.4:   # nearly equal: the dots touch
                    ax.annotate('', xy=(b, yy), xytext=(a, yy), zorder=3,
                                arrowprops=dict(arrowstyle='-|>', color=c, lw=3.2 if m == 'ExPert' else 1.8,
                                                mutation_scale=20, shrinkA=7, shrinkB=8))
                ax.plot(a, yy, 'o', ms=11, mfc='white', mec=c, mew=2, zorder=4)
                ax.plot(b, yy, 'o', ms=11, mfc=c, mec=VS.INK2, mew=0.8, zorder=5)
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    ax.text(max(a, b) * 1.35, yy, f'×{b / a:.2g}', va='center',
                            fontweight='bold' if m == 'ExPert' else 'normal')
            ax.set_xscale('log'); ax.xaxis.set_major_formatter(num); ax.minorticks_off()
            lo_v, hi_v = np.nanmin(v.values), np.nanmax(v.values)
            ax.set_xlim(lo_v / 1.8, hi_v * 6)
        ax.set_title(title, loc='left'); ax.set_xlabel(xlab); ax.grid(axis='y', visible=False)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels([NAME.get(m, m) + ('*' if m in combined else '') for m in r_order])
    for t, m in zip(axes[0, 0].get_yticklabels(), r_order):
        if m == 'ExPert':
            t.set_fontweight('bold'); t.set_color(VS.INK)
    axes[0, 0].invert_yaxis()
    if any(kind == 'measured' for kind, *_ in specs):
        fig.legend([Line2D([], [], marker='o', ls='', ms=11, mfc='white', mec=VS.INK2, mew=2),
                    Line2D([], [], marker='o', ls='', ms=11, mfc=VS.INK2, mec=VS.INK2)],
                   [f'{lo_C:,} perturbations', f'{hi_C:,} perturbations'],
                   loc='upper center', bbox_to_anchor=(0.5, 0.06 if len(specs) > 1 else -0.02), ncol=2)
    save(fig, out, name)




def _scaling(d):
    """(R, S, row order) for the scaling panels: rows sorted by the combined-cost exponent."""
    R = pd.read_csv(d / 'fig2e_res.csv')
    S = pd.read_csv(d / 'fig2e_slopes.csv')
    return R, S, list(S[S.what == 'burden'].sort_values('exponent').method)


def panel_g_compute(d, out):
    R, S, order = _scaling(d)
    scaling_grid(R, S, [('measured', 'per_fold', 'eval compute-h per fold', 'compute-hours (CPU core-h + GPU-h)')],
                 out, 'fig2g_scaling_compute', row_order=order)


def panel_h_jobs(d, out):
    R, S, order = _scaling(d)
    scaling_grid(R, S, [('measured', 'jobs_per_fold', 'eval SLURM jobs per fold', 'jobs')],
                 out, 'fig2h_scaling_jobs', row_order=order)


def panel_i_exponents(d, out):
    R, S, order = _scaling(d)
    scaling_grid(R, S, [('exp', 'burden', '', 'scaling exponent b  (compute-h × jobs ∝ perturbationsᵇ)')],
                 out, 'fig2i_scaling_exponents', row_order=order)


def main(a):
    d, out = Path(a.data_dir), Path(a.out_dir)
    setup(d, a.vizstyle_root)
    meta = json.loads((d / 'fig2_meta.json').read_text())
    metrics = [meta.get('f1', F.F1), F.R10, meta['mrr']]
    want = set(a.panels.split('+'))
    run = lambda *p: 'all' in want or bool(want & set(p))
    pb = d / 'pbulk'
    if run('a', 'b'):                                  # A + B: pseudobulk radars (shared ladder / per spoke)
        radar_pair(pb, out, 'fig2a_radar', 'fig2b_radar_spokes')
    if run('c'):                                       # C: OOD by shift, pseudobulk
        panel_b(pb, out, metrics)
    if run('d'):                                       # D: OOD by efficiency threshold
        panel_c(d, out)
    if run('e', 'f'): panel_d(d, out, metrics)         # E: cost vs performance, F: eval compute bars
    if run('g'): panel_g_compute(d, out)               # G/H/I: scaling
    if run('h'): panel_h_jobs(d, out)
    if run('i'): panel_i_exponents(d, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--panels', default='all', help="'+'-list of panel letters a..i or 'all'")
    ap.add_argument('--pdf', nargs='*', default=[], help='figure names to also save as .pdf')
    _a = ap.parse_args()
    PDF = set(_a.pdf)
    main(_a)
