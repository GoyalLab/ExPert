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

"""Fig. 3: zero-shot placement of unseen perturbations through the prior graph. One file per panel
plus the table behind it (<data>/fig3X_source.csv).

    a  (Fig. 3B) zero-shot per context across the 12 fold models, ID | OOD side by side, against the
       row-shuffled fill and chance (STEP=zs, fig3a_folds.sbatch, fig3a_seen.sbatch)
    b  (Fig. 3C) circular Ward tree of the trained prototypes, each zero-shot fill placed beside its
       nearest trained prototype (fig3b_layout.py, STEP=layout3)
    (the prior-pull panel, formerly 3D, is Supplementary Fig. S2B: figS2b_pull_plot.py calls
       panel_c_dumbbell below)
    d  (Fig. 3D) text prior generation: raw text source -> description writer -> embedding encoder, every
       writer x encoder pairing a ribbon coloured by zero-shot x bio quality
    e  (Fig. 3E) individual prior quality: KEGG neighbour lift (size-matched) vs efficiency-weighted
       zero-shot top-10 (x chance), per prior
    f  (Fig. 3F) prior combinations: every subset of the top-10 single pool by size k; in-sample and
       split-half held-out best, shuffled-gene twins, best single, chance

d-f read the deployment-panel prior screen (19,895-row vocabulary; never mix with the
candidates-only panel) through the screen's own loaders, so every number matches its REPORT.md.

Colour follows the entity: the prior-graph placement is ExPert's (green, as in Figs. 1 and 4),
references in neutral inks (oracle dark, shuffled null mid-grey), the shipped fill orange.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

EXPERT = None            # set in main() from fig2_common.EXPERT_COLOR (the paper-wide ExPert colour)
METRIC = 'w_zs_x'
ZS_LABEL = 'zero-shot top-10 (× chance)'
KEGG_LABEL = 'top-10 KEGG kNN (× chance)'   # share of a gene's 10 prior neighbours in a shared KEGG pathway, over the all-pairs rate
COL: dict[str, str] = {}


def _kde(x, grid, bw=None):
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(x, bw_method=bw)(grid)


# 'shipped' recipe (K562 held out); 'base' = the no-pull reference: BASE (no gate)
SHIPPED, BASE_C = 'EFFGATE3', 'BASE'
RES = 0.0040                              # pooled top-10 resolution: 2 sd, paired bootstrap on a near-replicate pair


# arms that pull EVERY prototype (prior_atom_gamma 0: soft gate, uniform weight); no offline simulation
# exists for them -- the dose x gate sweep stops at the rarest 45%
# shown: every prototype pulled at the shipped gate's PER-PROTOTYPE strength (9x the total pull);
# FULLOPEN3 (same loss weight as the gate, ~1/9 per prototype) stays out of the main panel
OPEN = (('FULLOPEN27', 'equal pull, all prototypes', 's', 'INK'),)


def arm_ranks(ranks_dir, keys, arms):
    """v017 readout ranks (0-based, same scorer as per_query) aligned to the per-query keys."""
    out = {}
    for arm in arms:
        c = pd.read_csv(Path(ranks_dir) / arm / f'{arm}_ranks.csv')
        r = pd.Series(c['rank'].to_numpy(), index=c.dataset + '|' + c.perturbation)
        out[arm] = r.reindex(keys).to_numpy()
        assert not np.isnan(out[arm]).any(), arm
    return out


def panel_c_dumbbell(out, per_query, ranks_dir, src, name='figS2b_pull_vs_base'):
    """Pooled top-10 of the trained models against base (BASE, no gate) on held-out K562 queries: the
    shipped gate and an arm that pulls every prototype, each against run-to-run noise around base.
    Formerly Fig. 3D; now Supplementary Fig. S2B, drawn by figS2b_pull_plot.py. Source table -> `src`."""
    src = Path(src)
    pq = pd.read_csv(per_query, sep='\t')       # query = (K562 screen, perturbation)
    rk = arm_ranks(ranks_dir, pq.key.to_numpy(), [a_ for a_, *_ in OPEN])
    arms = [(SHIPPED, 'gated pull (ExPert)', 'o', COL['expert'])] + \
        [(a_, lab, mk, getattr(VS, c)) for a_, lab, mk, c in OPEN]
    fig0, ax0 = plt.subplots(figsize=(4.0, 9.5))
    # run-to-run noise around base as a normal (sd = RES/2: 2 sd of a paired near-replicate
    # pair), every model marked on it with a two-sided p against that noise. Vertical: score on y,
    # density on x, labels to the right of each model's line
    from scipy.stats import norm
    n_q = len(pq)
    top = [('base', float((pq[f'rank_{BASE_C}'] < 10).mean()), VS.INK2)] + \
        [(lab, float(((rk[a_] if a_ in rk else pq[f'rank_{a_}'].to_numpy()) < 10).mean()), c)
         for a_, lab, mk, c in arms]
    b0, sd = top[0][1], RES / 2
    gy = np.linspace(b0 - 4 * sd, max(t_[1] for t_ in top) + 1.5 * sd, 400)
    pdf = norm.pdf(gy, b0, sd)
    ax0.fill_betweenx(gy, 0, pdf, color=VS.GRID, lw=0, zorder=1)
    ax0.plot(pdf, gy, color=VS.MUTED, lw=1.4, zorder=2)
    ptab = []
    vmax = max(t_[1] for t_ in top)
    for i, (lab, v, c) in enumerate(top):
        p_ = float(2 * norm.sf(abs(v - b0) / sd))
        ptab.append((lab, v, p_))
        ax0.hlines(v, 0, pdf.max() * 1.05, color=c, lw=2 if i else 1.4, ls='-' if i else ':', zorder=3)
        txt = f'{lab}\n{v:.4f}' + (f'\np = {p_:.2g}' if i else '')
        txt = txt.replace('equal pull, all prototypes', 'equal pull,\nall prototypes')
        # the best model reads upwards from its line, the next one downwards, base centred
        va = 'center' if i == 0 else ('bottom' if v >= vmax else 'top')
        ax0.text(pdf.max() * 1.1, v, txt, ha='left', va=va, color=c, linespacing=1.15,
                 fontweight='bold' if 'ExPert' in lab else 'normal')
    ax0.set_xlim(0, pdf.max() * 2.3); ax0.set_xticks([]); ax0.spines['bottom'].set_visible(False)
    ax0.set_ylim(gy[0], gy[-1] + 1.2 * sd)
    ax0.grid(False)
    ax0.set_ylabel(f'pooled top-10 ({n_q:,} held-out K562 queries)')
    VS.save(fig0, out, name)
    pd.DataFrame(ptab, columns=['model', 'top10', 'p_vs_base_noise']).assign(noise_sd=sd).to_csv(
        src / f'{name}_source.csv', index=False)


# ---- a -------------------------------------------------------------------------------------
def _zs_rows(z, ctx_map):
    """one fold record -> rows (context, split, score, shuffled-null mean, unique perturbations).
    split: OOD where the fold held the context out entirely (every dataset of it), else ID. The pooled
    rows come from the record's own seen / unseen-dataset pools."""
    unseen = set(z['unseen'])
    held_full = {lab for lab, g in ctx_map.groupby('label').dataset if set(g) <= unseen}
    out = []
    pools = {'pooled_seen_dataset': ('ID', 'all unseen perturbations'),
             'pooled_unseen_dataset': ('OOD', 'all unseen perturbations')}
    for g, v in z['real'].items():
        if g in pools:
            split, name = pools[g]
        elif g.startswith('ctx:'):
            name = g[4:]; split = 'OOD' if name in held_full else 'ID'
        else:
            continue
        sh = [r[g][METRIC] for r in z['shuffled']]
        out.append({'context': name, 'split': split, 'pooled': g in pools, 'n_unique': int(v['n_cand']),
                    'n_query': int(v['n_query']), METRIC: v[METRIC], 'random_mean': float(np.mean(sh))})
    return out


SEEN, UNSEEN = 'all seen perturbations', 'all unseen perturbations'
# Figure 3B as laid out in Fig3.pdf: fixed row order and display labels (summary rows first)
ORDER_3B = [SEEN, UNSEEN, 'K562 – essential', 'HepG2', 'Jurkat', 'RPE1', 'hESC', 'K562 – genome-wide', 'HEK293T',
            'HCT116', 'TeloHAEC', 'KOLF2.1J', 'CD4+ T – Rest', 'CD4+ T – 8 h', 'CD4+ T – 48 h']
LABEL_3B = {SEEN: 'all trained', UNSEEN: 'all zero-shot', 'K562 – essential': 'K562\nessential',
            'K562 – genome-wide': 'K562\ngenome-wide', 'CD4+ T – Rest': 'CD4+ T Rest.',
            'CD4+ T – 8 h': 'CD4+ T 8h', 'CD4+ T – 48 h': 'CD4+ T 48h'}
NBAR0 = 30              # left edge of the log-scaled unique-perturbation bars (bars start here, not at 0)


def _with_seen(t, d):
    """append the 'all seen perturbations' rows: every TRAINED perturbation of each fold's ID / OOD test
    split, scored on the zero-shot rows' table and units (fig3a_seen.py; absent until those jobs ran)"""
    import json
    rows = []
    for f in sorted((d / 'folds').glob('fig3a_seen_fold*.json')):
        r = json.load(open(f)); fold = int(f.stem.split('fold')[-1])
        for split in ('ID', 'OOD'):
            rows.append({'context': SEEN, 'split': split, 'pooled': True, 'n_unique': r[split]['n_unique'],
                         'fold': fold, METRIC: r[split]['w_zs_x'], 'random_mean': r[split]['random_mean']})
    return pd.concat([t, pd.DataFrame(rows)], ignore_index=True) if rows else t


def panel_a_side(d: Path, out, t=None, val=METRIC, log=True, xlabel='top-10 (× chance)',
                 name='fig3b_zeroshot', src=None):
    """Zero-shot top-10 per context across the fold models, ID (the context's cell line / donors were in
    training) and OOD (the fold held the context out) side by side on one row set, sharing the
    unique-perturbation bars on the right. IQR box + +-1 SD (log scale) line + median dot where a row has
    several fold models, else one dot; grey = shuffled-fill null. Source table -> `src` when given."""
    import json
    if t is None:
        cm = pd.read_csv(d / 'fig3a_contexts.tsv', sep='\t')
        recs = sorted((d / 'folds').glob('fig3a_fold*.json'))
        t = pd.concat([pd.DataFrame(_zs_rows(json.load(open(f)), cm)).assign(
            fold=int(f.stem.split('fold')[-1])) for f in recs], ignore_index=True)
        t = _with_seen(t, d)
    METRIC_ = val
    FLOOR = 0.1 if log else 0.0
    clip = (lambda v: np.maximum(np.asarray(v, float), FLOOR)) if log else (lambda v: np.asarray(v, float))
    have = set(t.context)
    order = [c for c in ORDER_3B if c in have] + sorted(have - set(ORDER_3B))   # fixed (Fig3.pdf) order
    npool = sum(c in (SEEN, UNSEEN) for c in order)
    pooled = order[0]
    y = {c: -(i + (0.5 if i >= npool else 0)) for i, c in enumerate(order)}
    fig, (axi, axo, axn) = plt.subplots(1, 3, figsize=(9.6, 0.36 * len(order) + 2.0), sharey=True,
                                        gridspec_kw={'width_ratios': [3, 3, 1.1], 'wspace': 0.07})
    for ax, split in ((axi, 'ID'), (axo, 'OOD')):
        q = t[t.split == split]
        for c in order:
            qq = q[q.context == c]
            if not len(qq):
                continue
            yy = y[c]
            if np.isfinite(qq.random_mean.mean()):
                ax.scatter(clip(qq.random_mean.mean()), yy, s=55, color=VS.MUTED, zorder=3)
            v = clip(qq[METRIC_].to_numpy())
            if len(v) > 1:
                q1, q3 = np.percentile(v, [25, 75])
                med = float(np.median(v))
                ax.add_patch(plt.Rectangle((q1, yy - 0.22), q3 - q1, 0.44, facecolor=VS.MUTED,
                                           alpha=0.25, edgecolor=VS.MUTED, lw=0.8, zorder=3))
                # +-1 SD as a dark-gray bar (thinner than the light IQR box), as in Fig3.pdf
                if log:
                    sdl = float(np.std(np.log10(v), ddof=1))
                    ax.hlines(yy, med / 10 ** sdl, med * 10 ** sdl, color=VS.INK2, lw=5, zorder=4)
                else:
                    sd_ = float(np.std(v, ddof=1))
                    ax.hlines(yy, max(med - sd_, 0), med + sd_, color=VS.INK2, lw=5, zorder=4)
                ax.scatter(med, yy, s=50, color=COL['expert'], lw=0, zorder=6)
            else:
                ax.scatter(v, yy, s=50, color=COL['expert'], lw=0, zorder=6)
        if log:
            ax.axvline(1, color=VS.MUTED, lw=1.3, ls='--')
            ax.set_xscale('log'); ax.set_xlim(FLOOR * 0.8, t[METRIC_].max() * 2)
            ax.set_xticks([FLOOR, 1, 10, 100, 1000]); ax.set_xticklabels(['0', '1', '10', '100', '1000'])
            ax.minorticks_off()
        else:
            ax.set_xlim(0, t[METRIC_].max() * 1.12)
        ax.grid(False)
        ax.set_title(split, loc='center', fontweight='bold', color=VS.INK)
    axi.set_yticks([y[c] for c in order]); axi.set_yticklabels([LABEL_3B.get(c, c) for c in order],
                                                               linespacing=0.95)
    ylo = min(y.values()) - 0.8
    axi.set_ylim(ylo, 1.1)
    fig.supxlabel(xlabel, x=0.47, y=0.035, fontsize=VS.FONT_MIN, color=VS.INK2)
    # direct labels once, on the pooled ID row
    pm = float(np.median(t[(t.split == 'ID') & (t.context == pooled)][METRIC_]))
    pr = float(t[(t.split == 'ID') & (t.context == pooled)].random_mean.mean())
    xmx = axi.get_xlim()[1]
    axi.text(pm if log else pm + 0.3 * xmx, 0.75, 'ExPert', ha='center', va='center', color=COL['expert'])
    axi.text(clip(pr) if log else max(pr, 0) + 0.12 * xmx, 0.75, 'random', ha='center', va='center',
             color=VS.INK2)
    nu = t[t.split == 'ID'].groupby('context').n_unique.median()
    for c in order:
        axn.barh(y[c], nu.get(c, np.nan) - NBAR0, left=NBAR0, height=0.5, color=VS.MUTED, zorder=3)
    axn.set_xscale('log'); axn.set_xlim(NBAR0, 30000)
    axn.set_xticks([100, 10000]); axn.set_xticklabels(['100', '10,000']); axn.minorticks_off()
    axn.grid(False); axn.tick_params(axis='y', length=0)
    axn.set_xlabel('unique\nperturbations')
    # row-group brackets left of the tick labels ("summary", "zero-shot"), placed from the drawn label extents
    fig.canvas.draw()
    inv = axi.transAxes.inverted()
    x0 = min(inv.transform(tl.get_window_extent())[0, 0] for tl in axi.get_yticklabels()) - 0.04
    groups = (('summary', [y[c] for c in order[:npool]]), ('zero-shot', [y[c] for c in order[npool:]]))
    for lab, ys in groups:
        if not ys:
            continue
        top, bot = max(ys) + 0.35, min(ys) - 0.35
        tr = matplotlib.transforms.blended_transform_factory(axi.transAxes, axi.transData)
        axi.plot([x0, x0], [bot, top], color=VS.INK2, lw=1.2, transform=tr, clip_on=False)
        axi.text(x0 - 0.03, (top + bot) / 2, lab, rotation=90, ha='right', va='center', color=VS.INK2,
                 transform=tr, clip_on=False)
    VS.save(fig, out, name)
    if src is not None:
        t.to_csv(Path(src) / 'fig3a_source.csv', index=False)


# ---- b, main-figure candidates -------------------------------------------------------------
# shades of grey only (orange = ExPert elsewhere in the figure); dark -> light = learned -> map output
KIND = {'trained': ('trained', '#3b3a38'), 'fill': ('zero-shot', '#bdbcb6')}


def _kc(k):
    return KIND[k][1]


def panel_b_tree(d: Path, out, name='fig3c_circular_tree', src=None):
    """Circular dendrogram, kind as a ring under the leaves. Joint Ward over trained + filled rows, or
    (meta tree == 'placement') Ward over trained rows with each fill beside its nearest trained one."""
    import json
    from matplotlib.collections import LineCollection
    from scipy.cluster.hierarchy import dendrogram
    lay = pd.read_parquet(d / 'fig3b_layout.parquet')
    meta = json.load(open(d / 'fig3b_layout_meta.json'))
    Z = np.load(d / 'fig3b_linkage.npz')['Z']
    n = len(lay)
    dn = dendrogram(Z, no_plot=True)
    ic, dc = np.asarray(dn['icoord']), np.asarray(dn['dcoord'])
    if meta.get('tree') == 'placement':
        # the linkage covers the TRAINED rows only (fills are grafted beside their nearest trained
        # prototype): move every trained leaf to its exact slot on the full ring, internal nodes with it
        nt = len(Z) + 1
        slot_x = 5 + 10 * lay.leaf_pos.to_numpy()[np.asarray(dn['leaves'])]
        ic = np.interp(ic, 5 + 10 * np.arange(nt), slot_x)
    # Ward heights are heavy-tailed: a sqrt keeps the fine structure near the leaves readable
    h = np.sqrt(dc / dc.max())
    R_LEAF, GAP = 1.0, 0.012
    rad = lambda hh: R_LEAF * (1 - hh) / (1 - h_cut)        # the cut sits at the ring
    ang = lambda xx: 2 * np.pi * (xx - 5) / (10 * n) * 0.985 + np.pi / 2   # small gap at top
    # the leaf level is ~n merges in a ring a few pixels deep, which renders as a solid band: draw
    # only merges above the cut; below it the ring itself carries the leaves
    h_cut = np.quantile(h[:, 1], 1 - 3000 / len(h))
    segs = []
    for xs, hs in zip(ic, h):
        if hs[1] < h_cut:
            continue
        hs = np.maximum(hs, h_cut)
        a0, a1 = ang(xs[1]), ang(xs[2])
        ra, rb, rtop = rad(hs[0]), rad(hs[3]), rad(hs[1])
        segs.append([(ra * np.cos(ang(xs[0])), ra * np.sin(ang(xs[0]))),
                     (rtop * np.cos(a0), rtop * np.sin(a0))])
        segs.append([(rb * np.cos(ang(xs[3])), rb * np.sin(ang(xs[3]))),
                     (rtop * np.cos(a1), rtop * np.sin(a1))])
        t = np.linspace(a0, a1, max(2, int(abs(a1 - a0) * 60) + 2))
        segs.append(np.c_[rtop * np.cos(t), rtop * np.sin(t)])
    fig, ax = plt.subplots(figsize=(10, 10.6))
    ax.add_collection(LineCollection(segs, colors=VS.INK2, linewidths=0.35))
    # ring: one radial tick per leaf, in leaf order (dn['leaves'] gives the row at each slot)
    leaves = np.argsort(lay.leaf_pos.to_numpy())          # row at each ring slot (== dn leaves for Ward)
    kind = lay.kind.to_numpy()[leaves]
    a = ang(5 + 10 * np.arange(n))
    r_in, r_out = R_LEAF + GAP, R_LEAF + GAP + 0.09
    ring = [[(r_in * np.cos(t), r_in * np.sin(t)), (r_out * np.cos(t), r_out * np.sin(t))]
            for t in a]
    ax.add_collection(LineCollection(ring, colors=[_kc(k) for k in kind], linewidths=0.6))
    lim = r_out * 1.03
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.axis('off')
    # the ring colours named where they can be read: a short block in the empty centre
    for i, (k, (lab, c)) in enumerate(KIND.items()):
        ax.text(0, 0.24 - 0.24 * i, f'{lab}\n(n = {int((lay.kind == k).sum()):,})', ha='center',
                va='center', color=c, fontweight='bold', linespacing=1.3,
                bbox=dict(fc='white', ec='none', pad=1.5), zorder=5)
    VS.save(fig, out, name)
    lay.groupby('cluster').agg(n=('kind', 'size'), fill_share=('kind', lambda s: (s == 'fill').mean())
                               ).to_csv(Path(src or d) / 'fig3b_tree_source.csv')


# ---- c -------------------------------------------------------------------------------------
_FLIP = (r'(?P<name>.+?)\s+self\s*\+\s*(?P<sg>\d+)\s*-\s*(?P<sl>\d+)\s*=\s*(?P<s>[+-]\d+) \[(?P<slo>[+-]\d+),(?P<shi>[+-]\d+)\]'
         r'\s+collateral\s*\+\s*(?P<cg>\d+)\s*-\s*(?P<cl>\d+)\s*=\s*(?P<c>[+-]\d+) \[(?P<clo>[+-]\d+),(?P<chi>[+-]\d+)\]'
         r'\s+total\s+(?P<t>[+-]\d+) \[(?P<tlo>[+-]\d+),(?P<thi>[+-]\d+)\]')


def read_decomp(path):
    """effgate_decomp/summary.txt -> realised (per arm) and simulated (per alpha) flip table, plus
    EFFGATE3's implied alpha. The file is the record (decomp.py); nothing here recomputes it. Also read by
    figS2_data.py (the post-hoc vs trained dose curve is Supplementary Fig. S2E)."""
    import re
    rows, alpha = [], None
    for line in open(path):
        m = re.search(r'IMPLIED ALPHA of EFFGATE3.*median ([\d.]+) IQR ([\d.]+)-([\d.]+)', line)
        if m:
            alpha = tuple(float(v) for v in m.groups())
        m = re.match(_FLIP, line.strip())
        if not m or '[' in m['name']:
            continue
        g = {k: int(v) for k, v in m.groupdict().items() if k != 'name'}
        nm = m['name'].strip()
        sim = re.match(r'sim nqxnds a=([\d.]+)', nm)
        rows.append({'kind': 'simulated' if sim else 'realised',
                     'arm': 'offline blend' if sim else nm,
                     'alpha': float(sim.group(1)) if sim else np.nan, **g})
    return pd.DataFrame(rows), alpha


# ---- d -------------------------------------------------------------------------------------
def panel_d(bio, zs, d: Path, out, src=None):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    t = PS._text_quality(bio, zs, METRIC, genept=True)
    encs = t.groupby('encoder').q.mean().sort_values(ascending=False).index.tolist()
    writers = [w for w in PS.DESCRIBERS if w in set(t.describer)] + \
        [w for w in PS.GENEPT_SRC if w in set(t.describer)]
    sources = PS.SOURCES
    src_of = {w: (sources if w in PS.DESCRIBERS else PS.GENEPT_SRC[w]) for w in writers}
    R, unit = PS._alluvial_nodes({e: int((t.encoder == e).sum()) for e in encs}, 0.012)

    def stack(h):
        gap = max((1 - sum(h.values())) / max(len(h) - 1, 1), 0.006)
        o, top = {}, 1.0
        for k, v in h.items():
            o[k] = (top - v, v); top = o[k][0] - gap
        return o
    M = stack({w: int((t.describer == w).sum()) * unit for w in writers})
    L = stack({s: sum(M[w][1] / len(src_of[w]) for w in writers if s in src_of[w])
               for s in sources})
    cmap = VS.seq_cmap()
    norm = Normalize(float(t.q.min()), float(t.q.max()))
    qw = t.groupby('describer').q.mean(); qe = t.groupby('encoder').q.mean()
    fig, ax = plt.subplots(figsize=(11.5, 14))     # portrait, not extreme
    xs, xm, xr, bw = 0.0, 0.33, 0.66, 0.03
    oL = {s: 0.0 for s in sources}; oMi = {w: 0.0 for w in writers}
    oM = {w: 0.0 for w in writers}; oR = {e: 0.0 for e in encs}
    for w in writers:
        h = M[w][1] / len(src_of[w])
        for s in src_of[w]:
            PS._ribbon(ax, xs + bw, xm, L[s][0] + oL[s], M[w][0] + oMi[w], h, VS.GRID,
                       alpha=1.0, edge=VS.SURFACE)
            oL[s] += h; oMi[w] += h
    for e in encs:
        for w in writers:
            r = t[(t.describer == w) & (t.encoder == e)]
            if not len(r):
                continue
            PS._ribbon(ax, xm + bw, xr, M[w][0] + oM[w], R[e][0] + oR[e], unit,
                       cmap(norm(float(r.q.iloc[0]))), alpha=0.95, edge=VS.SURFACE)
            oM[w] += unit; oR[e] += unit
    for s in sources:
        ax.add_patch(plt.Rectangle((xs, L[s][0]), bw, L[s][1], color=VS.INK2, zorder=3))
        ax.text(xs - 0.01, L[s][0] + L[s][1] / 2, s, ha='right', va='center')
    for w in writers:
        ax.add_patch(plt.Rectangle((xm, M[w][0]), bw, M[w][1], facecolor=cmap(norm(qw[w])),
                                   edgecolor=VS.INK2, lw=0.8, zorder=3))
        # left of the node, over the grey source ribbons: the gaps between writer nodes are too
        # thin for a label above each
        # on top of its block
        ax.text(xm + bw / 2, M[w][0] + M[w][1] + 0.004, w, ha='center', va='bottom', zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc=VS.SURFACE, ec='none', alpha=0.9))
    for e in encs:
        ax.add_patch(plt.Rectangle((xr, R[e][0]), bw, R[e][1], facecolor=cmap(norm(qe[e])),
                                   edgecolor=VS.INK2, lw=0.8, zorder=3))
        ax.text(xr + bw + 0.01, R[e][0] + R[e][1] / 2, e, ha='left', va='center')
    for x, lab in ((xs, 'raw text\nsource'), (xm, 'description\nwriter'),
                   (xr, 'embedding\nencoder')):
        ax.text(x + bw / 2, 1.035, lab, ha='center', va='bottom', color=VS.INK2,
                fontsize=VS.FONT_MIN + 1)
    ax.set_xlim(-0.2, 1.12); ax.set_ylim(-0.03, 1.13); ax.axis('off')
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal',
                      location='bottom', fraction=0.025, pad=0.02, aspect=30, shrink=0.8)
    cb.set_label(f'{ZS_LABEL} × {KEGG_LABEL}', color=VS.INK2)
    cb.outline.set_visible(False)
    VS.save(fig, out, 'fig3d_text_generation')
    t[['prior', 'describer', 'encoder', 'enc_family', 'x', 'bio', 'q']].rename(
        columns={'x': METRIC, 'bio': 'KEGG_lift10_matched', 'q': 'zs_x_bio'}).to_csv(
        Path(src or d) / 'fig3d_source.csv', index=False)


# ---- e -------------------------------------------------------------------------------------
# green (ExPert / held-out placement) and orange (fills) are taken in this figure; aqua reads as
# green next to them, so the prior families use the remaining slots
FAM = (('text', 'text', 'blue'), ('PPI (STRING)', 'STRING', 'violet'),
       ('DepMap', 'DepMap', 'magenta'))


def panel_e(bio, zs, d: Path, out, src=None):
    from scipy.stats import spearmanr
    m = zs[zs.group == PS.MAIN].merge(bio, on=['prior', 'family'])
    m = m[m.family != 'Random'].reset_index(drop=True)
    x, y = m['KEGG_lift10_matched'], m['x']
    fig, ax = plt.subplots(figsize=(9, 7))
    known = set()
    for fam, lab, c in FAM:
        s = m.family == fam; known.add(fam)
        ax.scatter(x[s], y[s], s=90, color=VS.C[c], edgecolors=VS.SURFACE, linewidths=1.5,
                   label=lab, zorder=3)
    s = ~m.family.isin(known)
    ax.scatter(x[s], y[s], s=90, color=VS.INK2, edgecolors=VS.SURFACE, linewidths=1.5,
               label='other external', zorder=3)
    ax.axhline(1.0, color=VS.MUTED, lw=1.3, ls=':')
    # trend over every prior (the 'all' Spearman): least-squares line, light-grey dotted, a visual guide
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.array([0, x.max() * 1.08])
    ax.plot(xx, b0 + b1 * xx, color='#c9c8c2', lw=1.8, ls=(0, (1, 2)), zorder=1)
    ax.set_xlim(0, x.max() * 1.08); ax.set_ylim(0, y.max() * 1.12)
    # name every non-text prior, and only the best text prior + the shipped text bank
    t = m.family == 'text'
    genept = m.prior.str.startswith('GenePT')                   # published reference: always named
    keep = set(m.loc[~t | genept, 'prior']) | {m.loc[m.x[t].idxmax(), 'prior'],
                                               'opus-4.8/qwen3-embed-8b'}
    texts = [ax.text(r['KEGG_lift10_matched'], r.x, PS.short(r.prior),
                     color=VS.INK if r.prior.startswith('GenePT') else VS.INK2,
                     fontweight='bold' if r.prior.startswith('GenePT') else 'normal')
             for _, r in m.iterrows() if r.prior in keep]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.2, 1.5),
                    arrowprops=dict(arrowstyle='-', color=VS.MUTED, lw=0.8))
    except ImportError:
        pass
    rho = lambda s: spearmanr(x[s], y[s])[0]
    ax.text(0.0, 1.03, f'Spearman  all {rho(np.ones(len(m), bool)):+.2f} (n={len(m)})\n'
            f'non-text {rho(~t):+.2f} (n={int((~t).sum())})   text {rho(t):+.2f} (n={int(t.sum())})',
            transform=ax.transAxes, ha='left', va='bottom', color=VS.INK)
    ax.set_xlabel(KEGG_LABEL)
    ax.set_ylabel(ZS_LABEL)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=4)
    VS.save(fig, out, 'fig3e_prior_quality')
    m[['prior', 'family', 'KEGG_lift10_matched', 'x', 'hits10', 'null_mean']].rename(
        columns={'x': METRIC}).to_csv(Path(src or d) / 'fig3e_source.csv', index=False)


# ---- f -------------------------------------------------------------------------------------
def panel_f(cb, pool, d: Path, out, src=None):
    g = PS.MAIN
    X, N = PS.cc(g, METRIC), PS.cc(g, METRIC, 'null')
    sizes = sorted(cb['size'].unique())
    hi = float(np.ceil(cb[X].max() * 1.1))
    grid = np.linspace(0, hi, 800)
    fig, ax = plt.subplots(figsize=(8, 2.4 + 1.2 * len(sizes)))
    best1 = cb[cb['size'] == 1][X].max()
    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(sizes):
        sub = cb[cb['size'] == s]; y0 = len(sizes) - 1 - i
        # no hue here (green / orange belong to other panels): subsets dark grey, the shuffled-gene
        # null hatched light grey
        for col, null in ((N, True), (X, False)):
            dd = _kde(sub[col], grid, 0.5) if len(sub) >= 3 else np.zeros_like(grid)
            dd = 0.6 * dd / max(dd.max(), 1e-12)
            if null:
                ax.fill_between(grid, y0, y0 + dd, facecolor='none', edgecolor=VS.MUTED,
                                hatch='////', lw=0)
                ax.plot(grid, y0 + dd, color=VS.MUTED, lw=1.3)
            else:
                ax.fill_between(grid, y0, y0 + dd, color=VS.INK2, alpha=0.25, lw=0)
                ax.plot(grid, y0 + dd, color=VS.INK2, lw=2.4)
        ax.plot(sub[X], y0 - 0.1 + rng.uniform(-0.03, 0.03, len(sub)), '|', color=VS.INK2,
                ms=6, mew=0.8)
        b = sub.loc[sub[X].idxmax()]
        ho = PS.heldout_best(sub, g, METRIC) if len(sub) > 1 else b[X]
        ax.plot([ho], [y0 + 0.03], 'o', ms=9, mfc=VS.SURFACE, mec=VS.INK, mew=2, zorder=5)
        ax.plot([b[X]], [y0 + 0.03], 'o', ms=9, color=VS.INK, mec=VS.SURFACE, mew=1.5,
                zorder=6)
        ax.text(1.01, y0 + 0.15, f'{b[X]:.1f}× | {ho:.1f}×', ha='left', va='center',
                color=VS.INK2, transform=ax.get_yaxis_transform())
        rows.append({'k': int(s), 'n_subsets': len(sub), 'max': float(b[X]), 'heldout_best': ho,
                     'median': float(sub[X].median()), 'null_median': float(sub[N].median()),
                     'best_subset': b['priors']})
    ax.text(1.01, len(sizes) - 0.2, 'max | held-out', ha='left', color=VS.INK2,
            transform=ax.get_yaxis_transform())
    ax.axvline(best1, color=VS.INK, lw=1.5, ls='--', zorder=1)
    ax.axvline(1.0, color=VS.MUTED, lw=1.3, ls=':', zorder=1)
    ax.set_yticks(range(len(sizes))); ax.set_yticklabels([f'k = {s}' for s in sizes[::-1]])
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, hi); ax.set_ylim(-0.3, len(sizes) - 0.25)
    ax.set_xlabel(ZS_LABEL)
    ax.legend(handles=[
        Patch(facecolor=VS.INK2, alpha=0.4, edgecolor=VS.INK2, label='all subsets of k priors'),
        Patch(facecolor='none', edgecolor=VS.MUTED, hatch='////', label='shuffled-gene twins'),
        Line2D([], [], marker='o', color=VS.INK, lw=0, ms=9, label='best subset'),
        Line2D([], [], marker='o', mfc=VS.SURFACE, mec=VS.INK, mew=2, lw=0, ms=9,
               label='best subset, split-half held-out'),
        Line2D([], [], color=VS.INK, lw=1.5, ls='--', label='best single prior'),
        Line2D([], [], color=VS.MUTED, lw=1.3, ls=':', label='chance (1×)')],
        loc='upper left', bbox_to_anchor=(0, -0.11), ncol=2)
    VS.save(fig, out, 'fig3f_prior_combinations')
    pd.DataFrame(rows).assign(pool='|'.join(n for _, n, _, _ in pool)).to_csv(
        Path(src or d) / 'fig3f_source.csv', index=False)


def setup(vs, prior_screen_root=None):
    """Style + module globals for the panel functions (what main() does before drawing; the figure
    notebook calls it directly). `vs` = the vizstyle module; `prior_screen_root` = the dir holding the
    prior screen's plot.py (needed by panels d-f only)."""
    global VS, PS
    if prior_screen_root is not None:
        sys.path.insert(0, str(Path(prior_screen_root)))
        import plot as PS             # the screen's loaders: same numbers as its REPORT.md
    VS = vs
    VS.apply()                        # after PS, whose import sets its own rcParams
    # paper figures: plain white background, not the vizstyle off-white surface
    VS.SURFACE = 'white'
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white',
                         # text as real <text> objects in the SVG (editable), not glyph outlines
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    # ExPert's colour is Fig. 2's (one colour for ExPert across the paper); fills are grey
    from fig2_common import EXPERT_COLOR
    COL.update({'expert': EXPERT_COLOR, 'oracle': VS.INK, 'null': VS.MUTED, 'fill': '#bdbcb6'})


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    setup(vs, a.prior_screen_root)
    d = Path(a.data_dir); d.mkdir(parents=True, exist_ok=True)
    want = set(a.panels.replace('+', ',').split(','))   # '+' survives sbatch --export
    run = lambda p: 'all' in want or p in want
    if run('b'):
        if (d / 'b_place' / 'fig3b_layout.parquet').exists():   # 3C: placement tree
            panel_b_tree(d / 'b_place', a.out_dir, 'fig3c_circular_tree')
        else:
            print('[b] no b_place/fig3b_layout.parquet: run STEP=layout3 first', flush=True)
    if run('a'):
        if (d / 'fig3a_contexts.tsv').exists():
            panel_a_side(d, a.out_dir, src=d)                # 3B: ID | OOD side by side
        else:
            print('[a] no fig3a_contexts.tsv: run STEP=zs first', flush=True)
    if run('d') or run('e'):
        bio, zs = PS.collect(a.screen_root, a.screen_tag, METRIC)
        print(f'[screen] panel {PS.PANEL}', flush=True)
        if PS.PANEL['panel'] == 'cand':
            raise SystemExit('screen tag is the 9,874-candidate panel; Fig. 3 uses the full vocabulary')
        if run('d'):
            panel_d(bio, zs, d, a.out_dir)
        if run('e'):
            panel_e(bio, zs, d, a.out_dir)
    if run('f'):
        cb, pool = PS.load_combos(f'{a.screen_root}/{a.screen_tag}')
        panel_f(cb, pool, d, a.out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--prior-screen-root', required=True, help='dir holding the screen plot.py')
    ap.add_argument('--screen-root', required=True, help='screen.py --out-dir')
    ap.add_argument('--screen-tag', required=True, help='full-vocabulary screen tag (H4_full)')
    ap.add_argument('--panels', default='all', help="comma/plus-list of a,b,d,e,f or 'all' (a = Fig. 3B, b = 3C; d-f = 3D-F)")
    main(ap.parse_args())
