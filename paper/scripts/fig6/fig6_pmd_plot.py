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

"""Fig. 6 (v2) panels from fig6_pmd.py output (--data-dir =
<paper/figures/fig6/data/pmd_mtmt>/score). One file per panel + its
source table. Fig. 6 layout: A schematic (by hand); B-D and F here; E fig6_pmd_panel_plot.py; G-H fig6_bulk_plot.py;
I fig6_flow_plot.py.

    b  single-cell DE log2FC vs ExPert loss score of the same PMD contrast, monocytes (fig6b_deg_vs_expert)
    c  monocyte UMAP of the ExPert latent, cells coloured by the uncentred OXPHOS loss score (fig6c_umap)
    d  circular trees: monocyte donors clustered label-free on the ExPert latent vs expression (fig6d_trees)
    f  PMD vs reference donors per cell type on the magnitude-aware set scores (fig6_donor_mag.py), one-sided
       donor-label permutation (fig6f_pathways; CLI key --panels e)

Trained bank, cosine readout unless noted (deflate changes complex scores by <= .004 here). No titles or legends.
"""
from __future__ import annotations

import argparse
import itertools
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig5_plot import P_BINS, p_size  # noqa: E402  (Fig. 5 dot-plot p bins, shared)

W = 7.0
# the Fig. 5 loss / gain scale (fig5_final.py): violet - grey - cyan, loss on the NEGATIVE / left end; cyan, not
# amber, for gain (amber reads as the ExPert orange)
LOSS, MID, GAIN = '#4a3aa7', '#d9d9d9', '#0e9aa7'


def lossgain_cmap():
    return LinearSegmentedColormap.from_list('lossgain', [LOSS, MID, GAIN])


def loss_score_cmap():
    """The Fig. 5 colours as a LOSS SCORE (high = violet = more loss-like than the reference, low = cyan): for values
    measured against other patients, where the low end is 'less loss-like', not a gain."""
    return LinearSegmentedColormap.from_list('lossscore', [GAIN, MID, LOSS])


def loss_score_cbar(cb, clip, label, fmt='{:g}'):
    cb.set_ticks([-clip, 0, clip])
    cb.set_ticklabels([fmt.format(-clip).replace('-', '−'), '0', fmt.format(clip)])
    cb.set_label(label)
    cb.outline.set_visible(False)


def lossgain_cbar(cb, clip, label, words=('loss', 'gain'), fmt='{:g}'):
    """Fig. 5 colourbar: three ticks, the tail words on the ends."""
    cb.set_ticks([-clip, 0, clip])
    cb.set_ticklabels([f'{words[0]}\n{fmt.format(clip)}', '0', f'{words[1]}\n{fmt.format(clip)}'])
    cb.set_label(label)
    cb.outline.set_visible(False)
# OXPHOS complexes and other mitochondrial genes (Okabe-Ito; avoids the ExPert orange and the loss/gain colours)
CAT_COL = {'CI': '#0072B2', 'CIII': '#009E73', 'CIV': '#CC79A7', 'CV': '#D55E00', 'other mito': '#56B4E9',
           'other': '#c8c8c6'}
PMD = ['PT54', 'PT43', 'PT67', 'PT46']
CT = ['Monocytes', 'Dendritic cells', 'B cells', 'NK cells', 'CD4+ T cells', 'CD8+ T cells', 'Nonclassical T cells']
CT_LAB = {'Monocytes': 'Mono', 'Dendritic cells': 'DC', 'B cells': 'B', 'NK cells': 'NK', 'CD4+ T cells': 'CD4 T',
          'CD8+ T cells': 'CD8 T', 'Nonclassical T cells': 'unconv. T'}
C_BLOCKS = ['OXPHOS', 'mito biogenesis', 'stress', 'mTOR']   # row-block labels, aligned with C_ROWS
C_ROWS = [
    [('Complex I', 'module:complex_i'), ('Complex III', 'module:complex_iii'),
     ('Complex IV', 'module:complex_iv'), ('Complex V', 'module:complex_v')],
    [('mitochondrial translation', 'reactome:Mitochondrial Translation'),
     ('mitochondrial protein import', 'reactome:Mitochondrial Protein Import'),
     ('Fe–S cluster biogenesis', 'reactome:Mitochondrial Iron-Sulfur Cluster Biogenesis'),
     ('cristae formation', 'reactome:Cristae Formation')],
    [('integrated stress response', 'module:isr'), ('ROS detoxification', 'reactome:Detoxification of Reactive Oxygen Species'),
     ('cap-dependent translation', 'module:cap_translation')],
    [('mTORC1 signalling', 'reactome:mTORC1-mediated Signalling'), ('macroautophagy', 'reactome:Macroautophagy')],
]


def colors():
    # three conditions, shown as PMD / non-PMD (the TBCK twins: diagnosed, not mitochondrial) / undiagnosed; greys for
    # the non-PMD conditions -- no orange (ExPert's colour), lighter than the dark grey used for DEG. 'TBCK' = the
    # condition key in the donor-set tables, drawn as non-PMD.
    return {'PMD': VS.C['blue'], 'non-PMD': '#cfcec8', 'TBCK': '#cfcec8', 'undiagnosed': '#a9a8a2'}


def group_of(d):
    return 'PMD' if d in PMD else ('non-PMD' if d in ('PT41', 'PT42') else 'undiagnosed')


def save(fig, out, name):
    VS.save(fig, out, name)


def clean(ax):
    ax.grid(False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_umap_density(d, out, which, name, star='PT67', uncentred=True):
    """Monocyte UMAP as filled density contours: every cell drawn in light grey, the per-cell OXPHOS-loss score
    (kNN-smoothed, B's metric) averaged on a grid (k nearest cells per grid point, grid points far from any cell
    masked) and drawn as filled contour levels; each donor named at its own density peak, PMD in bold, `star` (PT67,
    re-sequenced after ExPert flagged it) with a bold star drawn above everything."""
    from scipy.stats import gaussian_kde
    from sklearn.neighbors import NearestNeighbors
    from adjustText import adjust_text
    c = pd.read_csv(d / 'cells.csv.gz')
    c = c[c[f'umap1_{which}'].notna()].sample(frac=1, random_state=0)
    P = c[[f'umap1_{which}', f'umap2_{which}']].to_numpy()
    # uncentred (cos(z, u), no cohort-mean subtraction): one-sided, every monocyte is somewhat loss-like and the
    # patients more so -- same gradient as the centred score (Spearman 0.997), no 'gain' end
    v = (c.oxphos_raw_smooth if uncentred and 'oxphos_raw_smooth' in c else c.oxphos_smooth).to_numpy()
    fig, ax = plt.subplots(figsize=(W, W))
    ax.scatter(P[:, 0], P[:, 1], s=3, lw=0, color='#dcdcd8', zorder=1)
    pad = 0.04 * np.ptp(P, 0)
    gx, gy = np.mgrid[P[:, 0].min() - pad[0]:P[:, 0].max() + pad[0]:220j, P[:, 1].min() - pad[1]:P[:, 1].max() + pad[1]:220j]
    G = np.c_[gx.ravel(), gy.ravel()]
    nn = NearestNeighbors(n_neighbors=60).fit(P)
    dist, idx = nn.kneighbors(G)
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(v[idx].mean(1).reshape(gx.shape), sigma=3)   # smooth contour bands, no grid blockiness
    reach = np.quantile(nn.kneighbors(P[:2000], n_neighbors=60)[0][:, -1], 0.95)
    field = np.ma.masked_where(dist[:, -1].reshape(gx.shape) > reach, field)
    # OXPHOS loss score: cosine of the cell's shift from the cohort mean (all 9 donors, equal weight) with the OXPHOS
    # knockout direction; high = violet = more loss-like than the cohort mean, low = cyan = less (not a gain)
    if uncentred:
        lo, hi = np.quantile(v, [0.02, 0.98])
        lev = np.linspace(lo, hi, 9)
        cmap = LinearSegmentedColormap.from_list('lossonly', [MID, LOSS])
        cf = ax.contourf(gx, gy, np.clip(field, lo, hi), levels=lev, cmap=cmap, alpha=0.85, zorder=2)
    else:
        lim = float(np.quantile(np.abs(v), 0.98))
        lev = np.linspace(-lim, lim, 11)
        cf = ax.contourf(gx, gy, np.clip(field, -lim, lim), levels=lev, cmap=loss_score_cmap(),
                         norm=TwoSlopeNorm(0, -lim, lim), alpha=0.85, zorder=2, extend='neither')
    rng = np.random.default_rng(0)
    texts, pts = [], []
    for dn, s_ in c.groupby('donor'):
        q = s_[[f'umap1_{which}', f'umap2_{which}']].to_numpy()
        if len(q) < 50:
            continue
        sub = q[rng.choice(len(q), min(len(q), 3000), replace=False)]
        k = gaussian_kde(sub.T, bw_method=0.3)
        peak = sub[np.argmax(k(sub.T))]
        lab = f'{dn} ★' if dn == star else dn
        texts.append(ax.text(*peak, lab, ha='center', va='center', fontsize=VS.FONT_MIN - 2,
                             fontweight='bold' if dn in PMD else 'normal', color=VS.INK if dn in PMD else VS.INK2,
                             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9), zorder=9))
        pts.append(peak)
    adjust_text(texts, ax=ax, force_text=(0.4, 0.6), expand=(1.2, 1.4),
                arrowprops=dict(arrowstyle='-', color=VS.INK2, lw=0.8))
    bare(ax)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
    if uncentred:
        cb.set_ticks([lev[0], lev[-1]]); cb.set_ticklabels([f'{lev[0]:.2f}', f'{lev[-1]:.2f}'])
        cb.set_label('OXPHOS loss score'); cb.outline.set_visible(False)
    else:
        loss_score_cbar(cb, lim, 'OXPHOS loss score', fmt='{:.2f}')
    save(fig, out, name)


def _polar_tree(ax, L, labels, star='PT67', ring=(1.06, 0.13), label_r=1.27):
    """Circular dendrogram: root at the centre, leaves on the unit circle; an outer ring of condition boxes where
    neighbouring leaves of the same condition merge into ONE box, so how connected each group is reads directly."""
    from scipy.cluster.hierarchy import dendrogram
    dd = dendrogram(L, no_plot=True, labels=labels)
    n = len(labels)
    xs = np.array(dd['icoord']); ys = np.array(dd['dcoord'])
    hmax = ys.max()
    step = 2 * np.pi / n
    ang = lambda x: (x - 5.0) / 10.0 * step               # leaves at 5, 15, ... -> evenly round the full circle
    rad = lambda h: 1.0 - h / hmax * 0.95
    for xi, yi in zip(xs, ys):
        for (x0, y0), (x1, y1) in (((xi[0], yi[0]), (xi[1], yi[1])), ((xi[3], yi[3]), (xi[2], yi[2]))):
            ax.plot([ang(x0), ang(x1)], [rad(y0), rad(y1)], color=VS.INK2, lw=1.4, solid_capstyle='round')
        t = np.linspace(ang(xi[1]), ang(xi[2]), 40)
        ax.plot(t, np.full_like(t, rad(yi[1])), color=VS.INK2, lw=1.4)
    cc = colors()
    order = list(dd['ivl'])
    grp = [group_of(d_) for d_ in order]
    # runs of the same condition around the circle (the last run wraps onto the first)
    runs, k0 = [], 0
    for k in range(1, n + 1):
        if k == n or grp[k] != grp[k0]:
            runs.append((k0, k - 1, grp[k0]))
            k0 = k
    if len(runs) > 1 and runs[0][2] == runs[-1][2]:
        a0, a1, g_ = runs.pop()
        b0, b1, _ = runs.pop(0)
        runs.append((a0, b1 + n, g_))
    gap = 0.08 * step
    for k0, k1, g_ in runs:
        th0, th1 = (k0 - 0.5) * step + gap / 2, (k1 + 0.5) * step - gap / 2
        ax.bar((th0 + th1) / 2, ring[1], width=th1 - th0, bottom=ring[0], color=cc[g_], edgecolor='white', lw=0,
               align='center', zorder=4)
    for i, lab in enumerate(order):
        th = i * step
        rot = np.degrees(th)
        flip = 90 < rot % 360 < 270
        ax.text(th, label_r, f'{lab} ★' if lab == star else lab, rotation=rot + (180 if flip else 0),
                rotation_mode='anchor', ha='right' if flip else 'left', va='center', fontsize=VS.FONT_MIN - 3,
                color=VS.INK if lab in PMD else VS.INK2, fontweight='bold' if lab in PMD else 'normal', zorder=6)
        ax.plot([th, th], [1.0, ring[0]], color=VS.INK2, lw=1.0)
    ax.set_ylim(0, ring[0] + ring[1])
    ax.set_axis_off()


def panel_trees(d, rna_dir, out, space, name='fig6e_clustering'):
    """Panel E as two circular dendrograms: monocyte donors clustered label-free on the ExPert latent (fig6_pmd.py
    score clustering) and on expression (fig6_rna.py sccluster, 2,000 HVGs), each with its 2-cluster ARI."""
    cl = pd.read_csv(d / 'clustering.csv')
    r1 = cl[(cl.celltype == 'Monocytes') & (cl.space == space)].iloc[0]
    rc = pd.read_csv(rna_dir / 'rna_sc_clustering.csv')
    r2 = rc[(rc.celltype == 'Monocytes') & (rc.genes == 'hvg')].iloc[0]
    # supervised companion (fig6_supervised.py): held-out AUC of PMD vs non-PMD, same features as the trees
    sup = pd.read_csv(Path(rna_dir).parent / 'supervised' / 'supervised_auc.csv')
    sup = sup[sup.celltype == 'Monocytes'].set_index('features')
    fig, axs = plt.subplots(1, 2, figsize=(W + 1, (W + 1) / 2 + 1.1), subplot_kw={'projection': 'polar'})
    for ax, r, lab, feat in ((axs[0], r1, 'ExPert', 'expert'), (axs[1], r2, 'expression (2k HVG)', 'expression')):
        _polar_tree(ax, np.array(json.loads(r.linkage)), r.donors.split())
        fmt = lambda p: f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
        sp = sup.loc[feat]
        # clustering: label-free ARI + exact relabelling p; supervised: leave-one-donor-out AUC + its exact p
        ax.text(0.5, -0.3, (f'{lab}\nARI {r.ari:.2f}, {fmt(r.p)}\n'
                            f'held-out AUC {sp.loo_auc:.2f}, {fmt(sp.loo_p)}').replace('-0.', '−0.'),
                transform=ax.transAxes, ha='center', va='top', color=VS.INK, fontsize=VS.FONT_MIN - 2, linespacing=1.4)
    cc = colors()
    h = [plt.Rectangle((0, 0), 1, 1, color=cc[g_], label=g_) for g_ in ('PMD', 'non-PMD', 'undiagnosed')]
    fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.97),
               handlelength=1.2, columnspacing=1.5)   # three classes, each covering several donors
    fig.subplots_adjust(wspace=0.7)
    save(fig, out, name)
    print(f'[e] trees: ExPert ARI {r1.ari:.2f} p {r1.p:.3f} AUC {sup.loc["expert"].split_auc:.2f} | expression ARI '
          f'{r2.ari:.2f} p {r2.p:.3f} AUC {sup.loc["expression"].split_auc:.2f}', flush=True)


def _pmd_vs_nonpmd(Z, one_sided=False, scaled=False):
    """Per (family, set, celltype): mean donor set z of PMD minus non-PMD donors and its exact donor-label permutation
    p (every relabelling with the same number of PMD donors); two-sided, or one-sided in the loss direction (PMD more
    loss-like) when the direction is the prespecified hypothesis. scaled: the difference is divided by the SD of the
    reference donors (z vs reference donors; recomputed for every relabelling) -- for magnitude-aware projections
    (fig6_donor_mag.py), where the PMD mean of each donor's z vs the references equals this effect."""
    out = []
    for (fam, st, ct), x in Z.groupby(['family', 'set', 'celltype']):
        v = x.set_index('donor').z.dropna()
        ds = list(v.index)
        k = sum(d in PMD for d in ds)
        if k < 2 or len(ds) - k < 2:
            continue
        a = v.to_numpy()
        lab = np.array([d in PMD for d in ds])
        stat = (lambda x, r: (x.mean() - r.mean()) / r.std(ddof=1)) if scaled else (lambda x, r: x.mean() - r.mean())
        eff = stat(a[lab], a[~lab])
        null = np.array([stat(a[list(c)], np.delete(a, list(c)))
                         for c in itertools.combinations(range(len(ds)), k)])
        out.append(dict(family=fam, set=st, celltype=ct, effect=eff, n_pmd=k, n_ref=len(ds) - k, n_perm=len(null),
                        p=float(np.mean(null >= eff - 1e-12)) if one_sided
                        else float(np.mean(np.abs(null) >= abs(eff) - 1e-12))))
    return pd.DataFrame(out)


def panel_pmd_rows(Z, out, clip, cts, cb_label, ends, sign, name, src_dir, loss_score=False, blocks=None,
                   one_sided=False, summary=False, boxed=None, scaled=False, header='PMD vs non-PMD'):
    """C_ROWS (OXPHOS, mito biogenesis, stress, mTOR) x cell types, PMD vs non-PMD on per-donor set z (each donor vs
    the pooled non-PMD cells): colour = difference of mean set z, oriented by `sign` so loss / down is negative
    (violet) and gain / up positive (amber), as Fig. 5; size = exact two-sided donor-label permutation p bin (Fig. 5
    bins, ring at p < 0.05). `ends` = the colourbar's end words, e.g. ('loss', 'gain') or ('down', 'up')."""
    blocks = blocks or list(zip(C_BLOCKS, C_ROWS))              # [(block label, [(row label, family:set)])]
    Z = Z.assign(key=Z.family + ':' + Z.set)
    rows = [r for _, blk in blocks for r in blk]
    T = _pmd_vs_nonpmd(Z[Z.key.isin([k for _, k in rows])].drop(columns=['family']).rename(columns={'key': 'family'})
                       .assign(set=''), one_sided=one_sided, scaled=scaled)
    x = T.set_index(['family', 'celltype'])
    M = np.full((len(rows), len(cts)), np.nan)
    Pm = np.ones(M.shape)
    for i, (_, key) in enumerate(rows):
        for j, ct in enumerate(cts):
            if (key, ct) in x.index:
                M[i, j], Pm[i, j] = sign * x.loc[(key, ct), 'effect'], x.loc[(key, ct), 'p']
    spans, k0 = [], 0
    for lab, blk in blocks:
        spans.append((k0, k0 + len(blk) - 1, lab))
        k0 += len(blk)
    cmap = loss_score_cmap() if loss_score else lossgain_cmap()
    norm = TwoSlopeNorm(0, -clip, clip)
    nr, xs = len(rows), np.arange(len(cts), dtype=float)
    fig, ax = plt.subplots(figsize=(0.42 * len(xs) + 1.4, 0.42 * nr + 2.4))   # dense: label width comes on top
    if summary:
        # per-column summary above the grid: rows where PMD is significantly more loss-like (p < 0.05)
        nsig = [int(sum(1 for i in range(nr) if np.isfinite(M[i, j]) and Pm[i, j] < 0.05 and sign * M[i, j] > 0))
                for j in range(len(cts))]
        axs = ax.inset_axes([0, 1.0 + 0.2 / nr, 1, 1.6 / nr], sharex=ax)   # in row units: between grid and header
        axs.bar(xs, nsig, width=0.55, color=LOSS)
        for x_, v in zip(xs, nsig):
            axs.text(x_, v, f'{v}', ha='center', va='bottom', fontsize=VS.FONT_MIN - 4, color=VS.INK2)
        axs.set_ylim(0, max(nsig) * 1.35 + 0.5)
        axs.set_yticks([]); axs.tick_params(axis='x', labelbottom=False, length=0)
        axs.grid(False)
        for sp_ in axs.spines.values():
            sp_.set_visible(False)
        axs.set_ylabel('rows\np < 0.05', rotation=0, ha='right', va='center', fontsize=VS.FONT_MIN - 4)
    yy, xx = np.nonzero(np.isfinite(M))
    vals = np.clip(M[yy, xx], -clip, clip)
    sz = np.array([p_size(Pm[i, j]) * 0.75 for i, j in zip(yy, xx)])
    sig = sz > P_BINS[-1][2] * 0.75
    sc = ax.scatter(xs[xx[sig]], yy[sig], s=sz[sig], c=vals[sig], cmap=cmap, norm=norm, edgecolor=VS.INK, lw=0.7,
                    zorder=3)
    ax.scatter(xs[xx[~sig]], yy[~sig], s=sz[~sig], c=vals[~sig], cmap=cmap, norm=norm, lw=0, zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels([CT_LAB[c] for c in cts], rotation=90)
    ax.set_yticks(range(nr)); ax.set_yticklabels([r[0] for r in rows])
    bi = [i for i, r in enumerate(rows) if r[0] in (boxed or {})]   # rows holding a patient's suggested lesion gene
    for i in bi:
        ax.scatter(xs, np.full(len(xs), i), s=380, marker='s', facecolor='none', edgecolor=VS.INK, lw=1.0, zorder=4)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7); ax.set_ylim(nr - 0.5, -0.5)
    ax.grid(False)
    for y in range(nr):
        ax.axhline(y, color=VS.GRID, lw=0.8, zorder=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(length=0)
    for lo, hi, _ in spans[1:]:
        ax.axhline(lo - 0.5, color=VS.INK2, lw=1.0)
    off = (2.05 if summary else 0.45) / nr                      # header bar above the summary bars, if any
    tr = ax.get_xaxis_transform()
    ax.add_patch(plt.Rectangle((xs[0] - 0.45, 1.0 + off), len(xs) - 0.1, 0.3 / nr, color=colors()['PMD'],
                               clip_on=False, transform=tr))
    ax.text(xs.mean(), 1.0 + off + 0.45 / nr, header, ha='center', va='bottom', color=VS.INK,
            fontweight='bold', transform=tr)
    ax_w = ax.get_position().width * fig.get_size_inches()[0]
    fig.canvas.draw()                                          # bracket just left of the widest row label
    lab_w = max(t.get_window_extent().width for t in ax.get_yticklabels()) / fig.dpi
    xb = -(lab_w + 0.2) / ax_w
    for lo, hi, lab in spans:
        if not lab:                                            # unlabelled block (a single row)
            continue
        ax.annotate('', xy=(xb, lo - 0.4), xytext=(xb, hi + 0.4), xycoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color=VS.INK2, lw=1.5))
        ax.text(xb - 0.1 / ax_w, (lo + hi) / 2, lab, transform=ax.get_yaxis_transform(), rotation=90, ha='right',
                va='center', color=VS.INK2)
    h = [Line2D([], [], ls='', marker='o', ms=np.sqrt(s_ * 0.75), mfc=VS.MUTED,
                mec=VS.INK if s_ > P_BINS[-1][2] else 'none', mew=0.7, label=l) for _, l, s_ in P_BINS]
    if bi:
        h.append(Line2D([], [], ls='', marker='s', ms=np.sqrt(380), mfc='none', mec=VS.INK, mew=1.0,
                        label='holds a suggested\nlesion gene'))
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(1.0, 1.0), title='donor-label\npermutation p' + ('\n(one-sided)' if one_sided else ''),
              handletextpad=0.4, labelspacing=1.1, borderaxespad=0)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.03, pad=2.0 / (0.42 * nr + 2.4), aspect=18)
    if loss_score:
        loss_score_cbar(cb, clip, cb_label)
    else:
        lossgain_cbar(cb, clip, cb_label, words=ends)
    save(fig, out, name)
    T.rename(columns={'family': 'set_key'}).drop(columns='set').to_csv(src_dir / f'{name}_source.csv', index=False)
    t = T[T.celltype.isin(cts) & (T.p < 0.05)]
    print(f'[{name}] p < .05: ' + ', '.join(f'{r.family.split(":")[1][:22]}|{CT_LAB[r.celltype]} {sign * r.effect:+.1f}'
                                            f' p {r.p:.3f}' for r in t.itertuples()), flush=True)


# pathway groups of the curated rows (C_ROWS / C_BLOCKS), Okabe-Ito, no ExPert orange
PATH_COL = {'OXPHOS': '#0072B2', 'mito biogenesis': '#009E73', 'stress': '#CC79A7', 'mTOR': '#D55E00'}


# colour groups for the gene-level C panels, in PRIORITY order (a gene in several sets takes the first group): mTOR =
# mTORC1-mediated signalling only -- macroautophagy (107 genes, median rank ~3,400) would bury its small core, and the
# eIF4 effectors (EIF4G1 / EIF4B / EIF4E) belong to mTORC1 before cap-dependent translation
PATH_GROUPS = [('OXPHOS', [k for _, k in C_ROWS[0]]), ('mTOR', ['reactome:mTORC1-mediated Signalling']),
               ('mito biogenesis', [k for _, k in C_ROWS[1]]), ('stress', [k for _, k in C_ROWS[2]])]


def pathway_group(genes, a):
    """Gene -> first PATH_GROUPS group whose sets contain it ('' = none)."""
    from types import SimpleNamespace
    from fig6_pmd import load_sets
    sets = load_sets(SimpleNamespace(modules=a.modules, reactome_gmt=a.reactome_gmt), list(genes))
    out = {}
    for lab, keys in PATH_GROUPS:
        for key in keys:
            for g in sets.get(tuple(key.split(':', 1)), []):
                out.setdefault(g, lab)
    return out


def _signed_rank(v):
    """Rank from the centre outwards: the least negative value = -1 ... the most negative = -n_neg, the least positive
    = +1 ... the most positive = +n_pos (so 0 sits between the last-ranked down / loss and up / gain genes)."""
    from scipy.stats import rankdata
    v = np.asarray(v, float)
    r = np.zeros(len(v))
    neg, pos = v < 0, v >= 0
    r[neg] = -rankdata(-v[neg], method='ordinal')
    r[pos] = rankdata(v[pos], method='ordinal')
    return r


def _top_table(x, a, dest, n):
    """C's top-n ExPert loss genes with their single-cell DE (log2FC, Wilcoxon score, BH-adjusted p from scanpy),
    the pathway group (PATH_GROUPS priority), every curated set the gene is in, and PMD / no PMD (PMD = the groups
    highlighted in C: OXPHOS, mito biogenesis, mTOR; boolean)."""
    from types import SimpleNamespace
    from fig6_pmd import load_sets
    sets = load_sets(SimpleNamespace(modules=a.modules, reactome_gmt=a.reactome_gmt), list(x.gene))
    lab_of = {k: l for blk in C_ROWS for l, k in blk}
    member = {}
    for blk in C_ROWS:
        for _, key in blk:
            for g in sets.get(tuple(key.split(':', 1)), []):
                member.setdefault(g, []).append(lab_of[key])
    grp = pathway_group(x.gene, a)
    t = x.nsmallest(n, 'ex').copy()
    t.insert(0, 'expert_rank', np.arange(1, len(t) + 1))
    t['pathway_group'] = t.gene.map(grp).fillna('')
    t['pathways'] = t.gene.map(lambda g: '; '.join(member.get(g, [])))
    tags = complex_of(list(t.gene), a.modules)            # finer: complex / collaborator module / curated extra
    t['module'] = t.gene.map(tags).fillna('')
    t['PMD'] = t.pathway_group.isin(['OXPHOS', 'mito biogenesis', 'mTOR'])     # the groups highlighted in C
    t = t.rename(columns={'ex': 'expert_score', 'de_logfc': 'de_log2fc', 'de_score': 'de_wilcoxon_score',
                          'de_padj': 'de_padj_bh'})
    cols = ['expert_rank', 'gene', 'expert_score', 'zero_shot', 'de_log2fc', 'pct_pmd', 'pct_ref',
            'de_log2fc_scanpy', 'de_wilcoxon_score', 'de_padj_bh',
            'pathway_group', 'pathways', 'module', 'PMD']
    dest.mkdir(exist_ok=True)
    f = dest / f'fig6b_top{n}_expert.csv'
    t[[c for c in cols if c in t.columns]].to_csv(f, index=False)
    print(f'[c] module tags: {t.module.replace("", "none").value_counts().to_dict()}', flush=True)
    print(f'[c] wrote {f}: {len(t)} genes, PMD {int(t.PMD.sum())}, groups '
          f'{t.pathway_group.replace("", "none").value_counts().to_dict()}', flush=True)


def panel_c_quadrant(d, out, ct, contrast, a, mode='rank', name='fig6c_deg_vs_expert_quadrant', src_dir=None):
    """Every trained prototype gene of the cell type's PMD vs non-PMD contrast (cells pooled): x = single-cell DE
    log2 fold change (scanpy Wilcoxon), y = ExPert score (cosine; loss < 0 < gain), as signed ranks from the centre
    (mode='rank') or as values (mode='value'); pathway genes of B / F coloured by group, margins = their positions."""
    g = pd.read_csv(d / 'group_genes.csv.gz')
    x = g[(g.bank == 'trained') & (g.celltype == ct) & (g.contrast == contrast) & g.de_logfc.notna()].copy()
    lf = Path(a.rna_dir) / 'rna_sc_lfc.csv.gz'
    if lf.exists():                                       # Seurat-style log2FC (pseudocount 1 on CP10k), not scanpy's
        L = pd.read_csv(lf)
        L = L[L.celltype == ct].drop_duplicates('gene').set_index('gene')
        x['de_logfc_scanpy'] = x.de_logfc
        x['de_logfc'] = x.gene.map(L.log2fc)
        x['pct_pmd'], x['pct_ref'] = x.gene.map(L.pct_pmd), x.gene.map(L.pct_ref)
        x = x[x.de_logfc.notna()].copy()
    x['ex'] = -x.cos                                      # loss (resembles losing the gene) negative
    if mode == 'rank':
        x['X'], x['Y'] = _signed_rank(x.de_logfc), _signed_rank(x.ex)
        xl, yl = 'DE log2FC rank', 'ExPert rank'
    else:
        x['X'], x['Y'] = x.de_logfc, x.ex
        # y = -cosine: knockout-like (loss) points DOWN, as Fig. 5's loss-negative convention
        xl, yl = 'DE log2 fold change (single cell)', 'ExPert score'
    grp = pathway_group(x.gene, a)
    # the B / F pathway group; stress is not shown here
    x['group'] = x.gene.map(grp).fillna('').replace('stress', '')
    PMD_LAB = 'PMD-associated (OXPHOS, mito, mTOR)'
    pmd = set()
    if getattr(a, 'pmd_genes', None):
        # the collaborators' PMD gene list: drawn blue too (some sit outside the three groups, e.g. OPA1, DNM1L) and
        # the a.n_lab_list most loss-like of them are the named genes
        pmd = set(pd.read_csv(a.pmd_genes).iloc[:, 0].astype(str).str.strip())
        x.loc[x.gene.isin(pmd) & (x.group == ''), 'group'] = 'list'
        print(f'[{name}] PMD list: {len(pmd)} genes, {int(x.gene.isin(pmd).sum())} in the plot, '
              f'{int((x.group == "list").sum())} outside the pathway groups', flush=True)
    # ONE colour for every highlighted gene: the OXPHOS blue used for these pathways throughout the figure
    x['path'] = np.where(x.group != '', PMD_LAB, '')
    cats, pc = [PMD_LAB], {PMD_LAB: PATH_COL['OXPHOS']}
    fig = plt.figure(figsize=(W, W))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_r = fig.add_subplot(gs[1, 1], sharey=ax)
    if mode == 'value':
        lim = float(np.abs(x.X).max()) * 1.05
        ax.set_xlim(-lim, lim)
        for v in (-a.lfc_cut, a.lfc_cut):                 # the usual single-cell DEG cutoff, |log2FC| >= 0.25
            ax.axvline(v, color=VS.INK2, ls='--', lw=0.9, zorder=0)
    o = x[x.path == '']
    ax.scatter(o.X, o.Y, s=4, color=CAT_COL['other'], lw=0, zorder=1)
    for c in cats:
        y = x[x.path == c]
        ax.scatter(y.X, y.Y, s=18, color=pc[c], edgecolor='white', lw=0.4, zorder=3)
    ax.axhline(0, color=VS.MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=VS.MUTED, lw=0.8, zorder=0)
    # the top a.n_lab genes of every pathway group by ExPert score (strongest loss)
    if pmd:
        lab_ = x[x.gene.isin(pmd)].nsmallest(a.n_lab_list, 'ex')
    else:
        lab_ = pd.concat([x[x.group == c].nsmallest(a.n_lab, 'ex') for c in PATH_COL if (x.group == c).any()])
    # named in one stacked column at the left edge of the loss half, in score order (so leader lines do not cross)
    lab_ = lab_.sort_values('Y', ascending=False)
    ys = np.linspace(0.6, 0.03, len(lab_)) if len(lab_) > 1 else np.array([0.25])
    for yf, r in zip(ys, lab_.itertuples()):
        ax.annotate(r.gene, xy=(r.X, r.Y), xytext=(0.03, yf), textcoords='axes fraction', ha='left', va='center',
                    color=VS.INK, fontstyle='italic', fontsize=VS.FONT_MIN - 7, zorder=6,
                    arrowprops=dict(arrowstyle='-', color=VS.INK2, lw=0.4, alpha=0.7, shrinkA=1, shrinkB=2,
                                    relpos=(1, 0.5)))
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    # requested in-panel note: Spearman over every gene, magnitude and signed (ExPert loss < 0, as plotted)
    from scipy.stats import spearmanr
    r_abs = spearmanr(x.de_logfc.abs(), x.ex.abs()).statistic
    r_sig = spearmanr(x.de_logfc, x.ex).statistic
    ax.text(0.98, 0.98, f'Spearman ρ = {r_sig:.2f}'.replace('-', '−'), transform=ax.transAxes, ha='right', va='top',
            color=VS.INK2, fontsize=VS.FONT_MIN - 4)
    print(f'[{name}] Spearman |log2FC| vs |ExPert| {r_abs:.3f}, signed {r_sig:.3f} (n={len(x)})', flush=True)
    clean(ax)
    h = [Line2D([], [], ls='', marker='o', ms=6, color=pc[c], label=c) for c in cats]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.6, -0.14), ncol=1, frameon=False,
              handletextpad=0.2, columnspacing=1.2, borderaxespad=0)   # classes cover many dots (legend rule)
    pw = x[x.path != '']
    for axm, col, orient in ((ax_t, 'X', 'vertical'), (ax_r, 'Y', 'horizontal')):
        lo, hi = (ax.get_xlim() if col == 'X' else ax.get_ylim())
        tf = (ax.xaxis if col == 'X' else ax.yaxis).get_transform()   # equal-width bins in the DRAWN scale
        bins = tf.inverted().transform(np.linspace(*tf.transform(np.array([lo, hi])), 25))
        axm.hist([pw[pw.path == c][col] for c in cats], bins=bins, stacked=True, color=[pc[c] for c in cats],
                 orientation=orient, lw=0)
        (axm.axvline if orient == 'vertical' else axm.axhline)(0, color=VS.MUTED, lw=0.8)
        if mode == 'value' and col == 'X':
            for v in (-a.lfc_cut, a.lfc_cut):
                axm.axvline(v, color=VS.INK2, ls='--', lw=0.9)
        axm.grid(False)
        for sp_ in ('top', 'right'):
            axm.spines[sp_].set_visible(False)
        if orient == 'vertical':                          # count axis on the left, shared x hidden
            axm.spines['bottom'].set_visible(False)
            axm.tick_params(axis='x', labelbottom=False, length=0)
            axm.set_ylabel('genes')
        else:                                             # count axis at the bottom, shared y hidden
            axm.spines['left'].set_visible(False)
            axm.tick_params(axis='y', labelleft=False, length=0)
            axm.set_xlabel('genes')
        cnt = axm.yaxis if orient == 'vertical' else axm.xaxis   # count axis only: the other one is shared
        cnt.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, integer=True))
    # direction arrows beside the marginals (out of the gene-label column): DE down / up along x, loss / gain along y
    fs = VS.FONT_MIN - 2
    ax_t.text(0.0, 1.02, '← down', transform=ax_t.transAxes, ha='left', va='bottom', fontsize=fs, color=VS.INK2)
    ax_t.text(1.0, 1.02, 'up →', transform=ax_t.transAxes, ha='right', va='bottom', fontsize=fs, color=VS.INK2)
    ax_r.text(1.06, 1.0, 'gain →', transform=ax_r.transAxes, rotation=90, ha='left', va='top', fontsize=fs,
              color=VS.INK2)
    ax_r.text(1.06, 0.0, '← loss', transform=ax_r.transAxes, rotation=90, ha='left', va='bottom', fontsize=fs,
              color=VS.INK2)
    save(fig, out, name)
    x.to_csv(Path(src_dir or d) / f'{name}_source.csv', index=False)   # src_dir: notebook (default: the score dir)
    if mode == 'value':
        _top_table(x, a, Path(a.src_dir), a.n_table)                # the main B panel's table
    q = pd.crosstab(np.sign(pw.X), np.sign(pw.Y))
    print(f'[{name}] pathway genes {len(pw)}: DE down & ExPert loss {int(((pw.X < 0) & (pw.Y < 0)).sum())}, '
          f'DE up & loss {int(((pw.X > 0) & (pw.Y < 0)).sum())}, DE down & gain {int(((pw.X < 0) & (pw.Y > 0)).sum())}, '
          f'DE up & gain {int(((pw.X > 0) & (pw.Y > 0)).sum())}', flush=True)


COMPLEX_RE = [('CI', r'^(NDUF|MT-ND)'), ('CII', r'^SDH[A-D]'), ('CIII', r'^(UQCR|MT-CYB$|CYC1$)'),
              ('CIV', r'^(COX\d|MT-CO)'), ('CV', r'^(ATP5|MT-ATP)'), ('mito-ribosome', r'^MRP[SL]')]
MODULE_TAG = {'complex_i': 'CI', 'complex_ii': 'CII', 'complex_iii': 'CIII', 'complex_iv': 'CIV', 'complex_v': 'CV',
              'mitoribosome': 'mito-ribosome', 'iron_sulfur': 'Fe–S', 'mito_import': 'mito import',
              'mtdna_maintenance': 'mtDNA', 'mtdna_expression': 'mtDNA', 'coq_biosynthesis': 'CoQ',
              'pdh_tca': 'PDH / TCA', 'mito_dynamics': 'mito dynamics', 'mtor_autophagy': 'mTOR',
              'cap_translation': 'cap translation', 'isr': 'ISR'}


# mitochondrial genes outside the collaborator modules (curated by function, not by rank): LIAS = lipoic acid
# synthase (Fe-S enzyme), TAFAZZIN = cardiolipin remodelling, CCDC58 = MIX23 (mitochondrial protein import)
EXTRA_TAG = {'LIAS': 'Fe–S / lipoylation', 'LIPT1': 'Fe–S / lipoylation', 'LIPT2': 'Fe–S / lipoylation',
             'TAFAZZIN': 'cardiolipin', 'CCDC58': 'mito import'}


def complex_of(genes, modules):
    """Gene -> complex / module tag: OXPHOS complexes by family name first, then the collaborator modules."""
    import re
    mod = {}
    if modules:
        for line in open(modules):
            if line.startswith('#') or not line.strip():
                continue
            m, g = line.rstrip('\n').split('\t')[:2]
            if m in MODULE_TAG:
                mod.setdefault(g, MODULE_TAG[m])
    out = {}
    for g in genes:
        tag = next((t for t, rx in COMPLEX_RE if re.match(rx, g)), None) or EXTRA_TAG.get(g)
        if tag is None:
            tag = mod.get(g) or next((v for p, v in mod.items() if p.endswith('*') and g.startswith(p[:-1])), '')
        out[g] = tag
    return out


def setup(vizstyle_root):
    """Load and apply the paper style (sets this module's VS); returns it."""
    global VS
    sys.path.insert(0, str(Path(vizstyle_root)))
    from src import vizstyle as VS
    VS.apply()
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    return VS


def panel_f(a, out, cts, src_dir):
    """Fig. 6F: PMD vs reference donors per cell type on the magnitude-aware set scores (--donor-mag)."""
    # loss score = PMD minus reference donors (non-PMD + undiagnosed, fig6_pmd.CTRL) on the magnitude-aware
    # per-donor projection from one common baseline (fig6_donor_mag.py; the Fig. 6C score per set, as panel I),
    # in units of the reference donors' SD. Rows: OXPHOS complexes defined STRUCTURALLY (MitoCarta3.0 subunits +
    # assembly factors; PanelApp keeps only subunits with reported patients and drops the strongest ExPert hits,
    # e.g. UQCRC1, NDUFB4, COX6C, ATP5PD), the PanelApp groups (fig6_pmd_panel.py), stress, mTOR
    z = pd.read_csv(a.donor_mag).rename(columns={'proj': 'z'})
    blocks = None
    if a.panel_dir:
        pr = lambda gs: [(g, f'panel:{g}') for g in gs]
        mr = lambda gs: [(g, f'mitocarta:{g}') for g in gs]
        blocks = [('OXPHOS', mr(['CI', 'CII', 'CIII', 'CIV', 'CV']) + pr(['CoQ'])),
                  ('mtDNA', pr(['mtDNA maintenance', 'mt translation & RNA'])),
                  ('other mito', pr(['Fe-S & lipoate', 'import & dynamics', 'PDH & TCA', 'carriers'])),
                  ('stress', list(C_ROWS[2])), ('mTOR', list(C_ROWS[3]))]
    panel_pmd_rows(z, out, a.e_clip, cts, 'Δ loss score (z vs reference donors),\nPMD − reference', ('loss', 'gain'),
                   1, 'fig6f_pathways', src_dir, loss_score=True, blocks=blocks, one_sided=True, summary=True,
                   scaled=True, header='PMD vs reference')


def main(a):
    setup(a.vizstyle_root)
    d, out = Path(a.data_dir), Path(a.out_dir)
    want = set(a.panels.split('+'))
    run = lambda p: 'all' in want or p in want
    cts = [c for c in CT if c not in a.c_drop]
    cd = Path(a.cells_dir) if a.cells_dir else d
    rd = Path(a.rna_dir)
    dc = [c for c in cts if c != 'NK cells']
    # ---- main panels (A schematic by hand; E = fig6_pmd_panel_plot.py; G-H bulk, fig6_bulk_plot.py; I flow)
    if run('b'): panel_c_quadrant(d, out, 'Monocytes', a.contrast, a, mode='value', name='fig6b_deg_vs_expert')
    if run('c'): panel_umap_density(cd, out, 'mono', 'fig6c_umap')
    if run('d'): panel_trees(d, rd, out, a.space, name='fig6d_trees')
    if run('e'): panel_f(a, out, dc, d)                        # Fig. 6F (the 'e' key is kept for the CLI)

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--src-dir', required=True,
                    help='source tables behind the panels (paper/figures/<fig>/data)')
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--panels', default='all')
    ap.add_argument('--modules', help='collaborator modules.tsv (e: complex / module tags)')
    ap.add_argument('--panel-dir', help='fig6_pmd_panel.py output (PanelApp gene groups): e')
    ap.add_argument('--rna-dir', required=True, help='fig6_rna.py output (donor-set + PyDESeq2): b, c')
    ap.add_argument('--donor-mag', help='e: fig6_donor_mag.py table (donor_mag_trained.csv.gz)')
    ap.add_argument('--e-clip', type=float, default=5.0, help='e: colour clip (z vs reference donors)')
    ap.add_argument('--n-lab', type=int, default=5, help='c quadrant: genes named per pathway group')
    ap.add_argument('--pmd-genes', help='B: csv of PMD genes to highlight (first column)')
    ap.add_argument('--n-lab-list', type=int, default=15, help='B: listed genes named (most loss-like)')
    ap.add_argument('--n-table', type=int, default=200, help='c: top ExPert genes saved with their DE')
    ap.add_argument('--lfc-cut', type=float, default=0.25, help='c value panel: dashed |log2FC| cutoff')
    ap.add_argument('--cells-dir', help='dir with cells.csv.gz (main score dir) when --data-dir is a tagged rerun')
    ap.add_argument('--space', default='latent', choices=['latent', 'prototype'], help='b: clustering space')
    ap.add_argument('--reactome-gmt', help='c: Reactome gmt (gene membership for collapsing redundant rows)')
    ap.add_argument('--c-drop', nargs='*', default=['CD8+ T cells'])
    ap.add_argument('--contrast', default='pmd_vs_rest_cell', help='d/e group profile: pmd_vs_rest_cell (cells pooled; best in fig6_grid) or pmd_vs_rest (equal-weight donors)')
    return ap


if __name__ == '__main__':
    main(build_parser().parse_args())
