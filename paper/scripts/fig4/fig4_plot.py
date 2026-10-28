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

"""Fig. 4: perturbations of a different kind -- drugs and acute protein depletion, read through the
CRISPR prototypes with NO retraining. One file per panel plus the table behind it (<data>/fig4X_source.csv).

    a  schematic, vertical flow: three query types -> frozen encoder -> Δz vs matched controls ->
       cosine over the prototypes -> rank of the target gene
    b  sci-Plex3: target recovery by method on fingerprinting's own cells (B-E: sciplex_pb_* panels)
    c  sci-Plex3: recovery by dose
    d  sci-Plex3: HDAC machinery
    e  FlexPlex: similarity of each HDAC inhibitor to AR-42 by dose
    f  Tahoe-100M, responsive conditions only: target / partner recovery against the popularity-matched null
    g  Tahoe-100M: abemaciclib's off-target GSK3 vs the annotated CDK4/6 by dose (fig4_tahoe_mech.py)
    h  Tahoe-100M: clobetasol propionate among the glucocorticoids (fig4_tahoe_mech.py)
    i  degron blot (added by hand)
    j  degron pool: own-gene rank per responding clone vs the clone permutation null

No panel titles (identity sits in axis labels and legends). Multi-plot panels stack vertically.
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
# SVG text as real <text> objects (editable), not glyph outlines; PDF text as TrueType
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
from matplotlib.patches import FancyBboxPatch  # noqa: E402

EXPERT = '#E69F00'      # fig2_common.EXPERT_COLOR: Okabe-Ito orange, fixed across the paper
EXPERT_LIGHT = '#f6d08c'  # the same hue, lightened, for a second ExPert series in one panel
MECH_EXAMPLES = {'Abemaciclib': 'g', 'Clobetasol propionate': 'h'}  # drug -> panel that explains it
M = 'v0.1.0'
W = 7.0              # panel width (in); stacked rows add height
DEGRON_VAR = 'mtfree_mt20'
SRC_DIR = None       # where panels write their source tables; None = the data dir (the sbatch behaviour)


def _src(d):
    """Directory for a panel's source table: SRC_DIR when a caller (the notebook) sets it, else d."""
    return Path(SRC_DIR) if SRC_DIR is not None else d


def style(vs):
    """Make `vs` this module's style and apply the paper look (white surfaces)."""
    global VS
    VS = vs
    VS.apply()
    VS.SURFACE = 'white'                              # paper figures: plain white background
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white'})


def strip(s):
    return str(s).strip()


def short(drug, n=22):
    d = strip(drug).split(' (')[0]
    return d if len(d) <= n else d[:n - 1] + '…'


# ---- a ----------------------------------------------------------------------------------------
def panel_a(d, out):
    fig, ax = plt.subplots(figsize=(W, 10.2))
    ax.set_xlim(0, 7.1); ax.set_ylim(0, 10.2); ax.axis('off')

    def box(x, y, w, h, text, ec=VS.INK2, lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc=VS.SURFACE, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', color=VS.INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=VS.INK2, lw=1.8, shrinkA=2, shrinkB=2))
    q = [('sci-Plex3\ndrug → target\n188 compounds', 0.05),
         ('Tahoe-100M\ndrug → target\n222 drugs', 2.4),
         ('AID degron\nprotein → gene\n40 clones', 4.75)]
    for t, x in q:
        box(x, 7.75, 2.2, 1.65, t, ec=VS.MUTED)
        arrow(x + 1.1, 7.7, 3.5, 7.05)
    ax.text(3.5, 9.75, 'new: the query', ha='center', va='center', color=VS.INK2)
    box(1.6, 5.7, 3.8, 1.3, 'ExPert encoder\n(CRISPR-trained, frozen)', ec=EXPERT, lw=3)
    arrow(3.5, 5.65, 3.5, 5.0)
    box(1.6, 3.65, 3.8, 1.3, 'Δz vs matched controls\n(DMSO or vehicle)')
    arrow(3.5, 3.6, 3.5, 2.95)
    box(1.6, 1.6, 3.8, 1.3, 'cosine over 19,895\ngene prototypes', ec=EXPERT, lw=3)
    arrow(3.5, 1.55, 3.5, 0.95)
    ax.text(3.5, 0.75, 'rank of the target gene', ha='center', va='top', color=VS.INK)
    ax.plot([5.75, 5.75], [1.6, 7.0], color=VS.MUTED, lw=1.2)
    ax.text(5.95, 4.3, 'unchanged: labels, prototypes,\nreadout — no retraining', rotation=90,
            ha='left', va='center', color=VS.INK2)
    VS.save(fig, out, 'fig4a_schematic')


# ---- e: Tahoe exact + pathway recovery on responsive conditions -----------------------------------
# MT-masked: justified independently (every mtmask arm beat every unmasked arm on Tahoe, and
# paired per-condition on sci-Plex the full-bank gain is p=6e-8), not chosen on this figure.
SPB = f'{M}/sciplex_pb_mtmask_fpcells_cc200/panels'   # B-E: fingerprinting's exact cells, plates pooled, depth-matched null
def method_style():
    """Colour / marker per method. Built lazily: VS is imported inside main(), so a module-level
    dict referring to it fails at import."""
    return {'ExPert': (EXPERT, 'o', 2.6),
            'RNA fingerprinting': (VS.INK2, 's', 1.8),
            'training centroids': (VS.MUTED, '^', 1.8),
            'random': (VS.EXCLUDED, 'v', 1.6)}


def _end_labels(ax, rows, xpad=1.9, fs=None, gap=0.085):
    """Name each curve at its own right-hand end instead of in a legend box.

    `rows` is [(label, x_last, y_last, colour)]. Stacking is done in AXES-FRACTION space, which is
    the only thing that stays correct when the axis is log (panel b) or inverted (panel g, where
    rank 1 is at the top) -- doing it in data units silently piles every label on one spot.
    Curves that converge would otherwise overprint, so positions get a minimum gap and a leader
    runs back to the real point.
    """
    if not rows:
        return
    x0, x1 = ax.get_xlim()
    log = ax.get_xscale() == 'log'
    ax.set_xlim(x0, x1 * xpad if log else x1 + (x1 - x0) * (xpad - 1.0))
    xa = max(r[1] for r in rows)
    xt = xa * (1.0 + (xpad - 1.0) * 0.10) if log else xa + (x1 - x0) * (xpad - 1.0) * 0.08
    inv_ax, inv_dat = ax.transAxes.inverted(), ax.transData.inverted()
    to_f = lambda y: float(inv_ax.transform(ax.transData.transform((xa, y)))[1])
    to_y = lambda f: float(inv_dat.transform(ax.transAxes.transform((0.5, f)))[1])
    items = sorted(((to_f(y), lab, xp, y, c) for lab, xp, y, c in rows), reverse=True)
    step = [gap * (1 + lab.count('\n')) for _f, lab, _x, _y, _c in items]
    fs_out, prev = [], None
    for (f, *_), st in zip(items, step):
        f = min(f, 0.98)
        if prev is not None and f > prev - st:
            f = prev - st
        fs_out.append(f); prev = f
    # Stacking runs downward with no floor, so curves that all end near the bottom push the last
    # label below the axis onto the x tick labels (4C: random / fingerprinting / centroids at 10 uM).
    # Lift the whole bottom run back above the floor, keeping each label's spacing to the next.
    floor = 0.03
    for i in range(len(fs_out) - 1, -1, -1):
        lo = floor if i == len(fs_out) - 1 else fs_out[i + 1] + step[i]
        if fs_out[i] < lo:
            fs_out[i] = lo
    for (_f, lab, xp, yp, c), f in zip(items, fs_out):
        y = to_y(f)
        ax.plot([xp, xt], [yp, y], color=VS.MUTED, lw=0.6, zorder=4, clip_on=False)
        ax.text(xt + (xt * 0.02 if log else 0.01 * (x1 - x0)), y, lab, color=c,
                fontsize=fs or VS.FONT_MIN - 3, va='center', ha='left', zorder=6)


def panel_b_sciplex(d, out):
    """sci-Plex3, every method ranking the SAME 3,802 candidates (the fingerprinting
    dictionary intersected with our trained prototypes) and scored on the SAME drugs -- a drug
    enters only if all four methods can score it, since an unscorable drug is a hole in the
    annotation, not a miss. A drug counts as recovered at k when any of its doses puts an
    annotated target at or above k. ExPert reads the pseudobulks through the shipped predict()
    path: deflate readout, best of the loss and gain tails, no hub correction."""
    p_ = d / SPB / 'fig4b_sciplex_curve.csv'
    if not p_.exists():
        print('  [b] missing fig4b_sciplex_curve.csv; skipped'); return
    t = pd.read_csv(p_)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ends = []
    for m, g in t.groupby('method'):
        col, mk, lw = method_style().get(m, (VS.MUTED, 'o', 1.6))
        g = g.sort_values('k')
        ax.plot(g.k, 100 * g.recall, marker=mk, ms=6, lw=lw, color=col, mec=VS.SURFACE, mew=0.8)
        ends.append((m, float(g.k.iloc[-1]), 100 * float(g.recall.iloc[-1]), col))
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([1, 3, 10, 30, 100, 200]); ax.set_xticklabels(['1', '3', '10', '30', '100', '200'])
    ax.set_xlabel('rank cutoff')
    ax.set_ylabel(f'drugs with the target recovered (%)\n{int(t.n.iloc[0])} drugs')
    _end_labels(ax, ends)
    ax.grid(axis='x', visible=False)
    VS.save(fig, out, 'fig4b_sciplex_methods')
    t.to_csv(_src(d) / 'fig4b_source.csv', index=False)


def panel_c_sciplex(d, out, k=10):
    """The same four methods at each labelled dose, over the conditions that actually responded.
    Recovery holds above 20% at every dose once dead conditions are excluded; scoring them lets the
    gain tail manufacture hits at the lowest dose, which used to invert the dose-response."""
    p_ = d / SPB / 'fig4c_sciplex_dose.csv'
    if not p_.exists():
        print('  [c] missing fig4c_sciplex_dose.csv; skipped'); return
    t = pd.read_csv(p_)
    t = t[t.k == k]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ends = []
    for m, g in t.groupby('method'):
        col, mk, lw = method_style().get(m, (VS.MUTED, 'o', 1.6))
        g = g.sort_values('dose_uM')
        ax.plot(g.dose_uM, 100 * g.recall, marker=mk, ms=7, lw=lw, color=col, mec=VS.SURFACE,
                mew=0.8)
        ends.append((m, float(g.dose_uM.iloc[-1]), 100 * float(g.recall.iloc[-1]), col))
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(['0.01', '0.1', '1', '10'])
    ax.set_xlabel('sci-Plex3 dose (µM)')
    ax.set_ylabel(f'drugs with the target\nin the top {k} (%)')
    ax.set_ylim(0, 100 * t.recall.max() * 1.18)
    _end_labels(ax, ends)
    ax.grid(axis='x', visible=False)
    VS.save(fig, out, 'fig4c_sciplex_dose')
    t.to_csv(_src(d) / 'fig4c_source.csv', index=False)


def panel_d_sciplex(d, out):
    """HDAC inhibitors only, scored against the whole HDAC machinery (SIN3, CoREST, NCoR, NuRD and
    the HDAC1/2/3 catalytic subunits) rather than one annotated subunit. Recovery rises with dose,
    as a dose-dependent on-target effect should. Every method is scored over the SAME drug x dose
    pairs; ours carry one row per cell line because the cached fingerprinting table has none."""
    p_ = d / SPB / 'fig4d_hdac_by_dose.csv'
    if not p_.exists():
        print('  [d] missing fig4d_hdac_by_dose.csv; skipped'); return
    t = pd.read_csv(p_)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ends = []
    for m, g in t[t.dose_uM.notna()].groupby('method'):
        col, mk, lw = method_style().get(m, (VS.MUTED, 'o', 1.8))
        g = g.sort_values('dose_uM')
        ax.plot(g.dose_uM, 100 * g.frac_top10, marker=mk, ms=7, lw=lw, color=col, mec=VS.SURFACE,
                mew=0.8)
        ends.append((m, float(g.dose_uM.iloc[-1]), 100 * float(g.frac_top10.iloc[-1]), col))
    ax.text(0.03, 0.97, 'HDAC drugs', transform=ax.transAxes, ha='left', va='top', color=VS.INK,
            fontsize=VS.FONT_MIN - 1)
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(['0.01', '0.1', '1', '10'])
    ax.set_xlabel('sci-Plex3 dose (µM)')
    n_dr = int(t[t.dose_uM.notna()].n.iloc[0])
    ax.set_ylabel('conditions with the HDAC complex\nin the top 10 (%)')
    _end_labels(ax, ends)
    ax.grid(axis='x', visible=False)
    VS.save(fig, out, 'fig4d_hdac_machinery')
    t.to_csv(_src(d) / 'fig4d_source.csv', index=False)


def panel_e_ar42(d, out):
    """The fingerprinting paper's own FlexPlex HDAC screen, read through each model without any
    retraining: how much each HDAC inhibitor's readout looks like AR-42's at the same dose. The
    selective HDAC6 inhibitors converge on the pan-HDAC profile as the dose rises, which is the
    behaviour the screen was built to show."""
    p_ = d / SPB / 'fig4e_ar42.csv'
    if not p_.exists():
        print('  [e] missing fig4e_ar42.csv; skipped'); return
    t = pd.read_csv(p_)
    # the K562-free model (EFFGATE3) is a supplementary comparison now; the table keeps it for the SI
    t = t[t.method != 'ExPert, no K562 training']   # main panel; S17 relabels it, so it is kept there
    drugs = list(t.drug.unique())
    fig, axs = plt.subplots(1, len(drugs), figsize=(3.3 * len(drugs), 4.4), sharey=True,
                            gridspec_kw={'wspace': 0.12})
    style = dict(method_style())                 # one colour table for B-E (the SI patches it)
    style.setdefault('ExPert, no K562 training', (EXPERT_LIGHT, 'o', 2.0))
    aa = np.atleast_1d(axs)
    for i, (ax, drug) in enumerate(zip(aa, drugs)):
        ends = []
        for m, g in t[t.drug == drug].groupby('method'):
            col, mk, lw = style.get(m, (VS.MUTED, 'o', 1.6))
            g = g.sort_values('dose_nM')
            ax.plot(g.dose_nM, g.sim_AR42, marker=mk, ms=6, lw=lw, color=col, mec=VS.SURFACE,
                    mew=0.8)
            ends.append((m, float(g.dose_nM.iloc[-1]), float(g.sim_AR42.iloc[-1]), col))
        if i == len(aa) - 1:
            last_ends = ends           # label after the scale is set, not before (see below)
        ax.axhline(0, color=VS.MUTED, ls=':', lw=1.2)
        ax.set_xscale('log'); ax.minorticks_off()
        ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(['0.1', '1', '10'])
        ax.set_xlabel('FlexPlex dose (µM)')
        ax.text(0.5, 1.0, drug, transform=ax.transAxes, ha='center', va='bottom', color=VS.INK,
                fontsize=VS.FONT_MIN - 1)
        ax.grid(axis='x', visible=False)
    aa[0].set_ylabel('similarity to AR-42')
    # AFTER set_xscale('log'): widening a still-linear axis that spans 100-10,000 and only then
    # switching to log collapses the whole panel into the right-hand edge.
    _end_labels(aa[-1], last_ends, xpad=2.2, fs=VS.FONT_MIN - 4)
    VS.save(fig, out, 'fig4e_ar42')
    t.to_csv(_src(d) / 'fig4e_source.csv', index=False)


MECH = f'{M}/tahoe_mech'
# 1-mismatch barcode assignment + difference-in-differences + MT mask. nm2 adds 11 clones
# and 2.6x the cells but recovers FEWER (see the stringency table in the caption), and
# without the DiD the whole panel is a null (DMSO and auxin are separate libraries).
# The red that panel f uses to mark the two drugs g and h explain. g and h must NOT
# reuse the ExPert orange: in this figure that colour means 'the method', not 'this drug'.
MECH_COLOR = '#e8807a'
# clobetasol is told apart from the other worked example; Okabe-Ito sky blue, which is
# distinct from the reserved ExPert orange in both hue and luminance
CLOB_COLOR, CLOB_LIGHT = '#56B4E9', '#a9d6f2'
MECH_EX_COLOR = {'Clobetasol propionate': CLOB_COLOR}   # default: MECH_COLOR
DEGRON_PB_COL = 'clone_nm1'
DEGRON_PB_TAG = '_did_mt'


def panel_g_offtarget(d, out):
    """An explainable miss. Abemaciclib's annotated targets are CDK4/6, which the readout never
    ranks; GSK3 comes back at the top instead, and abemaciclib is a documented GSK3 inhibitor (Mol
    Cancer Res 2018). Palbociclib, a CDK4/6 inhibitor without that off-target, shows no GSK3
    signal at any dose, so this is not a class artefact. GSK3A and GSK3B are averaged, as are
    CDK4 and CDK6."""
    pd_ = d / MECH / 'dose_ranks.csv'
    if not pd_.exists():
        print('  [g] missing dose_ranks.csv; skipped'); return
    dr = pd.read_csv(pd_)
    dr = dr[dr.drug.isin(['Abemaciclib', 'palbociclib'])
            & dr.gene.isin(['GSK3A', 'GSK3B', 'CDK4', 'CDK6'])].copy()
    dr['set'] = np.where(dr.gene.str.startswith('GSK3'), 'GSK3 (off-target)',
                         'CDK4/6 (annotated target)')
    g = dr.groupby(['drug', 'set', 'dose'], as_index=False)['rank'].mean()
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    shape = {'Abemaciclib': 'o', 'palbociclib': 's'}        # shape = drug
    dash = {'GSK3 (off-target)': '-', 'CDK4/6 (annotated target)': '--'}   # line = gene set
    col = {'Abemaciclib': MECH_COLOR, 'palbociclib': VS.INK2}
    ends = []
    for (drug, gset), q in g.groupby(['drug', 'set']):
        q = q.sort_values('dose')
        ax.plot(q.dose, q['rank'], ls=dash[gset], marker=shape[drug], ms=8, lw=2.2,
                color=col[drug], mec=VS.SURFACE, mew=0.8)
        ends.append((f"{drug} · {gset.split(' (')[0]}", float(q.dose.iloc[-1]),
                     float(q['rank'].iloc[-1]), col[drug]))
    ax.set_yscale('log'); ax.set_xscale('log'); ax.minorticks_off(); ax.invert_yaxis()
    ax.set_yticks([1, 10, 100, 1000, 8964]); ax.set_yticklabels(['1', '10', '100', '1k', '8,964'])
    ax.set_xticks([0.05, 0.5, 5]); ax.set_xticklabels(['0.05', '0.5', '5'])
    ax.set_xlabel('Tahoe dose (µM)')
    ax.set_ylabel('rank of the gene pair\n(of 8,964)')
    _end_labels(ax, ends, xpad=3.0, fs=VS.FONT_MIN - 4, gap=0.075)
    ax.grid(axis='x', visible=False)
    VS.save(fig, out, 'fig4g_offtarget')
    g.to_csv(_src(d) / 'fig4g_source.csv', index=False)


def panel_h_clobetasol(d, out, n_label=5):
    """A miss that is an annotation error. Tahoe labels clobetasol propionate by its secondary
    activity (a selective CYP3A5 inhibitor, mechanism 'unclear'); it is a potent glucocorticoid.
    The readout puts it with dexamethasone and triamcinolone, and a model-free check agrees: the
    glucocorticoid target genes are induced. Two independent axes, so neither is circular."""
    p_ = d / MECH / 'clobetasol_scatter.csv'
    if not p_.exists():
        print('  [h] missing clobetasol_scatter.csv; skipped'); return
    t = pd.read_csv(p_)
    gc = t.moa.astype(str).str.contains('glucocorticoid', case=False, na=False)
    star = t.drug.astype(str).str.contains('Clobetasol', case=False, na=False)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.axhline(0, color=VS.MUTED, ls=':', lw=1.0); ax.axvline(0, color=VS.MUTED, ls=':', lw=1.0)
    ax.scatter(t.GR_expression_score[~gc & ~star], t.expert_sim_to_GC[~gc & ~star], s=14,
               color=VS.EXCLUDED, lw=0, alpha=0.7, zorder=2)
    ax.scatter(t.GR_expression_score[gc], t.expert_sim_to_GC[gc], s=42, color=CLOB_LIGHT,
               edgecolors=VS.SURFACE, lw=0.8, zorder=3, label='annotated glucocorticoids')
    ax.scatter(t.GR_expression_score[star], t.expert_sim_to_GC[star], s=110, color=CLOB_COLOR,
               edgecolors=VS.INK, lw=1.6, zorder=5)
    # Room on the right FIRST: the glucocorticoids sit in the top-right corner, so labels placed
    # before the axis is widened have nowhere to go and land on each other.
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.42 * (x1 - x0))
    from adjustText import adjust_text
    lab = t[star | (gc & (t.expert_sim_to_GC > t.expert_sim_to_GC.quantile(0.9)))]
    shown = lab.head(n_label)
    txt = [ax.text(r.GR_expression_score, r.expert_sim_to_GC, short(r.drug, 26),
                   fontsize=VS.FONT_MIN - 4, color=VS.INK, zorder=6)
           for r in shown.itertuples()]
    adjust_text(txt, ax=ax, expand=(1.5, 2.0), force_text=(0.6, 1.1), max_move=55, iter_lim=500,
                arrowprops=dict(arrowstyle='-', color=VS.MUTED, lw=0.6))
    ax.set_xlabel('glucocorticoid target genes induced\n(model-free, from expression)')
    ax.set_ylabel('readout similarity to the\nannotated glucocorticoids')
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.22 * (y1 - y0))      # the glucocorticoid cluster needs label room
    # The grey cloud is one dense blob, so a leader into it names it without ambiguity. The
    # glucocorticoids are five scattered points, four of which already carry their own name, so a
    # single leader there is always either ambiguous or a duplicate of one -- it kept landing on
    # Budesonide. A class spread over several marks is what a legend entry is for; that one gets it.
    q_other = t[~gc & ~star]
    yr = float(t.expert_sim_to_GC.max() - t.expert_sim_to_GC.min())
    xr = float(t.GR_expression_score.max() - t.GR_expression_score.min())
    ax.annotate('other drugs',
                xy=(float(q_other.GR_expression_score.quantile(0.10)),
                    float(q_other.expert_sim_to_GC.quantile(0.06))),
                xytext=(float(q_other.GR_expression_score.quantile(0.10)) - 0.04 * xr,
                        float(q_other.expert_sim_to_GC.quantile(0.06)) - 0.14 * yr),
                fontsize=VS.FONT_MIN - 4, color=VS.INK2, ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color=VS.MUTED, lw=0.6, shrinkA=2, shrinkB=4))
    ax.legend(loc='upper left', frameon=False, fontsize=VS.FONT_MIN - 4, handletextpad=0.4,
              borderaxespad=0.2)
    VS.save(fig, out, 'fig4h_clobetasol')
    t.to_csv(_src(d) / 'fig4h_source.csv', index=False)


# MAIN panel f: drugs whose MECHANISM IS KNOWN -- inhibitor/antagonist AND activator/agonist. The
# old inhibitor-only restriction was a loss-only leftover (an agonist is findable on the gain tail),
# but the 76 'unclear' drugs say nothing about mechanism.
TAH_PB = f'{M}/tahoe_pb/panels_knownmoa'


def panel_e_tahoe(d, out, k=10, n_label=5, src=None, name='fig4f_tahoe_recovery'):
    """Tahoe-100M, RESPONSIVE conditions only (|Δz| above the 95th percentile of DMSO-vs-DMSO noise).
    Per drug: the fraction of its own active (cell line x dose) conditions that put the annotated
    target -- or a partner of a target in a physical complex / tight pathway -- in the top k, beside
    the SAME statistic for size- and popularity-matched random gene sets drawn for that drug. Every
    drug is a point in its own distribution; the five highest are named with a leader line. Stars
    come from a paired sign-flip permutation test on the mean difference: a signed-rank test is
    dominated here by the SIGN of negligible negative differences on the 75% of drugs that sit at
    zero, while the claim is about the mean."""
    import seaborn as sns
    from matplotlib.ticker import PercentFormatter
    p_ = d / (src or TAH_PB) / 'fig4e_source.csv'
    if not p_.exists():
        print('  [e] missing fig4e_source.csv; skipped'); return
    t = pd.read_csv(p_)
    need = ['frac_exact_topk', 'frac_exact_null_topk', 'frac_complex_topk', 'frac_complex_null_topk']
    if any(c not in t.columns for c in need):
        print('  [e] fig4e_source.csv lacks the paired null (rerun STEP=pb_panels); skipped'); return
    lab = ('annotated\ntarget', 'complex /\npathway partner')
    obs_c, null_c = ('frac_exact_topk', 'frac_complex_topk'), ('frac_exact_null_topk', 'frac_complex_null_topk')
    sub = [t.dropna(subset=[o, n])[['drug', 'moa', 'n_active', o, n]].rename(
        columns={o: 'frac', n: 'null'}) for o, n in zip(obs_c, null_c)]
    long = pd.concat([pd.concat([q.assign(kind=l, source='ExPert'),
                                 q.assign(kind=l, source='matched-random', frac=q.null)])
                      for l, q in zip(lab, sub)])
    fig, ax = plt.subplots(figsize=(4.6, 7.6))
    sns.boxenplot(data=long, x='kind', y='frac', hue='source', order=list(lab),
                  hue_order=['matched-random', 'ExPert'],      # random left, ExPert right
                  palette={'ExPert': EXPERT, 'matched-random': VS.EXCLUDED}, ax=ax,
                  k_depth='trustworthy', showfliers=False, linewidth=0.9, width=0.62, gap=0.06)
    rj = np.random.default_rng(0)
    jxs = {}
    off = 0.155        # seaborn dodges hue boxes to x -/+ width/4
    for xi, (l, q) in enumerate(zip(lab, sub)):
        jx = xi + off + rj.uniform(-0.1, 0.1, len(q))       # one jitter, used for every drug
        jxs[xi] = jx
        ax.scatter(jx, q.frac, s=10, color=VS.INK2, alpha=0.45, lw=0, zorder=4)
        ax.scatter(xi - off + rj.uniform(-0.1, 0.1, len(q)), q.null, s=9, color=VS.INK2,
                   alpha=0.22, lw=0, zorder=4)
        top = q.nlargest(n_label, 'frac')
        ti = [q.index.get_loc(i) for i in top.index]
        ax.scatter(jx[ti], top.frac, s=22, color=VS.INK, lw=0, zorder=6)
        ty = []                                             # de-overlap the names only
        for v in top.frac:
            ty.append(v if not ty else min(v, ty[-1] - 0.055))
        # each group's names go on its OWN outer side, so neither stack crosses the other's box
        lx = xi - 0.40 if xi == 0 else xi + 0.40
        ha = 'right' if xi == 0 else 'left'
        for (r, y, x0) in zip(top.itertuples(), ty, jx[ti]):
            ax.plot([x0, lx], [r.frac, y], color=VS.MUTED, lw=0.7, zorder=5)
            ax.text(lx + (-0.03 if xi == 0 else 0.03), y, short(r.drug, 16),
                    fontsize=VS.FONT_MIN - 3, color=VS.INK, va='center', ha=ha)
        # paired test: each drug against the matched-random sets drawn for that same drug.
        # Sign-flip permutation on the mean difference (paired, and sensitive to magnitude).
        dif = (q.frac - q.null).to_numpy(float)
        flip = (rj.choice([-1.0, 1.0], (20000, len(dif))) * dif).mean(1)
        pv = float((np.sum(flip >= dif.mean()) + 1) / (len(flip) + 1))
        st = '***' if pv < 1e-3 else '**' if pv < 1e-2 else '*' if pv < 0.05 else 'n.s.'
        # ABOVE the label stack, not inside it: the top names sit at ~1.00 and their leaders ran
        # through the stars at 0.95. ty[0] is the highest label, so clear it by a fixed margin.
        yb = 1.10
        ax.plot([xi - off, xi - off, xi + off, xi + off], [yb - 0.018, yb, yb, yb - 0.018],
                color=VS.INK2, lw=1)
        ax.text(xi, yb + 0.008, st, ha='center', va='bottom', color=VS.INK2,
                fontsize=VS.FONT_MIN - 1)
    # the two drugs g and h explain sit at zero on the annotated-target axis: that IS the point,
    # so they are marked here in their own colour rather than left looking like plain failures
    # These two are the drugs (g) and (h) explain. Mark them in BOTH groups and run one label on
    # the right out to both points, so the reader sees the same drug failing on the annotated
    # target and on the complex axis.
    ex_col = MECH_COLOR          # per-drug below; abemaciclib keeps this red
    subs = [q.reset_index(drop=True) for q in sub]
    ey = np.linspace(0.30, 0.16, max(len(MECH_EXAMPLES), 1))
    for (drug, panel), y in zip(MECH_EXAMPLES.items(), ey):
        ex_col = MECH_EX_COLOR.get(drug, MECH_COLOR)
        pts = []
        for gi, q in enumerate(subs):
            m = np.flatnonzero((q.drug == drug).to_numpy())
            if len(m):
                i = int(m[0])
                pts.append((jxs[gi][i], float(q.frac.iloc[i])))
        if not pts:
            continue
        for (x0, y0) in pts:
            ax.scatter([x0], [y0], s=40, color=ex_col, edgecolors=VS.SURFACE, lw=0.8, zorder=7)
            ax.plot([x0, 1.24], [y0, y], color=ex_col, lw=0.8, ls=(0, (3, 2)), zorder=6)
        ax.text(1.28, y, f'{short(drug, 21)}\n→ ({panel})', color=ex_col,
                fontsize=VS.FONT_MIN - 3, va='center', ha='left', linespacing=1.15)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(['annotated\ntarget', 'complex /\npartner'])
    ax.tick_params(axis='x', labelsize=VS.FONT_MIN - 1)
    ax.set_xlabel(''); ax.set_xlim(-1.35, 1.80); ax.set_ylim(-0.03, 1.19)
    ax.set_ylabel(f"a drug's responsive conditions\nwith a hit in the top {k}")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=VS.FONT_MIN - 2, frameon=False, handlelength=1.2, borderaxespad=0.2,
              columnspacing=1.4)
    VS.save(fig, out, name)
    long.to_csv(_src(d) / 'fig4f_source.csv', index=False)


def _bridge_graph(ax, nodes, edges, gene, meta=None, max_drugs=3, max_ends=2, compact=False,
                  sub=True, centre=None, header='top'):
    """The neighbourhood around one bridge prototype: the drug targets whose shortest path to each
    other runs through it (the bridge is their only intermediate), the prototypes nearest to those
    nodes in the bank's own graph, and every kNN edge among them. The bridge is pinned at the centre
    and one target from each side is pinned left and right, so the layout keeps the reading
    'targets of mechanism A -- bridge -- targets of mechanism B'."""
    import networkx as nx
    from adjustText import adjust_text
    # Mask the edges a label crosses with an opaque bbox, NOT a stroke halo: a stroke on text makes
    # matplotlib emit the glyphs as outline paths, so the label stops being selectable text and
    # VS.assert_vector_svg rejects the figure (paper rule: real text in every SVG).
    halo_bb = dict(boxstyle='round,pad=0.12', fc=VS.SURFACE, ec='none')
    fs = VS.FONT_MIN - (3 if compact else 0)      # small multiples need everything a size down
    n, e = nodes[nodes.bridge == gene], edges[edges.bridge == gene]
    ends = n[n.role == 'endpoint'].copy()
    if not len(ends):
        ax.set_axis_off(); return None
    if len(ends) > max_ends:
        # a bridge that is one of several steps can carry many endpoints; keep the busiest few and
        # the neighbourhood around them, or the panel is unreadable. An endpoint whose drugs carry
        # a named mechanism is worth more here than a slightly busier unannotated one.
        key = ends.get('side', pd.Series(-1, index=ends.index)).ge(0).astype(int)
        ends = (ends.assign(_k=key).sort_values(['_k', 'n_pairs'], ascending=False)
                .head(max_ends).drop(columns='_k'))
        core = set(ends.gene) | {gene}
        adj = set(e[e.a.isin(core)].b) | set(e[e.b.isin(core)].a)
        n = n[n.gene.isin(core | adj)]
        e = e[e.a.isin(n.gene) & e.b.isin(n.gene)]
    tops = [x for x in ends.moa.fillna('').value_counts().index if x not in ('', 'unclear')][:2]
    tops = (tops + ['unannotated mechanism', 'unannotated mechanism'])[:2]
    if 'side' in ends and ends.side.ge(0).all():
        ends = ends.copy()                      # the export already decided which side is which
        tops = [ends[ends.side == sd].moa.iloc[0] if (ends.side == sd).any()
                else 'unannotated mechanism' for sd in (0, 1)]
    else:
        ends = ends.assign(side=[0 if m == tops[0] else 1 for m in ends.moa.fillna('')])
    side = dict(zip(ends.gene, ends.side))
    g = nx.Graph()
    g.add_nodes_from(n.gene)
    for r in e.itertuples():
        g.add_edge(r.a, r.b, weight=float(max(r.cos, 0.01)))
    rng = np.random.default_rng(0)
    init = {x: rng.uniform(-1, 1, 2) for x in g}
    init[gene] = np.zeros(2)
    fixed = [gene]
    for sd in (0, 1):
        q = ends[ends.side == sd].sort_values('n_pairs', ascending=False)
        if len(q):
            init[q.gene.iloc[0]] = np.array([-1.35 if sd == 0 else 1.35, 0.0])
            fixed.append(q.gene.iloc[0])
    pos = nx.spring_layout(g, pos=init, fixed=fixed, weight='weight', iterations=900,
                           k=2.1 / np.sqrt(max(len(g), 2)), seed=0)
    for r in e.itertuples():
        (x0, y0), (x1, y1) = pos[r.a], pos[r.b]
        ax.plot([x0, x1], [y0, y1], color=VS.GRID, lw=0.4 + 2.0 * max(r.cos, 0), alpha=0.75,
                zorder=1, solid_capstyle='round')
    tint = {0: EXPERT, 1: EXPERT_LIGHT}
    nb = n[n.role == 'neighbour']
    txt = []
    if len(nb):
        ax.scatter([pos[x][0] for x in nb.gene], [pos[x][1] for x in nb.gene], s=28,
                   color=VS.EXCLUDED, edgecolors=VS.SURFACE, linewidths=0.6, zorder=2)
        named = set(nb.gene)
        if compact:                       # a small cell cannot carry every name: keep 1-hop only
            core = set(ends.gene) | {gene}
            named = {x for x in nb.gene if set(g.neighbors(x)) & core}
        # start each label just below its node, then let adjustText push them apart
        txt += [ax.text(pos[x][0], pos[x][1] - 0.055, x, ha='center', va='top', color=VS.MUTED,
                        fontsize=fs - 6, fontstyle='italic', zorder=3,
                        bbox=halo_bb) for x in nb.gene if x in named]
    for r in ends.itertuples():
        x_, y_ = pos[r.gene]
        ax.scatter([x_], [y_], s=150, color=tint[side[r.gene]], edgecolors=VS.SURFACE,
                   linewidths=0.9, zorder=4)
        # label on the side AWAY from the bridge, so it cannot land on the centre box
        out = -1.0 if x_ < pos[gene][0] else 1.0
        txt.append(ax.text(x_ + 0.1 * out, y_, r.gene, ha='right' if out < 0 else 'left',
                           va='center', color=VS.INK, fontsize=fs - 3,
                           fontstyle='italic', zorder=6, bbox=halo_bb))
    if centre is None:
        ax.text(pos[gene][0], pos[gene][1], gene, ha='center', va='center', color=VS.INK,
                fontsize=fs - 2, fontweight='bold', fontstyle='italic', zorder=7,
                bbox=dict(boxstyle='round,pad=0.35', fc=VS.SURFACE, ec=VS.INK, lw=2))
    else:
        # Two half-boxes say where the evidence comes from: left = the learned prototypes, right =
        # the best prior. A bridge only the prototypes find is one solid colour; one a prior also
        # finds is split, the same two colours the strip between the graphs uses.
        # sized in POINTS, not data units: a patch radius would render as an ellipse because the
        # graph axes have no equal aspect
        from matplotlib.markers import MarkerStyle
        cx, cy = pos[gene]
        for fill, c in (('left', centre[0]), ('right', centre[1])):
            ax.scatter([cx], [cy], s=300, marker=MarkerStyle('o', fillstyle=fill), color=c,
                       edgecolors='none', zorder=6)
        ax.scatter([cx], [cy], s=300, marker='o', facecolor='none', edgecolors=VS.INK, lw=1.6,
                   zorder=7)
        ax.text(cx, cy - 0.10, gene, ha='center', va='top', color=VS.INK, fontsize=fs - 2,
                fontweight='bold', fontstyle='italic', zorder=8, bbox=halo_bb)
    ax.set_axis_off(); ax.margins(0.26)
    y0, y1 = ax.get_ylim()
    if header == 'top':
        ax.set_ylim(y0, y1 + 0.46 * (y1 - y0))      # room for the mechanism header above the graph
    else:
        ax.set_ylim(y0 - 0.46 * (y1 - y0), y1)      # ... or below it
    # the de-overlap has to know about the nodes AND the bridge box, or labels slide under them
    th = np.linspace(0, 2 * np.pi, 20)
    px = [pos[x][0] for x in g] + list(pos[gene][0] + 0.38 * np.cos(th))
    py = [pos[x][1] for x in g] + list(pos[gene][1] + 0.20 * np.sin(th))
    adjust_text(txt, x=np.array(px), y=np.array(py), ax=ax, expand=(1.6, 2.0),
                force_text=(0.9, 1.6), force_static=(0.6, 1.0), max_move=60, iter_lim=600,
                arrowprops=dict(arrowstyle='-', color=VS.MUTED, lw=0.5, shrinkA=1, shrinkB=2))
    for sd in (0, 1):
        q = ends[ends.side == sd]
        if not len(q):
            continue
        dr, extra = [], 0
        for r in q.itertuples():
            for x0 in str(r.drugs).split(';'):
                nm = short(x0, 14)
                if x0 and nm not in dr:
                    dr.append(nm)
            extra += max(int(getattr(r, 'n_drugs', 0)) - len(str(r.drugs).split(';')), 0)
        names = ', '.join(dr[:max_drugs]) + (f' +{extra + max(len(dr) - max_drugs, 0)}'
                                             if extra or len(dr) > max_drugs else '')
        head = {'DNA synthesis/repair inhibitor': 'DNA repair inh.',
                'Glucocorticoid receptor agonist': 'glucocorticoid agon.',
                'Androgen receptor antagonist': 'androgen rec. antag.',
                'unannotated mechanism': 'unannotated'}.get(tops[sd], tops[sd])
        head = (head.replace(' inhibitor', ' inh.').replace(' antagonist', ' antag.')
                .replace(' receptor', ' rec.').replace(' agonist', ' agon.'))
        hy, va_ = (1.0, 'top') if header == 'top' else (0.0, 'bottom')
        ax.text(0.0 if sd == 0 else 1.0, hy, short(head, 22 if compact else 40),
                transform=ax.transAxes, ha='left' if sd == 0 else 'right', va=va_,
                color=EXPERT if sd == 0 else '#b07a10', fontsize=fs - 3)
        if not compact:
            ax.text(0.01 if sd == 0 else 0.99, hy - 0.075 if header == 'top' else hy + 0.075,
                    names, transform=ax.transAxes, ha='left' if sd == 0 else 'right',
                    va=va_, color=VS.INK2, fontsize=fs - 5)
    m = next((x for x in (meta or {}).get('top', []) if x['gene'] == gene), None)
    if m and sub:
        sole = bool(n.get('sole', pd.Series([1])).iloc[0])
        best = max(((nm, x['fold']) for nm, v in (meta or {}).get('priors', {}).items()
                    for x in v['top'] if x['gene'] == gene), key=lambda z: z[1], default=None)
        if compact:
            sub = f"{gene}: {m['n_pairs']} pairs, {m['fold']:.0f}x expected"
            if not sole:
                sub += ', one of several steps'
            if best:
                sub += f"\nbest prior: {best[0]} {best[1]:.1f}x"
        else:
            sub = (f"{gene}: {m['n_pairs']} drug pairs routed through, {m['fold']:.0f}x expected"
                   + ('' if sole else ' (one of several steps on each path)')
                   + (f"  |  best prior ({best[0]}) {best[1]:.1f}x" if best else ''))
        ax.text(0.5, -0.015, sub, transform=ax.transAxes, ha='center', va='top', color=VS.INK2,
                fontsize=fs - 3)
    return pos[gene]


def _two_tone_tick(fig, ax, pos, name, c_first, c_second, fs, axis='x'):
    """One tick label whose first half is `c_first` and second half `c_second`.

    The second half is drawn at the tick and the first offset by that half's MEASURED extent in
    POINTS, which survives the tight-bbox redraw savefig does at its own dpi (a display-coordinate
    clip would not). PDPK1 is found by the prototypes and by a prior, so its name carries both.
    """
    from matplotlib.transforms import ScaledTranslation
    k = len(name) // 2 + len(name) % 2
    p1, p2 = name[:k], name[k:]
    kw = dict(fontsize=fs, fontweight='bold', fontstyle='italic')
    if axis == 'x':
        tr = ax.get_xaxis_transform()
        t2 = ax.text(pos, -0.02, p2, transform=tr, rotation=90, ha='center', va='top',
                     color=c_second, **kw)
        fig.canvas.draw()
        off = t2.get_window_extent().height * 72.0 / fig.dpi
        ax.text(pos, -0.02, p1, rotation=90, ha='center', va='top', color=c_first,
                transform=tr + ScaledTranslation(0, -off / 72.0, fig.dpi_scale_trans), **kw)
    else:
        tr = ax.get_yaxis_transform()
        t2 = ax.text(-0.01, pos, p2, transform=tr, ha='right', va='center', color=c_second, **kw)
        fig.canvas.draw()
        off = t2.get_window_extent().width * 72.0 / fig.dpi
        ax.text(-0.01, pos, p1, ha='right', va='center', color=c_first,
                transform=tr + ScaledTranslation(-off / 72.0, 0, fig.dpi_scale_trans), **kw)


def _bridge_strip(fig, ax, t, main, prior_backed, flip=False, floor_frac=0.45):
    """Every scored bridge: the learned enrichment against the best prior's, over a matched null.

    `flip` puts the bridges down the y axis and the enrichment along x. The two worked examples are
    named in the colours of their evidence -- one colour means the prototypes found it alone -- so
    the strip needs no connectors to the graphs beside it.
    """
    n = len(t)
    idx = np.arange(n)
    lf = t.fold.to_numpy(float)
    pv = t.best_prior.to_numpy(float)
    real = pv[pv > 0]
    fl = min(float(real.min()) if len(real) else 1.0, 1.0) * floor_frac
    pp = np.where(pv > 0, pv, fl)                 # a prior that routes nothing sits on its own line
    face = np.where(pv > 0, VS.INK2, 'white')
    lo, hi = np.minimum(pp, lf), np.maximum(pp, lf)
    val_lim = (fl * 0.6, max(lf.max(), pp.max()) * 3.2)
    ticks, tlabs = [fl, 1, 10, 100], ['none', '1', '10', '100']
    fsz = VS.FONT_MIN - 4

    if flip:
        pos = idx[::-1]                            # best bridge at the top
        ax.axvline(1.0, color=VS.MUTED, ls=':', lw=1.0, zorder=1)
        ax.hlines(pos, lo, hi, color=VS.GRID, lw=1.4, zorder=2)
        ax.scatter(pp, pos, s=30, color=face, edgecolors=VS.INK2, linewidths=1.2, zorder=4)
        ax.scatter(lf, pos, s=42, color=EXPERT, edgecolors=VS.SURFACE, linewidths=0.8, zorder=5)
        ax.set_xscale('log'); ax.minorticks_off()
        ax.set_xticks(ticks); ax.set_xticklabels(tlabs)
        ax.set_xlim(*val_lim); ax.set_ylim(-0.8, n + 0.9)   # room for the series names
        ax.set_yticks(pos); ax.set_yticklabels(t.gene, fontsize=fsz, fontstyle='italic')
        ax.set_xlabel('enrichment over random')
        ax.grid(axis='y', visible=False)
    else:
        pos = idx
        ax.axhline(1.0, color=VS.MUTED, ls=':', lw=1.0, zorder=1)
        ax.vlines(pos, lo, hi, color=VS.GRID, lw=1.4, zorder=2)
        ax.scatter(pos, pp, s=30, color=face, edgecolors=VS.INK2, linewidths=1.2, zorder=4)
        ax.scatter(pos, lf, s=42, color=EXPERT, edgecolors=VS.SURFACE, linewidths=0.8, zorder=5)
        ax.set_yscale('log'); ax.minorticks_off()
        ax.set_yticks(ticks); ax.set_yticklabels(tlabs)
        ax.set_ylim(*val_lim); ax.set_xlim(-0.8, n - 0.2)
        ax.set_xticks(pos)
        ax.set_xticklabels(t.gene, rotation=90, fontsize=fsz, fontstyle='italic')
        ax.set_ylabel('enrichment over random')
        ax.grid(axis='x', visible=False)

    # name the two series on a bridge where they are far apart, instead of a legend
    gap = np.where(pv > 0, lf / np.maximum(pv, 1e-9), 0.0)
    i0 = int(np.argmax(gap)) if gap.max() > 0 else 0
    for lab, val, mul, col in (('learned prototypes', lf[i0], 2.6, EXPERT),
                               ('best prior', pp[i0], 0.38, VS.INK2)):
        if flip:
            xy, xyt, ha = (val, pos[i0]), (val * mul, pos[i0] + 0.8), 'center'
        else:
            xy, xyt, ha = (pos[i0], val), (pos[i0] + 0.9, val * mul), 'left'
        ax.annotate(lab, xy=xy, xytext=xyt, fontsize=VS.FONT_MIN - 3, color=col, va='center',
                    ha=ha, arrowprops=dict(arrowstyle='-', color=col, lw=0.7))

    # the worked examples: circled, and named in their evidence colours
    labs = ax.get_yticklabels() if flip else ax.get_xticklabels()
    for lab in labs:
        if lab.get_text() in main:
            lab.set_visible(False)
    for gene in main:
        j = t.index[t.gene == gene]
        if not len(j):
            continue
        j = int(j[0])
        pt = (lf[j], pos[j]) if flip else (pos[j], lf[j])
        ax.scatter([pt[0]], [pt[1]], s=165, facecolor='none', edgecolors=VS.INK, lw=1.3, zorder=6)
        if prior_backed.get(gene, False):
            _two_tone_tick(fig, ax, pos[j], gene, EXPERT, VS.INK2, fsz, 'y' if flip else 'x')
        elif flip:
            ax.text(-0.01, pos[j], gene, transform=ax.get_yaxis_transform(), ha='right',
                    va='center', color=EXPERT, fontsize=fsz, fontweight='bold', fontstyle='italic')
        else:
            ax.text(pos[j], -0.02, gene, transform=ax.get_xaxis_transform(), rotation=90,
                    ha='center', va='top', color=EXPERT, fontsize=fsz, fontweight='bold',
                    fontstyle='italic')


def _degron_pb(d, col=None, bank='full'):
    """The pseudobulk degron table (fig4_degron_pb.py). Returns None when the step has not run."""
    col = col or DEGRON_PB_COL
    f = d / M / 'degron_pb' / f'degron_pb_ranks_{col}{DEGRON_PB_TAG}.csv'
    if not f.exists():
        return None
    t = pd.read_csv(f)
    return t[t.bank == bank].copy()


def panel_j_degron(d, out, col=None, bank='full', n_top=10):
    """Acute protein depletion, read through the same path as the drugs. One row per clone whose
    degron actually fired (IAA-vs-DMSO shift above the split-half noise of that clone's own DMSO
    cells, judged BEFORE the model). Rank of the degraded gene itself among all prototypes, against
    the clone permutation null. Filled = the gene has a trained prototype, open = zero-shot."""
    t = _degron_pb(d, col, bank)
    if t is None:
        print('  [j] missing degron_pb ranks (STEP=degron_pb); skipped'); return
    c = (t[t.active.fillna(False)].nsmallest(n_top, 'rank')
         .sort_values('rank', ascending=False).reset_index(drop=True))   # best at the top
    if not len(c):
        print('  [j] no active clones; skipped'); return
    nmax = int(c.n_candidates.iloc[0])
    y = np.arange(len(c))
    nm = float(c.rank_null.median())
    fig, ax = plt.subplots(figsize=(5.8, 0.22 * len(c) + 1.9))
    ax.axvline(nm, color=VS.MUTED, ls=':', lw=1.5)
    # NOT chance (that is ~nmax/2): this is the permutation null, the median rank the OTHER
    # clones' genes reach in this same query -- a strict control, since every clone gene is a
    # chromatin or nuclear regulator and the class is well placed as a class.
    ax.text(nm * 0.88, len(c) - 0.4, 'an unrelated\nclone\u2019s gene', color=VS.INK2, ha='right',
            va='center', fontsize=VS.FONT_MIN - 3, linespacing=1.15)
    # two encodings, no legend (every dot is the same size): COLOUR = which direction found it
    # (a degron removes the protein,
    # not the transcript, so a depleted gene need not look like its own knockout), FILL = whether
    # the gene had any CRISPR training data at all.
    c['by_loss'] = c.rank_loss <= c.rank_gain
    base = [EXPERT if b else VS.INK2 for b in c.by_loss]
    ax.hlines(y, c['rank'], nmax, color=VS.GRID, lw=1.8)
    ax.scatter(c['rank'], y, s=70,
               color=[VS.SURFACE if z else b for z, b in zip(c.zero_shot, base)],
               edgecolor=base, lw=1.8, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(c.clone, fontstyle='italic',
                                         fontsize=VS.FONT_MIN - 2)

    def note(sub, text, col, dx=6.0, dy=3.5):
        if not len(sub):
            return
        r = sub.loc[sub['rank'].idxmin()]
        yi = int(c.index[c.clone == r.clone][0])
        ax.annotate(text, xy=(r['rank'], yi), xytext=(r['rank'] * dx, yi - min(dy, yi)),
                    fontsize=VS.FONT_MIN - 3, color=col, va='center', linespacing=1.15,
                    arrowprops=dict(arrowstyle='-', color=VS.MUTED, lw=0.7))

    note(c[c.zero_shot], 'no CRISPR training\ndata for this gene', VS.INK2)
    note(c[~c.by_loss], 'found as the opposite\nof a knockout', VS.INK2, dx=0.12, dy=-2.5)
    ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.7, len(c) - 0.3)
    ax.set_xscale('log'); ax.set_xlim(0.7, nmax * 1.3); ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(['1', '10', '100', '1k', '10k'])
    ax.set_xlabel(f'rank of the depleted gene (of {nmax:,})')
    VS.save(fig, out, 'fig4j_degron')
    c.to_csv(_src(d) / 'fig4j_source.csv', index=False)


def main(a):
    global VS, DEGRON_VAR
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as vs
    style(vs)
    DEGRON_VAR = a.degron_variant
    d, out = Path(a.data_dir), Path(a.out_dir)
    # outputs of the earlier lettering (B-G became C-H when the summary became B)
    for stale in ('fig4b_sciplex_recall', 'fig4c_dose', 'fig4d_examples', 'fig4e_tahoe_recall',
                  'fig4e_named_hits', 'fig4f_attractor', 'fig4f_tahoe_hub', 'fig4h_moa_knn',
                  'fig4b_summary', 'fig4c_sciplex_recall', 'fig4d_dose', 'fig4e_tahoe_recovery'):
        for f in (out / f'{stale}.svg', out / f'{stale}.png'):
            f.unlink(missing_ok=True)
    for stale in ('fig4b_sciplex_recall_source.csv', 'fig4c_source.csv', 'fig4c_tests.csv', 'fig4c_per_drug_dose.csv',
                  'fig4e_tahoe_recall_source.csv', 'fig4e_hits_source.csv', 'fig4f_hits_source.csv',
                  'fig4f_tahoe_recall_source.csv', 'fig4g_scatter_source.csv', 'fig4h_source.csv',
                  'fig4h_covariate_source.csv'):
        (d / stale).unlink(missing_ok=True)
    want = set(a.panels.split('+'))
    run = lambda p: 'all' in want or p in want
    if run('a'): panel_a(d, out)
    if run('b'): panel_b_sciplex(d, out)
    if run('c'): panel_c_sciplex(d, out)
    if run('d'): panel_d_sciplex(d, out)
    if run('e'): panel_e_ar42(d, out)
    if run('f'): panel_e_tahoe(d, out)       # Tahoe recovery, random left / ExPert right
    if run('g'): panel_g_offtarget(d, out)
    if run('h'): panel_h_clobetasol(d, out)
    # 4i is a hand-made sketch + western; the prototype bridges are Figure S9 (figS9_plot.py)
    if run('i'):
        print("  [i] 4i is added by hand (sketch + western); bridges are Figure S9 (figS9_plot.py)")
    if run('j'): panel_j_degron(d, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--vizstyle-root', required=True, help='ExPert-data repo (src/vizstyle.py)')
    ap.add_argument('--panels', default='all', help="'+'-list of a..j or 'all' (no commas: sbatch --export)")
    ap.add_argument('--degron-variant', default='mtfree_mt20', help='fig4_degron.py output dir under degron/')
    main(ap.parse_args())
