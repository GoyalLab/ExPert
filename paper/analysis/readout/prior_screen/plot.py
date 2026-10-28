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

"""Collect bio / zs / combo JSONs -> TSV tables + figures (svg + png 300 dpi).

  fig_zs_singles   density of ZS x chance over all leak-free single priors, <=5 marked priors,
                   shuffled-twin null density + chance line
  fig_zs_combos    per combo size: every subset of the top-single pool, best per size, shuffled-twin
                   combo null, best-single reference
  fig_bio_vs_zs    bio lift@10 (KEGG, size-matched) vs ZS x chance, per prior
  fig_bio          bio lift@10 per prior x library (size-matched)
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

LIBS = ('KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023')
LIBLAB = {'KEGG_2021_Human': 'KEGG', 'Reactome_2022': 'Reactome', 'GO_Biological_Process_2023': 'GO BP'}
# reference palette (dataviz skill, light mode)
C = dict(s1='#2a78d6', s2='#eb6834', s3='#1baf7a', s4='#eda100', s5='#e87ba4', s6='#008300', s7='#4a3aa7', s8='#d03b3b',
         ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7', surf='#fcfcfb')
MARK_COLORS = [C['s1'], C['s2'], C['s3'], C['s7'], C['s5']]
FAM_COLOR = {'text': C['s1'], 'PPI (STRING)': C['s2'], 'DepMap': C['s3']}   # rest folds to "other"

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': C['axis'], 'axes.labelcolor': C['ink2'],
    'xtick.color': C['muted'], 'ytick.color': C['muted'], 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': C['surf'], 'axes.facecolor': C['surf'], 'savefig.facecolor': C['surf'],
    'legend.frameon': False, 'svg.fonttype': 'none'})


def save(fig, name, run_dir, sub=''):
    d = f'{run_dir}/figs' + (f'/{sub}' if sub else ''); os.makedirs(d, exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(f'{d}/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {d}/{name}.svg|png')


def pretty(name):
    return {'string': 'STRING PPI', 'string_exp': 'STRING experimental', 'string_coexp': 'STRING coexpression',
            'depmap': 'DepMap (raw)', 'depmap_dstd': 'DepMap (double-std)', 'depmap_drop2': 'DepMap (drop2)',
            'l1000': 'L1000', 'morph_crispr': 'Morphology CRISPR', 'morph_orf': 'Morphology ORF',
            'proteintalks_global': 'Proteome (global)', 'proteintalks_within': 'Proteome (within tissue)',
            'gff_pos': 'Genomic position', 'coexpr_pbulk38k': 'Screen coexpr v1', 'coexpr_pbulk38k_v2': 'Screen coexpr v2',
            'resp_consensus': 'Response consensus', 'random_gaussian': 'Random Gaussian',
            'sonnet-4.5/openai-3-large-1024/openai-3-large': 'sonnet-4.5/openai-3-large-1024'}.get(name, name)




GROUPS = ('all', 'seen_dataset', 'unseen_dataset')
PGROUPS = ('pooled_all', 'pooled_seen_dataset', 'pooled_unseen_dataset')
GLAB = {'all': 'all unseen-perturbation queries (per query)', 'seen_dataset': 'SEEN datasets (per query)',
        'unseen_dataset': 'UNSEEN datasets (per query)', 'pooled_all': 'all unseen perturbations',
        'pooled_seen_dataset': 'unseen perturbation, SEEN dataset', 'pooled_unseen_dataset': 'unseen perturbation, UNSEEN dataset (OOD)'}
MAIN = 'pooled_all'
METRIC = {'w_zs_x': ('efficiency-weighted zero-shot top-10', 'w_'), 'zs_x': ('zero-shot top-10', ''),
          'wraw_zs_x': ('raw-efficiency-weighted zero-shot top-10', 'wraw_')}
MDIR = {'w_zs_x': 'eff', 'zs_x': 'unweighted', 'wraw_zs_x': 'eff_raw'}      # one figure subfolder per readout


def short(name):
    """compact names for dense labels."""
    s = {'string': 'STRING', 'string_exp': 'STRING-exp', 'string_coexp': 'STRING-coexp', 'depmap': 'DepMap-raw',
         'depmap_dstd': 'DepMap-dstd', 'depmap_drop2': 'DepMap-drop2', 'proteintalks_global': 'Proteome-glob',
         'proteintalks_within': 'Proteome-within', 'morph_crispr': 'Morph-CRISPR', 'morph_orf': 'Morph-ORF',
         'l1000': 'L1000', 'gff_pos': 'GFF-pos'}.get(name)
    if s:
        return s
    if name.startswith('GenePT/'):
        return 'GenePT ' + name.split('/', 1)[1]
    if '/' in name:
        d, e = name.split('/', 1)
        return f'{e.split("/")[0]} ({d})'
    return name


PANEL = {'panel': 'cand', 'n_panel': 9874}      # filled from the zs records: what every query was ranked against


def atoms_of(tag):
    """Output tags are <atoms>[_full]; the atoms are the same, only the ranking panel differs."""
    return tag[:-5] if tag.endswith('_full') else tag


def panel_short():
    return (f"{PANEL['n_panel']:,}-candidate panel" if PANEL['panel'] == 'cand'
            else f"full {PANEL['n_panel']:,}-row vocabulary")


def panel_note():
    return (f"one {PANEL['n_panel']:,}-candidate panel for every dataset" if PANEL['panel'] == 'cand'
            else f"the full {PANEL['n_panel']:,}-row vocabulary (trained atoms included) for every dataset")


def collect(out_dir, tag, metric):
    bio = []
    for p in sorted(glob.glob(f'{out_dir}/bio/[0-9][0-9][0-9].json')):
        r = json.load(open(p)); row = dict(prior=r['prior'], family=r['family'], leak=r['leak'],
                                           n_covered=r['n_panel_covered'], dim=r['dim'])
        for lib in LIBS:
            f, m = r[lib]['full'], r[lib]['matched']
            row[f'{LIBLAB[lib]}_n'] = f['n']; row[f'{LIBLAB[lib]}_lift10_full'] = f['lift@10']
            row[f'{LIBLAB[lib]}_p10_full'] = f['p@10']
            row[f'{LIBLAB[lib]}_lift10_matched'] = m['lift@10'] if m else np.nan
            row[f'{LIBLAB[lib]}_lift10_matched_sd'] = r[lib]['matched_sd']['lift@10'] if m else np.nan
        bio.append(row)
    bio = pd.DataFrame(bio)
    zs = []
    for p in sorted(glob.glob(f'{out_dir}/{tag}/zs/[0-9][0-9][0-9].json')):
        r = json.load(open(p))
        if 'real' not in r:
            continue
        if r.get('n_panel'):
            PANEL.update(panel=r.get('panel', 'cand'), n_panel=int(r['n_panel']))
        for g in [k for k in r['real'] if k != 'cov']:
            s = r['real'][g]
            null = [r[k][g][metric] for k in r if k.startswith('shuf')]
            h = max(s['hits10'], 1)
            zs.append(dict(prior=r['prior'], family=r['family'], group=g, **s, x=s[metric],
                           ci95_x=1.96 * s[metric] / np.sqrt(h), null_mean=float(np.mean(null)), null_all=null))
    return bio, pd.DataFrame(zs)


def kde(x, grid, bw=None):
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(x, bw_method=bw)(grid)


def pick_marks(za):
    """<=5 on the main group: best single, best GenePT, EFFGATE3 text bank, STRING PPI, DepMap dstd."""
    real = za[za.family != 'Random']
    gp = real[real.prior.str.startswith('GenePT')]
    marks = [(real.loc[real.x.idxmax(), 'prior'], 'best single')]
    for nm, why in ((gp.loc[gp.x.idxmax(), 'prior'] if len(gp) else None, 'best GenePT'),
                    ('opus-4.8/qwen3-embed-8b', 'EFFGATE3 text bank'), ('string', 'PPI'), ('depmap_dstd', 'DepMap')):
        if nm is None or nm not in set(real.prior):
            continue
        hit = [k for k, (m_, _) in enumerate(marks) if m_ == nm]
        if hit:
            marks[hit[0]] = (nm, f'{marks[hit[0]][1]}; {why}')
        else:
            marks.append((nm, why))
    return marks[:5]


def fig_zs_singles(zs, tag, run_dir, metric, groups=(MAIN,), name='zs_singles_density', sub=None):
    sub = MDIR[metric] if sub is None else f'{MDIR[metric]}/{sub}'
    groups = [g for g in groups if g in set(zs.group)]
    marks = pick_marks(zs[zs.group == MAIN])
    H = 2.0 * len(groups) + 0.9
    fig = plt.figure(figsize=(7.4, H))
    gs = fig.add_gridspec(len(groups), 2, width_ratios=(3.3, 1.0), hspace=0.75, wspace=0.04,
                          top=1 - 0.62 / H, bottom=0.55 / H)
    for gi, g in enumerate(groups):
        ax = fig.add_subplot(gs[gi, 0]); lab = fig.add_subplot(gs[gi, 1]); lab.axis('off')
        z = zs[zs.group == g]; real = z[z.family != 'Random']
        null = np.concatenate([np.asarray(v) for v in real.null_all])
        hi = float(np.ceil(max(real.x.max(), np.percentile(null, 97.5)) * 1.08))
        grid = np.linspace(0, hi, 800)
        d = kde(real.x, grid, 0.3); d = d / max(d.max(), 1e-12)
        lo_n, hi_n = np.percentile(null, [2.5, 97.5])
        ax.axvspan(lo_n, hi_n, color=C['muted'], alpha=0.16, lw=0)
        ax.axvline(1.0, color=C['ink2'], lw=1, ls=':')
        ax.text(hi_n, 1.12, ' shuffled null (95%) / chance', fontsize=6.5, color=C['muted'], va='top', ha='left')
        ax.fill_between(grid, 0, d, color=C['s1'], alpha=0.18, lw=0)
        ax.plot(grid, d, color=C['s1'], lw=1.8)
        rows = []
        for j, (nm, why) in enumerate(marks):
            r = real[real.prior == nm]
            if len(r):
                x = float(r.x.iloc[0]); rows.append((x, j, nm))
                ax.axvline(x, color=MARK_COLORS[j], lw=1.6, zorder=3)
        ax.set_ylim(0, 1.15); ax.set_xlim(0, hi)
        ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.grid(axis='y', visible=False)
        ax.set_title(f'{GLAB[g]}  ({int(z.n_cand.iloc[0]):,} candidates; {len(real)} priors)', loc='left',
                     fontsize=8.5, color=C['ink'])
        if gi == len(groups) - 1:
            ax.set_xlabel(f'{METRIC[metric][0]} (x chance)')
        for k, (x, j, nm) in enumerate(sorted(rows, reverse=True)):
            yy = 0.95 - k * 0.2
            lab.plot([0.0, 0.1], [yy, yy], color=MARK_COLORS[j], lw=2, transform=lab.transAxes, clip_on=False)
            lab.text(0.14, yy, f'{x:5.1f}x  {short(nm)}', transform=lab.transAxes, fontsize=7, va='center', color=C['ink'])
    fig.suptitle(f'Zero-shot fill gain per single prior  ({atoms_of(tag)} atoms, ExPert graph fill, {panel_short()})\n'
                 f'marked: best single, best GenePT, EFFGATE3 text bank, STRING, DepMap-dstd',
                 x=0.01, y=1 - 0.08 / H, ha='left', va='top', color=C['ink'], fontsize=9)
    save(fig, name, run_dir, sub)


def fig_bio_vs_zs(bio, zs, run_dir, metric):
    from adjustText import adjust_text
    m = zs[zs.group == MAIN].merge(bio, on=['prior', 'family'])
    m = m[m.family != 'Random'].reset_index(drop=True)
    x = m['KEGG_lift10_matched']; y = m['x']
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for fam, lab in (('text', 'text'), ('PPI (STRING)', 'STRING'), ('DepMap', 'DepMap')):
        s = m.family == fam
        ax.scatter(x[s], y[s], s=34, color=FAM_COLOR[fam], edgecolors=C['surf'], linewidths=1.2, label=lab, zorder=3)
    s = ~m.family.isin(FAM_COLOR)
    ax.scatter(x[s], y[s], s=34, color=C['ink2'], edgecolors=C['surf'], linewidths=1.2, label='other external', zorder=3)
    ax.axhline(1.0, color=C['ink2'], lw=1, ls=':')
    ax.set_xlim(0, max(x) * 1.05); ax.set_ylim(0, max(y) * 1.1)
    marks = [nm for nm, _ in pick_marks(zs[zs.group == MAIN])]
    crowd = m.family.isin(['text', 'PPI (STRING)']) & (x > np.percentile(x, 40))     # the dense upper-right cluster
    mk = m[m.prior.isin(marks) & crowd].sort_values('x', ascending=False)
    ymax = ax.get_ylim()[1]
    for k, (_, r) in enumerate(mk.iterrows()):
        ax.annotate(short(r.prior), xy=(r['KEGG_lift10_matched'], r.x), xycoords='data',
                    xytext=(1.03, 0.92 - 0.1 * k), textcoords='axes fraction', fontsize=7, color=C['ink'], va='center',
                    arrowprops=dict(arrowstyle='-', color=C['axis'], lw=0.7, shrinkA=2, shrinkB=3))
    texts = [ax.text(r['KEGG_lift10_matched'], r.x, short(r.prior), fontsize=6.6, color=C['ink2'])
             for _, r in m.iterrows() if (r.family != 'text' or r.prior in marks) and r.prior not in set(mk.prior)]
    t = m.family == 'text'
    rho_all = spearmanr(x, y)[0]; rho_non = spearmanr(x[~t], y[~t])[0]; rho_txt = spearmanr(x[t], y[t])[0]
    ax.set_xlabel('bio quality: KEGG neighbour lift@10 (size-matched, n=2,000 genes)')
    ax.set_ylabel(f'{METRIC[metric][0]} (x chance)')
    ax.set_title('Bio quality vs zero-shot gain\n'
                 f'Spearman: all {rho_all:+.2f} (n={len(m)})   non-text {rho_non:+.2f} (n={int((~t).sum())})   '
                 f'text {rho_txt:+.2f} (n={int(t.sum())})', loc='left', color=C['ink'], fontsize=9.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.13), ncol=4, fontsize=7.5)
    adjust_text(texts, ax=ax, expand=(1.25, 1.5), arrowprops=dict(arrowstyle='-', color=C['axis'], lw=0.6))
    save(fig, 'bio_vs_zs', run_dir, MDIR[metric])
    return rho_all, rho_non, rho_txt


def cc(g, metric, kind='x'):
    """combos column names: kind in x / null / halfA / halfB."""
    p = METRIC[metric][1]
    return {'x': f'{g}.{metric}', 'null': f'{g}.{p}null_zs_x', 'halfA': f'{g}.{p}halfA', 'halfB': f'{g}.{p}halfB'}[kind]


def heldout_best(sub, g, metric):
    """split-half honest score of 'the best subset': select on one half of the candidates, score the other."""
    A = sub[cc(g, metric, 'halfA')].idxmax(); B = sub[cc(g, metric, 'halfB')].idxmax()
    return float((sub.loc[A, cc(g, metric, 'halfB')] + sub.loc[B, cc(g, metric, 'halfA')]) / 2)


def load_combos(run_dir):
    p = f'{run_dir}/combos.json'
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    return pd.json_normalize(d['rows']), d['pool']


def fig_zs_combos(cb, pool, run_dir, metric, g=MAIN):
    X, N = cc(g, metric), cc(g, metric, 'null')
    sizes = sorted(cb['size'].unique())
    hi = float(np.ceil(cb[X].max() * 1.1))
    grid = np.linspace(0, hi, 800)
    fig, ax = plt.subplots(figsize=(7.2, 1.3 + 0.62 * len(sizes)))
    best1 = cb[cb['size'] == 1][X].max()
    rng = np.random.default_rng(0)
    for i, s in enumerate(sizes):
        sub = cb[cb['size'] == s]; y0 = len(sizes) - 1 - i
        for col, c_, ls, al in ((N, C['muted'], ':', 0.16), (X, C['s1'], '-', 0.2)):
            dd = kde(sub[col], grid, 0.5) if len(sub) >= 3 else np.zeros_like(grid)
            dd = 0.5 * dd / max(dd.max(), 1e-12)
            ax.fill_between(grid, y0, y0 + dd, color=c_, alpha=al, lw=0)
            ax.plot(grid, y0 + dd, color=c_, lw=1.2 if ls == ':' else 1.8, ls=ls)
        ax.plot(sub[X], y0 - 0.1 + rng.uniform(-0.03, 0.03, len(sub)), '|', color=C['s1'], ms=4, mew=0.7)
        b = sub.loc[sub[X].idxmax()]; ho = heldout_best(sub, g, metric) if len(sub) > 1 else b[X]
        ax.plot([ho], [y0 + 0.02], 'o', ms=6, mfc=C['surf'], mec=C['s2'], mew=1.6, zorder=5)
        ax.plot([b[X]], [y0 + 0.02], 'o', ms=6.5, color=C['s2'], mec=C['surf'], mew=1.5, zorder=6)
        ax.text(1.01, y0, f'max {b[X]:.1f}x\nheld-out {ho:.1f}x\nn={len(sub)} subsets', ha='left',
                va='center', fontsize=6.3, color=C['ink2'], transform=ax.get_yaxis_transform())
    ax.axvline(best1, color=C['s2'], lw=1.1, ls='--', zorder=1)
    ax.axvline(1.0, color=C['ink2'], lw=1, ls=':', zorder=1)
    ax.set_yticks(range(len(sizes))); ax.set_yticklabels([f'k={s}' for s in sizes[::-1]])
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, hi); ax.set_ylim(-0.3, len(sizes) - 0.35)
    ax.set_xlabel(f'{METRIC[metric][0]}, {GLAB[g]} (x chance)')
    ax.set_title(f'Fusing the top singles: every subset of the pool, by number of priors k  ({panel_short()})', loc='left',
                 color=C['ink'], fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=C['s1'], lw=1.8, label='all subsets of size k'),
                       Line2D([], [], color=C['muted'], lw=1.2, ls=':', label='same subsets, shuffled-gene twins'),
                       Line2D([], [], marker='o', color=C['s2'], lw=0, ms=6, label='best subset (in-sample max)'),
                       Line2D([], [], marker='o', mfc=C['surf'], mec=C['s2'], mew=1.6, lw=0, ms=6,
                              label='best subset, split-half held-out'),
                       Line2D([], [], color=C['s2'], lw=1.1, ls='--', label='best single (in-sample)'),
                       Line2D([], [], color=C['ink2'], lw=1, ls=':', label='chance')],
              loc='upper left', bbox_to_anchor=(0.0, -0.1), ncol=3, fontsize=7)
    ax.text(0.0, -0.26, 'pool (top-10 singles, at most 3 text encoders): '
            + ', '.join(short(n) for _, n, _, _ in pool), transform=ax.transAxes, fontsize=6.3, color=C['muted'],
            va='top', wrap=True)
    save(fig, 'zs_combos_by_size', run_dir, MDIR[metric])


def fig_zs_saturation(cb, run_dir, metric, groups=(MAIN,), name='zs_combos_saturation', sub=None):
    sub = MDIR[metric] if sub is None else f'{MDIR[metric]}/{sub}'
    groups = [g for g in groups if cc(g, metric) in cb.columns]
    sizes = sorted(cb['size'].unique())
    fig, axs = plt.subplots(1, len(groups), figsize=(max(3.0 * len(groups), 4.6), 3.3))
    for ax, g in zip(np.atleast_1d(axs), groups):
        X = cc(g, metric)
        q = cb.groupby('size')[X].quantile([0.25, 0.5, 0.75]).unstack()
        mx = cb.groupby('size')[X].max()
        ho = [heldout_best(cb[cb['size'] == s], g, metric) if (cb['size'] == s).sum() > 1 else mx[s] for s in sizes]
        nl = cb.groupby('size')[cc(g, metric, 'null')].median()
        ax.fill_between(sizes, q[0.25], q[0.75], color=C['s1'], alpha=0.18, lw=0)
        ax.plot(sizes, q[0.5], '-', color=C['s1'], lw=2, label='median subset (IQR band)')
        ax.plot(sizes, mx, '-', color=C['s2'], lw=1, alpha=0.6, label='best subset, in-sample')
        ax.plot(sizes, ho, 'o-', color=C['s2'], lw=1.6, ms=6, mfc=C['surf'], mew=1.6, label='best subset, held-out')
        ax.plot(sizes, nl, ':', color=C['muted'], lw=1.5, label='shuffled twins, median')
        ax.axhline(1.0, color=C['ink2'], lw=0.8, ls=':')
        ax.set_xticks(sizes); ax.set_xlabel('priors fused (k)')
        ax.set_ylim(0, max(mx.max(), 1) * 1.12)
        ax.set_title(GLAB[g], loc='left', fontsize=8.5, color=C['ink'])
    np.atleast_1d(axs)[0].set_ylabel(METRIC[metric][0].replace(' zero-shot', '\nzero-shot') + ' (x chance)')
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2 if len(groups) == 1 else 4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Does fusing more priors keep helping?', x=0.01, ha='left', color=C['ink'], fontsize=10)
    fig.tight_layout(rect=(0, 0.13 if len(groups) == 1 else 0.07, 1, 0.97))
    save(fig, name, run_dir, sub)


def fig_zs_per_dataset(zs, cb, run_dir, unseen, metric):
    """SUPPLEMENT: per-query x over chance per dataset (log axis) for the marked priors, the spread of all priors, the
    shuffled null and the selection-free fusion of the whole top-10 pool; n queries on the right."""
    marks = pick_marks(zs[zs.group == MAIN])
    real = zs[zs.family != 'Random']
    dsg = sorted({g for g in real.group if g.startswith('ds:')})
    ref = real[real.prior == 'opus-4.8/qwen3-embed-8b'].set_index('group').x
    unseen = set(unseen)
    seen_ds = sorted([g for g in dsg if g[3:] not in unseen], key=lambda g: -ref.get(g, 0))
    ood_ds = sorted([g for g in dsg if g[3:] in unseen], key=lambda g: -ref.get(g, 0))
    order = ['all', 'seen_dataset', 'unseen_dataset', None] + seen_ds + [None] + ood_ds
    lab = {'all': 'ALL unseen-perturbation queries', 'seen_dataset': 'all SEEN datasets (ID)',
           'unseen_dataset': 'all UNSEEN datasets (OOD)'}
    full = None
    if cb is not None:
        f10 = cb[cb['size'] == cb['size'].max()]
        full = f10.iloc[0] if len(f10) == 1 else None
    fig, ax = plt.subplots(figsize=(7.4, 0.27 * len(order) + 1.6))
    lo_clip = 0.3
    ys = {}
    for i, g in enumerate(order):
        y = len(order) - 1 - i
        if g is None:
            ax.axhline(y, color=C['axis'], lw=0.6); continue
        ys[g] = y
        z = real[real.group == g]
        if z.empty:
            continue
        v = z.x.clip(lower=lo_clip)
        ax.plot([v.quantile(0.1), v.quantile(0.9)], [y, y], color=C['s1'], alpha=0.25, lw=7, solid_capstyle='butt', zorder=1)
        null = np.concatenate([np.asarray(n) for n in z.null_all])
        ax.plot(np.clip(np.percentile(null, [2.5, 97.5]), lo_clip, None), [y, y], color=C['muted'], lw=2.2, alpha=0.6, zorder=2)
        for j, (nm, why) in enumerate(marks):
            r = z[z.prior == nm]
            if len(r):
                ax.plot(max(float(r.x.iloc[0]), lo_clip), y, 'o', ms=5.5, color=MARK_COLORS[j], mec=C['surf'], mew=0.8, zorder=4)
        if full is not None and cc(g, metric) in full.index:
            ax.plot(max(float(full[cc(g, metric)]), lo_clip), y, 'D', ms=5, color=C['ink'], mec=C['surf'], mew=0.8, zorder=5)
        ax.text(1.01, y, f'{int(z.n_query.iloc[0]):,} q', transform=ax.get_yaxis_transform(), fontsize=6.5,
                va='center', color=C['ink2'])
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([lab.get(g, g[3:]) for g in ys], fontsize=7)
    ax.tick_params(axis='y', labelcolor=C['ink2'])
    for tl, g in zip(ax.get_yticklabels(), ys):
        if g in lab:
            tl.set_fontweight('bold')
    ax.set_xscale('log'); ax.set_xlim(lo_clip, None)
    ax.axvline(1.0, color=C['ink2'], lw=1, ls=':')
    ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel(f'{METRIC[metric][0]} per query\n(x chance, log scale; {panel_note()})')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C['s1'], alpha=0.25, lw=7, label='10th-90th pct of all single priors'),
         Line2D([], [], color=C['muted'], lw=2.2, alpha=0.6, label='shuffled-gene null (95%)')]
    h += [Line2D([], [], marker='o', lw=0, ms=5.5, color=MARK_COLORS[j], label=short(nm)) for j, (nm, _) in enumerate(marks)]
    if full is not None:
        h.append(Line2D([], [], marker='D', lw=0, ms=5, color=C['ink'], label='fusion of the whole top-10 pool (no selection)'))
    ax.legend(handles=h, loc='upper left', bbox_to_anchor=(0.0, -0.06 - 0.25 / (0.27 * len(order) + 1.6)), ncol=3, fontsize=6.8)
    ax.set_title('Zero-shot fill gain per dataset (supplement; per-query readout)', loc='left', color=C['ink'], fontsize=10)
    save(fig, 'zs_per_dataset', run_dir, f'{MDIR[metric]}/supplement')


ENC_FAMILY = {'qwen3-embed-0.6b': 'qwen3-embed', 'qwen3-embed-4b': 'qwen3-embed', 'qwen3-embed-8b': 'qwen3-embed',
              'nim-openai-3-large': 'OpenAI 3-large', 'openai-3-large': 'OpenAI 3-large',
              'openai-3-large-1024': 'OpenAI 3-large',
              'BioLORD': 'biomedical BERT', 'MedCPT-Article': 'biomedical BERT', 'PubMedBERT': 'biomedical BERT',
              'SapBERT': 'biomedical BERT', 'bge-m3-hf': 'bge-m3',
              'llama-embed-nemotron-8b': 'llama-nemotron', 'llama-nemotron-embed-1b': 'llama-nemotron'}
DESCRIBERS = ('opus-4.8', 'opus-4.7', 'sonnet-4.5')


ENC_CANON = {'nim-openai-3-large': 'openai-3-large'}      # NIM-hosted == OpenAI text-embedding-3-large (GenePT's release)


def text_table(zs, group=MAIN):
    """text priors as describer x embedding-encoder (GenePT has no describer axis and is dropped)."""
    t = zs[(zs.group == group) & (zs.family == 'text') & (~zs.prior.str.startswith('GenePT'))].copy()
    t['describer'] = t.prior.str.split('/').str[0]
    t['encoder'] = t.prior.str.split('/').str[1].replace(ENC_CANON)
    t['enc_family'] = t.encoder.map(ENC_FAMILY).fillna('other')
    return t


FAM_STYLE = {'qwen3-embed': (C['s1'], 'o'), 'OpenAI 3-large': (C['s2'], 's'), 'biomedical BERT': (C['s3'], '^'),
             'llama-nemotron': (C['s7'], 'D'), 'bge-m3': (C['s5'], 'v'), 'GenePT': (C['ink2'], 'P')}


def _hm(ax, piv, title, fmt='{:.1f}', cmap='Blues'):
    im = ax.imshow(piv.values, cmap=cmap, aspect='auto', vmin=np.nanmin(piv.values) * 0.9)
    hi = np.nanpercentile(piv.values, 65)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, fmt.format(v) if np.isfinite(v) else '--', ha='center', va='center', fontsize=7,
                    color=C['surf'] if np.isfinite(v) and v > hi else (C['ink'] if np.isfinite(v) else C['muted']))
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7.5)
    ax.tick_params(axis='both', labelcolor=C['ink2'], length=0)
    ax.grid(False)
    ax.set_title(title, loc='left', fontsize=8.5, color=C['ink'])
    return im


def fig_text_describer_encoder(bio, zs, run_dir, metric):
    """SUPPLEMENT-grade detail, main-figure worthy: description writer x embedding encoder on BOTH axes of quality --
    biological kNN (KEGG lift@10) and zero-shot fill -- plus their combined z-score."""
    t = _text_quality(bio, zs, metric)
    P = {k: t.pivot_table(index='encoder', columns='describer', values=v)[[d for d in DESCRIBERS if d in set(t.describer)]]
         for k, v in (('bio', 'bio'), ('zs', 'x'), ('combined', 'combined'))}
    order = P['combined'].mean(1).sort_values(ascending=False).index
    P = {k: v.loc[order] for k, v in P.items()}
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 0.30 * len(order) + 2.2))
    _hm(axs[0], P['bio'], 'biological kNN: KEGG lift@10')
    im = _hm(axs[1], P['zs'], f'{METRIC[metric][0]} (x chance)')
    axs[1].set_yticklabels([])
    fig.colorbar(im, ax=axs[1], fraction=0.035, pad=0.02).ax.tick_params(labelsize=6.5, labelcolor=C['ink2'])
    ax = axs[2]
    xs = np.arange(P['combined'].shape[1])
    lab_at = []
    for enc, row in P['combined'].iterrows():
        col = FAM_STYLE.get(ENC_FAMILY.get(enc, 'other'), (C['ink2'], 'o'))[0]
        ax.plot(xs, row.values, '-o', color=col, lw=1.3, ms=4, mec=C['surf'], mew=0.8, alpha=0.9)
        fin = np.flatnonzero(np.isfinite(row.values))
        if len(fin):
            lab_at.append([float(row.values[fin[-1]]), enc, col, float(xs[fin[-1]])])
    lab_at.sort()
    gap = (np.nanmax(P['combined'].values) - np.nanmin(P['combined'].values)) * 0.05
    for i in range(1, len(lab_at)):
        lab_at[i][0] = max(lab_at[i][0], lab_at[i - 1][0] + gap)
    for ly, enc, col, lx in lab_at:
        ax.text(lx + 0.08, ly, enc, fontsize=6.5, va='center', color=col)
    ax.set_xticks(xs); ax.set_xticklabels(P['combined'].columns, fontsize=8)
    ax.set_xlim(-0.25, len(xs) + 0.85)
    ax.set_ylabel('combined quality (mean z of bio kNN and zero-shot)', fontsize=8)
    ax.set_title('combined, same encoder across describers', loc='left', fontsize=8.5, color=C['ink'])
    ax.grid(axis='x', visible=False)
    fig.suptitle('Gene descriptions (writer) x embedding encoder  (rows ordered by combined quality)',
                 x=0.01, ha='left', color=C['ink'], fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    save(fig, 'text_describer_x_encoder_matrix', run_dir, f'{MDIR[metric]}/supplement')


GENEPT_AS_PAIR = {'GenePT/ada-002': ('NCBI (GenePT)', 'openai ada-002'),
                  'GenePT/3-large': ('NCBI+UniProt (GenePT)', 'openai-3-large')}


def _text_quality(bio, zs, metric, genept=False):
    """text priors with bio kNN, ZS and their combined z-score; `genept` adds GenePT as source x encoder pairings."""
    t = text_table(zs).merge(bio, on=['prior', 'family'])
    if genept:
        g = zs[(zs.group == MAIN) & (zs.prior.isin(GENEPT_AS_PAIR))].merge(bio, on=['prior', 'family'])
        g['describer'] = g.prior.map(lambda n: GENEPT_AS_PAIR[n][0])
        g['encoder'] = g.prior.map(lambda n: GENEPT_AS_PAIR[n][1])
        g['enc_family'] = 'GenePT'
        t = pd.concat([t, g], ignore_index=True)
    t['bio'] = t['KEGG_lift10_matched']
    t['combined'] = ((t.x - t.x.mean()) / t.x.std() + (t.bio - t.bio.mean()) / t.bio.std()) / 2
    # RAW combined score for the heat variants: both terms are already gains over a null (x over chance, KEGG lift over
    # the all-pairs rate), so their product is unit-free and needs no standardisation. Ranks with `combined` at rho 0.998.
    t['q'] = t.x * t.bio
    return t


def _ribbon(ax, x0, x1, y0, y1, h, color, alpha=0.85, hatch=None, edge=None):
    """one alluvium: a band of height h from (x0, y0) to (x1, y1), cubic on both edges."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + h), (xm, y1 + h), (xm, y0 + h), (x0, y0 + h), (x0, y0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor=(edge or (C['surf'] if hatch else 'none')),
                           lw=(0.6 if edge else 0), hatch=hatch, alpha=alpha, zorder=2))


def fig_text_alluvial(bio, zs, run_dir, metric, genept=False):
    """Description writer (left) -> embedding encoder (right); every pairing is one ribbon, coloured by combined quality."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    t = _text_quality(bio, zs, metric, genept)
    encs = t.groupby('encoder').combined.mean().sort_values(ascending=False).index.tolist()
    describers = [d for d in DESCRIBERS if d in set(t.describer)]
    describers += [d for d in t.describer.unique() if d not in describers]     # GenePT sources last
    cmap = LinearSegmentedColormap.from_list('div', [C['s8'], '#f0efec', C['s1']])   # low -> mid -> high
    v = np.abs(t.combined).max()
    norm = Normalize(-v, v)
    GAP_L, GAP_R = 0.035, 0.012
    nL = {d: int((t.describer == d).sum()) for d in describers}
    nR = {e: int((t.encoder == e).sum()) for e in encs}
    unit_L = (1 - GAP_L * (len(describers) - 1)) / sum(nL.values())
    unit_R = (1 - GAP_R * (len(encs) - 1)) / sum(nR.values())
    yL, topL = {}, 1.0
    for d in describers:
        yL[d] = topL - nL[d] * unit_L; topL = yL[d] - GAP_L
    yR, topR = {}, 1.0
    for e in encs:
        yR[e] = topR - nR[e] * unit_R; topR = yR[e] - GAP_R
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    x0, x1, bw = 0.10, 0.90, 0.028
    offL = {d: 0.0 for d in describers}; offR = {e: 0.0 for e in encs}
    for e in encs:                                   # ribbons grouped by encoder keeps crossings readable
        for d in describers:
            r = t[(t.describer == d) & (t.encoder == e)]
            if not len(r):
                continue
            _ribbon(ax, x0 + bw, x1, yL[d] + offL[d], yR[e] + offR[e], unit_R if unit_R < unit_L else unit_L,
                    cmap(norm(float(r.combined.iloc[0]))))
            offL[d] += unit_L; offR[e] += unit_R
    for d in describers:                             # nodes
        ax.add_patch(plt.Rectangle((x0, yL[d]), bw, nL[d] * unit_L, color=C['ink2'], zorder=3))
        ax.text(x0 - 0.015, yL[d] + nL[d] * unit_L / 2, d, ha='right', va='center', fontsize=8.5, color=C['ink'])
    for e in encs:
        ax.add_patch(plt.Rectangle((x1, yR[e]), bw, nR[e] * unit_R, color=C['ink2'], zorder=3))
        m = t[t.encoder == e]
        ax.text(x1 + bw + 0.015, yR[e] + nR[e] * unit_R / 2,
                f'{e}   {m.bio.mean():.1f} | {m.x.mean():.1f}', ha='left', va='center', fontsize=7.2, color=C['ink'])
    ax.set_xlim(-0.16, 1.32); ax.set_ylim(-0.03, 1.03); ax.axis('off')
    ax.text(x0 + bw / 2, 1.045, 'description writer', ha='center', fontsize=8.5, color=C['ink2'])
    ax.text(x1 + bw / 2, 1.045, 'embedding encoder   (mean bio kNN | zero-shot)', ha='center', fontsize=8.5, color=C['ink2'])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.14, aspect=28)
    cb.set_label('combined quality: mean z of bio kNN (KEGG lift@10) and zero-shot\n(0 = average text prior)', fontsize=7.5)
    cb.ax.tick_params(labelsize=7, labelcolor=C['ink2'])
    ttl = 'Every gene-description writer x embedding encoder pairing'
    if genept:
        ttl += '  (+ GenePT: its own text source x OpenAI encoder)'
        fig.text(0.01, 0.005, "GenePT ribbons are single pairings from published embeddings; its 'openai 3-large (3072d)' is the "
                 "same model family as our 1024-d openai rows, not the same vectors.", fontsize=6.3, color=C['muted'])
    fig.suptitle(ttl, x=0.01, ha='left', color=C['ink'], fontsize=10)
    fig.tight_layout(rect=(0, 0.02 if genept else 0, 1, 0.97))
    save(fig, 'text_describer_x_encoder' + ('_genept' if genept else ''), run_dir, MDIR[metric])


# generate_descriptions.py ALL_RESOURCES (stored set). ORDER = stacking order: NCBI and UniProt sit at the BOTTOM so their
# ribbons run straight into GenePT's concat nodes (also bottom) instead of crossing the whole figure.
SOURCES = ('PubMed', 'STRING', 'DepMap', 'NCBI', 'UniProt')
GENEPT_SRC = {'NCBI (GenePT)': ('NCBI',), 'NCBI+UniProt (GenePT)': ('NCBI', 'UniProt')}
# palette for the writer-coloured alluvials: validated with the dataviz validator (light surface, --pairs all:
# lightness band / chroma floor PASS, CVD dE 9.2 (>= 8 target) and normal-vision dE 27.6 (>= 15 floor); the one sub-3:1
# contrast WARN (aqua) is relieved by the visible node labels). GenePT keeps neutrals -- it has no writer.
WRITER_COLOR = {'opus-4.8': C['s7'], 'opus-4.7': C['s3'], 'sonnet-4.5': C['s2'],
                'NCBI (GenePT)': C['ink2'], 'NCBI+UniProt (GenePT)': C['muted']}
# greyscale variant: lightness carries identity (OKLab L 0.35 / 0.52 / 0.68, dL ~0.17 per step, measured); GenePT shares a
# grey and is marked by the texture channel (45 / 135 hatch) instead of a 4th step, which would fall too near the surface
WRITER_GREY = {'opus-4.8': ('#3b3a38', None), 'opus-4.7': ('#6b6a64', None), 'sonnet-4.5': ('#9a9891', None),
               'NCBI (GenePT)': ('#5c5b56', None), 'NCBI+UniProt (GenePT)': ('#8f8d86', None)}
# greyscale 3-step: the SHADE CARRIES THE SOURCE (measured OKLab L 0.30 / 0.42 / 0.53 / 0.64 / 0.75, ~0.11 per step);
# writer -> encoder ribbons are one neutral grey, since a written description mixes all five sources
SRC_GREY = dict(zip(SOURCES, ('#2f2e2c', '#55544f', '#7b7a74', '#a3a199', '#c9c7bd')))   # dark -> light down the column
NEUTRAL_GREY = '#8a8882'
# heat variant: SEQUENTIAL one-hue (neutral) ramp, light = worst -> dark = best. The light end stops at #e9e8e4 rather
# than white so the weakest ribbon still separates from the #faf9f7 surface; the dark end stops short of black.
HEAT_GREY = ('#ecebe7', '#c2c0ba', '#8e8c85', '#5a5854', '#232220')


def _alluvial_nodes(counts, gap):
    """stack nodes top-down; returns {name: (y_bottom, height)} and the unit height."""
    unit = (1 - gap * (len(counts) - 1)) / sum(counts.values())
    out, top = {}, 1.0
    for k, n in counts.items():
        out[k] = (top - n * unit, n * unit); top = out[k][0] - gap
    return out, unit


def fig_text_alluvial_bydescriber(bio, zs, run_dir, metric, three_step=False, grey=False, heat=False):
    """Same pairings and the same encoder ordering, but ribbons coloured by the DESCRIPTION WRITER.
    three_step prepends the raw text sources: source -> writer (or GenePT's template concat) -> encoder.
    heat replaces the writer colours with the RAW quality score (zero-shot x bio kNN) on a light->dark grey ramp."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    t = _text_quality(bio, zs, metric, genept=True)
    key = 'q' if heat else 'combined'
    encs = t.groupby('encoder')[key].mean().sort_values(ascending=False).index.tolist()
    writers = [d for d in DESCRIBERS if d in set(t.describer)] + [d for d in GENEPT_SRC if d in set(t.describer)]
    nR = {e: int((t.encoder == e).sum()) for e in encs}
    nM = {w: int((t.describer == w).sum()) for w in writers}
    R, unit = _alluvial_nodes(nR, 0.012)

    def _stack(heights):
        """stack boxes of GIVEN heights top-down, gaps sized to fill the axis exactly."""
        gap = max((1 - sum(heights.values())) / max(len(heights) - 1, 1), 0.006)
        out, top = {}, 1.0
        for k, h in heights.items():
            out[k] = (top - h, h); top = out[k][0] - gap
        return out
    src_of = {w: (SOURCES if w in DESCRIBERS else GENEPT_SRC[w]) for w in writers}
    M = _stack({w: nM[w] * unit for w in writers})                              # heights == their ribbon stacks
    L = _stack({s: sum(M[w][1] / len(src_of[w]) for w in writers if s in src_of[w]) for s in SOURCES})
    col = {w: (WRITER_GREY[w] if grey else (WRITER_COLOR[w], None)) for w in writers}
    if heat:                                    # sequential neutral ramp over the RAW score, no standardisation
        cmap = LinearSegmentedColormap.from_list('heatgrey', HEAT_GREY)
        norm = Normalize(float(t.q.min()), float(t.q.max()))
        qw = t.groupby('describer').q.mean(); qe = t.groupby('encoder').q.mean()
        col = {w: (cmap(norm(float(qw[w]))), None) for w in writers}
    fig, ax = plt.subplots(figsize=(9.8 if three_step else 7.6, 5.8))
    xs_, xm, xr, bw = 0.02, (0.40 if three_step else 0.10), 0.86, 0.026
    offL = {s: 0.0 for s in SOURCES}; offM_in = {w: 0.0 for w in writers}
    offM = {w: 0.0 for w in writers}; offR = {e: 0.0 for e in encs}
    if three_step:
        for w in writers:                                                       # stage 1: source -> writer
            h = M[w][1] / len(src_of[w])
            for s_ in SOURCES:
                if s_ not in src_of[w]:
                    continue
                _ribbon(ax, xs_ + bw, xm, L[s_][0] + offL[s_], M[w][0] + offM_in[w], h,
                        SRC_GREY[s_] if grey else col[w][0], alpha=0.55 if grey else (0.65 if heat else 0.5),
                        hatch=col[w][1], edge=C['surf'] if heat else None)
                offL[s_] += h; offM_in[w] += h
    for e in encs:                                                              # stage 2: writer -> encoder
        for w in writers:
            r = t[(t.describer == w) & (t.encoder == e)]
            if not len(r):
                continue
            fc = (cmap(norm(float(r.q.iloc[0]))) if heat else
                  (NEUTRAL_GREY if (grey and three_step) else col[w][0]))
            _ribbon(ax, xm + bw, xr, M[w][0] + offM[w], R[e][0] + offR[e], unit, fc,
                    alpha=0.92 if heat else 0.8, hatch=col[w][1], edge=C['surf'] if heat else None)
            offM[w] += unit; offR[e] += unit
    for w in writers:
        ax.add_patch(plt.Rectangle((xm, M[w][0]), bw, M[w][1],
                                   facecolor=NEUTRAL_GREY if (grey and three_step and not heat) else col[w][0],
                                   edgecolor=C['surf'], hatch=col[w][1], lw=0.6 if heat else 0, zorder=3))
        ax.text(xm + bw / 2, M[w][0] + M[w][1] + 0.008, w, ha='center', va='bottom', fontsize=7.5, color=C['ink'])
    for e in encs:
        ax.add_patch(plt.Rectangle((xr, R[e][0]), bw, R[e][1],
                                   facecolor=cmap(norm(float(qe[e]))) if heat else C['ink2'],
                                   edgecolor=C['surf'] if heat else 'none', lw=0.6 if heat else 0, zorder=3))
        ax.text(xr + bw + 0.012, R[e][0] + R[e][1] / 2, e, ha='left', va='center', fontsize=7.6, color=C['ink'])
    if three_step:
        for s_ in SOURCES:
            ax.add_patch(plt.Rectangle((xs_, L[s_][0]), bw, L[s_][1],
                                       color=SRC_GREY[s_] if grey else C['axis'], zorder=3))
            ax.text(xs_ - 0.012, L[s_][0] + L[s_][1] / 2, s_, ha='right', va='center', fontsize=7.5, color=C['ink'])
        ax.text(xs_ + bw / 2, 1.085, 'raw text source', ha='center', fontsize=8.5, color=C['ink2'])
    ax.set_xlim(-0.14 if three_step else -0.16, 1.30); ax.set_ylim(-0.06, 1.12); ax.axis('off')
    ax.text(xm + bw / 2, 1.085, 'description writer', ha='center', fontsize=8.5, color=C['ink2'])
    ax.text(xr + bw / 2, 1.085, 'embedding encoder', ha='center', fontsize=8.5, color=C['ink2'])
    if heat:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', location='bottom',
                          fraction=0.032, pad=0.01, aspect=45, shrink=0.5, anchor=(0.5, 1.0))
        cb.set_label(f'{METRIC[metric][0]} (x chance)  x  biological kNN (KEGG lift@10)', fontsize=7.5,
                     color=C['ink2'], labelpad=3)
        cb.ax.tick_params(labelsize=7, labelcolor=C['ink2'], length=2.5, pad=1.5); cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, 'text_describer_x_encoder_genept_' + ('3step' if three_step else 'bywriter') + ('_grey' if grey else ''),
         run_dir, MDIR[metric])


def fig_bio_vs_zs_text(bio, zs, run_dir, metric):
    """MAIN: text priors only, one panel, coloured (and shaped) by embedding encoder; every encoder appears once per
    description writer."""
    from adjustText import adjust_text
    t = text_table(zs).merge(bio, on=['prior', 'family'])
    gp = zs[(zs.group == MAIN) & (zs.prior.str.startswith('GenePT'))].merge(bio, on=['prior', 'family'])
    gp['enc_family'] = 'GenePT'; gp['encoder'] = gp.prior.str.split('/').str[1]
    t = pd.concat([t, gp], ignore_index=True)
    xl, yl = 'KEGG_lift10_matched', 'x'
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for f in ('qwen3-embed', 'OpenAI 3-large', 'biomedical BERT', 'llama-nemotron', 'bge-m3', 'GenePT'):
        sub = t[t.enc_family == f]
        if not len(sub):
            continue
        col, mk = FAM_STYLE[f]
        ax.scatter(sub[xl], sub[yl], s=38, marker=mk, color=col, edgecolors=C['surf'], linewidths=1.0, label=f, zorder=3)
    texts = []
    for enc, sub in t.groupby('encoder'):                       # label each encoder once, at its best describer
        r = sub.loc[sub[yl].idxmax()]
        texts.append(ax.text(r[xl], r[yl], short(r.encoder) if '/' in str(r.prior) and r.enc_family == 'GenePT' else r.encoder,
                             fontsize=6.4, color=C['ink2']))
    adjust_text(texts, ax=ax, expand=(1.3, 1.6), arrowprops=dict(arrowstyle='-', color=C['axis'], lw=0.5))
    rho = spearmanr(t[xl], t[yl])[0]
    ax.set_xlabel('biological kNN: KEGG neighbour lift@10 (size-matched, n=2,000 genes)')
    ax.set_ylabel(f'{METRIC[metric][0]} (x chance)')
    ax.set_title(f'Text priors: biological quality vs zero-shot gain  (Spearman {rho:+.2f})\n'
                 f'one point per description writer x encoder; labels mark each encoder at its best writer',
                 loc='left', color=C['ink'], fontsize=9.5)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.13), ncol=3, fontsize=7.5, title='embedding encoder',
              title_fontsize=7.5)
    save(fig, 'bio_vs_zs_text', run_dir, MDIR[metric])


def fig_bio_vs_zs_text_by_family(bio, zs, run_dir, metric):
    """SUPPLEMENT: the same, split into one panel per encoder family and coloured by description writer."""
    from adjustText import adjust_text
    t = text_table(zs).merge(bio, on=['prior', 'family'])
    fams = [f for f in ('qwen3-embed', 'OpenAI 3-large', 'biomedical BERT', 'llama-nemotron', 'bge-m3')
            if (t.enc_family == f).any()]
    n = len(fams)
    fig, axs = plt.subplots(1, n, figsize=(2.05 * n, 3.1), sharex=True, sharey=True)
    xl, yl = 'KEGG_lift10_matched', 'x'
    for ax, f in zip(np.atleast_1d(axs), fams):
        ax.scatter(t[xl], t[yl], s=16, color=C['grid'], edgecolors='none', zorder=1)   # all text priors as context
        sub = t[t.enc_family == f]
        for j, d in enumerate(DESCRIBERS):
            q = sub[sub.describer == d]
            ax.scatter(q[xl], q[yl], s=34, color=MARK_COLORS[j], edgecolors=C['surf'], linewidths=1.0, label=d, zorder=3)
        texts = [ax.text(r[xl], r[yl],
                         r.encoder.replace('qwen3-embed-', '').replace('-hf', '').replace('nim-openai-3-large', 'nim')
                         .replace('llama-embed-nemotron-8b', '8b').replace('llama-nemotron-embed-1b', '1b')
                         .replace('openai-3-large-1024/openai-3-large', '1024').replace('MedCPT-Article', 'MedCPT'),
                         fontsize=5.8, color=C['ink2']) for _, r in sub.iterrows()]
        adjust_text(texts, ax=ax, expand=(1.4, 1.7), arrowprops=dict(arrowstyle='-', color=C['axis'], lw=0.4))
        ax.set_title(f, loc='left', fontsize=8, color=C['ink'])
        ax.set_xlabel('')
    np.atleast_1d(axs)[0].set_ylabel(f'{METRIC[metric][0]}\n(x chance)', fontsize=8)
    fig.supxlabel('biological kNN: KEGG neighbour lift@10 (size-matched, n=2,000 genes)', fontsize=8, color=C['ink2'], y=0.12)
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04), title='description writer',
               title_fontsize=7.5)
    fig.suptitle('Text priors by encoder family (supplement; grey = all text priors)', x=0.01, ha='left',
                 color=C['ink'], fontsize=9.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    save(fig, 'bio_vs_zs_text_by_family', run_dir, f'{MDIR[metric]}/supplement')


def fig_bio(bio, run_dir):
    b = bio.sort_values('KEGG_lift10_matched', ascending=True).reset_index(drop=True)
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 0.16 * len(b) + 1.3), sharey=True)
    y = np.arange(len(b))
    col = [C['muted'] if l else FAM_COLOR.get(f, C['ink2']) for f, l in zip(b.family, b.leak)]
    for ax, lib in zip(axs, ('KEGG', 'Reactome', 'GO BP')):
        v = b[f'{lib}_lift10_matched']; sd = b[f'{lib}_lift10_matched_sd']
        ax.hlines(y, 1.0, v, color=C['grid'], lw=1, zorder=1)
        ax.errorbar(v, y, xerr=sd, fmt='none', ecolor=C['axis'], lw=0.8, zorder=2)
        ax.scatter(v, y, s=16, c=col, edgecolors=C['surf'], linewidths=0.6, zorder=3)
        ax.axvline(1.0, color=C['ink2'], lw=1, ls=':')
        ax.set_title(lib, loc='left', color=C['ink'], fontsize=9)
        ax.grid(axis='y', visible=False)
        ax.set_xlim(0, None)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([pretty(p) + (' [screen-derived]' if l else '') for p, l in zip(b.prior, b.leak)], fontsize=6.5)
    axs[0].set_ylim(-0.8, len(b) - 0.2)
    axs[1].set_xlabel('neighbour lift@10 over random gene pairs  (n=2,000 annotated genes per prior, mean ± sd of 5 draws)')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', lw=0, color=c_, label=l_) for l_, c_ in
                        (('text', C['s1']), ('STRING', C['s2']), ('DepMap', C['s3']), ('other external', C['ink2']),
                         ('screen-derived', C['muted']))],
               loc='upper left', bbox_to_anchor=(0.0, 0.975), ncol=5, fontsize=7.5)
    fig.suptitle('Biological quality of every prior (cosine kNN vs pathway co-membership)', x=0.01, ha='left',
                 y=0.995, color=C['ink'], fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, 'bio_quality', run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True, help='screen.py --out-dir')
    ap.add_argument('--tag', required=True, help='atoms tag used by screen.py zs / combos')
    ap.add_argument('--split-meta', required=True, help='splits metadata.json (held-out datasets per fold)')
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--metric', default='w_zs_x', choices=tuple(METRIC), help='readout shown in the figures')
    a = ap.parse_args()
    run_dir = f'{a.out_dir}/{a.tag}'
    bio, zs = collect(a.out_dir, a.tag, a.metric)
    os.makedirs(run_dir, exist_ok=True)
    if len(bio):
        bio.to_csv(f'{a.out_dir}/bio_quality.tsv', sep='\t', index=False, float_format='%.4f')
        fig_bio(bio, run_dir)
    cb, pool = load_combos(run_dir)
    if len(zs):
        zs.drop(columns=['null_all']).to_csv(f'{run_dir}/zs_singles.tsv', sep='\t', index=False, float_format='%.4f')
        for mt in ('w_zs_x', 'zs_x'):
            zs.pivot_table(index=['prior', 'family'], columns='group', values=mt).to_csv(
                f'{run_dir}/zs_per_group_{mt}.tsv', sep='\t', float_format='%.3f')
        fig_zs_singles(zs, a.tag, run_dir, a.metric)
        fig_zs_singles(zs, a.tag, run_dir, a.metric, ('pooled_seen_dataset', 'pooled_unseen_dataset'),
                       'zs_singles_density_split', 'supplement')
        md = json.load(open(a.split_meta))
        fig_zs_per_dataset(zs, cb, run_dir, [x for f in md['folds'] if int(f['fold']) == a.fold
                                             for x in f['held_out_datasets']], a.metric)
    if len(bio) and len(zs):
        print('[rho] bio vs zs (all, non-text, text):', fig_bio_vs_zs(bio, zs, run_dir, a.metric))
        fig_bio_vs_zs_text(bio, zs, run_dir, a.metric)
        fig_bio_vs_zs_text_by_family(bio, zs, run_dir, a.metric)
        fig_text_describer_encoder(bio, zs, run_dir, a.metric)
        fig_text_alluvial(bio, zs, run_dir, a.metric)
        fig_text_alluvial(bio, zs, run_dir, a.metric, genept=True)
        fig_text_alluvial_bydescriber(bio, zs, run_dir, a.metric)
        fig_text_alluvial_bydescriber(bio, zs, run_dir, a.metric, three_step=True, heat=True)
        fig_text_alluvial_bydescriber(bio, zs, run_dir, a.metric, grey=True)
        fig_text_alluvial_bydescriber(bio, zs, run_dir, a.metric, three_step=True, grey=True)
    if cb is not None:
        cb.sort_values(cc(MAIN, a.metric), ascending=False).to_csv(f'{run_dir}/zs_combos_{MDIR[a.metric]}.tsv', sep='\t',
                                                                   index=False, float_format='%.4f')
        fig_zs_combos(cb, pool, run_dir, a.metric)
        fig_zs_saturation(cb, run_dir, a.metric)
        fig_zs_saturation(cb, run_dir, a.metric, ('pooled_seen_dataset', 'pooled_unseen_dataset'),
                          'zs_combos_saturation_split', 'supplement')


if __name__ == '__main__':
    main()
