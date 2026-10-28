#!/usr/bin/env python
"""Figures for the gene-selection sweep. Writes SVG + 300-dpi PNG side by side.

Every panel is a light-mode static figure for a results directory, so the theme-aware
machinery of an interactive chart does not apply; the palette, mark specs and the
"direct-label rather than rely on a low-contrast hue" rule do.
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
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

# Validated categorical palette, fixed order -- never cycled, never reassigned by rank.
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
     'magenta': '#e87ba4', 'green': '#008300', 'violet': '#4a3aa7', 'red': '#e34948'}
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
       '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8983'
SURFACE, GRID = '#fcfcfb', '#e6e5e1'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'legend.frameon': False, 'figure.autolayout': False,
})

# strategy -> (label, colour) in fixed slot order
STRAT = {
    'per_dataset':    ('per-dataset (zero-padded)', C['blue']),
    'union':          ('union (keep all)',          C['orange']),
    'global':         ('global consensus',          C['aqua']),
    'hvg_union':      ('HVG union (current)',       C['yellow']),
    'intersection':   ('intersection (shared)',     C['magenta']),
    'random':         ('random',                    MUTED),
}


def save(fig, out: Path, name: str) -> None:
    for ext, kw in (('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)
    print(f'  wrote {name}.svg / .png', flush=True)


def thousands(x, _):
    return f'{int(x):,}'


def fig_size_curve(d: pd.DataFrame, out: Path) -> None:
    """Decodability vs ACTUAL gene-set size -- the sets are not the same size."""
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for strat, (lab, col) in STRAT.items():
        g = (d[d.strategy == strat].groupby('n')
             .agg(x=('n_genes', 'mean'), y=('top1', 'mean')).reset_index()
             .sort_values('x'))
        if g.empty:
            continue
        ax.plot(g.x, g.y, '-o', color=col, lw=2, ms=5,
                mec=SURFACE, mew=1.2, label=lab, zorder=3)

    best = d[d.strategy == 'per_dataset'].groupby('n')['top1'].mean()
    bx = d[(d.strategy == 'per_dataset') & (d.n == best.idxmax())]['n_genes'].mean()
    ax.axvline(bx, color=MUTED, lw=1, ls=':', zorder=1)
    lo, hi = ax.get_ylim()
    ax.annotate(f'optimum: {best.idxmax():,} genes/dataset', (bx, lo),
                xytext=(-6, 8), textcoords='offset points', color=MUTED,
                fontsize=8.5, va='bottom', ha='right')

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.set_xlabel('genes in the set (actual size, log scale)')
    ax.set_ylabel('top-1 replicate-matching accuracy')
    ax.set_title('More genes is worse past ~500 per dataset',
                 loc='left', color=INK, pad=12)
    # Six series is past the direct-label range, so identity rides on the legend; the
    # upper-right is empty because every curve has turned over by then.
    ax.legend(loc='upper left', bbox_to_anchor=(1.015, 1.0), fontsize=9,
              labelcolor=INK2, handlelength=2.2, labelspacing=0.7)
    ax.set_ylim(lo, hi)
    save(fig, out, 'decodability_vs_size')


def fig_per_dataset_curves(d: pd.DataFrame, out: Path) -> None:
    """Every dataset's own curve -- shows how flat the optimum is."""
    p = d[d.strategy == 'per_dataset']
    piv = p.pivot_table(index='dataset', columns='n', values='top1')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0),
                                  gridspec_kw={'width_ratios': [1.5, 1]})
    ns = np.array(sorted(piv.columns))
    for ds in piv.index:                       # one thin line per dataset, no legend
        ax.plot(ns, piv.loc[ds, ns].to_numpy() / piv.loc[ds, ns].max(),
                color=MUTED, lw=0.9, alpha=0.55, zorder=2)
    med = (piv[ns].div(piv[ns].max(axis=1), axis=0)).median()
    ax.plot(ns, med.to_numpy(), color=C['blue'], lw=2.5, zorder=4, label='median')
    ax.axhline(0.98, color=C['orange'], lw=1.5, ls='--', zorder=3)
    ax.annotate('98% of each dataset\'s own best', (ns[-1], 0.98), xytext=(-4, 6),
                textcoords='offset points', ha='right', color=C['orange'], fontsize=8.5)
    ax.annotate('median', (ns[0], med.iloc[0]), xytext=(6, -12),
                textcoords='offset points', color=C['blue'], fontsize=8.5)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2.5, 5)))
    ax.set_xlabel('genes selected per dataset')
    ax.set_ylabel('accuracy / that dataset\'s best')
    ax.set_title('Each dataset, normalised to its own optimum',
                 loc='left', color=INK, pad=12)

    best_n = piv.idxmax(axis=1).value_counts().sort_index()
    ax2.bar([str(int(i)) for i in best_n.index], best_n.to_numpy(),
            color=C['blue'], width=0.62, zorder=3)
    for x, v in zip(range(len(best_n)), best_n.to_numpy()):
        ax2.annotate(str(int(v)), (x, v), xytext=(0, 3), textcoords='offset points',
                     ha='center', color=INK2, fontsize=9)
    ax2.set_xlabel('that dataset\'s optimal n')
    ax2.set_ylabel('datasets')
    ax2.set_title(f'Optimum per dataset (median {int(piv.idxmax(axis=1).median()):,})',
                  loc='left', color=INK, pad=12)
    ax2.grid(axis='x', visible=False)
    fig.tight_layout()
    save(fig, out, 'per_dataset_optimum')


def fig_union_cost(d: pd.DataFrame, out: Path) -> None:
    """Zero-pad vs keep, per dataset, at each dataset's own optimal n."""
    p = d[d.strategy == 'per_dataset']
    piv = p.pivot_table(index='dataset', columns='n', values='top1')
    rows = []
    for ds in piv.index:
        n = int(piv.loc[ds].idxmax())
        u = d[(d.dataset == ds) & (d.strategy == 'union') & (d.n == n)]
        if len(u):
            rows.append({'dataset': ds, 'n': n, 'zero_pad': piv.loc[ds, n],
                         'keep': float(u['top1'].iloc[0])})
    r = pd.DataFrame(rows).sort_values('zero_pad')
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(r))
    ax.hlines(y, r['keep'], r['zero_pad'], color=GRID, lw=2.5, zorder=2)
    ax.plot(r['keep'], y, 'o', color=C['orange'], ms=7, mec=SURFACE, mew=1.4,
            label='keep all union genes', zorder=3)
    ax.plot(r['zero_pad'], y, 'o', color=C['blue'], ms=7, mec=SURFACE, mew=1.4,
            label='zero-pad outside own top-n', zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(r['dataset'], fontsize=8)
    ax.set_xlabel('top-1 replicate-matching accuracy')
    ax.set_title(f'Keeping the union hurts every dataset '
                 f'(mean {(r["keep"]-r["zero_pad"]).mean():+.3f})',
                 loc='left', color=INK, pad=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', visible=False)
    save(fig, out, 'union_cost')


def fig_overlap(out: Path) -> None:
    """Pairwise Jaccard of the per-dataset selections, for each selection rule."""
    panels = [(n, out / f'overlap_jaccard_{n}.csv') for n in ('fstat', 'expr', 'hvg')]
    panels = [(n, f) for n, f in panels if f.exists()]
    if not panels:
        print('  (no overlap_jaccard_*.csv yet, skipping)')
        return
    titles = {'fstat': 'separability (F-stat)', 'expr': 'mean expression',
              'hvg': 'seurat_v3 HVG (current)'}
    # one shared order + one shared scale, so the three panels are comparable; Marson
    # first so the within-study block is visible
    J0 = pd.read_csv(panels[0][1], index_col=0)
    names = list(J0.index)
    order = ([n for n in names if n.startswith('Marson')] +
             [n for n in names if not n.startswith('Marson')])
    vmax = max(float(np.nanmax(np.where(np.eye(len(names), dtype=bool), np.nan,
                                        pd.read_csv(f, index_col=0)
                                        .loc[order, order].to_numpy())))
               for _, f in panels)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', SEQ)
    n_mars = sum(1 for n in order if n.startswith('Marson'))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.1 * len(panels), 5.6))
    axes = np.atleast_1d(axes)
    for ax, (name, f) in zip(axes, panels):
        J = pd.read_csv(f, index_col=0).loc[order, order]
        m = J.to_numpy().copy()
        np.fill_diagonal(m, np.nan)          # self-overlap is 1 by definition
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        # mark the study-family block: Marson is 12 of 22 datasets, one publication
        for v in (n_mars - 0.5,):
            ax.axhline(v, color=C['orange'], lw=1.4)
            ax.axvline(v, color=C['orange'], lw=1.4)
        med = np.nanmedian(m)
        ax.set_title(f'{titles.get(name, name)}\nmedian {med:.3f}', loc='left',
                     color=INK, fontsize=11, pad=8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=5.5)
        ax.grid(visible=False)
        if ax is axes[0]:
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=5.5)
        else:
            ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label('Jaccard overlap of the two selections', color=INK2, fontsize=9)
    cb.outline.set_visible(False)
    fig.suptitle('Selections agree within a study, not across it '
                 '(orange line = Marson2025 block, 12 of 22 datasets)',
                 x=0.02, ha='left', color=INK, fontsize=12.5)
    save(fig, out, 'gene_overlap_heatmap')


def fig_sharing(out: Path) -> None:
    f = out / 'gene_sharing.csv'
    if not f.exists():
        print('  (no gene_sharing.csv yet, skipping)')
        return
    sh = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    n_ds = int(sh['n_datasets'].max())
    bins = np.arange(0.5, n_ds + 1.5)
    for col, lab, c in (('n_datasets', 'separability (F-stat)', C['blue']),
                        ('n_datasets_expr', 'mean expression', C['orange']),
                        ('n_datasets_hvg', 'HVG (current)', C['yellow'])):
        if col not in sh.columns:
            continue
        v = sh[col][sh[col] > 0]
        ax.hist(v, bins=bins, histtype='step', lw=2, edgecolor=c, label=lab, zorder=3)
    ax.set_xlabel('number of datasets that selected the gene')
    ax.set_ylabel('genes')
    ax.set_yscale('log')
    one = int((sh['n_datasets'] == 1).sum())
    tot = int((sh['n_datasets'] > 0).sum())
    ax.set_title(f'{one:,} of {tot:,} F-stat genes ({one/tot:.0%}) are picked by exactly '
                 f'one dataset', loc='left', color=INK, pad=12)
    ax.legend(fontsize=9)
    save(fig, out, 'gene_sharing')


def fig_concentration(out: Path) -> None:
    f = out / 'uniformity_ranks.csv.gz'
    if not f.exists():
        print('  (no uniformity_ranks.csv.gz yet, skipping)')
        return
    d = pd.read_csv(f)
    d['frac'] = d['rank'] / d['n_ref']
    RULE = (('fstat', 'separability (F-stat)', C['blue']),
            ('expr', 'mean expression', C['orange']),
            ('hvg', 'HVG (current)', C['yellow']),
            ('random', 'random', MUTED))
    fams = [x for x in ('Marson', 'other') if (d.family == x).any()]
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(5.0 * (len(fams) + 1), 4.9))

    # rank CDF per family -- the whole distribution, so a mean cannot hide a split
    for ax, fam in zip(axes, fams):
        sub = d[d.family == fam]
        for rule, lab, col in RULE:
            v = np.sort(sub[sub.rule == rule]['frac'].to_numpy())
            if not v.size:
                continue
            ax.plot(v, np.arange(1, v.size + 1) / v.size, color=col, lw=2, label=lab,
                    zorder=3)
        ax.set_xscale('log')
        ax.set_xlabel('rank of the true perturbation / n_perturbations')
        if ax is axes[0]:
            ax.set_ylabel('fraction of held-out perturbations')
        n = int((d.family == fam).sum() / d[d.family == fam].rule.nunique())
        ax.set_title(f'{fam}  ({n:,} perturbation replicates)', loc='left', color=INK,
                     pad=10)
        ax.legend(fontsize=8.5, loc='lower right')

    # per-dataset top1: is the gain spread across datasets or concentrated?
    ax = axes[-1]
    t = (d.groupby(['dataset', 'rule'])
         .apply(lambda g: (g['rank'] == 1).mean(), include_groups=False).unstack()
         .sort_values('fstat'))
    y = np.arange(len(t))
    ax.hlines(y, t['hvg'], t['fstat'], color=GRID, lw=2.5, zorder=2)
    ax.plot(t['hvg'], y, 'o', color=C['yellow'], ms=6, mec=SURFACE, mew=1.3,
            label='HVG (current)', zorder=3)
    ax.plot(t['expr'], y, 'o', color=C['orange'], ms=5, mec=SURFACE, mew=1.2,
            label='mean expression', zorder=4)
    ax.plot(t['fstat'], y, 'o', color=C['blue'], ms=6, mec=SURFACE, mew=1.3,
            label='separability', zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=6.5)
    ax.set_xlabel('top-1 accuracy')
    ax.set_title('Per dataset (n=500 genes each)', loc='left', color=INK, pad=10)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, out, 'concentration')


def fig_by_strength(out: Path) -> None:
    """Decodability stratified by how reproducible each perturbation is.

    The concern a mean cannot answer: does a gene set buy its win on strong perturbations
    while losing weak ones?  Perturbations are binned by the cosine between their OWN two
    replicates -- a reproducibility ceiling that is independent of the gene set being
    judged.  (An L2 norm of the effect was tried first and rejected: it correlates -0.96
    with cell count, so it measures sampling noise, not effect size.)
    """
    import glob
    rf = out / 'uniformity_ranks.csv.gz'
    sf = sorted(glob.glob(str(out / '_pertstats' / '*.csv')))
    if not rf.exists() or not sf:
        print('  (need uniformity_ranks + _pertstats, skipping)')
        return
    st = pd.concat([pd.read_csv(f) for f in sf])
    d = pd.read_csv(rf).merge(st, on=['dataset', 'label'], how='left')
    d['fam'] = np.where(d.dataset.str.startswith('Marson'), 'Marson', 'other')
    d['hit'] = (d['rank'] == 1).astype(float)
    d = d.dropna(subset=['rep_cos'])
    d['dec'] = d.groupby('dataset')['rep_cos'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 10, labels=False) + 1)

    RULE = (('fstat', 'separability (F-stat)', C['blue']),
            ('expr', 'mean expression', C['orange']),
            ('hvg', 'HVG (current)', C['yellow']),
            ('random', 'random', MUTED))
    fams = ['Marson', 'other']
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    for ax, fam in zip(axes[:2], fams):
        sub = d[d.fam == fam]
        for rule, lab, col in RULE:
            t = sub[sub.rule == rule].groupby('dec')['hit'].mean()
            if t.empty:
                continue
            ax.plot(t.index, t.to_numpy(), '-o', color=col, lw=2, ms=5, mec=SURFACE,
                    mew=1.2, label=lab, zorder=3)
        ax.set_xlabel('replicate-agreement decile  (1 = least reproducible)')
        if ax is axes[0]:
            ax.set_ylabel('top-1 accuracy')
        ax.set_title(f'{fam}', loc='left', color=INK, pad=10)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=8.5, loc='upper left')

    # where the F-stat gain over the CURRENT pipeline actually lands
    ax = axes[2]
    for fam, col, mk in (('Marson', C['blue'], 'o'), ('other', C['aqua'], 's')):
        sub = d[d.fam == fam]
        f_ = sub[sub.rule == 'fstat'].groupby('dec')['hit'].mean()
        h_ = sub[sub.rule == 'hvg'].groupby('dec')['hit'].mean()
        ax.plot(f_.index, (f_ - h_).to_numpy(), '-' + mk, color=col, lw=2, ms=5,
                mec=SURFACE, mew=1.2, label=fam, zorder=3)
    ax.axhline(0, color=MUTED, lw=1, zorder=2)
    ax.set_xlabel('replicate-agreement decile')
    ax.set_ylabel('top-1 gain of F-stat over HVG')
    ax.set_title('Where the gain lands', loc='left', color=INK, pad=10)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8.5)
    fig.suptitle('Weak perturbations are not sacrificed: on Marson the gain is LARGEST '
                 'for the least reproducible perturbations',
                 x=0.02, ha='left', color=INK, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, 'decodability_by_strength')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='results/gene_sweep/sweep_20ds_holdout.csv')
    ap.add_argument('--out-dir', default='results/gene_sweep')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.sweep)
    d = d[d.strategy != 'all_genes']
    print(f'plotting from {args.sweep} ({d.dataset.nunique()} datasets)', flush=True)
    fig_size_curve(d, out)
    fig_per_dataset_curves(d, out)
    fig_union_cost(d, out)
    fig_overlap(out)
    fig_sharing(out)
    fig_concentration(out)
    fig_by_strength(out)


if __name__ == '__main__':
    main()
