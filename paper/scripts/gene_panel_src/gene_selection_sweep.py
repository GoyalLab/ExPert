#!/usr/bin/env python
"""How many genes does perturbation decodability actually need, and does the union hurt?

Two questions, one sweep:

  1. SIZE.  Rank genes by within-dataset separability, take the top n for
     n in {1000..5000}, and score decodability inside that dataset.  Where the curve
     flattens is how many genes the gain needs.

  2. UNION.  The pipeline selects per dataset and then UNIONS across datasets.  For each
     dataset, compare its own top-n against the union of everyone's top-n.  The union is
     much larger, so the question is whether the extra genes dilute.

     Note a structural reason the union may be cheap: a gene a dataset never measured is
     ZERO-PADDED, and an all-zero column changes neither the dot product nor either norm
     in a cosine similarity.  Dilution can only come from genes that ARE measured in that
     dataset but are noisy there.

Strategies compared at each size, all scored per dataset and averaged:

    per_dataset    that dataset's own top-n           (upper bound; a different axis per dataset)
    union          union over datasets of each top-n  (what the pipeline would ship)
    global         top-n by mean percentile rank      (shared axis, n genes)
    global_matched the same ranking truncated to |union| genes
    intersection   top-n among genes measured in EVERY dataset (`var_merge: intersection`)
    hvg_union      union over datasets of top-n HVG   (what the pipeline ships TODAY)
    random         size-matched to `union`            (control)

`global` and `union` are NOT the same size -- a union of per-dataset top-n is far larger
than n.  Since bigger sets dilute, comparing them directly confounds ranking quality with
set size, which is what `global_matched` is for: same ranking, same size as the union.

Pass 1 computes a per-dataset F-stat vector (38,638 floats -- tiny) so the candidate sets
can be built without holding any expression in memory.  Pass 2 reloads one dataset at a
time to score.  Nothing ever holds two datasets' pseudobulks at once.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decode import (  # noqa: E402
    dataset_rows, decodability, delta_vs_control, f_stat_one, load_pseudobulk, normalize)


def split_labels(labels: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 50/50 split of the PERTURBATION set (not the cells).

    Selecting genes and scoring decodability on the same perturbations lets a gene that is
    quiet across replicates by chance be picked BECAUSE it is quiet and then score well
    BECAUSE it is quiet.  Splitting by perturbation breaks that: the gene set is global,
    so a set chosen on disjoint perturbations has to generalise to be worth anything.
    """
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    half = uniq.size // 2
    return uniq[perm[:half]], uniq[perm[half:]]


def load_one(path: Path, variant: str):
    """Load one dataset's pseudobulk and return (P, obs, measured-gene mask)."""
    X, obs, var = load_pseudobulk(path)
    keep = np.ones(len(obs), dtype=bool)
    if 'variant' in obs.columns:
        keep = (obs['variant'].to_numpy() == variant)
    X, obs = X[keep], obs[keep].reset_index(drop=True)
    measured = (X > 0).any(axis=0)
    P = delta_vs_control(normalize(X, obs), obs)
    return P, obs, measured, var


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sizes', type=int, nargs='+',
                    default=[100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000])
    ap.add_argument('--variant', default='rep')
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--var-prior-quantile', type=float, default=None,
                    help='moderate the F denominator by this quantile of the '
                         'positive within-variances; omit for the raw ratio')
    ap.add_argument('--min-coverage', type=int, default=2,
                    help='datasets a gene must be measured in to enter the global rank')
    ap.add_argument('--holdout', action='store_true',
                    help='select genes on half the perturbations, score on the other '
                         'half. Without this the F-stat is fit and scored on the SAME '
                         'replicates, which flatters any variance-based ranking.')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    store = Path(args.store)
    pb_dir = store / 'pseudobulk'
    datasets = args.datasets or sorted(p.stem for p in pb_dir.glob('*.h5ad'))
    print(f'{len(datasets)} datasets with pseudobulks\n')

    # ---------------- pass 1: per-dataset F-stat + HVG rank ----------------
    print('[pass 1] per-dataset separability')
    F, M, HV, n_var = {}, {}, {}, None
    SCORE_LABELS: dict = {}
    for ds in datasets:
        P, obs, measured, var = load_one(pb_dir / f'{ds}.h5ad', args.variant)
        n_var = len(var)
        sub = dataset_rows(obs, ds, variant=args.variant)
        if args.holdout:
            sel_labs, score_labs = split_labels(obs['label'].to_numpy()[sub])
            SCORE_LABELS[ds] = set(score_labs.tolist())
            sub = sub[np.isin(obs['label'].to_numpy()[sub], sel_labs)]
        if sub.size < 4:
            print(f'  {ds}: too few replicated perturbations, skipped')
            continue
        F[ds] = f_stat_one(P, obs, sub, var_prior_quantile=args.var_prior_quantile)
        M[ds] = measured
        h = pd.read_parquet(store / 'stats' / f'{ds}.hvg.parquet')
        col = ('highly_variable_rank' if 'highly_variable_rank' in h.columns
               else 'variances_norm')
        HV[ds] = h.sort_values(col, ascending=(col == 'highly_variable_rank')) \
                  .index.to_numpy()
        print(f'  {ds}: {sub.size:,} groups, {int(measured.sum()):,} genes measured')
        del P, obs
    datasets = [d for d in datasets if d in F]

    # ---------------- build candidate gene sets ----------------
    # global rank: mean PERCENTILE rank among datasets that measured the gene, so a
    # dataset with a large F scale cannot dominate and unmeasured genes do not drag.
    pct = np.full((len(datasets), n_var), np.nan)
    for i, ds in enumerate(datasets):
        f = np.where(M[ds], F[ds], np.nan)
        ok = np.isfinite(f)
        r = np.full(n_var, np.nan)
        r[ok] = pd.Series(f[ok]).rank(pct=True).to_numpy()
        pct[i] = r
    coverage = np.isfinite(pct).sum(0)
    with np.errstate(invalid='ignore'):
        global_score = np.nanmean(pct, axis=0)
    global_score[coverage < args.min_coverage] = -np.inf
    global_order = np.argsort(-global_score)
    print(f'\n  gene coverage across datasets: median {int(np.median(coverage))}, '
          f'{int((coverage >= args.min_coverage).sum()):,} genes in >= '
          f'{args.min_coverage} datasets')

    # genes measured in EVERY dataset -- what `var_merge: intersection` would ship, and
    # the literal reading of "shared genes across all datasets"
    shared = coverage >= len(datasets)
    inter_score = np.where(shared, global_score, -np.inf)
    inter_order = np.argsort(-inter_score)
    print(f'  measured in ALL {len(datasets)} datasets: {int(shared.sum()):,} genes')

    rng = np.random.default_rng(0)
    sets_by_n = {}
    for n in args.sizes:
        per = {ds: np.argsort(-np.where(M[ds], F[ds], -np.inf))[:n] for ds in datasets}
        union = np.unique(np.concatenate(list(per.values())))
        hvg_union = np.unique(np.concatenate([HV[ds][:n] for ds in datasets]))
        sets_by_n[n] = {
            'per_dataset': per,
            'union': union,
            'global': global_order[:n],
            # same ranking, SAME SIZE as the union -- isolates ranking from set size
            'global_matched': global_order[:len(union)],
            'intersection': inter_order[:n],
            'hvg_union': hvg_union,
            'random': rng.choice(n_var, size=len(union), replace=False),
        }
        print(f'  n={n:,}: union {len(union):,}, global_matched {len(union):,}, '
              f'intersection {min(n, int(np.isfinite(inter_score).sum())):,}, '
              f'hvg_union {len(hvg_union):,}')

    # ---------------- pass 2: score ----------------
    print('\n[pass 2] decodability per dataset')
    rows = []
    for ds in datasets:
        P, obs, _, _ = load_one(pb_dir / f'{ds}.h5ad', args.variant)
        if args.holdout:
            keep = (obs['is_ctrl'].to_numpy() |
                    np.isin(obs['label'].to_numpy(), list(SCORE_LABELS[ds])))
            P, obs = P[keep], obs[keep].reset_index(drop=True)
        base = decodability(P, obs, variant=args.variant, datasets=[ds])
        if base.empty:
            del P, obs
            continue
        n_ref = int(base['n_ref'].iloc[0])
        rows.append({'dataset': ds, 'strategy': 'all_genes', 'n': 0,
                     'n_genes': P.shape[1], 'n_ref': n_ref,
                     'top1': float(base['top1'].iloc[0]),
                     'top5': float(base['top5'].iloc[0]),
                     'median_rank': float(base['median_rank'].iloc[0])})
        for n, sets in sets_by_n.items():
            for strat in ('per_dataset', 'union', 'global', 'global_matched',
                          'intersection', 'hvg_union', 'random'):
                cols = sets[strat][ds] if strat == 'per_dataset' else sets[strat]
                cols = np.sort(np.asarray(cols))
                r = decodability(P[:, cols], obs, variant=args.variant, datasets=[ds])
                if r.empty:
                    continue
                rows.append({'dataset': ds, 'strategy': strat, 'n': n,
                             'n_genes': len(cols), 'n_ref': n_ref,
                             'top1': float(r['top1'].iloc[0]),
                             'top5': float(r['top5'].iloc[0]),
                             'median_rank': float(r['median_rank'].iloc[0])})
        print(f'  {ds}: done ({n_ref:,} perturbations)')
        del P, obs

    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    # ---------------- report ----------------
    print('\n' + '=' * 78)
    print('MEAN ACROSS DATASETS  (top1 replicate-matching accuracy)')
    piv = res.pivot_table(index='n', columns='strategy', values='top1', aggfunc='mean')
    order = [c for c in ('per_dataset', 'union', 'global', 'global_matched',
                         'intersection', 'hvg_union', 'random') if c in piv.columns]
    print(piv[order].round(4).to_string())
    print('\nGENE-SET SIZES (means; sets are NOT all the same size)')
    print(res[res['strategy'] != 'all_genes']
          .pivot_table(index='n', columns='strategy', values='n_genes', aggfunc='mean')
          [order].round(0).to_string())
    allg = res[res['strategy'] == 'all_genes']['top1'].mean()
    print(f'\nall {n_var:,} genes: {allg:.4f}')
    print('\nSIZE-MATCHED: union vs the same global ranking at the same size')
    mm = res[res['strategy'].isin(['union', 'global_matched'])].pivot_table(
        index='n', columns='strategy', values='top1')
    if {'union', 'global_matched'} <= set(mm.columns):
        mm['global_minus_union'] = mm['global_matched'] - mm['union']
        print(mm.round(4).to_string())

    print('\nUNION COST  (union minus that dataset\'s own top-n, per dataset)')
    d = res[res['strategy'].isin(['per_dataset', 'union'])]
    w = d.pivot_table(index=['dataset', 'n'], columns='strategy', values='top1')
    w['delta'] = w['union'] - w['per_dataset']
    w['rel'] = w['delta'] / w['per_dataset'].replace(0, np.nan)
    print(w.groupby('n')[['per_dataset', 'union', 'delta', 'rel']].mean()
           .round(4).to_string())
    print(f"\nworst per-dataset union cost: {w['delta'].min():+.4f} "
          f"({w['delta'].idxmin()})")
    print(f"datasets where union HELPS: {int((w['delta'] > 0).sum())}/{len(w)}")

    # ---- Q1: optimal n PER DATASET ----
    print('\n' + '=' * 78)
    print('OPTIMAL n PER DATASET  (own top-n, i.e. what zero-padding the union gives)')
    pd_ = res[res['strategy'] == 'per_dataset']
    best_n = (pd_.loc[pd_.groupby('dataset')['top1'].idxmax()]
              [['dataset', 'n', 'top1', 'n_ref']]
              .rename(columns={'n': 'best_n', 'top1': 'best_top1'}))
    # how flat is the optimum -- the smallest n within 2% of that dataset's best
    rows2 = []
    for ds, g in pd_.groupby('dataset'):
        b = g['top1'].max()
        ok = g[g['top1'] >= 0.98 * b]
        rows2.append({'dataset': ds, 'n_within_2pct': int(ok['n'].min())})
    best_n = best_n.merge(pd.DataFrame(rows2), on='dataset')
    print(best_n.sort_values('best_n').to_string(index=False))
    print(f"\n  median optimal n: {int(best_n['best_n'].median()):,}"
          f"   median n within 2% of best: {int(best_n['n_within_2pct'].median()):,}")

    # ---- Q2: zero-pad vs keep, stated as the decision ----
    print('\n' + '=' * 78)
    print('Q2: at each dataset\'s OWN optimal n -- zero-pad the rest, or keep them?')
    zp = []
    for _, r in best_n.iterrows():
        ds, n = r['dataset'], int(r['best_n'])
        u = res[(res.dataset == ds) & (res.strategy == 'union') & (res.n == n)]
        if u.empty:
            continue
        zp.append({'dataset': ds, 'n': n,
                   'zero_pad (= own top-n)': float(r['best_top1']),
                   'keep (= union)': float(u['top1'].iloc[0]),
                   'delta': float(u['top1'].iloc[0]) - float(r['best_top1'])})
    zp = pd.DataFrame(zp)
    print(zp.round(4).to_string(index=False))
    if len(zp):
        n_keep = int((zp['delta'] > 0).sum())
        print(f"\n  keeping the extra genes HELPS in {n_keep}/{len(zp)} datasets"
              f"  (mean delta {zp['delta'].mean():+.4f})")
        print(f"  VERDICT: {'KEEP the extra genes' if n_keep > len(zp)/2 else 'ZERO-PAD the rest'}")

    # ---------------- the objective: most decodability per gene ----------------
    # Only SHARED-AXIS strategies are shippable -- the training DB has ONE gene axis, so
    # `per_dataset` (a different axis per dataset) is an upper-bound reference, not an
    # option.  The frontier is over everything that can actually be built.
    SHIPPABLE = ('union', 'global', 'global_matched', 'intersection', 'hvg_union')
    print('\n' + '=' * 78)
    print('PARETO FRONTIER  (shippable shared-axis sets only)')
    cand = (res[res['strategy'].isin(SHIPPABLE)]
            .groupby(['strategy', 'n'])
            .agg(n_genes=('n_genes', 'mean'), top1=('top1', 'mean'),
                 top5=('top5', 'mean'))
            .reset_index().sort_values('n_genes'))
    best = cand['top1'].max()
    front, seen = [], -np.inf
    for _, r in cand.iterrows():
        if r['top1'] > seen:            # no smaller set scores at least as well
            front.append(r)
            seen = r['top1']
    fr = pd.DataFrame(front)
    fr['pct_of_best'] = fr['top1'] / best
    print(fr[['strategy', 'n', 'n_genes', 'top1', 'top5', 'pct_of_best']]
          .to_string(index=False,
                     formatters={'n_genes': '{:,.0f}'.format, 'top1': '{:.4f}'.format,
                                 'top5': '{:.4f}'.format,
                                 'pct_of_best': '{:.1%}'.format}))
    print(f'\nbest shippable top1 = {best:.4f}')
    for tol in (0.99, 0.95, 0.90):
        ok = cand[cand['top1'] >= tol * best]
        if len(ok):
            r = ok.loc[ok['n_genes'].idxmin()]
            print(f'  smallest set within {1-tol:.0%} of best: '
                  f"{r['n_genes']:,.0f} genes  ({r['strategy']}, n={r['n']:,}, "
                  f"top1 {r['top1']:.4f})")
    upper = res[res['strategy'] == 'per_dataset'].groupby('n')['top1'].mean().max()
    print(f'\nper-dataset upper bound (NOT shippable, one axis per dataset): {upper:.4f}')

    summary = {
        'sizes': args.sizes,
        'pareto': fr[['strategy', 'n', 'n_genes', 'top1']].to_dict('records'),
        'best_shippable_top1': float(best),
        'per_dataset_upper_bound': float(upper), 'variant': args.variant, 'datasets': datasets,
        'holdout': bool(args.holdout),
        'var_prior_quantile': args.var_prior_quantile,
        'all_genes_top1': float(allg),
        'union_sizes': {int(n): int(len(s['union'])) for n, s in sets_by_n.items()},
        'mean_top1': {c: {int(k): float(v) for k, v in piv[c].items()} for c in order},
        'mean_union_cost': {int(k): float(v)
                            for k, v in w.groupby('n')['delta'].mean().items()},
    }
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nwrote {out} and {out.with_suffix(".summary.json")}')


if __name__ == '__main__':
    main()
