#!/usr/bin/env python
"""Gene-set overlap across datasets, and whether the decodability gain is concentrated.

Two questions the mean top-1 accuracy cannot answer:

  OVERLAP      Which genes do the per-dataset selections actually share?  A selection
               that is nearly disjoint across datasets means the union is mostly
               dataset-specific and the zero-padded design is doing real work; a
               selection that is nearly identical means one shared axis would do.

  CONCENTRATION  Is the gain spread over all perturbations, or carried by a few?  A mean
               cannot tell "every perturbation got a bit easier" from "50 went from
               hopeless to perfect".  This reports the full per-perturbation rank
               distribution under the old and new gene sets.

One pass: each dataset's pseudobulk is loaded once and both analyses are computed from
it.  Writes tidy CSVs for the plotting script; makes no plots itself.
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
    dataset_rows, delta_vs_control, f_stat_one, load_pseudobulk, normalize,
    perturbation_ranks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n-select', type=int, default=500,
                    help='per-dataset gene count for the NEW selection')
    ap.add_argument('--n-hvg', type=int, default=3000,
                    help='per-dataset HVG count for the CURRENT baseline')
    ap.add_argument('--variant', default='rep')
    ap.add_argument('--datasets', nargs='+', default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    store = Path(args.store)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pb_dir = store / 'pseudobulk'
    datasets = args.datasets or sorted(p.stem for p in pb_dir.glob('*.h5ad'))
    print(f'{len(datasets)} datasets, n_select={args.n_select}, n_hvg={args.n_hvg}',
          flush=True)

    sel, hvg_sel, F, rank_rows, var = {}, {}, {}, [], None
    for ds in datasets:
        X, obs, v = load_pseudobulk(pb_dir / f'{ds}.h5ad')
        if 'variant' in obs.columns:
            k = obs['variant'].to_numpy() == args.variant
            X, obs = X[k], obs[k].reset_index(drop=True)
        var = v
        measured = (X > 0).any(axis=0)
        P = delta_vs_control(normalize(X, obs), obs)
        del X
        sub = dataset_rows(obs, ds, variant=args.variant)
        if sub.size < 4:
            print(f'  {ds}: skipped (too few replicated perturbations)', flush=True)
            del P, obs
            continue

        f = f_stat_one(P, obs, sub, var_prior_quantile=None)
        F[ds] = f
        sel[ds] = np.argsort(-np.where(measured, f, -np.inf))[:args.n_select]

        h = pd.read_parquet(store / 'stats' / f'{ds}.hvg.parquet')
        col = ('highly_variable_rank' if 'highly_variable_rank' in h.columns
               else 'variances_norm')
        hvg_sel[ds] = h.sort_values(
            col, ascending=(col == 'highly_variable_rank')).index.to_numpy()[:args.n_hvg]

        # ---- concentration: per-perturbation ranks under each gene set ----
        for name, cols in (('new', sel[ds]), ('hvg', hvg_sel[ds])):
            r = perturbation_ranks(P[:, np.sort(cols)], obs, ds, variant=args.variant)
            if len(r):
                r['gene_set'] = name
                r['n_genes'] = len(cols)
                rank_rows.append(r)
        print(f'  {ds}: {sub.size:,} groups, {int(measured.sum()):,} measured', flush=True)
        del P, obs

    datasets = [d for d in datasets if d in sel]

    # ---------------- overlap ----------------
    print('\n[overlap] pairwise Jaccard + gene sharing', flush=True)
    n = len(datasets)
    J = np.zeros((n, n))
    for i, a in enumerate(datasets):
        for j, b in enumerate(datasets):
            A, B = set(sel[a].tolist()), set(sel[b].tolist())
            J[i, j] = len(A & B) / max(len(A | B), 1)
    pd.DataFrame(J, index=datasets, columns=datasets).to_csv(out / 'overlap_jaccard.csv')

    # how many datasets selected each gene that anyone selected
    allsel = np.concatenate([sel[d] for d in datasets])
    genes, counts = np.unique(allsel, return_counts=True)
    share = pd.DataFrame({'gene_idx': genes, 'symbol': np.asarray(var)[genes],
                          'n_datasets': counts}).sort_values('n_datasets',
                                                             ascending=False)
    share.to_csv(out / 'gene_sharing.csv', index=False)
    # the same for the HVG baseline, so the two can be compared
    allh = np.concatenate([hvg_sel[d] for d in datasets])
    gh, ch = np.unique(allh, return_counts=True)
    pd.DataFrame({'gene_idx': gh, 'n_datasets': ch}).to_csv(
        out / 'gene_sharing_hvg.csv', index=False)

    union_size = len(genes)
    print(f'  union of per-dataset top-{args.n_select}: {union_size:,} genes')
    print(f'  selected by exactly 1 dataset: {int((counts == 1).sum()):,} '
          f'({(counts == 1).mean():.1%})')
    print(f'  selected by ALL {n}: {int((counts == n).sum()):,}')
    print(f'  median genes per dataset shared with a given other: '
          f'{np.median(J[~np.eye(n, dtype=bool)]):.3f} Jaccard')

    # ---------------- concentration ----------------
    print('\n[concentration] per-perturbation ranks', flush=True)
    ranks = pd.concat(rank_rows, ignore_index=True)
    ranks.to_csv(out / 'perturbation_ranks.csv.gz', index=False, compression='gzip')

    piv = ranks.pivot_table(index=['dataset', 'label', 'held'], columns='gene_set',
                            values='rank')
    piv = piv.dropna()
    piv['improved'] = piv['hvg'] - piv['new']
    tot = len(piv)
    better = int((piv['improved'] > 0).sum())
    worse = int((piv['improved'] < 0).sum())
    print(f'  {tot:,} held-out perturbation replicates')
    print(f'  rank improved: {better:,} ({better/tot:.1%})   '
          f'worsened: {worse:,} ({worse/tot:.1%})   '
          f'unchanged: {tot-better-worse:,}')
    for gs in ('hvg', 'new'):
        print(f"  {gs:4s}: top1 {(piv[gs] == 1).mean():.4f}  "
              f"top10 {(piv[gs] <= 10).mean():.4f}  median rank {piv[gs].median():.0f}")

    # is the top-1 gain carried by a few perturbations, or spread?
    gain = (piv['new'] == 1).astype(int) - (piv['hvg'] == 1).astype(int)
    flipped_in = int((gain > 0).sum())
    flipped_out = int((gain < 0).sum())
    print(f'  became top-1: {flipped_in:,}   stopped being top-1: {flipped_out:,}   '
          f'net {flipped_in - flipped_out:+,}')
    # concentration of the net gain across datasets
    by_ds = piv.reset_index().groupby('dataset').apply(
        lambda g: pd.Series({
            'n': len(g),
            'hvg_top1': (g['hvg'] == 1).mean(),
            'new_top1': (g['new'] == 1).mean(),
            'delta_top1': (g['new'] == 1).mean() - (g['hvg'] == 1).mean()}),
        include_groups=False).sort_values('delta_top1', ascending=False)
    by_ds.to_csv(out / 'concentration_by_dataset.csv')
    print('\n  per-dataset top1 (hvg -> new):')
    print(by_ds.round(4).to_string())

    (out / 'analysis_summary.json').write_text(json.dumps({
        'n_select': args.n_select, 'n_hvg': args.n_hvg, 'datasets': datasets,
        'union_size': int(union_size),
        'frac_selected_by_one': float((counts == 1).mean()),
        'n_selected_by_all': int((counts == n).sum()),
        'median_pairwise_jaccard': float(np.median(J[~np.eye(n, dtype=bool)])),
        'n_replicates': int(tot), 'frac_improved': float(better / tot),
        'frac_worsened': float(worse / tot),
        'hvg_top1': float((piv['hvg'] == 1).mean()),
        'new_top1': float((piv['new'] == 1).mean()),
    }, indent=2))
    print(f'\nwrote {out}/*.csv', flush=True)


if __name__ == '__main__':
    main()
