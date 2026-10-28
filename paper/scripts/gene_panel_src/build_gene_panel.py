#!/usr/bin/env python
"""Emit the actual gene panel: union of each dataset's top-n separability genes.

This is the artifact the DB would be built on.  Two files, because the zero-padded design
needs both:

    panel.tsv        the UNION -- one row per gene, in vocabulary order.  This is the
                     DB's gene axis (its column order).
    membership.npz   (n_genes x n_datasets) boolean -- which datasets selected each gene.
                     Under zero-padding a dataset's rows carry values ONLY for its own
                     selected genes, so this is what materialize needs to mask with, and
                     what `varm/feature_mask_per_dataset` has to reflect.  Without it the
                     zeros are indistinguishable from "gene not measured".

Two stages, because the per-dataset F-stat is embarrassingly parallel:

    --dataset DS   compute one dataset's F-stat over the FULL vocabulary and cache it to
                   _fstat/{ds}.npz (~300 KB).  Run as a slurm array, one task per dataset.
    --combine      read the cached vectors and emit the panel for a given --n.

Caching the whole F-stat vector rather than the top-n means a different panel size is a
combine away, with no recomputation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decode import (  # noqa: E402
    dataset_rows, delta_vs_control, f_stat_one, load_pseudobulk, normalize)
from src.vocab import load_vocabulary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--n', type=int, default=500, help='genes per dataset')
    ap.add_argument('--variant', default='rep')
    ap.add_argument('--vocab', default='resources/vocab/genes_v1.tsv')
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--dataset', default=None,
                    help='compute and cache ONE dataset\'s F-stat, then exit')
    ap.add_argument('--combine', action='store_true',
                    help='build the panel from the cached per-dataset F-stats')
    args = ap.parse_args()

    store = Path(args.store)
    out = Path(args.out_dir)
    cache = out / '_fstat'
    cache.mkdir(parents=True, exist_ok=True)
    pb = store / 'pseudobulk'

    # ---------------- stage 1: one dataset ----------------
    if args.dataset:
        ds = args.dataset
        # variant filtered at READ time: the file holds all three, and loading them
        # all to keep one costs 3x the peak memory
        X, obs, v = load_pseudobulk(pb / f'{ds}.h5ad', variant=args.variant)
        measured = (X > 0).any(axis=0)
        P = delta_vs_control(normalize(X, obs), obs)
        del X
        sub = dataset_rows(obs, ds, variant=args.variant)
        if sub.size < 4:
            print(f'{ds}: SKIPPED (too few replicated perturbations)', flush=True)
            return
        f = f_stat_one(P, obs, sub, var_prior_quantile=None)
        np.savez_compressed(cache / f'{ds}.npz', f_stat=f.astype(np.float32),
                            measured=measured, var=np.asarray(v).astype(str),
                            n_groups=np.array([sub.size]))
        print(f'{ds}: {int(measured.sum()):,} measured, {sub.size:,} groups '
              f'-> {cache/f"{ds}.npz"}', flush=True)
        return

    if not args.combine:
        raise SystemExit('pass --dataset <name> (stage 1) or --combine (stage 2)')

    # ---------------- stage 2: combine ----------------
    files = sorted(cache.glob('*.npz'))
    if args.datasets:
        files = [p for p in files if p.stem in set(args.datasets)]
    if not files:
        raise SystemExit(f'no cached F-stats in {cache}; run stage 1 first')
    datasets, sel, F, var = [], {}, {}, None
    for p_ in files:
        z = np.load(p_, allow_pickle=False)
        ds = p_.stem
        var = z['var'] if var is None else var
        f, measured = z['f_stat'].astype(np.float64), z['measured']
        F[ds] = f
        sel[ds] = np.argsort(-np.where(measured, f, -np.inf))[:args.n]
        datasets.append(ds)
    print(f'{len(datasets)} cached datasets, n={args.n} genes each', flush=True)

    union = np.unique(np.concatenate([sel[d] for d in datasets]))
    member = np.zeros((len(union), len(datasets)), dtype=bool)
    pos = {int(g): i for i, g in enumerate(union)}
    for j, ds in enumerate(datasets):
        member[[pos[int(g)] for g in sel[ds]], j] = True

    vocab = load_vocabulary(args.vocab)
    ens = np.asarray(var).astype(str)[union]
    sym = vocab['symbol'].to_numpy().astype(str)[union]
    # mean F across the datasets that MEASURED the gene, for ranking/inspection
    meanF = np.array([
        np.mean([F[d][g] for d in datasets if F[d][g] > 0]) if
        any(F[d][g] > 0 for d in datasets) else 0.0 for g in union])

    panel = pd.DataFrame({
        'vocab_idx': union, 'ensembl_id': ens, 'symbol': sym,
        'n_datasets': member.sum(1), 'mean_f_stat': meanF,
    })
    panel.to_csv(out / f'panel_n{args.n}.tsv', sep='\t', index=False)
    np.savez_compressed(out / f'panel_n{args.n}_membership.npz',
                        vocab_idx=union, datasets=np.array(datasets, dtype=object),
                        membership=member)

    n_dup = int(pd.Series(sym).duplicated().sum())
    print(f'\nPANEL: {len(union):,} genes '
          f'(union of {len(datasets)} x top-{args.n:,})')
    print(f'  sum of per-dataset selections : {args.n*len(datasets):,}')
    print(f'  compression from overlap      : {1 - len(union)/(args.n*len(datasets)):.1%}')
    print(f'  selected by exactly 1 dataset : {int((member.sum(1)==1).sum()):,} '
          f'({(member.sum(1)==1).mean():.1%})')
    print(f'  selected by ALL {len(datasets)}          : {int((member.sum(1)==len(datasets)).sum()):,}')
    print(f'  median datasets per gene      : {int(np.median(member.sum(1)))}')
    print(f'  matrix density (zero-pad)     : {member.mean():.1%} of cells carry values')
    print(f'  Ensembl IDs sharing a symbol  : {n_dup} (dedup happens at materialize)')
    print('\n  top 25 genes by cross-dataset breadth:')
    print(panel.sort_values(['n_datasets', 'mean_f_stat'], ascending=False)
          .head(25)[['symbol', 'ensembl_id', 'n_datasets', 'mean_f_stat']]
          .to_string(index=False, formatters={'mean_f_stat': '{:.1f}'.format}))

    (out / f'panel_n{args.n}_summary.json').write_text(json.dumps({
        'n_per_dataset': args.n, 'datasets': datasets, 'panel_size': int(len(union)),
        'frac_single_dataset': float((member.sum(1) == 1).mean()),
        'n_all_datasets': int((member.sum(1) == len(datasets)).sum()),
        'membership_density': float(member.mean()),
        'n_symbol_collisions': n_dup,
    }, indent=2))
    print(f'\nwrote {out}/panel_n{args.n}.tsv and _membership.npz', flush=True)


if __name__ == '__main__':
    main()
