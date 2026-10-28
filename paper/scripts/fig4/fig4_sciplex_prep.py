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

"""Fig. 5: sci-Plex3 -> a query in the Tahoe schema, so fig4_tahoe_encode / _score run unchanged.

No retraining: the model has never seen a drug label, and sci-Plex is read exactly like Tahoe --
drug response -> ranking over CRISPR prototypes -> is the drug's target gene near the top.

    rows     --time hours only; at most --max-per-group cells per (drug, dose, line), all vehicle
             controls kept (they set the encoder's input centre and the delta baseline)
    obs      drug_clean, dose_uM (nM / 1000), cell_name, plate, is_control, targets (curated
             genes, fig4_sciplex_targets.py), trained_targets / zeroshot_targets split on the bank's
             trained block, target_class in trained / mixed / zeroshot -- or 'nonlof' for agonists,
             activators and drugs without a human gene target, which the scorer reports apart
    X        raw integer counts as float32, gene symbols as var_names (predict() aligns by name)
"""
from __future__ import annotations

import argparse

from pathlib import Path
import numpy as np
import pandas as pd


def main(a):
    import anndata as ad
    rng = np.random.default_rng(a.seed)
    tg = pd.read_csv(a.targets_tsv, sep='\t', keep_default_na=False)
    tg['key'] = tg['drug'].str.strip()
    tg = tg.set_index('key')
    bank = np.load(a.bank, allow_pickle=True)
    trained = set(bank['labels'].astype(str)[:int(bank['n_labels'])])
    vocab = set(bank['labels'].astype(str))

    A = ad.read_h5ad(a.h5ad, backed='r')
    obs = A.obs[['perturbation', 'cell_line', 'plate', 'dose_value', 'time']].copy()
    obs['drug_clean'] = obs['perturbation'].astype(str).str.strip()
    ctrl = (obs['drug_clean'] == 'control').to_numpy()
    tm = (obs['time'].astype(float) == a.time).to_numpy()
    print(f'[obs] {len(obs):,} cells; {int(tm.sum()):,} at {a.time} h; '
          f'{int((ctrl & tm).sum()):,} controls at {a.time} h', flush=True)
    keep = []
    g = obs[tm & ~ctrl].groupby(['drug_clean', 'dose_value', 'cell_line'], observed=True).indices
    if a.keep_barcodes:
        # EXACTLY the cells another method saw (e.g. the per-line fingerprinting exports: <= 200 per
        # condition + every vehicle cell), instead of this script's own <= max_per_group draw
        bc = set()
        for f in a.keep_barcodes.split(','):
            bc |= set(Path(f).read_text().split())
        keep = np.flatnonzero(tm & obs.index.astype(str).isin(bc))
        print(f'[subset] --keep-barcodes: {len(keep):,} of {len(bc):,} listed cells found', flush=True)
    else:
        pos = np.flatnonzero(tm & ~ctrl)
        for k, ix in g.items():
            ix = pos[ix]
            keep.append(ix if len(ix) <= a.max_per_group else rng.choice(ix, a.max_per_group, replace=False))
        keep.append(np.flatnonzero(tm & ctrl))
        keep = np.sort(np.concatenate(keep))
    print(f'[subset] {len(keep):,} cells from {len(g)} (drug, dose, line) groups', flush=True)

    q = A[keep].to_memory()
    A.file.close()
    q.X = q.X.astype(np.float32).tocsr()
    o = q.obs
    d = o['perturbation'].astype(str).str.strip()
    is_c = (d == 'control').to_numpy()
    missing = sorted(set(d[~is_c]) - set(tg.index))
    if missing:
        raise SystemExit(f'drugs without a target row: {missing[:10]}')

    def per_drug(name):
        if name == 'control':
            return dict(targets='', trained_targets='', zeroshot_targets='', target_class='control',
                        single_target=0)
        r = tg.loc[name]
        genes = [x for x in r['genes'].split(';') if x]
        tr = [x for x in genes if x in trained]
        zs = [x for x in genes if x in vocab and x not in trained]
        if int(r['lof']) == 0 or not genes:
            cls = 'nonlof'
        else:
            cls = 'trained' if tr and not zs else 'zeroshot' if zs and not tr else 'mixed' if tr else 'none'
        return dict(targets=';'.join(genes), trained_targets=';'.join(tr),
                    zeroshot_targets=';'.join(zs), target_class=cls, single_target=int(len(genes) == 1))

    meta = pd.DataFrame({n: per_drug(n) for n in d.unique()}).T
    out = pd.DataFrame(index=o.index)
    out['drug_clean'] = d.to_numpy()
    out['dose_uM'] = o['dose_value'].astype(float).to_numpy() / 1000.0
    out['cell_name'] = o['cell_line'].astype(str).to_numpy()
    # --pool-plates: one label, so the encoder pseudobulks each line x drug x dose across its replicate
    # plates -- the grouping fingerprinting uses -- instead of one pseudobulk per plate
    out['plate'] = 'pooled' if a.pool_plates else o['plate'].astype(str).to_numpy()
    out['is_control'] = is_c
    for c in meta.columns:
        out[c] = meta.loc[d, c].to_numpy()
    out['single_target'] = out['single_target'].astype(int)
    q.obs = out
    q.obsm.clear(); q.layers.clear(); q.uns.clear()
    q.var = pd.DataFrame(index=q.var_names.astype(str))
    q.write_h5ad(a.out, compression='gzip')
    dm = meta[meta.target_class != 'control']
    print(f'[write] {q.n_obs:,} x {q.n_vars:,} -> {a.out}\n'
          f'[targets] drugs by class: {dm.target_class.value_counts().to_dict()}; '
          f'genes in bank {sum(x in vocab for t in dm.targets for x in t.split(";") if x)}/'
          f'{sum(1 for t in dm.targets for x in t.split(";") if x)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--targets-tsv', required=True)
    ap.add_argument('--bank', required=True, help='bank.npz (fig4_tahoe_encode.py --harvest-bank)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--time', type=float, default=24.0)
    ap.add_argument('--max-per-group', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-barcodes', default='', help='barcode files (comma separated): keep exactly these cells')
    ap.add_argument('--pool-plates', action='store_true', help='one plate label: pseudobulk across replicate plates')
    main(ap.parse_args())
