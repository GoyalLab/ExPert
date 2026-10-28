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

"""Table S5: the two benchmark training databases (200 perturbations; ctx7, perturbations measured in >= 7 cell
contexts) and their 12 held-out-context folds (related to Fig. 2). Every count is read from the databases' obs (h5py)
and the folds' own index files (0-based rows into each database); the metadata's per-fold counts are asserted against
them. Written to --out-dir (si/tables) as Table_S5.xlsx plus one CSV per part.

    S5A  databases: perturbations, cells, datasets, contexts, selection rule, caps
    S5B  per dataset x database: context, cells, perturbations
    S5C  per database x fold: held-out context(s) and datasets; cells and perturbations in id_train / id_val /
         id_test / ood_test
    S5D  per perturbation: in each database; contexts and datasets measuring it in the larger one
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SPLITS = [('id_train', 'n_train'), ('id_val', 'n_val'), ('id_test', 'n_test'), ('ood_test', 'n_ood')]
SHEET = {'S5A': 'S5A databases', 'S5B': 'S5B datasets', 'S5C': 'S5C folds', 'S5D': 'S5D perturbations'}


def obs_col(o, k):
    x = o[k]
    if isinstance(x, h5py.Group):
        return np.asarray(x['categories']).astype(str)[np.asarray(x['codes'])]
    return np.asarray(x).astype(str)


def read_idx(p):
    with gzip.open(p, 'rt') as f:
        return np.array([int(l) for l in f if l.strip()], dtype=np.int64)


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ctxmap = json.loads(Path(a.ctx7_subset_json).read_text())['dataset_to_context']
    dbs = {}
    for spec in a.db:
        name, h5, split_dir, rule = spec.split('=', 3)
        with h5py.File(h5, 'r') as f:
            ob = pd.DataFrame({'dataset': obs_col(f['obs'], 'dataset'),
                               'perturbation': obs_col(f['obs'], 'perturbation')})
        ob['context'] = ob.dataset.map(ctxmap)
        assert ob.context.notna().all(), f'{name}: datasets without a context: {set(ob.dataset[ob.context.isna()])}'
        dbs[name] = (ob, Path(split_dir), rule)

    rows_a, rows_b, rows_c = [], [], []
    for name, (ob, sd, rule) in dbs.items():
        meta = json.loads((sd / 'metadata.json').read_text())
        assert meta['source_shape'][0] == len(ob), f'{name}: metadata rows {meta["source_shape"][0]} != obs {len(ob)}'
        rows_a.append(dict(database=name, perturbations=ob.perturbation.nunique(), cells=len(ob),
                           datasets=ob.dataset.nunique(), contexts=ob.context.nunique(), selection=rule,
                           folds=meta['n_folds'], seed=meta['seed'], id_test_frac=meta['test_frac'],
                           id_eval_cap=meta['eval_cap'], ood_cap=meta.get('ood_cap', 'uncapped')))
        g = ob.groupby('dataset').agg(context=('context', 'first'), cells=('perturbation', 'size'),
                                      perturbations=('perturbation', 'nunique')).reset_index()
        rows_b += [dict(database=name, **r) for r in g.to_dict('records')]
        pert = ob.perturbation.to_numpy()
        for fm in meta['folds']:
            k = fm['fold']
            r = dict(database=name, fold=k, held_out=' + '.join(map(str, fm['held_out'].values())),
                     held_out_axis=' + '.join(fm['held_out'].keys()),
                     held_out_datasets='; '.join(fm['held_out_datasets']))
            for s, mk in SPLITS:
                ix = read_idx(sd / f'fold_{k}' / f'{s}.txt.gz')
                r[f'{s}_cells'] = len(ix)
                r[f'{s}_perturbations'] = int(pd.unique(pert[ix]).size)
                if mk in fm:
                    assert fm[mk] == len(ix), f'{name} fold {k} {s}: metadata {fm[mk]} vs index file {len(ix)}'
            rows_c.append(r)

    big = max(dbs, key=lambda n: len(dbs[n][0]))
    ob = dbs[big][0]
    nctx, nds = ob.groupby('perturbation').context.nunique(), ob.groupby('perturbation').dataset.nunique()
    allp = sorted(set().union(*[set(o.perturbation) for o, _, _ in dbs.values()]))
    s9d = pd.DataFrame({'perturbation': allp,
                        **{f'in_{n}': np.isin(allp, dbs[n][0].perturbation.unique()) for n in dbs},
                        f'contexts_in_{big}': [int(nctx.get(p, 0)) for p in allp],
                        f'datasets_in_{big}': [int(nds.get(p, 0)) for p in allp]})
    parts = {'S5A': pd.DataFrame(rows_a), 'S5B': pd.DataFrame(rows_b), 'S5C': pd.DataFrame(rows_c), 'S5D': s9d}
    with pd.ExcelWriter(out / 'Table_S5.xlsx') as w:
        for k, df in parts.items():
            df.to_excel(w, sheet_name=SHEET[k], index=False)
            df.to_csv(out / f'Table_{k}.csv', index=False)
    pd.set_option('display.width', 250)
    print(parts['S5A'].to_string(index=False))
    print(parts['S5B'].to_string(index=False))
    print(parts['S5C'].drop(columns=['held_out_datasets']).to_string(index=False))
    inc = s9d[[c for c in s9d if c.startswith('in_')]]
    print('perturbations in every database:', int(inc.all(axis=1).sum()), '| per database:', inc.sum().to_dict())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', nargs='+', required=True, help='name=h5ad=split_dir=selection rule (one per database)')
    ap.add_argument('--ctx7-subset-json', required=True, help='database_perturbed_ctx7.h5ad.subset.json (dataset -> context)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
