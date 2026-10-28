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

"""Fig. 1b data: perturbation x context occupancy + per-perturbation training evidence.

Reads only obs metadata -- never an X matrix:

    store obs   <store>/obs/<dataset>.parquet  every QC-passed cell, incl. perturbations the
                                                shared pool later excluded (-> zero-shot file)
    DB obs      database_perturbed.h5ad /obs   the 19.5M cells the model sees
    train split id_train.txt.gz (0-based)      the production fold the prior gate is built on

Reproduces two rules exactly and ASSERTS them against the recorded numbers:

    pool      ExPert-data src/select.py::build_pool -- present (>= min_cells_per_class cells)
              in >= max(3, int(n_groups * min_dataset_frac)) BROAD dataset groups; broad key
              = publication index + '_' + last token of dataset index, NaN key = no vote.
    evidence  ExPert-model _expert.py atom_nqxnds_qual -- n_c (train cells) x d_c (train
              datasets), ranked to [0,1] with a stable argsort over the label vocabulary.

Writes <out>/fig1c_cells.parquet (long: perturbation x dataset) and
<out>/fig1c_perts.parquet (one row per perturbation).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def broad_key(row) -> str | None:
    d = row.get('dataset index')
    if pd.isna(d) or str(d).strip() == '':
        return None
    return f"{row.get('publication index')}_{str(d).split('_')[-1]}"


def context_of(row) -> dict:
    """dataset -> (axis, context label, donor, condition) from the ood_* datasheet columns."""
    cl, dn, cd = (row.get('ood_cell_line'), row.get('ood_donor'), row.get('ood_condition'))
    if isinstance(dn, str) and dn.strip():
        return {'axis': 'donor', 'context': f'{dn} {cd}', 'donor': dn, 'condition': cd}
    return {'axis': 'cell_line', 'context': str(cl), 'donor': None, 'condition': None}


def read_h5_str(ds) -> np.ndarray:
    return np.asarray(ds.asstr()[:], dtype=object)


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    recipe = json.load(open(a.recipe))
    datasets = recipe['datasets']
    p = recipe['params']
    sheet = pd.read_csv(a.datasheet).set_index('index')

    # ---- store: every QC-passed perturbed cell, per (label, dataset) -------------------
    rows = []
    for ds in datasets:
        o = pd.read_parquet(Path(a.store_obs_dir) / f'{ds}.parquet',
                            columns=['label', 'is_ctrl', 'qc_pass'])
        o = o[o.qc_pass.astype(bool) & ~o.is_ctrl.astype(bool)]
        lab = o['label'].astype(str)
        lab = lab[~lab.isin(['', 'nan', 'None'])]
        vc = lab.value_counts()
        rows.append(pd.DataFrame({'perturbation': vc.index, 'dataset': ds,
                                  'store_cells': vc.to_numpy()}))
        print(f'[store] {ds}: {len(lab):,} cells, {len(vc):,} labels', flush=True)
    cells = pd.concat(rows, ignore_index=True)

    # ---- pool, recomputed ---------------------------------------------------------------
    keys = {ds: broad_key(sheet.loc[ds]) for ds in datasets}
    mcc = int(p['min_cells_per_class'])
    pres = cells[cells.store_cells >= mcc].assign(group=lambda d: d.dataset.map(keys))
    pres = pres[pres.group.notna()]
    n_groups_tot = pres.group.nunique()
    min_groups = max(3, int(n_groups_tot * float(p['min_dataset_frac'])))
    ngrp = pres.groupby('perturbation').group.nunique()
    pool = set(ngrp.index[ngrp >= min_groups])
    print(f'[pool] {len(pool):,} perturbations in >= {min_groups} of {n_groups_tot} groups '
          f'(recipe pool_size {recipe["pool_size"]:,})', flush=True)
    assert len(pool) == recipe['pool_size'], 'pool rule does not reproduce the recipe'

    # ---- DB + production train split ----------------------------------------------------
    with h5py.File(a.db, 'r') as f:
        pert = read_h5_str(f['obs/perturbation'])
        dsc = read_h5_str(f['obs/dataset'])
    pu, pi = np.unique(pert, return_inverse=True)
    du, di = np.unique(dsc, return_inverse=True)
    del pert, dsc
    with gzip.open(a.train_idx, 'rt') as fh:
        tr = np.loadtxt(fh, dtype=np.int64)
    is_tr = np.zeros(len(pi), bool); is_tr[tr] = True
    code = pi.astype(np.int64) * len(du) + di
    db_n = np.bincount(code, minlength=len(pu) * len(du))
    tr_n = np.bincount(code[is_tr], minlength=len(pu) * len(du))
    k = np.nonzero(db_n)[0]
    dbt = pd.DataFrame({'perturbation': pu[k // len(du)], 'dataset': du[k % len(du)],
                        'db_cells': db_n[k], 'train_cells': tr_n[k]})
    print(f'[db] {len(pi):,} cells, {len(pu):,} perturbations, train {is_tr.sum():,}',
          flush=True)
    assert len(pi) == json.load(open(a.db + '.stats.json'))['n_pert']
    assert set(pu) == pool, 'DB perturbations != recomputed pool'

    cells = cells.merge(dbt, on=['perturbation', 'dataset'], how='outer')
    cells[['store_cells', 'db_cells', 'train_cells']] = (
        cells[['store_cells', 'db_cells', 'train_cells']].fillna(0).astype(np.int64))
    ctx = pd.DataFrame([{'dataset': ds, 'group': keys[ds], **context_of(sheet.loc[ds])}
                        for ds in datasets])
    cells = cells.merge(ctx, on='dataset', how='left')

    # ---- per-perturbation table ---------------------------------------------------------
    g = cells.groupby('perturbation')
    ctx_store = cells[cells.store_cells >= mcc].groupby('perturbation').context.nunique()
    ctx_db = cells[cells.db_cells > 0].groupby('perturbation').context.nunique()
    per = pd.DataFrame({
        'store_cells': g.store_cells.sum(),
        'db_cells': g.db_cells.sum(),
        'n_train': g.train_cells.sum(),
        'd_train': (cells.train_cells > 0).groupby(cells.perturbation).sum(),
    })
    per['n_contexts_store'] = ctx_store.reindex(per.index).fillna(0).astype(int)
    per['n_contexts_db'] = ctx_db.reindex(per.index).fillna(0).astype(int)
    per['n_groups'] = ngrp.reindex(per.index).fillna(0).astype(int)
    per['selected'] = per.index.isin(pool)
    per = per[(per.store_cells > 0)]

    # evidence rank over the label vocabulary, as the model builds it (labels sorted)
    sel = per[per.selected].sort_index()
    sc = (sel.n_train.to_numpy(np.float32) * np.maximum(sel.d_train.to_numpy(np.float32), 1))
    o = np.argsort(sc, kind='stable')
    r = np.empty(len(sc), np.float32); r[o] = np.arange(len(sc)); r /= max(len(sc) - 1, 1)
    per.loc[sel.index, 'evidence'] = sc
    per.loc[sel.index, 'rank'] = r
    thr = float(np.quantile(r, float(a.pull_q)))
    per.loc[sel.index, 'w_hard'] = (r <= thr).astype(np.float32)
    per.loc[sel.index, 'w_soft'] = np.clip(1 - r, 0, 1) ** float(a.gamma)
    per.index.name = 'perturbation'
    print(f'[evidence] hard gate pull_q={a.pull_q}: {int((r <= thr).sum()):,} pulled; '
          f'soft gamma={a.gamma}: sum w {float((per.w_soft.fillna(0)).sum()):.1f}', flush=True)

    meta = {'n_datasets': len(datasets), 'n_contexts': int(ctx.context.nunique()),
            'n_cells_db': int(len(pi)), 'n_cells_train': int(is_tr.sum()),
            'n_selected': len(pool), 'n_store_perts': int(len(per)),
            'n_groups': int(n_groups_tot), 'min_groups': int(min_groups),
            'min_cells_per_class': mcc, 'pull_q': float(a.pull_q), 'gamma': float(a.gamma),
            'dataset_order': datasets}
    cells.to_parquet(out / 'fig1c_cells.parquet', index=False)
    per.reset_index().to_parquet(out / 'fig1c_perts.parquet', index=False)
    json.dump(meta, open(out / 'fig1c_meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != 'dataset_order'}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--datasheet', required=True)
    ap.add_argument('--store-obs-dir', required=True)
    ap.add_argument('--db', required=True, help='database_perturbed.h5ad')
    ap.add_argument('--train-idx', required=True, help='production fold id_train.txt.gz')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pull-q', type=float, required=True, help='prior_atom_pull_q (hard gate)')
    ap.add_argument('--gamma', type=float, required=True, help='prior_atom_gamma (soft gate)')
    main(ap.parse_args())
