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

"""SI summary of the prediction vocabulary: how many perturbations are trained, how many rows the production model
ranks in predict() (trained + zero-shot), and how many zero-shot perturbations can be scored (Fig. 3). Writes
--out (si/tables/vocabulary_summary.json). Every count is read from its source and cross-checked against a second one:

    trained         the production model's label registry (categories of `labels`) == embedding rows - filled zero-shot rows
    vocabulary      rows of the production model's prototype table (what predict() ranks against, trained + zero-shot)
    zero-shot rows  vocabulary - trained; all of them carry a fill (prior_missing_T) in the shipped model
    scored          zero-shot perturbations with >= --min-cells cells in at least one screen of the zero-shot database
                    (the query rule of cache_zs_queries.py, 25-cell cap per query) == the candidates of the Fig. 3
                    zero-shot cache; per fold model, from Table S5 (all unseen perturbations)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


def obs_col(o, k):
    x = o[k]
    if isinstance(x, h5py.Group):                     # categorical: codes + categories
        cats = np.asarray(x['categories']).astype(str)
        return cats[np.asarray(x['codes'])]
    return np.asarray(x).astype(str)


def n_trained_from_registry(attr):
    sr = attr['registry_']['field_registries']['labels']['state_registry']
    if 'n_cats' in sr:
        return int(sr['n_cats'])
    assert 'categorical_mapping' in sr, f'labels registry has neither n_cats nor categorical_mapping: {list(sr)}'
    return len(sr['categorical_mapping'])


def main(a):
    ck = torch.load(a.model, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    vocab = int(sd['cls_emb.embedding.weight'].shape[0])
    fill = sd['prior_missing_T'].float()
    assert fill.shape[0] == vocab, f'prior_missing_T has {fill.shape[0]} rows, prototype table {vocab}'
    filled = int((fill.norm(dim=1) > 0).sum())
    trained = n_trained_from_registry(ck['attr_dict'])
    assert trained + filled == vocab, f'trained {trained} + filled {filled} != vocabulary {vocab}'

    z = np.load(a.zs_cache, allow_pickle=True)
    rows = np.asarray(z['row_gene']).astype(str)
    assert len(rows) == vocab and int(z['n_trained']) == trained, 'zero-shot cache built on a different vocabulary'
    zs_genes = set(rows[trained:])
    cache_scored = len(np.unique(z['truth']))

    with h5py.File(a.zs_adata, 'r') as f:
        df = pd.DataFrame(dict(ds=obs_col(f['obs'], a.dataset_key), pert=obs_col(f['obs'], a.perturbation_key)))
    df = df[df.pert.isin(zs_genes)]
    n = df.groupby(['pert', 'ds']).size()
    measured = set(n.index.get_level_values(0))
    scored = set(n[n >= a.min_cells].index.get_level_values(0))
    assert len(scored) == cache_scored, f'{len(scored)} scorable here vs {cache_scored} in the Fig. 3 zero-shot cache'

    s5 = pd.read_csv(a.table_s5)
    s5 = s5[s5.context == 'all unseen perturbations']
    per_fold = {sp: dict(min=int(g.n_unique.min()), max=int(g.n_unique.max()), median=float(g.n_unique.median()))
                for sp, g in s5.groupby('split')}

    out = dict(
        production_model=a.model,
        trained_perturbations=trained,
        vocabulary_rows=vocab,
        zero_shot_rows=vocab - trained,
        zero_shot_rows_filled=filled,
        zero_shot_scored=dict(
            perturbations=len(scored),
            rule=f'>= {a.min_cells} cells in at least one screen of the zero-shot database (query capped at 25 cells)',
            per_screen_queries=int((n >= a.min_cells).sum()),
            fold_models_table_s5=per_fold),
        zero_shot_not_scored=dict(
            total=vocab - trained - len(scored),
            measured_below_min_cells=len(measured - scored),
            not_measured_in_any_screen=len(zs_genes - measured)),
        sources=dict(zs_adata=a.zs_adata, zs_cache=a.zs_cache, table_s5=a.table_s5))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='production model.pt (ExPert_v0.1.0)')
    ap.add_argument('--zs-cache', required=True, help='Fig. 3 zero-shot query cache (cache_zs_queries.py --min-cells)')
    ap.add_argument('--zs-adata', required=True, help='zero-shot database (database_perturbed_zs.h5ad)')
    ap.add_argument('--min-cells', type=int, required=True, help='query rule of the zero-shot cache')
    ap.add_argument('--dataset-key', default='dataset')
    ap.add_argument('--perturbation-key', default='perturbation')
    ap.add_argument('--table-s5', required=True, help='si/tables/Table_S6.csv')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
