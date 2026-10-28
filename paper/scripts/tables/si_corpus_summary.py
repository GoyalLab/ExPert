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

"""Condensed at-a-glance summaries of the canonical store and the training database, as JSON.

    store_summary.json        raw cells, the filters that remove cells (each with count and % of raw; they sum
                              to the raw count), retained cells (perturbed / controls), perturbations measured,
                              gene vocabulary, effect-size coverage, and a compact per-dataset table
    training_db_summary.json  datasets, contexts, cells (perturbed / control / train / validation / zero-shot),
                              perturbations, perturbation-context and -dataset pairs, input genes, selection
                              rule, prior-pulled set, and a compact per-dataset table

Reads only existing summaries: the store's summary/ + qc/ JSONs, the canonicalize logs (controls available per
dataset, for the control-cap filter), the DB recipe.json, and the Fig. 1 / Fig. S1 data tables.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd


def pct(x, tot):
    return round(100 * x / tot, 1)


def controls_available(log_dir: Path) -> dict:
    """dataset -> controls available before the cap, from 'control draw N of M available' log lines.
    A dataset logged more than once keeps the largest pool (smaller ones are fixture runs)."""
    out = {}
    for f in glob.glob(str(log_dir / '**' / '*.log'), recursive=True):
        for m in re.finditer(r'(\S+): control draw ([\d,]+) of ([\d,]+) available', open(f, errors='ignore').read()):
            ds, avail = m.group(1), int(m.group(3).replace(',', ''))
            out[ds] = max(out.get(ds, 0), avail)
    return out


def main(a):
    store, db, fd = Path(a.store), Path(a.db), Path(a.fig1_data)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------------- store ----------------
    avail = controls_available(Path(a.canon_logs))
    rows = []
    for f in sorted(glob.glob(str(store / 'summary' / '*.json'))):
        s = json.load(open(f)); q = json.load(open(str(f).replace('/summary/', '/qc/')))
        t = q['thresholds']
        rows.append(dict(dataset=s['dataset'], raw=s['n_obs_raw'], candidates=t['n_input'], retained=s['n_obs_kept'],
                         controls_retained=s['n_ctrl_kept'], controls_drawn=q['n_ctrl_drawn'],
                         controls_available=avail.get(s['dataset'])))
    S = pd.DataFrame(rows).set_index('dataset')
    assert S.controls_available.notna().all(), f'no control-draw log for {S.index[S.controls_available.isna()].tolist()}'
    raw = int(S.raw.sum())
    cap = int((S.controls_available - S.controls_drawn).sum())
    qc = int((S.candidates - S.retained).sum())
    kept = int(S.retained.sum())
    unassigned = raw - int(S.candidates.sum()) - cap        # no guide label; multiplets are 0 on this corpus
    ctrl_kept = int(S.controls_retained.sum())
    perts = pd.read_parquet(fd / 'fig1c_perts.parquet')
    cells = pd.read_parquet(fd / 'fig1c_cells.parquet')
    eff = pd.read_csv(Path(a.figS1_data) / 'figS1e_effect_size_summary.csv')
    vocab = pd.read_parquet(store / 'stats' / f'{S.index[0]}.hvg.parquet', columns=['ensembl_id'])
    store_sum = {
        'store': str(store),
        'datasets': len(S),
        'raw_cells': raw,
        'removed': {
            'unassigned_guide': {'cells': unassigned, 'pct_of_raw': pct(unassigned, raw),
                                 'note': 'no perturbation label (guide calling failed); multiplets: 0'},
            'controls_beyond_10000_cap': {'cells': cap, 'pct_of_raw': pct(cap, raw)},
            'quality_control': {'cells': qc, 'pct_of_raw': pct(qc, raw),
                                'pct_of_qc_candidates': pct(qc, int(S.candidates.sum())),
                                'note': 'MAD outliers (depth, genes, top-50 share) + mitochondrial fraction'},
        },
        'retained_cells': {'total': kept, 'pct_of_raw': pct(kept, raw), 'perturbed': kept - ctrl_kept,
                           'controls': ctrl_kept},
        'perturbations': {'with_any_cells': int((perts.store_cells > 0).sum()),
                          'with_at_least_10_cells_in_a_context': int(cells[cells.store_cells >= 10]
                                                                     .perturbation.nunique())},
        'gene_vocabulary_ensembl': len(vocab),
        'effect_size': {'perturbed_cells_scored': int(eff.n_scored.sum()),
                        'pct_scored': pct(int(eff.n_scored.sum()), int(eff.n_perturbed.sum())),
                        'note': 'Mixscale z; perturbations with < 5 DE genes are not scored'},
        'per_dataset': {ds: {'raw': int(r.raw), 'retained': int(r.retained), 'pct_retained': pct(r.retained, r.raw),
                             'controls_retained': int(r.controls_retained)} for ds, r in S.iterrows()},
    }
    assert unassigned + cap + qc + kept == raw
    json.dump(store_sum, open(out / 'store_summary.json', 'w'), indent=1)

    # ---------------- training DB ----------------
    rec = json.load(open(db / 'recipe.json'))
    meta = json.load(open(fd / 'fig1c_meta.json'))
    ds_ctx = cells.drop_duplicates('dataset').set_index('dataset')
    ctx = ds_ctx.apply(lambda r: f'CD4+ T, {r.condition}' if r.axis == 'donor' else r.context, axis=1)
    cells['context'] = cells.dataset.map(ctx)
    indb = cells[cells.db_cells > 0]
    sel = perts[perts.selected]
    import h5py
    with h5py.File(db / 'database_ctrl.h5ad', 'r') as h:
        nctrl = pd.Series(h['obs']['dataset'].asstr()[:]).value_counts()
    pd_ds = indb.groupby('dataset').agg(cells=('db_cells', 'sum'), perturbations=('perturbation', 'nunique'),
                                        train_cells=('train_cells', 'sum'))
    p = rec['params']
    db_sum = {
        'database': str(db),
        'datasets': len(rec['datasets']),
        'contexts': int(cells.context.nunique()),
        'contexts_list': sorted(cells.context.unique().tolist()),
        'cells': {'perturbed': meta['n_cells_db'], 'control': int(nctrl.sum()),
                  'total': meta['n_cells_db'] + int(nctrl.sum()),
                  'train_production_split': meta['n_cells_train'],
                  'validation_production_split': meta['n_cells_db'] - meta['n_cells_train'],
                  'zero_shot_eval_file': rec['n_cells_zero_shot']},
        'perturbations': {'selected': int(len(sel)),
                          'perturbation_context_pairs': int(indb[['perturbation', 'context']].drop_duplicates().shape[0]),
                          'perturbation_dataset_pairs': int(indb[['perturbation', 'dataset']].drop_duplicates().shape[0]),
                          'prior_pulled': int((sel.w_hard.fillna(0) > 0).sum()),
                          'prior_pulled_pct': pct(int((sel.w_hard.fillna(0) > 0).sum()), len(sel))},
        'selection_rule': {'min_cells_per_context': p['min_cells_per_class'],
                           'min_fraction_of_voting_contexts': p['min_dataset_frac'],
                           'voting_contexts': meta['n_groups'],
                           'min_cells_after_pooling': p['min_cells_per_perturbation']},
        'input_genes': {'n': rec['n_genes'], 'selection': p['gene_selection'],
                        'per_dataset': p['n_genes_per_dataset'], 'combination': 'union'},
        'per_dataset': {ds: {'context': ctx[ds], 'cells': int(r.cells), 'controls': int(nctrl.get(ds, 0)),
                             'perturbations': int(r.perturbations), 'train_cells': int(r.train_cells)}
                        for ds, r in pd_ds.loc[rec['datasets']].iterrows()},
    }
    json.dump(db_sum, open(out / 'training_db_summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in store_sum.items() if k != 'per_dataset'}, indent=1))
    print(json.dumps({k: v for k, v in db_sum.items() if k not in ('per_dataset', 'contexts_list')}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True, help='canonical store (store/HASH)')
    ap.add_argument('--db', required=True, help='materialised DB dir (store_out/HASH)')
    ap.add_argument('--canon-logs', required=True, help='ExPert-data canonicalize logs (logs/phase2)')
    ap.add_argument('--fig1-data', required=True, help='Fig. 1 data dir (fig1c_*)')
    ap.add_argument('--figS1-data', required=True, help='Fig. S1 data dir (figS1e_effect_size_summary.csv)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
