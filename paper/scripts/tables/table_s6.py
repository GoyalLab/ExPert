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

"""Table S6: zero-shot recovery per context, split and fold model -- the values behind Fig. 3B, from the per-fold
records Fig. 3 plots (fig3a_fold<k>.json: shipped fill of each fold model, every unseen perturbation ranked against
the full 19,895-row vocabulary; fig3a_seen_fold<k>.json: the trained perturbations of the same split, for reference).
Split: OOD where the fold model held the context out entirely, else ID (as Fig. 3B). Checked against Fig. 3B's own
source table. Written to --out-dir (si/tables) as Table_S6.xlsx plus Table_S6.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOLS = {'pooled_seen_dataset': 'ID', 'pooled_unseen_dataset': 'OOD'}
UNSEEN, SEEN = 'all unseen perturbations', 'all seen perturbations'
COLS = {'w_zs_x': 'top10_weighted_x_chance', 'w_top10': 'top10_weighted', 'zs_x': 'top10_x_chance',
        'top10': 'top10', 'hits10': 'hits_top10'}


def unseen_rows(z: dict, ctx: pd.DataFrame, fold: int) -> list[dict]:
    held = set(z['unseen'])
    held_full = {lab for lab, g in ctx.groupby('label').dataset if set(g) <= held}
    rows = []
    for g, v in z['real'].items():
        if g in POOLS:
            name, split = UNSEEN, POOLS[g]
        elif g.startswith('ctx:'):
            name = g[4:]; split = 'OOD' if name in held_full else 'ID'
        else:
            continue
        sh = np.array([r[g]['w_zs_x'] for r in z['shuffled']])
        rows.append({'fold': fold, 'context': name, 'split': split, 'perturbations': 'unseen (zero-shot)',
                     'n_unique': int(v['n_cand']), 'n_query': int(v['n_query']),
                     **{COLS[k]: v[k] for k in COLS},
                     'shuffled_null_x_chance_mean': float(sh.mean()), 'shuffled_null_x_chance_sd': float(sh.std(ddof=1)),
                     'n_shuffles': len(sh)})
    return rows


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fd = Path(a.folds_dir)
    ctx = pd.read_csv(a.contexts, sep='\t')
    rows = []
    for f in sorted(fd.glob('fig3a_fold*.json'), key=lambda p: int(p.stem.split('fold')[-1])):
        rows += unseen_rows(json.load(open(f)), ctx, int(f.stem.split('fold')[-1]))
    for f in sorted(fd.glob('fig3a_seen_fold*.json'), key=lambda p: int(p.stem.split('fold')[-1])):
        r = json.load(open(f)); fold = int(f.stem.split('fold')[-1])
        for split in ('ID', 'OOD'):
            v = r[split]; sh = np.asarray(v.get('shuffled', [np.nan]), float)
            rows.append({'fold': fold, 'context': SEEN, 'split': split, 'perturbations': 'trained (reference)',
                         'n_unique': int(v['n_unique']), 'n_query': int(v['n_query']),
                         'top10_weighted_x_chance': v['w_zs_x'], 'top10': v.get('top10', np.nan),
                         'shuffled_null_x_chance_mean': v['random_mean'],
                         'shuffled_null_x_chance_sd': float(np.nanstd(sh, ddof=1)) if len(sh) > 1 else np.nan,
                         'n_shuffles': int(np.isfinite(sh).sum())})
    t = pd.DataFrame(rows)
    order = {UNSEEN: 0, SEEN: 1}
    t = t.assign(_o=t.context.map(order).fillna(2)).sort_values(['_o', 'context', 'split', 'fold']) \
        .drop(columns='_o').reset_index(drop=True)

    # same values as Fig. 3B's source table
    src = pd.read_csv(a.check).rename(columns={'w_zs_x': 'top10_weighted_x_chance'})
    m = src.merge(t, on=['context', 'split', 'fold'], suffixes=('_src', ''))
    assert len(m) == len(src), f'{len(src) - len(m)} Fig. 3B rows not in the table'
    for c in ('top10_weighted_x_chance',):
        assert np.allclose(m[f'{c}_src'], m[c]), c
    assert np.allclose(m.random_mean, m.shuffled_null_x_chance_mean)
    med = t[t.context == UNSEEN].groupby('split').top10_weighted_x_chance.median()
    print(f'[S6] {len(t)} rows ({t.fold.nunique()} fold models); matches Fig. 3B source ({len(src)} rows); '
          f'median over fold models, all unseen: ID {med.ID:.2f}x, OOD {med.OOD:.2f}x', flush=True)

    with pd.ExcelWriter(out / 'Table_S6.xlsx') as w:
        t.to_excel(w, sheet_name='S6 zero-shot per screen', index=False)
    t.to_csv(out / 'Table_S6.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds-dir', required=True, help='Fig. 3 data folds/ (fig3a_fold<k>.json, fig3a_seen_fold<k>.json)')
    ap.add_argument('--contexts', required=True, help='Fig. 3 data fig3a_contexts.tsv (dataset -> context)')
    ap.add_argument('--check', required=True, help="Fig. 3 data fig3a_source.csv (Fig. 3B's own table)")
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
