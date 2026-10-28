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

"""Fig. 2 shared logic, ported from the benchmark pipeline (ExPert-benchmarks) so the figure
does not import from it: method naming, the performance-ranked palette, the distribution-shift
label of every OOD row, matched-cell selection and the measured cost overrides.

Sources (benchmark repo, as of commit 124659d):
    rank_methods / method_palette     workflow/scripts/collect_results.py  _rank_methods / _method_palette
    canon_split_shift                 workflow/scripts/collect_results.py  canon_split_shift
    apply_overrides                   workflow/scripts/analysis/cost.py    apply_overrides
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ExPert's colour: Okabe-Ito orange. Outside the viridis range the other methods use and far from the
# baseline greys, so ExPert reads at a glance without a label; colour-vision-deficiency safe, not red.
EXPERT_COLOR = '#E69F00'

# Paper names. Raw keys stay in every table; only labels change.
DISPLAY = {
    'ExPert': 'ExPert',
    'fingerprint_perdataset': 'Fingerprint (ensemble)',
    'fingerprint_merged': 'Fingerprint (merged)',
    'xgboost': 'XGBoost', 'xgboost_capped': 'XGBoost',
    'linreg': 'Linear regression',
    'cellot': 'CellOT', 'scgpt': 'scGPT', 'gears': 'GEARS', 'scouter': 'Scouter',
    'genepert': 'GenePert',
    'baseline_centroid_notarget': 'Centroid', 'baseline_centroid': 'Centroid (incl. target)',
    'baseline_neg_target': 'Negative target', 'baseline_prior': 'Frequency',
    'baseline_random': 'Random',
    'baselines': 'Baselines (all)',
}
# Scored, never drawn: the unmasked centroid reads the knocked-down target itself (replaced by
# the target-masked one), and the scGPT side arms exist only on a few folds of the small run.
HIDE = {'baseline_centroid', 'baseline_neg_target', 'scgpt_masked', 'scgpt_zf'}   # neg_target: only in the S2 target-mask panels
# the capped twin IS the same method at small (500k cap > small's 491k train cells)
ALIAS = {'xgboost': 'xgboost_capped'}
# cost-table method -> metrics method (fingerprint's per-dataset rules book as 'fingerprint')
COST_TO_METRIC = {'fingerprint': 'fingerprint_perdataset', 'xgboost': 'xgboost_capped'}

# headline metrics: efficiency-weighted, per perturbation where the table allows it
F1, R10, MRR = 'weighted_macro_f1', 'balanced_weighted_recall@10', 'weighted_mrr'
METRIC_LABEL = {F1: 'F1', R10: 'top-10 recall', MRR: 'MRR', 'balanced_weighted_mrr': 'MRR',
                'balanced_weighted_macro_f1': 'F1'}
# the threshold sweep (panel C) filters on efficiency instead of weighting by it. macro-F1 over
# the classes PRESENT in each dataset (collect's 'macro_f1' divides by all C classes, so on a
# per-dataset row every absent class adds a zero and the value shrinks ~C / n_present-fold)
SWEEP = {'macro_f1_no_ds_bias': 'F1', 'balanced_recall@10': 'top-10 recall', 'mrr': 'MRR'}
RADAR_LABEL = {'F1': 'F1', 'MRR': 'MRR', 'AUROC': 'AUROC', 'R1': 'top-1 recall', 'R5': 'top-5 recall',
               'R10': 'top-10 recall', 'R20': 'top-20 recall', 'R50': 'top-50 recall', 'R100': 'top-100 recall'}
SHIFTS = ('ood_cell_line', 'ood_condition', 'ood_donor')
SHIFT_LABEL = {'ood_cell_line': 'unseen cell line', 'ood_condition': 'unseen condition',
               'ood_donor': 'unseen donor'}
# in-distribution views, grouped by the shift type their FOLD held out (same folds as the OOD column)
ID_SHIFTS = ('id_cell_line', 'id_condition', 'id_donor')
SHIFT_LABEL.update({'id_cell_line': 'ID, cell-line folds', 'id_condition': 'ID, condition folds',
                    'id_donor': 'ID, donor folds'})


def disp(m: str) -> str:
    return DISPLAY.get(m, m)


def chance(metric: str, C: int) -> float:
    """Analytic uniform-random floor over C classes."""
    if 'recall@10' in metric:
        return min(10 / C, 1.0)
    if 'mrr' in metric:
        return float(np.sum(1 / np.arange(1, C + 1)) / C)
    return 1 / C                                   # macro-F1 / accuracy


def regime(split: pd.Series) -> pd.Series:
    return split.astype(str).str.split(':').str[0].str.replace('_test', '', regex=False)


def matched(df: pd.DataFrame, key=('fold', 'split')) -> pd.DataFrame:
    """Rows of the (fold, split) cells EVERY method in df has."""
    key = list(key)
    cells = None
    for _, g in df.groupby('method'):
        c = set(map(tuple, g[key].astype(str).to_numpy()))
        cells = c if cells is None else cells & c
    return df[[tuple(r) in cells for r in df[key].astype(str).to_numpy()]]


# ---- palette (collect_results._rank_methods / _method_palette) --------------------------------
def rank_methods(metrics_df: pd.DataFrame, recall_prefix: str = 'balanced_recall@') -> list[str]:
    hb = [c for c in metrics_df.columns
          if c in ('accuracy', 'macro_f1', 'knn_acc') or c.startswith(recall_prefix)]
    total, n = None, 0
    for col in hb:
        avg = metrics_df.groupby('method')[col].mean()
        if avg.dropna().empty:
            continue
        r = avg.rank(ascending=True)
        total = r if total is None else total.add(r, fill_value=0)
        n += 1
    if total is None:
        return sorted(metrics_df['method'].astype(str).unique())
    return list(total.sort_index().sort_values(ascending=False, kind='stable').index)


def method_palette(ranked, sub_range=(0.05, 0.75), grey_range=(0.30, 0.75)) -> dict[str, str]:
    """viridis for methods (best -> lime green), greys for baselines (best -> dark)."""
    import matplotlib as mpl
    ordered = list(dict.fromkeys(map(str, ranked)))
    non_b = [m for m in ordered if 'baseline' not in m.lower()]
    bsl = [m for m in ordered if 'baseline' in m.lower()]
    pal = {}
    for group, cmap, (lo, hi) in ((non_b, mpl.cm.viridis, sub_range), (bsl, mpl.cm.Greys, grey_range)):
        pos = [(lo + hi) / 2] if len(group) == 1 else np.linspace(hi, lo, len(group))
        for m, p in zip(group, pos):
            pal[m] = mpl.colors.to_hex(cmap(p))
    return pal


# ---- distribution-shift label (collect_results.canon_split_shift) ------------------------------
def _marson_parts(name):
    m = re.match(r'^(Marson\d*)_?(D\d+)_(.+)$', str(name))
    return (m.group(2), m.group(3)) if m else None


def _fold_holdouts(df):
    ood = df[df['split'].astype(str).str.startswith('ood')]
    return {f: set(g['split'].astype(str).str.split(':', n=1).str[-1]) for f, g in ood.groupby('fold')}


def _fold_shift_type(held):
    sib = [d for d in held if _marson_parts(d)]
    if sib:
        donors = {_marson_parts(d)[0] for d in sib}
        conds = {_marson_parts(d)[1] for d in sib}
        if len(donors) == 1 and len(conds) >= 2:
            return 'donor'
        if len(conds) == 1 and len(donors) >= 2:
            return 'condition'
    return 'cell_line'


def shift_label(df: pd.DataFrame) -> pd.Series:
    """ood rows -> ood_{cell_line,condition,donor} from the held-out dataset itself;
    id rows -> id_<fold's holdout type>."""
    top = df['split'].astype(str).str.split(':', n=1).str[0]
    ds = df['split'].astype(str).str.split(':', n=1).str[-1]
    held = _fold_holdouts(df)
    ftype = {f: _fold_shift_type(h) for f, h in held.items()}

    def one(t, f, d):
        if str(t).startswith('ood'):
            parts = _marson_parts(d)
            if parts is not None:
                donor, cond = parts
                sib = [x for x in held.get(f, ()) if _marson_parts(x)]
                same_d = [x for x in sib if _marson_parts(x)[0] == donor]
                same_c = [x for x in sib if _marson_parts(x)[1] == cond]
                if len(same_d) >= 2 and len(same_c) <= 1:
                    return 'ood_donor'
                if len(same_c) >= 2 and len(same_d) <= 1:
                    return 'ood_condition'
            return 'ood_cell_line'
        return f"id_{ftype.get(f, 'cell_line')}"
    return pd.Series([one(t, f, d) for t, f, d in zip(top, df['fold'], ds)], index=df.index)


# ---- costs (analysis/cost.py apply_overrides) ---------------------------------------------------
def apply_overrides(costs: pd.DataFrame, path) -> pd.DataFrame:
    """Replace sacct-booked values with measured ones (method -> stage -> column -> {size: v})."""
    import yaml
    ov = yaml.safe_load(open(path).read()) or {}
    out = costs.copy()
    for m, stages in ov.items():
        for st, cols in (stages or {}).items():
            for col, by_size in (cols or {}).items():
                if col not in out.columns or not isinstance(by_size, dict):
                    continue
                for sz, v in by_size.items():
                    sel = (out['method'] == m) & (out['stage'] == st) & (out['size'].astype(str) == str(sz))
                    if sel.any():
                        print(f'  [override] {m}/{st}/{sz} {col}: {out.loc[sel, col].tolist()} -> {v}')
                        out.loc[sel, col] = v
    return out
