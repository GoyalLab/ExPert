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

"""Supplementary Fig. S11 data (per-cell effect size predicted by the production model's label-aware effect-size head).

Input: the held-out id_val cells scored by analysis/effect_size/effect_stack.py (stack_val_cells.parquet; production
v0.1.0, up to 20,000 perturbed cells per dataset; controls are dropped by their missing Mixscale): measured effect size
(|Mixscale| with --abs-measured), the shipped readout (--pred-col ridge_head_global_main = predict()'s
obs['effect_size']: the trained efficiency head combined with the cosine confidence of the call) and the in-model head
`eff_hat` alone for reference. Measured and predicted are z-scored within each dataset, as in the model's
TensorBoard hexbins, so every panel shares one axis; Pearson r is unchanged by the z-scoring.

    figS11_cells.parquet        dataset, perturbation, eff_z (measured), eff_hat_z (predicted by --pred-col)
    figS11_per_dataset.csv      n, Pearson r, r^2, Spearman rho, R^2 (prediction vs identity) per dataset, for the
                                shipped head and (suffix _eff_hat) the in-model head
    figS11_essential.csv        the four essential-gene screens pooled (per-dataset z-scores), same columns
    figS11_pooled.csv           essential screens, the other 18 (non_essential) and all screens pooled (rows)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

ESSENTIAL = ('ReplogleWeissman2022_K562_essential', 'ReplogleWeissman2022_rpe1',
             'ReplogleWeissman2025_HepG2', 'ReplogleWeissman2025_Jurkat')


def zscore(df, col):
    g = df.groupby('dataset', observed=True)[col]
    return (df[col] - g.transform('mean')) / (g.transform('std', ddof=0) + 1e-12)


def stats(g):
    r = pearsonr(g.eff_hat_z, g.eff_z).statistic
    return pd.Series({'n': len(g), 'r': r, 'r_sq': r ** 2, 'rho': spearmanr(g.eff_hat_z, g.eff_z).statistic,
                      # coefficient of determination of the prediction AS IS (identity line, z-scored axes), as logged
                      # in TensorBoard; = 2r - 1 on z-scored data, so negative whenever r < 0.5
                      'R2': r2_score(g.eff_z, g.eff_hat_z)})


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    c = pd.read_parquet(a.cells, columns=['dataset', 'perturbation', 'efficiency', 'eff_hat', a.pred_col])
    c = c[np.isfinite(c.efficiency) & np.isfinite(c.eff_hat) & np.isfinite(c[a.pred_col])].copy()
    if a.abs_measured:                 # the readout's target: negative Mixscale is a real shift AGAINST the DEG direction
        c['efficiency'] = c.efficiency.abs()
    c['dataset'] = c.dataset.astype(str)
    if a.raw:      # RAW scale: |Mixscale| capped at --cap against the (calibrated) prediction, no per-dataset scaling
        pred = c[a.pred_col].astype(float)
        if a.raw_cal_cells:
            # the shipped readout predicts a within-dataset percentile: map it onto the |Mixscale| scale with ONE
            # linear fit on the training cells held out of every fit (split == 'holdout'), never on these cells
            h = np.load(a.head_npz, allow_pickle=True)
            H = pd.read_parquet(a.raw_cal_cells)
            H = H[(H.split == 'holdout') & ~H.is_ctrl & np.isfinite(H.efficiency)]
            ph = H[list(np.asarray(h['inputs']).astype(str))].to_numpy(np.float64) @ h['w'] + float(h['b'])
            ca, cb = np.polyfit(ph, np.minimum(H.efficiency.abs().to_numpy(), a.cap), 1)
            pred = ca * pred + cb
            print(f'[raw-cal] {len(H):,} held-out training cells: |Mixscale| = {ca:.4f} * effect_size + {cb:.4f}')
        c['eff_z'], c['eff_hat_z'] = np.minimum(c.efficiency.abs(), a.cap), pred
    else:
        c['eff_z'], c['eff_hat_z'] = zscore(c, 'efficiency'), zscore(c, a.pred_col)
    c['old_z'] = zscore(c, 'eff_hat')
    c[['dataset', 'perturbation', 'eff_z', 'eff_hat_z']].reset_index(drop=True).to_parquet(out / f'{a.prefix}_cells.parquet')
    old = c[['dataset', 'eff_z', 'old_z']].rename(columns={'old_z': 'eff_hat_z'})
    T = c.groupby('dataset').apply(stats).join(old.groupby('dataset').apply(stats).add_suffix('_eff_hat')).reset_index()
    T['n'] = T.n.astype(int)
    T.to_csv(out / f'{a.prefix}_per_dataset.csv', index=False)
    ess = c.dataset.isin(ESSENTIAL)
    E = pd.concat([stats(c[ess]), stats(old[ess]).add_suffix('_eff_hat')]).to_frame().T
    E.insert(0, 'datasets', ';'.join(ESSENTIAL)); E['n'] = E.n.astype(int)
    E.to_csv(out / f'{a.prefix}_essential.csv', index=False)
    Pl = pd.DataFrame({'essential': pd.concat([stats(c[ess]), stats(old[ess]).add_suffix('_eff_hat')]),
                       'non_essential': pd.concat([stats(c[~ess]), stats(old[~ess]).add_suffix('_eff_hat')]),
                       'all': pd.concat([stats(c), stats(old).add_suffix('_eff_hat')])}).T
    Pl.index.name = 'cells'
    Pl.to_csv(out / f'{a.prefix}_pooled.csv')
    meta = {'source': str(a.cells), 'model': 'ExPert v0.1.0 (production)', 'split': 'id_val (held-out cells)',
            'n_cells': int(len(c)), 'n_datasets': int(c.dataset.nunique()),
            'mean_r': float(T.r.mean()), 'median_r': float(T.r.median()), 'mean_rho': float(T.rho.mean()),
            'mean_r_sq': float(T.r_sq.mean()), 'mean_R2': float(T.R2.mean()), 'pred_col': a.pred_col,
            'mean_r_eff_hat': float(T.r_eff_hat.mean()), 'mean_rho_eff_hat': float(T.rho_eff_hat.mean()),
            'essential_mean_r': float(T[T.dataset.isin(ESSENTIAL)].r.mean())}
    (out / f'{a.prefix}_meta.json').write_text(json.dumps(meta, indent=2))
    pd.set_option('display.width', 200)
    print(T.round(3).to_string(index=False)); print(E.round(3).to_string(index=False)); print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', required=True, help='effect_head_fit.py val_cells.parquet')
    ap.add_argument('--pred-col', default='effect', help='prediction column (effect = shipped head; eff_hat = in-model)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--abs-measured', action='store_true', help='measured axis = |Mixscale| (the readout target)')
    ap.add_argument('--raw', action='store_true', help='raw scale: min(|Mixscale|, cap) vs the prediction, unscaled')
    ap.add_argument('--cap', type=float, default=10.0)
    ap.add_argument('--raw-cal-cells', help='conf_cells.parquet with split == holdout (raw-scale calibration)')
    ap.add_argument('--head-npz', help='effect_size_head.npz (inputs, w, b) for the raw-scale calibration')
    ap.add_argument('--prefix', default='figS11')
    main(ap.parse_args())
