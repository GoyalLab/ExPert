#!/usr/bin/env python
"""Held-out decodability of four per-dataset gene-ranking criteria.

`variances_norm` (seurat_v3) deliberately removes the mean-variance trend; RAW `variances`
keeps it and correlates with the F-statistic at rho=+0.76.  If an unsupervised statistic
decodes as well as the supervised one, the supervised machinery is not earning its keep.
"""
import sys
from pathlib import Path
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--expert-data-repo', required=True,
                help='checkout of GoyalLab/ExPert-data (provides src.decode)')
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--out-dir', default='results/gene_sweep/_criteria')
a = ap.parse_args()
sys.path.insert(0, a.expert_data_repo)
from src.decode import (dataset_rows, decodability, delta_vs_control,
                        load_pseudobulk, normalize)
S = Path(a.store); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ds = a.dataset

t = pd.read_parquet(S / 'fstat' / f'{ds}.parquet', columns=['f_stat', 'measured'])
h = pd.read_parquet(S / 'stats' / f'{ds}.hvg.parquet',
                    columns=['means', 'variances', 'variances_norm'])
meas = t['measured'].to_numpy()
crit = {'f_stat': t['f_stat'].to_numpy().astype(float),
        'variances': h['variances'].to_numpy(),
        'variances_norm': h['variances_norm'].to_numpy(),
        'means': h['means'].to_numpy()}
# OPTIONAL fifth criterion: joint non-linear importance from a tree ensemble trained to tell
# whether two pseudobulk rows are the SAME perturbation (see build_gene_panel_model.py). Added
# here rather than in a parallel scorer so it goes through the IDENTICAL held-out protocol and
# lands in the same table as the four published criteria. Silently skipped if not computed.
_imp = Path('results/gene_sweep/_imp_et') / f'{ds}.npz'
if _imp.exists():
    _z = np.load(_imp, allow_pickle=True)
    _iv = _z['importance'].astype(float)
    if _iv.shape[0] == crit['f_stat'].shape[0]:
        crit['model_imp'] = _iv
    else:
        print(f'  {ds}: model_imp SKIPPED (len {_iv.shape[0]} != {crit["f_stat"].shape[0]})')

X, obs, var = load_pseudobulk(S / 'pseudobulk' / f'{ds}.h5ad', variant='rep')
P = delta_vs_control(normalize(X, obs), obs); del X
sub = dataset_rows(obs, ds, variant='rep')
if sub.size < 4:
    print(f'{ds}: skipped'); sys.exit()
uniq = np.unique(obs['label'].to_numpy()[sub])
rng = np.random.default_rng(0); perm = rng.permutation(uniq.size)
score_labs = set(uniq[perm[uniq.size // 2:]].tolist())
keep = obs['is_ctrl'].to_numpy() | np.isin(obs['label'].to_numpy(), list(score_labs))
Pk, obsk = P[keep], obs[keep].reset_index(drop=True); del P

rows = []
for name, v in crit.items():
    cols = np.sort(np.argsort(-np.where(meas, v, -np.inf))[:a.n])
    r = decodability(Pk[:, cols], obsk, variant='rep', datasets=[ds])
    if not r.empty:
        rows.append({'dataset': ds, 'criterion': name, 'top1': float(r['top1'].iloc[0]),
                     'top5': float(r['top5'].iloc[0]), 'n_ref': int(r['n_ref'].iloc[0])})
        print(f'  {ds} {name}: top1 {rows[-1]["top1"]:.4f}', flush=True)
pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
