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

"""Input x latent centring under the CURRENT predict() on ALD (v0.1.0, mito_mt), scored against known biology.

stage encode  (one task per INPUT arm; one predict() call each; the scored rows are always one pseudobulk per donor x
              cell type, n_cells_key set)
    ct_normal_pools   center_by cell type, baseline = the Normal donors' 10 random pools (the Fig. 5 recipe)
    ct_normal_cells   center_by cell type, baseline = the Normal NUCLEI (mean log1p-CPM over single nuclei, as in
                      training where controls are single cells)
    pooled_normal     no center_by: ONE baseline from every Normal pool across cell types
    ct_selfmean       center_by cell type, baseline = every donor's pools of that cell type (query mean, no control)
    donorct_self      center_by donor x cell type, baseline = the group's OWN pools (negative control: removes the
                      group's signal by construction)
    trained           refresh_ctrl_means=False: the trained baseline of the placeholder dataset (no query control)
    Latent centring is a per-dataset shift and every ALD row shares one placeholder dataset, so it cancels in any
    contrast; it is applied after encoding (exact, linear) and only matters for the absolute read:
    none (z as encoded, center_mode=0) | trained (minus align_ds_mean of the placeholder) | query_mean | normal_mean
    (mean of the Normal groups) | ct_mean (each cell type's own mean) | ct_normal (each cell type's Normal mean).
stage score   (one task per input arm x bank) contrasts abs (per latent option) / ctrl / rest / rrep x cosine | deflate
              x setz | ora, the readout grid's targets and null (fig5_readout_grid.py).
stage summarize  scorecard.

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_centring_grid.py --stage encode --task 0 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig5_centring_audit import unit, pct_of, set_z
from fig5_readout_grid import CT, TARGETS, contrasts, deflated
from fig5_variants import MASKS, mask_query

ARMS = ['ct_normal_pools', 'ct_normal_cells', 'pooled_normal', 'ct_selfmean', 'donorct_self', 'trained']
LATENT = ['none', 'trained', 'query_mean', 'normal_mean', 'ct_mean', 'ct_normal']


def stage_encode(a):
    import anndata as ad
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    arm = ARMS[a.task]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    q, _ = mask_query(q, MASKS['mito_mt'])
    ct = q.obs['CellTypeSR'].astype(str).to_numpy()
    cond = q.obs['Condition'].astype(str).to_numpy()
    donor = q.obs['sample'].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    X, meta = [], []
    for t in CT:
        for d in sorted(set(donor[ct == t])):
            idx = np.flatnonzero((ct == t) & (donor == d))
            if len(idx) < 50:
                continue
            c = cond[idx[0]]
            X.append(np.asarray(q.X[idx].mean(0)).ravel())
            meta.append(dict(role='group', donor=d, condition=c, celltype=t, n_cells=len(idx)))
            for chunk in np.array_split(rng.permutation(idx), 10):
                X.append(np.asarray(q.X[chunk].mean(0)).ravel())
                meta.append(dict(role='pool', donor=d, condition=c, celltype=t, n_cells=len(chunk)))
    Xs = sp.csr_matrix(np.stack(X).astype(np.float32))
    obs = pd.DataFrame(meta)
    if arm == 'ct_normal_cells':            # Normal nuclei as the baseline rows
        nm = np.flatnonzero(np.isin(ct, CT) & (cond == 'Normal'))
        Xs = sp.vstack([Xs, q.X[nm].astype(np.float32)]).tocsr()
        obs = pd.concat([obs, pd.DataFrame(dict(role='cell', donor=donor[nm], condition='Normal',
                                                celltype=ct[nm], n_cells=1))], ignore_index=True)
    obs['grp'] = obs.donor + '|' + obs.celltype
    nor = obs.condition == 'Normal'
    ctrl = {'ct_normal_pools': (obs.role == 'pool') & nor, 'ct_normal_cells': obs.role == 'cell',
            'pooled_normal': (obs.role == 'pool') & nor, 'ct_selfmean': obs.role == 'pool',
            'donorct_self': obs.role == 'pool', 'trained': np.zeros(len(obs), bool)}[arm]
    keep = (obs.role == 'group') | ctrl
    obs, Xs = obs[keep.to_numpy()].reset_index(drop=True), Xs[np.flatnonzero(keep.to_numpy())]
    obs['perturbation'] = np.where(ctrl[keep].to_numpy(), 'control', 'query')
    obs.index = [f'r{i}' for i in range(len(obs))]
    pb = ad.AnnData(X=Xs, obs=obs, var=pd.DataFrame(index=pd.Index(q.var_names.astype(str))))
    n_shared = int(model.module.align_ds_mean.shape[1])
    kw = dict(return_latents=True, obs_only=True, progress_bar=False, head='A', readout='cosine',
              readout_confidence=False, n_cells_key='n_cells', center_mode=np.zeros(n_shared, np.float32),
              control_label='control')
    if arm in ('ct_normal_pools', 'ct_normal_cells', 'ct_selfmean'):
        kw['center_by'] = 'celltype'
    elif arm == 'donorct_self':
        kw['center_by'] = 'grp'
    elif arm == 'trained':
        kw['refresh_ctrl_means'] = False
    o = model.predict(pb, **kw)
    Z = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)
    g = (obs.role == 'group').to_numpy()
    # the placeholder dataset's trained z baseline (for latent option 'trained')
    ph = str(pb.obs['dataset'].iloc[0]) if 'dataset' in pb.obs else str(model.idx_to_batch[0])
    bi = list(map(str, model.idx_to_batch)).index(ph)
    np.savez(out / f'enc__{arm}.npz', Z=Z[g], align=model.module.align_ds_mean[bi].detach().cpu().numpy())
    obs[g].to_csv(out / f'obs__{arm}.csv', index=False)
    print(f'{arm}: {int(g.sum())} groups, {int(ctrl[keep].sum())} baseline rows, placeholder {ph}', flush=True)


def latent_shift(Z, obs, opt, align):
    nor = (obs.condition == 'Normal').to_numpy()
    Z = Z.astype(np.float64).copy()
    if opt == 'trained':
        return Z - align
    if opt == 'query_mean':
        return Z - Z.mean(0)
    if opt == 'normal_mean':
        return Z - Z[nor].mean(0)
    for t in CT:
        m = (obs.celltype == t).to_numpy()
        if opt == 'ct_mean':
            Z[m] -= Z[m].mean(0)
        elif opt == 'ct_normal' and (m & nor).any():
            Z[m] -= Z[m & nor].mean(0)
    return Z


def stage_score(a):
    arm, bank = ARMS[a.task // 2], ['trained', 'full'][a.task % 2]
    src = Path(a.out_dir)
    e = np.load(src / f'enc__{arm}.npz')
    obs = pd.read_csv(src / f'obs__{arm}.csv')
    noise = np.load(a.predict_npz, allow_pickle=True)['noise'].astype(np.float64)
    b = np.load(a.bank_npz, allow_pickle=True)
    rows_ = b['trained'].astype(bool) if bank == 'trained' else np.ones(len(b['labels']), bool)
    labels = [str(l) for l in b['labels'][rows_]]
    P = b['P'][rows_].astype(np.float64)
    Pn, N = unit(P), len(labels)
    from fig5_readout_grid import load_sets
    names, members, M = load_sets(a, labels)
    nset = M.sum(1)
    reads = []
    G0 = {(r.donor, r.celltype): (e['Z'][i].astype(np.float64), int(r.n_cells)) for i, r in obs.iterrows()}
    for kind in ('ctrl', 'rest', 'rrep'):
        keys, Q, ne = contrasts(G0, kind)
        reads.append((kind, 'n/a', keys, Q, ne))
    for opt in LATENT:
        Zs = latent_shift(e['Z'], obs, opt, e['align'])
        keys = list(zip(obs.donor, obs.celltype))
        reads.append(('abs', opt, keys, Zs, obs.n_cells.to_numpy(np.float64)))
    out = []
    for kind, lat, keys, Q, ne in reads:
        for rd, S in (('cosine', unit(Q) @ Pn.T), ('deflate', deflated(Q, P, ne, noise))):
            for i, (d, t) in enumerate(keys):
                zs = set_z(pct_of(S[i]), members)
                o = np.argsort(-S[i], kind='stable')
                for tl, ix in (('loss', o[:a.top]), ('gain', o[::-1][:a.top])):
                    h = M[:, ix].sum(1)
                    p = hypergeom.sf(h - 1, N, nset, a.top)
                    if tl == 'loss':
                        hl, pl = h, p
                    else:
                        hg, pg = h, p
                out.append(pd.DataFrame(dict(arm=arm, bank=bank, contrast=kind, latent=lat, readout=rd, donor=d,
                                             celltype=t, set=names, setz=[z[1] for z in zs], hits_loss=hl,
                                             p_loss=pl, hits_gain=hg, p_gain=pg)))
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(src / f'grid__{arm}__{bank}.parquet', index=False)
    print(f'{arm} {bank}: {len(df):,} rows', flush=True)


def stage_summarize(a):
    from fig5_readout_grid import score_frame
    df = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.out_dir).glob('grid__*.parquet'))])
    R, C = score_frame(df, ['arm', 'bank', 'contrast', 'latent', 'readout'])
    R.to_csv(Path(a.out_dir) / 'centring_scorecard.csv', index=False)
    C.to_csv(Path(a.out_dir) / 'centring_calls.csv.gz', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 400)
    print(R.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'score', 'summarize'])
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model-dir')
    ap.add_argument('--query')
    ap.add_argument('--predict-npz', help='fig5_predict.py predict_groups.npz (readout calibration)')
    ap.add_argument('--bank-npz')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--modules')
    ap.add_argument('--top', type=int, default=200)
    a = ap.parse_args()
    {'encode': stage_encode, 'score': stage_score, 'summarize': stage_summarize}[a.stage](a)
