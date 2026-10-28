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

"""Fig. 6F: ExPert on the sorted-cell bulk RNA-seq (CD4 T, CD8 T, monocytes x unstimulated / stimulated; PMD patients
vs controls) -- does the independent cohort reproduce 'PMD = OXPHOS / mitochondrial / mTORC1 loss, in monocytes'?

stage encode  (model load) one predict() call on every sample as a pseudobulk (salmon counts summed per gene symbol,
              --mask genes dropped, n_cells = input cells), center_by = cell type x stim: each group's input
              baseline = its control samples. predict() needs >= 10 control rows per group, so the controls are
              repeated as baseline-only rows (the mean is exactly the control mean; these rows are never scored).
              Explicit zero latent centre, cosine readout, no hub correction.
stage score   per sample, its latent minus the mean latent of the controls of the same cell type x stim (a control
              is compared with the OTHER controls); cosine to the trained bank; Fig. 6 set z (mean percentile vs
              random sets) for the curated rows (OXPHOS complexes, mito biogenesis, stress, mTOR); each PMD patient's
              own lesion-gene rank. PMD vs control per cell type x stim: difference of mean set z, exact patient-level
              permutation (every relabelling with the same number of PMD patients).

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_bulk_expert.py --stage encode|score ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_bulk_expert')
log.setLevel(logging.INFO)

ROWS = ['module:complex_i', 'module:complex_iii', 'module:complex_iv', 'module:complex_v',
        'reactome:Mitochondrial Translation', 'reactome:Respiratory Electron Transport',
        'reactome:Mitochondrial Protein Import', 'reactome:Mitochondrial Iron-Sulfur Cluster Biogenesis',
        'reactome:Cristae Formation', 'module:isr', 'reactome:Detoxification of Reactive Oxygen Species',
        'module:cap_translation', 'reactome:mTORC1-mediated Signalling', 'reactome:Macroautophagy']


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def load_meta(a):
    m = pd.read_csv(a.meta).set_index('sample_id')
    m['grp'] = m.cell_type.astype(str) + '|' + m.stim.astype(str)
    return m


def stage_encode(a):
    import anndata as ad
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(a)
    C = pd.read_csv(a.counts, sep='\t')
    C = C.drop(columns=['gene_id']).groupby('gene_name').sum()          # gene symbol x sample
    g = pd.Index(C.index.astype(str))
    drop = np.zeros(len(g), bool)
    for p in a.mask.split(','):
        drop |= np.asarray(g.str.startswith(p[:-1]) if p.endswith('*') else (g == p), bool)
    C = C[~drop]
    X = C[meta.index].T.astype(np.float32)                              # samples x genes
    rows, obs = [X.to_numpy()], []
    for s_, r in meta.iterrows():
        obs.append(dict(sample_id=s_, role='sample', grp=r.grp, n_cells=int(r.input_cells),
                        perturbation='query'))
    for grp, mg in meta.groupby('grp'):                                 # baseline: controls repeated to >= --n-base
        ctl = mg[mg.group == 'control'].index
        reps = int(np.ceil(a.n_base / max(len(ctl), 1)))
        for _ in range(reps):
            rows.append(X.loc[ctl].to_numpy())
            obs += [dict(sample_id=c, role='baseline', grp=grp, n_cells=int(meta.loc[c, 'input_cells']),
                         perturbation='control') for c in ctl]
    obs = pd.DataFrame(obs)
    obs.index = [f'r{i}' for i in range(len(obs))]
    q = ad.AnnData(X=sp.csr_matrix(np.vstack(rows)), obs=obs, var=pd.DataFrame(index=C.index.astype(str)))
    model = ExPert.load(a.model_dir)
    n_shared = int(model.module.align_ds_mean.shape[1])
    o = model.predict(q, return_latents=True, obs_only=True, soft_predictions=False, progress_bar=False, head='A',
                      center_by='grp', center_mode=np.zeros(n_shared, np.float32), control_label='control',
                      n_cells_key='n_cells', readout='cosine', hub_correction=False)
    s = (obs.role == 'sample').to_numpy()
    Z = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)[s]
    np.save(out / 'z_samples.npy', Z)
    obs[s].to_csv(out / 'obs_samples.csv', index=False)
    hit = int(np.isin(np.asarray(model.adata.var_names).astype(str), C.index.astype(str)).sum())
    json.dump(dict(model_dir=a.model_dir, n_genes_query=int(C.shape[0]), n_model_genes_hit=hit,
                   n_baseline_rows=int((~s).sum())), open(out / 'encode.json', 'w'), indent=1)
    log.info('%d samples encoded; %d of the model genes present in the query', int(s.sum()), hit)


def stage_score(a):
    from scipy.stats import rankdata
    from fig6_pmd import load_sets
    out = Path(a.out_dir)
    meta = load_meta(a)
    obs = pd.read_csv(out / 'obs_samples.csv')
    Z = np.load(out / 'z_samples.npy').astype(np.float64)
    M = meta.loc[obs.sample_id]
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = np.asarray([str(x) for x in b['labels'][tr]])
    Pn = unit(b['P'][tr].astype(np.float64))
    sets = load_sets(a, labels)
    pos = {g: i for i, g in enumerate(labels)}
    idx = {k: np.array([pos[g] for g in sets[tuple(k.split(':', 1))]]) for k in ROWS
           if tuple(k.split(':', 1)) in sets}
    R = np.zeros_like(Z)
    for grp in M.grp.unique():                                # vs the controls of the same cell type x stim
        ix = np.flatnonzero((M.grp == grp).to_numpy())
        ctl = [i for i in ix if M.iloc[i].group == 'control']
        for i in ix:
            ref = [c for c in ctl if c != i]
            R[i] = Z[i] - Z[ref].mean(0)
    S = unit(R) @ Pn.T
    N = S.shape[1]
    pct = (rankdata(S, axis=1) - 1) / (N - 1)                  # 1 = top of the loss tail
    rows, ranks = [], []
    for i, s_ in enumerate(obs.sample_id):
        r = M.loc[s_]
        for k, ix in idx.items():
            n_ = len(ix)
            z = (pct[i, ix].mean() - pct[i].mean()) / np.sqrt(pct[i].var() / n_ * (N - n_) / (N - 1))
            rows.append(dict(sample_id=s_, patient=r.patient, group=r.group, cell_type=r.cell_type, stim=r.stim,
                             set=k, z=z))
        if isinstance(r.gene, str) and r.gene in pos:
            rk = int((S[i] > S[i, pos[r.gene]]).sum()) + 1
            ranks.append(dict(sample_id=s_, patient=r.patient, cell_type=r.cell_type, stim=r.stim, gene=r.gene,
                              defect=r.defect, rank=rk, of=N))
    T = pd.DataFrame(rows)
    T.to_csv(out / 'bulk_expert_set_z.csv', index=False)
    pd.DataFrame(ranks).to_csv(out / 'bulk_expert_lesion_ranks.csv', index=False)
    tests = []
    for (ct, st, k), x in T.groupby(['cell_type', 'stim', 'set']):
        v = x.set_index('patient').z
        pts = list(v.index)
        lab = np.array([x.set_index('patient').group[p] == 'mito' for p in pts])
        vals = v.to_numpy()
        eff = vals[lab].mean() - vals[~lab].mean()
        null = np.array([vals[list(c)].mean() - np.delete(vals, list(c)).mean()
                         for c in itertools.combinations(range(len(pts)), int(lab.sum()))])
        tests.append(dict(cell_type=ct, stim=st, set=k, effect=eff, n_pmd=int(lab.sum()), n_ctrl=int((~lab).sum()),
                          p_loss=float(np.mean(null >= eff - 1e-12)), p_two=float(np.mean(np.abs(null) >= abs(eff) - 1e-12)),
                          n_perm=len(null)))
    tests = pd.DataFrame(tests)
    tests.to_csv(out / 'bulk_expert_tests.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    log.info('PMD - control set z (loss > 0)\n%s', tests.pivot_table(index='set', columns=['cell_type', 'stim'],
                                                                      values='effect').round(1).to_string())
    log.info('one-sided p (loss)\n%s', tests.pivot_table(index='set', columns=['cell_type', 'stim'],
                                                          values='p_loss').round(3).to_string())
    if ranks:
        log.info('lesion ranks\n%s', pd.DataFrame(ranks).pivot_table(index=['patient', 'gene'],
                                                                        columns=['cell_type', 'stim'],
                                                                        values='rank').to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'score'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--counts', required=True, help='salmon.merged.gene_counts.tsv')
    ap.add_argument('--meta', required=True)
    ap.add_argument('--model-dir')
    ap.add_argument('--bank-npz')
    ap.add_argument('--modules')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--mask', default='MT-*,MT1*,MT2A,MT3,MT4', help='genes dropped (the Fig. 6 ExPert input mask)')
    ap.add_argument('--n-base', type=int, default=12, help='baseline rows per cell type x stim (controls repeated)')
    a_ = ap.parse_args()
    {'encode': stage_encode, 'score': stage_score}[a_.stage](a_)
