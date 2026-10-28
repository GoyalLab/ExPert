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

"""Score the bulk RNA-seq livers (4 healthy, 6 AC, 6 AH; donor overlap with the snRNA-seq pools unknown) with ExPert, as the snRNA-seq Fig 5 readout:
does bulk reproduce the snRNA hepatocyte predictions, and are mRNA export / splicing enriched the same way?

stage encode (SLURM: loads the model)
    input    DESeq2-normalised counts per liver -> gene symbols (first SYMBOL; duplicates summed), mito_mt mask (as
             Fig 5), each sample rescaled to --lib-size total counts (a pseudobulk-of-nuclei scale); one row per liver
    predict  ONE call, center_by the single 'liver' type with the healthy livers as the input baseline ('control'),
             center_mode 0, head A (the Fig 5 encode settings)
    writes   bulk_latents.npz (Z per liver, sample table)
stage score (light)
    latent   subtract the unweighted mean over all livers (the Fig 5 per-cell-type centring), pool by condition
    readout  cosine to the TRAINED bank (--bank-npz, as Fig 5)
    compare  per condition: Spearman of bulk vs snRNA hepatocyte scores (all bank genes), top-k overlap, top-k ORA of
             mRNA export / splicing (hypergeometric, background = bank), and the 5l export families' mean z
    writes   fig5_bulk_expert_scores.csv.gz, fig5_bulk_expert_compare.csv, fig5_bulk_expert_families.csv

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_bulk_expert.py --stage encode|score ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

log = logging.getLogger('fig5_bulk_expert')
CONDS = ['AC', 'AH', 'Normal']


def cond_of(sample):
    return 'Normal' if sample.startswith('N') else sample[:2]


def bulk_matrix(a):
    """Liver x symbol count matrix from the normalised-counts table, symbols from the DESeq2 table."""
    nc = pd.read_csv(Path(a.bulk_dir) / 'normalized_counts_all_input_genes.csv')
    nc = nc.rename(columns={nc.columns[0]: 'gene_id'})
    sym = pd.read_csv(Path(a.bulk_dir) / 'DESeq2_all_genes_Healthy_vs_AC.csv', usecols=['gene_id', 'SYMBOL'],
                      low_memory=False)
    sym['symbol'] = sym.SYMBOL.astype(str).str.split('; ').str[0]
    nc = nc.merge(sym[sym.SYMBOL.notna()][['gene_id', 'symbol']], on='gene_id')
    M = nc.drop(columns=['gene_id']).groupby('symbol').sum()
    return M.T                                                    # livers x genes


def stage_encode(a):
    import anndata as ad
    import scipy.sparse as sp
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    from fig5_variants import MASKS, mask_query
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    if not (hasattr(model.module, 'has_readout_noise') and model.module.has_readout_noise()):
        log.info('no stored readout calibration -> %s', model.add_readout_calibration(a.ctrl_pools))
    M = bulk_matrix(a)
    X = M.to_numpy(np.float64)
    X = X / X.sum(1, keepdims=True) * a.lib_size                # per-liver library -> a pseudobulk scale
    obs = pd.DataFrame({'sample': M.index, 'condition': [cond_of(s) for s in M.index]})
    obs['celltype'] = 'liver'
    obs['n_cells'] = a.n_cells
    obs['perturbation'] = np.where(obs.condition == 'Normal', 'control', 'query')
    obs.index = [f'b{i}' for i in range(len(obs))]
    q = ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=M.columns.astype(str)))
    q, dropped = mask_query(q, MASKS[a.mask])
    log.info('bulk query %s, masked %d genes', q.shape, len(dropped))
    n_shared = int(model.module.align_ds_mean.shape[1])
    o = model.predict(q, return_latents=True, obs_only=True, progress_bar=False, head='A', center_by='celltype',
                      center_mode=np.zeros(n_shared, np.float32), control_label='control', n_cells_key='n_cells',
                      readout='deflate', hub_correction=False)
    Z = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)
    np.savez(out / 'bulk_latents.npz', Z=Z, sample=obs['sample'].to_numpy(), condition=obs.condition.to_numpy())
    log.info('wrote %s', out / 'bulk_latents.npz')


def stage_score(a):
    import argparse as ap_
    from scipy.stats import hypergeom, spearmanr
    import fig5_final as F
    out = Path(a.out_dir)
    e = np.load(out / 'bulk_latents.npz', allow_pickle=True)
    Z = e['Z'].astype(np.float64)
    Z = Z - Z.mean(0)                                            # latent centring over all livers (unweighted)
    cond = e['condition'].astype(str)
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = np.array([str(x) for x in b['labels'][tr]])
    P = F.unit(b['P'][tr].astype(np.float64))
    S = {c: F.unit(Z[cond == c].mean(0)) @ P.T for c in CONDS}
    sn = np.load(a.profiles, allow_pickle=True)
    snL = [str(x) for x in sn['labels']]
    snS = {str(k): sn['cond'][i] for i, k in enumerate(sn['cond_keys'])}
    assert snL == list(labels), 'bank order differs from the snRNA profiles'
    gs = F.gene_sets(ap_.Namespace(modules=a.modules, reactome_gmt=a.reactome_gmt), list(labels))
    sets = {'mRNA export': set(gs['mRNA export (grouped)']), 'mRNA splicing': set(gs['mRNA splicing'])}
    N = len(labels)
    pd.DataFrame({'gene': labels, **{f'bulk_{c}': S[c] for c in CONDS},
                  **{f'sn_{c}_hep': snS[f'{c}|Hepatocyte'] for c in CONDS}}).to_csv(
        out / 'fig5_bulk_expert_scores.csv.gz', index=False)
    rows = []
    for c in CONDS:
        sn_v = snS[f'{c}|Hepatocyte']
        r = spearmanr(S[c], sn_v)
        rows.append(dict(cond=c, metric='spearman_bulk_vs_sn_hep', value=r.statistic, p=r.pvalue))
        # cross-check: bulk condition vs the OTHER snRNA conditions (is the agreement condition-specific?)
        for c2 in CONDS:
            if c2 != c:
                rows.append(dict(cond=c, metric=f'spearman_bulk_vs_sn_{c2}_hep',
                                 value=spearmanr(S[c], snS[f'{c2}|Hepatocyte']).statistic, p=np.nan))
        tb, ts = set(labels[np.argsort(-S[c])[:a.top]]), set(labels[np.argsort(-sn_v)[:a.top]])
        ov = len(tb & ts)
        rows.append(dict(cond=c, metric=f'top{a.top}_overlap_bulk_sn', value=ov,
                         p=hypergeom.sf(ov - 1, N, a.top, a.top)))
        for nm, st in sets.items():
            m = st & set(labels)
            for src, v in (('bulk', S[c]), ('sn_hep', sn_v)):
                top = set(labels[np.argsort(-v)[:a.top]])
                h = len(top & m)
                rows.append(dict(cond=c, metric=f'{src}_{nm}_top{a.top}_hits', value=h,
                                 expected=len(m) * a.top / N, p=hypergeom.sf(h - 1, N, len(m), a.top)))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'fig5_bulk_expert_compare.csv', index=False)
    fam = []
    pos = {g: i for i, g in enumerate(labels)}
    for c in CONDS:
        zb = (S[c] - S[c].mean()) / S[c].std()
        zs = (snS[f'{c}|Hepatocyte'] - snS[f'{c}|Hepatocyte'].mean()) / snS[f'{c}|Hepatocyte'].std()
        for g in sets['mRNA export']:
            if g in pos:
                fam.append(dict(cond=c, gene=g, family=F.export_family(g), z_bulk=zb[pos[g]], z_sn=zs[pos[g]]))
    Fm = pd.DataFrame(fam)
    Fm.to_csv(out / 'fig5_bulk_expert_families.csv', index=False)
    pd.set_option('display.width', 200)
    print(R.round(4).to_string(index=False))
    print(Fm.groupby(['family', 'cond'])[['z_bulk', 'z_sn']].mean().unstack('cond').round(2).to_string())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', required=True, choices=['encode', 'score'])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--bulk-dir', help='BulkRNASeqAnalysis/tables')
    p.add_argument('--model-dir')
    p.add_argument('--ctrl-pools')
    p.add_argument('--mask', default='mito_mt')
    p.add_argument('--lib-size', type=float, default=5000.0, help='encode: total counts per liver row')
    p.add_argument('--n-cells', type=int, default=500, help='encode: nominal depth for the readout calibration')
    p.add_argument('--bank-npz')
    p.add_argument('--profiles', help='data/final/fig5_profiles.npz (snRNA condition profiles, same bank order)')
    p.add_argument('--modules')
    p.add_argument('--reactome-gmt')
    p.add_argument('--top', type=int, default=200)
    a = p.parse_args()
    {'encode': stage_encode, 'score': stage_score}[a.stage](a)
