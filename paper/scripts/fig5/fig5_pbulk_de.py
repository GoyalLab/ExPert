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

"""Donor-level pseudobulk DE for the ALD atlas (replaces the nucleus-level Wilcoxon table in Fig 5).

The old table (metabolic_weinberg/deg_tables.py) tested up to 6,000 NUCLEI per side, so every nucleus counted as a
replicate: p hit 0 for hundreds of genes, ~5-6k hepatocyte genes passed FDR < .05, and low-detection genes got
|logFC| ~ 20-30 from mean ratios near zero. Here the replicate is the DONOR:

    counts   raw counts summed per donor x cell type (donors with >= --min-nuclei nuclei of that type)
    filter   per cell type: genes with >= --min-count counts in >= --min-samples pseudobulks (filterByExpr-like)
    model    PyDESeq2, one fit per cell type over all its donors, design ~ condition (Normal = reference);
             Wald test AC vs Normal and AH vs Normal (2 vs 2 donors here), BH FDR with independent filtering
    logFC    apeGLM-style shrunk log2 fold change (lfc_shrink) -> low-count genes no longer explode; the MLE is kept
             as logfc_mle

Output (same columns the figure code reads, plus extras): deg.csv.gz with group ('AC|Hepatocyte'), gene,
score (Wald statistic), logfc (shrunk), logfc_mle, lfc_se, pval, pval_adj (NaN -> 1), base_mean, pct_group, pct_ctrl;
manifest.csv (donors / nuclei / genes tested / significant counts per group); pseudobulk_counts.npz.

    python paper/scripts/fig5/fig5_pbulk_de.py --h5ad ... --out-dir ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pseudobulk(a):
    """Summed counts and detected-nuclei counts per donor x cell type."""
    import anndata as ad
    import scipy.sparse as sp
    A = ad.read_h5ad(a.h5ad)
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    assert np.allclose(X.data[:10000], np.round(X.data[:10000])), 'X must hold raw counts'
    obs = A.obs[[a.donor_col, a.condition_col, a.celltype_col]].astype(str)
    key = obs[a.donor_col] + '|' + obs[a.celltype_col]
    keys = sorted(key.unique())
    ix = pd.Series(np.arange(len(obs)))
    S, D, meta = [], [], []
    for k in keys:
        rows = ix[(key == k).to_numpy()].to_numpy()
        sub = X[rows]
        S.append(np.asarray(sub.sum(0)).ravel())
        D.append(np.asarray((sub > 0).sum(0)).ravel())
        d, c = k.split('|')
        cond = obs[a.condition_col].iloc[rows].unique()
        assert len(cond) == 1, f'donor {d} has several conditions'
        meta.append(dict(sample=k, donor=d, celltype=c, condition=cond[0], n_nuclei=len(rows)))
    return np.stack(S), np.stack(D), pd.DataFrame(meta), np.asarray(A.var_names)


def fit_celltype(C, meta, genes, a):
    """PyDESeq2 for one cell type: {condition: results DataFrame} for every condition with >= 2 donors."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference
    inf = DefaultInference(n_cpus=a.n_cpus)
    keep = ((C >= a.min_count).sum(0) >= a.min_samples)
    counts = pd.DataFrame(C[:, keep].astype(np.int64), index=meta['sample'].to_numpy(), columns=genes[keep])
    md = pd.DataFrame({'condition': pd.Categorical(meta.condition.to_numpy(),
                                                   categories=[a.control] + sorted(set(meta.condition) - {a.control}))},
                      index=counts.index)
    try:
        dds = DeseqDataSet(counts=counts, metadata=md, design='~condition', refit_cooks=True, inference=inf, quiet=True)
    except TypeError:                                    # pydeseq2 < 0.5
        dds = DeseqDataSet(counts=counts, metadata=md, design_factors='condition', ref_level=['condition', a.control],
                           refit_cooks=True, inference=inf, quiet=True)
    dds.deseq2()
    coefs = list(dds.varm['LFC'].columns)
    out = {}
    for cond in md.condition.cat.categories[1:]:
        if (md.condition == cond).sum() < 2:
            continue
        st = DeseqStats(dds, contrast=['condition', cond, a.control], alpha=0.05, inference=inf, quiet=True)
        st.summary()
        mle = st.results_df.copy()
        coef = [c for c in coefs if cond in c and a.control not in c.split('vs')[0]]
        assert len(coef) == 1, (cond, coefs)
        st.lfc_shrink(coeff=coef[0])
        r = st.results_df
        out[cond] = pd.DataFrame({'gene': r.index, 'score': mle.stat.to_numpy(), 'logfc': r.log2FoldChange.to_numpy(),
                                  'logfc_mle': mle.log2FoldChange.to_numpy(), 'lfc_se': mle.lfcSE.to_numpy(),
                                  'pval': mle.pvalue.to_numpy(), 'pval_adj': mle.padj.fillna(1.0).to_numpy(),
                                  'base_mean': mle.baseMean.to_numpy()})
    return out, int(keep.sum())


def main(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S, D, meta, genes = pseudobulk(a)
    np.savez_compressed(out / 'pseudobulk_counts.npz', counts=S.astype(np.int64), detected=D.astype(np.int64),
                        genes=genes, **{c: meta[c].to_numpy() for c in meta.columns})
    meta.to_csv(out / 'pseudobulk_samples.csv', index=False)
    res, man = [], []
    for ct in sorted(meta.celltype.unique()):
        m = (meta.celltype == ct) & (meta.n_nuclei >= a.min_nuclei)
        sub = meta[m].reset_index(drop=True)
        if (sub.condition == a.control).sum() < 2:
            print(f'skip {ct}: < 2 control donors with >= {a.min_nuclei} nuclei', flush=True)
            continue
        R, n_tested = fit_celltype(S[m.to_numpy()], sub, genes, a)
        gi = {g: i for i, g in enumerate(genes)}
        for cond, r in R.items():
            grp, ctl = (sub.condition == cond).to_numpy(), (sub.condition == a.control).to_numpy()
            idx = np.array([gi[g] for g in r.gene])
            Dm = D[m.to_numpy()]
            r['pct_group'] = Dm[grp][:, idx].sum(0) / sub.n_nuclei[grp].sum()
            r['pct_ctrl'] = Dm[ctl][:, idx].sum(0) / sub.n_nuclei[ctl].sum()
            r.insert(0, 'group', f'{cond}|{ct}')
            res.append(r)
            sig = r.pval_adj < 0.05
            man.append(dict(group=f'{cond}|{ct}', donors_group=';'.join(sub.donor[grp]),
                            donors_ctrl=';'.join(sub.donor[ctl]), nuclei_group=int(sub.n_nuclei[grp].sum()),
                            nuclei_ctrl=int(sub.n_nuclei[ctl].sum()), genes_tested=n_tested, n_fdr05=int(sig.sum()),
                            **{f'n_fdr05_lfc{t:g}': int((sig & (r.logfc.abs() >= t)).sum()) for t in (0.25, 1, 2)},
                            max_abs_logfc=float(r.logfc.abs().max()), max_abs_logfc_mle=float(r.logfc_mle.abs().max()),
                            min_pval_adj=float(r.pval_adj.min())))
            print(man[-1], flush=True)
    pd.concat(res, ignore_index=True).to_csv(out / 'deg.csv.gz', index=False)
    pd.DataFrame(man).to_csv(out / 'manifest.csv', index=False)
    (out / 'run.json').write_text(json.dumps(vars(a), indent=2))
    print(pd.DataFrame(man).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5ad', required=True, help='nuclei with RAW counts in X')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--condition-col', default='cond_true')
    ap.add_argument('--control', default='Normal')
    ap.add_argument('--celltype-col', default='CellTypeSR')
    ap.add_argument('--donor-col', default='sample')
    ap.add_argument('--min-nuclei', type=int, default=30, help='donor x cell type pseudobulks below this are dropped')
    ap.add_argument('--min-count', type=int, default=10, help='gene filter: counts per pseudobulk')
    ap.add_argument('--min-samples', type=int, default=2, help='gene filter: pseudobulks reaching --min-count')
    ap.add_argument('--n-cpus', type=int, default=4)
    main(ap.parse_args())
