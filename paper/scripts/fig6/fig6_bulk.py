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

"""Fig. 6E: sorted-cell bulk RNA-seq (CD4 T, CD8 T, monocytes; unstimulated and stimulated) of PMD patients and
controls -- is the PMD signal a larger share of the transcriptome in monocytes than in T cells, relative to
stimulation?

Per cell type (salmon gene counts, genes with >= --min-count in >= --min-samples samples):
    vst      PyDESeq2 variance-stabilising transform (blind, design ~ 1), top --n-hvg genes by variance
    --norm tximport  the collaborators' normalisation (paper/figures/fig6/ext/261022_pca_minimal.R; DESeq2 from
             tximport): per-gene, per-library normalisation factors from the salmon average transcript lengths
             (--lengths) and median-of-ratios size factors, estimated ONCE over all libraries (as DESeq2
             estimateNormFactors) and kept when a cell type is subset; the vst is applied to counts / those factors
             (dispersion trend still fitted per cell type). --gene-filter mean: keep genes with mean normalised count
             >= --min-count (theirs) instead of >= --min-count in >= --min-samples libraries
    pca      PC1-PC2 coordinates + variance explained
    permanova  R^2 of stim and of group (PMD vs control) on Euclidean distances of the vst matrix (one factor at a time,
             and stim + group sequentially); p: stim permuted WITHIN patient (paired design), group permuted at the
             PATIENT level (every relabelling with the same number of PMD patients)
    effect   distance between the PMD and control centroids (within each stim condition) vs the median paired
             stim shift (each patient's stim - unstim) -> ratio
    pcr2     PC x covariate r^2 (PC1..PC6): eta^2 for categorical covariates, R^2 for numeric ones; permutation p --
             stim within patient, patient-level covariates (group, source, batch, age, sex) over patient relabellings,
             sample-level numeric ones (input cells, viability, genes detected, library size) over samples; plus each
             covariate's variance-weighted total over the PCs (sum of r^2 x the PC's share of variance)
    --fresh-only  restrict to fresh samples (P75 + the controls): shown, not testable (1 PMD patient)
    --regress  covariates removed from the vst matrix before the PCA (limma removeBatchEffect logic): one least-squares
             fit of the --protect factors + the covariates, then the covariates' fitted part subtracted (default protects
             stim only: protecting group with the true labels inflates the PMD permutation test). Donor cannot be removed (group is a donor property); batch 2 holds only PMD donors and
             pbmc_source differs from group only at P75, so those two corrections are partly aliased with PMD
    source   the PMD vs control axis built WITHOUT P75 (the only fresh PMD sample; all controls are fresh, the other PMD
             samples biobanked): P75's position on it (0 = control centroid, 1 = PMD centroid) -- separates disease
             from sample source

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_bulk.py --counts ... --meta ... --out-dir ...
"""
from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_bulk')
log.setLevel(logging.INFO)


def vst(counts, meta, n_cpus):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    dds = DeseqDataSet(counts=counts, metadata=meta[['stim']], design='~stim',
                       inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.vst(use_design=False)
    return pd.DataFrame(dds.layers['vst_counts'], index=counts.index, columns=counts.columns)


def tximport_norm_factors(C, L):
    """DESeq2 estimateNormFactors with a tximport length matrix: genes x libraries normalisation factors."""
    nm = L / np.exp(np.log(L).mean(1, keepdims=True))
    ec = C / nm
    with np.errstate(divide='ignore'):
        lg = np.log(ec)
    lgm = lg.mean(1)
    ok = np.isfinite(lgm)
    # estimateSizeFactorsForMatrix: median over genes with a finite log geometric mean and a non-zero count
    sf = np.exp([np.median((lg[ok, j] - lgm[ok])[ec[ok, j] > 0]) for j in range(C.shape[1])])
    nf = nm * sf[None, :]
    return nf / np.exp(np.log(nf).mean(1, keepdims=True))


def vst_nf(counts, nf, n_cpus):
    """Blind vst of counts / nf (samples x genes): PyDESeq2's dispersion trend, DESeq2's parametric vst formula."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    dds = DeseqDataSet(counts=counts, metadata=pd.DataFrame({'one': 'a'}, index=counts.index), design='~1',
                       inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.vst_fit(use_design=False)
    dds.layers['normed_counts'] = counts.to_numpy(float) / nf
    return pd.DataFrame(dds.vst_transform(), index=counts.index, columns=counts.columns)


def r2(D2, labels):
    """PERMANOVA R^2 of one factor on a squared-distance matrix."""
    n = len(labels)
    sst = D2.sum() / (2 * n)
    ssw = 0.0
    for g in np.unique(labels):
        m = labels == g
        ssw += D2[np.ix_(m, m)].sum() / (2 * m.sum())
    return 1 - ssw / sst


def regress(V, m, covs, protect=('stim',)):
    """Remove covariates from V (samples x genes) protecting `protect` (limma removeBatchEffect logic). Protecting
    group uses the true PMD labels and makes the patient-level PMD permutation anti-conservative (every relabelling is
    tested on data shaped by the real one), so the default protects stim only."""
    keep = pd.get_dummies(m[list(protect)].astype(str), drop_first=True).astype(float)
    parts = []
    for c in covs:
        x = m[c]
        if c in SAMPLE_NUMERIC + ['age_years', 'input_viability'] or x.dtype.kind in 'fiu':
            x = x.astype(float)
            parts.append(((x - x.mean()) / (x.std() or 1)).fillna(0).to_frame(c))
        else:
            parts.append(pd.get_dummies(x.astype(str), prefix=c, drop_first=True).astype(float))
    B = pd.concat(parts, axis=1)
    D = np.column_stack([np.ones(len(m)), keep.to_numpy(), B.to_numpy()])
    coef, *_ = np.linalg.lstsq(D, V.to_numpy(), rcond=None)
    nb = B.shape[1]
    return V - B.to_numpy() @ coef[-nb:]


PATIENT_LEVEL = ['group', 'pbmc_source', 'batch', 'age_years', 'sex']
SAMPLE_NUMERIC = ['log_input', 'input_viability', 'genes_detected', 'log_libsize']


def _r2(y, x):
    """eta^2 of a categorical x, or R^2 of a numeric x, for one PC score vector y."""
    ok = pd.notna(x)
    y, x = y[ok], x[ok]
    if len(np.unique(x)) < 2:
        return np.nan
    if x.dtype.kind in 'fiu':
        return float(np.corrcoef(y, x.astype(float))[0, 1] ** 2)
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - sum(((y[x == g] - y[x == g].mean()) ** 2).sum() for g in np.unique(x)) / tot)


def pc_r2(P, var, m, ct, a):
    """PC x covariate r^2 with permutation p (see the module docstring)."""
    rng = np.random.default_rng(0)
    pat = m.patient.to_numpy()
    pts = list(dict.fromkeys(pat))
    covs = ['group', 'stim', 'patient', 'batch', 'pbmc_source', 'age_years', 'sex'] + SAMPLE_NUMERIC
    rows = []
    for c in covs:
        x = m[c].to_numpy()
        x = x.astype(float) if c in SAMPLE_NUMERIC + ['age_years'] else x.astype(str)
        for k in range(min(a.n_pc, P.shape[1])):
            y = P[:, k]
            r = _r2(y, x)
            null = []
            if c == 'stim':                                   # within patient (paired)
                for _ in range(a.n_perm_pc):
                    xp = x.copy()
                    for p_ in pts:
                        ix = np.flatnonzero(pat == p_)
                        xp[ix] = rng.permutation(xp[ix])
                    null.append(_r2(y, xp))
            elif c in PATIENT_LEVEL:                          # shuffle the patients' values
                val = {p_: x[pat == p_][0] for p_ in pts}
                for _ in range(a.n_perm_pc):
                    sh = dict(zip(pts, rng.permutation([val[p_] for p_ in pts])))
                    null.append(_r2(y, np.array([sh[p_] for p_ in pat], dtype=x.dtype)))
            elif c != 'patient':                              # sample-level numeric
                null = [_r2(y, rng.permutation(x)) for _ in range(a.n_perm_pc)]
            p = float((np.sum(np.array(null) >= r - 1e-12) + 1) / (len(null) + 1)) if null else np.nan
            rows.append(dict(cell_type=ct, covariate=c, pc=f'PC{k + 1}', var=float(var[k]), r2=r, p=p,
                             weighted=r * float(var[k]) if np.isfinite(r) else 0.0))
    return pd.DataFrame(rows)


def main(a):
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA
    meta = pd.read_csv(a.meta).set_index('sample_id')
    C = pd.read_csv(a.counts, sep='\t')
    genes = C.gene_name.astype(str).to_numpy()
    C = C.drop(columns=['gene_id', 'gene_name']).round().astype(int)
    C.index = [f'{g}|{i}' for i, g in enumerate(genes)]
    NF = None
    if a.norm == 'tximport':                                  # over ALL libraries in the counts file, as theirs
        L = pd.read_csv(a.lengths, sep='\t').drop(columns=['gene_id', 'gene_name'])[C.columns]
        L.index = C.index
        NF = pd.DataFrame(tximport_norm_factors(C.to_numpy(float), L.to_numpy(float)), index=C.index,
                          columns=C.columns)
    if a.fresh_only:
        meta = meta[meta.pbmc_source.astype(str).str.startswith('fresh')]
    pca_rows, stats, eff, pcr2 = [], [], [], []
    drop = {}
    for part in filter(None, a.exclude_samples.split(';')):          # e.g. 'CD8:P43;Mono:P75,P83'
        ct_, pts_ = part.split(':')
        drop[ct_] = pts_.split(',')
    for ct, m in meta.groupby('cell_type'):
        m = m[~m.patient.isin(drop.get(ct, []))]
        X = C[m.index].T                                      # samples x genes
        m = m.assign(log_libsize=np.log10(X.sum(1).to_numpy()), genes_detected=(X > 0).sum(1).to_numpy(),
                     log_input=np.log10(m.input_cells.astype(float)))
        if a.gene_filter == 'mean':
            nfx = NF[m.index].T if NF is not None else 1.0
            X = X.loc[:, (X / nfx).mean(0) >= a.min_count]
        else:
            X = X.loc[:, (X >= a.min_count).sum(0) >= a.min_samples]
        V = vst_nf(X, NF.loc[X.columns, m.index].T.to_numpy(), a.n_cpus) if NF is not None else vst(X, m, a.n_cpus)
        if a.regress:
            V = regress(V, m, a.regress.split(','), tuple(a.protect.split(',')))
        V = V.loc[:, V.var(0).sort_values(ascending=False).index[:a.n_hvg]]
        Z = V.to_numpy() - V.to_numpy().mean(0)
        pc = PCA(n_components=min(6, len(Z) - 1)).fit(Z)
        P = pc.transform(Z)
        for i, s in enumerate(m.index):
            pca_rows.append(dict(cell_type=ct, sample_id=s, patient=m.loc[s, 'patient'], group=m.loc[s, 'group'],
                                 stim=m.loc[s, 'stim'], pbmc_source=m.loc[s, 'pbmc_source'],
                                 **{f'PC{k + 1}': P[i, k] for k in range(min(3, P.shape[1]))},
                                 **{f'var{k + 1}': pc.explained_variance_ratio_[k] for k in range(min(3, P.shape[1]))}))
        pcr2.append(pc_r2(P, pc.explained_variance_ratio_, m, ct, a))
        D2 = squareform(pdist(Z, 'sqeuclidean'))
        stim = m.stim.to_numpy()
        grp = m.group.to_numpy()
        pat = m.patient.to_numpy()
        r_stim, r_grp, r_src = r2(D2, stim), r2(D2, grp), r2(D2, m.pbmc_source.to_numpy())
        # stim: permute within patient (paired)
        rng = np.random.default_rng(0)
        null_s = []
        for _ in range(a.n_perm):
            sp = stim.copy()
            for p in np.unique(pat):
                ix = np.flatnonzero(pat == p)
                sp[ix] = rng.permutation(sp[ix])
            null_s.append(r2(D2, sp))
        # group: every patient-level relabelling with the same number of PMD patients
        pts = list(dict.fromkeys(pat))
        is_pmd = {p: grp[pat == p][0] == 'mito' for p in pts}
        k = sum(is_pmd.values())
        null_g = []
        for c in itertools.combinations(pts, k):
            gl = np.where(np.isin(pat, c), 'mito', 'control')
            null_g.append(r2(D2, gl))
        stats.append(dict(cell_type=ct, n_samples=len(m), n_genes=V.shape[1], r2_stim=r_stim,
                          p_stim=float((np.sum(np.array(null_s) >= r_stim - 1e-12) + 1) / (a.n_perm + 1)),
                          r2_pmd=r_grp, p_pmd=float(np.mean(np.array(null_g) >= r_grp - 1e-12)), n_perm_pmd=len(null_g),
                          r2_source=r_src, pmd_over_stim=r_grp / r_stim))
        # effect sizes: PMD vs control centroid distance within each stim condition vs paired stim shift
        cen = lambda mask: Z[mask].mean(0)
        d_pmd = {s_: float(np.linalg.norm(cen((grp == 'mito') & (stim == s_)) - cen((grp == 'control') & (stim == s_))))
                 for s_ in ('unstim', 'stim')}
        shifts = [float(np.linalg.norm(Z[(pat == p) & (stim == 'stim')][0] - Z[(pat == p) & (stim == 'unstim')][0]))
                  for p in pts if ((pat == p) & (stim == 'stim')).any() and ((pat == p) & (stim == 'unstim')).any()]
        # source check: PMD axis without P75, P75 projected (0 = control centroid, 1 = PMD centroid), per stim
        src = {}
        for s_ in ('unstim', 'stim'):
            pm = (grp == 'mito') & (stim == s_) & (pat != a.fresh_pmd)
            cm = (grp == 'control') & (stim == s_)
            t = (pat == a.fresh_pmd) & (stim == s_)
            if t.any():
                ax_ = cen(pm) - cen(cm)
                src[s_] = float((Z[t][0] - cen(cm)) @ ax_ / (ax_ @ ax_))
        eff.append(dict(cell_type=ct, d_pmd_unstim=d_pmd['unstim'], d_pmd_stim=d_pmd['stim'],
                        stim_shift_median=float(np.median(shifts)),
                        ratio_unstim=d_pmd['unstim'] / np.median(shifts), ratio_stim=d_pmd['stim'] / np.median(shifts),
                        p75_unstim=src.get('unstim', np.nan), p75_stim=src.get('stim', np.nan)))
        log.info('%s: R2 stim %.2f (p %.3f) | R2 PMD %.2f (p %.3f, %d perm) | R2 source %.2f | PMD/stim %.2f | '
                 'P75 on PMD axis unstim %.2f stim %.2f', ct, r_stim, stats[-1]['p_stim'], r_grp, stats[-1]['p_pmd'],
                 len(null_g), r_src, r_grp / r_stim, src.get('unstim', np.nan), src.get('stim', np.nan))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pca_rows).to_csv(out / 'bulk_pca.csv', index=False)
    pd.DataFrame(stats).to_csv(out / 'bulk_permanova.csv', index=False)
    pd.DataFrame(eff).to_csv(out / 'bulk_effects.csv', index=False)
    R = pd.concat(pcr2, ignore_index=True)
    R.to_csv(out / 'bulk_pc_r2.csv', index=False)
    w = R.groupby(['cell_type', 'covariate']).weighted.sum().unstack()
    log.info('variance-weighted r2 over PC1-%d\n%s', a.n_pc, w.round(3).to_string())
    log.info('effects\n%s', pd.DataFrame(eff).round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--counts', required=True, help='salmon.merged.gene_counts.tsv')
    ap.add_argument('--meta', required=True, help='sample metadata csv (sample_id, patient, group, cell_type, stim, ...)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--norm', default='celltype', choices=['celltype', 'tximport'],
                    help='celltype: size factors per cell type; tximport: length-aware factors over all libraries')
    ap.add_argument('--lengths', help='--norm tximport: salmon.merged.gene_lengths.tsv')
    ap.add_argument('--gene-filter', default='count', choices=['count', 'mean'])
    ap.add_argument('--min-count', type=int, default=10)
    ap.add_argument('--min-samples', type=int, default=3)
    ap.add_argument('--n-hvg', type=int, default=2000)
    ap.add_argument('--n-perm', type=int, default=9999, help='within-patient stim permutations')
    ap.add_argument('--n-pc', type=int, default=6, help='pcr2: PCs scored')
    ap.add_argument('--n-perm-pc', type=int, default=999, help='pcr2: permutations per PC x covariate')
    ap.add_argument('--exclude-samples', default='', help="per-cell-type patients dropped (QC), 'CD8:P43;Mono:P75,P83'")
    ap.add_argument('--fresh-only', action='store_true', help='fresh samples only (pbmc_source fresh*)')
    ap.add_argument('--protect', default='stim', help='regress: factors kept in the model (group = anti-conservative)')
    ap.add_argument('--regress', default='', help='comma list of covariates removed before the PCA (stim, group kept)')
    ap.add_argument('--fresh-pmd', default='P75', help='the PMD patient sampled fresh like the controls')
    ap.add_argument('--n-cpus', type=int, default=4)
    main(ap.parse_args())
