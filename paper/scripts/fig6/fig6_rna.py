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

"""Fig. 6 RNA-only controls: does expression alone show 'monocytes are the PMD-dysregulated cell type'?

The same donors, cell types (>= --min-cells per donor), groups and permutation designs as the ExPert panels, on donor
pseudobulks (summed raw counts per donor x cell type, MT-* genes dropped as in the ExPert input):
  cluster   panel B analogue: log-CPM of the --n-hvg most variable genes, each donor's deviation from the mean of all
            other donors, cosine distance, average linkage, 2-cluster cut vs diagnosis -> ARI + exact relabelling p
  donorset  panel C analogue: the 36 balanced donor splits of fig6_donorset.py; within a half, condition log-CPM
            centroid minus the mean of the half's condition centroids; Reactome / module sets scored by the mean
            percentile of their genes' own contrast (set z); threshold = non-PMD 1% |z|; replication rate per set
  sclfc     single-cell log2FC PMD vs non-PMD per cell type, Seurat-style pseudocount 1 on CP10k (panel C)
  sccluster panel E on expression, single-cell matched: per-donor mean of the cells' log expression (HVGs / all
            detected genes), donor vs mean of the others, cosine, average linkage, ARI + exact p
  scde      per-donor single-cell Wilcoxon DE in the same design (donor's cells vs pooled non-PMD cells) -> set z
            per donor: the cell-level DE Fig. 6B scores, matching the per-cell ExPert encoding
  shift     per-donor set z in the ExPert 'ctrl' design (PMD donor vs pooled non-PMD; non-PMD donor vs the other
            non-PMD), for the PMD vs non-PMD permutation dot plot (Fig. 6B)
  deseq     PyDESeq2 per cell type, PMD vs non-PMD on donor pseudobulks: DEGs at padj < 0.1 for the true labels and for
            every relabelling (4 of 9) -> exact p of the DEG count

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_rna.py --query <h5ad> --out-dir <d> --reactome-gmt ... --modules ...
"""
from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_rna')
log.setLevel(logging.INFO)

COND = {'PT43': 'PMD', 'PT46': 'PMD', 'PT54': 'PMD', 'PT67': 'PMD', 'PT41': 'TBCK', 'PT42': 'TBCK',
        'PT38': 'undiagnosed', 'PT50': 'undiagnosed', 'PT55': 'undiagnosed'}
PMD = [d for d, c in COND.items() if c == 'PMD']


def pseudobulk(a):
    import anndata as ad
    q = ad.read_h5ad(a.query)
    X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    genes = np.asarray(q.var_names).astype(str)
    keep = ~pd.Index(genes).str.startswith('MT-')
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    dn = q.obs[a.donor_col].astype(str).to_numpy()
    rows, meta = [], []
    for t in sorted(set(ct)):
        for d in COND:
            m = (ct == t) & (dn == d)
            if m.sum() >= a.min_cells:
                rows.append(np.asarray(X[m][:, keep].sum(0)).ravel())
                meta.append((t, d, int(m.sum())))
    return np.stack(rows), pd.DataFrame(meta, columns=['celltype', 'donor', 'n_cells']), genes[keep]


def logcpm(C):
    return np.log1p(C / C.sum(1, keepdims=True) * 1e6)


def stage_cluster(PB, meta, genes, a):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score
    out = []
    for t, m in meta.groupby('celltype'):
        if len(m) < 6:
            continue
        L = logcpm(PB[m.index])
        L = L[:, (PB[m.index] > 0).mean(0) >= 0.5]
        hv = np.argsort(-L.var(0))[:a.n_hvg]
        L = L[:, hv]
        ds = list(m.donor)
        dev = np.stack([L[i] - np.delete(L, i, 0).mean(0) for i in range(len(ds))])
        lab = fcluster(linkage(pdist(dev, 'cosine'), 'average'), 2, 'maxclust')
        truth = [d in PMD for d in ds]
        ari = adjusted_rand_score(truth, lab)
        null = []
        for c in itertools.combinations(range(len(ds)), sum(truth)):
            tt = np.zeros(len(ds), bool)
            tt[list(c)] = True
            null.append(adjusted_rand_score(tt, lab))
        out.append(dict(celltype=t, donors=' '.join(ds), clusters=' '.join(map(str, lab)), ari=ari,
                        p=float(np.mean(np.array(null) >= ari - 1e-12)), n_genes=L.shape[1]))
    return pd.DataFrame(out)


def splits():
    pmd = PMD
    tb = [d for d, c in COND.items() if c == 'TBCK']
    un = [d for d, c in COND.items() if c == 'undiagnosed']
    out = []
    for p in itertools.combinations(pmd, 2):
        if pmd[0] not in p:
            continue
        for t in tb:
            for k in (1, 2):
                for u in itertools.combinations(un, k):
                    A = list(p) + [t] + list(u)
                    out.append((A, [d for d in COND if d not in A]))
    return out


def load_sets(a, universe):
    u = set(universe)
    sets = {}
    for line in open(a.reactome_gmt):
        f = line.rstrip('\n').split('\t')
        g = sorted({x.split(',')[0] for x in f[2:] if x} & u)
        if 10 <= len(g) <= 500:
            sets[('reactome', f[0])] = g
    mods = {}
    for line in open(a.modules):
        if line.startswith('#') or not line.strip():
            continue
        m, g = line.rstrip('\n').split('\t')[:2]
        hits = [x for x in universe if (x.startswith(g[:-1]) if g.endswith('*') else x == g)]
        mods.setdefault(m, set()).update(hits)
    for m, g in mods.items():
        if len(g) >= 5:
            sets[('module', m)] = sorted(g)
    return sets


def stage_donorset(PB, meta, genes, a):
    from scipy.stats import rankdata
    SPL = splits()
    rows = []
    for t, m in meta.groupby('celltype'):
        L = logcpm(PB[m.index])
        ok = (PB[m.index] > 0).mean(0) >= 0.5                 # expressed genes only, as a DE test would
        L, g_ = L[:, ok], genes[ok]
        sets = load_sets(a, list(g_))
        keys = list(sets)
        pos = {x: i for i, x in enumerate(g_)}
        ii = np.concatenate([[pos[x] for x in sets[k]] for k in keys])
        jj = np.concatenate([[j] * len(sets[k]) for j, k in enumerate(keys)])
        M = sp.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(g_), len(keys)))
        sizes = np.asarray(M.sum(0)).ravel()
        cen = dict(zip(m.donor, L))
        Q, info = [], []
        for s, halves in enumerate(SPL):
            for h, members in enumerate(halves):
                ms = [d for d in members if d in cen]
                conds = sorted({COND[d] for d in ms})
                if len(conds) < 3:
                    continue
                C = {c: np.mean([cen[d] for d in ms if COND[d] == c], 0) for c in conds}
                mu = np.mean(list(C.values()), 0)
                for c in conds:
                    Q.append(C[c] - mu)
                    info.append((s, h, c))
        if not Q:
            continue
        Q = np.stack(Q)
        N = Q.shape[1]
        pct = (rankdata(Q, axis=1) - 1) / (N - 1)            # high = up in the condition
        mu = np.asarray(M.T @ pct.T).T / sizes
        sd = np.sqrt(pct.var(1, keepdims=True) / sizes * (N - sizes) / (N - 1))
        Z = (mu - pct.mean(1, keepdims=True)) / sd
        for (s, h, c), z in zip(info, Z):
            rows.append(pd.DataFrame({'celltype': t, 'split': s, 'half': h, 'condition': c,
                                      'family': [k[0] for k in keys], 'set': [k[1] for k in keys], 'z': z}))
        log.info('[donorset] %s: %d contrasts over %d genes', t, len(Q), N)
    T = pd.concat(rows, ignore_index=True)
    thr = float(np.quantile(T.z[T.condition != 'PMD'].abs(), 1 - a.alpha))
    w = T.pivot_table(index=['celltype', 'condition', 'family', 'set', 'split'], columns='half', values='z').dropna()
    g = pd.DataFrame({'up': (w[0] > thr) & (w[1] > thr), 'down': (w[0] < -thr) & (w[1] < -thr),
                      'z_mean': (w[0] + w[1]) / 2}).groupby(level=['celltype', 'condition', 'family', 'set'])
    R = pd.DataFrame({'rep_up': g.up.mean(), 'rep_down': g.down.mean(), 'z_mean': g.z_mean.mean()}).reset_index()
    R['threshold'] = thr
    return T, R


def stage_shift(PB, meta, genes, a):
    """Per-donor expression shifts in the ExPert 'ctrl' design (fig6_pmd.py score): a PMD donor vs the pooled non-PMD
    donors, a non-PMD donor vs the pooled OTHER non-PMD donors (summed counts = cell-weighted pooling), as log-CPM
    differences over expressed genes; every set scored by the set z of its genes' percentile (high = up)."""
    from scipy.stats import rankdata
    rows = []
    nonpmd = [d for d in COND if d not in PMD]
    for t, m in meta.groupby('celltype'):
        C = dict(zip(m.donor, PB[m.index]))
        ok = (PB[m.index] > 0).mean(0) >= 0.5
        g_ = genes[ok]
        sets = load_sets(a, list(g_))
        pos = {x: i for i, x in enumerate(g_)}
        keys = list(sets)
        idx = [np.array([pos[x] for x in sets[k]]) for k in keys]
        for d in C:
            ref = [r for r in nonpmd if r in C and r != d]
            if not ref:
                continue
            sh = (logcpm(C[d][None]) - logcpm(np.sum([C[r] for r in ref], 0)[None]))[0][ok]
            N = len(sh)
            pct = (rankdata(sh) - 1) / (N - 1)
            var = pct.var()
            z = [(pct[ix].mean() - pct.mean()) / np.sqrt(var / len(ix) * (N - len(ix)) / (N - 1)) for ix in idx]
            rows.append(pd.DataFrame({'donor': d, 'celltype': t, 'family': [k[0] for k in keys],
                                      'set': [k[1] for k in keys], 'z': z}))
    return pd.concat(rows, ignore_index=True)


def stage_sclfc(a):
    """Single-cell log2 fold change, PMD vs non-PMD cells per cell type, Seurat-style (FindMarkers): normalize_total
    1e4 per cell, then log2((mean_PMD + 1) / (mean_nonPMD + 1)) on that linear scale. The pseudocount of 1 keeps
    barely detected genes near 0 -- scanpy's rank_genes_groups uses 1e-9, which sends a gene seen in a few cells of one
    group and none of the other to |log2FC| ~ 20. Also writes each group's detection rate (pct_pmd / pct_ref)."""
    import anndata as ad
    import scanpy as sc
    q = ad.read_h5ad(a.query)
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    dn = q.obs[a.donor_col].astype(str).to_numpy()
    out = []
    for t in sorted(set(ct)):
        x = q[ct == t].copy()
        d_ = dn[ct == t]
        ok = np.isin(d_, [d for d in COND if (d_ == d).sum() >= a.min_cells])
        x, d_ = x[ok].copy(), d_[ok]
        pmd = np.isin(d_, PMD)
        if pmd.sum() == 0 or (~pmd).sum() == 0:
            continue
        det = x.X > 0
        sc.pp.normalize_total(x, target_sum=1e4)
        mA = np.asarray(x.X[pmd].mean(0)).ravel()
        mB = np.asarray(x.X[~pmd].mean(0)).ravel()
        out.append(pd.DataFrame({'celltype': t, 'gene': np.asarray(x.var_names).astype(str),
                                 'log2fc': np.log2((mA + 1) / (mB + 1)), 'mean_pmd': mA, 'mean_ref': mB,
                                 'pct_pmd': np.asarray(det[pmd].mean(0)).ravel(),
                                 'pct_ref': np.asarray(det[~pmd].mean(0)).ravel(),
                                 'n_pmd': int(pmd.sum()), 'n_ref': int((~pmd).sum())}))
        log.info('[sclfc] %s: %d vs %d cells', t, int(pmd.sum()), int((~pmd).sum()))
    return pd.concat(out, ignore_index=True)


def stage_sccluster(a):
    """Panel E's label-free clustering on expression, matched to the ExPert per-cell encoding: per cell type
    (normalize_total 1e4 + log1p on single cells, --mask genes dropped), each donor (>= --min-cells) = the MEAN of its
    cells' log expression over the cell type's --n-hvg highly variable genes (or every gene detected in >= --min-frac
    of cells: 'all'); each donor's deviation from the mean of all OTHER donors; cosine distance, average linkage;
    2-cluster cut vs diagnosis -> ARI + exact relabelling p (as fig6_pmd.py score)."""
    import json
    import anndata as ad
    import scanpy as sc
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score
    q = ad.read_h5ad(a.query)
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(len(v), bool)
    for p in a.mask.split(','):
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    q = q[:, ~drop].copy()
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    dn = q.obs[a.donor_col].astype(str).to_numpy()
    out = []
    for t in sorted(set(ct)):
        x = q[ct == t].copy()
        d_ = dn[ct == t]
        ds = [d for d in COND if (d_ == d).sum() >= a.min_cells]
        if len(ds) < 6:
            continue
        keep = np.isin(d_, ds)
        x, d_ = x[keep].copy(), d_[keep]
        x = x[:, np.asarray((x.X > 0).mean(0)).ravel() >= a.min_frac].copy()
        sc.pp.normalize_total(x, target_sum=1e4)
        sc.pp.log1p(x)
        sc.pp.highly_variable_genes(x, n_top_genes=min(a.n_hvg, x.n_vars))
        for genes, mask in (('hvg', x.var.highly_variable.to_numpy()), ('all', np.ones(x.n_vars, bool))):
            X = x.X[:, mask]
            M = np.stack([np.asarray(X[d_ == d].mean(0)).ravel() for d in ds])
            dev = np.stack([M[i] - np.delete(M, i, 0).mean(0) for i in range(len(ds))])
            L = linkage(pdist(dev, 'cosine'), 'average')
            lab = fcluster(L, 2, 'maxclust')
            truth = [d in PMD for d in ds]
            ari = adjusted_rand_score(truth, lab)
            null = []
            for c in itertools.combinations(range(len(ds)), sum(truth)):
                tt = np.zeros(len(ds), bool)
                tt[list(c)] = True
                null.append(adjusted_rand_score(tt, lab))
            out.append(dict(celltype=t, genes=genes, n_genes=int(mask.sum()), donors=' '.join(ds),
                            clusters=' '.join(map(str, lab)), ari=ari,
                            p=float(np.mean(np.array(null) >= ari - 1e-12)), linkage=json.dumps(L.tolist())))
            log.info('[sccluster] %s %s: ARI %.2f p %.3f | %s', t, genes, ari, out[-1]['p'],
                     ' '.join(f'{d}:{l}' for d, l in zip(ds, lab)))
    return pd.DataFrame(out)


def stage_scde(a):
    """Single-cell analogue of the ExPert per-donor shifts (fig6_pmd.py score, ref 'ctrl'), the cell-level DE Fig. 6B
    scores: per cell type (normalize_total 1e4 + log1p, --mask genes dropped, genes detected in >= --min-frac of its
    cells), each donor's cells vs the pooled non-PMD cells (a non-PMD donor vs the OTHER non-PMD donors), scanpy
    Wilcoxon; every set scored by the set z of its genes' percentile in the signed score ranking (high = up)."""
    import anndata as ad
    import scanpy as sc
    from scipy.stats import rankdata
    q = ad.read_h5ad(a.query)
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(len(v), bool)
    for p in a.mask.split(','):
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    q = q[:, ~drop].copy()
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    dn = q.obs[a.donor_col].astype(str).to_numpy()
    nonpmd = [d for d in COND if d not in PMD]
    rows = []
    for t in sorted(set(ct)):
        x = q[ct == t].copy()
        d_ = dn[ct == t]
        donors = [d for d in COND if (d_ == d).sum() >= a.min_cells]
        keep = np.isin(d_, donors)
        x, d_ = x[keep].copy(), d_[keep]
        X = x.X
        frac = np.asarray((X > 0).mean(0)).ravel()
        x = x[:, frac >= a.min_frac].copy()
        sc.pp.normalize_total(x, target_sum=1e4)
        sc.pp.log1p(x)
        genes = np.asarray(x.var_names).astype(str)
        sets = load_sets(a, list(genes))
        keys = list(sets)
        pos = {g: i for i, g in enumerate(genes)}
        idx = [np.array([pos[g] for g in sets[k]]) for k in keys]
        for d in donors:
            ref = [r for r in nonpmd if r in donors and r != d]
            if not ref:                                       # no other non-PMD donor in this cell type
                continue
            lab = np.where(d_ == d, 'A', np.where(np.isin(d_, ref), 'B', 'x'))
            y = x[lab != 'x'].copy()
            y.obs['lab'] = pd.Categorical(lab[lab != 'x'])
            sc.tl.rank_genes_groups(y, 'lab', groups=['A'], reference='B', method='wilcoxon')
            df = sc.get.rank_genes_groups_df(y, 'A').set_index('names')
            s_ = df.scores.reindex(genes).to_numpy(np.float64)
            N = len(s_)
            pct = (rankdata(s_) - 1) / (N - 1)
            var = pct.var()
            z = [(pct[ix].mean() - pct.mean()) / np.sqrt(var / len(ix) * (N - len(ix)) / (N - 1)) for ix in idx]
            rows.append(pd.DataFrame({'donor': d, 'celltype': t, 'family': [k[0] for k in keys],
                                      'set': [k[1] for k in keys], 'z': z}))
            log.info('[scde] %s %s: %d vs %d cells, %d genes', t, d, int((lab == 'A').sum()), int((lab == 'B').sum()), N)
    return pd.concat(rows, ignore_index=True)


def stage_deseq(PB, meta, genes, a):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference
    inf = DefaultInference(n_cpus=a.n_cpus)
    out = []
    for t, m in meta.groupby('celltype'):
        ds = list(m.donor)
        k = sum(d in PMD for d in ds)
        if k < 2 or len(ds) - k < 2:
            continue
        C = PB[m.index]
        ok = (C >= 10).sum(0) >= 3
        counts = pd.DataFrame(C[:, ok].astype(int), index=ds, columns=genes[ok])

        def n_deg(cases):
            md = pd.DataFrame({'g': ['case' if d in cases else 'ref' for d in ds]}, index=ds)
            dds = DeseqDataSet(counts=counts, metadata=md, design='~g', quiet=True, inference=inf)
            dds.deseq2()
            st = DeseqStats(dds, contrast=['g', 'case', 'ref'], quiet=True, inference=inf)
            st.summary()
            r = st.results_df
            return int((r.padj < a.padj).sum()), r
        obs, res = n_deg(set(PMD))
        null = [n_deg({ds[i] for i in c})[0] for c in itertools.combinations(range(len(ds)), k)]
        out.append(dict(celltype=t, n_donors=len(ds), n_pmd=k, n_genes=int(ok.sum()), deg_padj=obs,
                        null_median=float(np.median(null)), p=float(np.mean(np.array(null) >= obs)),
                        n_perm=len(null)))
        res.assign(celltype=t).to_csv(Path(a.out_dir) / f'deseq__{t.replace(" ", "_").replace("+", "p")}.csv.gz')
        log.info('[deseq] %s: %d DEGs (null median %.1f, p %.3f)', t, obs, np.median(null), out[-1]['p'])
    return pd.DataFrame(out)


def main(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = set(a.stages.split(','))
    if want == {'scde'}:
        stage_scde(a).to_csv(out / 'rna_scde_set_z.csv.gz', index=False)
        return
    if want == {'sclfc'}:
        stage_sclfc(a).to_csv(out / 'rna_sc_lfc.csv.gz', index=False)
        return
    if want == {'sccluster'}:
        stage_sccluster(a).to_csv(out / 'rna_sc_clustering.csv', index=False)
        return
    if a.cached and (out / 'pseudobulk.npz').exists():
        f = np.load(out / 'pseudobulk.npz', allow_pickle=True)
        PB, genes, meta = f['counts'], f['genes'].astype(str), pd.read_csv(out / 'pseudobulk_meta.csv')
    else:
        PB, meta, genes = pseudobulk(a)
        np.savez_compressed(out / 'pseudobulk.npz', counts=PB, genes=genes)
        meta.to_csv(out / 'pseudobulk_meta.csv', index=False)
    if 'cluster' in want:
        c = stage_cluster(PB, meta, genes, a)
        c.to_csv(out / 'rna_clustering.csv', index=False)
        log.info('[cluster]\n%s', c[['celltype', 'clusters', 'ari', 'p']].to_string(index=False))
    if 'donorset' in want:
        T, R = stage_donorset(PB, meta, genes, a)
        T.to_csv(out / 'rna_donorset_z.csv.gz', index=False)
        R.to_csv(out / 'rna_donorset_replication.csv', index=False)
        for t, y in R.groupby('celltype'):
            p = y[y.condition == 'PMD']
            for tail in ('rep_down', 'rep_up'):
                top = p[p[tail] > 0].nlargest(8, tail)
                log.info('[donorset] %s %s: %d PMD sets >= 50%% (non-PMD %d) | %s', t, tail,
                         int((p[tail] >= 0.5).sum()), int((y[y.condition != 'PMD'][tail] >= 0.5).sum()),
                         '; '.join(f'{s[:34]} {v:.2f}' for s, v in zip(top.set, top[tail])))
    if 'scde' in want:
        stage_scde(a).to_csv(out / 'rna_scde_set_z.csv.gz', index=False)
    if 'shift' in want:
        z = stage_shift(PB, meta, genes, a)
        z.to_csv(out / 'rna_donor_set_z.csv.gz', index=False)
        log.info('[shift] %d donor x cell type x set rows', len(z))
    if 'deseq' in want:
        d = stage_deseq(PB, meta, genes, a)
        d.to_csv(out / 'rna_deseq_counts.csv', index=False)
        log.info('[deseq]\n%s', d.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--query', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--modules', required=True)
    ap.add_argument('--stages', default='cluster,donorset,deseq', help='+ shift (per-donor set z, ExPert ctrl design)')
    ap.add_argument('--cached', action='store_true', help='reuse out-dir/pseudobulk.npz instead of re-reading --query')
    ap.add_argument('--celltype-col', default='singler_labels_main')
    ap.add_argument('--donor-col', default='patient_id')
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--n-hvg', type=int, default=2000)
    ap.add_argument('--mask', default='MT-*,MT1*,MT2A,MT3,MT4', help='scde: genes dropped (the ExPert input mask)')
    ap.add_argument('--min-frac', type=float, default=0.01, help='scde: genes detected in >= this share of cells')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--padj', type=float, default=0.1)
    ap.add_argument('--n-cpus', type=int, default=8)
    main(ap.parse_args())
