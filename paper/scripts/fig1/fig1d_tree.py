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

"""Fig. 1d tree: Ward hierarchy of the trained prototypes, clades annotated over several
gene-set resources, and a check of whether the "diffuse" share is the clustering's fault.

    tree        Ward linkage on unit prototypes (Euclidean on the sphere is monotone in cosine),
                the same construction as Fig. 3b but over the 8,964 TRAINED rows only
    libraries   Enrichr libraries via gseapy (GO BP, GO CC, KEGG, Reactome, CORUM, Hallmark)
    annotation  per cluster x gene set: hypergeometric over the prototypes that library annotates
                (background = atoms in the library's universe); BH within each clustering. A
                cluster's label is its smallest-q term over ALL libraries.
    coverage    share of prototypes in a WELL-annotated cluster (q < 0.05 and the term covers
                >= --min-frac of the cluster's annotated members), for Ward cuts and for Leiden
                at several resolutions -- the test of "is the diffuse mass a clustering artefact"

Writes <out>/fig1d_tree.npz, fig1d_tree_clusters.parquet, fig1d_clade_annotation.csv,
fig1d_clade_terms.csv (all passing terms per cluster, top 25),
fig1d_clustering_coverage.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

from expert.analysis.atom_umap import load_atoms

LIBS = {'GO BP': 'GO_Biological_Process_2023', 'GO CC': 'GO_Cellular_Component_2023',
        'KEGG': 'KEGG_2021_Human', 'Reactome': 'Reactome_2022', 'CORUM': 'CORUM',
        'Hallmark': 'MSigDB_Hallmark_2020'}


def load_libraries(labels: list[str], cache: Path, min_size: int, max_size: int) -> dict:
    """name -> (term list, sparse gene x term over `labels`, universe mask over `labels`)."""
    import gseapy
    cache.mkdir(parents=True, exist_ok=True)
    gi = {g: i for i, g in enumerate(labels)}
    out = {}
    for short, lib in LIBS.items():
        f = cache / f'{lib}.gmt'
        if f.exists():
            sets = {l.split('\t')[0]: l.rstrip('\n').split('\t')[2:]
                    for l in f.read_text().splitlines() if l}
        else:
            sets = gseapy.get_library(lib, organism='Human')
            f.write_text(''.join(f"{t}\t\t" + '\t'.join(g) + '\n' for t, g in sets.items()))
        terms, rows, cols = [], [], []
        for t, gs in sets.items():
            idx = [gi[g] for g in set(gs) if g in gi]
            if min_size <= len(idx) <= max_size:
                rows += idx; cols += [len(terms)] * len(idx); terms.append(t)
        S = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(len(labels), len(terms)))
        uni = np.asarray(S.sum(1)).ravel() > 0
        out[short] = (terms, S, uni)
        print(f'[lib] {short:8s} {len(terms):5d} terms, {int(uni.sum()):5d} prototypes annotated',
              flush=True)
    return out


def annotate(cl: np.ndarray, libs: dict, min_frac: float) -> pd.DataFrame:
    """Best term per cluster across libraries; q from BH over every test in this clustering."""
    ids = np.unique(cl)
    C = sp.csr_matrix((np.ones(len(cl), np.float32), (np.arange(len(cl)), np.searchsorted(ids, cl))),
                      shape=(len(cl), len(ids)))
    recs = []
    for lib, (terms, S, uni) in libs.items():
        Cu = C[uni]                                   # background = this library's universe
        Su = S[uni]
        N = int(uni.sum())
        n_c = np.asarray(Cu.sum(0)).ravel()           # annotated members per cluster
        K_t = np.asarray(Su.sum(0)).ravel()           # set size within the universe
        X = (Cu.T @ Su).toarray()                     # hits: clusters x terms
        ci, ti = np.nonzero(X >= 3)
        p = hypergeom.sf(X[ci, ti] - 1, N, K_t[ti], n_c[ci])
        recs.append(pd.DataFrame({'cluster': ids[ci], 'library': lib,
                                  'term': np.asarray(terms, dtype=object)[ti],
                                  'hits': X[ci, ti].astype(int), 'n_annot': n_c[ci].astype(int),
                                  'set_size': K_t[ti].astype(int), 'p': p}))
    t = pd.concat(recs, ignore_index=True)
    o = np.argsort(t.p.to_numpy())
    m = len(t)
    q = t.p.to_numpy()[o] * m / np.arange(1, m + 1)
    t.loc[t.index[o], 'q'] = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    t['frac'] = t.hits / t.n_annot
    size = pd.Series(cl).value_counts()
    best = (t[(t.q < 0.05) & (t.frac >= min_frac)].sort_values('p')
            .groupby('cluster').head(1).set_index('cluster'))
    out = pd.DataFrame({'cluster': ids, 'size': size.reindex(ids).to_numpy()}).set_index('cluster')
    out = out.join(best[['library', 'term', 'hits', 'n_annot', 'set_size', 'p', 'q', 'frac']])
    # every passing term per cluster (top 25 by p), so labels can fall back to the next-best
    # term when two clusters share the same best one
    terms = (t[(t.q < 0.05) & (t.frac >= min_frac)].sort_values('p')
             .groupby('cluster').head(25))
    return out.reset_index(), terms


def leiden(U: np.ndarray, k: int, res: float, seed: int = 0) -> np.ndarray:
    import igraph as ig
    import leidenalg
    import torch
    S = torch.from_numpy(U @ U.T); S.fill_diagonal_(-np.inf)
    nb = S.topk(k, dim=1).indices.numpy()
    edges = {(min(i, j), max(i, j)) for i in range(len(nb)) for j in nb[i]}
    g = ig.Graph(n=len(U), edges=list(edges))
    return np.asarray(leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed).membership)


def response_strength(db: str, labels: list[str]) -> np.ndarray:
    """Per perturbation: median within-dataset percentile of |efficiency| over its cells --
    the same quantity the production |eff| gate ranks on. Low = weak/no transcriptional
    response to the knockdown."""
    import h5py
    with h5py.File(db, 'r') as f:
        eff = np.abs(f['obs/efficiency'][:].astype(np.float32))
        ds = f['obs/dataset'].asstr()[:]
        pert = f['obs/perturbation'].asstr()[:]
    ok = np.isfinite(eff)
    eff, ds, pert = eff[ok], ds[ok], pert[ok]
    rk = np.empty_like(eff)
    for d in np.unique(ds):
        m = ds == d
        o = np.argsort(eff[m], kind='stable')
        r = np.empty(int(m.sum()), np.float32); r[o] = np.arange(m.sum()) / max(m.sum() - 1, 1)
        rk[m] = r
    med = pd.Series(rk).groupby(pert).median()
    return med.reindex(labels).to_numpy()


def main(a):
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels, A, _ = load_atoms(a.model_pt)
    labels = [str(x) for x in labels]
    U = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Z = linkage(U.astype(np.float64), method='ward')
    np.savez_compressed(out / 'fig1d_tree.npz', Z=Z, order=leaves_list(Z),
                        labels=np.asarray(labels, dtype=object))
    print('[tree] Ward linkage done', flush=True)
    libs = load_libraries(labels, Path(a.cache_dir), a.min_size, a.max_size)

    clusterings = {f'ward_{k}': fcluster(Z, k, criterion='maxclust') for k in a.ward_k}
    clusterings.update({f'leiden_{r:g}': leiden(U, 15, r) for r in a.leiden_res})
    cov, ann, all_terms = [], [], []
    for name, cl in clusterings.items():
        an, terms = annotate(cl, libs, a.min_frac)
        all_terms.append(terms.assign(clustering=name))
        good = an.dropna(subset=['term'])
        share = good['size'].sum() / len(cl)
        cov.append({'clustering': name, 'n_clusters': len(an), 'n_annotated': len(good),
                    'share_atoms_annotated': share,
                    'largest_unannotated': int(an.loc[an.term.isna(), 'size'].max()
                                               if an.term.isna().any() else 0)})
        print(f'[{name:11s}] {len(an):4d} clusters, {len(good):4d} annotated, '
              f'{share:.1%} of prototypes in an annotated cluster; libraries: '
              f'{good.library.value_counts().to_dict()}', flush=True)
        ann.append(an.assign(clustering=name))
    pd.DataFrame(cov).to_csv(out / 'fig1d_clustering_coverage.csv', index=False)
    pd.concat(ann).to_csv(out / 'fig1d_clade_annotation.csv', index=False)
    pd.concat(all_terms).to_csv(out / 'fig1d_clade_terms.csv', index=False)
    extra = {}
    if a.db:
        extra['response_q'] = response_strength(a.db, labels)
        print(f"[response] median |eff| percentile per perturbation: "
              f"{np.nanmedian(extra['response_q']):.3f}", flush=True)
    pd.DataFrame({'perturbation': labels, **clusterings, **extra}).to_parquet(
        out / 'fig1d_tree_clusters.parquet', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-pt', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--db', help='database_perturbed.h5ad (obs efficiency) for response strength')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ward-k', type=int, nargs='+', default=[20, 40, 60, 100, 150])
    ap.add_argument('--leiden-res', type=float, nargs='+', default=[1.0, 2.0, 4.0])
    ap.add_argument('--min-frac', type=float, default=0.2)
    ap.add_argument('--min-size', type=int, default=5)
    ap.add_argument('--max-size', type=int, default=500)
    main(ap.parse_args())
