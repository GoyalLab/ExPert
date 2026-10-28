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

"""Fig. 1d data: geometry of the learned perturbation prototypes (atoms) and their agreement
with external priors.

    atoms       expert.analysis.atom_umap.load_atoms -- the live block of the production
                dictionary exactly as the readout scores it
    layout      the model scores COSINE, so the geometry lives on the unit sphere. Cosine to the
                zero vector is undefined; the origin-anchored analogue is the MEAN ATOM DIRECTION
                m. radius = angle(a_c, m) in degrees (exact), azimuth = direction of the residual
                a_c - (a_c . m) m in its first two principal components (an azimuthal-equidistant
                view around m). A cosine UMAP is written alongside for comparison.
    categories  Reactome top-level category per gene (hierarchy + GMT downloaded from reactome.org)
    agreement   per prior: kNN overlap enrichment (k-NN in atom cosine vs k-NN in prior cosine,
                over chance k/(n-1)) and Spearman of pairwise cosines on random pairs; every prior
                also scored with its gene labels permuted (same marginal structure, no
                correspondence). Computed on ALL atoms and on atoms the prior term never PULLED,
                because text / DepMap / STRING were training inputs.

Writes <out>/fig1d_layout.parquet, <out>/fig1e_agreement.csv, <out>/fig1e_meta.json.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from expert.analysis.atom_umap import load_atoms
from expert.utils.prior_graph import load_prior_matrix

REACTOME = 'https://reactome.org/download/current/'
# top-level roots that describe a condition or a generic umbrella rather than a process
SKIP_ROOTS = {'Disease', 'Drug ADME'}


def fetch(url: str, cache: Path) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
    cache.write_bytes(b)
    return b


def reactome_categories(genes: list[str], cache_dir: Path) -> tuple[pd.Series, dict]:
    """gene -> Reactome top-level category, plus the gene-set dict (leaf pathways, human).

    A gene sits in many pathways; each membership votes for its pathway's root with weight
    1/|pathway|, so a gene's category is decided by its most SPECIFIC annotations rather than by
    the large umbrella sets every gene touches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rel = pd.read_csv(io.BytesIO(fetch(REACTOME + 'ReactomePathwaysRelation.txt',
                                       cache_dir / 'ReactomePathwaysRelation.txt')),
                      sep='\t', header=None, names=['parent', 'child'])
    names = pd.read_csv(io.BytesIO(fetch(REACTOME + 'ReactomePathways.txt',
                                         cache_dir / 'ReactomePathways.txt')),
                        sep='\t', header=None, names=['id', 'name', 'species'])
    names = names[names.species == 'Homo sapiens'].set_index('id').name
    zb = fetch(REACTOME + 'ReactomePathways.gmt.zip', cache_dir / 'ReactomePathways.gmt.zip')
    gmt = zipfile.ZipFile(io.BytesIO(zb)).read('ReactomePathways.gmt').decode()
    sets = {}
    for line in gmt.splitlines():
        p = line.split('\t')
        if len(p) > 2:
            sets[p[1]] = set(p[2:])                 # key = stable ID
    parent = rel[rel.child.isin(names.index)].groupby('child').parent.first().to_dict()

    def root(pid):
        seen = set()
        while pid in parent and pid not in seen:
            seen.add(pid)
            pid = parent[pid]
        return pid
    roots = {pid: names.get(root(pid), None) for pid in sets}
    votes: dict[str, dict[str, float]] = {}
    for pid, gs in sets.items():
        r = roots.get(pid)
        if r is None or r in SKIP_ROOTS or len(gs) < 5:
            continue
        w = 1.0 / len(gs)
        for g in gs:
            votes.setdefault(g, {}).setdefault(r, 0.0)
            votes[g][r] += w
    cat = pd.Series({g: max(v, key=v.get) for g, v in votes.items()})
    return cat.reindex(genes), {names.get(k, k): v for k, v in sets.items()}


def go_bp_prior(genes: list[str], out: Path) -> Path:
    """GO BP membership as an embedding (gene x term, binary): cosine = Ochiai overlap."""
    import gseapy
    lib = gseapy.get_library('GO_Biological_Process_2023', organism='Human')
    lib = {t: set(g) for t, g in lib.items() if 5 <= len(g) <= 500}
    terms = sorted(lib)
    gi = {g: i for i, g in enumerate(sorted(set().union(*lib.values())))}
    E = np.zeros((len(gi), len(terms)), np.float32)
    for j, t in enumerate(terms):
        E[[gi[g] for g in lib[t]], j] = 1.0
    np.savez(out, embeddings=E, genes=np.array(list(gi), dtype=object))
    return out


def cluster_and_annotate(C: torch.Tensor, labels: list[str], sets: dict, k: int,
                         resolution: float, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Leiden on the atoms' own cosine kNN graph, then each cluster's most enriched Reactome
    pathway (any level, 10-500 genes; hypergeometric over the annotated atoms; BH across ALL
    cluster x pathway tests). The names describe the geometry rather than impose on it."""
    import igraph as ig
    import leidenalg
    from scipy.stats import hypergeom
    S = C.clone(); S.fill_diagonal_(-np.inf)
    nb = S.topk(k, dim=1).indices.numpy()
    edges = {(min(i, j), max(i, j)) for i in range(len(nb)) for j in nb[i]}
    g = ig.Graph(n=len(labels), edges=list(edges))
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed)
    cl = np.asarray(part.membership)
    annotated = set().union(*[v for v in sets.values() if 10 <= len(v) <= 500])
    lab_arr = np.asarray(labels)
    bg = [x for x in labels if x in annotated]
    N = len(bg)
    bgset = set(bg)
    tests = []
    for c in np.unique(cl):
        mem = set(lab_arr[cl == c]) & bgset
        if len(mem) < 10:
            continue
        for name, gs in sets.items():
            if not 10 <= len(gs) <= 500:
                continue
            K = len(gs & bgset)
            x = len(gs & mem)
            if x < 3:
                continue
            p = hypergeom.sf(x - 1, N, K, len(mem))
            tests.append((int(c), name, x, len(mem), K, p))
    t = pd.DataFrame(tests, columns=['cluster', 'pathway', 'hits', 'n_annot', 'set_size', 'p'])
    o = np.argsort(t.p.to_numpy())
    q = t.p.to_numpy()[o] * len(t) / np.arange(1, len(t) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    t.loc[t.index[o], 'q'] = np.clip(q, 0, 1)
    t['frac'] = t.hits / t.n_annot
    return cl, t.sort_values(['cluster', 'p'])


def knn(S: torch.Tensor, idx: np.ndarray, k: int) -> torch.Tensor:
    sub = S[np.ix_(idx, idx)].clone()
    sub.fill_diagonal_(-np.inf)
    return sub.topk(k, dim=1).indices


def agreement(C: torch.Tensor, P: torch.Tensor, idx: np.ndarray, k: int, n_pairs: int,
              rng) -> dict:
    ka, kp = knn(C, idx, k), knn(P, idx, k)
    ov = (ka.unsqueeze(2) == kp.unsqueeze(1)).any(2).float().sum(1) / k
    chance = k / (len(idx) - 1)
    i = rng.integers(0, len(idx), n_pairs)
    j = rng.integers(0, len(idx), n_pairs)
    keep = i != j
    a = C[idx[i[keep]], idx[j[keep]]].numpy()
    b = P[idx[i[keep]], idx[j[keep]]].numpy()
    from scipy.stats import spearmanr
    return {'knn_overlap': float(ov.mean()), 'knn_enrich': float(ov.mean() / chance),
            'rsa_spearman': float(spearmanr(a, b).statistic), 'n': int(len(idx))}


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    labels, A, note = load_atoms(a.model_pt)
    print(f'[atoms] {A.shape} {note or ""}', flush=True)
    labels = [str(x) for x in labels]
    U = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    C = torch.from_numpy(U @ U.T).float()

    # ---- layout ------------------------------------------------------------------------------
    m = U.mean(0); m /= np.linalg.norm(m)
    cm = U @ m
    radius = np.degrees(np.arccos(np.clip(cm, -1, 1)))
    R = U - cm[:, None] * m[None]
    R /= np.clip(np.linalg.norm(R, axis=1, keepdims=True), 1e-12, None)
    Rc = R - R.mean(0)
    _, sv, Vt = np.linalg.svd(Rc, full_matrices=False)
    pc = Rc @ Vt[:2].T
    azim = np.degrees(np.arctan2(pc[:, 1], pc[:, 0]))
    import umap
    um = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=0).fit_transform(U)
    print(f'[layout] cos to mean direction: median {np.median(cm):.3f}; radius '
          f'{radius.min():.1f}-{radius.max():.1f} deg; residual PC1/2 var '
          f'{(sv[:2] ** 2 / (sv ** 2).sum()).round(4).tolist()}', flush=True)

    cat, sets = reactome_categories(labels, Path(a.cache_dir))
    per = pd.read_parquet(a.perts).set_index('perturbation')
    pulled = per.w_hard.reindex(labels).fillna(0).to_numpy() > 0
    lay = pd.DataFrame({'perturbation': labels, 'radius_deg': radius, 'azimuth_deg': azim,
                        'cos_mean_dir': cm, 'umap1': um[:, 0], 'umap2': um[:, 1],
                        'category': cat.to_numpy(), 'pulled': pulled,
                        'norm': np.linalg.norm(A, axis=1),
                        'n_train': per.n_train.reindex(labels).to_numpy(),
                        'evidence': per.evidence.reindex(labels).to_numpy()})
    cl, enr = cluster_and_annotate(C, labels, sets, a.cluster_k, a.resolution)
    lay['cluster'] = cl
    enr.to_csv(out / 'fig1d_cluster_enrichment.csv', index=False)
    best = enr[enr.q < 0.05].groupby('cluster').head(1)
    print(f'[clusters] {len(np.unique(cl))} Leiden clusters, {len(best)} with a Reactome '
          f'pathway at q<0.05', flush=True)
    for _, r in best.sort_values('p').head(25).iterrows():
        print(f'   c{r.cluster:<3d} n={int((cl == r.cluster).sum()):4d} q={r.q:.1e} '
              f'{r.hits}/{r.n_annot} {r.pathway}', flush=True)
    lay.to_parquet(out / 'fig1d_layout.parquet', index=False)
    print('[categories]', lay.category.value_counts(dropna=False).head(15).to_dict(), flush=True)

    # ---- agreement ---------------------------------------------------------------------------
    priors = dict(p.split('=', 1) for p in a.prior)
    priors['GO BP'] = str(go_bp_prior(labels, Path(a.cache_dir) / 'go_bp_2023.npz'))
    train_priors = set(a.train_prior or [])
    rows = []
    rng = np.random.default_rng(0)
    for name, path in priors.items():
        P, have, why = load_prior_matrix(path, labels)
        if P is None:
            print(f'[skip] {name}: {why}', flush=True); continue
        have = have.numpy()
        for subset, mask in (('all', have), ('not pulled', have & ~pulled)):
            idx = np.flatnonzero(mask)
            r = agreement(C, P, idx, a.k, a.n_pairs, rng)
            nulls = []
            for s in range(a.n_null):
                Ps, _, _ = load_prior_matrix(path, labels, shuffle=True, seed=s)
                nulls.append(agreement(C, Ps, idx, a.k, a.n_pairs, rng))
                del Ps
            for met in ('knn_enrich', 'knn_overlap', 'rsa_spearman'):
                nv = np.array([x[met] for x in nulls])
                rows.append({'prior': name, 'subset': subset, 'metric': met, 'value': r[met],
                             'null_mean': float(nv.mean()), 'null_sd': float(nv.std(ddof=1)),
                             'n': r['n'], 'used_in_training': name in train_priors})
            print(f'[{name:14s}] {subset:10s} n={r["n"]:5d} kNN x{r["knn_enrich"]:6.2f} '
                  f'(null x{np.mean([x["knn_enrich"] for x in nulls]):.2f})  rsa '
                  f'{r["rsa_spearman"]:+.3f} (null {np.mean([x["rsa_spearman"] for x in nulls]):+.3f})',
                  flush=True)
        del P
    pd.DataFrame(rows).to_csv(out / 'fig1e_agreement.csv', index=False)
    json.dump({'n_atoms': len(labels), 'dim': int(A.shape[1]), 'k': a.k, 'n_pairs': a.n_pairs,
               'n_null': a.n_null, 'n_pulled': int(pulled.sum()), 'atom_note': note,
               'model_pt': a.model_pt, 'priors': priors},
              open(out / 'fig1e_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-pt', required=True)
    ap.add_argument('--perts', required=True, help='fig1c_perts.parquet (pulled flag, evidence)')
    ap.add_argument('--prior', action='append', required=True, help='NAME=path.npz, repeatable')
    ap.add_argument('--train-prior', action='append', help='NAME of a prior the model trained on')
    ap.add_argument('--cache-dir', required=True, help='downloaded Reactome / GO files')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--cluster-k', type=int, default=15)
    ap.add_argument('--resolution', type=float, default=1.0)
    ap.add_argument('--n-pairs', type=int, default=300000)
    ap.add_argument('--n-null', type=int, default=3)
    main(ap.parse_args())
