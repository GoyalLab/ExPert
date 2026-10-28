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

"""Which prototypes sit BETWEEN drugs in the atom graph?

Model-independent: the trained prototype table and the drug -> target annotations only, no drug
responses. A drug's targets are terminals in the bank's own k-nearest-neighbour graph; for every
pair of drugs we take the shortest path between their target sets and count the genes the path
runs through. A gene that many drug pairs route through is a bridge.

    graph       each prototype joined to its --knn nearest by cosine, symmetrised,
                edge length = 1 - cos (so a short edge is a close pair)
    traversal   per gene, the number of drug pairs whose shortest path passes through it
                (endpoints and either drug's own targets excluded)
    null        the same count with each drug's target set replaced by a size- and
                popularity-matched random set -- a hub is traversed often by construction, so
                only the EXCESS over this null says a gene connects these drugs in particular
    cross-MoA   of a bridge's pairs, how many join drugs with different annotated mechanisms

Writes <prefix>_bridges.csv (per gene), <prefix>_bridge_pairs.csv (the drug pairs behind the top
bridges), <prefix>_bridge_graph_{nodes,edges}.csv (paths through the top bridge) and _meta.json.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

log = logging.getLogger("fig4_bridges")


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def gmt_similarity(path, L, min_size=3, max_size=500):
    """Gene-gene similarity from a GMT library: cosine over the binary gene x term matrix.

    The question this answers is whether a bridge is a property of the LEARNED prototype geometry
    or is already implied by generic annotation. Terms are size-filtered the same way for both.
    """
    pos = {g: i for i, g in enumerate(L)}
    rows, cols, t = [], [], 0
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        gs = {g.split(",")[0].strip() for g in f[2:] if g.strip()}
        gs = [pos[g] for g in gs if g in pos]
        if min_size <= len(gs) <= max_size:
            rows += gs; cols += [t] * len(gs); t += 1
    M = coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(L), t)).tocsr()
    nrm = np.sqrt(M.multiply(M).sum(1).A.ravel())
    cov = float((nrm > 0).mean())
    M = M.multiply(1.0 / np.maximum(nrm, 1e-9)[:, None]).tocsr()
    S = np.asarray((M @ M.T).todense(), np.float32)
    np.fill_diagonal(S, 1.0)
    return S, t, cov


def npz_similarity(path, L):
    """Gene-gene cosine from a prior embedding (.npz with 'genes' and 'embeddings').

    Genes the prior does not cover get no edges at all rather than spurious zero-cosine ones.
    """
    z = np.load(path, allow_pickle=True)
    g2i = {str(g): i for i, g in enumerate(z["genes"])}
    E = np.asarray(z["embeddings"], np.float32)
    M = np.zeros((len(L), E.shape[1]), np.float32)
    ok = np.zeros(len(L), bool)
    for i, g in enumerate(L):
        j = g2i.get(g)
        if j is not None:
            M[i] = E[j]; ok[i] = True
    ok &= np.linalg.norm(M, axis=1) > 0
    S = unit(M) @ unit(M).T
    S[~ok] = -1.0; S[:, ~ok] = -1.0
    np.fill_diagonal(S, 1.0)
    return S, int(E.shape[1]), float(ok.mean())


def knn_graph(G, k):
    """Symmetric kNN graph on the prototype-prototype cosine, edge length = 1 - cos."""
    n = len(G)
    nb = np.argsort(-G, axis=1)[:, 1:k + 1]
    r = np.repeat(np.arange(n), k)
    c = nb.ravel()
    keep = G[r, c] > -1.0                      # a prior that does not cover a gene gives it no edge
    r, c = r[keep], c[keep]
    w = np.maximum(1.0 - G[r, c], 1e-3)
    A = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    A = A.minimum(A.T) + A.maximum(A.T) - A.minimum(A.T)      # symmetrise, keep one weight
    A = A.maximum(A.T)
    return A.tocsr()


def traversal(A, sets, n_nodes, moa=None, keep_pairs=False):
    """Count, per gene, the drug pairs whose shortest path runs through it."""
    cnt = np.zeros(n_nodes, np.int64)
    cross = np.zeros(n_nodes, np.int64)
    rows = []
    D, P = {}, {}
    for i, s in enumerate(sets):
        d, pred, _ = dijkstra(A, indices=s, return_predecessors=True, min_only=True)
        D[i], P[i] = d, pred
    for i in range(len(sets)):
        own_i = set(sets[i])
        for j in range(i + 1, len(sets)):
            tgt = min(sets[j], key=lambda t: D[i][t])
            if not np.isfinite(D[i][tgt]):
                continue
            path, n_ = [tgt], tgt
            while P[i][n_] >= 0:
                n_ = P[i][n_]
                path.append(n_)
            path = path[::-1]                     # drug i's target first, drug j's target last
            mid = [p for p in path[1:-1] if p not in own_i and p not in set(sets[j])]
            if not mid:
                continue
            x = moa is not None and moa[i] != moa[j]
            for p in mid:
                cnt[p] += 1
                cross[p] += int(x)
            if keep_pairs:
                rows.append(dict(i=i, j=j, dist=float(D[i][tgt]), n_mid=len(mid),
                                 path=";".join(map(str, path)), cross=bool(x)))
    return cnt, cross, rows


def main(a):
    rng = np.random.default_rng(a.seed)
    b = np.load(a.bank, allow_pickle=True)
    nl = int(b["n_labels"]); L = b["labels"].astype(str)[:nl]
    P = unit(b["P"].astype(np.float32))[:nl]
    G = P @ P.T
    pos = {g: i for i, g in enumerate(L)}
    A = knn_graph(G, a.knn)
    deg = np.diff(A.indptr)
    log.info("kNN graph: %d nodes, %d edges (k=%d)", nl, A.nnz // 2, a.knn)

    m = pd.read_csv(a.drug_meta)
    sp = lambda s: [t.strip() for t in str(s).replace(",", ";").split(";")
                    if t.strip() and t.strip() != "nan"]
    tg = {r.drug: sorted({pos[g] for g in sp(r.targets) if g in pos}) for r in m.itertuples()}
    tg = {d: t for d, t in tg.items() if t}
    moa_of = dict(zip(m.drug, m.moa_fine.astype(str))) if "moa_fine" in m else {}
    drugs = sorted(tg)
    sets = [np.array(tg[d], int) for d in drugs]
    moa = [moa_of.get(d, "") for d in drugs]
    log.info("%d drugs with >=1 target in the bank (%d distinct mechanisms)",
             len(drugs), len(set(moa)))

    cnt, cross, pairs = traversal(A, sets, nl, moa, keep_pairs=True)
    n_pair = len(drugs) * (len(drugs) - 1) // 2
    log.info("observed: %d of %d drug pairs route through >=1 intermediate",
             len(pairs), n_pair)

    def matched_null(A_, G_, n_perm):
        """Traversal expected when each drug keeps its number of targets but draws them from its own
        popularity bin -- this is what makes a graph hub's traversal uninteresting."""
        pop_ = (G_.sum(1) - 1) / (len(G_) - 1)
        bin_ = np.clip((pop_.argsort().argsort() * a.n_bins) // len(G_), 0, a.n_bins - 1)
        bins_ = [np.flatnonzero(bin_ == j) for j in range(a.n_bins)]
        out = np.zeros((n_perm, nl), np.int64)
        for p_ in range(n_perm):
            sh = [np.array([rng.choice(bins_[bin_[i]]) for i in s_], int) for s_ in sets]
            out[p_] = traversal(A_, sh, nl)[0]
        return out, pop_

    null, pop = matched_null(A, G, a.n_perm)
    exp, sd = null.mean(0), null.std(0)
    z = (cnt - exp) / np.maximum(sd, 1.0)
    p_emp = ((null >= cnt[None]).sum(0) + 1) / (a.n_perm + 1)
    t = pd.DataFrame(dict(gene=L, n_pairs=cnt, n_cross_moa=cross, expected=exp, sd=sd, z=z,
                          p=p_emp, degree=deg, popularity=pop))
    t["excess"] = t.n_pairs - t.expected
    t["fold"] = t.n_pairs / np.maximum(t.expected, 0.5)
    t = t.sort_values("excess", ascending=False)
    t.to_csv(f"{a.out_prefix}_bridges.csv", index=False)
    # A graph hub is traversed often by construction AND in the null, so it survives the excess but
    # not the fold: require both, and the list becomes low-degree genes that specific drug pairs
    # route through.
    top = t[(t.n_pairs >= a.min_pairs) & (t.fold >= a.min_fold)].head(a.top_n)
    log.info("top bridges (excess over the matched null, fold >= %.1f, >= %d pairs):\n%s",
             a.min_fold, a.min_pairs,
             top[["gene", "n_pairs", "expected", "fold", "z", "p", "n_cross_moa", "degree"]]
             .to_string(index=False))

    # the drug pairs behind the top bridges, and the path subgraph through the named bridges
    pr = pd.DataFrame(pairs)
    if len(pr):
        pr["a"] = [drugs[i] for i in pr.i]; pr["b"] = [drugs[j] for j in pr.j]
        pr["moa_a"] = [moa[i] for i in pr.i]; pr["moa_b"] = [moa[j] for j in pr.j]
        pr["mid"] = [";".join(L[[int(x) for x in s.split(";")][1:-1]]) for s in pr.path]
        keep = set(top.gene)
        sel = pr[[bool(keep & set(s.split(";"))) for s in pr.mid]]
        sel.drop(columns=["path"]).to_csv(f"{a.out_prefix}_bridge_pairs.csv", index=False)
        log.info("%d drug pairs route through a top-%d bridge", len(sel), len(top))
        for g in top.gene:
            q = pr[[g in s.split(";") for s in pr.mid]]
            mo = (q.assign(k=[" / ".join(sorted({x, y})) for x, y in zip(q.moa_a, q.moa_b)])
                  .k.value_counts().head(3))
            log.info("  %-8s %4d pairs, %3d cross-mechanism | %s", g, len(q), int(q.cross.sum()),
                     "; ".join(f"{k_} ({v})" for k_, v in mo.items()))

        want = ([x.strip() for x in a.graph_genes.split(",") if x.strip()]
                if a.graph_genes.strip().lower() not in ("", "all") else list(top.gene))
        nbrs = np.argsort(-G, axis=1)[:, 1:a.knn + 1]
        nodes, edges = {}, {}
        for g in want:
            if g not in pos:
                log.warning("graph gene %r not in the bank", g); continue
            gi = pos[g]
            # the drug targets this bridge connects directly (it is the only node on the path);
            # some bridges are never the SOLE intermediate, so fall back to every path through it
            q = pr[pr.mid == g]
            sole = bool(len(q))
            if not sole:
                q = pr[[g in str(x).split(";") for x in pr.mid]]
            if not len(q):
                log.warning("graph gene %r is on no drug path", g); continue
            own = {}
            for r in q.itertuples():
                ip = [int(x) for x in r.path.split(";")]
                own.setdefault(ip[0], []).append((r.a, r.moa_a))
                own.setdefault(ip[-1], []).append((r.b, r.moa_b))
            # Which two mechanisms does this bridge actually join? Take the modal mechanism PAIR
            # over its paths, then the busiest endpoint carrying each side. Labelling a side by an
            # endpoint's own modal mechanism goes wrong for genes hit by promiscuous drugs.
            bad = ("", "nan", "unclear")
            mp = pd.Series([" | ".join(sorted((x.moa_a, x.moa_b))) for x in q.itertuples()
                            if x.moa_a not in bad and x.moa_b not in bad
                            and x.moa_a != x.moa_b]).value_counts()
            m0, m1 = (mp.index[0].split(" | ") if len(mp) else ("", ""))
            mc = {k_: pd.Series([m_ for _, m_ in v if m_ not in bad]).value_counts()
                  for k_, v in own.items()}
            side = {}
            for sd, mm in ((0, m0), (1, m1)):
                cand = [k_ for k_ in own if mm and mc[k_].get(mm, 0) > 0 and k_ not in side]
                if cand:
                    side[max(cand, key=lambda k_: (mc[k_].get(mm, 0), len(own[k_])))] = (sd, mm)
            ends = (sorted(side, key=lambda k_: side[k_][0])
                    + sorted((k_ for k_ in own if k_ not in side),
                             key=lambda k_: -len(own[k_])))[:a.graph_endpoints]
            # ... plus the neighbourhood each of them sits in, so the picture shows the structure
            # around the bridge and not only the three nodes on the path
            keep = {gi, *ends}
            for i in [gi, *ends]:
                keep.update(nbrs[i, :a.graph_knn].tolist())   # fewer than the graph's own k: the
                #                                               example panel has to stay readable
            keep = sorted(keep)
            for i in keep:
                role = "bridge" if i == gi else "endpoint" if i in own else "neighbour"
                dr = sorted({x for x, _ in own.get(i, [])})
                # "unclear" is a hole in Tahoe's own annotation, not a mechanism: never label a
                # side with it while a real mechanism is on the same endpoint
                mo = pd.Series([m_ for _, m_ in own.get(i, [])
                                if m_ not in ("", "nan", "unclear")]).value_counts()
                nodes[(g, i)] = dict(gene=L[i], role=role, bridge=g,
                                     drugs=";".join(dr[:a.graph_drugs]), n_drugs=len(dr),
                                     moa=(side[i][1] if i in side else
                                          (mo.index[0] if len(mo) else "")),
                                     side=side[i][0] if i in side else -1,
                                     n_pairs=len(q) if i == gi else len(own.get(i, [])),
                                     on_path=int(i in own or i == gi), sole=int(sole))
            ks = set(keep)
            for i in keep:
                for j in nbrs[i]:
                    if j in ks and i < j:
                        edges[(g, i, int(j))] = float(G[i, j])
                    elif j in ks and i > j:
                        edges[(g, int(j), i)] = float(G[i, j])
            log.info("  graph %s: %d nodes (%d endpoints), %d edges, %s on %d pairs | sides: %s",
                     g, len(ks), len(ends), sum(1 for k_ in edges if k_[0] == g),
                     "sole intermediate" if sole else "one of several intermediates", len(q),
                     " <-> ".join(f"{L[k_]} ({v[1]})" for k_, v in sorted(side.items(),
                                                                         key=lambda z: z[1][0])))
        nd = pd.DataFrame(list(nodes.values()))
        ed = pd.DataFrame([dict(bridge=g, a=L[x_], b=L[y_], cos=c)
                           for (g, x_, y_), c in edges.items()])
        nd.to_csv(f"{a.out_prefix}_bridge_graph_nodes.csv", index=False)
        ed.to_csv(f"{a.out_prefix}_bridge_graph_edges.csv", index=False)
        log.info("bridge subgraph: %d nodes, %d edges over %s", len(nd), len(ed), want)

    # ---- is the bridge already implied by a prior we have lying around? -------------------------
    # Same graph construction, same traversal, same matched-random null -- only the gene-gene
    # similarity changes. A bridge that is a property of the LEARNED geometry should not reappear.
    sources = []
    if a.annotation_gmt:
        sources.append(("GO BP", a.annotation_gmt, "gmt"))
    for spec in a.prior:
        nm, _, path = spec.partition("=")
        sources.append((nm.strip(), path.strip(), "npz"))
    priors, go = {}, {}
    for nm, path, kind in sources:
        try:
            Sg, n_feat, cov = (gmt_similarity(path, L) if kind == "gmt" else npz_similarity(path, L))
        except Exception as exc:                      # a missing prior must not kill the figure
            log.warning("prior %s (%s): %s", nm, path, exc); continue
        Ag = knn_graph(Sg, a.knn)
        cnt_g, _, pairs_g = traversal(Ag, sets, nl, moa, keep_pairs=True)
        null_g, _ = matched_null(Ag, Sg, max(a.n_perm // 4, 10))
        exp_g = null_g.mean(0)
        tg_ = pd.DataFrame(dict(gene=L, n_pairs=cnt_g, expected=exp_g,
                                fold=cnt_g / np.maximum(exp_g, 0.5),
                                degree=np.diff(Ag.indptr))).sort_values("n_pairs", ascending=False)
        tg_.to_csv(f"{a.out_prefix}_bridges_prior_{nm.replace(' ', '_').replace('.', '')}.csv",
                   index=False)
        ag = tg_.set_index("gene")
        rank = {g: i + 1 for i, g in enumerate(tg_.gene)}
        priors[nm] = dict(n_features=n_feat, coverage=cov, n_connected=len(pairs_g),
                          n_edges=int(Ag.nnz // 2),
                          top=[dict(gene=g, n_pairs=int(ag.n_pairs[g]), fold=float(ag.fold[g]),
                                    rank=int(rank[g])) for g in top.gene],
                          own_top=[dict(gene=r.gene, n_pairs=int(r.n_pairs), fold=float(r.fold))
                                   for r in tg_.head(a.top_n).itertuples()])
        log.info("%-14s %5.1f%% covered, %6d edges, connects %5d of %d pairs | overlap with the "
                 "atom top-%d: %d | atom bridges here: %s", nm, 100 * cov, Ag.nnz // 2,
                 len(pairs_g), n_pair, len(top),
                 len(set(top.gene) & {x["gene"] for x in priors[nm]["own_top"]}),
                 ", ".join(f"{x['gene']} {x['fold']:.1f}x" for x in priors[nm]["top"][:4]))
    if priors:
        pd.DataFrame({nm: {x["gene"]: x["fold"] for x in v["top"]}
                      for nm, v in priors.items()}).reindex(top.gene).to_csv(
            f"{a.out_prefix}_bridge_prior_folds.csv")
        go = priors.get("GO BP", {})

    json.dump(dict(knn=a.knn, n_drugs=len(drugs), n_pairs=n_pair, n_connected=len(pairs),
                   n_perm=a.n_perm, annotation=go,
                   min_pairs=a.min_pairs, min_fold=a.min_fold, priors=priors,
                   top=[dict(gene=r.gene, n_pairs=int(r.n_pairs), expected=float(r.expected),
                             fold=float(r.fold), z=float(r.z), p=float(r.p),
                             cross=int(r.n_cross_moa), degree=int(r.degree))
                        for r in top.itertuples()]),
              open(f"{a.out_prefix}_bridge_meta.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--drug-meta", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--graph-genes", default="PDPK1,RALGDS",
                    help="bridges whose drug paths are exported as a graph ('all' = every top bridge)")
    ap.add_argument("--graph-endpoints", type=int, default=8, help="most-used endpoints per bridge")
    ap.add_argument("--graph-drugs", type=int, default=5, help="drugs named per endpoint")
    ap.add_argument("--graph-knn", type=int, default=5,
                    help="neighbours drawn per node in the example panel (<= --knn)")
    ap.add_argument("--min-pairs", type=int, default=50, help="minimum drug pairs for the top list")
    ap.add_argument("--min-fold", type=float, default=2.0, help="minimum observed/expected")
    ap.add_argument("--annotation-gmt", default="", help="GMT library for the annotation-graph null")
    ap.add_argument("--prior", action="append", default=[], metavar="NAME=PATH",
                    help="prior embedding (.npz with genes/embeddings) to rebuild the graph from")
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
