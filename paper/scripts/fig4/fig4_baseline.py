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

"""Fig. 5: the cross-modal comparator, data-derived drug strength, and the MoA kNN test.

SIGNATURE CORRELATION (Connectivity-Map style). The reference is the training corpus itself: one
expression delta per perturbed gene, log1p(perturbation pseudobulk) - log1p(control pseudobulk)
within each dataset, averaged over the datasets that perturbed it. A drug's delta (log1p of
library-normalised counts, minus its matched controls -- the same strata as the ExPert arm) is
Pearson-correlated with every reference delta. Both are restricted to the model's input genes, so
the baseline sees exactly the information ExPert sees.

FAIR COMPARISON. The baseline can only rank genes that have a CRISPR reference; ExPert ranks the
full 19,895-row bank. Both are therefore re-ranked over the COMMON candidate set (reference genes
that are also bank rows) before recall@K, against the same target-permutation null.

Also written: |Δx| per drug x dose (data-derived strength, panel C) and, with --moa-tsv, the
kNN mechanism test (panel H): the fraction of each drug's k nearest drugs sharing its annotated
pathway / target family, in ExPert Δz vs in expression Δx, against a label shuffle.

Stage `ref` builds the reference once (cache npz); stage `query` scores one query.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig4_tahoe_score as fs


def model_genes(path):
    import h5py
    from anndata.io import read_elem
    with h5py.File(path, "r") as f:
        return np.asarray(read_elem(f["var"]).index).astype(str)


def build_ref(a):
    import h5py
    from anndata.io import read_elem
    genes = model_genes(a.model_genes_h5ad)
    with h5py.File(a.symbol_map_h5ad, "r") as f:
        v = read_elem(f["var"])
    sym2ens = dict(zip(v.index.astype(str), v[a.symbol_map_col].astype(str)))
    ens = [sym2ens.get(g) for g in genes]
    acc, cnt = {}, {}
    per_ds = {}
    for path in sorted(Path(a.pbulk_dir).glob("*.h5ad")):
        t0 = time.time()
        with h5py.File(path, "r") as f:
            o = read_elem(f["obs"])
            var = np.asarray(read_elem(f["var"]).index).astype(str)
            pos = {e: i for i, e in enumerate(var)}
            cols = np.array([pos.get(e, -1) for e in ens])
            have = cols >= 0
            keep = ((o["variant"].astype(str) == a.variant) | o["is_ctrl"].astype(bool)).to_numpy()
            rows = np.flatnonzero(keep)
            X = np.zeros((len(rows), len(genes)), np.float32)
            for s in range(0, len(rows), 2000):
                blk = f["X"][rows[s:s + 2000]]
                X[s:s + 2000][:, have] = blk[:, cols[have]]
        X = np.log1p(X)
        ob = o.iloc[rows]
        ctrl = ob["is_ctrl"].astype(bool).to_numpy()
        if ctrl.sum() == 0:
            print(f"[ref] {path.stem}: no control rows, skipped", flush=True)
            continue
        mu = X[ctrl].mean(0)
        lab = ob["label"].astype(str).to_numpy()
        n = 0
        for g in np.unique(lab[~ctrl]):
            d = X[(lab == g) & ~ctrl].mean(0) - mu
            acc[g] = acc.get(g, 0) + d
            cnt[g] = cnt.get(g, 0) + 1
            n += 1
        per_ds[path.stem] = dict(n_perts=n, n_ctrl_rows=int(ctrl.sum()), genes_found=int(have.sum()))
        print(f"[ref] {path.stem}: {n} perturbations, {int(ctrl.sum())} control rows, "
              f"{int(have.sum())}/{len(genes)} genes, {time.time() - t0:.0f}s", flush=True)
    labels = np.array(sorted(acc))
    D = np.stack([acc[g] / cnt[g] for g in labels]).astype(np.float32)
    np.savez_compressed(a.ref_npz, labels=labels, D=D, genes=genes,
                        n_datasets=np.array([cnt[g] for g in labels]))
    json.dump(per_ds, open(Path(a.ref_npz).with_suffix(".json"), "w"), indent=1)
    print(f"[ref] {len(labels):,} perturbed genes x {len(genes)} genes -> {a.ref_npz}", flush=True)


def query_deltas(a, genes):
    """Per-drug and per-(drug, dose) expression deltas vs controls in the same stratum."""
    import anndata as ad
    import scipy.sparse as sp
    q = ad.read_h5ad(a.query)
    o = q.obs
    X = sp.csr_matrix(q.X, dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    pos = {g: i for i, g in enumerate(q.var_names.astype(str))}
    cols = np.array([pos.get(g, -1) for g in genes])
    Xg = np.zeros((q.n_obs, len(genes)), np.float32)
    Xg[:, cols >= 0] = X[:, cols[cols >= 0]].toarray()
    Xg = np.log1p(Xg)
    ctrl = o["is_control"].astype(bool).to_numpy()
    strat = (np.zeros(q.n_obs, int) if a.stratum_col in ("", "none")
             else pd.factorize(o[a.stratum_col].astype(str))[0])
    drug = o["drug_clean"].astype(str).to_numpy()
    dose = o["dose_uM"].astype(float).to_numpy()
    out = {}
    for lev in ("drug", "drug_dose"):
        acc, w = {}, {}
        for s in np.unique(strat):
            m = strat == s
            if (m & ctrl).sum() < a.min_ctrl:
                continue
            mu = Xg[m & ctrl].mean(0)
            df = pd.DataFrame({"drug": drug[m & ~ctrl], "dose": dose[m & ~ctrl]})
            idx = np.flatnonzero(m & ~ctrl)
            keys = df["drug"] if lev == "drug" else df["drug"] + "|" + df["dose"].astype(str)
            for k, ii in df.groupby(keys.to_numpy()).indices.items():
                acc[k] = acc.get(k, 0) + (Xg[idx[ii]].mean(0) - mu) * len(ii)
                w[k] = w.get(k, 0) + len(ii)
        ks = sorted(k for k in acc if w[k] >= a.min_cells)
        out[lev] = (ks, np.stack([acc[k] / w[k] for k in ks]).astype(np.float32),
                    np.array([w[k] for k in ks]))
    meta = (o[~ctrl].groupby("drug_clean", observed=True)
            .agg(targets=("targets", "first"), target_class=("target_class", "first")).reset_index())
    return out, meta


def ranks_within(S, cand):
    """1-based rank of every candidate column within the candidate set only."""
    Sc = S[:, cand]
    order = np.argsort(-Sc, axis=1)
    rk = np.empty_like(order)
    np.put_along_axis(rk, order, np.arange(1, Sc.shape[1] + 1)[None, :], axis=1)
    return rk


def knn_share(V, lab, k, n_shuffle, rng):
    Vn = fs.unit(V)
    S = Vn @ Vn.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    obs = float((lab[nn] == lab[:, None]).mean())
    null = [float((p[nn] == p[:, None]).mean()) for p in (rng.permutation(lab) for _ in range(n_shuffle))]
    return obs, float(np.mean(null)), float(np.quantile(null, 0.975)), \
        float((np.sum(np.array(null) >= obs) + 1) / (n_shuffle + 1))


def score_query(a):
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ref = np.load(a.ref_npz, allow_pickle=True)
    genes, rlab, Dref = ref["genes"].astype(str), ref["labels"].astype(str), ref["D"]
    deltas, meta = query_deltas(a, genes)
    ex = np.load(Path(a.expert_dir) / "scores_drug.npz", allow_pickle=True)
    blab, ekeys = ex["labels"].astype(str), [k.strip() for k in ex["keys"].astype(str)]
    bpos = {g: i for i, g in enumerate(blab)}
    common = np.array([g for g in rlab if g in bpos])
    rpos = {g: i for i, g in enumerate(rlab)}
    ci_ref = np.array([rpos[g] for g in common]); ci_bank = np.array([bpos[g] for g in common])
    cpos = {g: i for i, g in enumerate(common)}

    dkeys, Dq, nq = deltas["drug"]
    dkeys = [k.strip() for k in dkeys]
    shared = [k for k in dkeys if k in ekeys]
    qi = np.array([dkeys.index(k) for k in shared]); ei = np.array([ekeys.index(k) for k in shared])
    # Pearson == cosine of row-centred vectors
    C = fs.unit(Dq[qi] - Dq[qi].mean(1, keepdims=True)) @ fs.unit(Dref - Dref.mean(1, keepdims=True)).T
    rk_sig = ranks_within(C, ci_ref)
    rk_exp = ranks_within(ex["S"].astype(np.float32)[ei], ci_bank)

    meta = meta.assign(drug_clean=meta["drug_clean"].str.strip()).set_index("drug_clean")
    sets = fs.gene_sets(a.gene_sets_json, a.max_set)
    cls = np.array([meta.loc[k, "target_class"] for k in shared])
    tgt = [np.array([cpos[g] for g in fs.split(meta.loc[k, "targets"]) if g in cpos], int) for k in shared]
    mate = [np.array([cpos[g] for g in fs.mates_of(fs.split(meta.loc[k, "targets"]), sets, a.mate_cap)
                      if g in cpos], int) for k in shared]
    rows, summ = [], {}
    for method, rk in (("expert", rk_exp), ("signature_corr", rk_sig)):
        for truth, idx in (("exact", tgt), ("mate", mate)):
            for cl in ("all", "trained", "mixed", "zeroshot"):
                m = (cls != "nonlof") if cl == "all" else (cls == cl)
                ii = [i for i in np.flatnonzero(m) if len(idx[i])]
                if len(ii) < 5:
                    continue
                obs_r = np.array([fs.best_rank(rk[i], idx[i]) for i in ii])
                null_r = np.array([[fs.best_rank(rk[i], idx[j]) for i, j in zip(ii, rng.permutation(ii))]
                                   for _ in range(a.n_perm)])
                for k in fs.KGRID:
                    nk = (null_r <= k).mean(1)
                    rows.append(dict(method=method, truth=truth, cls=cl, k=k, n=len(ii),
                                     recall=float((obs_r <= k).mean()), null=float(nk.mean()),
                                     null_lo=float(np.quantile(nk, 0.025)), null_hi=float(np.quantile(nk, 0.975))))
                summ[f"{method}|{truth}|{cl}"] = dict(n=len(ii), median_rank=float(np.median(obs_r)))
    pd.DataFrame(rows).to_csv(out / "baseline_curve.csv", index=False)
    per = pd.DataFrame({"drug": shared, "target_class": cls,
                        "rank_expert": [fs.best_rank(rk_exp[i], tgt[i]) for i in range(len(shared))],
                        "rank_signature": [fs.best_rank(rk_sig[i], tgt[i]) for i in range(len(shared))]})
    per.to_csv(out / "baseline_per_drug.csv", index=False)

    # data-derived strength: |Δx| per (drug, dose)
    dk, Dd, nd = deltas["drug_dose"]
    pd.DataFrame({"drug": [k.rsplit("|", 1)[0].strip() for k in dk],
                  "dose_uM": [float(k.rsplit("|", 1)[1]) for k in dk],
                  "delta_x_norm": np.linalg.norm(Dd, axis=1), "n_cells": nd}).to_csv(out / "dose_delta_x.csv", index=False)

    res = dict(query=a.query, stratum_col=a.stratum_col, n_common_candidates=int(len(common)),
               n_ref=int(len(rlab)), n_drugs=len(shared), summary=summ)
    if a.moa_tsv:
        moa = pd.read_csv(a.moa_tsv, sep="\t", keep_default_na=False)
        moa["key"] = moa["drug"].str.strip()
        moa = moa.set_index("key")
        knn = []
        for col in a.moa_cols.split(","):
            ok = [i for i, k in enumerate(shared) if k in moa.index and moa.loc[k, col] not in ("", "Others")]
            lab = np.array([moa.loc[shared[i], col] for i in ok])
            vc = pd.Series(lab).value_counts()
            keep = np.isin(lab, vc[vc >= 2].index)            # a singleton class can never be matched
            ok = np.array(ok)[keep]; lab = lab[keep]
            for space, V in (("expert_dz", ex["R"].astype(np.float32)[ei][ok]), ("expression_dx", Dq[qi][ok])):
                for k in (1, 3, 5, 10):
                    o_, n_, hi, p = knn_share(V, lab, k, a.n_perm, rng)
                    knn.append(dict(annotation=col, space=space, k=k, n_drugs=len(ok), n_classes=int(len(set(lab))),
                                    share=o_, null=n_, null_hi=hi, p=p))
        pd.DataFrame(knn).to_csv(out / "moa_knn.csv", index=False)
    json.dump(res, open(out / "baseline_metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("ref", "query"))
    ap.add_argument("--ref-npz", required=True)
    ap.add_argument("--model-genes-h5ad", help="ref: training DB (var = the model's input genes)")
    ap.add_argument("--pbulk-dir", help="ref: per-dataset pseudobulk h5ads")
    ap.add_argument("--symbol-map-h5ad", help="ref: an h5ad whose var index is symbols and has an Ensembl column")
    ap.add_argument("--symbol-map-col", default="ensembl_id")
    ap.add_argument("--variant", default="rep", help="ref: pseudobulk variant rows to use")
    ap.add_argument("--query", help="query: Tahoe-schema h5ad")
    ap.add_argument("--expert-dir", help="query: fig4_tahoe_score.py output dir (scores_drug.npz)")
    ap.add_argument("--out-dir")
    ap.add_argument("--stratum-col", default="", help="query: control stratum (match the ExPert arm)")
    ap.add_argument("--gene-sets-json")
    ap.add_argument("--moa-tsv", default="")
    ap.add_argument("--moa-cols", default="pathway,target_annotation")
    ap.add_argument("--max-set", type=int, default=80)
    ap.add_argument("--mate-cap", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--min-ctrl", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_ref(a) if a.stage == "ref" else score_query(a)
