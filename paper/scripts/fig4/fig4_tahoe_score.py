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

"""Fig. 5 stage 2 (Tahoe-100M): drug -> target recovery from one encoded centring arm. No model.

Response = control-centred z_shared, the readout path (not the softmax head the July run scored):
every cell's latent minus the mean latent of the DMSO cells that shared its forward pass (stratum =
the arm's subset), averaged into pseudobulks at three levels -- drug, drug x dose, drug x line --
and ranked by cosine over the FULL bank (trained atoms + f(missing) zero-shot rows, so a zero-shot
target competes with everything).

    exact     best rank over the drug's annotated targets that are in the bank (`any` policy;
              targets outside the bank are coverage, never misses)
    mate      best rank over complex/pathway mates (CORUM + Reactome + KEGG sets <= --max-set,
              smallest sets first, union capped at --mate-cap, targets themselves excluded)
    null      target (or mate) sets PERMUTED across drugs with the rankings held fixed: keeps the
              model's popularity bias, which the analytic K/C chance would miss
    dose      LABELLED dose: per drug rank at highest vs lowest dose; median rank, |delta| and eff_hat by dose
    attractor how often each gene is a drug's argmax / in its top 10; per-cell argmax frequency in
              drug vs DMSO cells; the CONDITIONAL read rebuilds drug pseudobulks from only the cells
              whose own argmax is outside the attractor set
    omp       readout.nnomp, stop off, first --omp-k picks per drug: does the complex come out

Writes <out>/ranks_<level>.csv, attractor.csv, omp.csv, metrics.json, curve.csv (recall on a log-K
grid with the permutation band), top20_drug.csv, scores_{drug,drug_dose}.npz (cosine table + Δz).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KS = (1, 10, 50, 100, 500)
KGRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)   # recall curves (panels B, E)
LIBRARIES = ("CORUM", "Reactome_Pathways_2024", "KEGG_2021_Human")


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def split(v):
    s = str(v).strip()
    return [t.strip() for t in s.split(";") if t.strip()] if s and s.lower() != "nan" else []


def rank_matrix(R, Pn, chunk=512):
    """1-based rank of every bank row for every response row (ties broken by order)."""
    out = np.empty((R.shape[0], Pn.shape[0]), np.int32)
    Rn = unit(R)
    for i in range(0, len(Rn), chunk):
        S = Rn[i:i + chunk] @ Pn.T
        order = np.argsort(-S, axis=1)
        rk = np.empty_like(order)
        np.put_along_axis(rk, order, np.arange(1, S.shape[1] + 1)[None, :], axis=1)
        out[i:i + chunk] = rk
    return out


def best_rank(rk_row, idx):
    return int(rk_row[idx].min()) if len(idx) else -1


def gene_sets(cache, max_set):
    p = Path(cache)
    if p.exists():
        sets = json.load(open(p))
    else:
        from expert.utils.performance import get_library
        sets = {}
        for lib in LIBRARIES:
            for name, genes in (get_library(library=lib, organism="Human") or {}).items():
                sets[f"{lib}::{name}"] = sorted(set(genes))
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(sets, open(p, "w"))
    return {k: v for k, v in sets.items() if 3 <= len(v) <= max_set}


def mates_of(targets, sets, cap):
    hit = sorted((s for s in sets.values() if set(s) & set(targets)), key=len)
    out = []
    for s in hit:
        for g in s:
            if g not in targets and g not in out:
                out.append(g)
        if len(out) >= cap:
            break
    return out[:cap]


def recall_with_null(ranks_by_drug, idx_sets, ks, n_perm, rng):
    """ranks_by_drug: (n_drug, C) rank matrix; idx_sets: per drug, bank indices of its truth."""
    have = np.array([len(s) > 0 for s in idx_sets])
    rows = np.flatnonzero(have)
    obs = np.array([best_rank(ranks_by_drug[i], idx_sets[i]) for i in rows])
    res = {f"recall@{k}": float((obs <= k).mean()) for k in ks}
    null = np.zeros((n_perm, len(ks)))
    sets = [idx_sets[i] for i in rows]
    for p in range(n_perm):
        perm = rng.permutation(len(rows))
        b = np.array([best_rank(ranks_by_drug[rows[i]], sets[perm[i]]) for i in range(len(rows))])
        null[p] = [(b <= k).mean() for k in ks]
    for j, k in enumerate(ks):
        res[f"null@{k}"] = float(null[:, j].mean())
        res[f"p@{k}"] = float(((null[:, j] >= res[f"recall@{k}"]).sum() + 1) / (n_perm + 1))
    res["n"] = int(len(rows))
    res["median_rank"] = float(np.median(obs)) if len(obs) else None
    return res, obs, rows


def main(a):
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    enc = Path(a.encode_dir)
    bank = np.load(a.bank, allow_pickle=True)
    P, labels, n_lab = bank["P"].astype(np.float32), bank["labels"].astype(str), int(bank["n_labels"])
    Pn = unit(P)
    pos = {g: i for i, g in enumerate(labels)}
    C = len(labels)

    obs = pd.read_parquet(enc / f"obs_{a.arm}.parquet")
    Z = np.load(enc / f"z_{a.arm}.npy").astype(np.float32)
    eff = np.load(enc / f"eff_{a.arm}.npy")
    ok = ~np.isnan(Z).any(1)
    obs, Z, eff = obs[ok].reset_index(drop=True), Z[ok], eff[ok]
    Z -= Z.mean(0, keepdims=True)          # whole-query latent centre (moves `mean`, never `delta`)
    is_ctrl = obs["is_control"].to_numpy(bool)
    stratum = {"global": np.zeros(len(obs), int)}.get(a.arm)
    if stratum is None:
        keys = ["cell_name"] if a.arm == "line" else ["plate", "cell_name"]
        stratum = pd.factorize(obs[keys].astype(str).agg("@".join, axis=1))[0]
    D = np.empty_like(Z)
    for s in np.unique(stratum):
        m = stratum == s
        c = m & is_ctrl
        if c.sum() == 0:
            D[m] = np.nan
            continue
        D[m] = Z[m] - Z[c].mean(0, keepdims=True)
    drug = ~is_ctrl & ~np.isnan(D).any(1)
    print(f"[{a.arm}] {len(obs):,} cells ({int(is_ctrl.sum()):,} DMSO), {int(drug.sum()):,} drug cells "
          f"with a matched control, {len(np.unique(stratum))} strata", flush=True)

    # ---- truth ----------------------------------------------------------------------------------
    dmeta = (obs[drug].groupby("drug_clean", observed=True)
             .agg(targets=("targets", "first"), target_class=("target_class", "first"),
                  single_target=("single_target", "first")).reset_index())
    sets = gene_sets(a.gene_sets_json, a.max_set)
    tgt_idx, mate_idx, cov = {}, {}, [0, 0]
    for r in dmeta.itertuples():
        t = split(r.targets)
        cov[0] += len(t); cov[1] += sum(g in pos for g in t)
        tgt_idx[r.drug_clean] = np.array([pos[g] for g in t if g in pos], int)
        mate_idx[r.drug_clean] = np.array([pos[g] for g in mates_of(t, sets, a.mate_cap) if g in pos], int)

    # ---- pseudobulks ----------------------------------------------------------------------------
    od = obs[drug].reset_index(drop=True)
    Dd, Ed = D[drug], eff[drug]
    levels = {"drug": ["drug_clean"], "drug_dose": ["drug_clean", "dose_uM"],
              "drug_line": ["drug_clean", "cell_name"]}
    metrics = dict(arm=a.arm, encode_dir=str(enc), n_cells=int(len(obs)), n_drug_cells=int(drug.sum()),
                   n_strata=int(len(np.unique(stratum))), bank_rows=C, n_trained=n_lab,
                   target_coverage=f"{cov[1]}/{cov[0]}", n_gene_sets=len(sets))
    rank_store = {}
    for lev, cols in levels.items():
        g = od.groupby(cols, observed=True).indices
        keys = list(g)
        n = np.array([len(g[k]) for k in keys])
        keep = n >= a.min_cells
        keys = [k for k, kp in zip(keys, keep) if kp]
        R = np.stack([Dd[g[k]].mean(0) for k in keys])
        E = np.array([np.nanmean(Ed[g[k]]) for k in keys])
        rk = rank_matrix(R, Pn)
        dname = [k if isinstance(k, str) else k[0] for k in keys]
        top = np.argsort(rk, axis=1)[:, :5]
        df = pd.DataFrame({"key": ["|".join(map(str, k)) if not isinstance(k, str) else k for k in keys],
                           "drug": dname, "n_cells": n[keep], "eff_hat": E,
                           "delta_norm": np.linalg.norm(R, axis=1),
                           "exact_rank": [best_rank(rk[i], tgt_idx[d]) for i, d in enumerate(dname)],
                           "mate_rank": [best_rank(rk[i], mate_idx[d]) for i, d in enumerate(dname)],
                           "top5": [";".join(labels[top[i]]) for i in range(len(keys))]})
        if lev != "drug":
            df[cols[1]] = [k[1] for k in keys]
        df = df.merge(dmeta.rename(columns={"drug_clean": "drug"}), on="drug", how="left")
        df.to_csv(out / f"ranks_{lev}.csv", index=False)
        rank_store[lev] = (keys, dname, rk, R)
        if lev in ("drug", "drug_dose"):
            # full cosine table + response, for the expression baseline (restricted to a common
            # candidate set) and the MoA kNN test; float16 keeps it ~10 MB
            np.savez_compressed(out / f"scores_{lev}.npz", keys=np.array(df["key"]), labels=labels,
                                S=(unit(R) @ Pn.T).astype(np.float16), R=R.astype(np.float16))

    # ---- drug-level recall + null, split by target class ----------------------------------------
    keys, dname, rk, R = rank_store["drug"]
    dclass = dmeta.set_index("drug_clean").loc[dname, "target_class"].to_numpy()
    rec = {}
    for truth_name, idx in (("exact", tgt_idx), ("mate", mate_idx)):
        sets_ = [idx[d] for d in dname]
        # headline = LoF drugs only: agonists / activators / no-gene drugs ('nonlof', sci-Plex) are
        # a separate class, never pooled in
        lof = dclass != "nonlof"
        rec[truth_name] = {"all": recall_with_null(rk[lof], [s for s, l in zip(sets_, lof) if l],
                                                   KS, a.n_perm, rng)[0]}
        for cl in ("trained", "mixed", "zeroshot", "nonlof"):
            m = dclass == cl
            if m.sum() >= 5:
                rec[truth_name][cl] = recall_with_null(rk[m], [s for s, mm in zip(sets_, m) if mm],
                                                       KS, a.n_perm, rng)[0]
    metrics["drug_recall"] = rec

    # recall curves on a log-K grid with the permutation band (panels B, E)
    rows = []
    for truth_name, idx in (("exact", tgt_idx), ("mate", mate_idx)):
        for cl in ("all", "trained", "mixed", "zeroshot"):
            m = (dclass != "nonlof") if cl == "all" else (dclass == cl)
            sets_ = [idx[d] for d, mm in zip(dname, m) if mm and len(idx[d])]
            rows_i = [i for i, (d, mm) in enumerate(zip(dname, m)) if mm and len(idx[d])]
            if len(rows_i) < 5:
                continue
            obs_r = np.array([best_rank(rk[i], s_) for i, s_ in zip(rows_i, sets_)])
            null_r = np.array([[best_rank(rk[i], sets_[j]) for i, j in
                                zip(rows_i, rng.permutation(len(rows_i)))] for _ in range(a.n_perm)])
            for k in KGRID:
                nk = (null_r <= k).mean(1)
                rows.append(dict(truth=truth_name, cls=cl, k=k, n=len(rows_i), recall=float((obs_r <= k).mean()),
                                 null=float(nk.mean()), null_lo=float(np.quantile(nk, 0.025)),
                                 null_hi=float(np.quantile(nk, 0.975))))
    pd.DataFrame(rows).to_csv(out / "curve.csv", index=False)

    # top-20 per drug with cosines (panel D worked examples)
    S_drug = unit(R) @ Pn.T
    top = np.argsort(-S_drug, axis=1)[:, :20]
    pd.DataFrame([dict(drug=d, rank=j + 1, gene=labels[top[i, j]], cos=float(S_drug[i, top[i, j]]),
                       trained=bool(top[i, j] < n_lab),
                       is_target=bool(top[i, j] in set(tgt_idx[d])), is_mate=bool(top[i, j] in set(mate_idx[d])))
                  for i, d in enumerate(dname) for j in range(top.shape[1])]).to_csv(out / "top20_drug.csv", index=False)

    # ---- dose -----------------------------------------------------------------------------------
    dd = pd.read_csv(out / "ranks_drug_dose.csv")
    dd = dd[dd.exact_rank > 0]
    piv = dd.pivot_table(index="drug", columns="dose_uM", values="exact_rank")
    pe = pd.read_csv(out / "ranks_drug_dose.csv").pivot_table(index="drug", columns="dose_uM",
                                                                values="eff_hat")
    pn = pd.read_csv(out / "ranks_drug_dose.csv").pivot_table(index="drug", columns="dose_uM",
                                                                values="delta_norm")
    # dose is the LABELLED concentration; eff_hat (strength head) and |delta| are two reads of it
    dose = dict(median_rank_by_dose={str(c): float(piv[c].median()) for c in piv.columns},
                mean_eff_by_dose={str(c): float(pe[c].mean()) for c in pe.columns},
                median_delta_norm_by_dose={str(c): float(pn[c].median()) for c in pn.columns})
    lo, hi = min(piv.columns), max(piv.columns)        # Tahoe 0.05 / 5 uM, sci-Plex 0.01 / 10 uM
    dose["lo_hi_uM"] = [float(lo), float(hi)]
    if lo != hi:
        b = piv[[lo, hi]].dropna()
        dose["frac_rank_improves_hi_vs_lo"] = float((b[hi] < b[lo]).mean())
        dose["n_drugs"] = int(len(b))
        e = pe[[lo, hi]].dropna()
        dose["frac_eff_up_hi_vs_lo"] = float((e[hi] > e[lo]).mean())
        nb = pn[[lo, hi]].dropna()
        dose["frac_delta_norm_up_hi_vs_lo"] = float((nb[hi] > nb[lo]).mean())
        # does the strength head see dose at the CELL level, pooled over drugs?
        dc = od["dose_uM"].to_numpy()
        dose["cell_eff_spearman_dose"] = float(pd.Series(Ed).corr(pd.Series(dc), method="spearman"))
        mates = dd[dd.mate_rank > 0].pivot_table(index="drug", columns="dose_uM", values="mate_rank")
        if {lo, hi} <= set(mates.columns):
            mb = mates[[lo, hi]].dropna()
            dose["frac_mate_rank_improves_hi_vs_lo"] = float((mb[hi] < mb[lo]).mean())
    metrics["dose"] = dose

    # ---- attractor ------------------------------------------------------------------------------
    am = rk.argmin(1)
    top10 = np.argsort(rk, axis=1)[:, :10]
    freq_arg = pd.Series(labels[am]).value_counts()
    freq_top = pd.Series(labels[top10.ravel()]).value_counts()
    att = pd.DataFrame({"drug_argmax_frac": freq_arg / len(am),
                        "drug_top10_frac": freq_top / len(am)}).fillna(0)
    # per-cell argmax in drug vs DMSO cells, on the same delta (DMSO delta = within-control spread)
    sub = rng.choice(np.flatnonzero(drug), min(a.percell_n, int(drug.sum())), replace=False)
    ctl = np.flatnonzero(is_ctrl & ~np.isnan(D).any(1))
    cam = lambda X: np.concatenate([(unit(X[i:i + 4096]) @ Pn.T).argmax(1)
                                    for i in range(0, len(X), 4096)])
    cell_am_drug = cam(D[sub]); cell_am_ctrl = cam(D[ctl])
    att["cell_argmax_frac_drug"] = pd.Series(labels[cell_am_drug]).value_counts() / len(sub)
    att["cell_argmax_frac_dmso"] = pd.Series(labels[cell_am_ctrl]).value_counts() / max(len(ctl), 1)
    att = att.fillna(0).sort_values("drug_top10_frac", ascending=False)
    att.index.name = "gene"
    att.head(200).to_csv(out / "attractor.csv")
    attractor_set = list(att.index[:a.n_attractor])
    metrics["attractor"] = dict(
        top_argmax=freq_arg.index[0], top_argmax_frac=float(freq_arg.iloc[0] / len(am)),
        n_distinct_argmax=int(len(freq_arg)),
        top6_share_of_top10=float(freq_top.iloc[:6].sum() / top10.size),
        attractor_set=attractor_set,
        cell_distinct_argmax_drug=int(len(np.unique(cell_am_drug))),
        cell_distinct_argmax_dmso=int(len(np.unique(cell_am_ctrl))),
        cell_attractor_frac_drug=float(np.isin(labels[cell_am_drug], attractor_set).mean()),
        cell_attractor_frac_dmso=float(np.isin(labels[cell_am_ctrl], attractor_set).mean()))

    # conditional read: drug pseudobulks from cells whose OWN argmax is not an attractor gene
    cell_am_all = cam(Dd)
    keep_c = ~np.isin(labels[cell_am_all], attractor_set)
    g = od[keep_c].groupby("drug_clean", observed=True).indices
    Dk = Dd[keep_c]
    cls_of = dmeta.set_index("drug_clean")["target_class"].to_dict()
    ck = [k for k in g if len(g[k]) >= a.min_cells and cls_of.get(k) != "nonlof"]
    rk_c = rank_matrix(np.stack([Dk[g[k]].mean(0) for k in ck]), Pn)
    cond = {}
    for truth_name, idx in (("exact", tgt_idx), ("mate", mate_idx)):
        cond[truth_name] = recall_with_null(rk_c, [idx[d] for d in ck], KS, a.n_perm, rng)[0]
    metrics["conditional"] = dict(frac_cells_kept=float(keep_c.mean()), n_drugs=len(ck), **cond)

    # ---- OMP picks ------------------------------------------------------------------------------
    import expert.utils.readout as ro
    rows = []
    for i, d in enumerate(dname):
        sel, w = ro.nnomp(R[i], P, k=a.omp_k, stop=None)
        t, mt = set(tgt_idx[d]), set(mate_idx[d])
        rows.append(dict(drug=d, picks=";".join(labels[sel]), coefs=";".join(f"{x:.3g}" for x in w),
                         exact_in_omp=int(bool(t & set(sel))), mate_in_omp=int(bool(mt & set(sel))),
                         n_mate_in_omp=len(mt & set(sel))))
    omp = pd.DataFrame(rows).merge(dmeta.rename(columns={"drug_clean": "drug"}), on="drug")
    omp.to_csv(out / "omp.csv", index=False)
    metrics["omp"] = dict(k=a.omp_k, frac_exact=float(omp.exact_in_omp.mean()),
                          frac_mate=float(omp.mate_in_omp.mean()))

    # ---- named exact hits -----------------------------------------------------------------------
    dr = pd.read_csv(out / "ranks_drug.csv")
    metrics["exact_top100"] = int((dr.exact_rank.between(1, 100)).sum())
    metrics["mate_top100"] = int((dr.mate_rank.between(1, 100)).sum())
    metrics["best_exact"] = (dr[dr.exact_rank > 0].nsmallest(15, "exact_rank")
                             [["drug", "targets", "target_class", "exact_rank"]].to_dict("records"))
    metrics["minutes"] = (time.time() - t0) / 60
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1, default=str)
    e = rec["exact"]["all"]; m = rec["mate"]["all"]
    print(f"[{a.arm}] exact r@50 {e['recall@50']:.4f} (null {e['null@50']:.4f}) r@100 "
          f"{e['recall@100']:.4f} (null {e['null@100']:.4f}) med {e['median_rank']} | mate r@100 "
          f"{m['recall@100']:.3f} (null {m['null@100']:.3f}) | argmax {metrics['attractor']['top_argmax']} "
          f"{metrics['attractor']['top_argmax_frac']:.2f} | {metrics['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encode-dir", required=True)
    ap.add_argument("--bank", required=True, help="bank.npz from fig4_tahoe_encode.py --harvest-bank")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gene-sets-json", required=True, help="cache; fetched from Enrichr if absent")
    ap.add_argument("--max-set", type=int, default=80)
    ap.add_argument("--mate-cap", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--percell-n", type=int, default=30000)
    ap.add_argument("--n-attractor", type=int, default=10)
    ap.add_argument("--omp-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
