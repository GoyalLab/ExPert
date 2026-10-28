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

"""Fig. 5 (degron): does acute protein depletion read as the CRISPR perturbation of the same gene?

Pooled AID-degron screen: 40 tagged clones (one gene each), DMSO vs auxin 24 h, two separate 10x
libraries. A minority of nuclei carry a clone call (--clone-col). Readout path as for the drugs:
one predict() with DMSO as the encoder's input controls, raw z_shared, cosine over the full bank.

    clone-matched  delta_G = mean z(IAA, G) - mean z(DMSO, G): the clone's own baseline cancels.
                   Falls back to the all-DMSO mean when clone G has < --min-cells DMSO nuclei
                   (flagged in the table)
    DiD            delta_G minus the mean delta over all clones: removes the shift every clone
                   shares, which includes the DMSO-vs-IAA LIBRARY difference
    per cell       each labelled nucleus' delta (vs the all-DMSO mean) ranked for its own gene,
                   IAA vs DMSO -- with the degron off the gene should NOT rank (paired control)
    null           clone labels permuted across clones, rankings fixed (rank of another clone's gene)

Also: eff_hat (strength head) IAA vs DMSO per clone, and the cosine of delta_G to its own prototype
as a z-score against all prototypes. Writes clones.csv, cells.csv.gz, metrics.json and
scores_{matched,did}.npz (the full clone x prototype cosine table).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KS = (1, 10, 50, 100, 500)


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def ranks_of(R, Pn):
    S = unit(R) @ Pn.T
    order = np.argsort(-S, axis=1)
    rk = np.empty_like(order)
    np.put_along_axis(rk, order, np.arange(1, S.shape[1] + 1)[None, :], axis=1)
    return rk, S


def main(a):
    import anndata as ad
    import expert.utils.readout as ro
    from expert import ExPert
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    model = ExPert.load(a.model_dir)
    n_shared = int(model.module.align_ds_mean.shape[1])
    bank = np.load(a.bank, allow_pickle=True)
    P, labels, n_lab = bank["P"].astype(np.float32), bank["labels"].astype(str), int(bank["n_labels"])
    Pn = unit(P)
    pos = {g: i for i, g in enumerate(labels)}

    q = ad.read_h5ad(a.query)
    q.obsm.clear(); q.layers.clear()
    if a.max_pct_mt is not None:
        # damaged / dying nuclei carry more mito reads: drop the high-MT tail BEFORE the forward
        # pass, so the DMSO input centre and every delta are built from the kept nuclei only
        keep = (q.obs[a.mt_col].astype(float) <= a.max_pct_mt).to_numpy()
        lab0 = (q.obs[a.clone_col].astype(str) != a.unassigned).to_numpy()
        print(f"[mt filter] {a.mt_col} <= {a.max_pct_mt}: keep {keep.mean():.3f} of nuclei, "
              f"{(keep & lab0).sum() / max(lab0.sum(), 1):.3f} of labelled", flush=True)
        q = q[keep].copy()
    q.obs = q.obs[[a.condition_col, a.clone_col]].copy()
    q.obs[a.clone_col] = q.obs[a.clone_col].astype(str)
    if a.mask_genes:
        pats = [x.strip() for x in a.mask_genes.split(",") if x.strip()]
        v = pd.Index(q.var_names).astype(str)
        drop = np.zeros(q.n_vars, bool)
        for p in pats:
            drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith("*") else v == p, bool)
        q = q[:, ~drop].copy()
        print(f"[mask] dropped {int(drop.sum())} genes ({pats})", flush=True)
    q.X = q.X.tocsr()
    ro.build_groups(q, a.condition_col, control_condition=a.control_condition)
    ro.mark_query(model, q)
    model.predict(q, inplace=True, return_latents=True, soft_predictions=False, head="A",
                  obs_only=True, center_mode=np.zeros(n_shared, np.float32), control_label="control",
                  batch_size=a.batch_size, n_cores=a.n_cores, progress_bar=False)
    Z = np.asarray(q.obsm[ro.LATENT_KEY], np.float32)
    eff = q.obs["eff_hat"].to_numpy(np.float32) if "eff_hat" in q.obs else np.full(q.n_obs, np.nan)
    cond = q.obs[a.condition_col].astype(str).to_numpy()
    clone = q.obs[a.clone_col].to_numpy()
    dmso, iaa = cond == a.control_condition, cond == a.treated_condition
    lab = clone != a.unassigned
    print(f"[query] {q.n_obs:,} nuclei; DMSO {int(dmso.sum()):,} IAA {int(iaa.sum()):,}; labelled "
          f"{int(lab.sum()):,} ({int((lab & dmso).sum())} DMSO / {int((lab & iaa).sum())} IAA)", flush=True)
    ct = pd.crosstab(clone[lab], cond[lab])
    print(ct.sort_values(a.treated_condition, ascending=False).head(40).to_string(), flush=True)

    mu_dmso = Z[dmso].mean(0)
    # ---- per clone ------------------------------------------------------------------------------
    rows, deltas = [], []
    for g in sorted(set(clone[lab])):
        mi, md = lab & iaa & (clone == g), lab & dmso & (clone == g)
        if mi.sum() < a.min_cells:
            continue
        matched = md.sum() >= a.min_cells
        base = Z[md].mean(0) if matched else mu_dmso
        deltas.append(Z[mi].mean(0) - base)
        rows.append(dict(clone=g, in_bank=g in pos, trained=g in pos and pos[g] < n_lab,
                         n_iaa=int(mi.sum()), n_dmso=int(md.sum()), dmso_matched=bool(matched),
                         eff_iaa=float(np.nanmean(eff[mi])), eff_dmso=float(np.nanmean(eff[md])) if md.any() else np.nan))
    cl = pd.DataFrame(rows)
    D = np.stack(deltas)
    Ddid = D - D.mean(0, keepdims=True)
    metrics = dict(model_dir=a.model_dir, query=a.query, clone_col=a.clone_col, mask_genes=a.mask_genes,
                   max_pct_mt=a.max_pct_mt, n_nuclei=int(q.n_obs), n_labelled=int(lab.sum()),
                   n_clones=int(len(cl)), min_cells=a.min_cells)
    for name, R in (("matched", D), ("did", Ddid)):
        rk, S = ranks_of(R, Pn)
        # full cosine table per clone: the cross-modality summary (fig4_summary.py) needs every
        # clone's ranking, not just its own gene's rank
        np.savez_compressed(out / f"scores_{name}.npz", keys=np.array(cl.clone), labels=labels,
                            S=S.astype(np.float16))
        gi = np.array([pos.get(g, -1) for g in cl.clone])
        own = np.array([rk[i, j] if j >= 0 else -1 for i, j in enumerate(gi)])
        z = np.array([(S[i, j] - S[i].mean()) / S[i].std() if j >= 0 else np.nan for i, j in enumerate(gi)])
        cl[f"rank_{name}"] = own
        cl[f"cos_z_{name}"] = z
        cl[f"top5_{name}"] = [";".join(labels[np.argsort(rk[i])[:5]]) for i in range(len(cl))]
        ok = gi >= 0
        obs_r = own[ok]
        null = []
        for _ in range(a.n_perm):
            p = rng.permutation(np.flatnonzero(ok))
            null.append([rk[i, gi[j]] for i, j in zip(np.flatnonzero(ok), p)])
        null = np.array(null)
        metrics[name] = {**{f"recall@{k}": float((obs_r <= k).mean()) for k in KS},
                         **{f"null@{k}": float((null <= k).mean()) for k in KS},
                         "median_rank": float(np.median(obs_r)), "null_median_rank": float(np.median(null)),
                         "p_median": float(((np.median(null, 1) <= np.median(obs_r)).sum() + 1) / (a.n_perm + 1)),
                         "mean_cos_z": float(np.nanmean(z)), "n": int(ok.sum())}
    cl.to_csv(out / "clones.csv", index=False)

    # ---- per cell: own-gene rank, IAA vs DMSO ---------------------------------------------------
    idx = np.flatnonzero(lab & np.isin(clone, cl.clone) & np.array([c in pos for c in clone]))
    rkc, _ = ranks_of(Z[idx] - mu_dmso, Pn)
    own_c = np.array([rkc[n, pos[clone[i]]] for n, i in enumerate(idx)])
    cells = pd.DataFrame({"clone": clone[idx], "condition": cond[idx], "own_rank": own_c, "eff_hat": eff[idx]})
    cells.to_csv(out / "cells.csv.gz", index=False)
    pc = cells.groupby(["clone", "condition"]).own_rank.median().unstack()
    if {a.control_condition, a.treated_condition} <= set(pc.columns):
        pc = pc.dropna()
        metrics["per_cell"] = dict(
            median_own_rank={c: float(cells[cells.condition == c].own_rank.median())
                             for c in (a.control_condition, a.treated_condition)},
            frac_clones_iaa_better=float((pc[a.treated_condition] < pc[a.control_condition]).mean()),
            n_clones=int(len(pc)))
    e = cl.dropna(subset=["eff_dmso"])
    metrics["eff_hat"] = dict(frac_clones_iaa_higher=float((e.eff_iaa > e.eff_dmso).mean()) if len(e) else None,
                              mean_iaa=float(np.nanmean(eff[iaa])), mean_dmso=float(np.nanmean(eff[dmso])))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1, default=str)
    print(json.dumps(metrics, indent=1, default=str), flush=True)
    print(cl.sort_values("rank_matched").to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--query", required=True, help="degron_labeled.h5ad (raw counts, clone calls)")
    ap.add_argument("--bank", required=True, help="bank.npz (fig4_tahoe_encode.py --harvest-bank)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--condition-col", required=True)
    ap.add_argument("--control-condition", required=True)
    ap.add_argument("--treated-condition", required=True)
    ap.add_argument("--clone-col", required=True)
    ap.add_argument("--unassigned", default="unassigned")
    ap.add_argument("--mask-genes", default="")
    ap.add_argument("--max-pct-mt", type=float, default=None,
                    help="drop nuclei above this mito-read percentage (dead / damaged), before predict()")
    ap.add_argument("--mt-col", default="pct_counts_mt")
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
