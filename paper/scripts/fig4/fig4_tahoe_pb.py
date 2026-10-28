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

"""Fig. 5 on the FULL Tahoe-100M: drug -> target recovery from streamed pseudobulks (no subsampling).

Input: paper/analysis/tahoe_pseudobulk.py output, one row per (plate, cell line, drug, dose), X = per-cell
mean raw counts, each plate x line with its own DMSO row.

    encode  one predict() over every pseudobulk with >= --min-cells cells; the encoder's input
            centre = all DMSO pseudobulks (the pooled-DMSO arm, best for Tahoe on the subset).
            Raw z_shared written per row. X is streamed in row blocks and cut to the model's genes
            (the full matrix is 65,918 x 62,710).
    score   Δz = z(drug row) - z(DMSO row of the SAME plate x line). Scored at four levels:
              entry      drug x line x dose (~60k)
              drug_dose  mean Δz over lines
              drug       mean Δz over lines and doses
              drug_line  mean Δz over doses
            Trained targets ranked among the trained prototypes (Fig. 5b convention) and all
            targets over the full bank. Entry-level null: the other drugs' trained-target sets
            ranked in the same entry's ranking, within the same (line, dose) block.
            Writes per-level scores_*.npz + ranks_*.csv in the fig4_summary.py format, entries.csv
            (one row per drug x line x dose) and summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_tahoe_pb")
CTRL = "__ctrl__"


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def split(v):
    s = str(v).strip()
    return [t.strip() for t in s.replace(",", ";").split(";") if t.strip()] if s and s.lower() != "nan" else []


def ranks(S):
    rk = np.empty(S.shape, np.int32)
    np.put_along_axis(rk, np.argsort(-S, axis=1), np.arange(1, S.shape[1] + 1)[None, :], axis=1)
    return rk


# ---- encode ---------------------------------------------------------------------------------------
def encode(a):
    import anndata as ad
    import h5py
    import scipy.sparse as sp
    from anndata.io import read_elem
    import expert.utils.readout as ro
    from expert import ExPert
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    n_shared = int(model.module.align_ds_mean.shape[1])
    mgenes = np.asarray(model.adata.var_names).astype(str)
    with h5py.File(a.pseudobulk, "r") as f:
        obs = read_elem(f["obs"])
        var = np.asarray(read_elem(f["var"]).index).astype(str)
        vpos = {g: i for i, g in enumerate(var)}
        masked = []
        if a.mask_genes:
            pats = [x.strip() for x in a.mask_genes.split(",") if x.strip()]
            hit = lambda g: any(g.startswith(p[:-1]) if p.endswith("*") else g == p for p in pats)
            masked = [g for g in mgenes if g in vpos and hit(g)]
        # a masked gene is simply absent from the query: predict() zero-fills it in every row AND in
        # the control mean, so its centred input is exactly 0 and nothing else moves
        cols = np.array([vpos[g] for g in mgenes if g in vpos and g not in set(masked)])
        keep_genes = var[cols]
        log.info("input mask %r: %d model genes dropped %s", a.mask_genes, len(masked), masked[:12])
        rows = np.flatnonzero((obs.n_cells.to_numpy() >= a.min_cells) | obs.is_control.astype(bool).to_numpy())
        ip = f["X/indptr"][:]
        blocks = []
        for s in range(0, len(rows), a.block):
            r = rows[s:s + a.block]
            r0, r1 = int(r[0]), int(r[-1]) + 1
            p0, p1 = int(ip[r0]), int(ip[r1])
            X = sp.csr_matrix((f["X/data"][p0:p1], f["X/indices"][p0:p1], ip[r0:r1 + 1] - p0),
                              shape=(r1 - r0, len(var)))
            blocks.append(X[r - r0][:, cols])
            log.info("  read rows %d / %d", s + len(r), len(rows))
    X = sp.vstack(blocks).tocsr()
    o = obs.iloc[rows].copy()
    log.info("query %d pseudobulks x %d model genes (%d of %d present); %d DMSO", X.shape[0], X.shape[1],
             len(cols), len(mgenes), int(o.is_control.astype(bool).sum()))
    o["condition"] = np.where(o.is_control.astype(bool), CTRL, o.drug.astype(str))
    q = ad.AnnData(X=X, obs=o[["condition"]].copy(), var=pd.DataFrame(index=keep_genes))
    ro.build_groups(q, "condition", control_condition=CTRL)
    ro.mark_query(model, q)
    cls_col = model.adata_manager.get_state_registry("labels").original_key
    is_c = (q.obs[cls_col].astype(str) == "control").to_numpy()
    # INPUT centring = which DMSO pseudobulks share the forward pass (the encoder centres on them)
    key = {"global": None, "line": ["cell_name"], "plate": ["plate"], "plate_line": ["plate", "cell_name"]}[a.input_center]
    lab = (np.zeros(len(o), int) if key is None else
           pd.factorize(o[key].astype(str).agg("|".join, axis=1))[0])
    Z = np.full((len(o), n_shared), np.nan, np.float32)
    for gi in np.unique(lab):
        idx = np.flatnonzero(lab == gi)
        cidx = idx[is_c[idx]]
        if len(cidx) == 0:
            continue
        # predict() keeps the TRAINING baseline below 10 controls: repeat the subset's controls up to 10
        # rows (the mean, and so the centring, is unchanged); the copies are dropped after
        rep_n = int(np.ceil(10 / len(cidx))) - 1 if len(cidx) < 10 else 0
        take = np.concatenate([idx] + [cidx] * rep_n)
        sub = q[take].copy()
        sub.obs_names_make_unique()
        model.predict(sub, inplace=True, return_latents=True, soft_predictions=False, head="A", obs_only=True,
                      center_mode=np.zeros(n_shared, np.float32), control_label="control",
                      batch_size=a.batch_size, n_cores=a.n_cores, progress_bar=False)
        Z[idx] = np.asarray(sub.obsm[ro.LATENT_KEY], np.float32)[:len(idx)]
        model.deregister_manager()                       # scvi keeps every registered query otherwise
    log.info("input centring %r: %d subsets; %d rows without a control in their subset",
             a.input_center, len(np.unique(lab)), int(np.isnan(Z).any(1).sum()))
    np.save(out / "z_raw.npy", Z)
    o.reset_index(names="pb_id").to_parquet(out / "obs.parquet")
    json.dump(dict(model_dir=a.model_dir, pseudobulk=a.pseudobulk, n_rows=int(len(o)), min_cells=a.min_cells,
                   mask_genes=a.mask_genes, n_masked=len(masked), input_center=a.input_center,
                   n_model_genes_present=int(len(cols)), minutes=(time.time() - t0) / 60),
              open(out / "encode.json", "w"), indent=1)
    log.info("encoded in %.1f min", (time.time() - t0) / 60)


# ---- score ----------------------------------------------------------------------------------------
def score(a):
    t0 = time.time()
    enc, out = Path(a.encode_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = np.load(a.bank, allow_pickle=True)
    P, labels, n_lab = b["P"].astype(np.float32), b["labels"].astype(str), int(b["n_labels"])
    Pn = unit(P); Pt = Pn[:n_lab]
    pos = {g: i for i, g in enumerate(labels)}
    o = pd.read_parquet(enc / "obs.parquet")
    # categoricals from parquet make groupby().indices enumerate UNOBSERVED combinations
    for c in ("drug", "cell_name", "plate", "targets", "target_class", "moa_broad"):
        o[c] = o[c].astype(str)
    o["dose_uM"] = o.dose_uM.astype(float)
    # drug_metadata lists targets comma-separated ("PSMB1, PSMB2, PSMB5"); every scorer here splits
    # on ';' -- unnormalised, each multi-target drug became ONE unknown token and dropped out
    o["targets"] = o.targets.str.replace(r"\s*[,;]\s*", ";", regex=True).str.strip(";")
    Z = np.load(enc / "z_raw.npy").astype(np.float32)
    ctrl = o.is_control.astype(bool).to_numpy()
    o["pl"] = o.plate.astype(str) + "|" + o.cell_name.astype(str)
    cz = pd.Series(np.flatnonzero(ctrl), index=o.pl[ctrl].to_numpy())
    cz = cz[~cz.index.duplicated()]
    drug_rows = np.flatnonzero(~ctrl & o.pl.isin(cz.index).to_numpy())
    D = Z[drug_rows] - Z[cz.loc[o.pl.iloc[drug_rows]].to_numpy()]
    e = o.iloc[drug_rows].reset_index(drop=True)
    e["dz_norm"] = np.linalg.norm(D, axis=1)
    if a.lof_only:
        lof = e.moa_broad.astype(str).eq("inhibitor/antagonist").to_numpy()
        e, D = e[lof].reset_index(drop=True), D[lof]
    tr_set = [np.array([pos[g] for g in split(t) if g in pos and pos[g] < n_lab], int) for t in e.targets]
    all_set = [np.array([pos[g] for g in split(t) if g in pos], int) for t in e.targets]
    has_tr = np.array([len(s) > 0 for s in tr_set])
    log.info("%d drug entries (%d drugs, %d lines, doses %s); %d with a trained target",
             len(e), e.drug.nunique(), e.cell_name.nunique(), sorted(e.dose_uM.unique()), int(has_tr.sum()))

    # entry level: own rank + same-(line, dose) null, trained targets among trained prototypes
    rank_tr = np.full(len(e), -1); null_tr = np.full(len(e), np.nan); rank_full = np.full(len(e), -1)
    top5 = np.empty(len(e), dtype=object)
    for (ln, ds), ix in e.groupby(["cell_name", "dose_uM"]).indices.items():
        ix = np.asarray(ix)
        Dn = unit(D[ix])
        rk_f = ranks(Dn @ Pn.T)
        for k, i in enumerate(ix):
            if len(all_set[i]):
                rank_full[i] = rk_f[k, all_set[i]].min()
        top = np.argsort(rk_f, axis=1)[:, :5]
        top5[ix] = [";".join(labels[t]) for t in top]
        sub = ix[has_tr[ix]]
        if len(sub) == 0:
            continue
        rk_t = ranks(unit(D[sub]) @ Pt.T)
        # R[p, q] = best rank of drug q's trained targets in drug p's ranking (same line x dose)
        R = np.stack([rk_t[:, tr_set[q]].min(1) for q in sub], axis=1).astype(float)
        rank_tr[sub] = np.diag(R)
        if len(sub) > 1:
            np.fill_diagonal(R, np.nan)
            null_tr[sub] = np.nanmedian(R, axis=1)
    e["rank_trained"], e["null_rank_trained"], e["rank_full"], e["top5"] = rank_tr, null_tr, rank_full, top5
    e.drop(columns=["pl"]).to_csv(out / "entries.csv", index=False)
    log.info("entries scored (%.1f min)", (time.time() - t0) / 60)

    # aggregated levels in the fig4_summary.py format (keys, labels, S) + ranks_<level>.csv
    levels = {"drug_dose": ["drug", "dose_uM"], "drug": ["drug"], "drug_line": ["drug", "cell_name"]}
    for lev, cols in levels.items():
        g = e.groupby(cols).indices
        keys = ["|".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in g]
        R = np.stack([D[ix].mean(0) for ix in g.values()])
        np.savez_compressed(out / f"scores_{lev}.npz", keys=np.array(keys), labels=labels,
                            S=(unit(R) @ Pn.T).astype(np.float16))
        first = e.groupby(cols).first().reset_index()
        first["key"] = keys
        first[["key", "drug", "targets", "target_class", "moa_broad"] + [c for c in cols if c != "drug"]
              ].to_csv(out / f"ranks_{lev}.csv", index=False)
        log.info("level %s: %d rows", lev, len(keys))

    # entry-level summary: recall@K vs the same-block null, per K
    m = e[e.rank_trained > 0]
    res = dict(n_entries=int(len(e)), n_entries_trained=int(len(m)), n_drugs=int(e.drug.nunique()),
               n_lines=int(e.cell_name.nunique()), n_trained_prototypes=n_lab)
    for k in (1, 10, 50):
        res[f"entry_recall@{k}"] = float((m.rank_trained <= k).mean())
    res["entry_median_rank"] = float(m.rank_trained.median())
    res["entry_median_null_rank"] = float(m.null_rank_trained.median())
    json.dump(res, open(out / "summary.json", "w"), indent=1)
    log.info("summary %s (%.1f min)", res, (time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("encode", "score"))
    ap.add_argument("--pseudobulk", help="encode: tahoe100m_pseudobulk_plate_line_drug_dose.h5ad")
    ap.add_argument("--model-dir", help="encode")
    ap.add_argument("--encode-dir", help="score: encode output dir")
    ap.add_argument("--bank", help="score: bank.npz (19,895 rows, n_labels trained)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--mask-genes", default="", help="encode: input mask, e.g. 'MT-*,MT1*,MT2A,MT3,MT4'")
    ap.add_argument("--input-center", default="global", choices=("global", "line", "plate", "plate_line"),
                    help="encode: which DMSO pseudobulks share a forward pass")
    ap.add_argument("--lof-only", action="store_true", help="score: inhibitor/antagonist drugs only")
    ap.add_argument("--block", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-cores", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    encode(a) if a.stage == "encode" else score(a)
