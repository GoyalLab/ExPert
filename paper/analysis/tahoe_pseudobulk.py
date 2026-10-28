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

"""Tahoe-100M -> pseudobulks per (plate, cell line, drug, dose), streamed over EVERY cell.

No subsampling. One plate per call (array over the 14 plates), then `--merge`.

    stage 'plate'  stream X in --block rows (sequential h5py reads; the source X is uncompressed)
                   and add each block into two dense accumulators through a sparse group-indicator
                   matmul:
                     sum of raw counts                 -> X        = mean counts per cell
                     sum of log1p(counts per 10k)      -> layers['log1p_cp10k'] = mean per cell
                   Group = drugname_drugconc x cell_name within the plate (each plate carries one
                   dose per drug plus its own DMSO). Cells with pass_filter != --pass-filter are
                   excluded and counted. Also: n_cells, mean library size, mean genes detected,
                   phase fractions.
    stage 'merge'  concatenate the plates, parse drug / dose, mark DMSO, attach targets and MoA
                   from the drug metadata and our target map.

X of the output is the per-cell MEAN of raw counts: the same form as the training-corpus
pseudobulks, so ExPert can encode it directly (predict() applies log1p itself).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("tahoe_pseudobulk")
DOSE_RGX = re.compile(r"'([^']*)',\s*([0-9.]+),\s*'uM'")


def _cat(obs, col):
    g = obs[col]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
    return g["codes"][:].astype(np.int64), cats


def plate_stage(a):
    t0 = time.time()
    with h5py.File(a.plate_h5ad, "r") as f:
        obs = f["obs"]
        dc, dcats = _cat(obs, "drugname_drugconc")
        lc, lcats = _cat(obs, "cell_name")
        pc, pcats = _cat(obs, "pass_filter")
        ph, phcats = (_cat(obs, "phase") if "phase" in obs and isinstance(obs["phase"], h5py.Group)
                      else (None, None))
        n_obs, n_vars = (int(x) for x in f["X"].attrs["shape"])
        genes = np.array([g.decode() if isinstance(g, bytes) else g for g in f["var"]["gene_name"][:]])
        keep = pcats[pc] == a.pass_filter
        if a.max_rows:                                   # smoke: count only the rows actually read
            keep &= np.arange(len(keep)) < a.max_rows
        gid_all = dc * len(lcats) + lc
        used = np.unique(gid_all[keep])
        remap = np.full(len(dcats) * len(lcats), -1, np.int64)
        remap[used] = np.arange(len(used))
        gid = np.where(keep, remap[gid_all], -1)
        G = len(used)
        log.info("%s: %d cells (%d %s), %d genes, %d groups", Path(a.plate_h5ad).name, n_obs,
                 int(keep.sum()), a.pass_filter, n_vars, G)

        S_raw = np.zeros((G, n_vars), np.float64)
        S_log = np.zeros((G, n_vars), np.float64)
        n_cells = np.bincount(gid[keep], minlength=G).astype(np.int64)
        excl = ~(pcats[pc] == a.pass_filter)
        if a.max_rows:
            excl &= np.arange(len(excl)) < a.max_rows
        re_ex = remap[gid_all[excl]]
        n_excl = np.bincount(re_ex[re_ex >= 0], minlength=G)
        lib_sum = np.zeros(G); ngen_sum = np.zeros(G)
        ph_cnt = np.zeros((G, len(phcats))) if ph is not None else None

        indptr = f["X/indptr"]
        data, indices = f["X/data"], f["X/indices"]
        stop = min(n_obs, a.max_rows) if a.max_rows else n_obs
        for r0 in range(0, stop, a.block):
            r1 = min(stop, r0 + a.block)
            ip = indptr[r0:r1 + 1].astype(np.int64)
            d = data[ip[0]:ip[-1]].astype(np.float32)
            ix = indices[ip[0]:ip[-1]]
            X = sp.csr_matrix((d, ix, ip - ip[0]), shape=(r1 - r0, n_vars))
            g = gid[r0:r1]
            ok = g >= 0
            if ok.any():
                rows = np.flatnonzero(ok)
                Gm = sp.csr_matrix((np.ones(len(rows), np.float32), (g[rows], rows)), shape=(G, r1 - r0))
                lib = np.asarray(X.sum(1)).ravel()
                S_raw += (Gm @ X).toarray()
                Xl = sp.diags((1e4 / np.maximum(lib, 1)).astype(np.float32)) @ X
                Xl.data = np.log1p(Xl.data)
                S_log += (Gm @ Xl).toarray()
                lib_sum += np.bincount(g[rows], weights=lib[rows], minlength=G)
                ngen_sum += np.bincount(g[rows], weights=np.diff(ip)[rows], minlength=G)
                if ph_cnt is not None:
                    np.add.at(ph_cnt, (g[rows], ph[r0:r1][rows]), 1)
            log.info("  rows %d-%d / %d  (%.1f min)", r0, r1, stop, (time.time() - t0) / 60)

    nz = np.maximum(n_cells, 1)[:, None]
    di, li = np.divmod(used, len(lcats))
    out = pd.DataFrame({"plate": Path(a.plate_h5ad).name.split("_")[0], "drugname_drugconc": dcats[di],
                        "cell_name": lcats[li], "n_cells": n_cells, "n_cells_excluded": n_excl[:G],
                        "mean_total_counts": lib_sum / nz[:, 0], "mean_genes_detected": ngen_sum / nz[:, 0]})
    if ph_cnt is not None:
        for j, p in enumerate(phcats):
            out[f"frac_{p}"] = ph_cnt[:, j] / nz[:, 0]
    np.savez(a.out, mean_raw=(S_raw / nz).astype(np.float32), mean_log=(S_log / nz).astype(np.float32),
             genes=genes, obs=out.to_json(orient="split"))
    log.info("wrote %s (%d groups) in %.1f min", a.out, G, (time.time() - t0) / 60)


def merge_stage(a):
    import anndata as ad
    parts = sorted(glob.glob(str(Path(a.parts_dir) / "plate*.npz")),
                   key=lambda s: int(re.search(r"plate(\d+)", s).group(1)))
    obs, Xr, Xl, genes = [], [], [], None
    for p in parts:
        z = np.load(p, allow_pickle=True)
        genes = z["genes"] if genes is None else genes
        if not np.array_equal(genes, z["genes"]):
            raise SystemExit(f"{p}: gene order differs")
        obs.append(pd.read_json(z["obs"].item(), orient="split"))
        Xr.append(sp.csr_matrix(z["mean_raw"])); Xl.append(sp.csr_matrix(z["mean_log"]))
        log.info("loaded %s: %d groups", Path(p).name, len(obs[-1]))
    o = pd.concat(obs, ignore_index=True)
    parsed = o.drugname_drugconc.str.extract(DOSE_RGX)
    o["drug"] = parsed[0].str.strip()
    o["dose_uM"] = parsed[1].astype(float)
    o["is_control"] = o.drug.eq(a.control_drug)
    meta = pd.read_parquet(a.drug_metadata).rename(columns={"moa-broad": "moa_broad", "moa-fine": "moa_fine"})
    meta["drug"] = meta.drug.str.strip()
    o = o.merge(meta[["drug", "targets", "moa_broad", "moa_fine"]].drop_duplicates("drug"),
                on="drug", how="left")
    if a.target_map:
        tm = pd.read_csv(a.target_map)
        tm["drug"] = tm.drug.str.strip()
        o = o.merge(tm[["drug", "target_class", "trained_targets", "zeroshot_targets"]].drop_duplicates("drug"),
                    on="drug", how="left")
    o.index = (o.plate + "|" + o.cell_name + "|" + o.drug + "|" + o.dose_uM.astype(str)).values
    for c in o.columns:
        if o[c].dtype == object:
            o[c] = o[c].fillna("").astype(str)
    A = ad.AnnData(X=sp.vstack(Xr).tocsr(), obs=o, var=pd.DataFrame(index=pd.Index(genes.astype(str))))
    A.layers["log1p_cp10k"] = sp.vstack(Xl).tocsr()
    A.uns["tahoe_pseudobulk"] = json.dumps(dict(
        X="per-cell MEAN of raw counts per (plate, cell_name, drug, dose)",
        log1p_cp10k="per-cell MEAN of log1p(counts per 10k)",
        pass_filter=a.pass_filter, n_plates=len(parts), source=a.parts_dir))
    A.write_h5ad(a.out, compression="gzip")
    log.info("merged %d pseudobulks x %d genes -> %s; %d drugs, %d lines, doses %s, DMSO rows %d",
             A.n_obs, A.n_vars, a.out, o.drug.nunique(), o.cell_name.nunique(),
             sorted(o.dose_uM.unique()), int(o.is_control.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("plate", "merge"))
    ap.add_argument("--plate-h5ad", help="plate: one plate*_filt_*.h5ad")
    ap.add_argument("--out", required=True, help="plate: .npz part; merge: .h5ad")
    ap.add_argument("--parts-dir", help="merge: directory of plate*.npz parts")
    ap.add_argument("--drug-metadata", help="merge: drug_metadata.parquet")
    ap.add_argument("--target-map", default=None, help="merge: tahoe100m_drug_target_map.csv")
    ap.add_argument("--control-drug", default="DMSO_TF")
    ap.add_argument("--pass-filter", default="full")
    ap.add_argument("--block", type=int, default=250_000,
                    help="rows per read; ~1,450 nnz/cell -> 250k rows ~ 3.6e8 elements, ~8 GB peak")
    ap.add_argument("--max-rows", type=int, default=0, help="plate: stop after N rows (smoke)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    plate_stage(a) if a.stage == "plate" else merge_stage(a)
