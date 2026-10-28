# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""sci-Plex3 -> RNA-fingerprinting export for ONE cell line, so fingerprinting can be run per line.

The published comparison (Figure 4B-D) ran fingerprinting on K562 only, while ExPert scored all three
lines; this rebuilds the fingerprinting input for any line with exactly the collaborator's two steps,
merged:

  02_prep_query.py      cells of --cell-line at 24 h with a perturbation; human (ENSG) genes only, the
                        ':N' PAR-Y duplicates dropped; condition = 'Vehicle' or '<drug>@<dose>nM'
  fp_sciplex_export.py  every Vehicle cell + up to --max-cells per condition (one seeded generator,
                        conditions in sorted order); genes with mean UMI >= --min-mean over the line

and writes the same genes x cells CSC binaries fp_sciplex_pip.R reads. For K562 this reproduces the
published export cell for cell (checked against its barcodes with --check-barcodes).

`--models` also writes input_genes_shared.txt: the input genes common to the given checkpoints, which
fp_sciplex_pip.R restricts fingerprinting to (ExPert v0.1.0 and EFFGATE3, 3,096 genes).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="SrivatsanTrapnell2020_sciplex3.h5ad (scPerturb)")
    ap.add_argument("--cell-line", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=24)
    ap.add_argument("--max-cells", type=int, default=200, help="0 = keep all cells")
    ap.add_argument("--min-mean", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="", help="model dirs, comma separated -> input_genes_shared.txt")
    ap.add_argument("--check-barcodes", default="", help="a published barcodes.txt to compare against")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- 02_prep_query.py ----------------------------------------------------------------------
    A = ad.read_h5ad(a.raw, backed="r")
    o = A.obs
    cells = np.flatnonzero((o["cell_line"].astype(str) == a.cell_line).to_numpy()
                           & (pd.to_numeric(o["time"], errors="coerce") == a.time).to_numpy()
                           & o["perturbation"].notna().to_numpy())
    vn = pd.Index(A.var_names.astype(str))
    gk = (A.var["ensembl_id"].astype(str).str.startswith("ENSG").to_numpy()
          & ~vn.str.match(r"^.+:\d+$"))
    q = A[cells].to_memory()[:, np.flatnonzero(gk)].copy()
    assert q.var_names.is_unique
    X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    s = q.obs
    drug = s["perturbation"].astype(str).str.strip()
    veh = drug.eq("control").to_numpy()
    dose = pd.to_numeric(s["dose_value"], errors="coerce").fillna(0).astype(int).to_numpy()
    drug = np.where(veh, "Vehicle", drug)
    cond = np.where(veh, "Vehicle", pd.Series(drug) + "@" + pd.Series(dose).astype(str) + "nM")
    print(f"{a.cell_line}: {len(cells):,} cells at {a.time:g} h, {X.shape[1]:,} human genes", flush=True)

    # ---- fp_sciplex_export.py ------------------------------------------------------------------
    mean = np.asarray(X.mean(0)).ravel()
    gkeep = np.flatnonzero(mean >= a.min_mean)
    rng = np.random.default_rng(a.seed)
    keep = []
    # GROUP ORDER MATTERS: one generator is shared across groups, so the draw for each group depends
    # on every group before it. The published export iterated its query's categorical in category
    # order, and anndata NATSORTS string categories when it writes an h5ad (2-Methoxyestradiol before
    # (+)-JQ1, @10nM before @10000nM). Plain sorted() keeps every <=200-cell group identical but
    # resamples all the larger ones differently (122,976 of 144,317 cells shared).
    from natsort import natsorted
    gi = pd.Series(cond).groupby(cond).indices
    for c in natsorted(gi):
        idx = gi[c]
        keep.append(idx if c == "Vehicle" or a.max_cells <= 0 else rng.permutation(idx)[:a.max_cells])
    keep = np.sort(np.concatenate(keep))
    X = X[keep][:, gkeep].tocsr()
    X.data = np.round(X.data).astype(np.int32)
    # R reads genes x cells CSC; the CSR of cells x genes has exactly that layout
    X.indptr.astype("<i4").tofile(out / "indptr.bin")
    X.indices.astype("<i4").tofile(out / "indices.bin")
    X.data.astype("<i4").tofile(out / "data.bin")
    (out / "genes.txt").write_text("\n".join(map(str, q.var_names[gkeep])) + "\n")
    bc = q.obs_names[keep].astype(str)
    (out / "barcodes.txt").write_text("\n".join(bc) + "\n")
    pd.DataFrame({"condition": cond[keep], "drug": drug[keep], "dose_nM": dose[keep]},
                 index=bc).to_csv(out / "meta.csv")
    info = (f"cells {len(keep)} (control Vehicle: {int((cond[keep] == 'Vehicle').sum())}, max per condition: "
            f"{a.max_cells or 'all'})\ngenes {len(gkeep)} of {q.n_vars} (mean UMI >= {a.min_mean})\n"
            f"conditions {len(np.unique(cond[keep]))}\n")
    (out / "export_info.txt").write_text(info)
    print(info, end="", flush=True)

    if a.check_barcodes:
        ref = Path(a.check_barcodes).read_text().split()
        same = list(bc) == ref
        print(f"[check] barcodes vs {a.check_barcodes}: {'IDENTICAL' if same else 'DIFFER'} "
              f"(ours {len(bc):,}, published {len(ref):,}, shared {len(set(bc) & set(ref)):,})", flush=True)

    if a.models:
        from expert import ExPert
        gs = None
        for m in a.models.split(","):
            g = set(map(str, ExPert.load(m).adata.var_names))
            gs = g if gs is None else gs & g
        (out / "input_genes_shared.txt").write_text("\n".join(sorted(gs)) + "\n")
        print(f"input_genes_shared.txt: {len(gs):,} genes common to {a.models.count(',') + 1} models", flush=True)


if __name__ == "__main__":
    main()
