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

"""Fig. 5 (degron inset): clone dropout under auxin vs DepMap essentiality. Data only, no model.

A degron of an essential gene should deplete its clone over 24 h of auxin. Per pool clone: nuclei
labelled under DMSO and under auxin (--clone-col), the auxin share normalised by the library sizes
(log2 of (n_iaa / N_iaa) / (n_dmso / N_dmso), +0.5 pseudocount), and the gene's mean DepMap Chronos
effect across lines (more negative = more essential). Writes degron_dropout.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main(a):
    import h5py
    from anndata.io import read_elem
    with h5py.File(a.query, "r") as f:
        o = read_elem(f["obs"])[[a.condition_col, a.clone_col]]
    pool = pd.read_csv(a.pool_tsv, sep="\t")
    cond = o[a.condition_col].astype(str)
    N = cond.value_counts()
    ct = pd.crosstab(o[a.clone_col].astype(str), cond)
    ct = ct.reindex(pool["gene"].astype(str).unique(), fill_value=0)
    dm = pd.read_csv(a.depmap_csv, index_col=0)
    dm.columns = [c.split(" (")[0] for c in dm.columns]
    eff = dm.mean(0)
    out = pd.DataFrame({"gene": ct.index, "n_dmso": ct.get(a.control_condition, 0).to_numpy(),
                        "n_iaa": ct.get(a.treated_condition, 0).to_numpy()})
    out["log2_iaa_vs_dmso"] = np.log2(((out.n_iaa + 0.5) / N[a.treated_condition]) /
                                      ((out.n_dmso + 0.5) / N[a.control_condition]))
    out["depmap_mean_effect"] = out.gene.map(eff)
    out["depmap_frac_lines_dependent"] = out.gene.map((dm < -0.5).mean(0))
    out.to_csv(a.out, index=False)
    lab = out[(out.n_dmso + out.n_iaa) >= a.min_labelled].dropna(subset=["depmap_mean_effect"])
    rho = lab["log2_iaa_vs_dmso"].corr(lab["depmap_mean_effect"], method="spearman")
    print(out.sort_values("log2_iaa_vs_dmso").to_string(index=False))
    print(f"[dropout] Spearman(log2 IAA/DMSO, DepMap effect) = {rho:.3f} over {len(lab)} clones "
          f"with >= {a.min_labelled} labelled nuclei", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query", required=True)
    ap.add_argument("--pool-tsv", required=True)
    ap.add_argument("--depmap-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition-col", required=True)
    ap.add_argument("--control-condition", required=True)
    ap.add_argument("--treated-condition", required=True)
    ap.add_argument("--clone-col", required=True)
    ap.add_argument("--min-labelled", type=int, default=5)
    main(ap.parse_args())
