# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Supplementary Fig. S7 data: Figure 4B-E read by a model that never saw K562 (EFFGATE3, fold 4).

EFFGATE3 was trained with ReplogleWeissman2022_K562_gwps and _K562_essential held out -- no K562 cell
and no K562 screen -- but it keeps the full label space, and 3,800 of the 3,802 candidates still have
training cells from other screens. So this asks whether Figure 4 rests on the K562 screen or on what the
model learned from the other contexts. The EFFGATE3 panels are scored on EXACTLY the conditions of
Figure 4 (the responsiveness mask comes from the shipped v0.1.0 encode, --resp-encode-dir), so the
fingerprinting, centroid and random arms are identical to the main figure and only ExPert changes.

Writes the panel tables (ExPert = EFFGATE3) and comparison.csv (v0.1.0 vs EFFGATE3, same conditions).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

LABEL = "ExPert - no K562"
PANELS = ["fig4b_sciplex_curve.csv", "fig4b_per_drug.csv", "fig4c_sciplex_dose.csv",
          "fig4c_per_dose.csv", "fig4d_hdac_by_dose.csv", "fig4d_per_pair.csv"]


def per_drug(d):
    t = pd.read_csv(Path(d) / "fig4b_per_drug.csv", keep_default_na=False)
    t["key"] = t["key"].replace("", "<unnamed>")
    t["rank"] = pd.to_numeric(t["rank"])
    return t[t.method == "ExPert"].set_index("key")["rank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eg3-panels", required=True, help="EFFGATE3 panels dir (fig4.sbatch MDIR_OVERRIDE run)")
    ap.add_argument("--v010-panels", required=True, help="shipped Figure 4 panels dir")
    ap.add_argument("--flexplex", required=True, help="fig4e_ar42.csv from fig4_flexplex.py (both models)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in PANELS:                             # the series is named for what it is in the SI
        t = pd.read_csv(Path(a.eg3_panels) / f, keep_default_na=False)
        t.loc[t.method == "ExPert", "method"] = LABEL
        # the shipped model as a reference line, straight from the Figure 4 table (same conditions)
        v = pd.read_csv(Path(a.v010_panels) / f, keep_default_na=False)
        t = pd.concat([t, v[v.method == "ExPert"]], ignore_index=True)
        t.to_csv(out / f, index=False)
    # the comparators must be the published ones: refuse to build if they drifted
    for f in ("fig4b_sciplex_curve.csv", "fig4d_hdac_by_dose.csv"):
        x, y = pd.read_csv(Path(a.eg3_panels) / f), pd.read_csv(Path(a.v010_panels) / f)
        cx, cy = x[x.method != "ExPert"].reset_index(drop=True), y[y.method != "ExPert"].reset_index(drop=True)
        assert cx.equals(cy), f"{f}: comparator arms differ from Figure 4 -- not the same conditions"
    e = pd.read_csv(a.flexplex)
    e = e.replace({"method": {"ExPert, no K562 training": LABEL}})     # both models + fingerprinting
    e.to_csv(out / "fig4e_ar42.csv", index=False)

    rows = []
    for lab, d in (("v0.1.0", a.v010_panels), ("EFFGATE3 (no K562)", a.eg3_panels)):
        c = pd.read_csv(Path(d) / "fig4b_sciplex_curve.csv")
        c = c[c.method == "ExPert"].set_index("k").recall
        h = pd.read_csv(Path(d) / "fig4d_hdac_by_dose.csv")
        h = h[h.method == "ExPert"].sort_values("dose_uM").frac_top10.to_numpy()
        rows.append({"model": lab, **{f"top{k}_recall": float(c[k]) for k in (1, 10, 50)},
                     **{f"hdac_top10_{z}": float(v) for z, v in zip(("0.01uM", "0.1uM", "1uM", "10uM"), h)}})
    a1, b1 = per_drug(a.v010_panels), per_drug(a.eg3_panels)
    j = pd.concat([a1.rename("v010"), b1.rename("eg3")], axis=1).dropna()
    x = int(((j.v010 <= 10) & (j.eg3 > 10)).sum()); y = int(((j.eg3 <= 10) & (j.v010 > 10)).sum())
    cmp_ = pd.DataFrame(rows)
    cmp_["n_drugs"] = len(j)
    cmp_.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame([dict(test="top-10 per drug, v0.1.0 vs EFFGATE3 (McNemar)", v010_only=x, eg3_only=y,
                       both=int(((j.v010 <= 10) & (j.eg3 <= 10)).sum()),
                       p=binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)]).to_csv(out / "comparison_test.csv",
                                                                                    index=False)
    print(cmp_.round(3).to_string(index=False))
    print(f"v0.1.0-only {x}, EFFGATE3-only {y} of {len(j)} drugs")


if __name__ == "__main__":
    main()
