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

"""Does the model's GENE-LEVEL reading of a drug predict which cell lines that drug hits?

ExPert reads every (drug, cell line, dose) response as a ranking over CRISPR knockdown prototypes.
If that reading is real, a drug should deplete the lines that actually DEPEND on the genes it
names. The dependency comes from DepMap (Chronos gene effect, orthogonal to anything here) and the
depletion from the screen's own cell counts.

    score(drug, line)   dependency-weighted reading: mean over the drug's top-k prototypes IN THAT
                        LINE of (-Chronos gene effect of that gene in that line). High = the model
                        named genes this line needs.
    growth(drug, line)  log2 cell count, with the DRUG main effect and the LINE main effect removed
                        (two-way median polish). The raw count is unusable: DMSO wells hold ~2.3x
                        more cells by design, so every drug looks depleted. The interaction is what
                        differential sensitivity lives in.
    test                within-drug Spearman of growth vs score across lines, averaged over drugs
                        (so drug potency and line growth rate are both held fixed).
    null                DepMap line labels permuted -- the same dependency matrix, wrong lines --
                        which keeps gene essentiality structure and kills only the line match.

Also reports the same test with the hub-filtered ranking, and a per-drug table.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_depmap")
norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def median_polish(T, n_iter=10):
    """Remove row and column main effects; return the interaction residual."""
    R = T.copy()
    for _ in range(n_iter):
        r = np.nanmedian(R, axis=1, keepdims=True); R -= r
        c = np.nanmedian(R, axis=0, keepdims=True); R -= c
    return R


def main(a):
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    b = np.load(a.bank, allow_pickle=True)
    nl = int(b["n_labels"]); L = b["labels"].astype(str)[:nl]; P = unit(b["P"].astype(np.float32))[:nl]

    # ---- DepMap: gene effect per (line, gene) ----------------------------------------------------
    models = pd.read_csv(a.depmap_models)
    id2name = {r.ModelID: norm(r.StrippedCellLineName) for r in models.itertuples()
               if isinstance(r.StrippedCellLineName, str)}
    dm = pd.read_csv(a.depmap_csv, index_col=0)
    dm.columns = [c.split(" (")[0] for c in dm.columns]
    dm = dm.loc[:, ~dm.columns.duplicated()]
    dm.index = [id2name.get(i, i) for i in dm.index]
    dm = dm[~dm.index.duplicated()]
    gcols = [g for g in L if g in dm.columns]
    gi = np.array([list(L).index(g) for g in gcols])
    DE = dm[gcols].to_numpy(np.float32)                          # lines x genes (subset of prototypes)
    log.info("DepMap %d lines x %d genes (of %d prototypes)", *DE.shape, nl)

    # ---- encodings, active responses -------------------------------------------------------------
    o = pd.read_parquet(Path(a.encode_dir) / "obs.parquet")
    Z = np.load(Path(a.encode_dir) / "z_raw.npy").astype(np.float32)
    for c in ("drug", "cell_name", "plate", "targets", "moa_broad", "moa_fine"):
        o[c] = o[c].astype(str)
    ok = ~np.isnan(Z).any(1)
    ctrl = o.is_control.astype(bool).to_numpy() & ok
    lm = pd.DataFrame(Z[ctrl], index=o.cell_name[ctrl].to_numpy()).groupby(level=0).mean()
    thr = float(np.quantile(np.linalg.norm(Z[ctrl] - lm.loc[o.cell_name[ctrl]].to_numpy(), axis=1), 0.95))
    d = (~o.is_control.astype(bool).to_numpy()) & ok & o.moa_broad.eq("inhibitor/antagonist").to_numpy()
    e = o[d].reset_index(drop=True)
    D = Z[d] - lm.loc[e.cell_name].to_numpy(np.float32)
    act = np.linalg.norm(D, axis=1) > thr
    e, D = e[act].reset_index(drop=True), D[act]
    e["line_key"] = e.cell_name.map(norm)
    have = e.line_key.isin(dm.index).to_numpy()
    e, D = e[have].reset_index(drop=True), D[have]
    S = unit(D) @ P.T
    log.info("active responses with DepMap: %d (%d drugs, %d lines)", len(e), e.drug.nunique(),
             e.line_key.nunique())

    # hub-filtered ranking: leave-one-DRUG-out column z over drug consensus profiles
    gd = e.groupby("drug").indices
    drugs_all = list(gd)
    Cp = np.stack([S[gd[x]].mean(0) for x in drugs_all])
    mu, sq, n = Cp.sum(0), (Cp ** 2).sum(0), len(Cp)
    mu_lo = (mu[None] - Cp) / (n - 1)
    sd_lo = np.sqrt(np.maximum((sq[None] - Cp ** 2) / (n - 1) - mu_lo ** 2, 1e-12))
    dpos = {x: i for i, x in enumerate(drugs_all)}
    Sz = (S - mu_lo[[dpos[x] for x in e.drug]]) / sd_lo[[dpos[x] for x in e.drug]]

    # ---- growth: interaction residual of log2 cell count -----------------------------------------
    cnt = (e.assign(lg=np.log2(e.n_cells.to_numpy() + 1))
           .pivot_table(index="drug", columns="line_key", values="lg", aggfunc="mean"))
    resid = pd.DataFrame(median_polish(cnt.to_numpy(np.float64)), index=cnt.index, columns=cnt.columns)
    log.info("growth matrix %s; residual sd %.3f", resid.shape, float(np.nanstd(resid.to_numpy())))

    # ---- dependency-weighted reading per (drug, line) --------------------------------------------
    lpos = {l: i for i, l in enumerate(dm.index)}
    rows = []
    for which, SC in (("raw", S), ("hub", Sz)):
        top = np.argsort(-SC, axis=1)[:, :a.top_k]
        for i in range(len(e)):
            li = lpos[e.line_key.iat[i]]
            g = [j for j in top[i] if j in set(gi)]
            if not g:
                continue
            cols = [gcols.index(L[j]) for j in g]
            rows.append(dict(which=which, drug=e.drug.iat[i], line=e.line_key.iat[i],
                             dose=float(e.dose_uM.iat[i]), score=float(-np.nanmean(DE[li, cols])),
                             n_genes=len(cols)))
    sc = pd.DataFrame(rows)
    sc = sc.groupby(["which", "drug", "line"], as_index=False).score.mean()
    sc["growth"] = [resid.at[r.drug, r.line] if (r.drug in resid.index and r.line in resid.columns)
                    else np.nan for r in sc.itertuples()]
    sc = sc.dropna(subset=["growth"])
    sc.to_csv(out / "depmap_scores.csv", index=False)

    # ---- test: within-drug Spearman across lines --------------------------------------------------
    from scipy.stats import spearmanr, wilcoxon
    res = {}
    for which in ("raw", "hub"):
        q = sc[sc.which == which]
        per = []
        for drug, g in q.groupby("drug"):
            if len(g) >= a.min_lines and g.score.std() > 0 and g.growth.std() > 0:
                per.append((drug, float(spearmanr(g.score, g.growth).statistic), len(g)))
        pr = pd.DataFrame(per, columns=["drug", "rho", "n_lines"])
        # null: permute DepMap line labels (same dependency matrix, wrong lines)
        # null: within each drug, shuffle which LINE each dependency score belongs to. The earlier
        # version remapped to the whole DepMap index, so almost every permuted line was absent from
        # the drug's 38 lines and the merge produced no rows (null = nan).
        null = []
        groups = [(drug, g) for drug, g in q.groupby("drug")
                  if len(g) >= a.min_lines and g.score.std() > 0 and g.growth.std() > 0]
        for _ in range(a.n_perm):
            vals = [spearmanr(rng.permutation(g.score.to_numpy()), g.growth.to_numpy()).statistic
                    for _, g in groups]
            null.append(float(np.nanmean(vals)))
        obs = float(pr.rho.mean())
        p = (np.sum(np.abs(np.array(null)) >= abs(obs)) + 1) / (len(null) + 1) if null else np.nan
        w = wilcoxon(pr.rho).pvalue if len(pr) > 5 else np.nan
        res[which] = dict(n_drugs=len(pr), mean_rho=obs, median_rho=float(pr.rho.median()),
                          null_mean=float(np.mean(null)) if null else np.nan, p_perm=float(p),
                          p_wilcoxon=float(w), frac_negative=float((pr.rho < 0).mean()))
        log.info("%s ranking: within-drug Spearman(growth residual, dependency score) mean %.4f "
                 "(median %.4f) over %d drugs; null %.4f; p_perm %.3f; wilcoxon p %.2g",
                 which, obs, pr.rho.median(), len(pr), res[which]["null_mean"], p, w)
        pr.to_csv(out / f"depmap_per_drug_{which}.csv", index=False)
        if len(pr):
            log.info("  most negative (model names genes the line needs -> fewer cells): %s",
                     ", ".join(f"{r.drug} {r.rho:+.2f}" for r in pr.nsmallest(6, "rho").itertuples()))
    json.dump(dict(encode_dir=a.encode_dir, top_k=a.top_k, min_lines=a.min_lines,
                   n_depmap_lines=int(DE.shape[0]), n_genes=int(DE.shape[1]), results=res),
              open(out / "depmap_meta.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encode-dir", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--depmap-csv", required=True)
    ap.add_argument("--depmap-models", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--min-lines", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
