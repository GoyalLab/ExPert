#!/usr/bin/env python
# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert: a contrastive perturbation-prediction model package and is
# released under the MIT License. See the LICENSE file in the project root for the full text.

"""What the harmonized space buys beyond target recovery: Tahoe-100M, active responses only.

The model puts 276 drugs x 50 cell lines into ONE gene-prototype space, so three questions that a
per-screen benchmark cannot ask:

  harmony   is a drug's GENE-LEVEL reading the same in two cell lines even when its raw expression
            response is not? Per drug, the between-line correlation of the response in expression
            (Δx), in the latent (Δz) and in the prototype profile (cosine to all trained atoms),
            each against a between-DRUG baseline computed the same way.
  graph     drug x drug similarity of the consensus prototype profile: does annotated MoA come out
            (kNN enrichment vs label shuffle), and which strong edges CROSS MoA labels
            (candidate shared mechanism / polypharmacology)?
  hidden    genes a drug ranks high in MANY cell lines, that are NOT its annotated targets and are
            specific to it (high z against the other drugs' profiles) -- hypothesis candidates.

Active entry = |Δz| above the 95th percentile of DMSO-vs-DMSO noise.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_harmonized")


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def split(v):
    s = str(v).strip()
    return [t.strip() for t in s.replace(",", ";").split(";") if t.strip() and t.strip() != "nan"] \
        if s and s.lower() != "nan" else []


def corr_rows(A, B):
    """Row-wise Pearson between paired matrices."""
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)


def main(a):
    import h5py
    import scipy.sparse as sp
    from anndata.io import read_elem
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    b = np.load(a.bank, allow_pickle=True)
    nl = int(b["n_labels"]); L = b["labels"].astype(str)[:nl]; P = unit(b["P"].astype(np.float32))[:nl]
    pos = {g: i for i, g in enumerate(L)}

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
    Sp = unit(D) @ P.T                                        # prototype profile per active entry
    log.info("active entries %d, drugs %d, lines %d", len(e), e.drug.nunique(), e.cell_name.nunique())

    # ---- harmony: the same drug in two cell lines ------------------------------------------------
    if a.pseudobulk:
        with h5py.File(a.pseudobulk, "r") as f:
            var = np.asarray(read_elem(f["var"]).index).astype(str)
            obs_all = read_elem(f["obs"])
            key_all = (obs_all.plate.astype(str) + "|" + obs_all.cell_name.astype(str) + "|"
                       + obs_all.drug.astype(str) + "|" + obs_all.dose_uM.astype(str)).to_numpy()
            rowmap = pd.Series(np.arange(len(key_all)), index=key_all)
            want = (e.plate + "|" + e.cell_name + "|" + e.drug + "|" + e.dose_uM.astype(str)).to_numpy()
            ctl_key = (obs_all.plate.astype(str) + "|" + obs_all.cell_name.astype(str))[
                obs_all.is_control.astype(bool).to_numpy()]
            ridx = rowmap.reindex(want).to_numpy()
            keep_x = ~np.isnan(ridx)
            lay = f["layers/log1p_cp10k"]; ip = lay["indptr"][:]
            hv = np.argsort(-np.asarray(read_elem(f["obs"]).n_cells))[:0]        # placeholder
            def rows_of(idx):
                blocks = []
                for r in idx:
                    s0, s1 = int(ip[r]), int(ip[r + 1])
                    blocks.append(sp.csr_matrix((lay["data"][s0:s1], lay["indices"][s0:s1], [0, s1 - s0]),
                                                shape=(1, len(var))))
                return sp.vstack(blocks).toarray()
            # control row per (plate, line) for the Δx baseline
            cmap = pd.Series(np.flatnonzero(obs_all.is_control.astype(bool).to_numpy()),
                             index=ctl_key.to_numpy())
            cmap = cmap[~cmap.index.duplicated()]
            pl = (e.plate + "|" + e.cell_name).to_numpy()
            sel = np.flatnonzero(keep_x & np.isin(pl, cmap.index))
            if a.max_x and len(sel) > a.max_x:
                sel = np.sort(rng.choice(sel, a.max_x, replace=False))
            X = rows_of(ridx[sel].astype(int)) - rows_of(cmap.loc[pl[sel]].to_numpy())
            # restrict to variable genes so the correlation is not dominated by zeros
            v = X.var(0)
            gi = np.argsort(-v)[:a.n_hvg]
            X = X[:, gi]
            ex = e.iloc[sel].reset_index(drop=True)
            Dx = D[sel]; Spx = Sp[sel]
            log.info("Δx matrix %s (%d HVGs)", X.shape, len(gi))
    rows = []
    src = ex if a.pseudobulk else e
    mats = ({"Δx (expression)": X, "Δz (latent)": Dx, "prototype profile": Spx} if a.pseudobulk
            else {"Δz (latent)": D, "prototype profile": Sp})
    for name, Mt in mats.items():
        same_i, same_j, diff_i, diff_j = [], [], [], []
        g = src.groupby("drug").indices
        for drug, ix in g.items():
            lines = src.cell_name.to_numpy()[ix]
            for p in range(len(ix)):
                for q in range(p + 1, len(ix)):
                    if lines[p] != lines[q]:
                        same_i.append(ix[p]); same_j.append(ix[q])
        # between-drug baseline: pairs from DIFFERENT drugs in DIFFERENT lines, same count
        allix = np.arange(len(src))
        for _ in range(len(same_i)):
            while True:
                p, q = rng.choice(allix, 2, replace=False)
                if src.drug.iat[p] != src.drug.iat[q] and src.cell_name.iat[p] != src.cell_name.iat[q]:
                    diff_i.append(p); diff_j.append(q); break
        s = corr_rows(Mt[same_i], Mt[same_j]); dd = corr_rows(Mt[diff_i], Mt[diff_j])
        rows.append(dict(space=name, n_pairs=len(s), same_drug_median=float(np.median(s)),
                         other_drug_median=float(np.median(dd)), gap=float(np.median(s) - np.median(dd))))
        log.info("%-20s same-drug across lines r=%.3f vs different drugs r=%.3f (n=%d pairs)",
                 name, np.median(s), np.median(dd), len(s))
        pd.DataFrame(dict(space=name, r_same=s)).to_csv(out / f"harmony_pairs_{name.split()[0]}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "harmony_summary.csv", index=False)

    # ---- consensus profile per drug, drug x drug graph -------------------------------------------
    gd = e.groupby("drug").indices
    drugs = [x for x, ix in gd.items() if len(ix) >= a.min_active]
    Cp = unit(np.stack([Sp[gd[x]].mean(0) for x in drugs]))
    meta = e.groupby("drug").first().loc[drugs]
    G = Cp @ Cp.T
    np.fill_diagonal(G, -np.inf)
    moa = meta.moa_fine.to_numpy()
    kn = np.argsort(-G, axis=1)[:, :a.knn]
    share = float(np.mean([(moa[kn[i]] == moa[i]).mean() for i in range(len(drugs))
                           if (moa == moa[i]).sum() > 1]))
    null = []
    for _ in range(a.n_perm):
        pm = rng.permutation(moa)
        null.append(np.mean([(pm[kn[i]] == pm[i]).mean() for i in range(len(drugs)) if (pm == pm[i]).sum() > 1]))
    log.info("drug-drug graph: %d of %d nearest neighbours share MoA (%.3f) vs shuffle %.3f (p=%.3f)",
             a.knn, len(drugs), share, np.mean(null),
             (np.sum(np.array(null) >= share) + 1) / (a.n_perm + 1))
    pairs = [dict(a=drugs[i], b=drugs[j], cos=float(G[i, j]), moa_a=moa[i], moa_b=moa[j],
                  targets_a=meta.targets.iat[i], targets_b=meta.targets.iat[j],
                  cross_moa=moa[i] != moa[j])
             for i in range(len(drugs)) for j in range(i + 1, len(drugs)) if G[i, j] > a.edge_cos]
    pd.DataFrame(pairs).sort_values("cos", ascending=False).to_csv(out / "drug_graph_edges.csv", index=False)
    np.savez_compressed(out / "drug_profiles.npz", drugs=np.array(drugs), profile=Cp.astype(np.float16),
                        labels=L)
    meta.assign(n_active=[len(gd[x]) for x in drugs],
                n_lines=[e.cell_name[gd[x]].nunique() for x in drugs]).to_csv(out / "drug_meta.csv")

    # ---- hidden targets ---------------------------------------------------------------------------
    rk = (-Sp).argsort(1).argsort(1) + 1
    hz = (Cp - Cp.mean(0)) / np.maximum(Cp.std(0), 1e-9)       # drug-specificity of each prototype
    rows = []
    for i, x in enumerate(drugs):
        ix = gd[x]
        frac = (rk[ix] <= a.top_k).mean(0)                      # share of this drug's lines ranking g top-k
        tg = set(pos[g] for g in split(meta.targets.iat[i]) if g in pos)
        for j in np.argsort(-frac)[:a.n_hidden]:
            if frac[j] < a.min_frac or j in tg:
                continue
            rows.append(dict(drug=x, moa=moa[i], gene=L[j], frac_lines_topk=float(frac[j]),
                             drug_specificity_z=float(hz[i, j]), annotated=";".join(sorted(L[list(tg)])),
                             n_active=len(ix)))
    hid = pd.DataFrame(rows)
    hid = hid[hid.drug_specificity_z >= a.min_z].sort_values(["drug_specificity_z"], ascending=False)
    hid.to_csv(out / "hidden_targets.csv", index=False)
    log.info("hidden-target candidates (top-%d in >= %.0f%% of a drug's lines, specificity z >= %g): %d "
             "over %d drugs", a.top_k, 100 * a.min_frac, a.min_z, len(hid), hid.drug.nunique())
    if len(hid):
        log.info("\n%s", hid.head(25).to_string(index=False))
    json.dump(dict(encode_dir=a.encode_dir, active_threshold=thr, n_drugs=len(drugs),
                   knn_moa_share=share, knn_moa_null=float(np.mean(null)), knn=a.knn),
              open(out / "harmonized_meta.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encode-dir", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pseudobulk", default=None, help="for the Δx arm of the harmony test")
    ap.add_argument("--min-active", type=int, default=5)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--edge-cos", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-hidden", type=int, default=8)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--min-z", type=float, default=3.0)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--max-x", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
