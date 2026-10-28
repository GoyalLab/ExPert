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

"""Fig. 4G/H data: Tahoe-100M mechanism examples (abemaciclib -> GSK3, clobetasol = glucocorticoid).

Port of the collaborator handoff (tahoe_mechanism_examples, scripts tahoe_pb_expr / 80_ / 82_ POS=1 / 84_ / 87_)
onto the repo's own inputs: the streamed Tahoe pseudobulk, the v0.1.0 mtmask_line encode of it
(fig4.sbatch STEP=pb_sweep_encode) and the public Tahoe sample / drug tables. Stages, in order:

    expr       plain-expression responses (no model). Per pseudobulk row (>= --min-cells): log1p(CP10k) over ALL
               genes, minus the same plate x line DMSO_TF row; unit = drug | dose | plate, mean over lines. MT-* and
               metallothionein genes removed; keeps the --n-hvg genes with the highest response variance.
               -> expr_units.npz (R_hvg, genes_hvg, units, n)
    positions  well position of every sample: Tahoe sample IDs run in well order within a plate, so position =
               order within plate. Plates that share their drug list form a block (4 blocks of ~90 drugs, one
               plate per dose; plate 14 is a fourth plate of one block); the mixed plate is block X.
               -> sample_positions.csv
    activity   model-free dose reproducibility of each drug x dose (dominant plate): max cosine of its plate-centred
               expression response to the drug's other doses vs 2,000 draws of other drugs' units; BH -> q_act.
               -> activity_units_expr_hvg.csv
    unitz      ExPert unit scores of ACTIVE drug x doses (q_act < 0.05). Response = latent minus the same plate x
               line DMSO, then minus the mean of all drugs on that plate x line; unit = mean over lines of the cosine
               to each trained prototype; z over the active units on the plate; well-position correction = minus the
               mean z of the units at the same position in the other blocks.
               -> mech_unit_z.npy, mech_units.csv
    doseranks  Fig. 4G: the same scores for ALL units (active or not) of --drugs, z over all drug units on the
               plate, position-corrected; rank of each of --genes among the trained prototypes (1 = best).
               -> dose_ranks.csv
    clobetasol Fig. 4H: per drug with >= 2 active doses, S = min over its active units of the unit z (unitz);
               y = mean Pearson r of S with that of the reference glucocorticoids (the drug itself excluded);
               x = mean z (across drugs) of the 5 uM expression log fold change of the GR target genes.
               -> clobetasol_scatter.csv

`--stage all` runs them in order; every stage reads the previous stages' outputs from --out-dir.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_tahoe_mech")
MT = re.compile(r"^(MT-|MT1|MT2A$|MT3$|MT4$)")
CTRL = "DMSO_TF"
STAGES = ("expr", "positions", "activity", "unitz", "doseranks", "clobetasol")


def bh(p):
    o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1)
    return out


def load_bank(path):
    b = np.load(path, allow_pickle=True)
    nl = int(b["n_labels"])
    L = b["labels"].astype(str)[:nl]
    P = b["P"].astype(np.float32)[:nl]
    return L, P / np.linalg.norm(P, axis=1, keepdims=True)


def stage_expr(a, out):
    import anndata as ad
    import scipy.sparse as sp
    A = ad.read_h5ad(a.pseudobulk, backed="r")
    o = A.obs.copy().reset_index(drop=True)
    rows = np.flatnonzero((o.n_cells >= a.min_cells).to_numpy())
    o = o.iloc[rows].reset_index(drop=True)
    var = np.array(list(map(str, A.var_names)))
    gkeep = np.array([not MT.match(g) for g in var])
    log.info("rows kept %d; genes %d (%d MT / metallothionein removed)", len(rows), len(var), (~gkeep).sum())

    def logrows(r):
        X = A.X[rows[r]]
        X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float64)
        X = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
        return X[:, gkeep].astype(np.float32)

    gv = var[gkeep]
    ctl = (o.drug.astype(str) == CTRL).to_numpy()
    cidx = np.flatnonzero(ctl)
    D = np.concatenate([logrows(cidx[s:s + a.chunk]) for s in range(0, len(cidx), a.chunk)])
    dkey = {(p, c): i for i, (p, c) in enumerate(zip(o.plate.iloc[cidx], o.cell_name.iloc[cidx]))}
    drows = np.flatnonzero(~ctl & np.array([(p, c) in dkey for p, c in zip(o.plate, o.cell_name)]))
    ukey = (o.drug.astype(str) + "|" + o.dose_uM.map("{:g}".format) + "|" + o.plate.astype(str)).to_numpy()[drows]
    uk, inv = np.unique(ukey, return_inverse=True)
    S = np.zeros((len(uk), gkeep.sum()), np.float64); n = np.bincount(inv, minlength=len(uk))
    s1 = np.zeros(gkeep.sum()); s2 = np.zeros(gkeep.sum())
    for s in range(0, len(drows), a.chunk):
        r = drows[s:s + a.chunk]
        X = logrows(r) - D[[dkey[(p, c)] for p, c in zip(o.plate.iloc[r], o.cell_name.iloc[r])]]
        np.add.at(S, inv[s:s + a.chunk], X)
        s1 += X.sum(0); s2 += (X.astype(np.float64) ** 2).sum(0)
    S /= n[:, None]
    hv = np.argsort(-(s2 / len(drows) - (s1 / len(drows)) ** 2))[:a.n_hvg]
    np.savez_compressed(out / "expr_units.npz", R_hvg=S[:, hv].astype(np.float32), genes_hvg=gv[hv], units=uk, n=n)
    log.info("expr: %d DMSO rows, %d drug rows, %d units, %d hvg", len(cidx), len(drows), len(uk), len(hv))


def stage_positions(a, out):
    s = pd.read_parquet(a.sample_metadata)
    s["snum"] = s["sample"].str.extract(r"(\d+)$", expand=False).astype(int)
    s["pos"] = s.groupby("plate").snum.rank(method="first").astype(int) - 1
    s["pnum"] = s.plate.str.extract(r"(\d+)$", expand=False).astype(int)
    dc = s.drugname_drugconc.map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    s["drug"] = [d[0][0] for d in dc]
    s["dose"] = [d[0][1] for d in dc]
    # blocks: plates whose (non-DMSO) drug lists mostly coincide; a plate matching no other plate is mixed (X)
    drugs = {p: set(q.drug.str.strip()) - {CTRL} for p, q in s.groupby("plate")}
    plates = sorted(drugs, key=lambda p: int(re.sub(r"\D", "", p)))
    block, nb = {}, 0
    for p in plates:
        if p in block:
            continue
        mates = [q for q in plates if q not in block and q != p
                 and len(drugs[p] & drugs[q]) > 0.5 * min(len(drugs[p]), len(drugs[q]))]
        if not mates:
            block[p] = "X"; continue
        lab = "ABCDEFGH"[nb]; nb += 1
        for q in [p] + mates:
            block[q] = lab
    s["block"] = s.plate.map(block)
    s = s.sort_values(["pnum", "pos"])[["sample", "plate", "pnum", "pos", "drug", "dose", "block"]]
    s.to_csv(out / "sample_positions.csv", index=False)
    log.info("positions: %d samples; blocks %s", len(s), s.groupby("block").plate.unique().to_dict())


def stage_activity(a, out):
    rng = np.random.default_rng(a.seed)
    ez = np.load(out / "expr_units.npz", allow_pickle=True)
    M = ez["R_hvg"].astype(np.float64)
    U = pd.DataFrame([k.split("|") for k in ez["units"].astype(str)], columns=["drug", "dose", "plate"])
    U["n_lines"] = ez["n"]; U["dose"] = U.dose.astype(float)
    for _, ix in U.groupby("plate").indices.items():
        M[ix] -= M[ix].mean(0)
    # one unit per drug x dose: most lines; ties (a 5 uM dose on its block plate AND on plate 13/14 with the same
    # lines) go to the lower plate number, i.e. the drug's own block plate. The handoff broke ties by an unstable
    # sort, which is platform-dependent.
    U["pnum"] = U.plate.str.extract(r"(\d+)$", expand=False).astype(int)
    keep = U.sort_values(["n_lines", "pnum"], ascending=[False, True], kind="stable") \
        .drop_duplicates(["drug", "dose"]).index.sort_values()
    U = U.loc[keep].reset_index(drop=True); M = M[keep]
    MU = M / np.linalg.norm(M, axis=1, keepdims=True); CS = MU @ MU.T
    pl, dg = U.plate.to_numpy(), U.drug.to_numpy()
    rep, prep = np.full(len(U), np.nan), np.full(len(U), np.nan)
    for i in range(len(U)):
        oth = np.flatnonzero((dg == dg[i]) & (pl != pl[i]))
        if not len(oth):
            continue
        rep[i] = CS[i, oth].max()
        pool = np.flatnonzero((dg != dg[i]) & (pl != pl[i]))
        nd = CS[i, rng.choice(pool, size=(a.n_null, len(oth)))].max(1)
        prep[i] = (1 + (nd >= rep[i]).sum()) / (a.n_null + 1)
    U["repro"] = rep; U["p_rep"] = prep
    ok = np.isfinite(prep)
    U["q_act"] = 1.0; U.loc[ok, "q_act"] = bh(prep[ok])
    U.to_csv(out / "activity_units_expr_hvg.csv", index=False)
    log.info("activity: %d units; q<%g: %d (%d drugs)", len(U), a.q_act, (U.q_act < a.q_act).sum(),
             U[U.q_act < a.q_act].drug.nunique())


def responses(a):
    """latent minus the same plate x line DMSO, then minus the mean of all drugs on that plate x line"""
    Z = np.load(Path(a.encode_dir) / "z_raw.npy").astype(np.float64)
    o = pd.read_parquet(Path(a.encode_dir) / "obs.parquet").reset_index(drop=True)
    ctl = (o.drug.astype(str) == CTRL).to_numpy()
    cidx = {(p, c): i for i, (p, c) in enumerate(zip(o.plate, o.cell_name)) if ctl[i]}
    dr = o[~ctl & np.array([(p, c) in cidx for p, c in zip(o.plate, o.cell_name)])].copy()
    dr["drug"] = dr.drug.astype(str)
    D = Z[dr.index.to_numpy()] - Z[[cidx[(p, c)] for p, c in zip(dr.plate, dr.cell_name)]]
    del Z
    for _, ix in dr.groupby(["plate", "cell_name"], observed=True).indices.items():
        D[ix] = D[ix] - D[ix].mean(0)
    return dr, D


def unit_scores(D, inv, n_units, P):
    """per unit: mean over its rows (cell lines) of the cosine to every prototype"""
    FQ = np.zeros((n_units, len(P)), np.float64)
    for s in range(0, len(D), 4096):
        np.add.at(FQ, inv[s:s + 4096], D[s:s + 4096].astype(np.float32) @ P.T)
    return FQ / np.bincount(inv, minlength=n_units)[:, None]


def plate_z(FQ, plate):
    Zu = np.zeros_like(FQ)
    for _, ix in pd.Series(plate).groupby(plate).indices.items():
        Zu[ix] = (FQ[ix] - FQ[ix].mean(0)) / np.maximum(FQ[ix].std(0), 1e-6)
    return Zu


def attach_positions(U, out, drug_col="drug", dose_col="dose"):
    sp_ = pd.read_csv(out / "sample_positions.csv")
    sp_["dk"] = sp_.drug.astype(str).str.strip() + "@" + sp_.dose.map("{:g}".format)
    pos = {(r.plate, r.dk): (r.pos, r.block) for r in sp_.itertuples()}
    pb = [pos.get((p, d.strip() + "@" + x), (np.nan, None)) for p, d, x in zip(U.plate, U[drug_col], U[dose_col])]
    U["pos"] = [x[0] for x in pb]; U["block"] = [x[1] for x in pb]


def position_correct(Zu, U, rows):
    """minus the mean z of the units at the same well position in the OTHER blocks (block X never used)"""
    Zc = Zu.copy()
    posv, blk = U.pos.to_numpy(), U.block.to_numpy()
    valid = (blk != "X") & U.block.notna().to_numpy()
    for i in rows:
        if pd.isna(posv[i]):
            continue
        same = np.flatnonzero((posv == posv[i]) & (blk != blk[i]) & valid)
        if len(same):
            Zc[i] = Zu[i] - Zu[same].mean(0)
    return Zc


def stage_unitz(a, out):
    L, P = load_bank(a.bank)
    dr, D = responses(a)
    dr["dd"] = dr.drug + "@" + dr.dose_uM.map("{:g}".format)
    A = pd.read_csv(out / "activity_units_expr_hvg.csv")
    A["dd"] = A.drug.astype(str) + "@" + A.dose.map("{:g}".format)
    act = set(A[A.q_act < a.q_act].dd)
    pmap = A.set_index("dd").plate.astype(str)               # the activity stage's unit plate for each drug x dose
    sel = (dr.dd.isin(act) & (dr.plate.astype(str) == dr.dd.map(pmap).astype(str))).to_numpy()
    dr, D = dr[sel], D[sel]
    nrm = np.linalg.norm(D, axis=1); ok = nrm > 0
    dr, D = dr[ok], D[ok] / nrm[ok, None]
    units = dr.dd.unique(); ui = {u: i for i, u in enumerate(units)}
    inv = dr.dd.map(ui).to_numpy()
    FQ = unit_scores(D, inv, len(units), P)
    U = pd.DataFrame({"unit": units, "drug": [u.rsplit("@", 1)[0] for u in units],
                      "dose": [u.rsplit("@", 1)[1] for u in units],
                      "plate": [str(pmap[u]) for u in units], "n_lines": np.bincount(inv)})
    Zu = plate_z(FQ, U.plate.to_numpy())
    attach_positions(U, out)
    Zu = position_correct(Zu, U, range(len(U)))
    np.save(out / "mech_unit_z.npy", Zu)
    U.to_csv(out / "mech_units.csv", index=False)
    log.info("unitz: %d active units, %d drugs, %d with a position", len(U), U.drug.nunique(), U.pos.notna().sum())


def stage_doseranks(a, out):
    L, P = load_bank(a.bank); li = {g: i for i, g in enumerate(L)}
    drugs = [x for x in a.drugs.split(",") if x]; genes = [x for x in a.genes.split(",") if x]
    dr, D = responses(a)
    n = np.linalg.norm(D, axis=1); k = n > 0
    dr, D = dr[k], D[k] / n[k, None]
    dr["unit"] = dr.drug + "@" + dr.dose_uM.map("{:g}".format) + "|" + dr.plate.astype(str)
    uk, inv = np.unique(dr.unit.to_numpy(), return_inverse=True)
    FQ = unit_scores(D, inv, len(uk), P)
    U = pd.DataFrame({"unit": uk})
    U["drug"] = U.unit.str.split("@").str[0]
    U["dose"] = U.unit.str.split("@").str[1].str.split("|").str[0]; U["plate"] = U.unit.str.split("|").str[1]
    Zu = plate_z(FQ, U.plate.to_numpy())
    attach_positions(U, out)
    rows = np.flatnonzero(U.drug.isin(drugs).to_numpy())
    Zc = position_correct(Zu, U, rows)
    A = pd.read_csv(out / "activity_units_expr_hvg.csv")
    qa = dict(zip(A.drug.astype(str) + "@" + A.dose.map("{:g}".format) + "|" + A.plate.astype(str), A.q_act))
    R = []
    for i in rows:
        s = Zc[i]
        for g in genes:
            R.append(dict(drug=U.drug.iat[i], dose=float(U.dose.iat[i]), plate=U.plate.iat[i], block=U.block.iat[i],
                          gene=g, rank=1 + int((s > s[li[g]]).sum()), z=float(s[li[g]]),
                          q_act=qa.get(U.unit.iat[i], np.nan), top3=",".join(L[np.argsort(-s)[:3]])))
    R = pd.DataFrame(R); R.to_csv(out / "dose_ranks.csv", index=False)
    log.info("doseranks:\n%s", R.pivot_table(index=["drug", "dose", "plate"], columns="gene", values="rank")
             .astype(int).to_string())


def stage_clobetasol(a, out):
    refs = a.gc_refs.split(","); gr = a.gr_genes.split(",")
    Zu = np.load(out / "mech_unit_z.npy").astype(np.float64); U = pd.read_csv(out / "mech_units.csv")
    drugs = [d for d, ix in U.groupby("drug").indices.items() if len(ix) >= 2]
    S = np.stack([Zu[np.flatnonzero(U.drug.to_numpy() == d)].min(0) for d in drugs])
    S -= S.mean(1, keepdims=True); S /= np.linalg.norm(S, axis=1, keepdims=True)
    R = S @ S.T; di = {d: i for i, d in enumerate(drugs)}
    ref = [di[d] for d in refs if d in di]
    ysim = {d: np.mean([R[di[d], j] for j in ref if j != di[d]]) for d in drugs}
    ez = np.load(out / "expr_units.npz", allow_pickle=True)
    gh = list(ez["genes_hvg"].astype(str))
    EU = pd.DataFrame([k.split("|") for k in ez["units"].astype(str)], columns=["drug", "dose", "plate"])
    EU["dose"] = EU.dose.astype(float)
    X = pd.concat([EU, pd.DataFrame(ez["R_hvg"][:, [gh.index(g) for g in gr]].astype(float), columns=gr)], axis=1)
    hi = X[X.dose == a.gr_dose].groupby("drug")[gr].mean()
    xgr = ((hi - hi.mean()) / hi.std()).mean(axis=1)
    md = pd.read_parquet(a.drug_metadata); moa = dict(zip(md.drug, md["moa-fine"]))
    T = pd.DataFrame({"drug": drugs, "expert_sim_to_GC": [ysim[d] for d in drugs],
                      "GR_expression_score": [xgr.get(d, np.nan) for d in drugs]})
    T["moa"] = T.drug.map(moa).fillna("unclear")
    T.to_csv(out / "clobetasol_scatter.csv", index=False)
    log.info("clobetasol: %d drugs; top by similarity:\n%s", len(T),
             T.sort_values("expert_sim_to_GC", ascending=False).head(8).round(3).to_string(index=False))


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    todo = STAGES if a.stage == "all" else a.stage.split(",")
    for st in todo:
        t0 = time.time()
        globals()[f"stage_{st}"](a, out)
        log.info("[%s] %.1f min", st, (time.time() - t0) / 60)
    (out / f"meta_{a.stage.replace(',', '_')}.json").write_text(json.dumps(vars(a), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, help="all, or a comma list of " + ", ".join(STAGES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pseudobulk", help="expr: plate x line x drug x dose pseudobulk h5ad")
    ap.add_argument("--sample-metadata", help="positions: Tahoe-100M metadata/sample_metadata.parquet")
    ap.add_argument("--drug-metadata", help="clobetasol: Tahoe-100M metadata/drug_metadata.parquet (moa-fine)")
    ap.add_argument("--encode-dir", help="unitz/doseranks: z_raw.npy + obs.parquet (mtmask_line encode)")
    ap.add_argument("--bank", help="unitz/doseranks: prototype bank.npz (trained rows are used)")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--q-act", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drugs", default="Idarubicin (hydrochloride),Doxorubicin (hydrochloride),"
                    "Epirubicin (hydrochloride),Mitoxantrone (dihydrochloride),Abemaciclib,palbociclib")
    ap.add_argument("--genes", default="RPA2,RAD9A,SSRP1,SUPT16H,TOP2A,TOP2B,HIST1H2AB,GSK3A,GSK3B,CDK4,CDK6")
    ap.add_argument("--gc-refs", default="Dexamethasone,Triamcinolone,Betamethasone dipropionate,Budesonide")
    ap.add_argument("--gr-genes", default="FKBP5,TSC22D3,PER1,DUSP1,KLF9,CEBPD,DDIT4,ERRFI1")
    ap.add_argument("--gr-dose", type=float, default=5.0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(ap.parse_args())
