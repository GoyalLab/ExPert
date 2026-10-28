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

"""Panel tables for the sci-Plex half of Fig. 5: ExPert against the other readouts.

Every method must rank the SAME candidates or the ranks mean different things, so the comparison
runs on the matched panel (3,802 genes = the fingerprinting dictionary intersected with our trained
prototypes; fig4_sciplex_annot.py).

    b  share of drugs whose annotated target is recovered by rank k, per method
       ExPert (this run, best of the loss and gain tails) | RNA fingerprinting (our per-line runs on
       the same cells) | training centroids | random
    c  the same at each labelled dose
    d  HDAC inhibitors only: share of conditions with any HDAC machinery gene in the top 10, by dose
    e  FlexPlex HDAC screen: similarity to AR-42 by dose, one line per model (fig4_flexplex.sbatch)

A drug counts as recovered when ANY of its annotated targets is at or above the rank.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_sciplex_panels")
KS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]


def centroid_ranks(a, ann, cand_genes):
    """CMap-style signature correlation, ranked on the SAME candidate panel as every other method.

    The shipped baseline_per_drug.csv ranks over all 19,895 prototypes, which cannot be compared
    with a 3,802-candidate rank, so the baseline is recomputed here.
    """
    import anndata as ad
    z = np.load(a.ref_npz, allow_pickle=True)
    D, lab_ref, gref = z["D"].astype(np.float32), z["labels"].astype(str), z["genes"].astype(str)
    pb = ad.read_h5ad(a.pseudobulk)
    gq = np.asarray(pb.var_names).astype(str)
    gi = {g: i for i, g in enumerate(gq)}
    use = [(i, gi[g]) for i, g in enumerate(gref) if g in gi]
    ri, qi = np.array([u[0] for u in use]), np.array([u[1] for u in use])
    log.info("signature baseline: %d of %d reference genes present in the pseudobulks",
             len(use), len(gref))
    X = np.asarray(pb[:, qi].X.todense(), np.float32)
    X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1e-9) * 1e4)
    o = pb.obs.reset_index(drop=True)
    ctrl = (o.drug == "control").to_numpy()
    cmean = {k: X[np.flatnonzero(ctrl)[v]].mean(0)
             for k, v in o[ctrl].groupby("cell_name", observed=True).indices.items()}
    d = (~ctrl) & o.cell_name.isin(cmean).to_numpy()
    Dx = X[d] - np.stack([cmean[c] for c in o.cell_name[d]])
    e2 = o[d].reset_index(drop=True)
    keep_ref = np.array([i for i, l in enumerate(lab_ref) if l in set(cand_genes)], int)
    R = D[np.ix_(keep_ref, ri)]
    lab_keep = lab_ref[keep_ref]
    zc = lambda M: (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-9)
    S = zc(Dx) @ zc(R).T / Dx.shape[1]          # Pearson of each condition against each signature
    rk = np.empty(S.shape, np.int32)
    np.put_along_axis(rk, np.argsort(-S, 1), np.arange(1, S.shape[1] + 1)[None], axis=1)
    p_ref = {g: i for i, g in enumerate(lab_keep)}
    rr = []
    for i, dr in enumerate(e2.drug):                 # the SAME annotation every method is scored on
        ix = [p_ref[g] for g in ann.get(str(dr), []) if g in p_ref]
        rr.append(int(rk[i, ix].min()) if ix else -1)
    return e2.assign(**{"rank": rr})[["drug", "dose_uM", "cell_name", "rank"]]




def expert_ranks(a, ann, n_cand, which=None, tail=None):
    """Rank the annotated targets of each condition in OUR readout, on the matched panel.

    Recomputed from the latents rather than read from ranks_arms.csv because every method has to be
    scored under the same annotation (`ann`).
    """
    from expert.utils.readout import calibrated_readout
    enc = Path(a.encode_dir)
    noise = np.load(enc / "noise.npy") if (enc / "noise.npy").exists() else None
    Z = np.load(enc / "z_raw.npy").astype(np.float32)
    o = pd.read_csv(enc / "obs.csv")
    b = np.load(enc / "bank.npz", allow_pickle=True)
    P, L, nl = b["P"].astype(np.float32), b["labels"].astype(str), int(b["n_labels"])
    want = set(np.load(a.candidates, allow_pickle=True)["genes"].astype(str)) if a.candidates else None
    sub = np.array([i for i, g in enumerate(L[:nl]) if want is None or g in want], int)
    Pb = P[sub] / np.maximum(np.linalg.norm(P[sub], axis=1, keepdims=True), 1e-12)
    pos = {g: i for i, g in enumerate(L[sub])}
    ctrl = (o.drug == "control").to_numpy()
    cm = {k: Z[np.flatnonzero(ctrl)[v]].mean(0)
          for k, v in o[ctrl].groupby("cell_name", observed=True).indices.items()}
    # Pooling keeps every line; the best rank per drug x dose is taken downstream, as for the
    # per-line fingerprinting runs.
    d = (~ctrl) & o.cell_name.isin(cm).to_numpy()
    if not getattr(a, "pool_lines", False):
        d &= (o.cell_name == a.line).to_numpy()
    D = Z[d] - np.stack([cm[c] for c in o.cell_name[d]])
    e = o[d].reset_index(drop=True)
    tail = tail or a.tail
    if (which or a.readout) == "deflate":
        # shipped settings: 3 rounds, the first 10 slots held by round 1 (that protection is why
        # the top 10 equals the cosine readout; the rounds buy a little between 10 and 50)
        r = calibrated_readout(D, Pb, e.n_cells.to_numpy(), noise=noise, rounds=3, head=10,
                               k=min(500, len(sub)))
        rank_in = lambda i, ix, t: ((np.flatnonzero(np.isin(r[t][i], ix))[:1].tolist()
                                     or [len(sub)])[0] + 1)
        get = lambda i, ix: (min(rank_in(i, ix, "loss"), rank_in(i, ix, "gain"))
                             if tail == "best" else rank_in(i, ix, "loss"))
    else:
        S = (D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)) @ Pb.T
        rk_l = np.empty(S.shape, np.int32); rk_g = np.empty(S.shape, np.int32)
        np.put_along_axis(rk_l, np.argsort(-S, 1), np.arange(1, S.shape[1] + 1)[None], axis=1)
        np.put_along_axis(rk_g, np.argsort(S, 1), np.arange(1, S.shape[1] + 1)[None], axis=1)
        get = lambda i, ix: int(min(rk_l[i, ix].min(), rk_g[i, ix].min()) if tail == "best"
                                else rk_l[i, ix].min())
    rr = []
    for i, dr in enumerate(e.drug):
        ix = [pos[g] for g in ann.get(str(dr), []) if g in pos]
        rr.append(get(i, ix) if ix else -1)
    out = e.assign(**{"rank": rr})[["drug", "dose_uM", "cell_name", "rank"]]
    log.info("ExPert rescored under the shared annotation: %d conditions, %d with a target in panel",
             len(out), int((out["rank"] > 0).sum()))
    return out


def fp_perline_ranks(a, ann, cand):
    """Fingerprinting ranks per cell line x drug x dose, from per-line pip/cor (fig4_sciplex_fp.sbatch).

    Same ordering as the collaborator's pipeline -- PIP descending, ties broken by |cor| descending --
    over the same candidate panel, and the same annotation `ann` ExPert is scored under, so the only
    difference left between the two methods is the method.
    """
    rows = []
    for ln in [x.strip() for x in a.fp_lines.split(",") if x.strip()]:
        d = Path(a.fp_perline_dir) / ln
        pip, cor = pd.read_csv(d / "pip.csv", index_col=0), pd.read_csv(d / "cor.csv", index_col=0)
        sh = sorted(set(pip.columns) & cand) if cand else sorted(pip.columns)
        pv, cv = pip[sh].to_numpy(float), cor[sh].to_numpy(float)
        R = np.empty(pv.shape, np.int64)
        for i in range(len(pv)):
            R[i, np.lexsort((-cv[i], -pv[i]))] = np.arange(1, len(sh) + 1)
        pos = {g: i for i, g in enumerate(sh)}
        for i, g in enumerate(pip.index.astype(str)):
            dr, dz = g.rsplit("@", 1)
            ix = [pos[x] for x in ann.get(dr, []) if x in pos]
            rows.append(dict(drug=dr, dose_uM=float(dz[:-2]) / 1000.0, cell_name=ln,
                             rank=int(R[i, ix].min()) if ix else -1))
        log.info("fingerprinting %s: %d conditions x %d candidates", ln, len(pip), len(sh))
    return pd.DataFrame(rows)


def responsive_conditions(a, q=0.95):
    """Which conditions actually responded, judged before any target is scored.

    The null has to be at the DEPTH OF ONE CONDITION. Comparing a single drug pseudobulk against
    the spread of two AVERAGES of many control pseudobulks passes 92-97% of conditions at every
    dose, because the control mean is far more stable than any one condition. So: take each control
    pseudobulk against the mean of the OTHER controls of its line (leave-one-out), and normalise
    for depth with the 1/sqrt(n) the readout already assumes, giving one threshold in units that a
    condition of any size can be held to.

    This matters because the gain tail scores as well at 10 nM as at 10 uM, which cannot be real --
    a dead condition simply gets two independent chances at a top-10 slot.

    Returns {(normalised drug, dose): bool} for the scored line.
    """
    # --resp-encode-dir: judge responsiveness in ANOTHER model's encode (e.g. the shipped v0.1.0) so a
    # second model is scored on exactly the same conditions and the comparators do not move
    enc = Path(getattr(a, "resp_encode_dir", "") or a.encode_dir)
    Z = np.load(enc / "z_raw.npy").astype(np.float32)
    o = pd.read_csv(enc / "obs.csv")
    ctrl = (o.drug == "control").to_numpy()
    nrm = lambda x: str(x).split("(")[0].strip().lower()
    out, info = {}, []
    for ln, g in o.groupby("cell_name", observed=True):
        ci = g.index[ctrl[g.index]].to_numpy()
        md, mc = o.n_cells.iloc[g.index[~ctrl[g.index]]].median(), o.n_cells.iloc[ci].median()
        if len(ci) and not 0.5 <= mc / max(md, 1) <= 2.0:
            log.warning("%s: control pseudobulks are %.0f cells but conditions %.0f -- the null is "
                        "NOT at the depth it judges (re-encode with --ctrl-chunk)", ln, mc, md)
        if len(ci) < 4:
            continue
        tot = Z[ci].sum(0)
        s = []
        for i in ci:                                  # leave-one-out, at this control's own depth
            mu = (tot - Z[i]) / (len(ci) - 1)
            s.append(float(np.linalg.norm(Z[i] - mu)))
        thr = float(np.quantile(s, q))
        mu_all = Z[ci].mean(0)
        di = g.index[~ctrl[g.index]].to_numpy()
        for i in di:
            v = float(np.linalg.norm(Z[i] - mu_all))
            out[(nrm(o.drug.iat[i]), float(o.dose_uM.iat[i]), ln)] = bool(v > thr)
        info.append((ln, len(ci), thr))
    for ln, n, thr in info:
        log.info("  responsiveness threshold %s: %d controls, 95th pct %.2f", ln, n, thr)
    if out:
        r = pd.DataFrame([dict(dose=d, ok=v) for (_, d, _l), v in out.items()])
        log.info("responded: %s", "  ".join(f"{d:g}uM {v:.2f}"
                                            for d, v in r.groupby("dose").ok.mean().items()))
    return out


def best_rank_per_drug(t, by=("key",)):
    """A drug is recovered at k if any of its conditions puts a target at or above k."""
    return t.groupby(list(by), observed=True)["rank"].min().reset_index()


def curve(r, n_cand, label, ks=KS):
    r = r[r["rank"] > 0]
    return pd.DataFrame([dict(method=label, k=k, recall=float((r["rank"] <= k).mean()),
                              n=len(r)) for k in ks])


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- the target annotation every method is scored against ----------------------------------
    # ChEMBL primary targets with the sci-Plex family genes as the fallback (fig4_sciplex_annot.py),
    # the same for every method.
    ann = {}
    if a.target_csv:
        ch = pd.read_csv(a.target_csv)
        for r in ch.itertuples():
            gs = [g.strip() for g in str(r.targets).split(";") if g.strip() and g.strip() != "nan"]
            if gs:
                ann[str(r.drug)] = gs
    if a.family_csv:
        fam = pd.read_csv(a.family_csv)
        fg = {r.family: [g.strip() for g in str(r.genes).split(";") if g.strip()]
              for r in fam.itertuples()}
        fpa = pd.read_csv(a.drug_family)[["drug", "family"]].drop_duplicates()
        # The family label is the fallback whenever ChEMBL leaves the drug with NOTHING THE PANEL
        # CAN SCORE -- not only when ChEMBL is silent. A ChEMBL target that sits outside the 3,802
        # candidates used to drop the drug entirely, which is how our denominator fell to 82 while
        # fingerprinting kept 109: it scored those drugs on exactly these family genes. The 27 lost
        # that way were systematic (the JAK and Aurora inhibitors), not a random shortfall.
        panel = (set(np.load(a.candidates, allow_pickle=True)["genes"].astype(str))
                 if a.candidates else None)
        in_panel = lambda gs: panel is None or any(g in panel for g in gs)
        n_fill = 0
        for r in fpa.itertuples():
            d = str(r.drug)
            if r.family in fg and not in_panel(ann.get(d, [])):
                ann[d] = sorted(set(ann.get(d, [])) | set(fg[r.family])) if d in ann else fg[r.family]
                n_fill += 1
        log.info("family fallback applied to %d drugs with no ChEMBL target inside the panel",
                 n_fill)
    log.info("annotation: %d drugs (%s)", len(ann), Path(a.target_csv).name if a.target_csv else "ours")

    # ---- ExPert: this run, matched panel, best of the two tails --------------------------------
    t = pd.read_csv(a.ranks)
    # NB: t.tail is a DataFrame METHOD -- take the column by name or the mask is all-False and
    # the panel silently empties
    ex = t[(t.bank == "matched") & (t.readout == a.readout)
           & (t["tail"] == a.tail)].copy()
    if not a.pool_lines:
        ex = ex[ex.cell_name == a.line]
    n_cand = int(a.n_candidates)
    if ann and a.encode_dir:
        ex = expert_ranks(a, ann, n_cand)
    log.info("ExPert: %d conditions, %d drugs on %s (bank=matched, %s, %s)", len(ex),
             ex.drug.nunique(), "all lines" if a.pool_lines else a.line, a.readout, a.tail)

    # ---- fingerprinting: our per-line runs (fig4_sciplex_fp.sbatch) ------------------------------
    # Run per line on the same cells as ExPert, so "recovered in ANY responsive condition" gives both
    # methods the same number of chances per drug x dose. Only drugs with a family label (the
    # evaluable set of fig4_sciplex_annot.py) carry one here.
    fam = pd.read_csv(a.drug_family).drop_duplicates("drug").set_index("drug").family
    cand_set = (set(np.load(a.candidates, allow_pickle=True)["genes"].astype(str))
                if a.candidates else set())
    fp = fp_perline_ranks(a, ann, cand_set)
    fp["family"] = fp.drug.map(fam)
    if not a.pool_lines:
        fp = fp[fp.cell_name == a.line]
    log.info("fingerprinting: %d conditions, %d drugs (per line: %s)", len(fp), fp.drug.nunique(),
             a.fp_lines)

    # drug names differ in punctuation between the two tables; match on the leading token
    norm = lambda s: str(s).split("(")[0].strip().lower()
    ex["key"], fp["key"] = ex.drug.map(norm), fp.drug.map(norm)
    shared = sorted(set(ex.key) & set(fp.key))
    log.info("%d drugs shared between ExPert and fingerprinting "
             "(ours %d, theirs %d)", len(shared), ex.key.nunique(), fp.key.nunique())
    ex, fp = ex[ex.key.isin(shared)], fp[fp.key.isin(shared)]
    cand_genes = set(np.load(a.candidates, allow_pickle=True)["genes"].astype(str)) if a.candidates \
        else set()

    # ---- training centroids: the CMap-style signature correlation, on the SAME panel -------------
    # The shipped baseline_per_drug.csv ranks over all 19,895 prototypes, which is not comparable to
    # a 3,802-candidate rank, so the centroid arm is recomputed here against the matched panel.
    cen = None
    if a.ref_npz and a.pseudobulk and Path(a.ref_npz).exists() and Path(a.pseudobulk).exists():
        cen = centroid_ranks(a, ann, cand_genes)
        if not a.pool_lines:
            cen = cen[cen.cell_name == a.line]
        cen = cen.assign(key=lambda q: q.drug.map(norm))
        cen = cen[cen.key.isin(shared)]
        log.info("centroids: %d conditions", len(cen))
    # ---- random: a uniform draw over the same candidate panel, per condition ---------------------
    rand = ex.copy()
    rand["rank"] = rng.integers(1, n_cand + 1, len(rand))
    rand["key"] = ex.key.to_numpy()

    # ---- responsiveness: one mask, applied to every method ---------------------------------------
    # The filter belongs to the CONDITION (did these cells move at all), not to the method, so the
    # same mask gates ExPert, fingerprinting, the centroids and the random draw alike.
    if a.active_filter and a.encode_dir:
        resp = responsive_conditions(a)
        if resp:
            anyline = {}
            for (k, dz, _l), v in resp.items():
                # scoring one line: a line-agnostic row may only count that line's responsiveness,
                # or fingerprinting keeps drug x doses that never responded there (4C denominators)
                if a.pool_lines or _l == a.line:
                    anyline[(k, dz)] = anyline.get((k, dz), False) or v

            def keep(q):
                if "cell_name" in q.columns:
                    m = [bool(resp.get((k, float(dz), ln), False))
                         for k, dz, ln in zip(q.key, q.dose_uM, q.cell_name)]
                else:                      # the fingerprint table carries no line
                    m = [bool(anyline.get((k, float(dz)), False))
                         for k, dz in zip(q.key, q.dose_uM)]
                return q[m]

            before = (len(ex), len(fp))
            ex, fp, rand = keep(ex), keep(fp), keep(rand)
            if cen is not None:
                cen = keep(cen)
            log.info("responsive conditions only: ExPert %d -> %d, fingerprinting %d -> %d",
                     before[0], len(ex), before[1], len(fp))

    # ---- is the annotation the thing limiting recovery? ------------------------------------------
    # A drug we cannot score (no annotated target inside the candidate panel) is not a miss, it is a
    # hole in the annotation. Score our readout under each source so the two are never confused.
    if a.encode_dir:
        ours = {}
        oo = pd.read_csv(Path(a.encode_dir) / "obs.csv")
        for r in oo[["drug", "trained_targets"]].drop_duplicates().itertuples():
            gs = [g.strip() for g in str(r.trained_targets).replace(",", ";").split(";")
                  if g.strip() and g.strip() != "nan"]
            if gs:
                ours[str(r.drug)] = gs
        chembl = {}
        if a.target_csv:
            for r in pd.read_csv(a.target_csv).itertuples():
                gs = [g.strip() for g in str(r.targets).split(";") if g.strip() and g.strip() != "nan"]
                if gs:
                    chembl[str(r.drug)] = gs
        union = {k: sorted(set(ours.get(k, [])) | set(chembl.get(k, [])) | set(ann.get(k, [])))
                 for k in set(ours) | set(chembl) | set(ann)}
        chk = []
        for nm, A in (("ours (sci-Plex prep)", ours), ("ChEMBL", chembl),
                      ("ChEMBL + family (shared)", ann), ("union of all", union)):
            if not A:
                continue
            q = expert_ranks(a, A, n_cand)
            q = q[q["rank"] > 0]
            per = q.groupby("drug", observed=True)["rank"].min()
            chk.append(dict(annotation=nm, n_drugs_annotated=len(A),
                            n_drugs_scorable=int(per.size),
                            median_targets=float(np.median([len(v) for v in A.values()])),
                            **{f"recall@{k}": float((per <= k).mean()) for k in (10, 50)}))
            log.info("%-26s %3d drugs annotated, %3d scorable in panel | recall@10 %.3f @50 %.3f",
                     nm, len(A), per.size, chk[-1]["recall@10"], chk[-1]["recall@50"])
        pd.DataFrame(chk).to_csv(out / "annotation_check.csv", index=False)

    # ---- b: recall over k, per drug --------------------------------------------------------------
    # a drug is in the comparison only if EVERY method could score it: an unscorable drug is a hole
    # in the annotation / candidate panel, not a miss, and letting the denominators differ (82 vs
    # 109 vs 80) silently compares different questions
    scorable = lambda q: set(q[q["rank"] > 0].key)
    common = scorable(ex) & scorable(fp) & (scorable(cen) if cen is not None else scorable(ex))
    log.info("scoring every method on the %d drugs ALL of them can score (ExPert %d, fp %d,"
             " centroids %s)", len(common), len(scorable(ex)), len(scorable(fp)),
             len(scorable(cen)) if cen is not None else "-")
    ex, fp, rand = [x[x.key.isin(common)] for x in (ex, fp, rand)]
    if cen is not None:
        cen = cen[cen.key.isin(common)]
    parts = [curve(best_rank_per_drug(ex), n_cand, "ExPert"),
             curve(best_rank_per_drug(fp), n_cand, "RNA fingerprinting"),
             curve(best_rank_per_drug(rand), n_cand, "random")]
    if cen is not None:
        parts.append(curve(best_rank_per_drug(cen), n_cand, "training centroids"))
    b = pd.concat(parts)
    b.to_csv(out / "fig4b_sciplex_curve.csv", index=False)
    # the per-drug and per-dose ranks behind the curves, so the ExPert-vs-comparator tests are run
    # on exactly the drugs the panels plot rather than on a re-derived set
    srcs = [("ExPert", ex), ("RNA fingerprinting", fp), ("random", rand)]
    if cen is not None:
        srcs.append(("training centroids", cen))
    pd.concat([best_rank_per_drug(q).assign(method=m) for m, q in srcs]).to_csv(
        out / "fig4b_per_drug.csv", index=False)
    pd.concat([best_rank_per_drug(g, by=("key",)).assign(method=m, dose_uM=dz)
               for m, q in srcs for dz, g in q.groupby("dose_uM")]).to_csv(
        out / "fig4c_per_dose.csv", index=False)
    log.info("panel b recall@10:\n%s", b[b.k == 10][["method", "recall", "n"]].to_string(index=False))

    # ---- c: the same by dose ---------------------------------------------------------------------
    rows = []
    for lab, src in (("ExPert", ex), ("RNA fingerprinting", fp), ("random", rand),
                     *( [("training centroids", cen)] if cen is not None else [] )):
        for dose, g in src.groupby("dose_uM"):
            q = curve(best_rank_per_drug(g, by=("key",)), n_cand, lab, ks=[10, 50])
            q["dose_uM"] = dose
            rows.append(q)
    c = pd.concat(rows)
    c.to_csv(out / "fig4c_sciplex_dose.csv", index=False)

    # ---- d: HDAC machinery in the top 10, by dose ------------------------------------------------
    hd = pd.read_csv(a.hdac_complexes)
    mach = sorted(set(hd.gene.astype(str)))
    log.info("HDAC machinery: %d genes over %d complexes", len(mach), hd.complex.nunique())
    hd_drugs = sorted(set(fp[fp.family == "HDAC"].key))
    log.info("%d HDAC inhibitors in the shared set", len(hd_drugs))
    pd.DataFrame(dict(gene=mach)).to_csv(out / "fig4d_hdac_machinery.csv", index=False)

    # Every method is scored against the SAME gene set and at the same doses: the HDAC machinery plus
    # the HDAC family's CORUM-core partners (fig4_sciplex_annot.py).
    if a.encode_dir:
        mach_set = sorted(set(mach))
        if a.partner_sets and Path(a.partner_sets).exists():
            ps = pd.read_csv(a.partner_sets)
            row = ps[ps.family == "HDAC"]
            if len(row):
                part = [x.split(":")[0] for x in str(row.partners_in_candidates.iloc[0]).split(";")
                        if x.strip()]
                mach_set = sorted(set(mach_set) | set(part))
                log.info("HDAC gene set: %d machinery + %d CORUM-core partners -> %d genes",
                         len(mach), len(part), len(mach_set))
        mach_ann = {d: mach_set for d in set(ex.drug) | set(cen.drug if cen is not None else [])}
        # One row per drug x dose for EVERY method, then the same drug x dose set for all of them.
        per = {}
        for lab, q in (("ExPert", expert_ranks(a, mach_ann, n_cand)),
                       *( [("training centroids", centroid_ranks(a, mach_ann, cand_genes))]
                          if a.ref_npz and a.pseudobulk else [] )):
            q = q.copy(); q["key"] = q.drug.map(norm)
            if "cell_name" in q and not a.pool_lines:
                q = q[q.cell_name == a.line]
            q = q[q.key.isin(hd_drugs) & (q["rank"] > 0)]
            # Keep the per-line conditions. Collapsing them to the best of three lines against a
            # ~50-gene machinery set saturates ExPert at 1.00 at every dose and the panel stops
            # saying anything; what has to match across methods is the drug x dose SET, below.
            per[lab] = q[["key", "dose_uM", "rank"]]
        q = fp_perline_ranks(a, {d: mach_set for d in set(fp.drug)}, set(np.load(
            a.candidates, allow_pickle=True)["genes"].astype(str)) if a.candidates else set())
        q["key"] = q.drug.map(norm)
        if not a.pool_lines:
            q = q[q.cell_name == a.line]
        per["RNA fingerprinting"] = q[q.key.isin(hd_drugs) & (q["rank"] > 0)][["key", "dose_uM", "rank"]]
        shared_d = set.intersection(*[set(map(tuple, q[["key", "dose_uM"]].to_numpy()))
                                      for q in per.values()]) if per else set()
        log.info("panel d: %d drug x dose pairs every method can score (rows per method: %s; "
                 "one row per condition of each method's own input)", len(shared_d),
                 ", ".join(f"{k} {len(v)}" for k, v in per.items()))
        rows_d = []
        for lab, q in per.items():
            q = q[[tuple(r) in shared_d for r in q[["key", "dose_uM"]].to_numpy()]]
            for dose, g in q.groupby("dose_uM"):
                rows_d.append(dict(method=lab, dose_uM=dose, n=len(g),
                                   frac_top10=float((g["rank"] <= 10).mean()),
                                   frac_top50=float((g["rank"] <= 50).mean())))
            log.info("%s HDAC set in top 10: %s", lab,
                     {r["dose_uM"]: round(r["frac_top10"], 3) for r in rows_d
                      if r["method"] == lab})
        pd.DataFrame(rows_d).to_csv(out / "fig4d_hdac_by_dose.csv", index=False)
        pd.concat([q[[tuple(r) in shared_d for r in q[["key", "dose_uM"]].to_numpy()]]
                   .assign(method=m) for m, q in per.items()]).to_csv(
            out / "fig4d_per_pair.csv", index=False)
        pd.DataFrame(sorted(hd_drugs), columns=["drug"]).to_csv(out / "fig4d_hdac_drugs.csv",
                                                                index=False)

    # ---- e: AR-42 convergence (fig4_flexplex.sbatch) ---------------------------------------------
    if a.flexplex and Path(a.flexplex).exists():
        fx = pd.read_csv(a.flexplex)
        fx = fx[fx.sim_AR42.notna()][["method", "drug", "dose_nM", "sim_AR42"]]
        fx.to_csv(out / "fig4e_ar42.csv", index=False)
        log.info("panel e: %d rows, methods %s, drugs %s", len(fx), list(fx.method.unique()),
                 list(fx.drug.unique()))

    json.dump(dict(line=a.line, readout=a.readout, tail=a.tail, n_candidates=n_cand,
                   n_drugs_shared=len(shared), hdac_drugs=len(hd_drugs)),
              open(out / "panels_meta.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ranks", required=True, help="ranks_arms.csv from fig4_sciplex_pb.py")
    ap.add_argument("--drug-family", required=True,
                    help="drug -> sci-Plex family label of the evaluable drugs (fig4_sciplex_annot.py)")
    ap.add_argument("--fp-perline-dir", required=True,
                    help="per-line fingerprinting runs (<dir>/<LINE>/{pip,cor}.csv)")
    ap.add_argument("--fp-lines", default="K562,A549,MCF7")
    ap.add_argument("--resp-encode-dir", default="",
                    help="take the responsiveness mask from this encode dir instead of --encode-dir")
    ap.add_argument("--hdac-complexes", required=True)
    ap.add_argument("--flexplex", default="")
    ap.add_argument("--centroid", default="")
    ap.add_argument("--ref-npz", default="", help="crispr_signature_ref.npz (CMap-style baseline)")
    ap.add_argument("--pseudobulk", default="", help="pseudobulk.h5ad written by the encode step")
    ap.add_argument("--candidates", default="", help="npz with the matched panel's 'genes'")
    ap.add_argument("--encode-dir", default="", help="rescore our readout under the shared annotation")
    ap.add_argument("--target-csv", default="", help="the shared drug -> target annotation (ChEMBL)")
    ap.add_argument("--family-csv", default="", help="family label genes, used where ChEMBL has none")
    ap.add_argument("--partner-sets", default="", help="CORUM-core partner sets per family (fig4_sciplex_annot.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool-lines", action=argparse.BooleanOptionalAction, default=True,
                    help="score every cell line and take the best rank per drug x dose")
    ap.add_argument("--active-filter", action=argparse.BooleanOptionalAction, default=True,
                    help="score only conditions that moved past control-vs-control noise")
    ap.add_argument("--line", default="K562")
    ap.add_argument("--readout", default="deflate")
    ap.add_argument("--tail", default="best")
    ap.add_argument("--n-candidates", type=int, default=3802)
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
