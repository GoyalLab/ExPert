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

"""The auxin-degron panel through the same path the drug screens settled on.

Four things carried over from sci-Plex and Tahoe:

  pseudobulks      one row per (clone, condition), mean raw counts, with n_cells so the deflate
                   readout judges each row at its own depth
  active filter    applied BEFORE the model, in expression space: a clone counts only if its
                   IAA-vs-DMSO shift is larger than the split-half noise of its OWN DMSO cells.
                   A clone whose degron did nothing is not a miss, and scoring it as one buries
                   the clones that did respond.
  readout          deflate at its shipped settings, BEST of the loss and gain tails. The gain tail
                   is not optional here either: degrading a repressor de-represses its programme.
  bank             trained prototypes (the zero-shot rows roughly halve recall, Tahoe and sci-Plex)

`--clone-col` selects the barcode assignment (clone_nm1 = 1 mismatch, clone_nm2 = 2); both are
scored so the panel can show what the looser assignment buys.

Writes <out>/degron_pb_ranks.csv, degron_pb_activity.csv and degron_pb_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_degron_pb")


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def cp10k(X):
    X = np.asarray(X, np.float32)
    return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1e-9) * 1e4)


def pseudobulk(X, codes, n):
    import scipy.sparse as sp
    M = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))),
                      shape=(n, len(codes)))
    cnt = np.asarray(M.sum(1)).ravel()
    return sp.csr_matrix((M @ X).multiply((1.0 / np.maximum(cnt, 1))[:, None])), cnt


def main(a):
    import anndata as ad
    import scipy.sparse as sp
    from expert import ExPert
    from expert.utils import readout as ro
    from expert.utils.readout import calibrated_readout

    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    q = ad.read_h5ad(a.query)
    o = q.obs.copy()
    o[a.clone_col] = o[a.clone_col].astype(str)
    o["condition"] = o[a.condition_col].astype(str)
    # 'unassigned' is a barcode that did not map, not a clone: it would otherwise get its own
    # pseudobulk and inflate the clone count (it is also the one label absent from the bank).
    drop = {x.strip().lower() for x in a.exclude_clones.split(",") if x.strip()}
    keep = ~o[a.clone_col].str.strip().str.lower().isin(drop | {"nan", ""})
    if a.max_pct_mt and a.mt_col in o:
        dead = o[a.mt_col].to_numpy(float) > a.max_pct_mt
        keep &= ~dead
        log.info("dead-nucleus filter %s <= %g: drops %d cells", a.mt_col, a.max_pct_mt, int(dead.sum()))
    log.info("%d cells, %d with a %s assignment, %d clones", q.n_obs, int(keep.sum()),
             a.clone_col, o.loc[keep, a.clone_col].nunique())

    # ---- active filter, in expression space, BEFORE the model -----------------------------------
    # Per clone: the IAA-vs-DMSO shift against the split-half noise of its own DMSO cells. The
    # null is built per clone so a shallow clone is judged against its own noise, not a global one.
    X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    if a.mask_genes:
        from fnmatch import fnmatch
        pats = [g.strip() for g in a.mask_genes.split(",") if g.strip()]
        hit = np.array([any(fnmatch(v, g) for g in pats) for v in q.var_names.astype(str)])
        if hit.any():      # diagonal multiply, not tolil: 58k x 38.6k would thrash
            X = (X @ sp.diags(np.where(hit, 0.0, 1.0).astype(np.float32))).tocsr()
        log.info("masked %d genes matching %s", int(hit.sum()), pats)
    act_rows = []
    for cl, idx in o[keep].groupby(a.clone_col, observed=True).indices.items():
        ii = np.flatnonzero(keep.to_numpy())[idx]
        ctl = ii[o.condition.to_numpy()[ii] == a.control_label]
        trt = ii[o.condition.to_numpy()[ii] != a.control_label]
        if len(ctl) < a.min_cells or len(trt) < a.min_cells:
            continue
        # normalise this clone's cells ONCE; the halves are then just row means of that matrix
        Nc = cp10k(X[ctl].toarray())
        xc, xt = Nc.mean(0), cp10k(X[trt].toarray()).mean(0)
        shift = float(np.linalg.norm(xt - xc))
        null = []
        for _ in range(a.n_split):
            pm = rng.permutation(len(ctl))
            h1, h2 = pm[: len(pm) // 2], pm[len(pm) // 2:]
            if min(len(h1), len(h2)) < 3:
                continue
            null.append(float(np.linalg.norm(Nc[h1].mean(0) - Nc[h2].mean(0))))
        del Nc
        # Depth correction. A half-vs-half split of n_c control cells carries noise
        # (1/(n_c/2) + 1/(n_c/2)) = 4/n_c, but the quantity being judged, mean(IAA) - mean(DMSO),
        # carries (1/n_t + 1/n_c). Comparing the two directly makes the threshold ~sqrt(2) too
        # strict, which under DiD left exactly ONE clone standing. Rescale the null to the
        # comparison's own depth.
        f = np.sqrt((1.0 / len(trt) + 1.0 / len(ctl)) / (4.0 / len(ctl)))
        null = np.asarray(null, float) * f
        thr = float(np.quantile(null, 0.95)) if len(null) else np.inf
        act_rows.append(dict(clone=cl, n_ctrl=len(ctl), n_treated=len(trt), noise95=thr,
                             depth_factor=float(f), _null=null, _dx=(xt - xc)))
    act = pd.DataFrame(act_rows)
    if len(act):
        # Difference-in-differences: 24 h of auxin moves every clone the same way (solvent, stress,
        # the TIR1 machinery). That common component is not the depletion and, left in, it is most
        # of ||Dx|| -- so the activity call and the readout both run on the clone-SPECIFIC residual.
        Dx = np.stack(act.pop("_dx").to_numpy())
        if a.did:
            Dx = Dx - Dx.mean(0, keepdims=True)
        act["shift"] = np.linalg.norm(Dx, axis=1)
        act["ratio"] = act["shift"] / act.noise95.clip(1e-9)
        act["active"] = act["shift"] > act.noise95
        nulls = act.pop("_null").to_numpy()
        act["p_active"] = [float((np.sum(nl >= sh) + 1) / (len(nl) + 1))
                           for sh, nl in zip(act["shift"], nulls)]
    act.to_csv(out / f"degron_pb_activity_{a.clone_col}{a.tag}.csv", index=False)
    log.info("%d clones with >= %d cells per condition; %d ACTIVE (shift > split-half noise)",
             len(act), a.min_cells, int(act.active.sum()))

    # ---- pseudobulks: one row per (clone, condition) ---------------------------------------------
    sel = keep.to_numpy() & o[a.clone_col].isin(act.clone).to_numpy()
    key = (o.loc[sel, a.clone_col] + "|" + o.loc[sel, "condition"]).to_numpy()
    codes, keys = pd.factorize(key, sort=True)
    Xp, cnt = pseudobulk(X[np.flatnonzero(sel)], codes, len(keys))
    meta = pd.DataFrame([k.split("|") for k in keys], columns=["clone", "condition"])
    meta["n_cells"] = cnt.astype(int)
    meta["is_control"] = meta.condition.eq(a.control_label)
    log.info("%d pseudobulks (%d control)", len(meta), int(meta.is_control.sum()))

    model = ExPert.load(a.model_dir, adata=None)
    pb = ad.AnnData(Xp.astype(np.float32), var=q.var.copy(), obs=meta)
    pb.obs["perturbation"] = np.where(meta.is_control, "control", meta.clone)
    pb.obs[model.original_batch_key] = a.dataset
    var_names = pd.Index(q.var_names.astype(str))
    Xn_pb = cp10k(Xp.toarray())            # the pseudobulks in log1p CP10K, for the probe below
    del q, X
    kw = dict(inplace=True, return_latents=True, soft_predictions=True, head="A",
              hub_correction=False, n_cells_key="n_cells", control_label="control",
              progress_bar=False, n_cores=a.n_cores)
    if a.center_by:
        # Fig. 5's lever: centre the INPUT on the control means of each group separately, instead
        # of one baseline pooled over every control row in the query.
        pb.obs[a.center_by] = pb.obs[a.center_by].astype(str)
        kw["center_by"] = a.center_by
        log.info("input baseline: per %s (center_by)", a.center_by)
    else:
        model.refresh_ctrl_gene_means(pb, min_ctrl=1)
        kw["refresh_ctrl_means"] = False
        log.info("input baseline: ONE pooled baseline over all %d control rows",
                 int(meta.is_control.sum()))
    model.predict(pb, **kw)
    Z = np.asarray(pb.obsm[ro.LATENT_KEY], np.float32)
    kk = [k for k in pb.uns if k.endswith("_latent_cls_proj")]
    P = np.asarray(pb.uns[kk[0]], np.float32)
    labels = np.asarray(model.idx_to_label).astype(str)[:P.shape[0]]
    _, lab_tr, _ = ro.perturbation_prototypes(model)
    nl = len(lab_tr)
    noise = (model.module.readout_noise.detach().cpu().numpy().astype(np.float64)
             if hasattr(model.module, "has_readout_noise") and model.module.has_readout_noise()
             else None)
    log.info("bank %s, %d trained; readout calibration %s", P.shape, nl,
             "present" if noise is not None else "ABSENT (deflate runs every round)")

    # ---- score: the clone's own gene, best of the two tails ---------------------------------------
    e = pb.obs.reset_index(drop=True)
    ctrl = e.is_control.to_numpy()
    cm = {r.clone: Z[i] for i, r in enumerate(e.itertuples()) if ctrl[i]}
    tr = np.flatnonzero(~ctrl)
    D = np.stack([Z[i] - cm[e.clone.iat[i]] for i in tr])
    D0 = D.copy()
    if a.did:
        D = D - D.mean(0, keepdims=True)      # same subtraction, in the latent
    et = e.iloc[tr].reset_index(drop=True)

    def _offset(M, tag):
        U = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        mu = U.mean(0)
        # ||mean of the unit rows||: 1.0 = every row points the same way, ~1/sqrt(n) = unrelated
        log.info("  shared offset %-14s ||mean(unit)||=%.3f  mean pairwise cos=%.3f  "
                 "(chance ~%.3f)", tag, float(np.linalg.norm(mu)),
                 float((U @ U.T)[np.triu_indices(len(U), 1)].mean()), 1 / np.sqrt(len(U)))
    log.info("shared-offset diagnostics on the %d scored contrasts:", len(D))
    _offset(np.stack([Z[i] for i in tr]), "z as encoded")
    _offset(D0, "z - own control")
    _offset(D, "after DiD" if a.did else "no DiD")
    np.save(out / f"degron_pb_delta_{a.clone_col}{a.tag}.npy", D)

    trained = set(map(str, lab_tr))
    in_full = set(map(str, labels))
    miss = sorted(set(et.clone) - in_full)
    log.info("%d/%d clone genes in the full bank, %d trained, ZERO-SHOT: %s%s",
             len(set(et.clone) & in_full), et.clone.nunique(),
             len(set(et.clone) & trained), sorted((set(et.clone) & in_full) - trained),
             f"; NOT IN BANK AT ALL: {miss}" if miss else "")
    # ---- probe: does the clone's OWN transcript fall? --------------------------------------------
    # A degron strips PROTEIN. If the transcript is untouched the response is not a knockout of that
    # gene, and nothing says the readout must find it on the loss side. Measured in the same frame
    # the model sees: the clone's IAA-minus-DMSO shift, and that shift after the across-clone mean
    # is removed (the difference in differences).
    ci = {r.clone: i for i, r in enumerate(e.itertuples()) if ctrl[i]}
    Dx = np.stack([Xn_pb[i] - Xn_pb[ci[e.clone.iat[i]]] for i in tr])
    Dx_did = Dx - Dx.mean(0, keepdims=True)
    gpos = {g: var_names.get_loc(g) for g in set(et.clone) if g in var_names}
    own = []
    for i, cl in enumerate(et.clone):
        j = gpos.get(cl)
        if j is None:
            continue
        own.append(dict(clone=cl, n_cells=int(et.n_cells.iat[i]),
                        dmso_log1p_cp10k=float(Xn_pb[ci[cl], j]),
                        iaa_log1p_cp10k=float(Xn_pb[np.flatnonzero(~ctrl)[i], j]),
                        own_delta=float(Dx[i, j]), own_delta_did=float(Dx_did[i, j])))
    own = pd.DataFrame(own)
    if len(own):
        own.to_csv(out / f"degron_pb_own_transcript_{a.clone_col}{a.tag}.csv", index=False)
        log.info("own transcript (IAA - DMSO, log1p CP10K): median %+.3f over %d clones; "
                 "rises in %d", own.own_delta.median(), len(own), int((own.own_delta > 0).sum()))

    rows = []
    for bank in ("trained", "full"):
        nb = nl if bank == "trained" else len(P)
        Pb, lb = unit(P[:nb]), labels[:nb]
        pos = {g: i for i, g in enumerate(lb)}
        r = calibrated_readout(D, Pb, et.n_cells.to_numpy(), noise=noise, rounds=3, head=10,
                               k=min(a.k, nb))
        # Collapse the two tails into one rank per prototype so the clone's own gene and the
        # permutation null are read off the SAME list. Unlisted prototypes sit past the cut.
        if bank == "full" and a.probe_clones:
            want = {x.strip() for x in a.probe_clones.split(",") if x.strip()}
            pr = []
            for i, cl in enumerate(et.clone):
                if cl not in want:
                    continue
                for t in ("loss", "gain"):
                    ix = np.asarray(r[t][i], np.int64)[:a.probe_top]
                    pr += [dict(clone=cl, tail=t, position=k + 1, gene=lb[j],
                                is_own_gene=bool(lb[j] == cl)) for k, j in enumerate(ix)]
            if pr:
                pd.DataFrame(pr).to_csv(
                    out / f"degron_pb_probe_{a.clone_col}{a.tag}.csv", index=False)
                log.info("wrote top-%d calls for %s", a.probe_top, sorted(want & set(et.clone)))
        for i, cl in enumerate(et.clone):
            if cl not in pos:
                continue
            rr = np.full(nb, nb + 1, np.int64)
            for t in ("loss", "gain"):
                ix = np.asarray(r[t][i], np.int64)
                rr[ix] = np.minimum(rr[ix], np.arange(1, len(ix) + 1))
            others = [pos[g] for g in set(et.clone) - {cl} if g in pos]
            rk = {}
            for t in ("loss", "gain"):
                hit = np.flatnonzero(np.asarray(r[t][i]) == pos[cl])
                rk[t] = int(hit[0]) + 1 if len(hit) else nb + 1
            rows.append(dict(bank=bank, clone=cl, n_cells=int(et.n_cells.iat[i]),
                             rank_loss=rk["loss"], rank_gain=rk["gain"],
                             rank=int(rr[pos[cl]]), n_candidates=nb,
                             zero_shot=cl not in trained,
                             rank_null=float(np.median(rr[others])) if others else np.nan))
    rk = pd.DataFrame(rows).merge(act[["clone", "active", "ratio", "shift", "noise95", "p_active", "n_treated"]], on="clone",
                                  how="left")
    rk["clone_col"] = a.clone_col
    rk.to_csv(out / f"degron_pb_ranks_{a.clone_col}{a.tag}.csv", index=False)

    summ = []
    for bank in ("trained", "full"):
        b = rk.bank.eq(bank)
        act_m = rk.active.fillna(False)
        subsets = [("all", b), ("active", b & act_m), ("inactive", b & ~act_m)]
        if bank == "full":   # the zero-shot clone genes only exist in the full bank
            subsets += [("active_trained", b & act_m & ~rk.zero_shot),
                        ("active_zeroshot", b & act_m & rk.zero_shot)]
        for subset, m in subsets:
            q_ = rk[m]
            if not len(q_):
                continue
            summ.append(dict(clone_col=a.clone_col, bank=bank, subset=subset, n=len(q_),
                             median_rank=float(q_["rank"].median()),
                             median_rank_null=float(q_["rank_null"].median()),
                             **{f"recall@{k}": float((q_["rank"] <= k).mean())
                                for k in (1, 5, 10, 50, 100)},
                             **{f"null@{k}": float((q_["rank_null"] <= k).mean())
                                for k in (10, 50)}))
            log.info("%-5s %-16s n=%2d  median rank %6.0f (null %6.0f)  recall@10 %.3f  @50 %.3f",
                     bank, subset, len(q_), summ[-1]["median_rank"],
                     summ[-1]["median_rank_null"], summ[-1]["recall@10"], summ[-1]["recall@50"])
    pd.DataFrame(summ).to_csv(out / f"degron_pb_summary_{a.clone_col}{a.tag}.csv", index=False)
    json.dump(dict(clone_col=a.clone_col, n_clones=int(len(act)),
                   n_active=int(act.active.sum()),
                   n_zeroshot=int(rk[rk.bank.eq("full")].zero_shot.sum())), open(out / f"degron_pb_meta_{a.clone_col}{a.tag}.json", "w"),
              indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--query", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clone-col", default="clone_nm2")
    ap.add_argument("--condition-col", default="condition")
    ap.add_argument("--control-label", default="DMSO")
    ap.add_argument("--dataset", default="degron_pb")
    ap.add_argument("--did", action=argparse.BooleanOptionalAction, default=True,
                    help="remove the auxin shift shared by all clones (difference-in-differences)")
    ap.add_argument("--mask-genes", default="", help="glob patterns to zero out, comma separated")
    ap.add_argument("--max-pct-mt", type=float, default=0.0, help="0 = no dead-nucleus filter")
    ap.add_argument("--mt-col", default="pct_counts_mt")
    ap.add_argument("--tag", default="", help="suffix for the output files (arm name)")
    ap.add_argument("--exclude-clones", default="unassigned,none,ambiguous,multiplet,doublet",
                    help="comma-separated non-clone labels in --clone-col")
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--k", type=int, default=20000, help="readout list depth (>= deepest rank read)")
    ap.add_argument("--n-split", type=int, default=50, help="split-half draws for the noise null")
    ap.add_argument("--center-by", default="",
                    help="obs column for a PER-GROUP input baseline (Fig. 5 recipe); empty = one pooled baseline")
    ap.add_argument("--probe-clones", default="",
                    help="comma list of clones to dump the full top-k call for")
    ap.add_argument("--probe-top", type=int, default=10)
    ap.add_argument("--n-cores", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
