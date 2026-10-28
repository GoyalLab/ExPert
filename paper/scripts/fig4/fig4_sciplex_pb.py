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

"""sci-Plex3 through the shipped predict() path, on pseudobulks.

STEP=encode   pseudobulk every (plate, cell line, drug, dose) with >= --min-cells cells, then ONE
              predict() over the lot. Mean raw counts per group, no hub correction (predict's
              default), prototypes harvested from the same model.
STEP=score    every readout arm on those latents, from one encode. The DEFAULT readout for the
              figure is best-of-loss-and-gain: a drug that activates its target, or whose feedback
              inverts the target's signature, is invisible to the loss tail (the three HSP90
              inhibitors are the clean case -- HSF1 fires and the chaperones go UP).

                bank       trained (8,964) | full (19,895, zero-shot rows compete)
                readout    deflate (calibrated_readout, depth-matched) | cosine (round 1 only)
                direction  loss (the drug looks like a knockdown of the target) | gain (it looks
                           like the opposite -- an activator) | best of the two

              A drug that ACTIVATES its target cannot be scored on the loss tail alone, which is
              what every earlier sci-Plex arm did; calibrated_readout returns both tails, so the
              gain side costs nothing extra here.

Writes <out>/z_raw.npy + obs.csv (encode) and ranks_<arm>.csv + summary.csv (score).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_sciplex_pb")
GRP = ["plate", "cell_name", "drug", "dose_uM"]


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def encode(a):
    import anndata as ad
    import scipy.sparse as sp
    from expert import ExPert

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    q = ad.read_h5ad(a.query)
    o = q.obs.copy()
    o["drug"] = o[a.drug_col].astype(str)
    o.loc[o.is_control.astype(bool), "drug"] = "control"
    o["dose_uM"] = o.dose_uM.astype(float)
    for c in ("plate", "cell_name"):
        o[c] = o[c].astype(str)
    key = o[GRP].astype(str).agg("|".join, axis=1)
    if a.ctrl_chunk:
        # Controls pool a whole vehicle well (median 274 cells, up to 689) while a drug condition
        # has ~74, and well-to-well systematics do NOT shrink with depth -- so a control-vs-control
        # spread cannot be rescaled to a condition's depth, it has to be MEASURED there. Cut the
        # controls into chunks the size of a real condition; the control mean is unchanged (same
        # cells, just averaged in more pieces) and the leave-one-out spread among them becomes a
        # null at the depth it is used to judge.
        ic = np.flatnonzero(o.is_control.astype(bool).to_numpy())
        rr = np.random.default_rng(0)
        chunk = pd.Series("", index=o.index, dtype=object)
        for _, g in o.iloc[ic].groupby(["plate", "cell_name"], observed=True):
            ii = rr.permutation(g.index.to_numpy())
            nch = max(1, int(round(len(ii) / a.ctrl_chunk)))
            for j, part in enumerate(np.array_split(ii, nch)):
                chunk.loc[part] = f"#c{j}"
        key = key + chunk
        log.info("control wells cut into ~%d-cell chunks: %d control groups",
                 a.ctrl_chunk, key[o.is_control.astype(bool).to_numpy()].nunique())
    codes, keys = pd.factorize(key, sort=True)
    n = len(keys)
    log.info("%d cells -> %d candidate pseudobulks", q.n_obs, n)

    # mean RAW counts per group: one sparse matrix multiply, not a python loop over groups
    M = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))),
                      shape=(n, q.n_obs))
    cnt = np.asarray(M.sum(1)).ravel()
    X = (M @ q.X).tocsr()                      # sparse ops keep returning coo; force csr back
    X = sp.csr_matrix(X.multiply((1.0 / np.maximum(cnt, 1))[:, None]))
    if a.mask_genes:
        # Masking is linear and so is the group mean, so masking the pseudobulks is identical to
        # masking the cells -- and far cheaper. Controls are masked too, so the baseline that
        # refresh_ctrl_gene_means builds has the same genes zeroed and Delta-z is unaffected.
        # On Tahoe this was worth 1.7x at recall@10 (every mtmask arm beat every unmasked one).
        from fnmatch import fnmatch
        pats = [g.strip() for g in a.mask_genes.split(",") if g.strip()]
        hit = np.array([any(fnmatch(v, g) for g in pats) for v in q.var_names.astype(str)])
        if hit.any():
            X = (X @ sp.diags(np.where(hit, 0.0, 1.0).astype(np.float32))).tocsr()
        log.info("input mask %r: %d of %d genes zeroed (%s)", a.mask_genes, int(hit.sum()),
                 X.shape[1], list(q.var_names.astype(str)[hit])[:10])
    meta = pd.DataFrame([k.split("#c")[0].split("|") for k in keys], columns=GRP)
    meta["dose_uM"] = meta.dose_uM.astype(float)
    meta["n_cells"] = cnt.astype(int)
    keep = meta.n_cells.to_numpy() >= a.min_cells
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    log.info("%d pseudobulks with >= %d cells (%d control)", len(meta), a.min_cells,
             int((meta.drug == "control").sum()))

    # the annotation travels with the group, not with the cell
    ann = (o.groupby(["cell_name", "drug"], observed=True)
           [["targets", "trained_targets", "zeroshot_targets", "target_class"]]
           .first().reset_index())
    meta = meta.merge(ann, on=["cell_name", "drug"], how="left")

    model = ExPert.load(a.model_dir, adata=None)
    pb = ad.AnnData(X.astype(np.float32), var=q.var.copy(), obs=meta)
    pb.obs["perturbation"] = pb.obs.drug.to_numpy()
    # refresh_ctrl_gene_means needs the encoder's batch key as well as the class column; a name the
    # model never trained on is what we want here -- it then centres on THIS query's own controls
    pb.obs[model.original_batch_key] = a.dataset
    del q
    # predict() keeps the TRAINING baseline below 10 control rows; a pseudobulk query has far
    # fewer, so set the control means once and tell predict() not to recompute them.
    model.refresh_ctrl_gene_means(pb, min_ctrl=1)
    # soft_predictions=True (obs_only=False) so uns carries the FULL scoring bank: the trained
    # atoms plus the f(missing) zero-shot rows, which is what the full-bank arm ranks against.
    from expert.utils import readout as ro
    model.predict(pb, inplace=True, return_latents=True, soft_predictions=True,
                  head="A", hub_correction=False, refresh_ctrl_means=False,
                  n_cells_key="n_cells", control_label="control",
                  progress_bar=False, n_cores=a.n_cores)
    Z = np.asarray(pb.obsm[ro.LATENT_KEY], np.float32)
    np.save(out / "z_raw.npy", Z)
    pb.obs.to_csv(out / "obs.csv", index=False)
    # the expression pseudobulks go out too: the signature / centroid baseline is scored on the
    # same rows, and recomputing them from 346k cells for every comparator is wasteful
    ad.AnnData(X.astype(np.float32), var=pb.var[[]].copy(), obs=meta).write_h5ad(
        out / "pseudobulk.h5ad", compression="gzip")
    key = [k for k in pb.uns if k.endswith("_latent_cls_proj")]
    if not key:
        raise SystemExit(f"no *_latent_cls_proj in uns; keys {list(pb.uns)}")
    P = np.asarray(pb.uns[key[0]], np.float32)
    labels = np.asarray(model.idx_to_label).astype(str)[:P.shape[0]]
    _, lab_tr, _ = ro.perturbation_prototypes(model)
    np.savez(out / "bank.npz", P=P, labels=labels, n_labels=len(lab_tr))
    log.info("bank %s, %d trained", P.shape, len(lab_tr))
    mod = model.module
    if hasattr(mod, "has_readout_noise") and mod.has_readout_noise():
        np.save(out / "noise.npy",
                mod.readout_noise.detach().cpu().numpy().astype(np.float64))
    else:
        log.warning("no readout calibration on this checkpoint: deflate runs every round unstopped")
    log.info("z %s -> %s", Z.shape, out)


def score(a):
    from expert.utils.readout import calibrated_readout

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    enc = Path(a.encode_dir)
    Z = np.load(enc / "z_raw.npy").astype(np.float32)
    o = pd.read_csv(enc / "obs.csv")
    b = np.load(enc / "bank.npz", allow_pickle=True)
    P, L, nl = b["P"].astype(np.float32), b["labels"].astype(str), int(b["n_labels"])
    noise = np.load(enc / "noise.npy") if (enc / "noise.npy").exists() else None
    pos = {g: i for i, g in enumerate(L)}

    ctrl = (o.drug == "control").to_numpy()
    # latent reference: the control pseudobulks of the same plate x line, else the same line
    ref = {}
    for keys in (["plate", "cell_name"], ["cell_name"]):
        for k, idx in o[ctrl].groupby(keys, observed=True).indices.items():
            ref.setdefault(tuple(np.atleast_1d(k)), Z[np.flatnonzero(ctrl)[idx]].mean(0))
    d = ~ctrl
    base = np.stack([ref.get((r.plate, r.cell_name), ref[(r.cell_name,)])
                     for r in o[d].itertuples()])
    D = Z[d] - base
    e = o[d].reset_index(drop=True)
    log.info("%d treated pseudobulks, %d drugs, %d lines, doses %s", len(e), e.drug.nunique(),
             e.cell_name.nunique(), sorted(e.dose_uM.unique()))

    # ranks only compare across methods that score the SAME candidate set, so the arm that gets
    # compared to fingerprinting is restricted to its dictionary (the collaborator's matched panel)
    cand = None
    if a.candidates:
        z = np.load(a.candidates, allow_pickle=True)
        want = set(np.asarray(z["genes"] if "genes" in z else z[z.files[0]]).astype(str))
        cand = np.array([i for i, g in enumerate(L[:nl]) if g in want], int)
        log.info("matched candidate panel: %d of %d requested genes are trained prototypes",
                 len(cand), len(want))

    rows, summ = [], []
    banks = ["trained", "full"] + (["matched"] if cand is not None else [])
    for bank in banks:
        sub = cand if bank == "matched" else np.arange(nl if bank == "trained" else len(P))
        nb = len(sub)
        Pb = unit(P[sub])
        lab_b = L[sub]
        pos_b = {g: i for i, g in enumerate(lab_b)}
        tcol = "targets" if bank == "full" else "trained_targets"
        tgt = [[pos_b[g.strip()] for g in str(s).replace(",", ";").split(";")
                if g.strip() and g.strip() != "nan" and g.strip() in pos_b]
               for s in e[tcol].fillna("")]
        for ro_mode in ("deflate", "cosine"):
            if ro_mode == "deflate":
                r = calibrated_readout(D, Pb, e.n_cells.to_numpy(), noise=noise,
                                       rounds=a.rounds, head=a.head, k=min(a.top, nb))
                order = {"loss": r["loss"], "gain": r["gain"]}
                S = None
            else:
                S = unit(D) @ Pb.T
                order = {"loss": np.argsort(-S, 1)[:, :min(a.top, nb)],
                         "gain": np.argsort(S, 1)[:, :min(a.top, nb)]}
            rk = {}
            for tail, idx in order.items():
                # rank of each row's own target in that tail's merged list (nb + 1 = not listed)
                rr = np.full(len(e), nb + 1, int)
                for i, t in enumerate(tgt):
                    if not t:
                        rr[i] = -1; continue
                    hit = np.flatnonzero(np.isin(idx[i], t))
                    if len(hit):
                        rr[i] = int(hit[0]) + 1
                rk[tail] = rr
            rk["best"] = np.where((rk["loss"] < 0) | (rk["gain"] < 0), -1,
                                  np.minimum(rk["loss"], rk["gain"]))
            for tail in ("loss", "gain", "best"):
                arm = f"{bank}_{ro_mode}_{tail}"
                rows.append(pd.DataFrame(dict(
                    arm=arm, bank=bank, readout=ro_mode, tail=tail,
                    condition=e.drug + "@" + e.dose_uM.astype(str) + "@" + e.cell_name,
                    drug=e.drug, dose_uM=e.dose_uM, cell_name=e.cell_name,
                    n_cells=e.n_cells, target_class=e.target_class, rank=rk[tail])))
                q = rk[tail][rk[tail] > 0]
                n_ann = int((rk[tail] >= 0).sum())
                summ.append(dict(arm=arm, bank=bank, readout=ro_mode, tail=tail, n=n_ann,
                                 **{f"recall@{k}": float((q <= k).sum() / max(n_ann, 1))
                                    for k in (1, 5, 10, 20, 50, 100)}))
                log.info("%-24s n=%4d  recall@10 %.3f  recall@50 %.3f", arm, n_ann,
                         summ[-1]["recall@10"], summ[-1]["recall@50"])
    # ---- how much does deflation actually move, once the protected head stops masking it? -------
    # The merged list holds its first `head` slots for round 1, so with head=10 the top 10 IS the
    # cosine readout by construction and deflate==cosine there whatever the rounds do. This sweep
    # lowers the protection and deepens the rounds to measure the readout itself. The stopping rule
    # needs add_readout_calibration, which v0.1.0 does NOT carry, so every round always runs.
    if a.deflate_sweep:
        sub = cand if cand is not None else np.arange(nl)
        Pb, lab_b = unit(P[sub]), L[sub]
        pos_b = {g: i for i, g in enumerate(lab_b)}
        tgt = [[pos_b[g.strip()] for g in str(x).replace(",", ";").split(";")
                if g.strip() in pos_b] for x in e["trained_targets"].fillna("")]
        has = np.array([bool(t) for t in tgt])
        srow = []
        for rounds in [int(x) for x in a.sweep_rounds.split(",")]:
            for head in [int(x) for x in a.sweep_head.split(",")]:
                r = calibrated_readout(D, Pb, e.n_cells.to_numpy(), noise=noise, rounds=rounds,
                                       head=head, k=min(a.top, len(sub)))
                rec = {}
                for tail in ("loss", "gain", "best"):
                    if tail == "best":
                        rr = np.minimum(rec["_loss"], rec["_gain"])
                    else:
                        idx = r[tail]
                        rr = np.array([(np.flatnonzero(np.isin(idx[i], t))[:1].tolist() or
                                        [len(sub)])[0] + 1 if t else len(sub) + 1
                                       for i, t in enumerate(tgt)])
                        rec[f"_{tail}"] = rr
                    for k in (10, 50):
                        rec[f"{tail}@{k}"] = float((rr[has] <= k).mean())
                srow.append(dict(rounds=rounds, head=head, n_rounds_used=float(np.mean(r["n_rounds"])),
                                 **{k: v for k, v in rec.items() if not k.startswith("_")}))
                log.info("deflate rounds=%2d head=%2d  used %.2f rounds | loss@10 %.4f best@10 %.4f "
                         "best@50 %.4f", rounds, head, srow[-1]["n_rounds_used"],
                         srow[-1]["loss@10"], srow[-1]["best@10"], srow[-1]["best@50"])
        pd.DataFrame(srow).to_csv(out / "deflate_sweep.csv", index=False)

    pd.concat(rows).to_csv(out / "ranks_arms.csv", index=False)
    s = pd.DataFrame(summ).sort_values("recall@10", ascending=False)
    s.to_csv(out / "summary_arms.csv", index=False)
    log.info("best arms:\n%s", s.head(6).to_string(index=False))
    json.dump(dict(n_conditions=int(len(e)), n_drugs=int(e.drug.nunique()),
                   best_arm=str(s.arm.iloc[0])), open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", required=True, choices=["encode", "score"])
    ap.add_argument("--query")
    ap.add_argument("--model-dir")
    ap.add_argument("--encode-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drug-col", default="drug_clean")
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--ctrl-chunk", type=int, default=0,
                    help="encode: cut control wells into ~N-cell pseudobulks so control-vs-control "
                         "noise is measured at a real condition's depth (0 = one per well)")
    ap.add_argument("--mask-genes", default="", help="encode: input mask, e.g. 'MT-*,MT1*,MT2A,MT3,MT4'")
    ap.add_argument("--dataset", default="sciplex3_pb", help="value for the encoder's batch key")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--head", type=int, default=10)
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--candidates", default="", help="npz with a 'genes' array: the matched panel")
    ap.add_argument("--deflate-sweep", action="store_true", help="sweep deflate rounds x protected head")
    ap.add_argument("--sweep-rounds", default="1,3,6,10")
    ap.add_argument("--sweep-head", default="0,1,3,5,10")
    ap.add_argument("--n-cores", type=int, default=8)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main_args = ap.parse_args()
    (encode if main_args.step == "encode" else score)(main_args)
