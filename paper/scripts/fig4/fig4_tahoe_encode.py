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

"""Fig. 5 stage 1 (Tahoe-100M): encode the drug screen under one INPUT-centring arm.

The encoder centres log1p(x) on the mean of the cells labelled 'control' IN THE CALL
(`center_mode='ctrl'`), so the only way to choose what a drug cell is centred on is to choose which
cells share its forward pass. The arms:

    global              one pass, all DMSO pooled (the July read, up to the readout path)
    line                one pass per cell line, that line's DMSO
    plate_line          one pass per plate x line, that plate's DMSO for that line. Needs
                        --controls (plate-matched DMSO): the subset's own 5,000 DMSO are 98% plate1
    plate_line_alldrug  one pass per plate x line with EVERY cell labelled control, so the input is
                        centred on the plate x line all-drug mean -- generic drug stress is
                        subtracted before the encoder rather than after it

The latent is written RAW (center_mode = zero vector); latent centring is a subtraction and happens in
stage 2. One extra pass on a subsample harvests the FULL scoring bank (trained atoms + f(missing)
zero-shot rows) that predict(obs_only=False) scores against, so zero-shot targets are rankable.

Writes <out>/z_<arm>.npy (float16), eff_<arm>.npy, obs_<arm>.parquet, encode_<arm>.json and, with
--harvest-bank, bank.npz (P, labels, n_labels).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(format="%(name)s | %(message)s", level=logging.INFO, force=True)
log = logging.getLogger("fig4_tahoe_encode")

ARMS = ("global", "line", "plate_line", "plate_line_alldrug")
OBS_KEEP = ["drug_clean", "dose_uM", "cell_name", "plate", "is_control", "targets",
            "trained_targets", "zeroshot_targets", "target_class", "single_target"]
CTRL = "__ctrl__"


def load_query(path, controls, max_cells):
    import anndata as ad
    q = ad.read_h5ad(path)
    q.obs = q.obs[OBS_KEEP].copy()
    q.obs["is_control"] = q.obs["is_control"].astype(bool)
    if controls:
        c = ad.read_h5ad(controls)
        c.obs = c.obs[OBS_KEEP].copy()
        c.obs["is_control"] = c.obs["is_control"].astype(bool)
        if not c.obs["is_control"].all():
            raise SystemExit(f"--controls {controls} holds non-control cells")
        if list(c.var_names) != list(q.var_names):
            raise SystemExit("--controls var differs from --query var")
        log.info("replacing %d subset DMSO with %d plate-matched DMSO", int(q.obs.is_control.sum()),
                 c.n_obs)
        q = ad.concat([q[~q.obs["is_control"].to_numpy()], c], join="outer", merge="first")
        q.obs_names_make_unique()
    if max_cells and q.n_obs > max_cells:
        rng = np.random.default_rng(0)
        q = q[np.sort(rng.choice(q.n_obs, max_cells, replace=False))].copy()
    for col in ("drug_clean", "cell_name", "plate", "targets", "trained_targets",
                "zeroshot_targets", "target_class"):
        q.obs[col] = q.obs[col].astype(str)
    q.obs["condition"] = np.where(q.obs["is_control"], CTRL, q.obs["drug_clean"])
    q.X = q.X.tocsr()
    return q


def subsets(obs, arm):
    if arm == "global":
        return {"all": np.ones(len(obs), bool)}
    keys = ["cell_name"] if arm == "line" else ["plate", "cell_name"]
    lab = obs[keys].astype(str).agg("@".join, axis=1).to_numpy()
    return {k: lab == k for k in sorted(set(lab))}


def main(a):
    import expert.utils.readout as ro
    from expert import ExPert

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    model = ExPert.load(a.model_dir)
    mod, enc = model.module, model.module.z_encoder
    n_shared = int(mod.align_ds_mean.shape[1])
    facts = dict(n_shared=n_shared, n_labels=int(mod.n_labels),
                 encoder_center_input=bool(getattr(enc, "center_input", False)),
                 encoder_center_mode=str(getattr(enc, "center_mode", None)))
    log.info("model facts %s", facts)

    q = load_query(a.query, a.controls, a.max_cells)
    # predict() drops every non-model gene anyway; trimming once here keeps the per-subset copies
    # small (the line arm peaked at 91 GB on Tahoe's 62,710 columns)
    mg = set(np.asarray(model.adata.var_names).astype(str))
    q = q[:, np.array([v in mg for v in q.var_names.astype(str)])].copy()
    log.info("query %d cells x %d genes, %d DMSO", q.n_obs, q.n_vars, int(q.obs.is_control.sum()))
    ro.build_groups(q, "condition", control_condition=CTRL)
    ro.mark_query(model, q)
    cls_col = model.adata_manager.get_state_registry("labels").original_key
    is_ctrl = (q.obs[cls_col].astype(str) == "control").to_numpy()
    if not np.array_equal(is_ctrl, q.obs["is_control"].to_numpy()):
        raise RuntimeError("mark_query control labels disagree with is_control")

    def run(sub):
        model.predict(sub, inplace=True, return_latents=True, soft_predictions=False, head="A",
                      obs_only=True, center_mode=np.zeros(n_shared, np.float32),
                      control_label="control", batch_size=a.batch_size, n_cores=a.n_cores,
                      progress_bar=False)
        eff = sub.obs["eff_hat"].to_numpy(np.float32) if "eff_hat" in sub.obs else \
            np.full(sub.n_obs, np.nan, np.float32)
        rp = sub.obs["response_probs"].to_numpy(np.float32) if "response_probs" in sub.obs else \
            np.full(sub.n_obs, np.nan, np.float32)
        return np.asarray(sub.obsm[ro.LATENT_KEY], np.float32), eff, rp

    parts = subsets(q.obs, a.arm)
    Z = np.full((q.n_obs, n_shared), np.nan, np.float32)
    E = np.full(q.n_obs, np.nan, np.float32)
    RP = np.full(q.n_obs, np.nan, np.float32)
    report, skipped = {}, {}
    for name, m in parts.items():
        n_ctrl = int((m & is_ctrl).sum())
        if a.arm != "plate_line_alldrug" and n_ctrl < a.min_ctrl:
            # never encode on a fabricated baseline: below 10 controls predict() silently keeps
            # the training one
            skipped[name] = dict(n_cells=int(m.sum()), n_ctrl=n_ctrl)
            continue
        t0 = time.time()
        sub = q[m].copy()
        if a.arm == "plate_line_alldrug":
            sub.obs[cls_col] = "control"
        Z[m], E[m], RP[m] = run(sub)
        # scvi keeps an AnnDataManager -- and with it the AnnData -- for every query predict()
        # registers: ~0.4 GB per subset, which OOMs the ~700-subset plate x line arms at 96 GB.
        # predict() registers an internal COPY, so deregister(sub) misses it: clear every manager
        # except the model's own
        model.deregister_manager()
        if len(report) % 50 == 0:
            log.info("%d/%d subsets, maxrss %.1f GB", len(report), len(parts),
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        report[name] = dict(n_cells=int(m.sum()), n_ctrl=n_ctrl, sec=round(time.time() - t0, 1))
        del sub
        gc.collect()
    if skipped:
        n_drug_lost = sum(v["n_cells"] - v["n_ctrl"] for v in skipped.values())
        log.warning("%d subsets skipped (< %d controls): %d drug cells unencoded", len(skipped),
                    a.min_ctrl, n_drug_lost)
    log.info("encoded %d subsets in %.1f min", len(report), (time.time() - t_all) / 60)

    np.save(out / f"z_{a.arm}.npy", Z.astype(np.float16))
    np.save(out / f"eff_{a.arm}.npy", E)
    np.save(out / f"respprob_{a.arm}.npy", RP)
    q.obs[OBS_KEEP].reset_index(names="cell").to_parquet(out / f"obs_{a.arm}.parquet")

    bank = None
    if a.harvest_bank:
        rng = np.random.default_rng(0)
        idx = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(is_ctrl), min(1000, int(is_ctrl.sum())), replace=False),
            rng.choice(np.flatnonzero(~is_ctrl), 1000, replace=False)]))
        sv = q[idx].copy()
        model.predict(sv, inplace=True, return_latents=True, soft_predictions=True, head="A",
                      obs_only=False, center_mode="query_mean", control_label="control",
                      batch_size=a.batch_size, n_cores=a.n_cores, progress_bar=False)
        key = [k for k in sv.uns if k.endswith("_latent_cls_proj")]
        if not key:
            raise SystemExit(f"no *_latent_cls_proj in uns; keys {list(sv.uns)}")
        P = np.asarray(sv.uns[key[0]], np.float32)
        labels = np.asarray(model.idx_to_label).astype(str)[:P.shape[0]]
        src, lab_tr, A = ro.perturbation_prototypes(model)
        n_lab = len(lab_tr)
        rc = (ro._unit(P[:n_lab].astype(np.float64)) * ro._unit(A)).sum(1)
        # head-A argmax over the full softmax must equal argmax cosine(z, bank)
        soft = sv.obsm["soft_predictions"]
        zs = np.asarray(sv.obsm[ro.LATENT_KEY], np.float64)
        am = np.asarray(soft.columns).astype(str)[np.asarray(soft.values).argmax(1)]
        am_bank = labels[(ro._unit(zs) @ ro._unit(P.astype(np.float64)).T).argmax(1)]
        bank = dict(shape=list(P.shape), n_labels=n_lab, atoms_source=src,
                    trained_labels_match=bool((lab_tr == labels[:n_lab]).all()),
                    trained_rows_vs_atoms_rowcos_min=float(rc.min()),
                    head_argmax_equals_bank_argmax=float((am == am_bank).mean()))
        log.info("bank %s", bank)
        np.savez(out / "bank.npz", P=P, labels=labels, n_labels=n_lab)

    json.dump(dict(model_dir=a.model_dir, query=a.query, controls=a.controls, arm=a.arm,
                   model_facts=facts, n_cells=int(q.n_obs), n_ctrl=int(is_ctrl.sum()),
                   n_subsets=len(report), skipped=skipped, subsets=report, bank=bank,
                   minutes=(time.time() - t_all) / 60),
              open(out / f"encode_{a.arm}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--query", required=True, help="tahoe100m_ood_subset.h5ad")
    ap.add_argument("--controls", default=None,
                    help="plate-matched DMSO h5ad; REPLACES the query's own DMSO when given")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--min-ctrl", type=int, default=30)
    ap.add_argument("--harvest-bank", action="store_true")
    ap.add_argument("--max-cells", type=int, default=0, help="random subsample (smoke)")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-cores", type=int, default=8)
    main(ap.parse_args())
