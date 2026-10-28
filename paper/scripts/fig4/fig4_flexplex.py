# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Figure 4E from the FlexPlex HDAC screen (GSE339463), computed here rather than re-plotted.

Similarity of the nominally HDAC6-selective compounds (ricolinostat, ACY-738) to the pan-HDAC
inhibitor AR-42, as the Spearman correlation between their rankings at the SAME concentration over
the candidates shared by ExPert and the fingerprinting dictionary. Scored identically for every arm:

  ExPert arms        predict() on every CELL with the latent centre fixed at zero, then each
                     condition's delta = its cells' mean latent minus the NT cells' mean
                     (readout.group_response), scored by cosine against the trained atoms
                     (readout.score_prototypes), restricted to the shared candidates

ENCODE CELLS, NOT PSEUDOBULKS. The encoder is nonlinear, so averaging cells and then encoding is not
the same as encoding cells and then averaging: the pseudobulk route agreed with the published
FlexPlex scores at a median per-condition Spearman of only 0.66. The cell route is the one the
published panel used (the collaborator's run_readout.py: --control NT --center-mode zero
--min-cells 30 --k 200), and every call below is this repo's own expert.utils.readout.
  RNA fingerprinting the published per-condition pip/cor, ranked PIP descending with ties broken by
                     |cor| (its own pipeline's order), scored as -rank so that higher is better

Spearman is invariant to the monotone score -> rank map, so correlating scores and correlating
rankings give the same number; we correlate the rankings, as the Methods state.

`--keep-genes` reproduces the collaborator's `_fpgenes` runs: every MODEL input gene outside the
keep list is DROPPED from the query (it is then zero-filled at the model's input), while genes the
model never reads are left alone -- exactly make_genemask_query.py, so neither method sees genes the
other cannot.

`--check-dir` points at a published run directory (latents/z.npy, scores_pooled.npz) and reports how
closely this run reproduces it, latent by latent and score by score.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger("flexplex")
unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def encode(q, model_dir, a):
    """The published readout, per cell: returns (scores groups x trained labels, latents, labels)."""
    from expert import ExPert
    from expert.utils import readout as ro
    model = ExPert.load(model_dir)
    q = q.copy()
    if a.keep_genes:
        keep = {l.strip() for l in open(a.keep_genes) if l.strip()}
        mg = set(map(str, model.adata.var_names))
        assert keep <= mg, f"{len(keep - mg)} keep genes are not model input genes"
        drop = q.var_names.isin(sorted(mg - keep))
        log.info("--keep-genes: model inputs %d, kept %d, dropping %d model genes from the query",
                 len(mg), len(keep), int(drop.sum()))
        q = q[:, ~drop].copy()
    ro.build_groups(q, a.condition_col, None, control_condition=a.control_drug)
    ro.mark_query(model, q)
    adm = getattr(model.module, "align_ds_mean", None)
    assert adm is not None and adm.ndim == 2, "no align_ds_mean on this model"
    model.predict(q, inplace=True, return_latents=True,
                  center_mode=np.zeros(int(adm.shape[1]), np.float32),
                  control_label=q.uns["expert_readout"]["control_label"])
    Z = np.asarray(q.obsm[ro.LATENT_KEY], np.float32)
    ro.build_groups(q, a.condition_col, None, control_condition=a.control_drug, key_added="g_pooled")
    ro.group_response(q, group_key="g_pooled", min_cells=a.min_cells, uns_key="ro_pooled")
    ro.score_prototypes(model, q, contrast="delta", k=a.k, uns_key="ro_pooled")
    S = ro.get_scores(q, "delta", uns_key="ro_pooled")
    S.index = [g.split("|")[0] for g in S.index]
    return S, Z, q.obs_names.to_numpy(str)


def check(lab, S, Z, cells, ref_dir):
    """How closely this run reproduces a published run directory."""
    ref = Path(ref_dir)
    zf = ref / "latents" / "z.npy"
    if zf.exists():
        Zr = np.load(zf).astype(np.float32)
        if Zr.shape == Z.shape:
            d = np.abs(Z - Zr)
            log.info("[check %s] latents %s: max|diff| %.3e, mean|diff| %.3e", lab, Z.shape,
                     float(d.max()), float(d.mean()))
        else:
            log.info("[check %s] latent shapes differ: ours %s vs published %s", lab, Z.shape, Zr.shape)
    sf = ref / "scores_pooled.npz"
    if sf.exists():
        z = np.load(sf, allow_pickle=True)
        R = pd.DataFrame(z["S"], index=[g.split("|")[0] for g in z["groups"].astype(str)],
                         columns=z["labels"].astype(str))
        rows = [r for r in S.index if r in R.index]
        cols = [c for c in S.columns if c in R.columns]
        d = np.abs(S.loc[rows, cols].to_numpy() - R.loc[rows, cols].to_numpy())
        sp = [spearmanr(S.loc[r, cols], R.loc[r, cols]).correlation for r in rows]
        log.info("[check %s] scores %d x %d: max|diff| %.3e; per-condition Spearman median %.6f min %.6f",
                 lab, len(rows), len(cols), float(d.max()), float(np.median(sp)), float(np.min(sp)))


def sim_to_ref(rank_df, ref_drug, drugs):
    """Spearman of each compound's ranking against the reference drug's at the SAME dose."""
    out = []
    idx = {c: c for c in rank_df.index}
    for c in rank_df.index:
        if "@" not in c:
            continue
        d, dose = c.split("@")[0], c.split("@")[1]
        if d not in drugs:
            continue
        ref = f"{ref_drug}@{dose}"
        if ref not in idx:
            continue
        out.append(dict(drug=d, dose_nM=int(dose[:-2]),
                        sim_AR42=float(spearmanr(rank_df.loc[c], rank_df.loc[ref]).correlation)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--models", required=True,
                    help="label=model_dir, SEMICOLON separated (a label may contain a comma)")
    ap.add_argument("--fp-pip", required=True)
    ap.add_argument("--fp-cor", required=True)
    ap.add_argument("--fp-label", default="RNA fingerprinting")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-genes", default="", help="model input genes to KEEP; other model genes are dropped")
    ap.add_argument("--check-dirs", default="", help="label=published_run_dir, semicolon separated")
    ap.add_argument("--ref-drug", default="AR-42")
    ap.add_argument("--drugs", default="Ricolinostat,ACY-738")
    ap.add_argument("--condition-col", default="condition")
    ap.add_argument("--drug-col", default="drug")
    ap.add_argument("--control-drug", default="NT")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--k", type=int, default=200, help="score_prototypes k (as published)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    drugs = [x.strip() for x in a.drugs.split(",") if x.strip()]

    q = ad.read_h5ad(a.query)
    checks = dict(c.split("=", 1) for c in a.check_dirs.split(";") if c.strip())
    pip = pd.read_csv(a.fp_pip, index_col=0)
    cor = pd.read_csv(a.fp_cor, index_col=0)
    rows, ranks = [], {}
    for spec in a.models.split(";"):
        lab, mdir = spec.split("=", 1)
        S, Z, cells = encode(q, mdir, a)
        if lab in checks:
            check(lab, S, Z, cells, checks[lab])
        sh = sorted(set(pip.columns) & set(S.columns))
        sc = S[sh]
        # the score matrix goes out too: it is what a published scores_pooled.npz can be diffed against
        sc.to_csv(out / f"scores_{lab.replace(',', '').replace(' ', '_')}.csv")
        ranks[lab] = sc.rank(axis=1, ascending=False)
        log.info("[%s] %d conditions x %d shared candidates", lab, *ranks[lab].shape)
        rows.append(sim_to_ref(ranks[lab], a.ref_drug, drugs).assign(method=lab))

    sh = sorted(set(pip.columns) & set(ranks[list(ranks)[0]].columns))
    p_, c_ = pip[sh], cor[sh]
    rk = {c: pd.Series(np.argsort(np.lexsort((-c_.loc[c].to_numpy(), -p_.loc[c].to_numpy()))) + 1,
                       index=sh) for c in p_.index}
    ranks[a.fp_label] = pd.DataFrame(rk).T
    log.info("[%s] %d conditions x %d shared candidates", a.fp_label, *ranks[a.fp_label].shape)
    rows.append(sim_to_ref(ranks[a.fp_label], a.ref_drug, drugs).assign(method=a.fp_label))

    D = pd.concat(rows, ignore_index=True)[["method", "drug", "dose_nM", "sim_AR42"]]
    D.to_csv(out / "fig4e_ar42.csv", index=False)
    print(D.pivot_table(index=["drug", "dose_nM"], columns="method", values="sim_AR42").round(3).to_string())
    print(f"\nwrote {out / 'fig4e_ar42.csv'}")


if __name__ == "__main__":
    main()
