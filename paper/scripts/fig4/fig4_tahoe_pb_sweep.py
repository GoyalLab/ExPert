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

"""Fig. 5 full-Tahoe readout sweep: score every encoding arm x latent reference x bank. No model.

Encoding arms (fig4_tahoe_pb.py --stage encode) vary the MT input mask and the INPUT centre. Here:

    latent reference  what Δz subtracts from a drug pseudobulk's z:
                        pl     the DMSO pseudobulk of the SAME plate x line
                        line   the mean DMSO z of that line over all plates
                        plate  the mean DMSO z of that plate over all lines
    bank              trained  trained targets only, ranked among the trained prototypes
                      full     all annotated targets, ranked among all 19,895 (zero-shot rows compete)
    level             drug (mean Δz over lines and doses), drug_dose, entry (drug x line x dose)

Metric per cell: recall@1/10/50 and the target-permutation null (other queries' target sets ranked
in each query's own ranking; at entry level within the same line x dose), median rank vs median
null rank, one-sided paired t-test on log10 rank. Inhibitor / antagonist drugs only.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_tahoe_pb_sweep")
KS = (1, 10, 50)


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def split(v):
    s = str(v).strip()
    return [t.strip() for t in s.replace(",", ";").split(";") if t.strip()] if s and s.lower() != "nan" else []


def ranks(S):
    rk = np.empty(S.shape, np.int32)
    np.put_along_axis(rk, np.argsort(-S, axis=1), np.arange(1, S.shape[1] + 1)[None, :], axis=1)
    return rk


def rank_matrix(R, Pb, sets, pen=None):
    """M[p, q] = best rank of query q's target set in query p's ranking (cos - pen if given)."""
    S = unit(R) @ Pb.T
    rk = ranks(S if pen is None else S - pen[None, :])
    return np.stack([rk[:, s].min(1) for s in sets], axis=1).astype(float)


def collect(M_blocks, C=None, set_sizes=None):
    """Observed (diag) and per-query null (off-diagonal) ranks pooled over blocks."""
    obs, null_hit = [], {k: [] for k in KS}
    null_med, null_log = [], []
    for M in M_blocks:
        n = len(M)
        if n < 2:
            continue
        off = ~np.eye(n, dtype=bool)
        obs.append(np.diag(M))
        for k in KS:
            null_hit[k].append((M[off] <= k).reshape(n, n - 1).mean(1))
        null_med.append(np.median(M[off].reshape(n, n - 1), axis=1))
        null_log.append(np.log10(M[off].reshape(n, n - 1)).mean(1))
    obs = np.concatenate(obs)
    from scipy.stats import ttest_rel
    out = dict(n=int(len(obs)), median_rank=float(np.median(obs)),
               median_null_rank=float(np.median(np.concatenate(null_med))),
               p_ttest=float(ttest_rel(np.log10(obs), np.concatenate(null_log), alternative="less").pvalue))
    for k in KS:
        out[f"r@{k}"] = float((obs <= k).mean())
        out[f"null@{k}"] = float(np.concatenate(null_hit[k]).mean())
        if C is not None:
            # analytic chance: a random ranking puts >=1 of m targets in the top k
            m = np.asarray(set_sizes, float)
            out[f"chance@{k}"] = float(np.mean(1 - (1 - k / C) ** m))
    return out


def main(a):
    pen_all = np.load(a.hub_pen)["pen"][0].astype(np.float32) if a.hub_pen else None
    b = np.load(a.bank, allow_pickle=True)
    P, labels, n_lab = unit(b["P"].astype(np.float32)), b["labels"].astype(str), int(b["n_labels"])
    pos = {g: i for i, g in enumerate(labels)}
    rows = []
    for enc in a.encode_dirs:
        enc = Path(enc)
        o = pd.read_parquet(enc / "obs.parquet")
        for c in ("drug", "cell_name", "plate", "targets", "moa_broad"):
            o[c] = o[c].astype(str)
        o["dose_uM"] = o.dose_uM.astype(float)
        Z = np.load(enc / "z_raw.npy").astype(np.float32)
        ok = ~np.isnan(Z).any(1)
        ctrl = o.is_control.astype(bool).to_numpy() & ok
        o["pl"] = o.plate + "|" + o.cell_name
        refs = {
            "pl": pd.DataFrame(Z[ctrl], index=o.pl[ctrl].to_numpy()).groupby(level=0).mean(),
            "line": pd.DataFrame(Z[ctrl], index=o.cell_name[ctrl].to_numpy()).groupby(level=0).mean(),
            "plate": pd.DataFrame(Z[ctrl], index=o.plate[ctrl].to_numpy()).groupby(level=0).mean(),
        }
        keycol = {"pl": "pl", "line": "cell_name", "plate": "plate"}
        drug = (~o.is_control.astype(bool).to_numpy()) & ok & o.moa_broad.eq("inhibitor/antagonist").to_numpy()
        e = o[drug].reset_index(drop=True)
        Ze = Z[drug]
        for ref, tab in refs.items():
            if a.refs and ref not in a.refs:
                continue
            kk = e[keycol[ref]].to_numpy()
            have = np.isin(kk, tab.index)
            D = Ze[have] - tab.loc[kk[have]].to_numpy(np.float32)
            ee = e[have].reset_index(drop=True)
            for bank in ("trained", "full"):
                if a.banks and bank not in a.banks:
                    continue
                Pb = P[:n_lab] if bank == "trained" else P
                lim = n_lab if bank == "trained" else len(labels)
                pen = None if pen_all is None else pen_all[:lim]
                sets = [np.array([pos[g] for g in split(t) if g in pos and pos[g] < lim], int) for t in ee.targets]
                use = np.array([len(s) > 0 for s in sets])
                for level, cols in (("drug", ["drug"]), ("drug_dose", ["drug", "dose_uM"])):
                    g = ee[use].groupby(cols).indices
                    idx = np.flatnonzero(use)
                    R = np.stack([D[idx[ix]].mean(0) for ix in g.values()])
                    S = [sets[idx[ix[0]]] for ix in g.values()]
                    res = collect([rank_matrix(R, Pb, S, pen)], C=lim, set_sizes=[len(x) for x in S])
                    rows.append(dict(arm=enc.name, latent_ref=ref, bank=bank, level=level,
                                     hub_correction=pen is not None, **res))
                blocks, sizes = [], []
                for _, ix in ee[use].groupby(["cell_name", "dose_uM"]).indices.items():
                    ii = np.flatnonzero(use)[ix]
                    if len(ii) > 1:
                        sizes += [len(sets[i]) for i in ii]
                    blocks.append(rank_matrix(D[ii], Pb, [sets[i] for i in ii], pen))
                rows.append(dict(arm=enc.name, latent_ref=ref, bank=bank, level="entry",
                                 hub_correction=pen is not None, **collect(blocks, C=lim, set_sizes=sizes)))
                r = rows[-3]
                log.info("%s | %s | %s | drug r@10 %.3f (null %.4f) r@50 %.3f p %.1e", enc.name, ref, bank,
                         r["r@10"], r["null@10"], r["r@50"], r["p_ttest"])
    t = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    log.info("wrote %s (%d rows)", a.out, len(t))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encode-dirs", nargs="+", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-pen", default=None, help="npz with pen (2, n_bank): rank by cos - pen[0]")
    ap.add_argument("--refs", nargs="*", default=None, help="latent refs to score (default all)")
    ap.add_argument("--banks", nargs="*", default=None, help="trained / full (default both)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(a)
