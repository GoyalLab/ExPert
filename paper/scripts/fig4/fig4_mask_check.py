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

"""Is masking the metallothionein / MT genes worth it on sci-Plex?

Tahoe has run MT-masked from the start (every mtmask arm beat every unmasked one), sci-Plex never
had. This pairs the two sci-Plex runs CONDITION BY CONDITION -- same pseudobulk, same readout, one
with those genes zeroed -- so the comparison is not two independent recall numbers that happen to
differ. Reports, per bank: the mean log10 rank improvement with a paired sign-flip permutation,
and McNemar on the top-k hit (the discordant conditions are the only ones that carry information
about a change in recall).

Writes <out>/mask_check.csv.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("fig4_mask_check")


def main(a):
    from scipy.stats import binomtest
    key = ["bank", "readout", "tail", "condition"]
    un = pd.read_csv(a.unmasked)
    mt = pd.read_csv(a.masked)
    for nm, t in (("unmasked", un), ("masked", mt)):
        d = int(t.duplicated(key).sum())
        if d:
            log.warning("%s: %d duplicate rows on %s -- the merge would fan out", nm, d, key)
    # `condition` is drug@dose@line and does NOT carry the plate, so a (drug, dose, line) run on
    # two plates appears twice and the rows cannot be paired 1:1. Collapse to the condition first,
    # keeping its best rank across plates -- the same aggregation the panels read.
    agg = lambda t: t.groupby(key, as_index=False)["rank"].min()
    un, mt = agg(un), agg(mt)
    m = un.merge(mt, on=key, suffixes=("_un", "_mt"), validate="one_to_one")
    log.info("paired %d conditions", len(m))
    rj = np.random.default_rng(a.seed)
    rows = []
    for bank in sorted(m.bank.unique()):
        for tail in sorted(m["tail"].unique()):      # NB: .tail is a DataFrame method
            q = m[(m.bank == bank) & (m.readout == a.readout) & (m["tail"] == tail)]
            q = q.dropna(subset=["rank_un", "rank_mt"])
            q = q[(q.rank_un > 0) & (q.rank_mt > 0)]
            if len(q) < 10:
                continue
            d = np.log10(q.rank_un.to_numpy(float)) - np.log10(q.rank_mt.to_numpy(float))
            # permute in chunks: 20k x n at once is what OOM'd this on a login node
            obs, ge = d.mean(), 0
            for _ in range(a.n_perm // a.chunk):
                f = (rj.choice([-1.0, 1.0], (a.chunk, len(d))) * d).mean(1)
                ge += int(np.sum(f >= obs))
            p = (ge + 1) / (a.n_perm + 1)
            r = dict(bank=bank, tail=tail, n=len(q), mean_log10_gain=float(obs), p_signflip=p,
                     better=int((d > 0).sum()), worse=int((d < 0).sum()), tied=int((d == 0).sum()))
            for k in (1, 10, 50):
                hu, hm = q.rank_un <= k, q.rank_mt <= k
                b01, b10 = int((~hu & hm).sum()), int((hu & ~hm).sum())
                r |= {f"recall@{k}_un": float(hu.mean()), f"recall@{k}_mt": float(hm.mean()),
                      f"mask_only@{k}": b01, f"unmasked_only@{k}": b10,
                      f"p_mcnemar@{k}": float(binomtest(b01, b01 + b10, 0.5).pvalue)
                      if b01 + b10 else np.nan}
            rows.append(r)
            log.info("%-8s %-5s n=%5d  log10 gain %+.4f (p=%.4g)  r@10 %.4f -> %.4f  "
                     "discordant %d/%d (p=%.3g)", bank, tail, len(q), obs, p,
                     r["recall@10_un"], r["recall@10_mt"], r["mask_only@10"],
                     r["unmasked_only@10"], r["p_mcnemar@10"])
    # drug level = panel B's unit: a drug counts as found if ANY of its doses finds it, so an
    # improvement that moves a different dose of an already-found drug does not show up here.
    dkey = ["bank", "readout", "tail", "drug"]
    for src, suf in ((un, "_un"), (mt, "_mt")):
        src["drug"] = src.condition.astype(str).str.split("@").str[0]
    du = un.groupby(dkey, as_index=False)["rank"].min()
    dm = mt.groupby(dkey, as_index=False)["rank"].min()
    dd = du.merge(dm, on=dkey, suffixes=("_un", "_mt"), validate="one_to_one")
    for bank in sorted(dd.bank.unique()):
        q = dd[(dd.bank == bank) & (dd.readout == a.readout) & (dd["tail"] == "best")]
        q = q[(q.rank_un > 0) & (q.rank_mt > 0)]
        if not len(q):
            continue
        for k in (10, 50):
            hu, hm = q.rank_un <= k, q.rank_mt <= k
            log.info("DRUG level %-8s recall@%-2d %.4f -> %.4f  (n=%d, mask-only %d, "
                     "unmasked-only %d)", bank, k, hu.mean(), hm.mean(), len(q),
                     int((~hu & hm).sum()), int((hu & ~hm).sum()))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unmasked", required=True, help="ranks_arms.csv from the unmasked run")
    ap.add_argument("--masked", required=True, help="ranks_arms.csv from the MT-masked run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout", default="deflate")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
