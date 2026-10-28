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

"""Fig. 5 Tahoe panels E + F data: target and pathway recovery on RESPONSIVE conditions.

Active entry = |Δz| above the 95th percentile of DMSO-vs-DMSO noise (each DMSO pseudobulk minus its
cell line's DMSO mean). A drug is scored from the mean of its active entries (>= --min-active);
non-responding conditions are not scored at all, so the numbers describe what the model does where
the drug does something.

    E   recall@k over k, on active drugs and on active entries:
          exact    the drug's annotated target (any of them)
          complex  a PARTNER of a target in a physical complex / tight pathway (--libs, sets <=
                   --max-set genes), with the targets themselves REMOVED -- so it is an independent
                   test, not "the target is in its own pathway"
        Null: size- and hubness-matched RANDOM gene sets (--n-perm draws per drug). Prototypes are
        binned by mean cosine over all drugs' responses and each true mate is replaced by a random
        prototype from its own bin, so a set of equally popular genes is the comparison. A
        drug-permutation null is NOT used for the complex statistic: a dozen Tahoe drugs are
        PI3K/mTOR inhibitors and many more are kinase inhibitors, so another drug's complex
        partners are often this drug's as well, and the "null" carries real signal.
    F   worked examples: top-10 prototypes of the drug's active response, raw and hub-filtered,
        each labelled target / pathway mate of a target / other, with the cosine and the hub z.
        Hub filter = leave-one-drug-out column z of cos over the other drugs' active responses: a
        prototype that scores high for EVERY drug is down-weighted, and the drug being scored never
        enters its own statistics.

Writes <out>/fig4e_source.csv, fig4e_curve.csv, fig4f_examples.csv, panels_meta.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from expert.utils.readout import calibrated_readout

log = logging.getLogger("fig4_tahoe_panels")
KGRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def split(v):
    s = str(v).strip()
    return [t.strip() for t in s.replace(",", ";").split(";") if t.strip() and t.strip() != "nan"] \
        if s and s.lower() != "nan" else []


def rank_matrix(D, P, n_cells, readout="deflate", tail="best", noise=None, rounds=3, head=10):
    """Ranks of every prototype for every row, under the readout the figure settled on.

    sci-Plex settled this: deflate at its shipped settings (3 rounds, the first 10 slots held by
    round 1) and the BEST of the loss and gain tails. A drug that activates its target, or whose
    feedback inverts the target's signature, is invisible to the loss tail alone -- on sci-Plex the
    gain tail is what recovers the HSP90 inhibitors, and it adds ~30% relative recall overall.
    """
    n, m = len(D), len(P)
    if readout == "deflate":
        r = calibrated_readout(D, P, n_cells, noise=noise, rounds=rounds, head=head, k=m)
        out = np.full((n, m), m + 1, np.int32)
        for t in (("loss", "gain") if tail == "best" else (tail,)):
            idx = r[t]
            rr = np.full((n, m), m + 1, np.int32)
            rows = np.repeat(np.arange(n), idx.shape[1])
            rr[rows, idx.ravel()] = np.tile(np.arange(1, idx.shape[1] + 1), n)
            out = np.minimum(out, rr)
        return out
    S = unit(D) @ P.T
    out = np.full((n, m), m + 1, np.int32)
    for t in (("loss", "gain") if tail == "best" else (tail,)):
        rr = np.empty(S.shape, np.int32)
        np.put_along_axis(rr, np.argsort(-S if t == "loss" else S, axis=1),
                          np.arange(1, m + 1)[None, :], axis=1)
        out = np.minimum(out, rr)
    return out


def main(a):
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    noise = np.load(a.noise_npy) if a.noise_npy and Path(a.noise_npy).exists() else None
    b = np.load(a.bank, allow_pickle=True)
    nl = int(b["n_labels"])
    # 'full' keeps the zero-shot rows (the f(missing) text fill) in the candidate set, so a drug
    # whose only annotated targets have no trained prototype becomes scorable at all. On the
    # trained block those drugs silently get an EMPTY target list and drop out of the panel.
    nrow = nl if a.bank_rows == "trained" else len(b["labels"])
    L = b["labels"].astype(str)[:nrow]; P = unit(b["P"].astype(np.float32))[:nrow]
    log.info("bank rows: %s -> %d candidates (%d trained)", a.bank_rows, nrow, nl)
    pos = {g: i for i, g in enumerate(L)}

    # ---- active responses ----------------------------------------------------------------------
    o = pd.read_parquet(Path(a.encode_dir) / "obs.parquet")
    Z = np.load(Path(a.encode_dir) / "z_raw.npy").astype(np.float32)
    for c in ("drug", "cell_name", "plate", "targets", "moa_broad", "moa_fine"):
        o[c] = o[c].astype(str)
    ok = ~np.isnan(Z).any(1)
    ctrl = o.is_control.astype(bool).to_numpy() & ok
    lm = pd.DataFrame(Z[ctrl], index=o.cell_name[ctrl].to_numpy()).groupby(level=0).mean()
    noise = np.linalg.norm(Z[ctrl] - lm.loc[o.cell_name[ctrl]].to_numpy(), axis=1)
    thr = float(np.quantile(noise, 0.95))
    # The MoA restriction is a leftover from a LOSS-ONLY readout: only an inhibitor can look like
    # a knockdown. The shipped readout scores both tails, so an agonist is findable on the gain
    # side and there is no reason to drop it. 'all' keeps every MoA class.
    d = (~o.is_control.astype(bool).to_numpy()) & ok
    if a.moa_broad != "all":
        keep = {x.strip() for x in a.moa_broad.split(",") if x.strip()}
        d &= o.moa_broad.isin(keep).to_numpy()
    log.info("MoA classes scored: %s -> %d drugs", a.moa_broad,
             o.drug[d].nunique() if d.any() else 0)
    e = o[d].reset_index(drop=True)
    D = Z[d] - lm.loc[e.cell_name].to_numpy(np.float32)
    act = np.linalg.norm(D, axis=1) > thr
    tgt = {x: [pos[g] for g in split(t) if g in pos]
           for x, t in e.groupby("drug").targets.first().items()}

    # ---- gene sets -> pathway mates per drug -----------------------------------------------------
    libs = tuple(x.strip() for x in a.libs.split(",") if x.strip())
    sets = json.load(open(a.gene_sets_json))
    sets = {k: [g for g in v if g in pos] for k, v in sets.items() if k.startswith(libs)}
    sets = {k: v for k, v in sets.items() if 3 <= len(v) <= a.max_set}
    g2s = {}
    for k, v in sets.items():
        for g in v:
            g2s.setdefault(g, set()).add(k)
    def mates(ti):
        """Complex / pathway PARTNERS of the targets, smallest sets first, capped at --mate-cap,
        with the targets themselves excluded (independent of the exact-target statistic)."""
        # sort and iterate deterministically: ties on set size, and the member order inside a set,
        # otherwise follow Python's per-process string hashing and the partner lists change run to run
        ks = sorted(set().union(*[g2s.get(L[i], set()) for i in ti]) if ti else set(),
                    key=lambda k: (len(sets[k]), k))
        m = []
        for k in ks:
            for g in sorted(sets[k]):
                if pos[g] not in m and pos[g] not in set(ti):
                    m.append(pos[g])
            if len(m) >= a.mate_cap:
                break
        return np.array(sorted(m[:a.mate_cap]), int)

    drugs, R, n_act = [], [], []
    for x, ix in e.groupby("drug").indices.items():
        ia = ix[act[ix]]
        if tgt.get(x) and len(ia) >= a.min_active:
            drugs.append(x); R.append(D[ia].mean(0)); n_act.append(len(ia))
    R = unit(np.stack(R)); S = R @ P.T
    rk = rank_matrix(R, P, np.array([max(n, 1) for n in n_act]), a.readout, a.tail, noise)
    mate_idx = [mates(tgt[x]) for x in drugs]
    log.info("active |Δz| > %.2f (%d/%d entries); %d drugs with >= %d active entries; sets %s n=%d, "
             "median complex partners/drug %d; drugs with >=1 partner %d", thr, int(act.sum()), len(act),
             len(drugs), a.min_active, a.libs, len(sets), int(np.median([len(m) for m in mate_idx])),
             int(sum(len(m) > 0 for m in mate_idx)))

    # ---- E: recall curves with the shared permutation null ---------------------------------------
    # hubness bins: prototypes ranked by mean cosine over all active drug responses
    pop = S.mean(0)
    nb = a.n_bins
    bin_of = np.clip((pop.argsort().argsort() * nb) // len(pop), 0, nb - 1)
    bins = [np.flatnonzero(bin_of == j) for j in range(nb)]

    def matched_draw(idx):
        """A random gene set of the same size, each gene drawn from its own popularity bin."""
        return np.array([rng.choice(bins[bin_of[i]]) for i in idx], int)

    def curve(level, rkm, tsets, msets):
        rows = []
        n = len(rkm)
        # per drug: n_perm matched random sets for the complex statistic (drawn once, reused per k)
        has = [i for i in range(n) if len(msets[i])]          # drugs with >= 1 complex partner
        mn = np.array([[rkm[i, matched_draw(msets[i])].min() for _ in range(a.n_perm)] for i in has], float)
        for k in KGRID:
            ex = np.mean([rkm[i, tsets[i]].min() <= k for i in range(n)])
            pw = float(np.mean([rkm[i, msets[i]].min() <= k for i in has])) if has else np.nan
            exn = []
            for _ in range(a.n_perm):
                p = rng.permutation(n)
                exn.append(np.mean([rkm[i, tsets[p[i]]].min() <= k for i in range(n)]))
            pwn = (mn <= k).mean(0)                     # one null value per matched draw
            rows.append(dict(level=level, k=k, n=n, n_complex=len(has), exact=ex, exact_null=float(np.mean(exn)),
                             exact_null_lo=float(np.quantile(exn, .025)), exact_null_hi=float(np.quantile(exn, .975)),
                             exact_p=float((np.sum(np.array(exn) >= ex) + 1) / (a.n_perm + 1)),
                             pathway=pw, pathway_fold=float(pw / max(np.mean(pwn), 1e-9)),
                             exact_fold=float(ex / max(np.mean(exn), 1e-9)),
                             pathway_null=float(np.mean(pwn)),
                             pathway_null_lo=float(np.quantile(pwn, .025)), pathway_null_hi=float(np.quantile(pwn, .975)),
                             pathway_p=float((np.sum(np.asarray(pwn) >= pw) + 1) / (len(pwn) + 1))))
        return rows
    rows = curve("drug", rk, [np.array(tgt[x]) for x in drugs], mate_idx)
    # active entries: same statistic per (drug, line, dose)
    ea = e[act].reset_index(drop=True)
    keep = np.array([bool(tgt.get(x)) for x in ea.drug])
    Da = D[act][keep]
    Sa = unit(Da) @ P.T
    rka = rank_matrix(Da, P, ea.n_cells.to_numpy()[keep] if "n_cells" in ea else
                      np.full(len(Da), 10), a.readout, a.tail, noise)
    dn = ea.drug[keep].to_numpy()
    mate_of = dict(zip(drugs, mate_idx))
    rows += curve("entry", rka, [np.array(tgt[x]) for x in dn],
                  [mate_of.get(x, np.array(tgt[x])) for x in dn])
    ent_idx = pd.Series(np.arange(len(dn))).groupby(dn).apply(lambda s_: s_.to_numpy()).to_dict()
    ent_idx = {x: ent_idx.get(x, np.array([], int)) for x in drugs}
    pd.DataFrame(rows).to_csv(out / "fig4e_curve.csv", index=False)

    # per-drug table behind E: consensus ranks AND the fraction of a drug's own active conditions
    # that put the target / a complex partner in the top --top-k (the reproducibility read)
    gda = e[act_keep].groupby("drug").indices if False else None
    ea_all = e[act].reset_index(drop=True) if False else None
    rows_e = []
    for i, x in enumerate(drugs):
        ia = ent_idx[x]
        fe = float(np.mean([rka[j, tgt[x]].min() <= a.top_k for j in ia])) if len(ia) else np.nan
        fm = (float(np.mean([rka[j, mate_idx[i]].min() <= a.top_k for j in ia]))
              if len(ia) and len(mate_idx[i]) else np.nan)
        # per-drug null: the same statistic for size- and popularity-matched random gene sets, so
        # panel E can show observed and null as PAIRED distributions over drugs
        fen = (float(np.mean([[rka[j, matched_draw(tgt[x])].min() <= a.top_k for j in ia]
                              for _ in range(a.n_null_draw)])) if len(ia) else np.nan)
        fmn = (float(np.mean([[rka[j, matched_draw(mate_idx[i])].min() <= a.top_k for j in ia]
                              for _ in range(a.n_null_draw)]))
               if len(ia) and len(mate_idx[i]) else np.nan)
        rows_e.append(dict(drug=x, targets=";".join(L[tgt[x]]), moa=e[e.drug == x].moa_fine.iloc[0],
                           n_active=n_act[i], n_entries=int((e.drug == x).sum()),
                           rank_exact=int(rk[i, tgt[x]].min()),
                           rank_pathway=int(rk[i, mate_idx[i]].min()) if len(mate_idx[i]) else -1,
                           n_mates=len(mate_idx[i]), frac_exact_topk=fe, frac_complex_topk=fm,
                           frac_exact_null_topk=fen, frac_complex_null_topk=fmn))
    pd.DataFrame(rows_e).to_csv(out / "fig4e_source.csv", index=False)

    # ---- F: worked examples, raw vs hub-filtered -------------------------------------------------
    mu = S.sum(0); sq = (S ** 2).sum(0); n = len(S)
    mu_lo = (mu[None] - S) / (n - 1)
    sd_lo = np.sqrt(np.maximum((sq[None] - S ** 2) / (n - 1) - mu_lo ** 2, 1e-12))
    Sz = (S - mu_lo) / sd_lo
    want = [x.strip() for x in a.examples.split(",") if x.strip()]
    rows = []
    for x in want:
        if x not in drugs:
            log.warning("example %r not among active drugs", x); continue
        i = drugs.index(x)
        ti, mi = set(tgt[x]), set(mate_idx[i].tolist())
        for which, sc in (("raw", S[i]), ("hub", Sz[i])):
            top = np.argsort(-sc)[:a.top_k]
            for r_, g in enumerate(top, 1):
                rows.append(dict(drug=x, targets=";".join(L[tgt[x]]), ranking=which, rank=r_, gene=L[g],
                                 cos=float(S[i, g]), hub_z=float(Sz[i, g]),
                                 kind="target" if g in ti else ("mate" if g in mi else "other"),
                                 n_active=n_act[i]))
    pd.DataFrame(rows).to_csv(out / "fig4f_examples.csv", index=False)

    json.dump(dict(encode_dir=a.encode_dir, active_threshold=thr, frac_entries_active=float(act.mean()),
                   n_drugs=len(drugs), n_active_entries=int(keep.sum()), n_gene_sets=len(sets),
                   max_set=a.max_set, min_active=a.min_active, examples=want),
              open(out / "panels_meta.json", "w"), indent=1)
    c = pd.DataFrame(rows[:0]) if not rows else None
    cur = pd.read_csv(out / "fig4e_curve.csv")
    for lev in ("drug", "entry"):
        q = cur[(cur.level == lev) & cur.k.isin([10, 50, 100])]
        log.info("%s: %s", lev, "; ".join(
            f"k={int(r.k)} exact {100*r.exact:.1f}% (null {100*r.exact_null:.1f}, p {r.exact_p:.3f}) "
            f"complex {100*r.pathway:.1f}% (null {100*r.pathway_null:.1f}, p {r.pathway_p:.3f})"
            for r in q.itertuples()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encode-dir", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--gene-sets-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-active", type=int, default=3)
    ap.add_argument("--max-set", type=int, default=30, help="only specific sets: complexes / tight pathways")
    ap.add_argument("--libs", default="CORUM,Reactome_Pathways_2024",
                    help="gene-set libraries to use (prefix match); KEGG sets are mostly > 30 genes")
    ap.add_argument("--n-bins", type=int, default=20, help="popularity bins for the matched-random null")
    ap.add_argument("--mate-cap", type=int, default=25, help="max complex partners per drug (smallest sets first)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-null-draw", type=int, default=20,
                    help="matched random gene sets per drug for the paired per-drug null (panel E)")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--examples", default="Bortezomib,Ixazomib citrate,Temsirolimus,Selinexor,"
                                          "Pitavastatin (Calcium),GSK1059615")
    ap.add_argument("--moa-broad", default="inhibitor/antagonist",
                    help="comma list of moa_broad classes, or 'all' (the signed "
                         "readout can find agonists on the gain tail)")
    ap.add_argument("--bank-rows", default="trained", choices=("trained", "full"),
                    help="full = include the zero-shot prototype rows as candidates")
    ap.add_argument("--readout", default="deflate", choices=("deflate", "cosine"))
    ap.add_argument("--tail", default="best", choices=("loss", "gain", "best"))
    ap.add_argument("--noise-npy", default="", help="readout calibration, if the checkpoint has one")
    ap.add_argument("--seed", type=int, default=0)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    main(ap.parse_args())
