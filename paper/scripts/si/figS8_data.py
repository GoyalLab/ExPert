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

"""Supplementary Fig. S8 plotting data (drug-target recovery with the zero-shot prototypes added to the candidate bank,
as fold over random; related to Fig. 4). Raw recall cannot be read across banks: the matched panel ranks 3,802
candidates, the trained bank 8,964 and the padded bank 19,895, and the padded bank also scores MORE conditions (a drug
whose targets have no trained prototype becomes scorable at all). Each arm is therefore divided by its own chance rate.
One CSV per panel in --out-dir (paper/figures/si/figS8/data):

    figS8a_sciplex3_fold.csv   per fill x bank x k: candidates, scored conditions and drugs, top-k recall, chance, fold
                               over random with a 95% bootstrap interval over DRUGS (doses of one drug are not
                               independent). Chance is exact per condition: with m of the drug's annotated targets
                               in a pool of N, a uniform ranking puts one in the top k with probability
                               1 - C(N-m, k)/C(N, k); conditions are averaged, so the baseline follows each arm's own
                               condition set. A condition with no target in the pool is unscorable and dropped.
    figS8b_tahoe_top10.csv     per arm x statistic (annotated target, complex partner): drugs scored, drugs recovered
                               (any responsive condition with the gene in the top k), mean fraction of each drug's
                               responsive conditions with it in the top k, the same for size- and popularity-matched
                               random gene sets (Fig. 4F's null), their ratio (fold) and a bootstrap interval over
                               drugs. Read from fig4_tahoe_panels.py's fig4e_source.csv; nothing is re-ranked
    figS8aux_tahoe_curves.csv  the same arms as drug-level recall curves (consensus response per drug) over k with the
                               permutation null (fig4e_curve.csv, level = drug)
    figS8_meta.json            arms, inputs, k, bootstrap draws, seed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KS = [1, 5, 10, 20, 50, 100]
BANKS = {'matched': 'matched panel', 'trained': 'trained', 'full': 'zero-shot padded'}


def chance(k, N, m):
    """P(at least one of m targets in the top k of N), vectorised over m."""
    m = np.asarray(m, float)
    lo = np.ones_like(m)
    for i in range(int(k)):                      # prod_{i<k} (N-m-i)/(N-i)
        lo *= np.clip(N - m - i, 0, None) / (N - i)
    return 1.0 - lo


def boot_ratio(num: np.ndarray, den: np.ndarray, groups: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """95% interval of sum(num)/sum(den), resampling whole groups (drugs)"""
    u, inv = np.unique(groups, return_inverse=True)
    gn, gd = np.bincount(inv, num, len(u)), np.bincount(inv, den, len(u))
    bs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(u), len(u))
        bs.append(gn[s].sum() / gd[s].sum() if gd[s].sum() > 0 else np.nan)
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def sciplex(a, rng) -> pd.DataFrame:
    tg = pd.read_csv(a.sciplex_targets, sep='\t')
    genes = {r.drug: {g for g in str(r.genes).split(';') if g and g != 'nan'} for r in tg.itertuples()}
    b = np.load(a.sciplex_bank, allow_pickle=True)
    L, nl = b['labels'].astype(str), int(b['n_labels'])
    pools = {'matched': set(np.load(a.sciplex_matched, allow_pickle=True)['genes'].astype(str)),
             'trained': set(L[:nl]), 'full': set(L)}
    N = {'matched': len(pools['matched']), 'trained': nl, 'full': len(L)}
    rows = []
    for spec in a.sciplex_ranks:
        fill, path = spec.split('=', 1)
        d = pd.read_csv(path)
        d = d[d.arm.str.endswith(a.sciplex_arm)]
        for bank, g in d.groupby('bank'):
            m = np.array([len(genes.get(dr, set()) & pools[bank]) for dr in g.drug])
            keep = m > 0
            r, m, dr = g['rank'].to_numpy(float)[keep], m[keep], g.drug.to_numpy()[keep]
            for k in KS:
                hit, exp = (r <= k).astype(float), chance(k, N[bank], m)
                lo, hi = boot_ratio(hit, exp, dr, a.n_boot, rng)
                rows.append(dict(fill=fill, bank=bank, bank_label=BANKS[bank], n_cand=N[bank],
                                 n_cond=int(keep.sum()), n_drugs=int(len(set(dr))), k=k, recall=float(hit.mean()),
                                 random=float(exp.mean()), fold=float(hit.mean() / exp.mean()), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def tahoe(a, rng) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows, curves, top_k = [], [], set()
    for spec in a.tahoe_panels:
        arm, fill, path = spec.split('=', 2)
        d = Path(path)
        top_k.add(int(json.loads((d / 'panels_meta.json').read_text()).get('top_k', a.tahoe_top_k)))
        t = pd.read_csv(d / 'fig4e_source.csv')
        for stat, obs, null in (('annotated target', 'frac_exact_topk', 'frac_exact_null_topk'),
                                ('complex partner', 'frac_complex_topk', 'frac_complex_null_topk')):
            s = t.dropna(subset=[obs, null])
            lo, hi = boot_ratio(s[obs].to_numpy(), s[null].to_numpy(), s.drug.to_numpy(), a.n_boot, rng)
            rows.append(dict(arm=arm, fill=fill, statistic=stat, n_drugs=len(s), n_recovered=int((s[obs] > 0).sum()),
                             frac=float(s[obs].mean()), null=float(s[null].mean()),
                             fold=float(s[obs].mean() / s[null].mean()), ci_lo=lo, ci_hi=hi))
        c = pd.read_csv(d / 'fig4e_curve.csv')
        curves.append(c[c.level == 'drug'].assign(arm=arm, fill=fill))
    assert len(top_k) == 1, f'Tahoe arms scored at different top-k: {top_k}'
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True), top_k.pop()


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    A = sciplex(a, np.random.default_rng(a.seed))
    A.to_csv(out / 'figS8a_sciplex3_fold.csv', index=False)
    print(A.pivot_table(index=['fill', 'bank', 'n_cand', 'n_cond'], columns='k', values='fold').round(2).to_string(),
          flush=True)
    B, C, top_k = tahoe(a, np.random.default_rng(a.seed))
    B.to_csv(out / 'figS8b_tahoe_top10.csv', index=False)
    C.to_csv(out / 'figS8aux_tahoe_curves.csv', index=False)
    print(B.round(3).to_string(index=False), flush=True)
    (out / 'figS8_meta.json').write_text(json.dumps(dict(
        sciplex_ranks=a.sciplex_ranks, sciplex_arm=a.sciplex_arm, ks=KS, tahoe_panels=a.tahoe_panels,
        tahoe_top_k=top_k, n_boot=a.n_boot, seed=a.seed), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sciplex-ranks', nargs='+', required=True, help='fill=ranks_arms.csv (cos4=..., cos=...)')
    ap.add_argument('--sciplex-targets', required=True, help='sciplex3_drug_targets.tsv')
    ap.add_argument('--sciplex-bank', required=True, help='bank.npz (labels + n_labels)')
    ap.add_argument('--sciplex-matched', required=True, help='npz with the 3,802-gene matched candidate list')
    ap.add_argument('--sciplex-arm', default='deflate_best', help='readout suffix of the shipped arm')
    ap.add_argument('--tahoe-panels', nargs='+', required=True,
                    help='arm=fill=panels dir (fig4_tahoe_panels.py output: fig4e_source.csv, fig4e_curve.csv)')
    ap.add_argument('--tahoe-top-k', type=int, default=10, help='used only if panels_meta.json lacks top_k')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
