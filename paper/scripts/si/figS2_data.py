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

"""Supplementary Fig. S2 plotting data (the prior pull: gain by training support, and applied after vs during
training; related to Fig. 3). Fold-4 models (K562 held out), the 10,061 held-out K562 queries of trained
perturbations, top-10 hits of each pulled model minus the no-pull model (BASE). One CSV per panel in --out-dir
(paper/figures/si/figS2/data):

    figS2a_pull_set_size.csv        offline pull-set-size x pull-strength sweep on the no-pull model (net top-10 flips vs
                                    no pull), copied from the effgate_decomp record
    figS2c_deciles.csv              per arm x support decile: queries, share of queries whose prototype is gated, net
                                    top-10 hits (arm - BASE) with 95% CI, and the same split into gated / other queries.
                                    Support = the trained model's own gate evidence (training cells x training datasets,
                                    ranked; gates.npz from decomp.py), so the gate is exactly the lowest-ranked queries;
                                    deciles over the QUERIES (ties broken in query order); bootstrap over queries
    figS2d_halves.csv               per arm x half (lower / upper five deciles): net top-10 hits, 95% CI; base_top10 (here
                                    and in A) = the no-pull model's hits in the group, the denominator for a % change
    figS2e_posthoc_vs_trained.csv   self (gated prototypes' own queries) / collateral / total flips vs BASE: the
                                    post-hoc static blend of the gated prototypes per pull strength alpha, and the
                                    trained gated-pull model. Read from the pull decomposition record
                                    (analysis/readout/effgate_decomp/decomp.py summary.txt, parsed by
                                    fig3_plot.read_decomp; nothing is recomputed)
    figS2_meta.json                 arms, queries, bootstrap draws, seed; gated prototypes / queries and the gate's edge
                                    on the decile axis; the trained model's implied alpha (median, IQR)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig3_plot as P3  # noqa: E402


def ranks(ranks_dir: Path, arm: str) -> pd.Series:
    t = pd.read_csv(ranks_dir / arm / f'{arm}_ranks.csv')
    return pd.Series(t['rank'].to_numpy(), index=t.dataset + '|' + t.perturbation)


def ci(d: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    bs = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def posthoc_vs_trained(summary: str, trained: str):
    """C: the post-hoc static blend per alpha and the trained gated-pull model, from the decomposition record."""
    tab, alpha = P3.read_decomp(summary)
    sim = tab[tab.kind == 'simulated'].sort_values('alpha').assign(kind='post-hoc', arm='static blend')
    tr = tab[(tab.kind == 'realised') & (tab.arm == trained)].assign(kind='trained', alpha=alpha[0])
    assert len(tr) == 1, f'{trained} not in {summary}'
    gate = re.search(r'GATE SETS at pull_q ([\d.]+): nqxnds (\d+)', Path(summary).read_text())
    meta = dict(summary=summary, trained=trained, implied_alpha_median=alpha[0], implied_alpha_iqr=list(alpha[1:]),
                pull_q=float(gate.group(1)), n_gated=int(gate.group(2)))
    return pd.concat([sim, tr], ignore_index=True), meta


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    # ---- A: pulled-set size sweep (fold-4 OOD, offline; formerly Supplementary Fig. S2A) ----
    sweep = pd.read_csv(a.pull_sweep, sep='\t')
    sweep.to_csv(out / 'figS2a_pull_set_size.csv', index=False)
    pull_q = json.loads((Path(a.fig1_data) / 'fig1c_meta.json').read_text())['pull_q']
    print(f'[A] pull sweep: {sweep.pull_q.nunique()} set sizes x {sweep.alpha.nunique()} strengths', flush=True)
    G = np.load(a.gates, allow_pickle=True)                         # the model's own evidence and gate
    labels = G['labels'].astype(str)
    sup = pd.Series(G['q_nqxnds'].astype(float), index=labels)     # rank of training cells x training datasets
    gate = pd.Series(G['in_nqxnds'].astype(bool), index=labels)
    rd = Path(a.ranks_dir)
    base = ranks(rd, a.base)
    pert = pd.Series(base.index.str.split('|').str[1])
    s = pert.map(sup).to_numpy()
    assert not np.isnan(s).any(), 'query perturbation without gate evidence'
    gq = pert.map(gate).to_numpy(bool)
    dec = pd.qcut(pd.Series(s).rank(method='first'), 10, labels=False).to_numpy()
    order = np.argsort(np.argsort(s, kind='stable'), kind='stable')        # query position along the support axis
    assert gq[order < gq.sum()].all(), 'gate is not the lowest-support block of queries'
    per_dec = len(s) / 10
    b10 = (base.to_numpy() < 10).astype(int)                 # the no-pull model's hits: denominators for % change
    rows, halves = [], []
    for arm in a.arms:
        d = (ranks(rd, arm).reindex(base.index).to_numpy() < 10).astype(int) - (base.to_numpy() < 10).astype(int)
        rng = np.random.default_rng(a.seed)            # halves first: the intervals the text quotes
        for name, m in (('lower half', dec < 5), ('upper half', dec >= 5)):
            lo, hi = ci(d[m], a.n_boot, rng)
            halves.append(dict(arm=arm, half=name, n_queries=int(m.sum()), net_top10=int(d[m].sum()),
                               ci_lo=lo, ci_hi=hi, base_top10=int(b10[m].sum())))
        rng = np.random.default_rng(a.seed)
        for q in range(10):
            m = dec == q
            lo, hi = ci(d[m], a.n_boot, rng)
            rows.append(dict(arm=arm, decile=q + 1, n_queries=int(m.sum()), gated_share=float(gq[m].mean()),
                             support_rank_min=float(s[m].min()), support_rank_max=float(s[m].max()),
                             net_top10=int(d[m].sum()), ci_lo=lo, ci_hi=hi,
                             net_top10_gated=int(d[m & gq].sum()), net_top10_other=int(d[m & ~gq].sum()),
                             base_top10=int(b10[m].sum())))
    R, H = pd.DataFrame(rows), pd.DataFrame(halves)
    R.to_csv(out / 'figS2c_deciles.csv', index=False)
    H.to_csv(out / 'figS2d_halves.csv', index=False)
    for arm in a.arms:
        print(f'[A] {arm}: {R[R.arm == arm].net_top10.tolist()}', flush=True)
    print(H.round(1).to_string(index=False), flush=True)
    C, mc = posthoc_vs_trained(a.summary, a.trained)
    assert abs(mc['pull_q'] - pull_q) < 1e-9, (mc['pull_q'], pull_q)   # the record's gate = the shipped pull_q (A)
    C.to_csv(out / 'figS2e_posthoc_vs_trained.csv', index=False)
    print(C[['kind', 'arm', 'alpha', 's', 'c', 't', 'tlo', 'thi']].to_string(index=False), flush=True)
    (out / 'figS2_meta.json').write_text(json.dumps(dict(
        pull_sweep=str(a.pull_sweep), base=a.base, arms=a.arms, n_queries=int(len(base)), n_boot=a.n_boot, seed=a.seed,
        n_gated=int(gate.sum()), n_gated_queries=int(gq.sum()),
        base_top10={'total': int(b10.sum()), 'gated': int(b10[gq].sum()), 'other': int(b10[~gq].sum())},
        gate_edge_decile=float(0.5 + gq.sum() / per_dec), **{k: v for k, v in mc.items() if k != 'n_gated'},
        n_gated_posthoc=mc['n_gated']), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gates', required=True, help='effgate_decomp gates.npz (the model\'s gate evidence and set)')
    ap.add_argument('--ranks-dir', required=True, help='v017 readout ranks/<ARM>/<ARM>_ranks.csv')
    ap.add_argument('--base', default='BASE', help='no-pull reference arm')
    ap.add_argument('--arms', nargs='+', default=['FULLOPEN27', 'EFFGATE3'], help='pulled arms (pull-all, gated)')
    ap.add_argument('--summary', required=True, help='analysis_out/effgate_decomp/summary.txt (panel C)')
    ap.add_argument('--trained', default='EFFGATE3', help='trained gated-pull arm in the record (panel C)')
    ap.add_argument('--pull-sweep', required=True, help='effgate_decomp dose_gate_sweep.tsv (panel A)')
    ap.add_argument('--fig1-data', required=True, help='Fig. 1 data dir (fig1c_meta.json: the shipped pull_q)')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
