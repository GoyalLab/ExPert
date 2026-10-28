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

"""Supplementary Fig. S4 plotting data (Fingerprint's cell assignment, related to Fig. 2), from the benchmark's
cached per-cell scores (score_selective.py; nothing is rescored). Reuses fig2_data.py's loaders. One table per
panel in --out-dir (paper/figures/si/figS4/data); figS4_plot.py draws from those alone.

    figS4_palette.csv                 method -> display name, colour, draw order (as Fig. 2)
    figS4b_coverage.csv               per method x regime x coverage: efficiency-weighted top-1 accuracy and
                                      recall@10 on each fold-view's most confident cells; Fingerprint (ensemble)'s
                                      no-calls rank last, and its call rate is recorded
    figS4c_called_vs_all.csv          per method x regime: per-perturbation F1 / recall@10 / MRR on all cells and
                                      on the cells Fingerprint (ensemble) called (mean over matched views)
    figS4c_called_vs_all_views.csv    ... per (method, fold, view, subset)
    figS4a_abstention.csv             Fingerprint (ensemble): share of cells with no call, per dataset x regime
                                      (cell-weighted and efficiency-weighted)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig2_common as F  # noqa: E402
import fig2_data as D2  # noqa: E402

FP = 'fingerprint_perdataset'
NO_CONF = {'baseline_random', 'baseline_prior'}      # cell-independent scores: no confidence to rank by
COVERAGE = np.round(np.arange(0.05, 1.0001, 0.05), 2)


# ---- A: performance over coverage ------------------------------------------------------------------
def panel_a(sel: pd.DataFrame, out: Path):
    """Within each (fold, split) a method keeps its most confident cells."""
    rows = []
    for (m, reg), g in sel[~sel.method.isin(NO_CONF)].groupby(['method', 'regime']):
        acc = {q: [0.0, 0.0, 0.0] for q in COVERAGE}           # sum w*top1, sum w*hit10, sum w
        for _, f in g.groupby(['fold', 'split']):
            conf = np.where(f.abstain.to_numpy(), -np.inf, f.conf.to_numpy(np.float64))
            order = np.argsort(-conf, kind='stable')
            w = f.w.to_numpy()[order]
            top1 = (f['rank'].to_numpy()[order] < 1) * w
            hit10 = (f['rank'].to_numpy()[order] < 10) * w
            cw, c1, c10 = np.cumsum(w), np.cumsum(top1), np.cumsum(hit10)
            for q in COVERAGE:
                k = max(int(np.ceil(q * len(f))), 1) - 1
                acc[q][0] += c1[k]; acc[q][1] += c10[k]; acc[q][2] += cw[k]
        call_rate = float((~g.abstain).mean())
        for q, (a1, a10, ww) in acc.items():
            rows.append(dict(method=m, regime=reg, coverage=q, accuracy=a1 / ww, recall10=a10 / ww,
                             call_rate=call_rate))
    pd.DataFrame(rows).to_csv(out / 'figS4b_coverage.csv', index=False)


# ---- B: every method on all cells vs on the cells Fingerprint (ensemble) called ---------------------
def panel_b(sel: pd.DataFrame, out: Path):
    fp = sel[sel.method == FP]
    called = fp.loc[~fp.abstain, ['fold', 'split', 'row']].assign(fp_called=True)
    s = sel.merge(called, on=['fold', 'split', 'row'], how='left')
    s['fp_called'] = s['fp_called'].fillna(False).astype(bool)
    rows = []
    for subset, sub in (('all cells', s), ('fingerprint-called cells', s[s.fp_called])):
        for (m, fold, split, ds), v in sub.groupby(['method', 'fold', 'split', 'dataset']):
            r = D2.perpert(v)
            if r:
                rows.append(dict(method=m, fold=fold, split=f'{split}:{ds}', subset=subset,
                                 regime=F.regime(pd.Series([split])).iloc[0], n=len(v), **r))
    mt = pd.DataFrame(rows)
    mt = pd.concat([F.matched(g) for _, g in mt.groupby('subset')], ignore_index=True)
    mt.to_csv(out / 'figS4c_called_vs_all_views.csv', index=False)
    mt.groupby(['subset', 'regime', 'method'])[['F1', 'recall@10', 'MRR']].mean().reset_index() \
      .to_csv(out / 'figS4c_called_vs_all.csv', index=False)
    print(f'  [B] {mt.split.nunique()} views')


# ---- C: Fingerprint (ensemble) no-call rate per dataset ---------------------------------------------
def panel_c(sel: pd.DataFrame, out: Path):
    fp = sel[sel.method == FP]
    ab = fp.groupby(['regime', 'dataset']).apply(lambda x: pd.Series(dict(
        n=len(x), abstain_cells=x.abstain.mean(),
        abstain_eff=np.average(x.abstain, weights=x.w) if x.w.sum() > 0 else np.nan))).reset_index()
    ab.to_csv(out / 'figS4a_abstention.csv', index=False)
    print(f'  [C] no-call rate: ' + ', '.join(f'{r} {fp[fp.regime == r].abstain.mean():.3f}' for r in ('id', 'ood')))


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    D2.panel_palette(Path(a.ctx7_dir), out, name='figS4_palette.csv')
    sel = D2.load_selective(Path(a.selective_dir))
    panel_a(sel, out)
    panel_b(sel, out)
    panel_c(sel, out)
    (out / 'figS4_meta.json').write_text(json.dumps(dict(vars(a), n_cells=int(len(sel))), indent=2, default=str))
    print('wrote', sorted(p.name for p in out.glob('figS4*')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctx7-dir', required=True, help='benchmark collect output, 1,157-class run (palette)')
    ap.add_argument('--selective-dir', required=True, help='score_selective.py per-cell cache (ctx7)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
