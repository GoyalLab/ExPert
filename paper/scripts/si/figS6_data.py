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

"""Supplementary Fig. S6 plotting data (zero-shot prototypes: geometry, how the fill is built, and their cost to trained
perturbations; related to Fig. 3). One CSV per panel in --out-dir (paper/figures/si/figS6/data):

    figS6a_angle.csv                per row: angle (degrees) to the mean trained-prototype direction, for trained
                                    prototypes, zero-shot fills (cos^4 fill) and trained genes re-placed by the same fill
                                    map as if unseen (Fig. 3's radial table, fig3_data.py)
    figS6b_graph_k.csv (+ _summary) graph fill with cos^4 neighbour weights at k = 25 / 50 / 100, per fold model (12):
                                    relative change vs the rebuilt cos fill (k = 25) in zero-shot top-10 (x chance, ID and
                                    OOD) and in trained-query top-10 (ID and OOD); summary = median over folds, folds
                                    improved, two-sided sign test, and the row sum the shipped k was chosen by
    figS6c_weighting.csv (+ _summary) the same for the neighbour weighting at k = 25: cos^4 (shipped), cos^8, softmax
                                    (tau = 0.05)
    figS6d_fill_map.csv             zero-shot (x chance) of each fill map (MLP, linear ridge, graph with cos weights at k =
                                    25), for every prior-bank set (offline fill-map scan)
    figS6e_posthoc_vs_trained.csv   top-10 hits each pulled model recovers over the post-hoc model when the zero-shot fills
                                    (cos^4) join the candidates
    figS6aux_fold4_arms.csv         fold 4 only: every weighting / k arm tested there (incl. cos^2, softmax tau = 0.02,
                                    support- and held-out-recall-weighted neighbours, cos^4 at k = 10), relative to the
                                    rebuilt cos fill
    figS6_meta.json                 counts, medians, inputs

B, C and aux read the fill sweeps (analysis/readout/fill_fix/fill_fix.py on fills rebuilt by build_weight_fills.py with
each model's own prior refit; Fig. 3B's scorer, full 19,895-row vocabulary; 12-fold tables by summarize.py). D reads the
fill-map scan (analysis/readout/mapcache_score.py, one TSV per map: pull-side OOD flips, zero-shot x chance, map, bank
set), whose zero-shot scorer ranks each unseen perturbation's pooled query among the untrained rows only, on a fold-4
model trained without priors (H8; K562 held out), with the earlier cos weighting -- not Fig. 3B's scorer. Bank sets with
co-expression are dropped: that bank is built from the screens and leaks zero-shot truth. E reads fig3_extension.py
(run for S5 by figS6_ext.sbatch with --sim-pow 4): post-hoc = a model that never saw a filled row (H4), pulled = ExPert's
gated pull (EFFGATE3) and the full pull (FULLOPEN27), both trained with the fills as negatives; recovered = arm loss -
post-hoc loss, 95% interval = the two bootstrap half-widths in quadrature; percent = recovered / post-hoc loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Fig. 3's radial-table kinds -> the three shown here (the shuffled-prior control stays in Fig. 3's data)
KINDS = {'trained prototype': 'trained', 'untrained, shipped fill': 'fill',
         'trained, prior-placed (held out)': 'held-out placement'}
# fill-map scan name -> (family, parameter); the scan's nqxnds-evidence run (the shipped gate's evidence; its
# zero-shot column does not depend on the evidence, the pull column does)
MAPS = {'mlp': ('MLP', None), 'ridge0.02': ('linear', 0.02), 'ridge0.5': ('linear', 0.5),
        'graph10': ('graph', 10), 'graph25': ('graph', 25), 'graph50': ('graph', 50)}
METRICS = ['zs_ID', 'zs_OOD', 'trained_ID', 'trained_OOD']
# fill-sweep variants -> (label, parameter); the reference is the rebuilt cos fill at k = 25 ('base')
# 'plain' = the cos fill stored in each model before the change (vs the rebuilt one: its spread is the rebuild noise)
K_ARMS = {'plain': ('cos, k = 25', 25), 'simpow4_k25': ('cos⁴, k = 25', 25), 'simpow4_k50': ('cos⁴, k = 50', 50),
          'simpow4_k100': ('cos⁴, k = 100', 100)}
W_ARMS = {'simpow4': ('cos⁴', 4), 'simpow8': ('cos⁸', 8), 'simtau0.05': ('softmax, τ = 0.05', None)}
SHIPPED = {'b': 'simpow4_k25', 'c': 'simpow4'}
# E: the extension table's post-hoc reference and pulled arms; the candidate set = the graph fill
POSTHOC = 'H4'
PULLED = {'EFFGATE3': 'ExPert', 'FULLOPEN27': 'full pull'}
FILLS = '+ fills (graph map, own atoms)'
# fill_fix.py record paths (summarize.py's COLS)
PATHS = {'zs_ID': ('zs', 'pooled_seen_dataset', 'w_zs_x'), 'zs_OOD': ('zs', 'pooled_unseen_dataset', 'w_zs_x'),
         'trained_ID': ('seen_id', 'top10'), 'trained_OOD': ('seen_ood', 'top10')}


def relative(t: pd.DataFrame, arms: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """per fold x arm x metric: value / base - 1 (summarize.py's definition); summary over folds"""
    base = t[t.variant == 'base'].set_index('fold')
    rows = []
    for v, (lab, par) in arms.items():
        x = t[t.variant == v].set_index('fold')
        for m in METRICS:
            rel = x[m] / base.loc[x.index, m] - 1
            rows += [dict(fold=f, variant=v, label=lab, param=par, metric=m, value=x.loc[f, m],
                          base=base.loc[f, m], rel=r) for f, r in rel.items()]
    long = pd.DataFrame(rows)
    summ = []
    for (v, m), g in long.groupby(['variant', 'metric'], sort=False):
        w, n = int((g.rel > 0).sum()), int((g.rel != 0).sum())
        summ.append(dict(variant=v, label=g.label.iloc[0], param=g.param.iloc[0], metric=m, n_folds=len(g),
                         median_rel=float(np.median(g.rel)), folds_improved=w, folds_changed=n,
                         sign_p=binomtest(w, n).pvalue if n else 1.0))
    s = pd.DataFrame(summ)
    # the selection criterion: zero-shot ID + zero-shot OOD + trained OOD (median relative changes)
    rs = s[s.metric.isin(['zs_ID', 'zs_OOD', 'trained_OOD'])].groupby('variant').median_rel.sum()
    s['row_sum'] = s.variant.map(rs)
    return long, s


def fold4_arms(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        rec = json.loads(p.read_text())['4']
        for v, x in rec.items():
            if v in ('n_fill', 'trained_only', 'plain'):
                continue
            vals = {}
            for m, pth in PATHS.items():
                r = x
                for k in pth:
                    r = r[k]
                vals[m] = float(r)
            rows.append(dict(source=p.parent.name, variant=v, **vals))
    t = pd.DataFrame(rows).drop_duplicates('variant')
    base = t[t.variant == 'base'].iloc[0]
    for m in METRICS:
        t[f'{m}_rel'] = t[m] / base[m] - 1
    return t


def read_scan(d: Path, shipped: str) -> pd.DataFrame:
    rows = []
    for mp, (fam, par) in MAPS.items():
        t = pd.read_csv(d / f'scores_nqxnds_{mp}.tsv', sep='\t', header=None, names=['pull_flips', 'zero_shot_x_chance',
                                                                                  'map', 'banks'])
        assert (t['map'] == mp).all(), mp
        rows.append(t.assign(family=fam, param=par))
    t = pd.concat(rows, ignore_index=True)
    t = t[~t.banks.str.contains('coexpr')].copy()                       # leaks zero-shot truth
    t['n_banks'] = t.banks.str.count(r'\+') + 1
    t['shipped_banks'] = t.banks == shipped
    assert t.shipped_banks.sum() == len(MAPS), f'shipped bank set {shipped} missing for some map'
    return t


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    # ---- A: angle to the mean trained-prototype direction ----
    r = pd.read_parquet(a.radial)
    r = r[r.kind.isin(KINDS)].assign(kind=lambda t: t.kind.map(KINDS))
    r[['perturbation', 'kind', 'angle_deg']].to_csv(out / 'figS6a_angle.csv', index=False)
    med_a = r.groupby('kind').angle_deg.median().round(2).to_dict()
    print(f'[A] median angle: {med_a}', flush=True)

    # ---- B / C: k and neighbour weighting, 12 fold models, vs the rebuilt cos fill ----
    summaries = {}
    for panel, path, arms, name in (('b', a.wk12, K_ARMS, 'figS6b_graph_k'), ('c', a.w12, W_ARMS, 'figS6c_weighting')):
        long, s = relative(pd.read_csv(path), arms)
        long.to_csv(out / f'{name}.csv', index=False)
        s.to_csv(out / f'{name}_summary.csv', index=False)
        summaries[panel] = s
        print(f'[{panel.upper()}] median change vs the rebuilt cos fill (%)\n'
              + s.pivot(index='label', columns='metric', values='median_rel').mul(100).round(1).to_string() + '\n'
              + s[['label', 'metric', 'folds_improved', 'folds_changed', 'sign_p', 'row_sum']].round(4).to_string(index=False),
              flush=True)

    # ---- aux: every arm tested on fold 4 ----
    f4 = fold4_arms([Path(a.w_fold4), Path(a.wk_fold4)])
    f4.to_csv(out / 'figS6aux_fold4_arms.csv', index=False)

    # ---- D: the fill-map scan ----
    t = read_scan(Path(a.map_scores_dir), a.shipped_banks)
    cols = ['map', 'family', 'param', 'banks', 'n_banks', 'shipped_banks', 'zero_shot_x_chance', 'pull_flips']
    t[cols].to_csv(out / 'figS6d_fill_map.csv', index=False)
    s = t.groupby('map').zero_shot_x_chance.agg(['median', 'max']).join(
        t[t.shipped_banks].set_index('map').zero_shot_x_chance.rename('shipped'))
    print(f'[D] {t.banks.nunique()} co-expression-free bank sets per map\n{s.round(2).to_string()}', flush=True)

    # ---- E: the fill's cost on trained queries, post-hoc vs trained ----
    e = pd.read_csv(a.extension)
    em = json.loads((Path(a.extension).parent / 'fig3_extension_meta.json').read_text())
    n_trained = e[e.candidates == 'trained only'].set_index('arm').n_candidates
    e = e[e.candidates == FILLS].set_index('arm')
    hw = lambda arm: (float(e.loc[arm, 'hi']) - float(e.loc[arm, 'lo'])) / 2
    post = float(e.loc[POSTHOC, 'delta'])
    rows = []
    for arm, lab in PULLED.items():
        lift = float(e.loc[arm, 'delta']) - post
        h = float(np.hypot(hw(arm), hw(POSTHOC)))
        rows.append(dict(arm=arm, model=lab, delta=float(e.loc[arm, 'delta']), delta_posthoc=post, recovered=lift,
                         ci_lo=lift - h, ci_hi=lift + h, recovered_pct=100 * lift / -post,
                         ci_lo_pct=100 * (lift - h) / -post, ci_hi_pct=100 * (lift + h) / -post,
                         n_rows_added=int(e.loc[arm, 'n_candidates'] - n_trained[arm]),
                         sim_pow=em.get('sim_pow')))
    imp = pd.DataFrame(rows)
    imp.to_csv(out / 'figS6e_posthoc_vs_trained.csv', index=False)
    print(f'[E] fill cos^{em.get("sim_pow")}; post-hoc loss {post:.0f}\n'
          f'{imp[["model", "delta", "recovered", "ci_lo", "ci_hi", "recovered_pct"]].round(1).to_string(index=False)}',
          flush=True)

    (out / 'figS6_meta.json').write_text(json.dumps(dict(
        n={k: int(v) for k, v in r.kind.value_counts().items()}, median_angle_deg=med_a,
        shipped=SHIPPED, n_folds=int(summaries['b'].n_folds.max()), extension_sim_pow=em.get('sim_pow'),
        shipped_banks=a.shipped_banks, n_bank_sets=int(t.banks.nunique()), maps=MAPS,
        zero_shot_shipped={m: float(v) for m, v in s.shipped.items()},
        zero_shot_median={m: float(v) for m, v in s['median'].items()}), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--radial', required=True, help='Fig. 3 data fig3b_radial.parquet')
    ap.add_argument('--wk12', required=True, help='12-fold k sweep table (fill_fix summarize.py: wk12_summary.csv)')
    ap.add_argument('--w12', required=True, help='12-fold weighting table (fill_fix summarize.py: w12_summary.csv)')
    ap.add_argument('--w-fold4', required=True, help='fold-4 weighting arms (fill_fix.py: w_fold4/k.json)')
    ap.add_argument('--wk-fold4', required=True, help='fold-4 k arms under cos^4 (fill_fix.py: wk_fold4/k.json)')
    ap.add_argument('--map-scores-dir', required=True, help='fill-map scan TSVs (analysis_out/mapcache scores_<map>.tsv)')
    ap.add_argument('--shipped-banks', required=True, help="the shipped prior-bank set, '+'-joined as in the scan")
    ap.add_argument('--extension', required=True, help='S5 fig3_extension.csv (figS6_ext.sbatch, --sim-pow 4)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
