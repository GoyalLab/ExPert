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

"""Supplementary Fig. S3 plotting data (benchmark, related to Fig. 2), read from the same CACHED benchmark
tables as fig2_data.py (whose loaders it reuses; nothing is rescored). One table per panel in --out-dir
(paper/figures/si/figS3/data); figS3_plot.py draws from those alone.

    figS3_palette.csv                          method -> display name, colour, draw order (as Fig. 2)
    figS3_meta.json                            metric columns, class counts, inputs
    figS3a_ood_condition.csv                   pseudobulk, per method x (fold, held-out condition) view: F1 /
                                               recall@10 / MRR (the condition counterpart of Fig. 2C)
    figS3b_increase_over_next_best.csv         per fold and regime: ExPert / best other method (single cells,
                                               efficiency-weighted, per-perturbation recall@10 etc.)
    figS3b_increase_over_next_best_summary.csv median / range over folds, folds won, Wilcoxon signed-rank
    figS3c_target_mask.csv                     own target gene masked in the query vs plain, pseudobulk, every
                                               ctx7 perturbation
    figS3d_scoring_level.csv                   single cells, efficiency-weighted cells and pseudobulk, per method x
                                               regime (mean +- SEM over views matched across methods AND levels)
    figS3d_scoring_level_views.csv             ... per (method, fold, view, level)
    figS3e_subsample_control.csv               OOD: cells >= effect-size threshold vs a random sample of as many
                                               cells (mean +- SEM over matched views)
    figS3e_subsample_control_views.csv         ... per (method, fold, view, threshold, arm)
  aux (alternatives / earlier analyses, drawn into si/figS3/aux)
    figS3aux_target_mask_cells.csv             target-mask ablation per cell set (target is / is not an input)
    figS3aux_train_res.csv / _train_slopes.csv training cost vs class-space size and its scaling exponent

Fingerprint's no-calls (abstention, called vs all cells, coverage) are Fig. S4 (figS4_data.py).
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

# ---- A: held-out conditions, pseudobulk -----------------------------------------------------------
def panel_a(t: pd.DataFrame, metrics, out: Path):
    """The pooled matched views (D2.pbulk_pick), labelled by the shift their holdout encodes; condition rows."""
    cols = {'F1': metrics[0], 'R10': F.R10, 'MRR': metrics[2]}
    m = F.matched(t.rename(columns=cols)).copy()
    m['shift'] = F.shift_label(m)
    a = m[m['shift'] == 'ood_condition'][['method', 'fold', 'split', 'shift', *metrics]]
    a.to_csv(out / 'figS3a_ood_condition.csv', index=False)
    print(f'  [A] {a.split.nunique()} held-out condition views x {a.method.nunique()} methods')


# ---- B: ExPert over the next-best method, per fold ------------------------------------------------
def panel_b(w: pd.DataFrame, metrics, out: Path):
    r, s = D2.headline_tables(F.matched(w), metrics)
    r.to_csv(out / 'figS3b_increase_over_next_best.csv', index=False)
    s.to_csv(out / 'figS3b_increase_over_next_best_summary.csv', index=False)
    print(f'  [B] {r.fold.nunique()} folds')


# ---- C: own target gene masked in the query, pseudobulk -------------------------------------------
def _read_groups(f: Path, method: str, arm: str) -> pd.DataFrame:
    s = pd.read_parquet(f)
    for c in ('method', 'fold', 'split', 'dataset', 'label', 'pred', 'agg'):
        s[c] = s[c].astype(str)
    s = s[s['thr'].isna() & (s['agg'] == 'mean')].drop(columns=['thr'])
    s['split'] = s['split'] + ':' + s['dataset']
    return s.assign(method=method, arm=arm)


def panel_c(mask_csvs: list, main_pbulk: Path, query_h5ad: Path, out: Path):
    """Target-mask ablation at PSEUDOBULK level ('all perturbations' version): every fold's ablation outputs
    pooled per (view, perturbation) group by score_pbulk_groups.py (utils/abl_mask_target/pbulk.sbatch),
    plain vs masked, one vote per perturbation. Fingerprint was only scored on target-input cells, so its
    'all' = the arm's groups for perturbations whose target IS an input + the published pooled groups for
    the rest (the mask cannot reach them)."""
    import h5py
    with h5py.File(query_h5ad, 'r') as f:
        var = f['var']; vn = set(var[var.attrs['_index']].asstr()[:])
    parts = []
    for c in mask_csvs:
        root = Path(c).parent.parent / 'pbulk_groups'
        for f in sorted(root.glob('*/*.parquet')):
            name, tag = f.stem, f.parent.name
            if tag == 'results':
                base, arm = name.rsplit('_', 1)
            else:
                base, arm = name, tag.replace('baselines_', '')
            if arm not in ('plain', 'masked'):
                continue
            s = _read_groups(f, F.ALIAS.get(base, base), arm)
            if base.startswith('fingerprint'):
                fold = s['fold'].iloc[0]
                pub = _read_groups(main_pbulk / f'{base}.parquet', s['method'].iloc[0], arm)
                pub = pub[(pub['fold'] == fold) & ~pub['label'].isin(vn)]
                s = pd.concat([s[s['label'].isin(vn)], pub], ignore_index=True)
            parts.append(s)
    if not parts:
        raise SystemExit('[C] no pooled target-mask ablation groups found next to --mask-ablation')
    s = pd.concat(parts, ignore_index=True)
    rows = []
    for arm, g in s.groupby('arm'):
        v = D2.pbulk_views(g.drop(columns='arm'))
        rows.append(v.assign(arm=arm))
    v = pd.concat(rows, ignore_index=True)
    g = v.groupby(['method', 'arm', 'regime']).agg(F1=('F1', 'mean'), R10=('R10', 'mean'), MRR=('MRR', 'mean'),
                                                   n=('n', 'sum'), n_views=('F1', 'size')).reset_index()
    g['cells'] = 'all'
    g.to_csv(out / 'figS3c_target_mask.csv', index=False)
    print(f'  [C] {s.fold.nunique()} folds, {len(s):,} groups')


# ---- D: single cells vs efficiency-weighted cells vs pseudobulk -----------------------------------
def panel_d(sel: pd.DataFrame, pdir: Path, agg: str, out: Path):
    """Same views, three scorings: single cells unweighted, single cells efficiency-weighted (Fig. 2's
    per-cell metric) and pseudobulk (pooled groups). Per-perturbation F1 / recall@10 / MRR per (fold, view),
    matched across methods AND levels, then mean +- sem over views per method x regime x level."""
    rows = []
    for (m, fold, split, ds), v in sel.groupby(['method', 'fold', 'split', 'dataset'], observed=True):
        for lvl, vv in (('cell', v.assign(w=1.0)), ('cell_eff', v)):
            r = D2.perpert(vv)
            if r:
                rows.append(dict(method=m, fold=fold, split=f'{split}:{ds}', level=lvl, **r))
    c = pd.DataFrame(rows)
    s = D2.load_pbulk(pdir)
    s = s[s['thr'].isna() & (s['agg'] == agg)].drop(columns='thr')
    pb = D2.pbulk_views(s).rename(columns={'R10': 'recall@10'})[['method', 'fold', 'split', 'F1', 'recall@10', 'MRR']]
    t = pd.concat([c, pb.assign(level='pbulk')], ignore_index=True)
    t['regime'] = F.regime(t['split'])
    t = t[t['regime'].isin(['id', 'ood'])]
    keep = None                                      # views every (method, level) has
    for _, g in t.groupby(['method', 'level']):
        k = set(map(tuple, g[['fold', 'split']].astype(str).to_numpy()))
        keep = k if keep is None else keep & k
    t = t[[tuple(r) in keep for r in t[['fold', 'split']].astype(str).to_numpy()]]
    t.to_csv(out / 'figS3d_scoring_level_views.csv', index=False)
    g = t.groupby(['method', 'regime', 'level'])[['F1', 'recall@10', 'MRR']].agg(['mean', 'sem'])
    g.columns = [f'{a}_{b}' for a, b in g.columns]
    g = g.reset_index()
    g.to_csv(out / 'figS3d_scoring_level.csv', index=False)
    ex = g[g.method == 'ExPert'].set_index(['regime', 'level'])[['F1_mean', 'recall@10_mean', 'MRR_mean']]
    print('  [D] ExPert (matched views):\n' + ex.round(4).to_string())


# ---- E: is the effect-size trend (Fig. 2D) a sample-size effect? ----------------------------------
def panel_e(sel: pd.DataFrame, out: Path, thresholds=(0.0, 1.0, 2.0, 3.0), n_draws=5, min_cells=20, seed=0):
    """OOD, per (method, fold, dataset) view: per-perturbation F1 / recall@10 / MRR (one vote per
    perturbation, cells unweighted) on (a) the cells with knockdown effect size >= t and (b) a random
    sample of the SAME number of cells drawn from all of the view's cells regardless of effect size
    (n_draws draws, averaged). Views must have >= min_cells cells at every threshold and are matched
    across methods; mean +- SEM over views."""
    rng = np.random.default_rng(seed)
    s = sel[sel['regime'] == 'ood']
    rows = []
    for (m, fold, ds), v in s.groupby(['method', 'fold', 'dataset'], observed=True):
        eff = pd.to_numeric(v['efficiency'], errors='coerce').to_numpy(float)
        base = v.assign(w=1.0).reset_index(drop=True)
        n_t = [len(v) if t <= 0 else int(np.nansum(eff >= t)) for t in thresholds]
        if min(n_t) < min_cells:
            continue
        for t, n in zip(thresholds, n_t):
            keep = np.ones(len(v), bool) if t <= 0 else np.nan_to_num(eff, nan=-np.inf) >= t
            r = D2.perpert(base[keep])
            rows.append(dict(method=m, fold=fold, dataset=ds, eff=t, arm='effect size', n_cells=n, **r))
            draws = [D2.perpert(base.iloc[rng.choice(len(base), n, replace=False)]) for _ in range(n_draws)]
            rows.append(dict(method=m, fold=fold, dataset=ds, eff=t, arm='random', n_cells=n,
                             **{k: float(np.mean([d[k] for d in draws])) for k in draws[0]}))
    t = pd.DataFrame(rows)
    t['split'] = t['fold'].astype(str) + ':' + t['dataset'].astype(str)
    t = F.matched(t, key=('split',))
    t.to_csv(out / 'figS3e_subsample_control_views.csv', index=False)
    g = (t.groupby(['method', 'arm', 'eff'])[['F1', 'recall@10', 'MRR']].agg(['mean', 'sem'])
          .stack(level=0, future_stack=True).reset_index().rename(columns={'level_3': 'metric'}))
    g['n_views'] = t['split'].nunique()
    g.to_csv(out / 'figS3e_subsample_control.csv', index=False)
    ex = g[(g.method == 'ExPert') & (g.metric == 'recall@10')].pivot(index='eff', columns='arm', values='mean')
    print(f'  [E] {t["split"].nunique()} OOD views; ExPert recall@10 by threshold:\n{ex.round(3)}')


# ---- aux --------------------------------------------------------------------------------------------
def aux_mask_cells(csvs: list, out: Path):
    """Target-mask ablation: mean over (fold, dataset) views of each (method, arm, split, cell set),
    pooling every fold's table."""
    t = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    print(f'  [mask] {t["fold"].nunique() if "fold" in t else 1} folds: '
          f'{sorted(t["fold"].unique()) if "fold" in t else "?"}')
    t['regime'] = F.regime(t['split'])
    g = t.groupby(['method', 'arm', 'regime', 'cells']).agg(
        F1=('F1', 'mean'), R10=('R10', 'mean'), MRR=('MRR', 'mean'), n=('n', 'sum'),
        n_views=('F1', 'size')).reset_index()
    g.to_csv(out / 'figS3aux_target_mask_cells.csv', index=False)


def aux_train(sizes, costs: pd.DataFrame, drop: set, n_folds: int, expert_train: dict | None, out: Path):
    """Training cost vs class-space size. train + train+eval rows of the cost table.
    ExPert is trained ONCE per fold on the full vocabulary and serves every class space, so its row
    is the same measured training cost at every size (exponent 0 by construction)."""
    res = []
    for name, p in sizes:
        C = D2.n_classes(p)
        r = D2.stage_costs(costs, name, ('train', 'train+eval'))
        r = r[~r['method'].isin(drop)].copy()
        r['per_fold'] = r['compute_h'] / n_folds
        r['jobs_per_fold'] = r['n_jobs'] / n_folds
        if expert_train:
            r = pd.concat([r, pd.DataFrame([dict(method='ExPert', cpu_h=0.0, gpu_h=expert_train['gpu_h'],
                                                 compute_h=expert_train['gpu_h'], ram=expert_train['ram_gb'],
                                                 n_jobs=n_folds, combined=False, per_fold=expert_train['gpu_h'],
                                                 jobs_per_fold=1.0)])], ignore_index=True)
        r['size'], r['C'] = name, C
        res.append(r)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(out / 'figS3aux_train_res.csv', index=False)
    D2.slopes(res).to_csv(out / 'figS3aux_train_slopes.csv', index=False)


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ctx7, small = Path(a.ctx7_dir), Path(a.small_dir)
    metrics = [a.f1, F.R10, a.mrr]
    costs = F.apply_overrides(pd.read_csv(a.costs), a.cost_overrides)

    D2.panel_palette(ctx7, out, name='figS3_palette.csv')
    w = D2.load_weighted(ctx7, Path(a.extra_ctx7) if a.extra_ctx7 else None)
    for col in (a.f1, a.mrr):
        if col not in w.columns:
            raise SystemExit(f'{col} is not in the ctx7 tables (needs --extra-ctx7?)')
    _, t, pick = D2.pbulk_pick(Path(a.pbulk_dir), a.pbulk_agg)
    panel_a(t, metrics, out)
    panel_b(w, metrics, out)
    panel_c(a.mask_ablation, Path(a.pbulk_dir), Path(a.mask_query), out)
    sel = D2.load_selective(Path(a.selective_dir))
    panel_d(sel, Path(a.pbulk_dir), pick, out)
    panel_e(sel, out)

    aux_mask_cells(a.mask_ablation, out)
    timing = D2.expert_train_timing(Path(a.expert_train_root), range(a.folds))
    aux_train([('small', small), ('ctx7', ctx7)], costs, set(a.drop_scaling), a.folds,
              dict(gpu_h=float(timing.train_gpu_h.median()), ram_gb=a.expert_train_ram_gb), out)

    meta = dict(vars(a), pbulk_agg=pick, n_classes={'small': D2.n_classes(small), 'ctx7': D2.n_classes(ctx7)})
    (out / 'figS3_meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print('wrote', sorted(p.name for p in out.glob('figS3*')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctx7-dir', required=True, help='benchmark collect output, 1,157-class run')
    ap.add_argument('--small-dir', required=True, help='benchmark collect output, 200-class run (training cost)')
    ap.add_argument('--costs', required=True, help='slurm_costs.csv (size-keyed)')
    ap.add_argument('--cost-overrides', required=True, help='benchmark config/cost_overrides.yaml')
    ap.add_argument('--pbulk-dir', required=True, help='score_pbulk_groups.py output (metrics/ctx7/pbulk_groups)')
    ap.add_argument('--selective-dir', required=True, help='score_selective.py per-cell cache (ctx7)')
    ap.add_argument('--mask-ablation', nargs='+', required=True,
                    help='target-mask ablation tables, every fold (utils/abl_mask_target/score.py)')
    ap.add_argument('--mask-query', required=True,
                    help='ctx7 query h5ad (var names = input genes) for the pooled target-mask table')
    ap.add_argument('--expert-train-root', required=True,
                    help='dir with ExPert_fold<k>/run_0 (training logs) -> ExPert training cost')
    ap.add_argument('--expert-train-ram-gb', type=float, required=True,
                    help='measured peak RAM of an ExPert training job (sacct MaxRSS)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--extra-ctx7', default=None, help='score_extra.py output for ctx7 (optional)')
    ap.add_argument('--mrr', default='weighted_mrr', choices=('weighted_mrr', 'balanced_weighted_mrr'))
    ap.add_argument('--f1', default='weighted_macro_f1', choices=('weighted_macro_f1', 'balanced_weighted_macro_f1'))
    ap.add_argument('--pbulk-agg', default=None, choices=('mean', 'effw'),
                    help="pool cells by plain or efficiency-weighted mean; default: as fig2_data.py (effw only if "
                         "it scores better)")
    ap.add_argument('--drop-scaling', nargs='*', default=['scgpt', 'gears', 'cellot'],
                    help='left out of the training-cost scaling (as Fig. 2G-I)')
    ap.add_argument('--folds', type=int, default=12)
    main(ap.parse_args())
