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

"""Fig. 2 data: the benchmark, read from the CACHED tables the benchmark pipeline's collect step
wrote (no logits are loaded, nothing is rescored). Writes one source table per panel to --out-dir;
fig2_plot.py draws from those alone.

Headline metrics everywhere (A, B, D, E): efficiency-weighted macro-F1, recall@10 and MRR.
recall@10 is averaged per perturbation (balanced_weighted_recall@10). F1 defaults to collect's
weighted_macro_f1, which averages per-class F1 by weighted SUPPORT; --f1 balanced_weighted_macro_f1
gives every perturbation one vote. MRR defaults to the cell-level weighted_mrr; --mrr
balanced_weighted_mrr is its per-perturbation form. Both balanced columns come from score_extra.py
(--extra-ctx7 / --extra-small). Every mean is over a MATCHED (fold, split) cell set --
the cells every drawn method has -- so differences are method, not fold composition.

    fig2_palette.csv      method -> display name, colour (the benchmark's performance-ranked
                          palette: viridis for methods, greys for baselines), draw order
    fig2a_radar.csv       per method: F1 / recall@10 / MRR, ID and OOD
    fig2b_shift.csv       per method x (fold, held-out dataset): the three metrics, OOD rows
                          labelled by the shift their holdout encodes (cell line / condition / donor)
    fig2c_lines.csv       OOD threshold sweep: mean +- SEM per method x min-efficiency (unweighted
                          metrics on the cells passing the threshold; the x axis IS efficiency)
    fig2c_corr.csv        per method x (fold, dataset) cell: Pearson r between threshold and metric
    fig2d_cost.csv        eval compute per method (ctx7) + its OOD headline metrics
    fig2e_perf.csv        x over random per method x size x regime (class space changes with size,
                          so chance does too -- the one place a ratio is the honest scale)
    fig2e_res.csv         eval compute / jobs / peak RAM per method x size
    fig2e_slopes.csv      scaling exponent b (cost ~ C^b) between the two sizes

Supplementary Fig. S3 tables are figS3_data.py's (paper/figures/si/figS3/data); it reuses this
module's loaders.
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

KEY = ['method', 'split', 'direction', 'fold']


def load_weighted(mdir: Path, extra: Path | None) -> pd.DataFrame:
    w = pd.read_csv(mdir / 'weighted_metrics.csv')
    w['method'] = w['method'].replace(F.ALIAS)
    w = w[~w['method'].isin(F.HIDE)]
    if extra is not None and extra.is_file():
        x = pd.read_csv(extra)
        x['method'] = x['method'].replace(F.ALIAS)
        for k in KEY:
            w[k], x[k] = w[k].astype(str), x[k].astype(str)
        w = w.merge(x.drop(columns=['n', 'path'], errors='ignore'), on=KEY, how='left')
        for col in ('weighted_mrr', 'weighted_macro_f1'):
            if f'{col}_check' in w:
                d = (w[col] - w[f'{col}_check']).abs().max()
                print(f'  [extra] merged {extra.name}: max |{col} - check| = {d:.2e}')
                if d > 1e-6:
                    raise SystemExit('extra-metrics rows do not match weighted_metrics.csv')
    w['regime'] = F.regime(w['split'])
    return w[w['regime'].isin(['id', 'ood'])]


def n_classes(mdir: Path) -> int:
    return len(pd.read_csv(mdir / 'train_class_support.csv'))


def panel_palette(ctx7: Path, out: Path, name: str = 'fig2_palette.csv'):
    sw = pd.read_csv(ctx7 / 'sweep_metrics.csv')
    sw['method'] = sw['method'].replace(F.ALIAS)
    ranked = F.rank_methods(sw)
    pal = F.method_palette(ranked)                 # ranked on ALL methods, as collect does
    pal['ExPert'] = F.EXPERT_COLOR                 # ExPert stands apart from the viridis/grey palette
    rows = [dict(method=m, display=F.disp(m), color=c, order=i)
            for i, (m, c) in enumerate(pal.items()) if m not in F.HIDE]
    rows.append(dict(method='baselines', display=F.disp('baselines'),
                     color=pal.get('baseline_centroid_notarget', '#808080'), order=len(rows)))
    pd.DataFrame(rows).to_csv(out / name, index=False)


def panel_a(w: pd.DataFrame, metrics, out: Path) -> pd.DataFrame:
    m = F.matched(w)
    # + median rank for the alternative radar (cell-level weighted median; no per-perturbation form)
    a = m.groupby(['method', 'regime'])[[*metrics, 'weighted_median_rank']].mean().reset_index()
    a.to_csv(out / 'fig2a_radar.csv', index=False)
    return a


def panel_b(w: pd.DataFrame, metrics, out: Path):
    m = F.matched(w).copy()
    m['shift'] = F.shift_label(m)
    b = m[m['shift'].isin(F.SHIFTS + F.ID_SHIFTS)][['method', 'fold', 'split', 'shift', *metrics]]
    b.to_csv(out / 'fig2b_shift.csv', index=False)
    headline(m, metrics, out)


def headline(m: pd.DataFrame, metrics, out: Path):
    r, s = headline_tables(m, metrics)
    r.to_csv(out / 'fig2_headline_folds.csv', index=False)
    s.to_csv(out / 'fig2_headline.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 80):
        print(s.round(4).to_string(index=False))


def headline_tables(m: pd.DataFrame, metrics) -> tuple[pd.DataFrame, pd.DataFrame]:
    """'ExPert's median increase over the next-best method across folds': per fold and regime, each
    method's mean over that fold's matched views; ratio = ExPert / the best OTHER method in that fold
    (all methods, and published methods only = no baseline_*). Median and range over folds, plus a
    two-sided Wilcoxon signed-rank test of ExPert vs the per-fold next-best (n = folds)."""
    from scipy.stats import wilcoxon
    m = m.copy()
    m['regime'] = F.regime(m['split'])
    f = m.groupby(['regime', 'fold', 'method'])[list(metrics)].mean().reset_index()
    rows = []
    for pool, keep in (('all methods', lambda s: s != 'ExPert'),
                       ('published methods', lambda s: (s != 'ExPert') & ~s.str.startswith('baseline'))):
        for (reg, fold), g in f.groupby(['regime', 'fold']):
            ex = g[g.method == 'ExPert']
            oth = g[keep(g.method)]
            if ex.empty or oth.empty:
                continue
            for met in metrics:
                j = oth[met].idxmax()
                rows.append(dict(pool=pool, regime=reg, fold=fold, metric=met, expert=float(ex[met].iloc[0]),
                                 next_best=float(oth.loc[j, met]), next_method=oth.loc[j, 'method']))
    r = pd.DataFrame(rows)
    r['ratio'] = r.expert / r.next_best
    s = []
    for (pool, reg, met), g in r.groupby(['pool', 'regime', 'metric']):
        p = wilcoxon(g.expert, g.next_best).pvalue if len(g) >= 5 else np.nan
        s.append(dict(pool=pool, regime=reg, metric=met, n_folds=len(g), median_ratio=g.ratio.median(),
                      min_ratio=g.ratio.min(), max_ratio=g.ratio.max(), folds_won=int((g.ratio > 1).sum()),
                      wilcoxon_p=p, stars='***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'n.s.',
                      next_best=g.next_method.value_counts().to_dict()))
    return r, pd.DataFrame(s)


def panel_c(ctx7: Path, out: Path):
    s = pd.read_csv(ctx7 / 'sweep_metrics_per_fold.csv')
    s['method'] = s['method'].replace(F.ALIAS)
    s = s[~s['method'].isin(F.HIDE)]
    sp = s['split'].astype(str)
    s = s[sp.str.startswith('ood_test:') | (sp == 'ood_test')].copy()
    s['eff'] = pd.to_numeric(s['min_efficiency'], errors='coerce').fillna(0.0)
    mets = list(F.SWEEP)
    s = F.matched(s, key=('fold', 'split', 'eff'))
    lines = (s.groupby(['method', 'eff'])[mets].agg(['mean', 'sem'])
             .stack(level=0, future_stack=True).reset_index()
             .rename(columns={'level_2': 'metric'}))
    lines.to_csv(out / 'fig2c_lines.csv', index=False)
    rows = []
    for (meth, fold, split), g in s.groupby(['method', 'fold', 'split']):
        g = g.sort_values('eff')
        for met in mets:
            x, y = g['eff'].to_numpy(), g[met].to_numpy()
            ok = np.isfinite(y)
            r = (np.corrcoef(x[ok], y[ok])[0, 1]
                 if ok.sum() >= 3 and np.ptp(y[ok]) > 0 else np.nan)
            rows.append(dict(method=meth, fold=fold, split=split, metric=met, r=r))
    pd.DataFrame(rows).to_csv(out / 'fig2c_corr.csv', index=False)


def stage_costs(costs: pd.DataFrame, size: str, stages) -> pd.DataFrame:
    c = costs[costs['size'].astype(str) == size].copy()
    c['method'] = c['method'].map(lambda m: F.COST_TO_METRIC.get(m, m))
    c = c[c['stage'].isin(stages)]
    c['compute_h'] = c['cpu_hours'].fillna(0) + c['gpu_hours'].fillna(0)
    return c.groupby('method').agg(
        cpu_h=('cpu_hours', 'sum'), gpu_h=('gpu_hours', 'sum'), compute_h=('compute_h', 'sum'),
        ram=('max_ram_GB', 'max'), n_jobs=('n_jobs', 'sum'),
        combined=('stage', lambda x: bool((x == 'train+eval').any()))).reset_index()


def eval_costs(costs: pd.DataFrame, size: str) -> pd.DataFrame:
    return stage_costs(costs, size, ('eval', 'train+eval'))


def slopes(res: pd.DataFrame) -> pd.DataFrame:
    """Scaling exponent b (cost ~ C^b) between the smallest and largest class space, per method."""
    lo, hi = res['C'].min(), res['C'].max()
    res = res.assign(burden=res['per_fold'] * res['jobs_per_fold'])
    rows = []
    for m, g in res.groupby('method'):
        g = g.set_index('C')
        if lo in g.index and hi in g.index:
            for col, lab in (('per_fold', 'compute'), ('jobs_per_fold', 'jobs'),
                             ('burden', 'burden'), ('ram', 'peak RAM')):
                fc = g.loc[hi, col] / g.loc[lo, col]
                rows.append(dict(method=m, what=lab, fold_change=fc,
                                 exponent=np.log(fc) / np.log(hi / lo), C_lo=lo, C_hi=hi))
    return pd.DataFrame(rows)


def panel_d(costs: pd.DataFrame, a: pd.DataFrame, metrics, out: Path):
    d = eval_costs(costs, 'ctx7')
    ood = a[a['regime'] == 'ood'].set_index('method')[metrics]
    d = d.join(ood, on='method')
    d.to_csv(out / 'fig2d_cost.csv', index=False)


def panel_e(sizes: list[tuple[str, Path]], costs: pd.DataFrame, extra: dict, metrics,
            drop: set, n_folds: int, out: Path):
    ws = {n: load_weighted(p, extra.get(n)) for n, p in sizes}
    common = set.intersection(*(set(w['method']) for w in ws.values())) - drop
    perf, res = [], []
    for name, p in sizes:
        C = n_classes(p)
        w = F.matched(ws[name][ws[name]['method'].isin(common)])
        g = w.groupby(['method', 'regime'])[metrics].mean().reset_index()
        for met in metrics:
            g[f'{met}_x_random'] = g[met] / F.chance(met, C)
        g['size'], g['C'] = name, C
        perf.append(g)
        r = eval_costs(costs, name)
        r = r[~r['method'].isin(drop)]
        r['per_fold'] = r['compute_h'] / n_folds
        r['jobs_per_fold'] = r['n_jobs'] / n_folds
        r.loc[r['method'] == 'ExPert', 'jobs_per_fold'] = 1      # one eval job per fold, by design
        r['size'], r['C'] = name, C
        res.append(r)
    perf, res = pd.concat(perf), pd.concat(res)
    perf.to_csv(out / 'fig2e_perf.csv', index=False)
    res.to_csv(out / 'fig2e_res.csv', index=False)
    slopes(res).to_csv(out / 'fig2e_slopes.csv', index=False)


def expert_train_timing(train_root: Path, folds) -> pd.DataFrame:
    """Per fold: median seconds per training epoch (tqdm 's/it' of the epoch bar), steps per epoch
    (checkpoint 'epoch=E-step=S'), batch size and epochs from config.yaml -> seconds per training cell.
    Epoch time includes the per-epoch validation/monitor passes, so it is an UPPER bound."""
    import re
    import yaml
    rows = []
    for k in folds:
        run = train_root / f'ExPert_fold{k}' / 'run_0'
        logs = sorted((run / '.logs').glob('*.out'))
        sit, steps = [], []
        for f in logs:
            txt = f.read_text(errors='ignore')
            sit += [float(x) for x in re.findall(r'<[0-9:]+, ([0-9.]+)s/it', txt)]
            # checkpoint names printed in the log; the console wrap can CUT trailing digits (which
            # only ever shrinks a step count), so the largest steps/epoch ratio is the true one
            steps += [int(st) / (int(e) + 1) for e, st in re.findall(r'epoch=(\d+)-step=(\d+)', txt)]
        steps = [max(steps)] if steps else []
        cfg = yaml.safe_load((run / 'config.yaml').read_text())
        if not sit or not steps:
            continue
        bs, ep = cfg['data']['batch_size'], cfg['train']['max_epochs']
        spe = float(np.median(steps))
        t_ep = float(np.median(sit))
        rows.append(dict(fold=f'fold_{k}', epoch_s=t_ep, steps_per_epoch=spe, batch_size=bs, max_epochs=ep,
                         s_per_train_cell=t_ep / (spe * bs), train_gpu_h=ep * t_ep / 3600))
    return pd.DataFrame(rows)


def _encode_seconds(log: Path) -> float:
    """Seconds inside predict() (encoding) in one ExPert eval log: from each '[Predict] scoring head'
    to the next 'Aggregating predictions', using the last rich timestamp at or before each line."""
    import re
    from datetime import datetime
    ts, start, total = None, None, 0.0
    for line in log.read_text(errors='ignore').splitlines():
        m = re.match(r'\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\]', line)
        if m:
            ts = datetime.strptime(m.group(1), '%m/%d/%y %H:%M:%S')
        if '[Predict] scoring head' in line and ts is not None:
            start = ts
        elif 'Aggregating predictions' in line and start is not None and ts is not None:
            total += (ts - start).total_seconds(); start = None
    return total


def expert_gpu_estimate(timing: pd.DataFrame, sel: pd.DataFrame, costs: pd.DataFrame, eval_logs: Path,
                        n_folds: int, cpus: int, infer_frac: float, out: Path):
    """ExPert eval on a GPU node, per fold: the measured CPU job with ONLY its encoding swapped.
        wall_cpu   = measured eval CPU core-h (cost table, after overrides) / folds / cpus
        enc_cpu    = time inside predict() in that fold's CPU eval log
        enc_gpu    = eval cells x s per training cell x infer_frac (training throughput, forward only)
        wall_gpu   = wall_cpu - enc_cpu + enc_gpu          (startup, data loading, scoring, writing kept)
        compute_h  = wall_gpu x (cpus + 1 GPU)             (the cost table's 1:1 CPU-core/GPU billing)"""
    import re
    cpu_h = float(eval_costs(costs, 'ctx7').set_index('method').loc['ExPert', 'cpu_h'])
    wall_cpu = cpu_h / n_folds / cpus * 3600
    logs = {}
    for f in sorted(eval_logs.glob('*.out'), key=lambda p: p.stat().st_mtime):
        txt = f.read_text(errors='ignore')
        m = re.search(r'ctx7/(fold_\d+)', txt)
        if m and txt.count('Aggregating predictions') >= 2:
            logs[m.group(1)] = f                       # newest complete log per fold wins
    cells = sel[sel.method == 'ExPert'].groupby('fold').size()
    rows = []
    for _, r in timing.iterrows():
        fold = r['fold']
        if fold not in logs or fold not in cells.index:
            continue
        enc_cpu = _encode_seconds(logs[fold])
        enc_gpu = cells[fold] * r['s_per_train_cell'] * infer_frac
        wall_gpu = wall_cpu - enc_cpu + enc_gpu
        rows.append(dict(fold=fold, log=logs[fold].name, eval_cells=int(cells[fold]), wall_cpu_s=wall_cpu,
                         enc_cpu_s=enc_cpu, enc_gpu_s=enc_gpu, wall_gpu_s=wall_gpu,
                         gpu_h=wall_gpu / 3600, cpu_h=wall_gpu / 3600 * cpus))
    t = pd.DataFrame(rows)
    t.to_csv(out / 'fig2d_expert_gpu_estimate.csv', index=False)
    return t


def panel_c_rho(sel: pd.DataFrame, out: Path):
    """Per-cell link between knockdown efficiency and how well the cell is scored: Spearman rho
    between efficiency and (top-1 correct | top-10 hit | reciprocal rank) over the cells of each OOD
    (fold, dataset) view -- the per-cell counterparts of F1 / recall@10 / MRR in Fig. 2c."""
    from scipy.stats import spearmanr
    outcome = {'macro_f1_no_ds_bias': lambda r: (r < 1).astype(float),
               'balanced_recall@10': lambda r: (r < 10).astype(float),
               'mrr': lambda r: 1.0 / (r + 1.0)}
    o = sel[(sel.regime == 'ood') & np.isfinite(sel.efficiency)]
    rows = []
    for (m, fold, split, ds), v in o.groupby(['method', 'fold', 'split', 'dataset'], observed=True):
        if len(v) < 20:
            continue
        e = v.efficiency.to_numpy(float); rk = v['rank'].to_numpy(float)
        for met, f in outcome.items():
            y = f(rk)
            rho = spearmanr(e, y).statistic if np.ptp(y) > 0 else np.nan
            rows.append(dict(method=m, fold=fold, split=f'{split}:{ds}', metric=met, rho=rho, n=len(v)))
    pd.DataFrame(rows).to_csv(out / 'fig2c_rho.csv', index=False)


def panel_a_ext(sel: pd.DataFrame, C: int, w_check: pd.DataFrame, out: Path,
                ks=(1, 5, 10, 20, 50, 100)):
    """Radar candidates from the per-cell cache: per-perturbation, efficiency-weighted F1, MRR,
    recall@k and rank-AUROC, per (fold, dataset) view, mean over the matched views. F1 / recall@10 /
    MRR are checked against the cached collect + score_extra values (same definitions)."""
    rows = []
    for (m, fold, split, ds), v in sel.groupby(['method', 'fold', 'split', 'dataset'], observed=True):
        v = v[v.w > 0]
        if v.empty:
            continue
        sup = v.groupby('label').w.sum()
        tp = v.w.where(v.label == v.pred, 0.0).groupby(v.label).sum()
        predw = v.groupby('pred').w.sum().reindex(sup.index).fillna(0.0)
        prec = (tp / predw.replace(0, np.nan)).fillna(0.0); rec = tp / sup
        f1 = (2 * prec * rec / (prec + rec).replace(0, np.nan)).fillna(0.0)
        r = dict(method=m, fold=fold, split=f'{split}:{ds}', F1=f1.mean(),
                 MRR=(v.w / (v['rank'] + 1)).groupby(v.label).sum().div(sup).mean(),
                 AUROC=(v.w * (1 - v['rank'] / (C - 1))).groupby(v.label).sum().div(sup).mean())
        for k in ks:
            r[f'R{k}'] = ((v['rank'] < k) * v.w).groupby(v.label).sum().div(sup).mean()
        rows.append(r)
    t = F.matched(pd.DataFrame(rows))
    t['regime'] = F.regime(t['split'])
    # consistency with the cached tables the main radar uses
    chk = t.merge(w_check, on=['method', 'fold', 'split'], how='inner', suffixes=('', '_cache'))
    for a_, b_ in (('F1', 'balanced_weighted_macro_f1'), ('R10', 'balanced_weighted_recall@10'),
                   ('MRR', 'balanced_weighted_mrr')):
        if b_ in chk:
            print(f'  [radar-ext] max |{a_} - cached {b_}| = {(chk[a_] - chk[b_]).abs().max():.2e}')
    mets = ['F1', 'MRR', 'AUROC'] + [f'R{k}' for k in ks]
    g = t.groupby(['method', 'regime'])[mets].mean().reset_index()
    g.to_csv(out / 'fig2a_radar_ext.csv', index=False)
    for reg in ('id', 'ood'):
        x = g[g.regime == reg].set_index('method')[mets]
        best_other = x.drop(index='ExPert').max()
        print(f'  [radar-ext] {reg}: ExPert / best other ->',
              ' '.join(f'{c}={x.loc["ExPert", c] / best_other[c]:.2f}' for c in mets))


# ---- pseudobulk (group-level) versions of A-D (from score_pbulk_groups.py) ----------------------
PB_KS = (1, 5, 10, 20, 50, 100)


def load_pbulk(d: Path) -> pd.DataFrame:
    s = pd.concat([pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))], ignore_index=True)
    for c in ('method', 'fold', 'split', 'dataset', 'label', 'pred', 'agg'):
        s[c] = s[c].astype(str)
    s['method'] = s['method'].replace(F.ALIAS)
    s = s[~s['method'].isin(F.HIDE)]
    s['split'] = s['split'] + ':' + s['dataset']                  # the per-dataset view key of the cell tables
    return s


def pbulk_views(s: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, fold, view[, thr]): one vote per perturbation (= per group); F1 over the
    perturbations PRESENT in the view (precision counts every group predicted as that class)."""
    rows = []
    keys = ['method', 'fold', 'split'] + (['thr'] if 'thr' in s else [])
    for k, v in s.groupby(keys, dropna=False):
        hit = v['label'].to_numpy() == v['pred'].to_numpy()
        tp = pd.Series(hit.astype(float), index=v['label']).groupby(level=0).sum()
        npred = v['pred'].value_counts().reindex(tp.index).fillna(0.0)
        prec = (tp / npred.replace(0, np.nan)).fillna(0.0)
        f1 = (2 * prec * tp / (prec + tp).replace(0, np.nan)).fillna(0.0)   # recall = tp (one group each)
        r, C = v['rank'].to_numpy(float), float(v['C'].iloc[0])
        row = dict(zip(keys, k), n=len(v), F1=f1.mean(), MRR=np.mean(1 / (r + 1)), AUROC=np.mean(1 - r / (C - 1)),
                   median_rank=float(np.median(r) + 1))
        row.update({f'R{kk}': np.mean(r < kk) for kk in PB_KS})
        rows.append(row)
    t = pd.DataFrame(rows)
    t['regime'] = F.regime(t['split'])
    return t[t['regime'].isin(['id', 'ood'])]


def pbulk_pick(pdir: Path, agg: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Pooled groups, their matched (fold, view) table at the chosen pooling, and the pooling. agg=None picks
    'effw' over 'mean' only if ExPert scores better with it (mean of the matched-view F1 / recall@10 / MRR over
    both regimes); the pick is applied to every method. Returns (groups, matched views, pick)."""
    s = load_pbulk(pdir)
    base = s[s['thr'].isna()]
    v = {g: F.matched(pbulk_views(base[base['agg'] == g].drop(columns='thr'))) for g in ('mean', 'effw')}
    score = {g: v[g][['F1', 'R10', 'MRR']].mean().mean() for g in v}
    ex = {g: v[g][v[g].method == 'ExPert'][['F1', 'R10', 'MRR']].mean().round(4).to_dict() for g in v}
    exs = {g: float(np.mean(list(ex[g].values()))) for g in v}
    pick = agg or ('effw' if exs['effw'] > exs['mean'] else 'mean')   # applied to EVERY method either way
    print(f'  [pbulk] pooled score mean {score["mean"]:.4f} vs effw {score["effw"]:.4f} -> using {pick!r}; '
          f'ExPert mean {ex["mean"]} effw {ex["effw"]}')
    return s, v[pick], pick


def pbulk_panels(pdir: Path, metrics, costs, ctx7: Path, out: Path, agg: str | None = None) -> str:
    """Writes the A-D data tables for the POOLED scoring into out (same file names as the cell-level ones,
    so fig2_plot.py draws them unchanged), at the pooling pbulk_pick chooses."""
    s, t, pick = pbulk_pick(pdir, agg)
    out.mkdir(parents=True, exist_ok=True)
    cols = {'F1': metrics[0], 'R10': F.R10, 'MRR': metrics[2], 'median_rank': 'weighted_median_rank'}
    w = t.rename(columns=cols)
    radar = panel_a(w, metrics, out)
    t.groupby(['method', 'regime'])[['F1', 'MRR', 'AUROC'] + [f'R{k}' for k in PB_KS]].mean() \
        .reset_index().to_csv(out / 'fig2a_radar_ext.csv', index=False)
    panel_b(w, metrics, out)
    # C: OOD lines over the efficiency floor (eff 0 = all cells), same columns as panel_c's table
    th = s[(s['agg'] == pick)].copy()
    th['eff'] = th['thr'].fillna(0.0)
    tv = pbulk_views(th.drop(columns='thr').rename(columns={'eff': 'thr'}))
    tv = tv[tv['regime'] == 'ood'].rename(columns={'thr': 'eff', 'F1': 'macro_f1_no_ds_bias',
                                                     'R10': 'balanced_recall@10', 'MRR': 'mrr'})
    tv = F.matched(tv, key=('fold', 'split', 'eff'))
    mets = list(F.SWEEP)
    (tv.groupby(['method', 'eff'])[mets].agg(['mean', 'sem']).stack(level=0, future_stack=True).reset_index()
       .rename(columns={'level_2': 'metric'}).to_csv(out / 'fig2c_lines.csv', index=False))
    panel_d(costs, radar, metrics, out)
    return pick


# ---- per-cell cache (score_selective.py); also used by figS3_data.py -------------------------------
def load_selective(d: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    s = pd.concat(parts, ignore_index=True)
    for c in ('method', 'fold', 'split', 'dataset', 'label', 'pred'):
        s[c] = s[c].astype(str)
    s['method'] = s['method'].replace(F.ALIAS)
    s = s[~s['method'].isin(F.HIDE)]
    s['regime'] = F.regime(s['split'])
    s['w'] = np.nan_to_num(np.clip(s['efficiency'].to_numpy(np.float64), 0, None))
    return s


def perpert(v: pd.DataFrame) -> dict:
    """Per-perturbation, efficiency-weighted F1 / recall@10 / MRR on one (fold, dataset) view --
    the score_extra definitions (one vote per perturbation present)."""
    v = v[v.w > 0]
    if v.empty:
        return {}
    g = v.groupby('label')
    sup = g.w.sum()
    hit = (v['rank'] < 10) * v.w
    rr = v.w / (v['rank'] + 1.0)
    tp = v.w.where(v.label == v.pred, 0.0).groupby(v.label).sum()
    predw = v.groupby('pred').w.sum().reindex(sup.index).fillna(0.0)
    prec = (tp / predw.replace(0, np.nan)).fillna(0.0)
    rec = tp / sup
    f1 = (2 * prec * rec / (prec + rec).replace(0, np.nan)).fillna(0.0)
    return {'F1': f1.mean(), 'recall@10': (hit.groupby(v.label).sum() / sup).mean(),
            'MRR': (rr.groupby(v.label).sum() / sup).mean()}


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ctx7, small = Path(a.ctx7_dir), Path(a.small_dir)
    extra = {'ctx7': Path(a.extra_ctx7) if a.extra_ctx7 else None,
             'small': Path(a.extra_small) if a.extra_small else None}
    metrics = [a.f1, F.R10, a.mrr]
    costs = F.apply_overrides(pd.read_csv(a.costs), a.cost_overrides)

    panel_palette(ctx7, out)
    w = load_weighted(ctx7, extra['ctx7'])
    for col in (a.f1, a.mrr):
        if col not in w.columns:
            raise SystemExit(f'{col} is not in the ctx7 tables (needs --extra-ctx7?)')
    radar = panel_a(w, metrics, out)
    panel_b(w, metrics, out)
    panel_c(ctx7, out)
    panel_d(costs, radar, metrics, out)
    panel_e([('small', small), ('ctx7', ctx7)], costs, extra, metrics,
            drop=set(a.drop_scaling), n_folds=a.folds, out=out)
    sel = load_selective(Path(a.selective_dir)) if a.selective_dir else None
    if sel is not None:
        panel_a_ext(sel, n_classes(ctx7), w, out)
        panel_c_rho(sel, out)
    if a.expert_train_root and sel is not None:
        timing = expert_train_timing(Path(a.expert_train_root), range(a.folds))
        if a.expert_eval_logs:
            g = expert_gpu_estimate(timing, sel, costs, Path(a.expert_eval_logs), a.folds, a.expert_eval_cpus,
                                    a.infer_frac, out)
            print(f'  [gpu-est] {len(g)} folds: encoding {g.enc_cpu_s.sum():.0f} s CPU -> {g.enc_gpu_s.sum():.0f} s GPU '
                  f'of {g.wall_cpu_s.sum():.0f} s wall; GPU job {g.gpu_h.sum():.2f} GPU-h + {g.cpu_h.sum():.1f} '
                  f'CPU core-h (CPU job: {g.wall_cpu_s.sum() / 3600 * a.expert_eval_cpus:.1f} core-h)')
    if a.pbulk_dir:
        pick = pbulk_panels(Path(a.pbulk_dir), metrics, costs, ctx7, out / 'pbulk', agg=a.pbulk_agg)
        (out / 'pbulk' / 'fig2_palette.csv').write_text((out / 'fig2_palette.csv').read_text())
    meta = dict(vars(a), n_classes={'small': n_classes(small), 'ctx7': n_classes(ctx7)},
                matched_cells_ctx7=int(F.matched(w)[['fold', 'split']].drop_duplicates().shape[0]))
    (out / 'fig2_meta.json').write_text(json.dumps(meta, indent=2, default=str))
    if a.pbulk_dir:
        # same metric column names, so the plot script reads the pooled tables unchanged
        (out / 'pbulk' / 'fig2_meta.json').write_text(json.dumps(dict(meta, level='pbulk', pbulk_agg=pick),
                                                                indent=2, default=str))
    print('wrote', sorted(p.name for p in out.glob('fig2*')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctx7-dir', required=True, help='benchmark collect output, 1,157-class run')
    ap.add_argument('--small-dir', required=True, help='benchmark collect output, 200-class run')
    ap.add_argument('--costs', required=True, help='slurm_costs.csv (size-keyed)')
    ap.add_argument('--cost-overrides', required=True, help='benchmark config/cost_overrides.yaml')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--extra-ctx7', default=None, help='score_extra.py output for ctx7 (optional)')
    ap.add_argument('--extra-small', default=None, help='score_extra.py output for small (optional)')
    ap.add_argument('--mrr', default='weighted_mrr', choices=('weighted_mrr', 'balanced_weighted_mrr'))
    ap.add_argument('--f1', default='weighted_macro_f1',
                    choices=('weighted_macro_f1', 'balanced_weighted_macro_f1'),
                    help='weighted_macro_f1 averages classes by weighted support; the balanced one '
                         'gives every perturbation one vote (score_extra.py)')
    ap.add_argument('--drop-scaling', nargs='*', default=['scgpt', 'gears', 'cellot'],
                    help='left out of E: scGPT changed readout between sizes, GEARS changed '
                         'hardware (CPU -> GPU), CellOT changed code')
    ap.add_argument('--folds', type=int, default=12)
    ap.add_argument('--expert-train-root', default=None,
                    help='dir with ExPert_fold<k>/run_0 (training logs + checkpoints) -> GPU estimate')
    ap.add_argument('--expert-eval-logs', default=None,
                    help='dir of the ExPert benchmark eval logs (run_expert_logits.py stdout) -> encoding time')
    ap.add_argument('--expert-eval-cpus', type=int, default=8, help='CPU cores of an ExPert eval job')
    ap.add_argument('--infer-frac', type=float, default=0.5,
                    help='inference / training time per cell (forward only, no generative step)')
    ap.add_argument('--pbulk-dir', default=None,
                    help='score_pbulk_groups.py output (metrics/ctx7/pbulk_groups): writes <out>/pbulk/ tables '
                         'for the pseudobulk versions of A-D')
    ap.add_argument('--pbulk-agg', default=None, choices=('mean', 'effw'),
                    help="pool cells by plain or efficiency-weighted mean; default: effw only if it scores better")
    ap.add_argument('--selective-dir', default=None,
                    help='score_selective.py per-cell cache (ctx7) -> radar candidates, efficiency rho, GPU estimate')
    main(ap.parse_args())
