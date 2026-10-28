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

"""Tables S2 and S3: the ablations behind the model design, one part per development stage. Each part is scored on
ONE OOD set, so numbers are comparable within a part, not across:

  Table S2 (full scale, the prototype source and the fold_4 ladders)
  S2A source       LEGACY full scale (old merge DB, own fold_0): where the prototypes come from --
                   frozen prior + learned adapter (CLIP-style) vs random learnable vs free classifier.
  S2B capacity     fold_4 v0.1.2-v0.1.4 ladders  } paired fold_4 OOD probe: K562_essential +
  S2C invariance   fold_4 v0.1.5/5b ladders      } K562_gwps, 10,061 pseudobulk groups (<=25 cells)
  S2D prior atoms  fold_4 v0.1.5 PA - v0.1.7     } vs the 8,964-atom panel (ood_readout_probe)

  Table S3 (the training objective)
  S3A objective    v0.1.0 + v0.1.1 handoff, fold_0. In-training OOD top-10 on the live held-out contexts
                   (HepG2 cell line + Marson D2 donor), each arm on the readout it TRAINS
                   (classifier head H for CE, cosine to learned atoms A for proxy-anchor arms).
  S3B init         SMALL scale (200-perturbation DB): prototype table initialised from the prior
                   text embeddings and fine-tuned by PA, vs random init. Only tested at small scale.
  S3C / S3D        small-scale objective swaps from the arm survey: CLIP-style prototypes vs classifier head
                   (legacy DB), proxy anchor vs classifier head (current DB, fold_4)

Columns: what changed vs the arm's baseline (from the run configs), scores, the top-10 change vs
the baseline, and whether it exceeds the paired resolution (+-0.0040, 2 sd). v0.1.7 arms are all
compared to BASE (their common reference; `parent` = the arm each was copied from). `carried_forward`
marks arms that became the baseline of a later stage (in S2D: the selection path to the recipe).

Inputs: the arm inventory (configs, baselines) and the long-format score table, plus the v0.1.1
in-training summary log for S3A. Writes Table_S2.xlsx / Table_S3.xlsx (one sheet per part) and one CSV per part
(Table_S2A.csv ...) to --out-dir.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE = {'FULLOPEN3'}        # user decision: not part of the reported ablations
# Every v0.1.7 arm is BASE (zero-shot machinery, no pull) plus switches, so all of them -- and the
# fold-4 replicate of the shipped recipe -- are compared to BASE. The arm each was COPIED from in the
# config generators is kept as `parent`; the change column is the real hparams diff vs BASE.
COMMON_BASE = 'BASE'
REBASE_LADDERS = lambda l: l.startswith('v0.1.7') or l.startswith('12-fold benchmark')
# FULLOPEN27's recorded parent is the excluded FULLOPEN3; its real parent is the 80-epoch recipe
PARENT_OVERRIDE = {'FULLOPEN27': 'ExPert_fold4'}
DESC_OVERRIDE = {'FULLOPEN27': 'pull ALL atoms (soft gate, gamma 0) at weight 27; parent is the '
                               'gated recipe at the same 80 epochs'}
# the path that produced the shipped recipe (S2D)
SELECTION_PATH = ['BASE', 'PULL', 'PULLW15', 'NQ15', 'GRAPH0', 'EFFGATE3']
# derived from class_emb_extra_paths (tensors / names), so redundant in a diff
_DIFF_SKIP = re.compile(r'extra_cls_emb_kwargs\.')
_PREFIX = re.compile(r'^(model_params\.|train_params\.|plan_params\.anneal_schedules\.|plan_params\.)')


def _flat(d: dict, p: str = '') -> dict:
    o = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            o.update(_flat(v, f'{p}{k}.'))
        else:
            o[p + k] = v
    return o


def load_hparams(root: Path, arm: str) -> dict | None:
    import yaml
    p = root / arm / 'run_0' / 'config' / 'ExPert_v0' / 'hparams.yaml'
    if not p.exists():
        return None
    h = _flat(yaml.safe_load(p.read_text()))
    if 'model_params.class_emb_extra_paths' in h and isinstance(h['model_params.class_emb_extra_paths'], list):
        h['model_params.class_emb_extra_paths'] = [Path(str(x)).name for x in
                                                   h['model_params.class_emb_extra_paths']]
    return {k: v for k, v in h.items() if not _DIFF_SKIP.search(k)}


def hparams_diff(a: dict, b: dict) -> str:
    """'key: base -> arm' for every differing key, section prefixes trimmed."""
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, '<unset>'), b.get(k, '<unset>')
        if va != vb:
            out.append(f'{_PREFIX.sub("", k)}: {va} -> {vb}')
    return ' | '.join(out)
RES = 0.0040
TRAIN_FLAT = 0.005           # |delta train recall| below this counts as 'same fit'


def fit_columns(curves: pd.DataFrame, run: str, head: bool) -> dict:
    """Final-epoch per-cell top-10 recall on train / val / OOD from the training logs, on the
    readout the arm trains (head for classifier arms, atoms otherwise)."""
    pre = '' if head else 'clip_'
    c = curves[curves.run == run].set_index('metric')
    g = lambda m, k='final': float(c.loc[m, k]) if m in c.index else np.nan
    tr, va, oo = (g(f'{pre}top10_recall_{x}') for x in ('train', 'validation', 'ood_test'))
    return {'train_top10_cell': tr, 'val_top10_cell': va, 'ood_top10_cell_intraining': oo,
            'train_over_val': tr / va if va and not np.isnan(va) else np.nan,
            'last_epoch_logged': g(f'{pre}top10_recall_train', 'last_epoch')}


def diagnose(d_ood: float, d_train: float) -> str:
    """Only for arms that LOSE beyond resolution, judged on the per-cell train fit vs the baseline:
    train up -> overfit; train down -> underfit; within +-TRAIN_FLAT -> equal fit, worse transfer."""
    if np.isnan(d_ood) or np.isnan(d_train) or d_ood >= -RES:
        return ''
    if d_train > TRAIN_FLAT:
        return 'overfit (train up, OOD down)'
    if d_train < -TRAIN_FLAT:
        return 'underfit (train and OOD down)'
    return 'equal fit, worse transfer'                 # paired 2-sd resolution of fold_4 top-10
STAGES = {
    'S2B_capacity': ['v0.1.2 PA ladder', 'v0.1.3 PA ladder', 'v0.1.4 capacity ladder'],
    'S2C_invariance': ['v0.1.5 invariance ladder', 'v0.1.5b invariance re-weight'],
    'S2D_prior_atoms': ['v0.1.5 PA (prior-atom pull)', 'v0.1.6 PAGW (agreement gate)',
                         'v0.1.7 soft-pull ladder', 'v0.1.7 pull dose ladder',
                         'v0.1.7 evidence arms', 'v0.1.7 graph-map arms',
                         'v0.1.7 gate arms (shipped recipe)', 'v0.1.7 stopping-point arms',
                         'v0.1.7 gate vs pull-all', 'v0.1.7 head vs PA', 'v0.1.7 resume/LR arms',
                         '12-fold benchmark (EFFGATE3 recipe)'],
}
METRICS = ['top1', 'top10', 'w_top10', 'rank_med', 'w_rankmed', 'top10_essential', 'top10_gwps']
HANDOFF = {   # run label -> (arm, change vs that version's PA baseline, readout)
    'pa/run_0': ('v0.1.0 PA (proxy anchor)', 'baseline: proxy anchor 200 on learned atoms, latent 512',
                 'atoms (A)'),
    'ce/run_0': ('v0.1.0 CE (classifier)', 'proxy_anchor_weight 200 -> 0, cls_head_weight 0 -> 200',
                 'head (H)'),
    'pa_wide/run_0': ('v0.1.0 PA, latent 768', 'n_latent = n_shared 512 -> 768', 'atoms (A)'),
    'v2_ce/run_0': ('CE (classifier)', 'proxy_anchor_weight 200 -> 0, cls_head_weight 0 -> 200', 'head (H)'),
    'v2_pa/run_0': ('PA (proxy anchor)', 'baseline: proxy anchor 200 on learned atoms, latent 768', 'atoms (A)'),
    'v2_pa_w/run_0': ('PA weight 400', 'proxy_anchor_weight 200 -> 400', 'atoms (A)'),
    'v2_pa_w/run_1': ('PA weight 1000', 'proxy_anchor_weight 200 -> 1000', 'atoms (A)'),
    'v2_pa_w/run_2': ('PA weight 5000', 'proxy_anchor_weight 200 -> 5000', 'atoms (A)'),
    'v2_pa_selbo/run_0': ('PA, normalised ELBO', 'scale_elbo false -> true, proxy_anchor_weight 200 -> 4',
                          'atoms (A)'),
    'v2_pa_depmap/run_0': ('PA, DepMap-only margin prior', 'relax priors text+DepMap -> DepMap', 'atoms (A)'),
    'v2_pa_textp/run_0': ('PA, text-only margin prior', 'relax priors text+DepMap -> text', 'atoms (A)'),
    'v2_pa_relax/run_0': ('PA, relax_k 450', 'proxy_anchor_relax_k 200 -> 450', 'atoms (A)'),
    'v2_pa_relax/run_1': ('PA, relax_k 900', 'proxy_anchor_relax_k 200 -> 900', 'atoms (A)'),
}


def stage1(curves: pd.DataFrame) -> pd.DataFrame:
    """v0.1.0 / v0.1.1 handoff pairs, fold_0, from the epoch-keyed training logs. OOD/val = live-macro
    top-10 (held-out HepG2 + Marson D2) on the readout each arm trains; train/val per-cell top-10
    on the same readout for the fit diagnosis. Final epoch (49)."""
    c = curves.set_index(['run', 'metric'])
    g = lambda run, m: float(c.loc[(run, m), 'final']) if (run, m) in c.index else np.nan
    out = []
    for run, (arm, change, ro) in HANDOFF.items():
        k = 'H' if ro.startswith('head') else 'A'
        pre = '' if k == 'H' else 'clip_'
        ver = 'v0.1.0' if not run.startswith('v2_') else 'v0.1.1'
        tr, vc = g(run, f'{pre}top10_recall_train'), g(run, f'{pre}top10_recall_validation')
        out.append({'stage': 'S3A_objective', 'version': ver, 'arm': arm, 'run': run,
                    'baseline': 'v0.1.0 PA (proxy anchor)' if ver == 'v0.1.0' else 'PA (proxy anchor)',
                    'change_vs_baseline': change, 'readout': ro,
                    'ood_top10': g(run, f'ood_test_live_macro_top10_recall_{k}'),
                    'val_top10': g(run, f'val_live_macro_top10_recall_{k}'),
                    'train_top10_cell': tr, 'val_top10_cell': vc,
                    'train_over_val': tr / vc if vc else np.nan,
                    'last_epoch': c.loc[(run, f'{pre}top10_recall_train'), 'last_epoch']
                    if (run, f'{pre}top10_recall_train') in c.index else np.nan})
    df = pd.DataFrame(out)
    by = df.set_index('arm')
    df['delta_ood_top10_vs_baseline'] = [round(o - by.loc[b, 'ood_top10'], 4) if a != b else np.nan
                                         for a, b, o in zip(df.arm, df.baseline, df.ood_top10)]
    df['delta_train_vs_baseline'] = [round(t - by.loc[b, 'train_top10_cell'], 4) if a != b else np.nan
                                     for a, b, t in zip(df.arm, df.baseline, df.train_top10_cell)]
    # CE and PA train on different readouts, so their train fit is not comparable -> no diagnosis
    df['fit_diagnosis'] = [diagnose(o, t) if 'classifier' not in a else ''
                           for a, o, t in zip(df.arm, df.delta_ood_top10_vs_baseline,
                                              df.delta_train_vs_baseline)]
    df['note'] = ''
    df.loc[df.arm.str.contains('relax_k'), 'note'] = ('VOID: the response graph was lost on resume, '
                                                     'so the relax_k change never took effect')
    df.loc[df.arm.str.contains('classifier'), 'note'] = (
        'scored on its head (its atoms are untrained); train fit is on a different readout from PA, '
        'so no over/underfit call')
    return df


SMALL = {   # 200-perturbation ladder (v0.1.3 small): arm -> (change vs s3_base, description)
    's3_base': ('baseline: PA, atoms randomly initialised (randn * 0.02), effect-size weighting on',
                'random-init prototype table'),
    's3_ti': ('extra_cls_emb_kwargs.atom_init: random -> text (full-rank PCA of the qwen3 text '
              'prior, rescaled to the random-init norm)',
              'CLIP-style: prototype table initialised FROM the prior embeddings, fine-tuned by PA'),
    's3_base_ne': ('use_efficiency_pa_weight: true -> false',
                   'effect-size weighting of the PA loss off'),
}


LEGACY = {   # legacy full-scale v3 ladder (old merge DB): arm -> (prototype source, change vs baseline)
    'full_v3_baseline': ('CLIP-style: frozen PRIOR (text) prototypes + learned adapter; the prior is the '
                         'starting point, the adapter learns the rest',
                         'baseline: CLIP alignment (weight 4) of z to adapter(text prior)'),
    'full_v3_classifier': ('free classifier head (no prototypes, no alignment)',
                           'use_cls_head true, cls_head_weight 1; alignment, anchor, centre and '
                           'contrastive weights -> 0; adapter regularisers off'),
}


def stage0(scan_log: Path, ckpt_root: Path) -> pd.DataFrame:
    """Legacy full-scale prototype-source ladder. OOD/val = in-training metrics, MAX over evaluations
    (from the 2026-08-17 tfevents scan; the tfevents were deleted afterwards); final val top-10 and
    epochs trained from the checkpoint filenames, which record the monitored val metric."""
    rows = []
    for line in scan_log.read_text().splitlines():
        p = line.split()
        if p and p[0] in LEGACY and len(p) == 8:      # random-init CLIP arm deliberately omitted
            arm = p[0]
            ck = sorted((ckpt_root / arm / 'run_0/config/ExPert_v0/checkpoints').glob('epoch=*'),
                        key=lambda q: int(re.search(r'epoch=(\d+)', q.name).group(1)))
            last = ck[-1].name if ck else ''
            ep = int(re.search(r'epoch=(\d+)', last).group(1)) if last else np.nan
            vf = float(re.search(r'=([\d.]+)$', last).group(1)) if last else np.nan
            rows.append({'stage': 'S2A_prototype_source_legacy', 'arm': arm,
                         'prototype_source': LEGACY[arm][0], 'baseline': 'full_v3_baseline',
                         'change_vs_baseline': LEGACY[arm][1],
                         'ood_top1_best': float(p[5]), 'ood_top10_best': float(p[7]),
                         'val_top10_best': float(p[6]), 'val_top10_last_ckpt': vf,
                         'last_epoch': ep})
    df = pd.DataFrame(rows)
    base = df.set_index('arm').loc['full_v3_baseline', 'ood_top10_best']
    df['delta_ood_top10_vs_baseline'] = (df.ood_top10_best - base).round(4)
    df.loc[df.arm == 'full_v3_baseline', 'delta_ood_top10_vs_baseline'] = np.nan
    df['note'] = ('legacy database (merge 86ecffa), its own fold_0 split; in-training metrics, best '
                  'over evaluations; arms ran to different epochs (see last_epoch)')
    return df


def stage_small(log: Path) -> pd.DataFrame:
    """Step-matched in-training OOD top-10 of the small ladder (common epoch across arms)."""
    rows = []
    for line in log.read_text().splitlines():
        m = re.match(r'^\s+(s3_\w+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$', line)
        if m and m.group(1) in SMALL:
            arm = m.group(1)
            rows.append({'stage': 'S3B_prototype_init_small', 'arm': arm, 'baseline': 's3_base',
                         'change_vs_baseline': SMALL[arm][0], 'description': SMALL[arm][1],
                         'readout': 'atoms (A)', 'ood_top10_at_common_epoch': float(m.group(2)),
                         'ood_top10_best': float(m.group(3)), 'best_epoch': int(m.group(4)),
                         'val_top10_at_common_epoch': float(m.group(5))})
    df = pd.DataFrame(rows).drop_duplicates('arm')
    base = df.set_index('arm').loc['s3_base']
    df['delta_ood_top10_best_vs_baseline'] = (df.ood_top10_best - base.ood_top10_best).round(4)
    df.loc[df.arm == 's3_base', 'delta_ood_top10_best_vs_baseline'] = np.nan
    df['note'] = ('200-perturbation DB (small scale), 25 step-matched epochs; run-to-run sd ~0.0045. '
                  'No full-scale prior-initialised arm was trained.')
    return df


# Small-scale objective comparisons recovered from the arm survey (configs diffed: each pair is a
# single objective swap; mlpA_cdan / s_c0head have configs IDENTICAL to rep_base_r1 / rep_cls_r1).
# Each arm is scored on the readout it trains: CLIP / PA = cosine to the prior / atoms, CE = head.
SMALL_OBJ = {
    'S3C_clip_vs_head_small': [
        ('CLIP vs head, replicated (legacy DB, fold_2)', 'fold_2/rep_base_r1',
         ['fold_2/rep_base_r1', 'fold_2/rep_base_r2', 'fold_2/mlpA_cdan',
          'fold_2/rep_cls_r1', 'fold_2/rep_cls_r2', 'fold_2/s_c0head']),
        ('CLIP vs head, no CDAN (legacy DB, fold_2)', 'fold_2/mlpA_linadapter',
         ['fold_2/mlpA_linadapter', 'fold_2/mlpA_clshead']),
        ('CLIP vs head, donor + cell-line holdout (legacy DB, v1 fold_0)', 'v1/s_e9_v1_align10',
         ['v1/s_e9_v1_align10', 'v1/s_e9_v1_ctrlcenter']),
    ],
    'S3D_pa_vs_head_small_fold4': [
        ('PA vs classifier head (current DB, fold_4)', 'PA_F4',
         ['PA_F4', 'CLS20_F4', 'CLS200_F4', 'CLSMLP20_F4', 'CLSMLP200_F4', 'GRAPH0_F4']),
    ],
}
SMALL_OBJ_LABEL = {
    'CLIP(None)': 'CLIP-style: frozen text prior + linear adapter, cosine readout',
    'CE_head': 'classifier head (CE)', 'PA_rand-init': 'proxy-anchor, random-init atoms',
}


def stage_small_objective(survey: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = survey.set_index('arm')
    out = {}
    for table, ladders in SMALL_OBJ.items():
        rows = []
        for ladder, base, arms in ladders:
            b = s.loc[base]
            for arm in arms:
                r = s.loc[arm]
                is_base = arm == base
                same_obj = r.family == b.family
                rows.append({
                    'ladder': ladder, 'arm': arm.split('/')[-1], 'db': r.db, 'split': r.split,
                    'epochs': int(r.max_epochs), 'scored_epoch': r.scored_epoch,
                    'objective': SMALL_OBJ_LABEL.get(r.family, r.family),
                    'baseline': base.split('/')[-1],
                    'change_vs_baseline': '' if is_base else (
                        'replicate (identical config)' if same_obj and r.varies.endswith('replicate')
                        else r.varies),
                    'readout': r.metric,
                    'train_top10_cell': r.train_top10, 'val_top10_cell': r.val_top10,
                    'ood_top10_cell': r.ood_top10,
                    'delta_ood_top10_vs_baseline': np.nan if is_base else round(r.ood_top10 - b.ood_top10, 4),
                    'delta_train_vs_baseline': np.nan if is_base else round(r.train_top10 - b.train_top10, 4),
                    'train_over_val': round(r.train_top10 / r.val_top10, 2),
                    'posthoc_predict_ood_r10': r.get('posthoc_predict_ood_r10', np.nan),
                    'score_source': r.score_source,
                })
            if 'replicated' in ladder:
                # replicate means: the spread between identical configs is the noise floor
                df = pd.DataFrame(rows)
                cur = df[df.ladder == ladder]
                for obj, g in cur.groupby('objective', sort=False):
                    rows.append({'ladder': ladder, 'arm': f'mean (n={len(g)})', 'objective': obj,
                                 'db': g.db.iloc[0], 'split': g.split.iloc[0], 'epochs': g.epochs.iloc[0],
                                 'train_top10_cell': g.train_top10_cell.mean(),
                                 'val_top10_cell': g.val_top10_cell.mean(),
                                 'ood_top10_cell': g.ood_top10_cell.mean(),
                                 'ood_top10_sd': g.ood_top10_cell.std(ddof=1),
                                 'train_over_val': round(g.train_top10_cell.mean() / g.val_top10_cell.mean(), 2)})
        df = pd.DataFrame(rows)
        df['note'] = ''
        df.loc[df.split == 'fold_2', 'note'] = ('fold_2 keeps the held-out donor\'s other condition in '
                                                'TRAIN: a cell-context holdout, not a clean donor holdout')
        df.loc[df.arm == 'GRAPH0_F4', 'note'] = 'not an objective swap: PA with the graph prior map + nqxnds evidence'
        out[table] = df
    return out


# sheet titles (Excel caps them at 31 characters)
SHEET = {'S2A': 'S2A prototype source', 'S2B': 'S2B capacity', 'S2C': 'S2C invariance', 'S2D': 'S2D prior pull',
         'S3A': 'S3A objective', 'S3B': 'S3B prototype init', 'S3C': 'S3C prototype vs head',
         'S3D': 'S3D proxy anchor vs head'}


def write_tables(parts: dict, out: Path):
    """Table_S2.xlsx / Table_S3.xlsx, one sheet per part, plus one CSV per part (Table_S2A.csv ...)."""
    for table in ('S2', 'S3'):
        keys = sorted(k for k in parts if k.startswith(table))
        with pd.ExcelWriter(out / f'Table_{table}.xlsx') as w:
            for k in keys:
                part = k.split('_')[0]
                parts[k].to_excel(w, sheet_name=SHEET[part], index=False)
                parts[k].to_csv(out / f'Table_{part}.csv', index=False)
        print(f'Table {table}: {", ".join(k.split("_")[0] for k in keys)}')


def main(a):
    inv = pd.read_csv(a.inventory)
    sc = pd.read_csv(a.scores)
    curves = pd.read_csv(a.curves)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = {}                                     # stage -> table, written per Table at the end

    s0 = stage0(Path(a.legacy_scan_log), Path(a.legacy_models))
    parts['S2A_prototype_source_legacy'] = s0
    print(s0[['arm', 'ood_top1_best', 'ood_top10_best', 'val_top10_best', 'val_top10_last_ckpt',
              'last_epoch', 'delta_ood_top10_vs_baseline']].to_string(index=False))
    s1 = stage1(curves)
    parts['S3A_objective'] = s1
    s1b = stage_small(Path(a.small_log))
    parts['S3B_prototype_init_small'] = s1b
    for name, df in stage_small_objective(pd.read_csv(a.small_survey)).items():
        parts[name] = df
        print(df[['ladder', 'arm', 'objective', 'train_top10_cell', 'val_top10_cell', 'ood_top10_cell',
                  'delta_ood_top10_vs_baseline', 'delta_train_vs_baseline']].round(4).to_string(index=False))

    f4 = sc[(sc.fold == 4) & ~sc.arm.str.contains('@')]
    wide = f4.pivot_table(index='arm', columns='metric', values='value', aggfunc='first')
    # baselines that later stages build on = carried forward
    later = {}
    order = [l for ls in STAGES.values() for l in ls]
    for _, r in inv.iterrows():
        if r.ladder in order:
            later.setdefault(r.baseline, set()).add(order.index(r.ladder))
    hp_root = Path(a.hparams_root)
    base_hp = load_hparams(hp_root, COMMON_BASE)
    assert base_hp is not None, f'no hparams for {COMMON_BASE} under {hp_root}'
    allrows = []
    for stage, ladders in STAGES.items():
        sub = inv[inv.ladder.isin(ladders)].copy()
        # the other folds of the 12-fold benchmark are scored on their own query sets
        sub = sub[~sub.arm.str.match(r'ExPert_fold(?!4$)')]
        sub = sub[~sub.arm.isin(EXCLUDE)]
        sub['ladder_rank'] = sub.ladder.map({l: i for i, l in enumerate(ladders)})
        sub = sub.sort_values(['ladder_rank', 'arm'])
        rows = []
        for _, r in sub.iterrows():
            r = r.copy()
            parent = PARENT_OVERRIDE.get(r.arm, r.baseline)
            r['description'] = DESC_OVERRIDE.get(r.arm, r.description)
            if REBASE_LADDERS(r.ladder) and r.arm != COMMON_BASE:
                r['baseline'] = COMMON_BASE
                ha = load_hparams(hp_root, r.arm)
                r['diff_vs_baseline'] = (hparams_diff(base_hp, ha) if ha is not None
                                         else f'<no hparams; vs parent {parent}: {r.diff_vs_baseline}>')
            row = {'stage': stage, 'ladder': r.ladder, 'arm': r.arm, 'baseline': r.baseline,
                   'parent': parent,
                   'change_vs_baseline': r.diff_vs_baseline, 'description': r.description,
                   'readout': r.readout_family, 'n_latent': r.n_latent, 'n_hidden': r.n_hidden,
                   'epochs': r.max_epochs, 'status': r.status}
            for m in METRICS:
                row[m] = wide.loc[r.arm, m] if r.arm in wide.index and m in wide.columns else np.nan
            base = str(r.baseline).split(' ')[0]
            if base in wide.index and 'top10' in wide.columns and not np.isnan(row['top10']):
                d = row['top10'] - wide.loc[base, 'top10']
                row['delta_top10_vs_baseline'] = round(d, 4)
                row['beyond_resolution'] = bool(abs(d) > RES)
            else:
                row['delta_top10_vs_baseline'] = np.nan
                row['beyond_resolution'] = np.nan
            row['scored'] = not np.isnan(row['top10'])
            if curves is not None:
                row.update(fit_columns(curves, f'{r.arm}/run_0', head=r.arm == 'HEAD200'))
            row['carried_forward'] = r.arm in later and max(later[r.arm]) > order.index(r.ladder)
            rows.append(row)
        df = pd.DataFrame(rows)
        if curves is not None:
            # baselines can sit in another stage: look them up in the full fit table
            fit_all = {arm: fit_columns(curves, f'{arm}/run_0', head=arm == 'HEAD200')
                       for arm in set(df.baseline.astype(str).str.split(' ').str[0])}
            df['delta_train_vs_baseline'] = [
                round(t - fit_all[str(b).split(' ')[0]]['train_top10_cell'], 4)
                for t, b in zip(df.train_top10_cell, df.baseline)]
            df['delta_val_vs_baseline'] = [
                round(v - fit_all[str(b).split(' ')[0]]['val_top10_cell'], 4)
                for v, b in zip(df.val_top10_cell, df.baseline)]
            df['fit_diagnosis'] = [diagnose(o, t) for o, t in
                                   zip(df.delta_top10_vs_baseline, df.delta_train_vs_baseline)]
            # train fit is only comparable on the same readout: head-trained arms vs atom baselines are not
            df.loc[df.arm == 'HEAD200', ['delta_train_vs_baseline', 'delta_val_vs_baseline',
                                         'fit_diagnosis']] = [np.nan, np.nan, '']
        if stage == 'S2D_prior_atoms':
            # v0.1.7 arms all share BASE, so 'carried forward' = the selection path to the recipe
            v017 = df.ladder.map(REBASE_LADDERS)
            df.loc[v017, 'carried_forward'] = df.loc[v017, 'arm'].isin(SELECTION_PATH)
            df.loc[df.arm == 'HEAD200', 'note'] = ('classifier head on the final setup; scored on '
                                                   'its head (atoms untrained). Also drops the prior pull')
            df.loc[df.arm.isin(['ExPert_fold4', 'EFFGATE3_R20CONST']), 'note'] = (
                'replicate of the final recipe at 80 epochs; the spread between replicates is the noise floor')
        df.loc[df.arm == 'G16', 'note'] = ('16k-gene panel DB; scored on its own capped fold_4 split, '
                                           'not paired with the other rows')
        df.loc[~df.scored, 'note'] = df.loc[~df.scored].get('note', pd.Series(dtype=str)).fillna('') + \
            ' not scored (no final model / unfinished)'
        parts[stage] = df
        allrows.append(df)
        print(f'{stage}: {len(df)} arms, {int(df.scored.sum())} scored')
    write_tables(parts, out)
    pd.set_option('display.width', 250)
    print(s1[['arm', 'readout', 'ood_top10', 'train_top10_cell', 'val_top10_cell',
              'delta_ood_top10_vs_baseline', 'delta_train_vs_baseline', 'fit_diagnosis']].to_string(index=False))
    print(s1b[['arm', 'ood_top10_at_common_epoch', 'ood_top10_best', 'delta_ood_top10_best_vs_baseline']].to_string(index=False))
    for df in allrows:
        print(df[['arm', 'baseline', 'top10', 'delta_top10_vs_baseline', 'train_top10_cell',
                  'val_top10_cell', 'delta_train_vs_baseline', 'fit_diagnosis',
                  'carried_forward']].round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inventory', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--small-log', required=True, help='small s3 ladder step-matched readout log')
    ap.add_argument('--legacy-scan-log', required=True, help='2026-08-17 tfevents scan of the legacy DB')
    ap.add_argument('--legacy-models', required=True, help='legacy merge DB models dir (checkpoint names)')
    ap.add_argument('--curves', required=True,
                    help='ablations_training_curves.csv (epoch-keyed train/val/OOD from the logs)')
    ap.add_argument('--small-survey', required=True,
                    help='ablations_small_arm_survey.csv (per-arm configs + final-epoch per-cell scores)')
    ap.add_argument('--hparams-root', required=True,
                    help='full-scale models dir (<arm>/run_0/config/ExPert_v0/hparams.yaml)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
