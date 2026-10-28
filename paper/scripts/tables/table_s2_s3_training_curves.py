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

"""In-training fit summary for every ablation run: per-cell top-10 recall on TRAIN, VALIDATION and
OOD, final epoch and best epoch, on both readouts.

    atoms (A)  clip_top10_recall_{train,validation,ood_test}   cosine of z to the prototype table
    head  (H)  top10_recall_{train,validation,ood_test}         classifier head

These are the per-cell diagnostics that expose over-/under-fitting; the train/val LOSS ratio does
not (it is dominated by reconstruction). Also kept: the live-macro OOD/val recall used by the
v0.1.x handoff summaries.

Writes one long CSV: run (path below models/), metric, n_evals, last_epoch, final, best, best_epoch.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

TAGS = [f'{p}top10_recall_{s}' for p in ('clip_', '') for s in ('train', 'validation', 'ood_test')]
TAGS += ['ood_test_live_macro_top10_recall_A', 'ood_test_live_macro_top10_recall_H',
         'val_live_macro_top10_recall_A', 'val_live_macro_top10_recall_H', 'epoch']


def summarise(run_dir: str) -> list[dict]:
    """Values keyed by EPOCH, not step: runs resumed before the global_step fix restart the step
    counter in each segment, so a step-keyed union lets an early segment overwrite late epochs.
    Within each event file, step -> epoch comes from that file's own 'epoch' scalar; later files
    (later segments) win for the same epoch."""
    import bisect
    sc: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents*'), recursive=True),
                    key=os.path.getmtime):
        try:
            ea = event_accumulator.EventAccumulator(p, size_guidance={'scalars': 0})
            ea.Reload()
        except Exception:                                   # noqa: BLE001
            continue
        have = set(ea.Tags()['scalars'])
        if 'epoch' not in have:
            continue
        ep = sorted((e.step, int(round(e.value))) for e in ea.Scalars('epoch'))
        steps = [s_ for s_, _ in ep]
        def to_epoch(step):
            i = bisect.bisect_right(steps, step) - 1
            return ep[max(i, 0)][1]
        for t in TAGS:
            if t in have and t != 'epoch':
                sc.setdefault(t, {}).update({to_epoch(e.step): e.value for e in ea.Scalars(t)})
    rows = []
    for t, d in sc.items():
        eps = sorted(d)
        rows.append({'metric': t, 'n_evals': len(eps), 'last_epoch': eps[-1],
                     'final': d[eps[-1]], 'best': max(d.values()),
                     'best_epoch': max(d, key=d.get)})
    return rows


def main(a):
    out = []
    for root in a.roots:
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            name = os.path.basename(d)
            if '_smoke' in name or not os.path.isdir(d):
                continue
            for run in sorted(glob.glob(os.path.join(d, 'run_*'))) or [d]:
                rows = summarise(run)
                label = f"{name}/{os.path.basename(run)}" if run != d else name
                for r in rows:
                    out.append({'root': root, 'run': label, **r})
                if rows:
                    print(f'{label}: {len(rows)} metrics', flush=True)
    df = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f'wrote {len(df)} rows for {df.run.nunique()} runs -> {a.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', required=True, help='model dirs containing <arm>/run_*')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
