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

"""Fig. 2 panel: prototype alignment (PA) vs an MLP classifier head on the same encoder.

Two models trained on the same fold (K562 held out as an entire context), same architecture and
data; they differ in the objective that reads perturbations out of the latent:

    ExPert (PA)          cosine to learned perturbation prototypes (the shipped readout)
    ExPert (Classifier)  an MLP classification head over the same 8,964 trained perturbations

Both are scored on the held-out K562 screens with the same readout script and capped split, at two
levels: per QUERY (one pseudobulked latent per screen x perturbation) and per CELL. For the classifier
the query score is the head applied to the query's mean latent -- the stronger of its two group
constructions (averaging per-cell head scores is in the source table, `--clf-alt`), so the lift is
not inflated by a weak readout.

    metrics   top-1, top-10, and efficiency-weighted top-10 (weight = within-screen percentile of the
              perturbation's knockdown efficiency)
    lift      PA / classifier, with a paired bootstrap that resamples QUERIES (cells of one query are
              not independent), 95% interval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
# SVG text as real <text> objects (editable), not glyph outlines; PDF text as TrueType
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
from matplotlib.patches import Patch  # noqa: E402

EXPERT = '#7ad151'
LAB = {'pa': 'ExPert (PA)', 'clf': 'ExPert (Classifier)'}


def load(path):
    c = pd.read_csv(path)
    c['key'] = c.dataset + '|' + c.perturbation
    return c


def eff_weight(c):
    """within-screen percentile of clipped efficiency (the zero-shot screen's convention)."""
    w = np.zeros(len(c))
    e = c.efficiency.to_numpy(float)
    for d in c.dataset.unique():
        m = (c.dataset == d).to_numpy() & np.isfinite(e)
        o = np.argsort(np.argsort(np.clip(e[m], 0, None), kind='stable'), kind='stable')
        w[m] = o / max(m.sum() - 1, 1)
    return w


METRICS = {'top-1': lambda r: (r == 0).astype(float), 'top-10': lambda r: (r < 10).astype(float)}


def stats(pa, clf, n_boot, rng, level):
    assert (pa.key.to_numpy() == clf.key.to_numpy()).all(), f'{level}: row order differs'
    w = eff_weight(pa)
    grp = pd.factorize(pa.key)[0]
    n_g = grp.max() + 1
    rows = []
    vals = {}
    for m, f in list(METRICS.items()) + [('efficiency-weighted top-10', None)]:
        if f is None:
            a, b, ww = (pa['rank'] < 10) * w, (clf['rank'] < 10) * w, w
        else:
            a, b, ww = f(pa['rank'].to_numpy()), f(clf['rank'].to_numpy()), np.ones(len(w))
        a, b = np.asarray(a, float), np.asarray(b, float)
        # per-query sums so a bootstrap over queries is a weighted resample of these
        sa, sb, sw = (np.bincount(grp, x, n_g) for x in (a, b, ww))
        va, vb = sa.sum() / sw.sum(), sb.sum() / sw.sum()
        bs = []
        for _ in range(n_boot):
            k = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
            bs.append((k @ sa / (k @ sw), k @ sb / (k @ sw)))
        bs = np.array(bs)
        lo_a, hi_a = np.percentile(bs[:, 0], [2.5, 97.5])
        lo_b, hi_b = np.percentile(bs[:, 1], [2.5, 97.5])
        ratio = bs[:, 0] / np.clip(bs[:, 1], 1e-12, None)
        rows.append({'level': level, 'metric': m, 'n_rows': len(pa), 'n_queries': n_g,
                     'pa': va, 'pa_lo': lo_a, 'pa_hi': hi_a, 'clf': vb, 'clf_lo': lo_b, 'clf_hi': hi_b,
                     'lift': va / vb, 'lift_lo': np.percentile(ratio, 2.5),
                     'lift_hi': np.percentile(ratio, 97.5), 'diff': va - vb})
    return rows


def main(a):
    sys.path.insert(0, str(Path(a.vizstyle_root)))
    from src import vizstyle as VS
    VS.apply()
    rng = np.random.default_rng(0)
    d = Path(a.data_dir); d.mkdir(parents=True, exist_ok=True)
    rows = stats(load(a.pa_query), load(a.clf_query), a.n_boot, rng, 'per query')
    rows += stats(load(a.pa_cell), load(a.clf_cell), a.n_boot, rng, 'per cell')
    alt = stats(load(a.pa_query), load(a.clf_alt), a.n_boot, rng, 'per query, classifier per-cell-score mean')
    t = pd.DataFrame(rows)
    pd.concat([t, pd.DataFrame(alt)]).to_csv(d / 'fig2_pa_vs_classifier_source.csv', index=False)
    print(pd.concat([t, pd.DataFrame(alt)])[['level', 'metric', 'pa', 'clf', 'lift', 'lift_lo',
                                               'lift_hi']].round(4).to_string(index=False), flush=True)

    levels = list(dict.fromkeys(t.level))
    fig, axs = plt.subplots(1, len(levels), figsize=(6.2 * len(levels), 5.6), sharey=True,
                            gridspec_kw={'wspace': 0.08})
    for ax, lev in zip(np.atleast_1d(axs), levels):
        q = t[t.level == lev].reset_index(drop=True)
        x = np.arange(len(q)); h = 0.38
        for dx, k, c in ((-h / 2, 'pa', EXPERT), (h / 2, 'clf', VS.MUTED)):
            ax.bar(x + dx, q[k], width=h * 0.95, color=c, zorder=3)
            ax.errorbar(x + dx, q[k], yerr=[q[k] - q[k + '_lo'], q[k + '_hi'] - q[k]], fmt='none',
                        ecolor=VS.INK, lw=1.3, capsize=3, zorder=4)
        top = np.maximum(q.pa_hi, q.clf_hi)
        for xi, r, tp in zip(x, q.itertuples(), top):
            ax.text(xi, tp + 0.012, f'{r.lift:.1f}×', ha='center', va='bottom', color=VS.INK,
                    fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('efficiency-weighted ', 'eff.-weighted\n') for m in q.metric])
        ax.grid(axis='x', visible=False)
        n = int(q.n_rows.iloc[0])
        ax.set_title(f'{lev} (n = {n:,})', loc='left')
    np.atleast_1d(axs)[0].set_ylabel('held-out K562 recall\n(8,964 trained perturbations)')
    ymax = float(np.maximum(t.pa_hi, t.clf_hi).max())
    np.atleast_1d(axs)[0].set_ylim(0, ymax * 1.22)
    np.atleast_1d(axs)[0].legend(handles=[Patch(color=EXPERT, label=LAB['pa']),
                                          Patch(color=VS.MUTED, label=LAB['clf'])],
                                 loc='upper left', ncol=2, bbox_to_anchor=(0, -0.2))
    VS.save(fig, a.out_dir, 'fig2_pa_vs_classifier')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pa-query', required=True, help='ranks csv, PA, one row per query')
    ap.add_argument('--clf-query', required=True, help='ranks csv, classifier on the mean latent')
    ap.add_argument('--clf-alt', required=True, help='ranks csv, classifier per-cell scores averaged')
    ap.add_argument('--pa-cell', required=True, help='ranks csv, PA, one row per cell')
    ap.add_argument('--clf-cell', required=True, help='ranks csv, classifier, one row per cell')
    ap.add_argument('--vizstyle-root', required=True)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
