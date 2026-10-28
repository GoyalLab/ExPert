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

"""Supplementary Fig. S9 plotting data (prototype bridges between drug classes; related to Fig. 4). Reads the output of
fig4_atom_bridges.py (fig4.sbatch STEP=pb_bridges: a kNN graph over the trained prototypes, the shortest path between
the target sets of every pair of Tahoe-100M drugs, and the traversal count of each gene against size- and
popularity-matched random target sets; the same graph rebuilt from each prior). No drug responses are used. Nothing is
recomputed; one CSV per panel in --out-dir (paper/figures/si/figS9/data):

    figS9_bridges.csv          every gene passing the bridge criteria (>= --min-pairs drug pairs routed through it and
                               >= --min-fold the matched-random expectation), with its traversal fold in each prior's
                               graph and the best biological prior (genomic position, the negative control, excluded)
    figS9a_strip.csv           the --n-strip bridges with the largest excess over the null: learned vs best-prior fold
    figS9_graph_nodes.csv      the neighbourhood drawn around each bridge (panels B-D): endpoints = drug targets whose
    figS9_graph_edges.csv      shortest path runs through the bridge, neighbours = their nearest prototypes
    figS9_meta.json            graph and null parameters, counts quoted in the caption, the prior-support threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CONTROL = 'genomic position'


def main(a):
    src, out = Path(a.bridges_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = lambda s: src / f'{a.prefix}_{s}'
    br = pd.read_csv(f('bridges.csv'))
    pr = pd.read_csv(f('bridge_prior_folds.csv'))
    meta = json.loads(f('bridge_meta.json').read_text())
    assert (meta['min_pairs'], meta['min_fold']) == (a.min_pairs, a.min_fold), 'criteria differ from the bridge run'
    bio = [c for c in pr.columns if c not in ('gene', CONTROL)]
    pr['best_prior'] = pr[bio].max(1)
    pr['best_name'] = pr[bio].idxmax(1)
    passing = br[(br.n_pairs >= a.min_pairs) & (br.fold >= a.min_fold)].sort_values('excess', ascending=False)
    allb = passing.merge(pr, on='gene', how='left').reset_index(drop=True)
    allb.insert(0, 'rank', range(1, len(allb) + 1))
    allb.to_csv(out / 'figS9_bridges.csv', index=False)
    # the strip: the top bridges by excess that every prior was scored on (prior folds exist for the exported top only)
    strip = allb.dropna(subset=['best_prior']).head(a.n_strip)
    assert len(strip) == a.n_strip, f'only {len(strip)} bridges carry prior folds'
    strip = strip.assign(prior_backed=strip.best_prior >= a.prior_fold).sort_values('fold', ascending=False)
    strip.to_csv(out / 'figS9a_strip.csv', index=False)
    for s in ('graph_nodes', 'graph_edges'):
        pd.read_csv(f(f'bridge_{s}.csv')).to_csv(out / f'figS9_{s}.csv', index=False)

    backed = strip[strip.prior_backed]
    counts = dict(n_bridges=int(len(allb)), n_bridges_p05=int((allb.p <= 0.05).sum()), n_strip=int(len(strip)),
                  n_prior_backed=int(len(backed)), prior_fold=a.prior_fold,
                  prior_backed_by={c: int((backed[c] >= a.prior_fold).sum()) for c in bio},
                  prior_backed_genes=backed.gene.tolist())
    print(json.dumps(counts, indent=1), flush=True)
    print(strip[['gene', 'n_pairs', 'fold', 'p', 'best_prior', 'best_name']].round(2).to_string(index=False), flush=True)
    (out / 'figS9_meta.json').write_text(json.dumps(dict(
        source=str(src), main=a.main, **counts, **meta), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bridges-dir', required=True, help='fig4 tahoe_pb dir holding the bridge run (STEP=pb_bridges)')
    ap.add_argument('--prefix', required=True, help='bridge run --out-prefix basename (atom_structure)')
    ap.add_argument('--min-pairs', type=int, required=True, help='bridge criterion (must match the run)')
    ap.add_argument('--min-fold', type=float, required=True, help='bridge criterion (must match the run)')
    ap.add_argument('--n-strip', type=int, default=24, help='bridges shown in panel A')
    ap.add_argument('--prior-fold', type=float, default=2.0,
                    help='a prior supports a bridge if it routes >= this fold of pairs through it in its own graph')
    ap.add_argument('--main', nargs=2, default=['PDPK1', 'RALGDS'], help='the two worked bridges (panels B, C)')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
