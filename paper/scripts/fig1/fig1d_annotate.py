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

"""Fig. 1d per-prototype annotation by NEIGHBOURHOOD enrichment (SAFE-style; Baryshnikova 2016).

Instead of clustering first and naming clusters afterwards (which colours many prototypes by a
term they do not carry), every prototype is tested on its own:

  1. support   prototype g supports term t if g is annotated with t AND its k nearest
               neighbours in the 2048-D prototype space (cosine) contain more members of t than
               expected: hypergeometric P(X >= x | N prototypes, |t| members, k draws). BH over
               every (prototype, term) membership pair tested; supported = q < alpha and x >= min_hits.
  2. selection greedy maximum coverage over terms: repeatedly take a term that adds the most
               prototypes not yet covered -- among terms within --near (80%) of that best gain, the
               SMALLEST gene set, so specific names beat umbrella terms -- until the gain drops
               below --min-gain or --max-terms terms are chosen. Near-duplicate terms add ~nothing
               and are skipped.
  3. assignment each covered prototype takes, among the selected terms it supports, the one with
               the smallest p. Everything else stays unannotated.

Terms: GO BP / GO CC / KEGG / Reactome / CORUM / Hallmark sets of 5-300 prototypes, infection and
disease terms excluded (same 'specific' definition as the cluster labels).

Writes <out>/fig1d_annot_prototypes.parquet (perturbation, term, label, library, p, q, hits) and
<out>/fig1d_annot_terms.csv (selected terms with coverage and local enrichment).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import hypergeom

from expert.analysis.atom_umap import load_atoms
import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig1d_plot import short_label, theme_of
from fig1d_tree import load_libraries


def main(a):
    labels, A, _ = load_atoms(a.model_pt)
    labels = [str(x) for x in labels]
    n = len(labels)
    U = torch.from_numpy(A / np.linalg.norm(A, axis=1, keepdims=True)).float()
    S = U @ U.T
    S.fill_diagonal_(-np.inf)
    nb = S.topk(a.k, dim=1).indices.numpy()
    del S
    Knn = sp.csr_matrix((np.ones(n * a.k, np.float32), (np.repeat(np.arange(n), a.k), nb.ravel())),
                        shape=(n, n))
    libs = load_libraries(labels, Path(a.cache_dir), a.min_size, a.max_size)

    names, libn, cols = [], [], []
    for lib, (terms, M, _) in libs.items():
        keep = [j for j, t in enumerate(terms) if theme_of(t) is not None]
        cols.append(M[:, keep]); names += [terms[j] for j in keep]; libn += [lib] * len(keep)
    M = sp.hstack(cols).tocsr()                           # prototypes x terms (binary)
    size = np.asarray(M.sum(0)).ravel()
    X = (Knn @ M).tocsr()                                 # neighbour members per prototype x term
    Mc = M.tocoo()                                        # test only (g, t) with g in t
    g, t = Mc.row, Mc.col
    x = np.asarray(X[g, t]).ravel()
    p = hypergeom.sf(x - 1, n, size[t] - 1, a.k)          # g itself excluded from the term pool
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    sup = (q < a.alpha) & (x >= a.min_hits)
    print(f'[support] {len(p):,} membership tests; {int(sup.sum()):,} supported pairs over '
          f'{len(np.unique(g[sup])):,} prototypes ({len(np.unique(g[sup])) / n:.1%})', flush=True)
    pairs = pd.DataFrame({'g': g[sup], 't': t[sup], 'p': p[sup], 'q': q[sup], 'x': x[sup]})

    # greedy maximum coverage
    members = pairs.groupby('t').g.apply(lambda s: set(s.tolist()))
    covered, chosen = set(), []
    while len(chosen) < a.max_terms:
        gain = members.apply(lambda s: len(s - covered))
        if gain.max() < a.min_gain:
            break
        # SPECIFICITY PREFERENCE: among terms within --near of the best gain, take the smallest
        # gene set, so 'spliceosome' wins over 'gene expression' at almost no cost in coverage
        near = gain.index[gain >= a.near * gain.max()]
        tbest = min(near, key=lambda tj: (size[tj], -gain[tj]))
        chosen.append(tbest); covered |= members[tbest]
    sel = pairs[pairs.t.isin(chosen)].sort_values('p').drop_duplicates('g')
    term_name = np.asarray(names, dtype=object)
    out = pd.DataFrame({'perturbation': np.asarray(labels)[sel.g.to_numpy()],
                        'term': term_name[sel.t.to_numpy()],
                        'library': np.asarray(libn, dtype=object)[sel.t.to_numpy()],
                        'p': sel.p.to_numpy(), 'q': sel.q.to_numpy(), 'hits': sel.x.to_numpy()})
    out['label'] = [short_label(s) for s in out.term]
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(a.out_dir) / 'fig1d_annot_prototypes.parquet', index=False)
    exp = size / n * a.k
    rows = []
    for rank, tj in enumerate(chosen):
        m = sel.t == tj
        rows.append({'rank': rank + 1, 'term': names[tj], 'label': short_label(names[tj]),
                     'library': libn[tj], 'set_size': int(size[tj]),
                     'supported': len(members[tj]), 'assigned': int(m.sum()),
                     'median_neighbour_hits': float(np.median(sel.x[m])) if m.any() else np.nan,
                     'median_local_fold': float(np.median(sel.x[m] / exp[tj])) if m.any() else np.nan,
                     'share_of_term_supported': len(members[tj]) / size[tj]})
    terms_df = pd.DataFrame(rows)
    terms_df.to_csv(Path(a.out_dir) / 'fig1d_annot_terms.csv', index=False)
    print(f'[select] {len(chosen)} terms cover {len(covered):,} prototypes '
          f'({len(covered) / n:.1%}); assigned {len(out):,}', flush=True)
    pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 45)
    print(terms_df.drop(columns='term').to_string(index=False, float_format=lambda v: f'{v:.3g}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-pt', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--min-hits', type=int, default=3)
    ap.add_argument('--min-size', type=int, default=5)
    ap.add_argument('--max-size', type=int, default=300)
    ap.add_argument('--max-terms', type=int, default=60)
    ap.add_argument('--min-gain', type=int, default=15)
    ap.add_argument('--near', type=float, default=0.8)
    main(ap.parse_args())
