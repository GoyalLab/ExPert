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

"""Table S9: the Tahoe-100M drug misses whose ExPert ranking matches the drug's known pharmacology (related to Fig. 4;
Supplementary Note 4). The curated workbook (--source; header on its third row) is copied as is, with one column added,
counted_as_explained (False for --not-counted, the likely sample mix-up). The Tahoe targets it lists are checked against
Tahoe's own annotation (--annotations: fig4 tahoe_pb failing_drugs.csv; --bridge-nodes for the glucocorticoids, which
that table does not list). Written to --out-dir as Table_S9.xlsx + .csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(a):
    c = pd.read_excel(a.source, header=a.header_row).dropna(how='all').dropna(axis=1, how='all')
    c['counted_as_explained'] = ~c.Drug.isin(a.not_counted)
    ann = pd.read_csv(a.annotations)[['drug', 'targets', 'moa_fine']]
    nodes = pd.read_csv(a.bridge_nodes)
    for d in nodes.dropna(subset=['drugs']).itertuples():
        for drug in str(d.drugs).split(';'):
            if drug not in set(ann.drug):
                ann.loc[len(ann)] = (drug, d.gene, d.moa)
    m = c.merge(ann, left_on='Drug', right_on='drug', how='left')
    assert m.targets.notna().all(), f'no Tahoe annotation for {m[m.targets.isna()].Drug.tolist()}'
    norm = lambda s: sorted(x.strip() for x in str(s).replace('(only)', '').split(','))  # noqa: E731
    bad = m[[norm(x) != norm(y) for x, y in zip(m['Tahoe targets'], m.targets)]]
    assert bad.empty, f'targets differ from Tahoe: {bad[["Drug", "Tahoe targets", "targets"]].values}'
    out = Path(a.out_dir)
    c.to_csv(out / 'Table_S9.csv', index=False)
    with pd.ExcelWriter(out / 'Table_S9.xlsx') as w:
        c.to_excel(w, sheet_name='S9 explained misses', index=False)
    print(f'{len(c)} rows, {int(c.counted_as_explained.sum())} counted as explained; targets match Tahoe')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', required=True,
                    help='curated workbook ('
                         'paper/figures/si/tables/data/tahoe_misses_source.xlsx)')
    ap.add_argument('--header-row', type=int, default=2, help='0-based row of the column header')
    ap.add_argument('--not-counted', nargs='*', default=[], help='drugs listed but not counted as explained')
    ap.add_argument('--annotations', required=True, help='fig4 tahoe_pb failing_drugs.csv (drug, targets, moa_fine)')
    ap.add_argument('--bridge-nodes', required=True, help='fig4 tahoe_pb atom_structure_bridge_graph_nodes.csv')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
