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

"""Fig. 3a row labels: dataset -> "context - condition", from Fig. 1's own context table so the two
figures name screens the same way (no publication names).

    T-cell donors   pooled per condition: "<donor cell type> - Rest / 8 h / 48 h"
    K562            kept split by screen: "K562 - essential" / "K562 - genome-wide"
    others          the cell line ("RPE1", "HepG2", ...)

Writes a TSV dataset<TAB>label for screen.py panels --group-map.
"""
from __future__ import annotations

import argparse

import pandas as pd

COND = {'Rest': 'Rest', 'Stim8hr': '8 h', 'Stim48hr': '48 h'}
K562 = {'essential': 'essential', 'gwps': 'genome-wide'}


def main(a):
    c = pd.read_parquet(a.cells, columns=['dataset', 'axis', 'context', 'condition'])
    c = c.drop_duplicates('dataset')
    lab = []
    for r in c.itertuples():
        if r.axis == 'donor':
            lab.append(f'{a.donor_celltype} – {COND.get(r.condition, r.condition)}')
        elif r.context == 'K562':
            lab.append(f'K562 – {K562.get(r.dataset.rsplit("_", 1)[-1], r.dataset)}')
        else:
            lab.append(str(r.context))
    c['label'] = lab
    c[['dataset', 'label']].to_csv(a.out, sep='\t', index=False)
    print(c[['dataset', 'label']].to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', required=True, help='Fig. 1 fig1c_cells.parquet')
    ap.add_argument('--donor-celltype', required=True, help='label for the donor-axis screens')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
