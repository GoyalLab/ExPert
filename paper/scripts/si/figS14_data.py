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

"""Supplementary Fig. S14 plotting data (PBMC overview: cell types, donors and the OXPHOS-loss score; related to
Fig. 6). Reads the per-cell table of Fig. 6's encode (fig6_pmd.py --stage cells: ExPert latent UMAP of all 28,513 ex vivo
PBMCs, per-cell OXPHOS-loss score) and the query h5ad's SingleR labels (obs only, h5py). One CSV per panel in
--out-dir (paper/figures/si/figS14/data):

    figS14abc_cells.csv     per cell: UMAP (all cells), cell type, donor, donor group (PMD / non-PMD / undiagnosed),
                            uncentred OXPHOS-loss score (kNN-smoothed)
    figS14d_counts.csv      cells per donor x cell type, flagged below --min-cells (the Fig. 6F inclusion rule)
    figS14e_unconv_t.csv    the "Nonclassical T cells" (SingleR main label) split by the SingleR fine label, per donor
    figS14_meta.json        groups, totals, min-cells
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PMD = ['PT54', 'PT43', 'PT67', 'PT46']
NON_PMD = ['PT41', 'PT42']


def group_of(d):
    return 'PMD' if d in PMD else ('non-PMD' if d in NON_PMD else 'undiagnosed')


def obs_col(o, k):
    x = o[k]
    if isinstance(x, h5py.Group):
        return np.asarray(x['categories']).astype(str)[np.asarray(x['codes'])]
    return np.asarray(x).astype(str)


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(a.cells)
    c = c[c.umap1_all.notna()]
    c = c.assign(group=c.donor.map(group_of))[['donor', 'group', 'celltype', 'umap1_all', 'umap2_all', 'oxphos_raw_smooth']]
    c.to_csv(out / 'figS14abc_cells.csv', index=False)
    n = c.groupby(['donor', 'celltype']).size().rename('n_cells').reset_index()
    n['below_min'] = n.n_cells < a.min_cells
    n['group'] = n.donor.map(group_of)
    n.to_csv(out / 'figS14d_counts.csv', index=False)
    with h5py.File(a.h5ad, 'r') as f:
        o = f['obs']
        ob = pd.DataFrame({'donor': obs_col(o, a.donor_key), 'main': obs_col(o, 'singler_labels_main'),
                           'fine': obs_col(o, 'singler_labels_fine')})
    ob = ob[ob.donor.isin(set(c.donor))]
    # the h5ad and the scored table must describe the same cells (the cell-type labels ARE SingleR main)
    chk = ob.groupby(['donor', 'main']).size()
    ref = c.groupby(['donor', 'celltype']).size()
    assert chk.sort_index().equals(ref.rename_axis(['donor', 'main']).sort_index()), 'h5ad labels differ from cells.csv'
    u = ob[ob.main == a.unconv_label].groupby(['donor', 'fine']).size().rename('n_cells').reset_index()
    u.to_csv(out / 'figS14e_unconv_t.csv', index=False)
    meta = dict(cells=a.cells, h5ad=a.h5ad, n_cells=int(len(c)), min_cells=a.min_cells, pmd=PMD, non_pmd=NON_PMD,
                undiagnosed=sorted(set(c.donor) - set(PMD) - set(NON_PMD)), unconv_label=a.unconv_label,
                unconv_fine=u.groupby('fine').n_cells.sum().to_dict(),
                below_min=n[n.below_min][['donor', 'celltype', 'n_cells']].to_dict('records'))
    (out / 'figS14_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells', required=True,
                    help='paper/figures/fig6/data/pmd_mtmt/score/cells.csv.gz')
    ap.add_argument('--h5ad', required=True, help='PBMC query h5ad (SingleR labels in obs)')
    ap.add_argument('--donor-key', default='patient_id')
    ap.add_argument('--unconv-label', default='Nonclassical T cells')
    ap.add_argument('--min-cells', type=int, default=100, help='Fig. 6F: donor x cell type with fewer cells dropped')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
