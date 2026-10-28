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

"""Fig. 6E / 6F: magnitude-aware per-donor gene-set scores from ONE common baseline.

The within-profile set z (fig6_pmd.py donor_set_z) ranks the bank by the cosine of a donor's shift, so it sees only
the shift's DIRECTION: a reference donor with a small, noisy shift scores as loudly as a patient with a large one, and
each reference donor is measured against the other four only. Here every donor x cell type is measured from one
baseline, the cell-weighted centroid of the five reference donors (non-PMD + undiagnosed, fig6_pmd.CTRL), and
projected on the set's unit mean prototype direction (the Fig. 6C OXPHOS loss score, per set):

    proj(d, set) = (m_d - c) . u_set,   c = sum_r w_r m_r / W over the reference donors r,  u_set = unit(mean P_set)

m_d - c is recovered exactly from the fig6_pmd.py 'ctrl' shifts: a PMD donor's shift is already m_d - c; a reference
donor's leave-one-out shift m_d - c_(-d) equals (m_d - c) / (1 - w_d / W). Projections are linear, so any difference
of donor means is baseline-free; downstream z = (proj - mean of the reference donors) / their SD.

Sets: module:* (--modules), reactome:* (--reactome-gmt, 10-500 in the bank), mitocarta:CI..CV (structural complexes,
fig6_pmd_categories.py pmd_categories.csv), panel:<group> / panel:all PMD genes / panel:CI..CV (PanelApp green genes,
fig6_pmd_panel.py pmd_panel_genes_*.csv), all:mitocarta / all:panel (the Fig. 6E 'all' row: CI-CV of either definition
+ panel CoQ + panel PDH & TCA), gene:<suggested>.

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_donor_mag.py --score-dir ... --bank-npz ... --out ...
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig6_pmd_categories import unit
from fig6_pmd_panel import COND, CTRL, SUGGESTED

logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_donor_mag')
log.setLevel(logging.INFO)


def readout_module(path):
    """expert/utils/readout.py loaded by path (numpy / pandas only; skips the package __init__ and torch)."""
    spec = importlib.util.spec_from_file_location('expert_readout', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(a):
    b = np.load(a.bank_npz, allow_pickle=True)
    keep = b['trained'].astype(bool) if a.bank == 'trained' else np.ones(len(b['labels']), bool)
    labels = [str(x) for x in b['labels'][keep]]
    Pn = unit(b['P'][keep].astype(np.float64))
    pos = {g: i for i, g in enumerate(labels)}
    ro = readout_module(a.readout_py)
    sets = {f'module:{k}': v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({f'reactome:{k}': v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    cat = pd.read_csv(Path(a.categories_dir) / 'pmd_categories.csv').set_index('category')
    cx = {k: [g for g in cat.loc[k, 'genes'].split() if g in pos] for k in ('CI', 'CII', 'CIII', 'CIV', 'CV')}
    sets.update({f'mitocarta:{k}': v for k, v in cx.items()})
    pg = pd.read_csv(Path(a.panel_dir) / f'pmd_panel_genes_{a.bank}.csv')
    pg = pg[pg.model_symbol.isin(pos)].drop_duplicates('model_symbol')
    for g_, x in pg.groupby('group'):
        sets[f'panel:{g_}'] = sorted(x.model_symbol)
    sets['panel:all PMD genes'] = sorted(pg.model_symbol)
    extra = set(sets.get('panel:CoQ', [])) | set(sets.get('panel:PDH & TCA', []))
    sets['all:mitocarta'] = sorted(set().union(*cx.values()) | extra)   # Fig. 6E 'all', structural complexes
    sets['all:panel'] = sorted(set().union(*[sets.get(f'panel:{k}', []) for k in cx]) | extra)   # PanelApp complexes
    sets.update({f'gene:{g}': [g] for g in SUGGESTED.values() if g in pos})
    names = list(sets)
    U = unit(np.stack([Pn[[pos[g] for g in sets[k]]].mean(0) for k in names]))
    log.info('%d sets (%d module, %d reactome); all = %d genes', len(names),
             sum(k.startswith('module:') for k in names), sum(k.startswith('reactome:') for k in names),
             len(sets['all:mitocarta']))
    sh = np.load(Path(a.score_dir) / 'shifts.npz', allow_pickle=True)
    keys = [k.split('|', 2) for k in sh['keys'].astype(str)]   # cell types may contain '|'
    R = sh['R'].astype(np.float64)
    w = pd.read_csv(a.cells, usecols=['donor', 'celltype']) \
        .groupby(['celltype', 'donor']).size()
    out = []
    for ct in sorted({k[2] for k in keys if k[0] == 'ctrl'}):
        prof = {k[1]: R[i] for i, k in enumerate(keys) if k[0] == 'ctrl' and k[2] == ct}
        refs = [d for d in CTRL if d in prof]
        W = sum(w[(ct, d)] for d in refs)
        D = list(prof)
        X = np.stack([prof[d] * (1 - w[(ct, d)] / W) if d in refs else prof[d] for d in D])
        resid = np.linalg.norm(sum(w[(ct, d)] * X[D.index(d)] for d in refs)) / np.linalg.norm(X, axis=1).max()
        if resid > 1e-3:                                          # the reference centroid must sit at the origin
            raise ValueError(f'{ct}: baseline reconstruction off by {resid:.2e} (cell counts != score counts?)')
        proj = X @ U.T
        for j, d in enumerate(D):
            out.append(pd.DataFrame(dict(key=names, donor=d, condition=COND.get(d, '?'), celltype=ct,
                                         n_genes=[len(sets[k]) for k in names], proj=proj[j])))
    T = pd.concat(out, ignore_index=True)
    T[['family', 'set']] = T.key.str.split(':', n=1, expand=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    T.drop(columns='key').to_csv(a.out, index=False)
    log.info('wrote %s (%d rows)', a.out, len(T))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--score-dir', required=True, help='fig6_pmd.py score dir (shifts.npz)')
    ap.add_argument('--cells', required=True, help='per-cell donor / celltype table of the encode (cells.csv.gz or '
                    'obs_cells.csv.gz): the cell weights of the reference centroid')
    ap.add_argument('--bank-npz', required=True)
    ap.add_argument('--bank', default='trained', choices=['trained', 'full'])
    ap.add_argument('--modules', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--readout-py', required=True, help='expert/utils/readout.py (set loaders)')
    ap.add_argument('--categories-dir', required=True, help='fig6_pmd_categories.py out dir')
    ap.add_argument('--panel-dir', required=True, help='fig6_pmd_panel.py out dir')
    ap.add_argument('--out', required=True, help='output .csv.gz')
    main(ap.parse_args())
