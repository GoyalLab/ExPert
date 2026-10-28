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

"""Fig. 6I (category version): which primary-mitochondrial-disease category does each patient's profile land in?

Source table: the MitoCarta3.0 human inventory page (the .xls host refuses connections from the cluster).
The suggested lesion genes are candidates, not ground truth, so each patient is scored against EVERY PMD gene
category and the suggested gene's category is only highlighted. Categories = MitoCarta3.0 MitoPathways (Rath et al.
2021) at the level the PMD gene reviews group disease genes (Frazier, Thorburn & Compton 2019): each OXPHOS complex
(subunits + assembly factors), CoQ, Fe-S, mtDNA maintenance, mitochondrial translation (ribosome, tRNA synthetases,
translation factors), pyruvate / PDH, TCA, protein import, dynamics, cardiolipin. --categories maps a short label to
one or more MitoPathway names ('|'-separated).

Profiles: the PBMC readout shifts (fig6_pmd.py score: a PMD donor vs the pooled non-PMD cells, a non-PMD donor vs the
other non-PMD donors) and, with --bulk-dir, the sorted-bulk samples (each vs its cell type x stim controls). Score =
Fig. 6 set z of the category's genes in the trained bank (loss tail). p (within the profile) = upper tail of the set
z; the reference donors / control samples scored the same way are kept for display.

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_pmd_categories.py --mitocarta ... --score-dir ... --out-dir ...
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_pmd_categories')
log.setLevel(logging.INFO)

COND = {'PT43': 'PMD', 'PT46': 'PMD', 'PT54': 'PMD', 'PT67': 'PMD', 'PT41': 'non-PMD', 'PT42': 'non-PMD',
        'PT38': 'undiagnosed', 'PT50': 'undiagnosed', 'PT55': 'undiagnosed'}
# short label: MitoCarta3.0 MitoPathway path(s), '|'-separated; a gene in any sub-pathway of a path belongs to it
CATEGORIES = [
    ('CI', 'OXPHOS > Complex I'), ('CII', 'OXPHOS > Complex II'), ('CIII', 'OXPHOS > Complex III'),
    ('CIV', 'OXPHOS > Complex IV'), ('CV', 'OXPHOS > Complex V'),
    ('CoQ', 'Metabolism > Metals and cofactors > Coenzyme Q metabolism'),
    ('Fe-S', 'Metabolism > Metals and cofactors > Fe-S cluster biosynthesis'),
    ('mtDNA maintenance', 'Mitochondrial central dogma > mtDNA maintenance'),
    ('mtRNA metabolism', 'Mitochondrial central dogma > mtRNA metabolism'),
    ('mito-ribosome', 'Mitochondrial central dogma > Translation > Mitochondrial ribosome|'
                      'Mitochondrial central dogma > Translation > Mitochondrial ribosome assembly'),
    ('mt-tRNA synthetases', 'Mitochondrial central dogma > Translation > mt-tRNA synthetases'),
    ('translation factors', 'Mitochondrial central dogma > Translation > Translation factors'),
    ('pyruvate (PDH)', 'Metabolism > Carbohydrate metabolism > Pyruvate metabolism'),
    ('TCA & anaplerosis', 'Metabolism > Carbohydrate metabolism > TCA cycle|'
                          'Metabolism > Carbohydrate metabolism > TCA-associated'),
    ('protein import', 'Protein import, sorting and homeostasis > Protein import and sorting'),
    ('cardiolipin', 'Metabolism > Lipid metabolism > Cardiolipin synthesis')]
# the suggested lesion gene's category (highlight only)
LESION_CAT = {'NDUFS8': 'CI', 'NDUFA10': 'CI', 'BCS1L': 'CIII', 'SURF1': 'CIV', 'MT-ATP6': 'CV',
              'PC': 'TCA & anaplerosis'}


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def load_mitopathways(path):
    """{MitoPathway path (every level, 'A > B > C'): set of genes} from the MitoCarta3.0 human HTML table
    (columns Symbol ... MitoPathways; paths separated by ';<br>')."""
    import html
    import re
    t = open(path).read()
    out = {}
    for sym, rest in re.findall(r'<tr><td>([^<]+)</td>(.*?)</tr>', t, re.S):
        tds = re.findall(r'<td[^>]*>(.*?)</td>', rest, re.S)
        for p in re.split(r';\s*<br>', tds[5] if len(tds) > 5 else ''):
            parts = [x.strip() for x in html.unescape(p.replace('&nbsp', ' ')).split('>') if x.strip()]
            for k in range(1, len(parts) + 1):
                out.setdefault(' > '.join(parts[:k]), set()).add(sym.strip())
    return out


def set_z(S, ix):
    from scipy.stats import rankdata
    N = S.shape[1]
    pct = (rankdata(S, axis=1) - 1) / (N - 1)
    k = len(ix)
    return (pct[:, ix].mean(1) - pct.mean(1)) / np.sqrt(pct.var(1) / k * (N - k) / (N - 1))


def main(a):
    from scipy.stats import norm
    mp = load_mitopathways(a.mitocarta)
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = [str(x) for x in b['labels'][tr]]
    Pn = unit(b['P'][tr].astype(np.float64))
    pos = {g: i for i, g in enumerate(labels)}
    cats = {}
    for lab, names in CATEGORIES:
        genes = set().union(*[mp.get(n, set()) for n in names.split('|')])
        miss = [n for n in names.split('|') if n not in mp]
        ix = np.array(sorted(pos[g] for g in genes if g in pos))
        log.info('%-18s %3d genes in MitoCarta, %3d in the bank%s', lab, len(genes), len(ix),
                 f'  (pathway not found: {miss})' if miss else '')
        if len(ix) >= 5:
            cats[lab] = (ix, genes)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(category=k, n_bank=len(v[0]), genes=' '.join(sorted(v[1]))) for k, v in cats.items()]) \
        .to_csv(Path(a.out_dir) / 'pmd_categories.csv', index=False)
    rows = []
    # ---- PBMC
    sh = np.load(Path(a.score_dir) / 'shifts.npz', allow_pickle=True)
    keys = [k.split('|', 2) for k in sh['keys'].astype(str)]   # cell types may contain '|'
    sel = [i for i, k in enumerate(keys) if k[0] == 'ctrl']
    S = unit(sh['R'].astype(np.float64)[sel]) @ Pn.T
    for lab, (ix, _) in cats.items():
        z = set_z(S, ix)
        for j, i in enumerate(sel):
            _, d, ct = keys[i]
            rows.append(dict(source='pbmc', donor=d, condition=COND.get(d, '?'), cell_type=ct, stim='', category=lab,
                             set_z=float(z[j]), p=float(norm.sf(z[j]))))
    # ---- bulk
    if a.bulk_dir:
        Bd = Path(a.bulk_dir)
        Z = np.load(Bd / 'z_samples.npy').astype(np.float64)
        ob = pd.read_csv(Bd / 'obs_samples.csv')
        meta = pd.read_csv(a.bulk_meta).set_index('sample_id').loc[ob.sample_id]
        grp = (meta.cell_type.astype(str) + '|' + meta.stim.astype(str)).to_numpy()
        R = np.zeros_like(Z)
        for g_ in np.unique(grp):
            ix_ = np.flatnonzero(grp == g_)
            ctl = [i for i in ix_ if meta.iloc[i].group == 'control']
            for i in ix_:
                R[i] = Z[i] - Z[[c for c in ctl if c != i]].mean(0)
        Sb = unit(R) @ Pn.T
        for lab, (ix, _) in cats.items():
            z = set_z(Sb, ix)
            for i in range(len(ob)):
                r = meta.iloc[i]
                rows.append(dict(source='bulk', donor=r.patient, condition='PMD' if r.group == 'mito' else 'control',
                                 cell_type=r.cell_type, stim=r.stim, category=lab, set_z=float(z[i]),
                                 p=float(norm.sf(z[i])), fresh=str(r.pbmc_source).startswith('fresh')))
    T = pd.DataFrame(rows)
    T.to_csv(Path(a.out_dir) / 'pmd_category_setz.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    x = T[(T.source == 'pbmc') & (T.cell_type == a.celltype)]
    log.info('[pbmc %s] set z per donor\n%s', a.celltype,
             x.pivot_table(index='category', columns=['condition', 'donor'], values='set_z')
             .reindex([c for c, _ in CATEGORIES if c in cats]).round(1).to_string())
    if a.bulk_dir:
        y = T[(T.source == 'bulk') & (T.cell_type == 'Mono') & (T.stim == 'unstim')]
        log.info('[bulk Mono unstim] set z per sample\n%s',
                 y.pivot_table(index='category', columns=['condition', 'donor'], values='set_z')
                 .reindex([c for c, _ in CATEGORIES if c in cats]).round(1).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mitocarta', required=True, help='MitoCarta3.0 human HTML table (human.mitocarta3.0.html)')
    ap.add_argument('--score-dir', required=True, help='fig6_pmd.py score dir (shifts.npz)')
    ap.add_argument('--bank-npz', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bulk-dir', help='fig6_bulk_expert.py out dir')
    ap.add_argument('--bulk-meta')
    ap.add_argument('--celltype', default='Monocytes', help='log summary only')
    main(ap.parse_args())
