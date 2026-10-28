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

"""Fig. 6E: do the patients' profiles land on the genetically established primary-mitochondrial-disease genes?

Gene set = the Genomics England PanelApp 'Mitochondrial disorders' panel (--panel-json), green (diagnostic-grade)
genes, mapped to the model's symbols (current symbol, then aliases, then known HGNC renames). Rows:
    all PMD genes   one module of every mapped green gene
    groups          the same genes split by MitoCarta3.0 MitoPathway (--mitocarta, first matching group wins):
                    each OXPHOS complex (subunits + assembly factors), CoQ, Fe-S & lipoate, mtDNA maintenance,
                    mt translation & RNA, PDH & TCA, import & dynamics, carriers, other
    suggested gene  the patient's candidate lesion gene alone (a one-gene set z)
    all             the union of the Fig. 6E rows: MitoCarta3.0 OXPHOS CI-CV (subunits + assembly factors, as
                    fig6_pmd_categories.py) + the PanelApp CoQ and PDH & TCA genes
Score = Fig. 6 set z of the genes in the donor's profile (fig6_pmd.py score shifts, ref 'ctrl': a PMD donor vs the
pooled non-PMD cells, a non-PMD donor vs the other non-PMD donors), --bank trained | full. p = upper tail.

The Fig. 6E / 6F magnitude-aware scores are fig6_donor_mag.py (it reads pmd_panel_genes_*.csv from here).

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_pmd_panel.py --panel-json ... --mitocarta ... --score-dir ...
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig6_pmd_categories import CATEGORIES, load_mitopathways, set_z, unit

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_pmd_panel')
log.setLevel(logging.INFO)

CTRL = ['PT38', 'PT50', 'PT55', 'PT41', 'PT42']                 # fig6_pmd.CTRL: the reference donors
COND = {'PT43': 'PMD', 'PT46': 'PMD', 'PT54': 'PMD', 'PT67': 'PMD', 'PT41': 'non-PMD', 'PT42': 'non-PMD',
        'PT38': 'undiagnosed', 'PT50': 'undiagnosed', 'PT55': 'undiagnosed'}
SUGGESTED = {'PT54': 'NDUFS8', 'PT46': 'NDUFA10', 'PT43': 'BCS1L', 'PT67': 'PC'}
# HGNC renames PanelApp has not caught up with (the panel symbol -> the current symbol)
RENAMES = {'ATP5A1': 'ATP5F1A', 'ATP5D': 'ATP5F1D', 'ATP5E': 'ATP5F1E', 'ATP5G3': 'ATP5MC3', 'ATP5O': 'ATP5PO',
           'C12orf65': 'MTRFR', 'C19orf70': 'MICOS13', 'APOPT1': 'COA8', 'TAZ': 'TAFAZZIN', 'KIAA0391': 'PRORP',
           'KARS': 'KARS1', 'GARS': 'GARS1', 'QARS': 'QARS1', 'SPATA5': 'AFG2A', 'C2orf69': 'C2orf69'}
# group: MitoPathway prefixes (first matching group wins, in this order)
GROUPS = [
    ('CI', ['OXPHOS > Complex I']), ('CII', ['OXPHOS > Complex II']), ('CIII', ['OXPHOS > Complex III']),
    ('CIV', ['OXPHOS > Complex IV']), ('CV', ['OXPHOS > Complex V']),
    ('CoQ', ['Metabolism > Metals and cofactors > Coenzyme Q metabolism']),
    ('Fe-S & lipoate', ['Metabolism > Metals and cofactors > Fe-S cluster biosynthesis',
                        'Metabolism > Lipid metabolism > Lipoate insertion']),
    ('mtDNA maintenance', ['Mitochondrial central dogma > mtDNA maintenance']),
    ('mt translation & RNA', ['Mitochondrial central dogma > Translation',
                              'Mitochondrial central dogma > mtRNA metabolism']),
    ('PDH & TCA', ['Metabolism > Carbohydrate metabolism > Pyruvate metabolism',
                   'Metabolism > Carbohydrate metabolism > TCA cycle',
                   'Metabolism > Carbohydrate metabolism > TCA-associated']),
    ('import & dynamics', ['Protein import, sorting and homeostasis', 'Mitochondrial dynamics and surveillance']),
    ('carriers', ['Small molecule transport'])]


def panel_genes(path, labels):
    """PanelApp green genes -> {panel symbol: model symbol} (current symbol, aliases, then RENAMES)."""
    lab = set(labels)
    d = json.load(open(path))
    out, miss = {}, []
    for x in d['genes']:
        if x.get('confidence_level') != '3':
            continue
        gd = x['gene_data']
        cand = [gd.get('gene_symbol'), gd.get('hgnc_symbol'), RENAMES.get(gd.get('gene_symbol'))] + \
            list(gd.get('alias') or [])
        hit = next((c for c in cand if c and c in lab), None)
        if hit:
            out[gd['gene_symbol']] = hit
        else:
            miss.append(gd['gene_symbol'])
    return out, miss, d.get('name'), d.get('version')


def main(a):
    from scipy.stats import norm
    b = np.load(a.bank_npz, allow_pickle=True)
    keep = b['trained'].astype(bool) if a.bank == 'trained' else np.ones(len(b['labels']), bool)
    labels = [str(x) for x in b['labels'][keep]]
    Pn = unit(b['P'][keep].astype(np.float64))
    pos = {g: i for i, g in enumerate(labels)}
    mapped, miss, pname, pver = panel_genes(a.panel_json, labels)
    log.info('%s v%s: %d green genes in the %s bank, %d not (%s)', pname, pver, len(mapped), a.bank, len(miss),
             ' '.join(miss[:40]))
    mp = load_mitopathways(a.mitocarta)
    genes = sorted(set(mapped.values()))
    grp = {}
    for g in genes:
        grp[g] = next((lab for lab, prefixes in GROUPS
                       if any(g in mp.get(p, set()) for p in prefixes)), 'other')
    rows = [('all PMD genes', genes)] + [(lab, [g for g in genes if grp[g] == lab]) for lab, _ in GROUPS + [('other', 0)]]
    rows = [(r, gs) for r, gs in rows if len(gs) >= a.min_genes]
    cx = set().union(*[mp.get(n, set()) for lab, n in CATEGORIES if lab in ('CI', 'CII', 'CIII', 'CIV', 'CV')])
    rows.append(('all', sorted((cx & set(pos)) | {g for g in genes if grp[g] in ('CoQ', 'PDH & TCA')})))
    sh = np.load(Path(a.score_dir) / 'shifts.npz', allow_pickle=True)
    keys = [k.split('|', 2) for k in sh['keys'].astype(str)]
    sel = [i for i, k in enumerate(keys) if k[0] == 'ctrl']
    S = unit(sh['R'].astype(np.float64)[sel]) @ Pn.T
    out = []
    for r, gs in rows + [(f'gene:{g}', [g]) for g in sorted(set(SUGGESTED.values()) & set(pos))]:
        ix = np.array([pos[g] for g in gs])
        z = set_z(S, ix)
        for j, i in enumerate(sel):
            _, d, ct = keys[i]
            out.append(dict(row=r, n_genes=len(ix), donor=d, condition=COND.get(d, '?'), cell_type=ct,
                            set_z=float(z[j]), p=float(norm.sf(z[j]))))
    T = pd.DataFrame(out)
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    T.to_csv(o / f'pmd_panel_setz_{a.bank}.csv', index=False)
    pd.DataFrame([dict(panel_symbol=k, model_symbol=v, group=grp[v]) for k, v in sorted(mapped.items())]) \
        .to_csv(o / f'pmd_panel_genes_{a.bank}.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    x = T[T.cell_type == a.celltype]
    log.info('[%s %s] set z\n%s', a.bank, a.celltype,
             x.pivot_table(index=['row', 'n_genes'], columns=['condition', 'donor'], values='set_z')
             .round(1).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--panel-json', required=True, help='PanelApp panel JSON (Mitochondrial disorders)')
    ap.add_argument('--mitocarta', required=True, help='MitoCarta3.0 human HTML table')
    ap.add_argument('--score-dir', required=True, help='fig6_pmd.py score dir (shifts.npz)')
    ap.add_argument('--bank-npz', required=True)
    ap.add_argument('--bank', default='trained', choices=['trained', 'full'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-genes', type=int, default=3)
    ap.add_argument('--celltype', default='Monocytes', help='log summary only')
    main(ap.parse_args())
