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

"""Table S1: the training corpus, built from the store / DB counts written by fig1b_data.py (--fig1-data) and
figS1_data.py (--data-dir). Written to --out-dir (paper/figures/si/tables) as Table_S1.xlsx (one sheet per table) plus one CSV
per sheet.

Sheets
    S1A data sources  one row per dataset used in the paper (--data-sources): public status, accession, the figures
                      that use it; for the training screens of the canonical store also study, DOI, perturbation
                      type, cell line / donor / condition, QC-passed perturbed cells, perturbations measured
                      (>= min_cells_per_class cells), share of perturbed cells with a measured effect size.
                      Without --data-sources: the training screens only (sheet 'S1A store')
    S1B training DB   one row per screen in the training database: whether it counts toward the pool
                      rule, perturbed and control cells, selected perturbations, cells per perturbation,
                      production train / validation split, prior-pulled perturbations
    S1C gene classes  share of each functional class among the selected vs all measured perturbations
                      (classes overlap; Fisher test selected vs not selected)

Gene classes: transcription factor = Lambert et al. 2018 human TF census ("Is TF?" == Yes);
chromatin / signaling / metabolism / transcription = membership of the Reactome top-level
pathway of that name; complex subunit = member of any CORUM complex.

Also prints [N]: perturbation-context and perturbation-dataset pairs with cells in the DB.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

TF_URL = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv'
REACTOME_TOP = {'chromatin regulator': 'R-HSA-4839726',        # Chromatin organization
                'signaling component': 'R-HSA-162582',         # Signal Transduction
                'metabolic enzyme': 'R-HSA-1430728',           # Metabolism
                'transcription (Reactome)': 'R-HSA-74160'}     # Gene expression (Transcription)


def fetch(url: str, cache: Path) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
    cache.write_bytes(b)
    return b


def gene_classes(genes: pd.Index, cache: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=genes)
    tf = pd.read_csv(io.BytesIO(fetch(TF_URL, cache / 'humantfs_v1.01.csv')))
    sym = next(c for c in tf.columns if c.strip().lower() in ('hgnc symbol', 'name'))
    istf = next(c for c in tf.columns if 'is tf' in c.lower())
    out['transcription factor'] = genes.isin(set(tf.loc[tf[istf].astype(str).str.lower() == 'yes', sym]))
    gmt = zipfile.ZipFile(cache / 'ReactomePathways.gmt.zip').read('ReactomePathways.gmt').decode()
    react = {}
    for line in gmt.splitlines():
        p = line.split('\t')
        if len(p) > 2:
            react[p[1]] = set(p[2:])
    for name, rid in REACTOME_TOP.items():
        out[name] = genes.isin(react[rid])
    corum = set()
    for line in (cache / 'CORUM.gmt').read_text().splitlines():
        corum |= set(line.rstrip('\n').split('\t')[2:])
    out['complex subunit (CORUM)'] = genes.isin(corum)
    return out


def add_total(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Append a TOTAL row: 'sum' sums the column, a number is used as is (unique counts do not add)."""
    tot = {c: '' for c in df.columns}
    for c, v in spec.items():
        tot[c] = int(df[c].sum()) if v == 'sum' else v
    out = pd.concat([df.astype(object), pd.DataFrame([tot], index=['TOTAL'])])
    out.index.name = df.index.name
    return out



def with_data_sources(s1a, src):
    """S1A = every dataset the paper uses (--data-sources: training screens, inference screens and this study's
    unpublished data, with public status, accession and the figures that use it). The training screens' store
    columns are taken from the freshly computed s1a; their totals row follows the last training screen."""
    s1a = s1a.copy()
    tot = s1a.loc[['TOTAL']] if 'TOTAL' in s1a.index else None
    s1a = s1a.drop(index='TOTAL', errors='ignore')
    missing = s1a.index.difference(src.index)
    if len(missing):
        raise SystemExit(f'--data-sources lacks training datasets: {list(missing)}')
    out = src.astype(object)
    cols = [c for c in s1a.columns if c in out.columns]
    out.loc[s1a.index, cols] = s1a[cols].to_numpy()
    if tot is not None:
        tot = tot.reindex(columns=out.columns).astype(object)
        tot.loc['TOTAL', 'Public'] = ''
        k = out.index.get_loc(s1a.index[-1]) + 1
        out = pd.concat([out.iloc[:k], tot.rename(index={'TOTAL': 'TOTAL (training corpus)'}), out.iloc[k:]])
    out.index.name = 'dataset'
    for c in out.columns:                                    # counts stay integers (the CSV read gave floats)
        if out[c].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            continue
        v = pd.to_numeric(out[c], errors='coerce')
        if v.notna().any() and v.isna().eq(out[c].isna() | out[c].eq('')).all() and (v.dropna() % 1 == 0).all():
            out[c] = v.astype('Int64')
    return out

def main(a):
    d, f1 = Path(a.data_dir), Path(a.fig1_data)
    cells = pd.read_parquet(f1 / 'fig1c_cells.parquet')
    per = pd.read_parquet(f1 / 'fig1c_perts.parquet').set_index('perturbation')
    meta = json.load(open(f1 / 'fig1c_meta.json'))
    mcc = meta['min_cells_per_class']
    sheet = pd.read_csv(a.datasheet).set_index('index')
    order = meta['dataset_order']

    # context: K562 screens merged, donors pooled per condition (the pool rule's contexts)
    ds_ctx = cells.drop_duplicates('dataset').set_index('dataset')
    ctx = ds_ctx.apply(lambda r: f'CD4+ T, {r.condition}' if r.axis == 'donor' else r.context, axis=1)
    cells['context12'] = cells.dataset.map(ctx)

    # ---- S1A store (per dataset) ----------------------------------------------------------
    g = cells.groupby('dataset')
    eff = pd.read_csv(d / 'figS1e_effect_size_summary.csv').set_index('dataset')
    s1a = pd.DataFrame({
        'study': sheet.loc[order, 'publication index'],
        'DOI': sheet.loc[order, 'doi link'].str.replace('https://doi.org/', '', regex=False),
        'title': sheet.loc[order, 'publication title'],
        'perturbation type': sheet.loc[order, 'perturbation_type'],
        'cell line / cell type': [r.context if r.axis != 'donor' else 'primary CD4+ T cells'
                                  for r in ds_ctx.loc[order].itertuples()],
        'donor': ds_ctx.loc[order, 'donor'].fillna(''),
        'condition': ds_ctx.loc[order, 'condition'].fillna(''),
        'context': ctx.loc[order],
        'QC-passed perturbed cells': g.store_cells.sum().loc[order],
        'perturbations with cells': cells[cells.store_cells > 0].groupby('dataset').perturbation.nunique()
            .reindex(order),
        'perturbations measured (>=%d cells)' % mcc:
            cells[cells.store_cells >= mcc].groupby('dataset').perturbation.nunique().reindex(order),
        'median cells per perturbation': cells[cells.store_cells > 0].groupby('dataset').store_cells.median()
            .reindex(order),
        'perturbed cells with effect size (%)': (100 * eff.frac_scored).reindex(order).round(1),
        'median effect size (Mixscale z)': eff.median_z.reindex(order).round(2),
    })
    s1a.index.name = 'dataset'
    s1a = add_total(s1a, {'QC-passed perturbed cells': 'sum',
                          'perturbations with cells': int((per.store_cells > 0).sum()),
                          'perturbations measured (>=%d cells)' % mcc: int((per.n_contexts_store > 0).sum())})

    if a.data_sources:
        s1a = with_data_sources(s1a, pd.read_csv(a.data_sources).set_index('dataset'))

    # ---- S1B training DB (per dataset) ------------------------------------------------------
    import h5py
    with h5py.File(a.ctrl_db, 'r') as f:
        n_ctrl = pd.Series(f['obs']['dataset'].asstr()[:]).value_counts()
    indb = cells[cells.db_cells > 0]
    pulled = set(per.index[per.w_hard.fillna(0) > 0])
    s1b = pd.DataFrame({
        'context': ctx.loc[order],
        'counted toward pool rule': ds_ctx.loc[order, 'group'].notna(),
        'perturbed cells': g.db_cells.sum().loc[order],
        'control cells': n_ctrl.reindex(order).fillna(0).astype(int),
        'selected perturbations': indb.groupby('dataset').perturbation.nunique().reindex(order),
        'median cells per perturbation': indb.groupby('dataset').db_cells.median().reindex(order),
        'train cells (production split)': g.train_cells.sum().loc[order],
        'validation cells (production split)': (g.db_cells.sum() - g.train_cells.sum()).loc[order],
        'prior-pulled perturbations': indb[indb.perturbation.isin(pulled)].groupby('dataset')
            .perturbation.nunique().reindex(order).fillna(0).astype(int),
    })
    s1b.index.name = 'dataset'
    s1b = add_total(s1b, {'perturbed cells': 'sum', 'control cells': 'sum',
                          'selected perturbations': int(per.selected.sum()),
                          'train cells (production split)': 'sum', 'validation cells (production split)': 'sum',
                          'prior-pulled perturbations': len(pulled)})

    # ---- S1C gene-class composition of the selected perturbations ----------------------------
    meas = per[per.n_contexts_store > 0].copy()
    cls = gene_classes(meas.index, Path(a.cache_dir))
    if a.essentiality:
        e = pd.read_csv(a.essentiality).drop_duplicates('perturbation').set_index('perturbation')
        cls['common essential (DepMap)'] = (e.depmap_class.reindex(cls.index) == 'common essential').to_numpy()
    rows = []
    sel = meas.selected.to_numpy()
    for c in cls.columns:
        x = cls[c].to_numpy()
        t = [[int((x & sel).sum()), int((~x & sel).sum())],
             [int((x & ~sel).sum()), int((~x & ~sel).sum())]]
        orr, p = fisher_exact(t)
        rows.append({'class': c, 'selected (n)': t[0][0],
                     'selected (%)': 100 * t[0][0] / sel.sum(),
                     'all measured (n)': int(x.sum()), 'all measured (%)': 100 * x.mean(),
                     'not selected (%)': 100 * t[1][0] / (~sel).sum(),
                     'odds ratio selected vs not': orr, 'Fisher p': p})
    s1c = pd.DataFrame(rows).set_index('class')

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / 'Table_S1.xlsx') as w:
        s1a.to_excel(w, sheet_name='S1A data sources' if a.data_sources else 'S1A store')
        s1b.to_excel(w, sheet_name='S1B training DB')
        s1c.to_excel(w, sheet_name='S1C gene classes')
    for name, df in (('A', s1a), ('B', s1b), ('C', s1c)):
        df.to_csv(out / f'Table_S1{name}.csv')

    # ---- [N] ------------------------------------------------------------------------------
    pc = cells[cells.db_cells > 0][['perturbation', 'context12']].drop_duplicates()
    pdp = cells[cells.db_cells > 0][['perturbation', 'dataset']].drop_duplicates()
    pc_tr = cells[cells.train_cells > 0][['perturbation', 'context12']].drop_duplicates()
    print(f'[N] perturbation-context pairs (12 contexts) in DB: {len(pc):,} '
          f'(train split {len(pc_tr):,}); perturbation-dataset pairs (22 datasets): {len(pdp):,}')
    print(f'    contexts per selected perturbation: median {pc.groupby("perturbation").size().median():.0f}, '
          f'mean {pc.groupby("perturbation").size().mean():.2f}')
    pd.set_option('display.width', 250)
    print(s1a.drop(columns=['title', 'DOI']).to_string())
    print(s1b.to_string())
    print(s1c.round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fig1-data', required=True, help='Fig. 1 data dir (fig1c cells / perts parquet, meta)')
    ap.add_argument('--data-dir', required=True, help='Fig. S1 data dir (figS1e_effect_size_summary.csv)')
    ap.add_argument('--datasheet', required=True)
    ap.add_argument('--cache-dir', required=True, help='Reactome / CORUM / TF census cache')
    ap.add_argument('--essentiality', help='figS1aux_essential_cells.csv (DepMap class per perturbation)')
    ap.add_argument('--ctrl-db', required=True, help='database_ctrl.h5ad (control cells per dataset)')
    ap.add_argument('--data-sources', help='curated table of every dataset used (public status, accession, figures); '
                    'makes S1A that table with the training screens\' store columns filled in')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
