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

"""Supplementary Fig. S1 plotting data (light panels). One CSV per panel in --out-dir
(paper/figures/si/figS1/data), named
after the panel:

    figS1a_cells_per_perturbation.csv          (dataset, perturbation): QC-passed perturbed cells in the store,
                                               and whether the perturbation is in the training DB
    figS1a_cells_per_perturbation_summary.csv  per dataset: cells / perturbations in and out of the DB
    figS1b_pull_cutoff.csv                     selected perturbations: training support (training cells x datasets
                                               with training cells) and whether the perturbation is pulled toward
                                               its prior neighbours (Fig. 1's corpus table, so the pulled set is the
                                               one Fig. 1 marks)
    figS1e_effect_size.csv                     per-cell measured effect (raw signed Mixscale z), sampled
    figS1e_effect_size_summary.csv             per dataset: cells scored, median / 99th percentile
    figS1f_gene_selection.csv                  held-out decodability by gene-selection strategy and size
    figS1aux_essential_cells.csv               (dataset, perturbation): store cells, DepMap class, selected (aux
                                               panel; also Table S1C's DepMap class)
    figS1_meta.json                            shipped pull_q, selected and pulled counts

Panels C (cross-dataset response similarity) and D (variance explained) are computed by their own scripts
(figS1c_response_data.py, figS1d_variance_data.py). Store / efficiency counts reuse ExPert-data's collectors
(scripts/plot_store_overview.py, scripts/plot_efficiency_overview.py) so the numbers match the data paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ESS_LABEL = ['common essential', 'selective', 'non-essential']
SAMPLE_PER_DS = 20_000


def depmap_class(genes: pd.Index, gene_effect: Path) -> pd.Series:
    """DepMap Chronos: fraction of cell lines with gene effect < -0.5 -> class per gene."""
    ge = pd.read_csv(gene_effect, index_col=0)
    ge.columns = [c.split(' (')[0] for c in ge.columns]
    ge = ge.loc[:, ~ge.columns.duplicated()]
    frac = (ge < -0.5).mean(0).reindex(genes)
    cls = np.select([frac >= 0.9, frac > 0, frac == 0], ESS_LABEL, 'not in DepMap')
    return pd.Series(cls, index=genes), frac


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.data_repo) / 'scripts'))
    import plot_store_overview as so
    import plot_efficiency_overview as eo
    store, db = Path(a.store), Path(a.db)

    # ---- A: cells per perturbation per dataset (store, QC-passed perturbed cells) ----
    pp, extra = so.collect(store, db, 'dataset')
    pp = pp.rename(columns={'label': 'perturbation', 'included': 'in_db'})
    pp[['dataset', 'perturbation', 'n_cells', 'in_db']].to_csv(out / 'figS1a_cells_per_perturbation.csv',
                                                              index=False)
    extra['summary'].to_csv(out / 'figS1a_cells_per_perturbation_summary.csv', index=False)
    print(f'[A] {len(pp):,} (dataset, perturbation) rows over {pp.dataset.nunique()} datasets', flush=True)

    # ---- B: training support of selected perturbations and the pulled tail ----
    perts = pd.read_parquet(Path(a.fig1_data) / 'fig1c_perts.parquet').set_index('perturbation')
    meta = json.loads((Path(a.fig1_data) / 'fig1c_meta.json').read_text())
    f = perts[perts.selected][['evidence', 'n_train', 'd_train', 'w_hard']].copy()
    f['pulled'] = f.w_hard > 0
    f.drop(columns='w_hard').reset_index().to_csv(out / 'figS1b_pull_cutoff.csv', index=False)
    print(f'[B] {len(f):,} selected, {int(f.pulled.sum()):,} pulled', flush=True)

    (out / 'figS1_meta.json').write_text(json.dumps(dict(
        pull_q=meta['pull_q'], n_selected=int(len(f)), n_pulled=int(f.pulled.sum())), indent=2))

    # ---- aux: same counts + DepMap essentiality + training selection ----
    genes = pd.Index(pp.perturbation.unique())
    cls, frac = depmap_class(genes, Path(a.depmap))
    b = pp[['dataset', 'perturbation', 'n_cells', 'in_db']].copy()
    b['depmap_dep_frac'] = b.perturbation.map(frac)
    b['depmap_class'] = b.perturbation.map(cls)
    b['selected'] = b.perturbation.map(perts.selected).fillna(False).astype(bool)
    b.to_csv(out / 'figS1aux_essential_cells.csv', index=False)
    t = b.groupby('depmap_class').agg(perts=('perturbation', 'nunique'),
                                      median_cells=('n_cells', 'median'))
    print(f'[aux] DepMap classes over measured perturbations:\n{t}', flush=True)

    # ---- E: per-cell measured effect size (raw signed Mixscale z) ----
    summ, _pert, samp = eo.collect(store, None)
    samp = (samp.groupby('dataset', group_keys=False)
            .apply(lambda g: g.sample(min(len(g), SAMPLE_PER_DS), random_state=0)))
    samp.to_csv(out / 'figS1e_effect_size.csv', index=False)
    summ.to_csv(out / 'figS1e_effect_size_summary.csv', index=False)
    print(f'[E] {len(samp):,} sampled cells', flush=True)

    # ---- F: gene-selection sweep (ExPert-data scripts/gene_selection_sweep.py, 22 datasets, held-out perturbation
    # halves; reference copy of the panel code under paper/scripts/gene_panel_src/) ----
    s = pd.read_csv(a.gene_sweep)
    s = s[s.strategy != 'all_genes']
    g = (s.groupby(['strategy', 'n'])
         .agg(top1=('top1', 'mean'), top1_sd=('top1', 'std'), n_datasets=('dataset', 'nunique'),
              n_genes_median=('n_genes', 'median')).reset_index())
    g.to_csv(out / 'figS1f_gene_selection.csv', index=False)
    print(f'[F] {len(g)} (strategy, size) rows from {s.dataset.nunique()} datasets', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True, help='canonical store (store/HASH)')
    ap.add_argument('--db', required=True, help='materialised DB dir (store_out/HASH)')
    ap.add_argument('--data-repo', required=True, help='ExPert-data repo root')
    ap.add_argument('--fig1-data', required=True, help='Fig. 1 data dir (fig1c_perts.parquet, fig1c_meta.json)')
    ap.add_argument('--depmap', required=True, help='DepMap CRISPRGeneEffect.csv')
    ap.add_argument('--gene-sweep', required=True, help='ExPert-data results/gene_sweep/sweep_holdout.csv')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
