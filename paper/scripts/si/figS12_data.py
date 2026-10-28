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

"""Supplementary Fig. S12 plotting data (ExPert scores of the bulk RNA-seq livers against the snRNA-seq hepatocyte
scores; related to Fig. 5). Reads the output of fig5_bulk_expert.py (stage encode on SLURM, stage score: each liver
encoded as one pseudobulk, latents centred over all livers, pooled by condition, cosine to the trained bank, compared
with the snRNA-seq hepatocyte profile of the same condition). Nothing is re-scored; one CSV per panel in --out-dir
(paper/figures/si/figS12/data):

    figS12a_families.csv   mean predicted-loss z of each mRNA export family (the Fig. 5L families), snRNA-seq
                           hepatocytes vs bulk livers, per condition
    figS12a_rho.csv        per condition: Spearman of snRNA vs bulk over the families shown, and over the export genes
    figS12b_enrichment.csv per condition x gene set (mRNA export, mRNA splicing) x source (snRNA, bulk): top-k hits,
                           expected hits, fold, hypergeometric p (background = trained bank)
    figS12_meta.json       livers per condition, top-k, whole-bank Spearman bulk vs snRNA and top-k overlap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_final as F  # noqa: E402

CONDS = ['AC', 'AH', 'Normal']


def main(a):
    src, out = Path(a.bulk_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Fm = pd.read_csv(src / 'fig5_bulk_expert_families.csv')
    Fm = Fm[Fm.family != 'other']
    fams = [f for f, _ in F.EXPORT_FAMILIES if f in set(Fm.family)]
    rows, rho = [], []
    for c in CONDS:
        g = Fm[Fm.cond == c]
        m = g.groupby('family')[['z_sn', 'z_bulk']].mean().loc[fams]
        rows += [dict(cond=c, family=f, z_sn=r.z_sn, z_bulk=r.z_bulk) for f, r in m.iterrows()]
        rg = spearmanr(g.z_sn, g.z_bulk)
        rho.append(dict(cond=c, rho_families=spearmanr(m.z_sn, m.z_bulk).statistic, n_families=len(fams),
                        rho_genes=rg.statistic, p_genes=rg.pvalue, n_genes=len(g)))
    pd.DataFrame(rows).to_csv(out / 'figS12a_families.csv', index=False)
    pd.DataFrame(rho).to_csv(out / 'figS12a_rho.csv', index=False)

    R = pd.read_csv(src / 'fig5_bulk_expert_compare.csv')
    e = []
    for r in R[R.metric.str.contains(f'_top{a.top}_hits')].itertuples():
        source, gset = r.metric.split('_mRNA ')[0], 'mRNA ' + r.metric.split('_mRNA ')[1].split(f'_top{a.top}')[0]
        e.append(dict(cond=r.cond, gene_set=gset, source={'bulk': 'bulk', 'sn_hep': 'snRNA'}[source], hits=int(r.value),
                      expected=r.expected, fold=r.value / r.expected, p=r.p))
    E = pd.DataFrame(e)
    E.to_csv(out / 'figS12b_enrichment.csv', index=False)

    z = np.load(src / 'bulk_latents.npz', allow_pickle=True)
    cond = pd.Series(z['condition'].astype(str)).value_counts().to_dict()
    whole = R[R.metric == 'spearman_bulk_vs_sn_hep'].set_index('cond')[['value', 'p']]
    ov = R[R.metric == f'top{a.top}_overlap_bulk_sn'].set_index('cond')[['value', 'p']]
    meta = dict(source=str(src), livers=cond, top=a.top, families=fams,
                spearman_whole_bank={c: dict(rho=float(whole.loc[c, 'value']), p=float(whole.loc[c, 'p'])) for c in CONDS},
                top_overlap={c: dict(n=int(ov.loc[c, 'value']), p=float(ov.loc[c, 'p'])) for c in CONDS})
    (out / 'figS12_meta.json').write_text(json.dumps(meta, indent=1))
    print(pd.DataFrame(rho).round(3).to_string(index=False))
    print(E.round(4).to_string(index=False))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bulk-dir', required=True,
                    help='paper/figures/fig5/data/bulk_expert (fig5_bulk_expert.py --out-dir)')
    ap.add_argument('--top', type=int, default=200, help='top-k of the score stage')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
