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

"""Table S8: the HDAC machinery gene set that Fig. 4D scores the HDAC inhibitors against, and the class I corepressor
subunits the CORUM-core partner rule was calibrated on (related to Fig. 4). Read from the same two tables
fig4_sciplex_panels.py builds the set from (--hdac-complexes: the curated complexes; --partner-sets: the
collaborator's CORUM-core partners of the HDAC family, gene:S_g, restricted to the candidate panel), so the union is
the panel's 48 genes. Written to --out-dir (si/tables) as Table_S8.xlsx plus one CSV per part.

    S8A  HDAC machinery (48 genes): complexes the gene belongs to, whether it is a CORUM-core partner and its S_g, and
         whether it is a candidate of the shared sci-Plex3 panel
    S8B  class I corepressor subunits (the curated complexes without the catalytic HDAC1-3): complex, and whether
         the CORUM-core partner rule recovers it (with S_g); recall and precision of the rule printed
    S8C  vendor-datasheet targets of the sci-Plex3 drugs without a usable ChEMBL target (--vendor-targets: the
         curated workbook, header on its fourth row; copied as is, empty rows dropped)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATALYTIC = ('HDAC1', 'HDAC2', 'HDAC3')
SHEET = {'S8A': 'S8A HDAC machinery', 'S8B': 'S8B corepressor subunits', 'S8C': 'S8C vendor targets'}


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    hd = pd.read_csv(a.hdac_complexes)
    ps = pd.read_csv(a.partner_sets)
    part = {x.split(':')[0]: float(x.split(':')[1])
            for x in str(ps[ps.family == a.family].partners_in_candidates.iloc[0]).split(';') if x.strip()}
    cand = set(np.load(a.candidates, allow_pickle=True)['genes'].astype(str))
    cx = hd.groupby('gene').complex.apply(lambda s: '; '.join(sorted(set(s))))
    genes = sorted(set(hd.gene) | set(part))
    assert len(genes) == a.expect_n, f'{len(genes)} genes, Fig. 4 scores {a.expect_n}'
    s7a = pd.DataFrame(dict(gene=genes, complexes=[cx.get(g, '') for g in genes],
                            curated_complex_member=[g in cx.index for g in genes],
                            corum_core_partner=[g in part for g in genes],
                            S_g=[part.get(g, np.nan) for g in genes],
                            in_sciplex3_candidates=[g in cand for g in genes]))
    s7a = s7a.sort_values(['curated_complex_member', 'S_g'], ascending=[False, False]).reset_index(drop=True)
    sub = hd[~hd.gene.isin(CATALYTIC)].groupby('gene').complex.apply(lambda s: '; '.join(sorted(set(s))))
    s7b = pd.DataFrame(dict(gene=sub.index, complex=sub.values, recovered_by_partner_rule=sub.index.isin(list(part)),
                            S_g=[part.get(g, np.nan) for g in sub.index]))
    s7b = s7b.sort_values(['complex', 'gene']).reset_index(drop=True)
    hit = int(s7b.recovered_by_partner_rule.sum())
    print(f'[S8A] {len(s7a)} genes: {int(s7a.curated_complex_member.sum())} curated complex members, '
          f'{len(part)} CORUM-core partners, {int((s7a.curated_complex_member & s7a.corum_core_partner).sum())} both; '
          f'{int(s7a.in_sciplex3_candidates.sum())} in the sci-Plex3 candidates')
    print(f'[S8B] {len(s7b)} subunits; partner rule recovers {hit}: recall {hit / len(s7b):.3f}, '
          f'precision {hit / len(part):.3f} (partners restricted to the candidate panel)')
    s7c = pd.read_excel(a.vendor_targets, header=a.vendor_header_row).dropna(how='all')
    s7c = s7c.dropna(axis=1, how='all')
    print(f'[S8C] {len(s7c)} drugs with vendor-datasheet targets')
    with pd.ExcelWriter(out / 'Table_S8.xlsx') as w:
        for k, df in (('S8A', s7a), ('S8B', s7b), ('S8C', s7c)):
            df.to_excel(w, sheet_name=SHEET[k], index=False)
            df.to_csv(out / f'Table_{k}.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hdac-complexes', required=True, help='curated HDAC complexes (complex, gene)')
    ap.add_argument('--partner-sets', required=True, help='family_partners_corum_core_sets.csv (gene:S_g)')
    ap.add_argument('--candidates', required=True, help='npz with the shared sci-Plex3 candidate genes')
    ap.add_argument('--vendor-targets', required=True, help='curated vendor-target workbook (S8C)')
    ap.add_argument('--vendor-header-row', type=int, default=3, help='0-based row of the column header')
    ap.add_argument('--family', default='HDAC')
    ap.add_argument('--expect-n', type=int, default=48, help='size of the set Fig. 4D scores')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
