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

"""Collaborator export: the top-k predicted perturbations per disease donor x cell type, exactly as scored for
Fig. 5 (fig5_data.py output fig5_top500_per_group.csv, trained bank), annotated with
  replicated   the gene is also in the top-k of the OTHER donor of the same condition, same cell type and tail
  fig5b_rows   which Fig. 5b pathway rows (Reactome sets / curated modules) contain the gene
  model_input  the gene is one of the model's input genes (its own transcript is read by the encoder)
plus a README describing every column.

    python paper/scripts/fig5/fig5_export_top.py --top <fig5_top500_per_group.csv> --reactome-gmt <gmt> \
        --modules <readout_panels.tsv> --encode-json <variant encode.json> --out <csv> [--k 200]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig5_plot as fp
from fig5_centring_audit import NPC
from fig5_variants import PAUSE_BRAKE


def row_sets(gmt, modules):
    reac = {}
    for line in open(gmt):
        f = line.rstrip('\n').split('\t')
        reac[f[0]] = set(f[2:])
    mods = {}
    for line in open(modules):
        if line.startswith('#') or not line.strip():
            continue
        m, g = line.rstrip('\n').split('\t')[:2]
        mods.setdefault(m, set()).add(g)
    out = {}
    for lab, fam, st, _, _ in fp.B_ROWS:
        out[lab] = {'reactome': reac.get(st, set()), 'module': mods.get(st, set()), 'npc': set(NPC),
                    'custom': set(PAUSE_BRAKE)}[fam]
    return out


def main(a):
    t = pd.read_csv(a.top)
    t = t[(t.bank == 'trained') & ~t.donor.str.startswith('Nor') & (t['rank'] <= a.k)].copy()
    rs = row_sets(a.reactome_gmt, a.modules)
    t['fig5b_rows'] = [';'.join(lab for lab, g in rs.items() if gene in g) for gene in t.gene]
    inputs = set(json.load(open(a.encode_json))['input_genes'])
    t['model_input'] = t.gene.isin(inputs)
    key = ['condition', 'celltype', 'tail', 'gene']
    nd = t.groupby(key).donor.nunique().rename('n_donors').reset_index()
    t = t.merge(nd, on=key)
    t['replicated'] = t.n_donors == 2
    t['direction'] = t['tail'].map({'loss': 'loss of function', 'gain': 'gain of function'})
    if 'score' not in t:                       # older tables: raw cosine only; gain = -cos so larger = stronger
        t['score'] = t.cosine.where(t['tail'] == 'loss', -t.cosine)
    t = t.rename(columns={'cosine': 'raw_cosine', 'z': 'score_z', 'own_logfc': 'own_transcript_logfc',
                          'own_padj': 'own_transcript_padj'})
    cols = ['condition', 'donor', 'celltype', 'tail', 'direction', 'rank', 'gene', 'score', 'raw_cosine', 'score_z',
            'replicated',
            'fig5b_rows', 'model_input', 'own_transcript_logfc', 'own_transcript_padj']
    t = t[cols].sort_values(['condition', 'donor', 'celltype', 'tail', 'rank'])
    out = Path(a.out)
    t.to_csv(out, index=False)
    rep = t[t.replicated].drop_duplicates(['condition', 'celltype', 'tail', 'gene'])
    rep[['condition', 'celltype', 'tail', 'direction', 'gene', 'fig5b_rows', 'model_input']].to_csv(
        out.with_name(out.stem + '_replicated.csv'), index=False)
    out.with_name(out.stem + '_README.txt').write_text(f"""ExPert predictions, alcohol-associated liver disease snRNA-seq (Chembazhi et al. 2025, Nat Commun)
Model: ExPert v0.1.0 (production), no retraining. Mitochondrially encoded genes and metallothioneins removed from the
model input (a per-donor nuclei-prep artefact). Prototype bank: the {8964:,} perturbations seen in training.

How the scores were made
  For each cell type, nuclei of each donor were pooled into pseudobulks and encoded by ExPert. Within each donor set
  ({{AC1, AH1, Normal1}} and {{AC2, AH2, Normal2}}), each condition's latent was compared with the mean of the three
  conditions; 'score' is the cosine of that difference to the perturbation's prototype. The two donor sets are
  independent replicates.

Rows: the top {a.k} perturbations per disease donor x cell type x tail ({fp.CT}).

Columns
  condition / donor / celltype     AC = alcohol-associated cirrhosis, AH = alcoholic hepatitis
  tail, direction                  loss: the group resembles a KNOCKOUT of the gene (gene function lost/impaired)
                                   gain: the group resembles the opposite of that knockout
  rank                             1 = strongest within the group and tail
  score                            ranking score: cosine to the gene's prototype in this tail (loss: +cos, gain: -cos)
                                   minus the model's stored per-prototype popularity offset (hubness correction, as
                                   used by ExPert's predict()); larger = stronger
  raw_cosine                       the uncorrected cosine (signed as stored: loss positive, gain negative)
  score_z                          score standardised over all {8964:,} prototypes of the group
  replicated                       also in the top {a.k} of the other donor of the same condition (same cell type, tail);
                                   the stronger candidates for follow-up (also listed alone in *_replicated.csv)
  fig5b_rows                       Fig. 5b pathway rows containing the gene (';'-separated; empty = none)
  model_input                      TRUE = the gene's own mRNA is a model input; treat with care when its own
                                   transcript also changes (possible self-signal)
  own_transcript_logfc / _padj     the gene's own mRNA change vs Normal in that cell type (Wilcoxon, BH); predictions
                                   are largely independent of these (Fig. 5c)
""")
    print(f'{len(t):,} rows -> {out}; replicated unique {len(rep):,}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--top', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--modules', required=True)
    ap.add_argument('--encode-json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=200)
    main(ap.parse_args())
