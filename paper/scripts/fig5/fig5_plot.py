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

"""Fig. 5 constants shared with other scripts: the cell types of the ALD readout, the curated pathway rows
(fig5_data.py, fig5_export_top.py) and the dot-plot p bins (Fig. 6's dot plots, fig6_pmd_plot.py). The panels
themselves are drawn by fig5_final.py.
"""
from __future__ import annotations

CT = ['HSC', 'Hepatocyte', 'Kupffer Cell', 'LSEC', 'T cell']

# Panel b rows: (label, Reactome pathway, literature status, reference). Status 'established' needs a
# human/rodent ALD paper on that process; 'not reported' = no PubMed hit for the process x ALD.
B_ROWS = [  # (label, family, set, block, reference)
    ('mRNA export to cytoplasm', 'reactome', 'Transport of Mature Transcript to Cytoplasm', 'new', ''),
    ('nuclear pore complex', 'npc', 'nuclear pore', 'new', ''),
    ('TREX mRNA export', 'module', 'rna_export_trex', 'new', ''),
    ('Pol II elongation', 'reactome', 'RNA Polymerase II Transcription Elongation', 'new', ''),
    ('Pol II pause brake (7SK / NELF)', 'custom', 'pause brake', 'new', ''),
    # what the dataset's own paper reports (Chembazhi et al. 2025, Nat Commun, same cohort)
    ('mRNA splicing', 'reactome', 'mRNA Splicing - Major Pathway', 'cohort', 'Chembazhi 2025'),
    ('adult hepatocyte TFs', 'module', 'adult_hep_tf', 'cohort', 'Chembazhi 2025'),
    ('WNT / TCF signalling', 'reactome', 'Signaling by WNT', 'cohort', 'Chembazhi 2025'),
    ('Hippo signalling', 'reactome', 'Signaling by Hippo', 'cohort', 'Chembazhi 2025'),
    ('TGF-β signalling', 'reactome', 'Signaling by TGFB Family Members', 'cohort', 'Chembazhi 2025'),
    ('IL-1 signalling', 'reactome', 'Interleukin-1 Signaling', 'cohort', 'Chembazhi 2025'),
    ('TNFR1 → NF-κB', 'reactome', 'TNFR1-induced NF-kappa-B Signaling Pathway', 'cohort', 'Chembazhi 2025'),
    ('PLK1 / mitotic entry', 'reactome', 'Polo-like Kinase Mediated Events', 'cohort', 'Chembazhi 2025'),
    # other ALD literature (PubMed-checked 2026-10-12)
    ('cytosolic translation', 'reactome', 'Cap-dependent Translation Initiation', 'literature', 'Rothschild 1971'),
    ('mitochondrial translation', 'reactome', 'Mitochondrial Translation', 'literature', 'Hao 2020'),
    ('SUMOylation', 'reactome', 'SUMOylation', 'literature', 'Chandla 2026'),
    ('chromatin modification', 'reactome', 'Chromatin Modifying Enzymes', 'literature', 'Ghare 2023'),
    ('macroautophagy', 'reactome', 'Macroautophagy', 'literature', 'Chao 2018'),
    ('amino acids → mTORC1', 'reactome', 'Amino Acids Regulate mTORC1', 'literature', 'Davuluri 2021'),
    ('proteasome assembly', 'reactome', 'Proteasome Assembly', 'literature', 'Wang 2016'),
]

P_BINS = [(1e-3, 'p < 0.001', 380), (1e-2, 'p < 0.01', 270), (5e-2, 'p < 0.05', 180), (1.01, 'p ≥ 0.05', 80)]


def p_size(p):
    for thr, _, s in P_BINS:
        if p < thr:
            return s
    return P_BINS[-1][2]
