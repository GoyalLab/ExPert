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

"""Fig. 5: curated sci-Plex3 drug -> target-gene table.

sci-Plex3 annotates each compound with a protein FAMILY ('HDAC', 'Aurora Kinase', 'JAK'), not a
gene. Recovery needs gene symbols, so every drug gets an explicit list:

    drug-specific   selective compounds get their known targets (alisertib -> AURKA, PCI-34051 ->
                    HDAC8, tazemetostat -> EZH2), and mis-annotations are corrected (entacapone is
                    a COMT inhibitor annotated 'Histone Methyltransferase'; G007-LK is tankyrase-
                    selective, annotated 'PARP')
    family          anything else expands each family token to its genes (union)
    lof             1 = inhibitor / antagonist / degrader: the knockdown analogy holds. 0 = agonist,
                    activator, stabiliser, neo-substrate modulator or no human gene target (DNA
                    alkylators, bacterial/viral targets). lof=0 drugs are scored separately, never
                    pooled into the headline

Input: a per-drug CSV with columns perturbation, target, pw (from the sci-Plex3 obs).
"""
from __future__ import annotations

import argparse

import pandas as pd

FAM = {
    'ALK': 'ALK', 'c-Met': 'MET', 'AMPK': 'PRKAA1;PRKAA2', 'Androgen Receptor': 'AR',
    'Aromatase': 'CYP19A1', 'Aurora Kinase': 'AURKA;AURKB;AURKC', 'Bcr-Abl': 'ABL1', 'FLT3': 'FLT3',
    'JAK': 'JAK1;JAK2;JAK3;TYK2', 'c-RET': 'RET', 'FGFR': 'FGFR1;FGFR2;FGFR3;FGFR4',
    'CDK': 'CDK1;CDK2;CDK4;CDK6;CDK7;CDK9', 'VEGFR': 'KDR;FLT1;FLT4', 'ROCK': 'ROCK1;ROCK2',
    'Sirtuin': 'SIRT1', 'Bcl-2': 'BCL2;BCL2L1;BCL2L2', 'c-Kit': 'KIT', 'Src': 'SRC',
    'Gamma-secretase': 'PSEN1;PSEN2;NCSTN;APH1A;PSENEN', 'CCR': 'CCR5', 'COX': 'PTGS2;PTGS1',
    'CSF-1R': 'CSF1R', 'PDGFR': 'PDGFRA;PDGFRB', 'DNA Methyltransferase': 'DNMT1;DNMT3A;DNMT3B',
    'Dehydrogenase': 'ALDH2;ALDH1A1', 'Dopamine Receptor': 'DRD2;DRD3', 'TNF-alpha': 'TNF',
    'EGFR': 'EGFR', 'HER2': 'ERBB2', 'HDAC': 'HDAC1;HDAC2;HDAC3;HDAC6;HDAC8',
    'Estrogen/progestogen Receptor': 'ESR1', 'FAAH': 'FAAH', 'FAK': 'PTK2',
    'Glucocorticoid Receptor': 'NR3C1', 'PI3K': 'PIK3CA;PIK3CB;PIK3CD;PIK3CG',
    'HSP (e.g. HSP90)': 'HSP90AA1;HSP90AB1', 'Histamine Receptor': 'HRH2',
    'Histone Acetyltransferase': 'EP300;KAT2B;KAT5', 'IDH1': 'IDH1', 'IGF-1R': 'IGF1R',
    'LPA Receptor': 'LPAR1;LPAR3', 'Lipoxygenase': 'ALOX5', 'MAO': 'MAOA;MAOB',
    'MEK': 'MAP2K1;MAP2K2', 'MT Receptor': 'MTNR1A;MTNR1B', 'PARP': 'PARP1;PARP2',
    'Raf': 'RAF1;BRAF', 'PKA': 'PRKACA', 'PKC': 'PRKCA;PRKCB', 'PLK': 'PLK1', 'Pim': 'PIM1;PIM2;PIM3',
    'STAT': 'STAT3', 'Survivin': 'BIRC5', 'TGF-beta/Smad': 'TGFBR1;ACVR1B;ACVR1C',
    'Telomerase': 'TERT', 'Tie-2': 'TEK', 'Trk receptor': 'NTRK1;NTRK2;NTRK3',
    'Wnt/beta-catenin': 'TNKS;TNKS2', 'mTOR': 'MTOR', 'Microtubule Associated': 'TUBB',
    'Topoisomerase': 'TOP2A;TOP2B', 'NF-κB': 'NFKB1;RELA', 'Nrf2': 'NFE2L2',
}
# tokens that carry no gene on their own (always covered by a drug-specific entry or a co-token)
NO_GENE = {'Autophagy', 'E3 Ligase', 'p53', 'Beta Amyloid', 'GABA Receptor',
           'Immunology & Inflammation related', 'DNA alkylator', 'DNA/RNA Synthesis', 'Others', 'HIF',
           'Epigenetic Reader Domain', 'Histone Methyltransferase', 'Histone Demethylase'}

# (genes, lof, note)
DRUG = {
    'AICAR (Acadesine)': ('PRKAA1;PRKAA2', 0, 'AMPK activator'),
    'Andarine': ('AR', 0, 'SARM agonist'),
    'Triamcinolone Acetonide': ('NR3C1', 0, 'GR agonist'),
    'Meprednisone': ('NR3C1', 0, 'GR agonist'),
    'Prednisone': ('NR3C1', 0, 'GR agonist'),
    'Resveratrol': ('SIRT1', 0, 'sirtuin activator'),
    'SRT3025 HCl': ('SIRT1', 0, 'SIRT1 activator'),
    'SRT2104 (GSK2245840)': ('SIRT1', 0, 'SIRT1 activator'),
    'SRT1720 HCl': ('SIRT1', 0, 'SIRT1 activator'),
    'Ramelteon': ('MTNR1A;MTNR1B', 0, 'MT receptor agonist'),
    'ITSA-1 (ITSA1)': ('HDAC1;HDAC2;HDAC3', 0, 'HDAC activator (TSA suppressor)'),
    'Patupilone (EPO906, Epothilone B)': ('TUBB', 0, 'microtubule stabiliser'),
    'Epothilone A': ('TUBB', 0, 'microtubule stabiliser'),
    'Ofloxacin': ('', 0, 'bacterial gyrase; no human target'),
    'Clevudine': ('', 0, 'viral polymerase; no human target'),
    'Altretamine': ('', 0, 'DNA alkylator'),
    'Busulfan': ('', 0, 'DNA alkylator'),
    'Streptozotocin (STZ)': ('', 0, 'DNA alkylator'),
    'Lomustine': ('', 0, 'DNA alkylator'),
    'Thiotepa': ('', 0, 'DNA alkylator'),
    'Mesna': ('', 0, 'antioxidant; no target'),
    'Cyclocytidine HCl': ('', 0, 'cytarabine prodrug; DNA synthesis'),
    'Thalidomide': ('CRBN', 0, 'CRBN modulator (neo-substrate degrader)'),
    'Lenalidomide (CC-5013)': ('CRBN;IKZF1;IKZF3', 0, 'CRBN modulator; IKZF1/3 degraded'),
    'Raltitrexed': ('TYMS', 1, ''),
    'Fluorouracil (5-Fluoracil, 5-FU)': ('TYMS', 1, ''),
    'Capecitabine': ('TYMS', 1, '5-FU prodrug'),
    'Carmofur': ('TYMS;ASAH1', 1, ''),
    'Mercaptopurine (6-MP)': ('PPAT;IMPDH1;IMPDH2', 1, ''),
    'Alendronate sodium trihydrate': ('FDPS', 1, ''),
    'Aminoglutethimide': ('CYP19A1;CYP11A1', 1, ''),
    'Disulfiram': ('ALDH2;ALDH1A1', 1, ''),
    'Entacapone': ('COMT', 1, 'annotated HMT; COMT inhibitor'),
    'Tranylcypromine (2-PCPA) HCl': ('MAOA;MAOB;KDM1A', 1, ''),
    'Pirarubicin': ('TOP2A;TOP2B', 1, ''),
    'Ellagic acid': ('TOP1;TOP2A', 1, 'weak'),
    'Costunolide': ('TERT', 1, 'weak'),
    'JNJ-26854165 (Serdemetan)': ('MDM2', 1, ''),
    'YM155 (Sepantronium Bromide)': ('BIRC5', 1, ''),
    'XAV-939': ('TNKS;TNKS2', 1, ''),
    'G007-LK': ('TNKS;TNKS2', 1, 'annotated PARP; tankyrase-selective'),
    'Roxadustat (FG-4592)': ('EGLN1;EGLN2;EGLN3', 1, 'PHD inhibitor (HIF stabiliser)'),
    'IOX2': ('EGLN1', 1, 'PHD2 inhibitor (HIF stabiliser)'),
    '2-Methoxyestradiol (2-MeOE2)': ('HIF1A', 1, ''),
    'Alisertib (MLN8237)': ('AURKA', 1, ''),
    'MLN8054': ('AURKA', 1, ''),
    'MK-5108 (VX-689)': ('AURKA', 1, ''),
    'Aurora A Inhibitor I': ('AURKA', 1, ''),
    'Barasertib (AZD1152-HQPA)': ('AURKB', 1, ''),
    'Hesperadin': ('AURKB', 1, ''),
    'ZM 447439': ('AURKB;AURKA', 1, ''),
    'GSK1070916': ('AURKB;AURKC', 1, ''),
    'PCI-34051': ('HDAC8', 1, ''),
    'Tubastatin A HCl': ('HDAC6', 1, ''),
    'TMP195': ('HDAC4;HDAC5;HDAC7;HDAC9', 1, ''),
    'MC1568': ('HDAC4;HDAC5;HDAC6', 1, ''),
    'Mocetinostat (MGCD0103)': ('HDAC1;HDAC2;HDAC3;HDAC11', 1, ''),
    'Entinostat (MS-275)': ('HDAC1;HDAC2;HDAC3', 1, ''),
    'Tacedinaline (CI994)': ('HDAC1;HDAC2;HDAC3', 1, ''),
    'Tucidinostat (Chidamide)': ('HDAC1;HDAC2;HDAC3;HDAC10', 1, ''),
    'Droxinostat': ('HDAC3;HDAC6;HDAC8', 1, ''),
    'M344': ('HDAC1;HDAC6', 1, ''),
    'Valproic acid sodium salt (Sodium valproate)': ('HDAC1;HDAC2;HDAC3', 1, ''),
    'Divalproex Sodium': ('HDAC1;HDAC2;HDAC3', 1, ''),
    'Sodium Phenylbutyrate': ('HDAC1;HDAC2;HDAC3', 1, ''),
    'CUDC-907': ('HDAC1;HDAC2;HDAC3;HDAC10;PIK3CA;PIK3CB;PIK3CD', 1, ''),
    'CUDC-101': ('HDAC1;HDAC2;HDAC3;EGFR;ERBB2', 1, ''),
    '(+)-JQ1': ('BRD4;BRD2;BRD3', 1, ''),
    'PFI-1 (PF-6405761)': ('BRD4;BRD2', 1, ''),
    'EED226': ('EED', 1, 'PRC2 EED binder'),
    'Anacardic Acid': ('EP300;KAT2B;KAT5', 1, ''),
    'GSK J1': ('KDM6B;KDM6A', 1, ''),
    'GSK-LSD1 2HCl': ('KDM1A', 1, ''),
    'UNC0379': ('KMT5A', 1, ''),
    'Tazemetostat (EPZ-6438)': ('EZH2', 1, ''),
    'UNC0631': ('EHMT2;EHMT1', 1, ''),
    'UNC1999': ('EZH2;EZH1', 1, ''),
    'BRD4770': ('EHMT2', 1, ''),
    'A-366': ('EHMT2;EHMT1', 1, ''),
    'RG108': ('DNMT1', 1, ''),
    'BMS-536924': ('IGF1R;INSR', 1, ''),
    'BMS-754807': ('IGF1R;INSR;MET;NTRK1;NTRK2;NTRK3', 1, ''),
    'Ruxolitinib (INCB018424)': ('JAK1;JAK2', 1, ''),
    'S-Ruxolitinib (INCB018424)': ('JAK1;JAK2', 1, ''),
    'Baricitinib (LY3009104, INCB028050)': ('JAK1;JAK2', 1, ''),
    'Tofacitinib (CP-690550) Citrate': ('JAK1;JAK3', 1, ''),
    'Fedratinib (SAR302503, TG101348)': ('JAK2', 1, ''),
    'Filgotinib (GLPG0634)': ('JAK1', 1, ''),
    'Gandotinib (LY2784544)': ('JAK2', 1, ''),
    'AZD1480': ('JAK2;JAK1', 1, ''),
    'NVP-BSK805 2HCl': ('JAK2', 1, ''),
    'BMS-911543': ('JAK2', 1, ''),
    'CEP-33779': ('JAK2', 1, ''),
    'AZ 960': ('JAK2', 1, ''),
    'Momelotinib (CYT387)': ('JAK1;JAK2', 1, ''),
    'Cerdulatinib (PRT062070, PRT2070)': ('JAK1;JAK2;JAK3;TYK2;SYK', 1, ''),
    'WP1066': ('JAK2;STAT3', 1, ''),
    'FLLL32': ('JAK2;STAT3', 1, ''),
    'Flavopiridol HCl': ('CDK9;CDK1;CDK2;CDK4;CDK6;CDK7', 1, ''),
    'Roscovitine (Seliciclib,CYC202)': ('CDK1;CDK2;CDK5;CDK7;CDK9', 1, ''),
    'BMS-265246': ('CDK1;CDK2', 1, ''),
    'Obatoclax Mesylate (GX15-070)': ('BCL2;BCL2L1;BCL2L2;MCL1', 1, ''),
    'Nilotinib (AMN-107)': ('ABL1;KIT;PDGFRA;PDGFRB', 1, ''),
    'Dasatinib': ('ABL1;SRC;KIT;LCK;YES1;FYN', 1, ''),
    'Bosutinib (SKI-606)': ('SRC;ABL1', 1, ''),
    'TGX-221': ('PIK3CB', 1, ''),
    'Daphnetin': ('PRKACA;EGFR;PRKCA', 1, 'weak'),
    'Enzastaurin (LY317615)': ('PRKCB', 1, ''),
    'Bisindolylmaleimide IX (Ro 31-8220 Mesylate)': ('PRKCA;PRKCB;PRKCG;GSK3B', 1, ''),
    'PD98059': ('MAP2K1', 1, ''),
    'Temsirolimus (CCI-779, NSC 683864)': ('MTOR;FKBP1A', 1, ''),
    'Selisistat (EX 527)': ('SIRT1', 1, ''),
    'Sirtinol': ('SIRT1;SIRT2', 1, ''),
    'Quercetin': ('SRC;PIK3CG;PRKCA;SIRT1', 1, 'promiscuous'),
    'Curcumin': ('NFKB1;RELA;EP300;HDAC1;NFE2L2', 1, 'promiscuous'),
    'Iniparib (BSI-201)': ('PARP1', 1, 'not a true PARP inhibitor'),
    'Fulvestrant': ('ESR1', 1, 'ER degrader'),
    'Fasudil (HA-1077) HCl': ('ROCK1;ROCK2', 1, ''),
    'Linifanib (ABT-869)': ('KDR;FLT1;FLT4;PDGFRA;PDGFRB;CSF1R;FLT3', 1, ''),
}


def main(a):
    t = pd.read_csv(a.drugs_csv)
    rows = []
    for r in t.itertuples():
        d = r.perturbation                       # kept verbatim: source names carry trailing spaces
        if d.strip() == 'control':
            continue
        if d.strip() in DRUG:
            g, lof, note = DRUG[d.strip()]
        else:
            toks = [x.strip() for x in str(r.target).split(',') if x.strip()]
            miss = [x for x in toks if x not in FAM and x not in NO_GENE]
            if miss:
                raise SystemExit(f'{d!r}: unmapped family tokens {miss}')
            gs = []
            for x in toks:
                for gg in FAM.get(x, '').split(';'):
                    if gg and gg not in gs:
                        gs.append(gg)
            g, lof, note = ';'.join(gs), 1, 'family expansion'
        rows.append(dict(drug=d, target_annotation=r.target, pathway=r.pw, genes=g, lof=lof, note=note))
    df = pd.DataFrame(rows)
    bad = df[(df.lof == 1) & (df.genes == '')]
    if len(bad):
        raise SystemExit(f'lof drugs without genes: {list(bad.drug)}')
    df.to_csv(a.out, sep='\t', index=False)
    print(f'{len(df)} drugs; lof {int(df.lof.sum())}; no gene {int((df.genes == "").sum())}; '
          f'family-expanded {int((df.note == "family expansion").sum())}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--drugs-csv', required=True)
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
