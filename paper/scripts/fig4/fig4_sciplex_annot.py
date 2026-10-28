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

"""Annotation inputs of the sci-Plex3 comparison (Fig. 4B-D, S8, Table S8), built from our own files.

    candidates.npz       'genes': the matched panel = fingerprinting dictionary (columns of a per-line pip.csv)
                         intersected with ExPert's trained prototypes (bank labels[:n_labels]); 3,802 for v0.1.0
    drug_family.csv      drug -> sci-Plex family label(s) of the EVALUABLE drugs: the drug's target_annotation
                         (comma-separated families) maps, through the curated family gene table, to at least one
                         candidate gene (109 drugs)
    corum_core_sets.csv  per family, its CORUM-core partners inside the panel: complexes of <= 20 members that contain
                         a family target; partner score = (# such complexes with the gene) / sqrt(# such complexes x
                         # CORUM complexes with the gene anywhere), kept at >= 0.15 (cap and threshold fixed by
                         agreement with the hand-written HDAC core list, never by method ranks)
    chembl_targets.csv   drug-specific primary targets from the ChEMBL REST API (curated mechanism, else
                         single-protein pChEMBL >= 7 within 1 log of the best), plus the manual fixes below; raw API
                         responses are cached in --chembl-cache, so a rerun with the cache is offline and exact

The curated inputs (family gene table, HDAC complex list) and the rules above are J. Schlamowitz's
(sci-Plex fingerprinting benchmark, 2026-10); the code is a port of his 26/50/56 scripts without the plotting.
"""
from __future__ import annotations

import argparse, collections, json, re, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

API = 'https://www.ebi.ac.uk/chembl/api/data'


def candidates(pip_csv, bank_npz):
    dic = set(pd.read_csv(pip_csv, index_col=0, nrows=1).columns.astype(str))
    b = np.load(bank_npz, allow_pickle=True)
    trained = set(b['labels'].astype(str)[:int(b['n_labels'])])
    return np.array(sorted(dic & trained))


def drug_family(targets_tsv, fam_genes, cand):
    t = pd.read_csv(targets_tsv, sep='\t')
    out = []
    for d, ann in zip(t.drug.astype(str).str.strip(), t.target_annotation.astype(str)):
        labs = [x.strip() for x in ann.split(',') if x.strip()]
        if any(g in cand for l in labs for g in fam_genes.get(l, [])):
            out.append((d, ','.join(labs)))
    return pd.DataFrame(out, columns=['drug', 'family']).sort_values('drug', ignore_index=True)


def corum_core_sets(gmt, fam_genes, cand, max_size=20, th=0.15):
    corum = {}
    for line in open(gmt):
        q = line.rstrip('\n').split('\t')
        if q[0].endswith('(human)'):
            corum[q[0]] = {g for g in q[2:] if g}
    deg = collections.Counter(g for v in corum.values() for g in v)
    rows = []
    for f, t in fam_genes.items():
        t = set(t)
        hit = [n for n, v in corum.items() if v & t and len(v) <= max_size]
        co = collections.Counter(g for n in hit for g in corum[n] - t)
        sc = {g: c / np.sqrt(len(hit) * deg[g]) for g, c in co.items()}
        keep = sorted(sorted(g for g, v in sc.items() if v >= th and g in cand), key=lambda g: -sc[g])  # ties A-Z
        rows.append(dict(family=f, n_complexes=len(hit),
                         partners_in_candidates=';'.join(f'{g}:{sc[g]:.2f}' for g in keep)))
    return pd.DataFrame(rows)


# ---- ChEMBL --------------------------------------------------------------------------------------
# Manual fixes after checking every non-exact name match by molecular formula (2026-10-13): the first search hit was a
# different compound for these; SRT3025 is not in ChEMBL.
OVERRIDE = {'Epothilone A': ('CHEMBL329993', 'EPOTHILONE A'), 'JNJ-7706621': ('CHEMBL191003', 'JNJ-7706621'),
            'AG-14361': ('CHEMBL65892', 'AG-14361'), 'SRT3025 HCl': (None, None)}
# Literature targets for drugs ChEMBL gives no target, or a target that is an assay artefact or an unrelated mechanism
# (original reports / vendor annotation, not a database).
MANUAL = {
    'Droxinostat': 'HDAC3;HDAC6;HDAC8', 'MC1568': 'HDAC4;HDAC5;HDAC7;HDAC9', 'Sodium Phenylbutyrate': 'HDAC1;HDAC2;HDAC3',
    'Valproic acid sodium salt (Sodium valproate)': 'HDAC1;HDAC2;HDAC3', 'Divalproex Sodium': 'HDAC1;HDAC2;HDAC3',
    'INO-1001 (3-Aminobenzamide)': 'PARP1;PARP2', 'Obatoclax Mesylate (GX15-070)': 'BCL2;BCL2L1;MCL1',
    'MK-0752': 'PSEN1;PSEN2', 'BMS-265246': 'CDK1;CDK2', 'RG108': 'DNMT1', 'JNJ-26854165 (Serdemetan)': 'MDM2',
    'AG-490 (Tyrphostin B42)': 'JAK2;EGFR', 'WHI-P154': 'JAK3;EGFR', 'Toremifene Citrate': 'ESR1;ESR2',
    'FLLL32': 'JAK2;STAT3', 'WP1066': 'JAK2;STAT3', 'Anacardic Acid': 'EP300;KAT2B', 'BRD4770': 'EHMT2',
    'UNC0379': 'KMT5A', 'PD98059': 'MAP2K1;MAP2K2', 'SL-327': 'MAP2K1;MAP2K2', 'Epothilone A': 'TUBB',
    'Daphnetin': 'PRKACA;PRKCA;EGFR', 'S3I-201': 'STAT3', 'SRT1720 HCl': 'SIRT1', 'SRT3025 HCl': 'SIRT1',
    'Sirtinol': 'SIRT1;SIRT2', 'YM155 (Sepantronium Bromide)': 'BIRC5', 'Tie2 kinase inhibitor': 'TEK',
    'Pirarubicin': 'TOP2A;TOP2B'}
EXCLUDE = {'Curcumin': 'promiscuous (PAINS); ChEMBL potency target APP',
           'Cyclocytidine HCl': 'ChEMBL potency target LMNA (HTS artefact); prodrug of cytarabine'}
SALT = (r'\b(hcl|2hcl|dihydrochloride|hydrochloride|sodium|salt|phosphate|citrate|mesylate|maleate|tosylate|sulfate|'
        r'acetate|tartrate|fumarate|succinate|lactate|dimaleate|hbr|hydrobromide|potassium|calcium|monohydrate|hydrate|'
        r'free base|ditosylate|besylate)\b')
FUSION_KINASES = {'ABL1', 'ALK', 'ROS1', 'RET'}
_norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())


class Chembl:
    def __init__(self, cache):
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True); self.tc = {}

    def get(self, path, **params):
        key = re.sub(r'[^A-Za-z0-9]+', '_', path + '_' + '_'.join(f'{k}{v}' for k, v in sorted(params.items())))[:180]
        f = self.cache / f'{key}.json'
        if f.exists():
            return json.loads(f.read_text())
        url = f'{API}/{path}.json?' + urllib.parse.urlencode(params)
        for i in range(4):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=60)); f.write_text(json.dumps(d)); return d
            except Exception as e:
                err = e; time.sleep(2 * (i + 1))
        raise err

    def target_genes(self, tid):
        if tid not in self.tc:
            t = self.get(f'target/{tid}')
            genes = []
            if t.get('organism') == 'Homo sapiens':
                for comp in t.get('target_components', []):
                    genes += [s['component_synonym'] for s in comp.get('target_component_synonyms', [])
                              if s['syn_type'] == 'GENE_SYMBOL'][:1]
            self.tc[tid] = (t.get('pref_name'), t.get('target_type'), genes)
        return self.tc[tid]

    @staticmethod
    def queries(name):
        main = re.sub(r'\(.*?\)', '', name).strip()
        qs = [name, main]
        for al in re.findall(r'\((.*?)\)', name):
            qs += [x.strip() for x in re.split(r'[,;]', al) if x.strip()]
        qs += [re.sub(SALT, '', main, flags=re.I).strip(), re.sub(SALT, '', main, flags=re.I).replace('?', '').strip()]
        out = []
        for q in qs:
            q = re.sub(r'\s+', ' ', q.replace('?', ' ')).strip()
            if q and q not in out and len(q) > 2:
                out.append(q)
        return out

    def molecule(self, name):
        first = None
        for q in self.queries(name):
            hits = self.get('molecule/search', q=q, limit=10).get('molecules', [])
            for h in hits:
                names = [h.get('pref_name') or ''] + [s['molecule_synonym'] for s in h.get('molecule_synonyms', [])]
                if _norm(q) in {_norm(n) for n in names if n}:
                    return ((h['molecule_hierarchy'] or {}).get('parent_chembl_id') or h['molecule_chembl_id'],
                            h.get('pref_name'), q, 'exact')
            if hits and first is None:
                h = hits[0]
                first = ((h['molecule_hierarchy'] or {}).get('parent_chembl_id') or h['molecule_chembl_id'],
                         h.get('pref_name'), q, 'first_hit')
        return first or (None, None, None, 'not_found')

    def one(self, d):
        if d in OVERRIDE:
            cid, pname = OVERRIDE[d]; q, match = d, 'manual' if cid else 'not_found'
        else:
            cid, pname, q, match = self.molecule(d)
        r = dict(drug=d, chembl_id=cid, chembl_name=pname, query=q, match=match, source='none', targets='',
                 target_names='')
        if not cid:
            return r
        # potency over human SINGLE PROTEIN targets (pChEMBL >= 6); BCR-ABL assays credit both genes
        pot = {}
        for x in self.get('activity', molecule_chembl_id=cid, pchembl_value__gte=6, target_organism='Homo sapiens',
                          limit=1000).get('activities', []):
            if x.get('pchembl_value') and x.get('target_chembl_id'):
                _, tt, g = self.target_genes(x['target_chembl_id'])
                if tt == 'SINGLE PROTEIN' and g:
                    pot[g[0]] = max(pot.get(g[0], 0), float(x['pchembl_value']))
                elif tt == 'CHIMERIC PROTEIN':
                    for gg in g:
                        pot[gg] = max(pot.get(gg, 0), float(x['pchembl_value']))
        genes, tn = [], []
        for m in self.get('mechanism', molecule_chembl_id=cid, limit=100).get('mechanisms', []):
            if not m.get('target_chembl_id'):
                continue
            n, tt, g = self.target_genes(m['target_chembl_id'])
            if not g:
                continue
            tn.append(f"{n} [{tt}; {m.get('action_type')}]")
            if tt in ('SINGLE PROTEIN', 'CHIMERIC PROTEIN'):
                genes += g
            else:
                # family / complex target: members the drug hits within 1 log of its best member, else all members
                hp = {x: pot[x] for x in g if x in pot}
                genes += [x for x, v in hp.items() if v >= max(hp.values()) - 1] if hp else g
        if genes:
            r.update(source='mechanism', targets=';'.join(sorted(set(genes))), target_names=' | '.join(sorted(set(tn))))
        elif pot and max(pot.values()) >= 7:
            top = max(pot.values()); keep = sorted(x for x, v in pot.items() if v >= max(7, top - 1))
            r.update(source='potency', targets=';'.join(keep), target_names=f'pChEMBL max {top:.1f}; {len(keep)} targets within 1 log')
        return r


def chembl_targets(drugs, cache, threads=8):
    c = Chembl(cache)
    with ThreadPoolExecutor(threads) as ex:
        T = pd.DataFrame(list(ex.map(c.one, drugs)))
    for d, g in MANUAL.items():
        i = T.index[T.drug == d]
        assert len(i) == 1, d
        T.loc[i, ['source', 'targets', 'target_names']] = ['manual', g, f"literature; ChEMBL gave: {T.loc[i[0], 'targets'] or 'none'}"]
    for d, why in EXCLUDE.items():
        T.loc[T.drug == d, ['source', 'targets', 'target_names']] = ['excluded', '', why]
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets-tsv', required=True, help='sciplex3_drug_targets.tsv (drug, target_annotation)')
    ap.add_argument('--family-genes', required=True, help='curated sci-Plex family -> genes csv')
    ap.add_argument('--fp-pip', required=True, help='pip.csv of a per-line fingerprinting run (dictionary columns)')
    ap.add_argument('--bank', required=True, help='ExPert prototype bank.npz (labels, n_labels)')
    ap.add_argument('--corum-gmt', required=True, help='Enrichr CORUM library (gmt)')
    ap.add_argument('--chembl-cache', required=True, help='dir of raw ChEMBL API responses (filled on first run)')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    fam = {r.family: [g for g in str(r.genes).split(';') if g]
           for r in pd.read_csv(a.family_genes).itertuples()}
    cand = candidates(a.fp_pip, a.bank)
    np.savez_compressed(out / 'candidates.npz', genes=cand)
    cs = set(cand)
    df = drug_family(a.targets_tsv, fam, cs); df.to_csv(out / 'drug_family.csv', index=False)
    cs_sets = corum_core_sets(a.corum_gmt, fam, cs); cs_sets.to_csv(out / 'corum_core_sets.csv', index=False)
    drugs = sorted(pd.read_csv(a.targets_tsv, sep='\t').drug.astype(str).str.strip().unique())
    ch = chembl_targets(drugs, a.chembl_cache); ch.to_csv(out / 'chembl_targets.csv', index=False)
    print(f'[annot] candidates {len(cand)} | evaluable drugs {len(df)} | families {len(cs_sets)} | '
          f'ChEMBL {len(ch)} drugs: {ch.source.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
