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

"""Fig. 5 readout grid scored against known ALD biology (v0.1.0, mito_mt input, no hubness correction).

One task per encoding unit x bank; inside, every contrast x readout x scoring.
  unit      cells_cos   per-nucleus cos(z_cell - reference, P), averaged per donor x cell type (single-cell readout;
                        deflate is a no-op on single nuclei, so cosine only)
            cells_z     nuclei averaged in z per donor x cell type, then contrasted
            group       one pseudobulk per donor x cell type (fig5_predict.py encode, current predict())
            pools       10 pseudobulk pools per donor x cell type averaged in z (fig5_variants pbulk_x)
  contrast  ctrl (donor - mean of the Normal donors), rest (donor - every donor outside its condition), rrep (condition
            - mean of its donor set's AC / AH / Normal); a Normal donor is compared with the other Normal donor
  readout   cosine | deflate (predict()'s deflate_rows + transplant_scores at the contrast's depth)
  scoring   setz (whole-ranking mean-percentile set z, loss +) | ora (top-200 hypergeometric, per tail)
Significance: empirical p against every Normal-row test of the same combination over all Reactome sets + modules.
Targets (TARGETS below): literature expectations + the RNA-FISH export result, each with condition / cell type /
direction; a donor x cell type counts as a hit when p_emp < .05 in the expected direction, 'wrong' when p_emp < .05 in
the opposite one.

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_readout_grid.py --task 0 ...
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig5_centring_audit import NPC, unit, pct_of, set_z
from fig5_data import read_gmt, export_parts

CT = ['HSC', 'Hepatocyte', 'Kupffer Cell', 'LSEC', 'T cell']
# (label, set key, conditions, cell types, expected direction, reference)
TARGETS = [
    ('mRNA export (grouped)', 'export', ('AC', 'AH'), ('Hepatocyte',), 'loss', 'RNA FISH (this study)'),
    ('mRNA splicing', 'r:mRNA Splicing - Major Pathway', ('AH',), ('Hepatocyte',), 'loss', 'Chembazhi 2025'),
    ('TGF-β signalling', 'r:Signaling by TGFB Family Members', ('AC', 'AH'), tuple(CT), 'any', 'Chembazhi 2025'),
    ('adult hepatocyte TFs', 'm:adult_hep_tf', ('AC', 'AH'), ('Hepatocyte',), 'loss', 'Chembazhi 2025'),
    ('mitochondrial translation', 'r:Mitochondrial Translation', ('AH',), ('Hepatocyte',), 'gain', 'Hao 2020'),
    ('chromatin modification', 'r:Chromatin Modifying Enzymes', ('AH',), ('Hepatocyte',), 'loss', 'Ghare 2023'),
    ('SUMOylation', 'r:SUMOylation', ('AC', 'AH'), tuple(CT), 'loss', 'Chandla 2026'),
    ('cytosolic translation', 'r:Cap-dependent Translation Initiation', ('AC', 'AH'), ('Hepatocyte',), 'loss',
     'Rothschild 1971'),
    ('proteasome assembly', 'r:Proteasome Assembly', ('AC', 'AH'), ('Hepatocyte',), 'loss', 'Wang 2016'),
    ('macroautophagy', 'r:Macroautophagy', ('AH',), ('Hepatocyte',), 'loss', 'Chao 2018'),
    ('amino acids → mTORC1', 'r:Amino Acids Regulate mTORC1', ('AC', 'AH'), ('Hepatocyte',), 'any', 'Davuluri 2021'),
]


def group_means(Z, obs, min_cells=50):
    """(donor, celltype) -> (mean z, n)."""
    out = {}
    for (d, t), ix in obs.groupby(['donor', 'celltype']).indices.items():
        if t in CT and len(ix) >= min_cells:
            out[(d, t)] = (Z[ix].astype(np.float64).mean(0), len(ix))
    return out


def references(G, kind):
    """(donor, celltype) -> list of reference donors for the contrast."""
    cond = lambda d: 'Normal' if d.startswith('Nor') else d[:2]
    ref = {}
    for t in CT:
        ds = sorted(d for d, tt in G if tt == t)
        nor = [d for d in ds if cond(d) == 'Normal']
        for d in ds:
            if cond(d) == 'Normal':
                r = [e for e in nor if e != d]
            elif kind == 'ctrl':
                r = nor
            elif kind == 'rest':
                r = [e for e in ds if cond(e) != cond(d)]
            else:                                           # rrep: the donor's own set {AC_k, AH_k, Nor_k}
                k = d[-1]
                r = [e for e in ds if e.endswith(k) and e != d]
                if not any(cond(e) == 'Normal' for e in r):
                    r += nor[:1]
            if r:
                ref[(d, t)] = r
    return ref


def contrasts(G, kind):
    """keys, Q, n_eff. rrep = d - mean(d and its set), the others d - mean(refs)."""
    ref = references(G, kind)
    keys, Q, ne = [], [], []
    for (d, t), r in sorted(ref.items()):
        zd, nd = G[(d, t)]
        if kind == 'rrep':
            members = [d] + r
            Q.append(zd - np.mean([G[(e, t)][0] for e in members], 0))
            ne.append(9.0 / (4.0 / nd + sum(1.0 / G[(e, t)][1] for e in r)))
        else:
            Q.append(zd - np.mean([G[(e, t)][0] for e in r], 0))
            ne.append(1.0 / (1.0 / nd + sum(1.0 / G[(e, t)][1] for e in r) / len(r) ** 2))
        keys.append((d, t))
    return keys, np.stack(Q), np.asarray(ne)


def cell_profiles(Z, obs, G, kind, Pn):
    """Per-nucleus cos(z - reference mean) averaged per donor x cell type (reference as in `contrasts`)."""
    ref = references(G, kind)
    idx = obs.groupby(['donor', 'celltype']).indices
    keys, S = [], []
    for (d, t), r in sorted(ref.items()):
        members = [d] + r if kind == 'rrep' else r
        c = np.mean([G[(e, t)][0] for e in members], 0)
        ix = idx[(d, t)]
        acc = np.zeros(Pn.shape[0])
        for i in range(0, len(ix), 4096):
            acc += (unit(Z[ix[i:i + 4096]].astype(np.float64) - c) @ Pn.T).sum(0)
        keys.append((d, t))
        S.append(acc / len(ix))
    return keys, np.stack(S)


def deflated(Q, P, ne, noise):
    import expert.utils.readout as ro
    S1 = unit(Q) @ unit(P).T
    dr = ro.deflate_rows(Q, P, ne, noise, rounds=3, head=10, k=200)
    S = S1.copy()
    for j, i in enumerate(dr['rows']):
        S[i] = ro.transplant_scores(S1[i][None], dr['loss'][j][None], dr['gain'][j][None])[0]
    return S


def load_sets(a, labels):
    """Every Reactome set (10-500) + readout modules (the null) + the TARGETS sets: names, member index arrays,
    boolean (sets x bank) matrix."""
    pos = {g: i for i, g in enumerate(labels)}
    N = len(labels)
    sets = {f'r:{k}': v for k, v in read_gmt(a.reactome_gmt, labels).items()}
    mods = {}
    for line in open(a.modules):
        if line.startswith('#') or '\t' not in line:
            continue
        m, p = line.rstrip('\n').split('\t')[:2]
        mods.setdefault(m, []).append(p)
    for m, pats in mods.items():
        g = sorted({x for p in pats for x in labels if (x.startswith(p[:-1]) if p.endswith('*') else x == p)})
        if len(g) >= 5:
            sets[f'm:{m}'] = g
    sets['export'] = export_parts(labels, a.reactome_gmt, a.modules)['mRNA export (grouped)']
    allr = read_gmt(a.reactome_gmt, labels, lo=1, hi=10_000)
    for _, k, *_ in TARGETS:
        if k not in sets and k.startswith('r:'):
            sets[k] = allr.get(k[2:], [])
    names = [k for k, v in sets.items() if len(v) >= 5]
    members = [np.array([pos[g] for g in sets[k] if g in pos]) for k in names]
    M = np.zeros((len(names), N), bool)
    for i, m in enumerate(members):
        M[i, m] = True
    return names, members, M


def main(a):
    units = ['cells_cos', 'cells_z', 'group', 'pools']
    un, bank = units[a.task // 2], ['trained', 'full'][a.task % 2]
    b = np.load(a.bank_npz, allow_pickle=True)
    rows_ = b['trained'].astype(bool) if bank == 'trained' else np.ones(len(b['labels']), bool)
    labels = [str(l) for l in b['labels'][rows_]]
    P = b['P'][rows_].astype(np.float64)
    Pn = unit(P)
    N = len(labels)
    pos = {g: i for i, g in enumerate(labels)}
    e = np.load(Path(a.predict_dir) / 'predict_groups.npz', allow_pickle=True)
    noise = e['noise'].astype(np.float64)
    names, members, M = load_sets(a, labels)
    nset = M.sum(1)
    # encodings
    if un == 'group':
        og = pd.read_csv(Path(a.predict_dir) / 'obs_groups.csv')
        G = {(r.donor, r.celltype): (e['Z'][i].astype(np.float64), int(r.n_cells)) for i, r in og.iterrows()}
    elif un == 'pools':
        op = pd.read_csv(Path(a.variant_dir) / 'obs_pbulk.csv')
        Zp = np.load(Path(a.variant_dir) / 'z_pbulk.npy')
        G = {k: (Zp[ix].astype(np.float64).mean(0), int(op.n_cells.iloc[ix].sum()))
             for k, ix in op.groupby(['donor', 'celltype']).indices.items()}
    else:
        oc = pd.read_csv(Path(a.variant_dir) / 'obs_cells.csv.gz')
        Zc = np.load(Path(a.variant_dir) / 'z_cells.npy', mmap_mode='r')
        G = group_means(Zc, oc)
    out = []
    for kind in ('ctrl', 'rest', 'rrep'):
        if un == 'cells_cos':
            keys, S = cell_profiles(Zc, oc, G, kind, Pn)
            prof = {'cosine': S}
        else:
            keys, Q, ne = contrasts(G, kind)
            prof = {'cosine': unit(Q) @ Pn.T, 'deflate': deflated(Q, P, ne, noise)}
        for rd, S in prof.items():
            for i, (d, t) in enumerate(keys):
                zs = set_z(pct_of(S[i]), members)
                o = np.argsort(-S[i], kind='stable')
                top = {'loss': o[:a.top], 'gain': o[::-1][:a.top]}
                hits = {tl: M[:, ix].sum(1) for tl, ix in top.items()}
                for j, nm in enumerate(names):
                    row = dict(unit=un, bank=bank, contrast=kind, readout=rd, donor=d, celltype=t, set=nm,
                               setz=zs[j][1])
                    for tl in ('loss', 'gain'):
                        row[f'hits_{tl}'] = int(hits[tl][j])
                        row[f'p_{tl}'] = float(hypergeom.sf(hits[tl][j] - 1, N, nset[j], a.top))
                    out.append(row)
    df = pd.DataFrame(out)
    df.to_parquet(Path(a.out_dir) / f'grid__{un}__{bank}.parquet', index=False)
    print(f'{un} {bank}: {len(df):,} rows', flush=True)


def score_frame(df, keycols):
    """Target scorecard + per-cell calls for every combination of `keycols` (hit = p_emp < .05 in the expected
    direction against every Normal-donor test of that combination)."""
    cond = lambda d: 'Normal' if d.startswith('Nor') else d[:2]
    df = df.assign(condition=df.donor.map(cond))
    res, calls = [], []
    for key, x in df.groupby(keycols):
        key = key if isinstance(key, tuple) else (key,)
        x = x.copy()
        nor = x.condition == 'Normal'
        nz = np.sort(np.abs(x.setz[nor].to_numpy()))
        npv = np.sort(np.r_[x.p_loss[nor].to_numpy(), x.p_gain[nor].to_numpy()])
        for sc in ('setz', 'ora'):
            if sc == 'setz':
                pz = (1 + len(nz) - np.searchsorted(nz, x.setz.abs(), 'left')) / (1 + len(nz))
                pl, pg = np.where(x.setz > 0, pz, 1), np.where(x.setz < 0, pz, 1)
            else:
                pl = (1 + np.searchsorted(npv, x.p_loss.to_numpy(), 'right')) / (1 + len(npv))
                pg = (1 + np.searchsorted(npv, x.p_gain.to_numpy(), 'right')) / (1 + len(npv))
            x['pl'], x['pg'] = pl, pg
            hit = wrong = rep = cells = 0
            ex = {}
            for lab, sk, conds, cts, dirn, ref in TARGETS:
                y = x[(x.set == sk) & x.condition.isin(conds) & x.celltype.isin(cts)]
                if not len(y):
                    continue
                ok = ((y.pl < .05) if dirn == 'loss' else (y.pg < .05) if dirn == 'gain'
                      else (y.pl < .05) | (y.pg < .05))
                bad = ((y.pg < .05) if dirn == 'loss' else (y.pl < .05) if dirn == 'gain' else ok & False)
                cells += len(y); hit += int(ok.sum()); wrong += int(bad.sum())
                rep += int((y.assign(ok=ok.values).groupby(['condition', 'celltype']).ok.sum() == 2).sum())
                for r in y.assign(ok=ok.values, bad=bad.values).itertuples():
                    calls.append(dict(zip(keycols, key), scoring=sc, target=lab, donor=r.donor, celltype=r.celltype,
                                      call='hit' if r.ok else 'opposite' if r.bad else ''))
                if sk == 'export':
                    h = y.assign(ok=ok.values)
                    h = h[h.celltype == 'Hepatocyte']
                    ex = {f'export_hep_{c}': int(h[h.condition == c].ok.sum()) for c in ('AC', 'AH')}
            res.append(dict(zip(keycols, key), scoring=sc, target_hits=hit, of=cells, opposite=wrong,
                            both_donors=rep, **ex,
                            normal_fp=round(float(((x.pl[nor] < .05) | (x.pg[nor] < .05)).mean()), 3)))
    R = pd.DataFrame(res)
    R['net'] = R.target_hits - R.opposite
    return R.sort_values(['net', 'both_donors'], ascending=False), pd.DataFrame(calls)


def summarize(a):
    df = pd.concat([pd.read_parquet(f) for f in sorted(Path(a.out_dir).glob('grid__*.parquet'))])
    R, C = score_frame(df, ['unit', 'bank', 'contrast', 'readout'])
    R.to_csv(Path(a.out_dir) / 'grid_scorecard.csv', index=False)
    C.to_csv(Path(a.out_dir) / 'grid_calls.csv.gz', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 200)
    print(R.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='run', choices=['run', 'summarize'])
    ap.add_argument('--task', type=int, default=0, help='unit x bank: 0..7')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--predict-dir', help='fig5_predict.py encode output (predict_groups.npz, obs_groups.csv)')
    ap.add_argument('--variant-dir', help='fig5_variants encode (z_cells, z_pbulk)')
    ap.add_argument('--bank-npz')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--modules')
    ap.add_argument('--top', type=int, default=200)
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (main if a.stage == 'run' else summarize)(a)
