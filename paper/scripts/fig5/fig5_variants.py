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

"""Fig. 5 variant grid: model x input mask x bank x readout, every variant scored the same way.

stage encode  (one task per model x mask)
    input mask   none | mito (MT-*) | mito_mt (MT-*, metallothioneins MT1*/MT2A/MT3/MT4 = the 'mtfree' query)
    cells        one predict() per cell type, that cell type's Normal nuclei = 'control' (input baseline)
    pbulk_x      each donor x cell type split into --n-rep random pools; each pool's MEAN raw counts is one
                 row; one predict() per cell type with the Normal pools as 'control' (>= 10 needed, else
                 predict() keeps a training baseline). The encoder then sees pseudobulked x.
stage score   (one task per encode x bank: trained | full = with the zero-shot f(missing) rows)
    readouts, all donor-vs-rest within a cell type (donor - equal-weight mean of the other donors):
    pbulk_z      cosine(unit(mean z_donor - rest), P)
    cell_ia      invariant_align mirror: z centred on the CELL TYPE's mean z (the context, as training
                 centres on each dataset's mean), per-cell cosine to P, averaged per donor, then donor - rest
    pbulk_x      as pbulk_z on the encoded pseudobulk pools
    pbulk_x_zall group vs ALL other groups (every donor x cell type): each prototype's score z-scored against
                 the other groups' scores for that prototype
    pbulk_x_vsOut each donor against every group of its cell type OUTSIDE its condition (AC1 vs Nor1, Nor2,
                 AH1, AH2); Normal donors against the other Normal donor = healthy donor-vs-donor null
    pbulk_x_zgroup the original ALD readout: each donor's profile vs the cell type's centre, every prototype
                 z-scored ACROSS the donors of that cell type (plot_perturbation_enrichment score='z_score')
    pbulk_x_zcond  the same with groups = conditions (donors pooled; AC/AH/Normal), every donor gets its
                 condition's vector
    pbulk_x_zrep   pbulk_x_zcond computed separately on the donor sets {AC1,AH1,Nor1} and {AC2,AH2,Nor2}
    pbulk_x_rrep   zrep without the per-prototype z: cos(condition - mean of its donor set's AC/AH/Normal). zrep's z
                 saturates at ~sqrt2 with 3 groups (top-200 spans ~0.005), so rank on this
    pbulk_x_vsN  pbulk_x against a FIXED healthy reference: donor - mean of the Normal donors of that cell type
                 (a Normal donor is compared with the other Normal donor, which is the null)
    gene sets (Reactome 10-500 in bank, ALD modules + ctrl panels, nuclear-pore set) -> signed set z;
    threshold = 99th pct of the Normal donors' own set z (they hit 1%).
stage merge   scorecard over all variants.

    PYTHONPATH=paper/scripts python paper/scripts/fig5/fig5_variants.py --stage encode --task 0 ...
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig5_centring_audit import NPC, ESTABLISHED, encode, unit, pct_of, set_z

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig5_variants')
log.setLevel(logging.INFO)

# 7SK/HEXIM-LARP7-MEPCE + NELF: the brakes on P-TEFb / promoter-proximal pausing (loss = pause release)
PAUSE_BRAKE = ['HEXIM1', 'HEXIM2', 'LARP7', 'MEPCE', 'NELFA', 'NELFB', 'NELFCD', 'NELFE']
MASKS = {'none': [], 'mito': ['MT-*'], 'mito_mt': ['MT-*', 'MT1*', 'MT2A', 'MT3', 'MT4']}
EXPORT = ['Transport of Mature Transcript to Cytoplasm', 'rna_export_trex', 'nuclear_pore',
          'Transport of Mature mRNA Derived From an Intron-Containing Transcript',
          'Nuclear Pore Complex (NPC) Disassembly']


def grid(a):
    models = a.models.split(',')
    return [(m, k) for m in models for k in MASKS]


def mask_query(q, pats):
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(q.n_vars, bool)
    for p in pats:
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    return q[:, ~drop].copy(), sorted(v[drop])


def stage_encode(a):
    import anndata as ad
    from fig5_centring_audit import load_model
    mtag, mask = grid(a)[a.task]
    out = Path(a.out_dir) / f'{mtag}__{mask}'
    out.mkdir(parents=True, exist_ok=True)
    model, facts = load_model(dict(zip(a.models.split(','), a.model_dirs.split(',')))[mtag])
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    q, dropped = mask_query(q, MASKS[mask])
    genes = np.asarray(q.var_names).astype(str)
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    cond = q.obs[a.condition_col].astype(str).to_numpy()
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    keep = np.isin(ct, a.celltypes.split(','))
    normal = cond == a.control_condition
    # cells: one call per cell type, Normal nuclei of that type are the input baseline
    Z = np.full((q.n_obs, int(model.module.align_ds_mean.shape[1])), np.nan, np.float32)
    for t in a.celltypes.split(','):
        m = keep & (ct == t)
        z, _ = encode(model, q.X[m], genes, normal[m], a)
        Z[m] = z
    np.save(out / 'z_cells.npy', Z.astype(np.float16))
    pd.DataFrame({'donor': donor, 'condition': cond, 'celltype': ct, 'done': ~np.isnan(Z[:, 0])}).to_csv(
        out / 'obs_cells.csv.gz', index=False)
    # pbulk_x: random pools per donor x cell type, mean raw counts per pool
    rng = np.random.default_rng(0)
    rows, Zp = [], []
    for t in a.celltypes.split(','):
        X, meta, ref = [], [], []
        for d in sorted(set(donor[keep & (ct == t)])):
            idx = rng.permutation(np.flatnonzero(keep & (ct == t) & (donor == d)))
            if len(idx) < a.min_cells:
                continue
            for r, chunk in enumerate(np.array_split(idx, a.n_rep)):
                X.append(np.asarray(q.X[chunk].mean(0)).ravel())
                meta.append(dict(donor=d, condition=cond[idx[0]], celltype=t, rep=r, n_cells=len(chunk)))
                ref.append(cond[idx[0]] == a.control_condition)
        z, _ = encode(model, sp.csr_matrix(np.stack(X).astype(np.float32)), genes, np.array(ref), a)
        Zp.append(z)
        rows += meta
    np.save(out / 'z_pbulk.npy', np.concatenate(Zp).astype(np.float32))
    pd.DataFrame(rows).to_csv(out / 'obs_pbulk.csv', index=False)
    mg = set(np.asarray(model.adata.var_names).astype(str))
    json.dump(dict(model=mtag, mask=mask, dropped=dropped, n_dropped=len(dropped), model_facts=facts,
                   input_genes=[g for g in genes if g in mg]),
              open(out / 'encode.json', 'w'), indent=1)
    log.info('%s %s: dropped %d genes, %d cells, %d pools', mtag, mask, len(dropped), int(keep.sum()), len(rows))


def profiles(Zc, obs, Zp, obsp, Pn, a):
    """{readout: {(donor, celltype): donor-vs-rest prototype score vector}}."""
    out = {'pbulk_z': {}, 'cell_ia': {}, 'pbulk_x': {}, 'pbulk_x_vsN': {}, 'pbulk_x_zall': {}, 'pbulk_x_vsOut': {},
           'pbulk_x_zgroup': {}, 'pbulk_x_zcond': {}, 'pbulk_x_zrep': {}, 'pbulk_x_rrep': {}}
    # group vs ALL other groups (every donor x cell type): cosine of each group's deviation from the grand
    # mean, then each prototype z-scored against the other groups' scores for that prototype
    keys = [(d, t) for t in a.celltypes.split(',') for d in sorted(obsp.donor[(obsp.celltype == t).to_numpy()].unique())]
    G = np.stack([Zp[((obsp.celltype == t) & (obsp.donor == d)).to_numpy()].mean(0) for d, t in keys])
    Sg = unit(G - G.mean(0)) @ Pn.T
    for i, k in enumerate(keys):
        o = np.delete(Sg, i, axis=0)
        out['pbulk_x_zall'][k] = (Sg[i] - o.mean(0)) / (o.std(0) + 1e-9)
    for t in a.celltypes.split(','):
        mt = obs.done.to_numpy() & (obs.celltype == t).to_numpy()
        ds = [d for d in sorted(obs.donor[mt].unique()) if (mt & (obs.donor == d).to_numpy()).sum() >= a.min_cells]
        mz, mc = {}, {}
        centre = Zc[mt].mean(0)                                          # the context's mean z
        for d in ds:
            m = np.flatnonzero(mt & (obs.donor == d).to_numpy())
            mz[d] = Zc[m].mean(0)
            acc = np.zeros(Pn.shape[0])
            for i in range(0, len(m), 4096):
                acc += (unit(Zc[m[i:i + 4096]] - centre) @ Pn.T).sum(0)
            mc[d] = acc / len(m)
        for d in ds:
            rest = [e for e in ds if e != d]
            out['pbulk_z'][(d, t)] = unit(mz[d] - np.mean([mz[e] for e in rest], 0)) @ Pn.T
            out['cell_ia'][(d, t)] = mc[d] - np.mean([mc[e] for e in rest], 0)
        pt = (obsp.celltype == t).to_numpy()
        pd_ = sorted(obsp.donor[pt].unique())
        pm = {d: Zp[pt & (obsp.donor == d).to_numpy()].mean(0) for d in pd_}
        nor = [e for e in pd_ if obsp.condition[obsp.donor == e].iloc[0] == a.control_condition]
        for d in pd_:
            out['pbulk_x'][(d, t)] = unit(pm[d] - np.mean([pm[e] for e in pd_ if e != d], 0)) @ Pn.T
            cnd_of = {e: obsp.condition[obsp.donor == e].iloc[0] for e in pd_}
            # vsOut: against every group of this cell type OUTSIDE the donor's condition; a Normal donor is
            # compared with the other Normal donor instead, so the Normal rows are healthy donor-vs-donor (the null)
            outside = ([e for e in nor if e != d] if cnd_of[d] == a.control_condition
                       else [e for e in pd_ if cnd_of[e] != cnd_of[d]])
            if outside:
                out['pbulk_x_vsOut'][(d, t)] = unit(pm[d] - np.mean([pm[e] for e in outside], 0)) @ Pn.T
            ref = [e for e in nor if e != d]                          # fixed healthy reference (other Normal
            if ref:                                                   # donor for a Normal donor = the null)
                out['pbulk_x_vsN'][(d, t)] = unit(pm[d] - np.mean([pm[e] for e in ref], 0)) @ Pn.T
        # the original ALD readout (expert.utils.plotting.plot_perturbation_enrichment, score='z_score'): every
        # group's profile, then each PROTOTYPE z-scored ACROSS the groups of this cell type
        def zacross(vecs):
            S = np.stack([unit(v - np.mean(vecs, 0)) @ Pn.T for v in vecs])
            return (S - S.mean(0)) / np.where(S.std(0) > 1e-9, S.std(0), 1.0)
        Zg = zacross([pm[d] for d in pd_])                            # groups = donors
        for i, d in enumerate(pd_):
            out['pbulk_x_zgroup'][(d, t)] = Zg[i]
        conds = sorted({obsp.condition[obsp.donor == e].iloc[0] for e in pd_})
        pc = {c: np.mean([pm[e] for e in pd_ if obsp.condition[obsp.donor == e].iloc[0] == c], 0) for c in conds}
        Zcond = zacross([pc[c] for c in conds])                       # groups = conditions (donors pooled)
        for d in pd_:
            out['pbulk_x_zcond'][(d, t)] = Zcond[conds.index(obsp.condition[obsp.donor == d].iloc[0])]
        # zrep: the same condition z-scoring done twice on independent donor sets {AC1, AH1, Nor1} and
        # {AC2, AH2, Nor2}; donor d carries its own set's vector, so 'both donors' = replicated across sets.
        # A set missing its Normal donor (T cells: Nor2 too small) borrows the other Normal donor.
        for k in ('1', '2'):
            trip = {c: [e for e in pd_ if e.endswith(k) and obsp.condition[obsp.donor == e].iloc[0] == c]
                    for c in conds}
            if not trip.get(a.control_condition):
                trip[a.control_condition] = nor[:1]
            if any(not v for v in trip.values()):
                continue
            Zr = zacross([pm[trip[c][0]] for c in conds])
            for c in conds:
                d = trip[c][0]
                if d.endswith(k):
                    out['pbulk_x_zrep'][(d, t)] = Zr[conds.index(c)]
            vecs = [pm[trip[c][0]] for c in conds]
            for c in conds:                     # rrep: the same design, UN-normalised (zrep saturates at ~sqrt2)
                d = trip[c][0]
                if d.endswith(k):
                    out['pbulk_x_rrep'][(d, t)] = unit(pm[d] - np.mean(vecs, 0)) @ Pn.T
    return out


def stage_score(a):
    import expert.utils.readout as ro
    encs = grid(a)
    banks = ['trained', 'full']
    (mtag, mask), bank = encs[a.task // 2], banks[a.task % 2]
    src = Path(a.out_dir) / f'{mtag}__{mask}'
    b = np.load(dict(zip(a.models.split(','), a.banks.split(',')))[mtag], allow_pickle=True)
    rows_ = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
    labels = [str(l) for l in b['labels'][rows_]]
    Pn = unit(b['P'][rows_].astype(np.float32))
    pen = None
    if a.hub_pen:                                     # stored predict() offset, (2, n) aligned to bank labels
        hz = np.load(a.hub_pen, allow_pickle=True)
        assert [str(x) for x in hz['labels']] == [str(l) for l in b['labels']], 'hub_pen not aligned to bank'
        pen = hz['pen'][:, rows_].astype(np.float64)
    lpos = {l: i for i, l in enumerate(labels)}
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    Z = np.load(src / 'z_cells.npy').astype(np.float32)
    obsp = pd.read_csv(src / 'obs_pbulk.csv')
    Zp = np.load(src / 'z_pbulk.npy')
    P = profiles(Z, obs, Zp, obsp, Pn, a)
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in lpos]
    sets[('custom', 'pause brake')] = [g for g in PAUSE_BRAKE if g in lpos]
    keys = list(sets)
    members = [np.array([lpos[x] for x in sets[k] if x in lpos]) for k in keys]
    inputs = set(json.load(open(src / 'encode.json'))['input_genes'])
    deg = pd.read_csv(a.deg)
    down = set(zip(deg.group[(deg.pval_adj < 0.05) & (deg.logfc <= -0.25)],
                   deg.gene[(deg.pval_adj < 0.05) & (deg.logfc <= -0.25)]))
    rows, genes = [], []
    for rd, R in P.items():
        for (d, t), v in R.items():
            if pen is None:
                for (fam, sname), (mu, z, p) in zip(keys, set_z(pct_of(v), members)):
                    rows.append((rd, d, t, fam, sname, mu, z))      # loss tail; gain = -z
            else:   # each tail ranked on its own corrected score (predict(): loss cos - pen0, gain -cos - pen1)
                zl, zg = set_z(pct_of(v - pen[0]), members), set_z(pct_of(-v - pen[1]), members)
                for (fam, sname), (mu, z1, _), (_, z2, _) in zip(keys, zl, zg):
                    rows.append((rd, d, t, fam, sname, mu, (z1 - z2) / 2))   # loss-positive; = z when pen flat
            top = np.argsort(-v)[:50]
            cnd = obs.condition[obs.donor == d].iloc[0] if (obs.donor == d).any() else obsp.condition[obsp.donor == d].iloc[0]
            genes.append(dict(readout=rd, donor=d, celltype=t, condition=cnd,
                              top50=','.join(labels[j] for j in top),
                              self_signal=float(np.mean([labels[j] in inputs and (f'{cnd}|{t}', labels[j]) in down
                                                         for j in top]))))
    tag = f'{mtag}__{mask}__{bank}' + ('__hub' if pen is not None else '')
    t_ = pd.DataFrame(rows, columns=['readout', 'donor', 'celltype', 'family', 'set', 'mean_pct', 'z'])
    t_.insert(0, 'variant', tag)
    t_.to_csv(Path(a.out_dir) / f'tests__{tag}.csv.gz', index=False)
    pd.DataFrame(genes).assign(variant=tag).to_csv(Path(a.out_dir) / f'genes__{tag}.csv', index=False)
    log.info('%s: %d tests', tag, len(t_))


def stage_merge(a):
    out = Path(a.out_dir)
    card = []
    for f in sorted(out.glob('tests__*.csv.gz')):
        t = pd.read_csv(f)
        tag = t.variant.iloc[0]
        g = pd.read_csv(out / f"genes__{tag}.csv")
        model, mask, bank = tag.split('__')
        for rd, x in t.groupby('readout'):
            nor = x.donor.str.startswith('Nor')
            zz = np.r_[x.z[nor], -x.z[nor]]                            # both tails of the Normal donors
            thr = np.quantile(zz, 1 - a.alpha)
            x = x.assign(loss=x.z > thr, gain=-x.z > thr, hit=(x.z.abs() > thr))
            dis = ~x.donor.str.startswith('Nor')
            ald = (x.family == 'module') & ~x.set.str.startswith('ctrl_')
            ctl = (x.family == 'module') & x.set.str.startswith('ctrl_')
            est = (x.family == 'reactome') & x.set.isin(ESTABLISHED)
            rea = x.family == 'reactome'
            row = dict(model=model, mask=mask, bank=bank, readout=rd, threshold=thr,
                       hit_rate_disease=x.hit[dis & rea].mean(),
                       established_hits=int(x.hit[dis & est].sum()), established_tests=int((dis & est).sum()),
                       ald_module_hits=int(x.hit[dis & ald].sum()), ctrl_panel_hits=int(x.hit[dis & ctl].sum()))

            def cons(d1, d2):
                k = ['celltype', 'set']
                h1 = x[(x.donor == d1) & x.hit & rea][k + ['z']]
                h2 = x[(x.donor == d2) & x.hit & rea][k + ['z']]
                m = h1.merge(h2, on=k)
                return int((np.sign(m.z_x) == np.sign(m.z_y)).sum())
            row['replicated_AC'] = cons('AC1', 'AC2')
            row['replicated_AH'] = cons('AH1', 'AH2')
            row['replicated_null_cross'] = float(np.mean([cons(p, q) for p, q in
                                                          (('AC1', 'AH1'), ('AC1', 'AH2'), ('AC2', 'AH1'), ('AC2', 'AH2'))]))
            npc = x[x.family == 'npc'].set_index(['donor', 'celltype']).z
            for dn in ('AC1', 'AC2', 'AH1', 'AH2'):
                v = npc[npc.index.get_level_values(0) == dn]
                row[f'npc_z_{dn}'] = ' '.join(f'{z:.1f}' for z in v.values)
                row[f'npc_n_over_{dn}'] = int((v > thr).sum())
            ex = x[x.set.isin(EXPORT)].copy()
            allr = x[x.family != 'npc'].copy()
            allr['rank'] = allr.groupby(['donor', 'celltype']).z.rank(ascending=False)
            er = allr[allr.set.isin(EXPORT)].groupby(['donor', 'celltype'])['rank'].min()
            row['AC2_best_export_rank'] = ' '.join(f'{int(v)}' for v in er[er.index.get_level_values(0) == 'AC2'].values)
            row['self_signal_top50'] = g[(g.readout == rd) & ~g.donor.str.startswith('Nor')].self_signal.mean()
            card.append(row)
    card = pd.DataFrame(card).sort_values(['readout', 'bank', 'model', 'mask'])
    card.to_csv(out / 'variant_scorecard.csv', index=False)
    pd.set_option('display.width', 300, 'display.max_columns', 40, 'display.max_colwidth', 40)
    log.info('\n%s', card.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'score', 'merge'])
    ap.add_argument('--task', type=int)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--models', default='v0.1.0,v0.1.1,v0.1.2')
    ap.add_argument('--model-dirs', help='comma list aligned with --models')
    ap.add_argument('--banks', help='comma list of fullbank/bank.npz aligned with --models')
    ap.add_argument('--query', help='alcohol_readout.h5ad (UNMASKED; masks are applied here)')
    ap.add_argument('--modules')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--deg', help='readout_deg/deg.csv.gz')
    ap.add_argument('--condition-col', default='Condition')
    ap.add_argument('--control-condition', default='Normal')
    ap.add_argument('--celltype-col', default='CellTypeSR')
    ap.add_argument('--donor-col', default='sample')
    ap.add_argument('--celltypes', default='HSC,Hepatocyte,Kupffer Cell,LSEC,T cell')
    ap.add_argument('--min-cells', type=int, default=50)
    ap.add_argument('--n-rep', type=int, default=10, help='pseudobulk pools per donor x cell type')
    ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--hub-pen', default='', help='score: stored hubness offset npz (fig5_hubpen.py), adds __hub')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--n-cores', type=int, default=8)
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    {'encode': stage_encode, 'score': stage_score, 'merge': stage_merge}[a.stage](a)
