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

"""Fig. 6 data: the production model read on paediatric mitochondrial-disease PBMCs (Weinberg cohort).

There is NO healthy baseline: 'control' patients are undiagnosed ('unknown'), tbck = TBCK disease (non-mitochondrial,
genetically confirmed), leigh = Leigh syndrome (NDUFS8 / NDUFA10 / BCS1L). Only disease CONTRASTS are readable.
Two groupings are scored side by side: 'orig' (leigh / tbck / unknown) and 'regrouped' (PT67, re-sequenced after
ExPert flagged it, moves to the mitochondrial group: mito / tbck / unknown).

stage encode  (model load) one predict() per cell type on random pseudobulk pools (--n-rep per donor x cell type,
              mean raw counts); ALL pools of the cell type are the input baseline (cell-type mean; no healthy
              reference exists). Per-cell latents for the UMAP the same way. MT-* + metallothioneins dropped
              (Fig. 5 'mito_mt'). Writes bank.npz = full prototype table (trained + zero-shot fills, idx_to_label
              order) + the model's stored hubness offset (atom_hub_pen, rows loss/gain).
stage deg     own-transcript DE (scanpy wilcoxon, log1p CP10k) per cell type (--task) for every contrast scored.
stage score   profiles per cell type, for bank in {trained, full} x hub in {on, off}:
                donor      cos(unit(donor - mean of the other donors), P)
                donor_z    Fig. 5 original z-score: cos(unit(donor - mean of donors), P), z-scored per prototype ACROSS donors
                cond       cos(unit(condition - mean of conditions), P); condition = equal-weight mean of its donors
                pair       cos(unit(condition a - condition b), P)
              ranking score = cos - pen[tail] (hub on); set z = mean percentile of the set vs random sets (Fig. 5),
              loss and gain tails separately. Condition/pair significance = exact donor-label permutation null
              (every partition of the cell type's donors into the grouping's sizes). Union top-k tables,
              lesion-gene ranks, leave-one-donor-out condition assignment.
stage genes   every prototype's score for the pair contrasts + lesion donors (gene panels), with own DE.
stage umap    UMAP of the per-cell latents.

    python paper/scripts/fig6/fig6_data.py --stage encode --query <h5ad> --model-dir <model> --bank-npz <bank> --out-dir <d>
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_data')
log.setLevel(logging.INFO)

MASK = ['MT-*', 'MT1*', 'MT2A', 'MT3', 'MT4']
LESION = {'PT54': 'NDUFS8', 'PT46': 'NDUFA10', 'PT43': 'BCS1L'}
LESION_SETS = {'NDUFS8': 'complex_i', 'NDUFA10': 'complex_i', 'BCS1L': 'complex_iii'}


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def grouping(donor_cond, reassign, etc=False):
    """{'orig': donor -> leigh/tbck/unknown, 'regrouped': donor -> mito/tbck/unknown}; with etc=True also
    'etc': Leigh (ETC lesions) -> etc, the reassigned non-ETC PMD donor(s) -> pc, everyone else -> nonpmd."""
    orig = {d: {'control': 'unknown'}.get(c, c) for d, c in donor_cond.items()}
    reg = {d: ('mito' if c == 'leigh' else c) for d, c in orig.items()}
    for kv in reassign:
        d, c = kv.split('=')
        reg[d] = c
    out = {'orig': orig, 'regrouped': reg}
    if etc:
        moved = {kv.split('=')[0] for kv in reassign}
        out['etc'] = {d: 'etc' if c == 'leigh' else ('pc' if d in moved else 'nonpmd') for d, c in orig.items()}
    return out


def celltypes(obs, a):
    """Cell-type label per cell; with --split-col (in-vitro stim_condition) the unit is '<celltype> <split>'."""
    ct = obs[a.celltype_col].astype(str).to_numpy()
    if a.split_col:
        ct = np.char.add(np.char.add(ct.astype(str), ' '), obs[a.split_col].astype(str).to_numpy().astype(str))
    return ct


def mask_query(q, pats):
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(q.n_vars, bool)
    for p in pats:
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    return q[:, ~drop].copy(), sorted(v[drop])


# ------------------------------------------------------------------------------------------- encode
def stage_encode(a):
    import anndata as ad
    import torch
    from fig5_centring_audit import encode, load_model
    import expert.utils.readout as ro
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, facts = load_model(a.model_dir)
    # prototype bank (harvested from this model's predict(); verify the trained block) + stored hub offset
    b = np.load(a.bank_npz, allow_pickle=True)
    _, lab_tr, P_tr = ro.perturbation_prototypes(model)
    tr = b['trained'].astype(bool)
    rowcos = (unit(b['P'][:len(lab_tr)].astype(np.float64)) * unit(P_tr)).sum(1)
    assert tr[:len(lab_tr)].all() and not tr[len(lab_tr):].any()
    assert list(map(str, b['labels'][:len(lab_tr)])) == list(map(str, lab_tr)), 'bank labels != model labels'
    assert rowcos.min() > 0.9999, f'bank trained rows differ from the model atoms (min row cos {rowcos.min()})'
    pen = ro.hub_penalty(model, len(b['labels']))
    if pen is None or pen.shape != (2, len(b['labels'])):            # no stored correction: hub arms = plain cosine
        log.warning('model carries no full-width hub correction; storing a zero offset')
        pen = np.zeros((2, len(b['labels'])), np.float32)
    np.savez(out / 'bank.npz', P=b['P'].astype(np.float32), labels=b['labels'], trained=tr, hub_pen=pen.astype(np.float32))
    log.info('bank %s, trained %d, hub pen loss median trained %.3f / filled %.3f', b['P'].shape, tr.sum(),
             np.median(pen[0, tr]), np.median(pen[0, ~tr]))

    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    q, dropped = mask_query(q, MASK)
    genes = np.asarray(q.var_names).astype(str)
    mg = set(np.asarray(model.adata.var_names).astype(str))
    ct = celltypes(q.obs, a)
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    cond = q.obs[a.condition_col].astype(str).to_numpy()
    cts = sorted(set(ct))
    n_sh = int(model.module.align_ds_mean.shape[1])
    Z = np.full((q.n_obs, n_sh), np.nan, np.float32)
    rng = np.random.default_rng(0)
    rows, Zp = [], []
    for t in cts:
        m = ct == t
        Z[m], _ = encode(model, q.X[m], genes, np.ones(m.sum(), bool), a)   # baseline = this cell type's mean
        X, meta = [], []
        for d in sorted(set(donor[m])):
            idx = rng.permutation(np.flatnonzero(m & (donor == d)))
            if len(idx) < a.min_cells:
                continue
            for r, chunk in enumerate(np.array_split(idx, a.n_rep)):
                X.append(np.asarray(q.X[chunk].mean(0)).ravel())
                meta.append(dict(donor=d, condition=cond[idx[0]], celltype=t, rep=r, n_cells=len(chunk)))
        if len({x['donor'] for x in meta}) < 4:
            log.info('%s: <4 donors with >= %d cells, skipped for pools', t, a.min_cells)
            continue
        z, _ = encode(model, sp.csr_matrix(np.stack(X).astype(np.float32)), genes, np.ones(len(X), bool), a)
        Zp.append(z)
        rows += meta
        log.info('%s: %d cells, %d pools over %d donors', t, int(m.sum()), len(meta), len({x['donor'] for x in meta}))
    np.save(out / 'z_cells.npy', Z.astype(np.float16))
    pd.DataFrame({'donor': donor, 'condition': cond, 'celltype': ct}).to_csv(out / 'obs_cells.csv.gz', index=False)
    np.save(out / 'z_pbulk.npy', np.concatenate(Zp).astype(np.float32))
    pd.DataFrame(rows).to_csv(out / 'obs_pbulk.csv', index=False)
    json.dump(dict(model_dir=a.model_dir, query=a.query, model_facts=facts, dropped=dropped,
                   n_input_model=len(mg), n_input_present=int(len(mg & set(genes))),
                   hub_info=dict(n=int(pen.shape[1])), torch=torch.__version__),
              open(out / 'encode.json', 'w'), indent=1)
    log.info('input genes present %d / %d; dropped %d', len(mg & set(genes)), len(mg), len(dropped))


# ------------------------------------------------------------------------------------------- deg
def contrasts_for(donors, groups):
    """[(name, members_a, members_b)] for one cell type; b=None means 'every other donor'."""
    out = []
    for gname, g in groups.items():
        conds = sorted({g[d] for d in donors})
        for c in conds:
            out.append((f'{gname}:{c}_vs_rest', [d for d in donors if g[d] == c], [d for d in donors if g[d] != c]))
        for c1, c2 in itertools.combinations(conds, 2):
            out.append((f'{gname}:{c1}_vs_{c2}', [d for d in donors if g[d] == c1], [d for d in donors if g[d] == c2]))
    for d in donors:
        out.append((f'donor:{d}_vs_rest', [d], [e for e in donors if e != d]))
    return out


def stage_deg(a):
    import anndata as ad
    import scanpy as sc
    q = ad.read_h5ad(a.query)
    ct = celltypes(q.obs, a)
    t = sorted(set(ct))[a.task]
    q = q[ct == t].copy()
    sc.pp.normalize_total(q, target_sum=1e4)
    sc.pp.log1p(q)
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    dc = dict(zip(donor, q.obs[a.condition_col].astype(str)))
    vc = pd.Series(donor).value_counts()
    donors = sorted(vc.index[vc >= a.min_cells])
    groups = grouping(dc, a.reassign, etc=a.etc_grouping)
    res = []
    for name, A, B in contrasts_for(donors, groups):
        if not A or not B:
            continue
        lab = np.where(np.isin(donor, A), 'A', np.where(np.isin(donor, B), 'B', 'x'))
        x = q[lab != 'x'].copy()
        x.obs['lab'] = pd.Categorical(lab[lab != 'x'])
        sc.tl.rank_genes_groups(x, 'lab', groups=['A'], reference='B', method='wilcoxon')
        df = sc.get.rank_genes_groups_df(x, 'A')
        res.append(pd.DataFrame({'celltype': t, 'contrast': name, 'gene': df.names.astype(str), 'score': df.scores,
                                 'logfc': df.logfoldchanges, 'pval_adj': df.pvals_adj}))
        log.info('%s %s: %d vs %d cells', t, name, int((lab == 'A').sum()), int((lab == 'B').sum()))
    out = Path(a.out_dir) / 'deg'
    out.mkdir(parents=True, exist_ok=True)
    if not res:
        log.info('%s: no scorable contrast (<2 donors with >= %d cells)', t, a.min_cells)
        return
    pd.concat(res).to_csv(out / f'deg__{t.replace(" ", "_").replace("+", "p")}.csv.gz', index=False)


# ------------------------------------------------------------------------------------------- score
def pct_rows(S):
    """Row-wise tie-averaged percentile (1 = top)."""
    from scipy.stats import rankdata
    return (rankdata(S, axis=1) - 1) / (S.shape[1] - 1)


def set_z_rows(pct, M, sizes):
    """Set z for every row of pct (rows x N) and every set (sparse N x S membership): analytic finite-population
    normal null with the profile's own variance (Fig. 5 set_z, vectorised)."""
    N = pct.shape[1]
    var = pct.var(1, keepdims=True)
    mu = np.asarray(M.T @ pct.T).T / sizes                       # rows x S
    sd = np.sqrt(var / sizes * (N - sizes) / (N - 1))
    return (mu - pct.mean(1, keepdims=True)) / sd


def partitions(donors, sizes):
    """Every assignment of donors to labelled groups of the given sizes (dict label -> n)."""
    labs = list(sizes)

    def rec(pool, i):
        if i == len(labs) - 1:
            yield {labs[i]: tuple(pool)}
            return
        for comb in itertools.combinations(pool, sizes[labs[i]]):
            rest = [d for d in pool if d not in comb]
            for tail in rec(rest, i + 1):
                yield {labs[i]: comb, **tail}
    yield from rec(list(donors), 0)


def score_vec(R, Pn, pen, hub):
    """R (rows x d) contrast vectors -> raw cosine S, loss ranking score, gain ranking score."""
    S = unit(R) @ Pn.T
    if not hub:
        return S, S, -S
    return S, S - pen[0], -S - pen[1]


def stage_score(a):
    import expert.utils.readout as ro
    src = Path(a.out_dir)
    out = src / 'score'
    out.mkdir(exist_ok=True)
    b = np.load(src / 'bank.npz', allow_pickle=True)
    obsp = pd.read_csv(src / 'obs_pbulk.csv')
    Zp = np.load(src / 'z_pbulk.npy').astype(np.float64)
    dc = dict(zip(obsp.donor, obsp.condition))
    groups = grouping(dc, a.reassign)
    degs = pd.concat([pd.read_csv(f) for f in sorted((src / 'deg').glob('deg__*.csv.gz'))]) \
        if (src / 'deg').exists() else None
    deg_ix = degs.set_index(['celltype', 'contrast', 'gene']) if degs is not None else None
    cts = sorted(obsp.celltype.unique())
    tests, tops, lesion, loo, permz, rho = [], [], [], [], [], []
    for bank in ('trained', 'full'):
        rows_ = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
        labels = np.asarray([str(l) for l in b['labels'][rows_]])
        zs = ~b['trained'].astype(bool)[rows_]
        Pn = unit(b['P'][rows_].astype(np.float64))
        pen_all = b['hub_pen'][:, rows_].astype(np.float64)
        lpos = {l: i for i, l in enumerate(labels)}
        sets = {('module', k): v for k, v in ro.load_modules(a.modules, list(labels)).items()}
        sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=list(labels)).items()})
        keys = list(sets)
        ii = np.concatenate([[lpos[g] for g in sets[k]] for k in keys])
        jj = np.concatenate([[j] * len(sets[k]) for j, k in enumerate(keys)])
        M = sp.csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(labels), len(keys)))
        sizes = np.asarray(M.sum(0)).ravel()
        log.info('[%s] %d prototypes, %d sets', bank, len(labels), len(keys))
        for t in cts:
            pt = (obsp.celltype == t).to_numpy()
            ds = sorted(obsp.donor[pt].unique())
            pm = np.stack([Zp[pt & (obsp.donor == d).to_numpy()].mean(0) for d in ds])
            # ---- per-donor profiles
            Rd = np.stack([pm[i] - np.delete(pm, i, 0).mean(0) for i in range(len(ds))])
            Rz = pm - pm.mean(0)
            Sz = unit(Rz) @ Pn.T
            Zacross = (Sz - Sz.mean(0)) / np.where(Sz.std(0) > 1e-9, Sz.std(0), 1.0)
            prof = {}                                                   # (kind, name) -> (raw, loss, gain)
            for hub in (True, False):
                S, L, G = score_vec(Rd, Pn, pen_all, hub)
                for i, d in enumerate(ds):
                    prof[('donor', d, hub)] = (S[i], L[i], G[i])
                    prof[('donor_z', d, hub)] = (Zacross[i], Zacross[i], -Zacross[i])  # hub offset cancels in z
            # ---- condition / pair profiles + permutation null, per grouping
            for gname, g in groups.items():
                conds = sorted({g[d] for d in ds})
                size = {c: sum(g[d] == c for d in ds) for c in conds}
                pairs = list(itertools.combinations(conds, 2))

                def contrast_rows(assign):
                    C = {c: pm[[ds.index(d) for d in assign[c]]].mean(0) for c in conds}
                    cen = np.mean([C[c] for c in conds], 0)
                    R = [C[c] - cen for c in conds] + [C[c1] - C[c2] for c1, c2 in pairs]
                    names = [f'{c}' for c in conds] + [f'{c1}_vs_{c2}' for c1, c2 in pairs]
                    return np.stack(R), names
                obs_assign = {c: tuple(d for d in ds if g[d] == c) for c in conds}
                Robs, names = contrast_rows(obs_assign)
                perms = list(partitions(ds, size))
                for hub in (True, False):
                    S, L, G = score_vec(Robs, Pn, pen_all, hub)
                    for i, n in enumerate(names):
                        prof[(f'{gname}', n, hub)] = (S[i], L[i], G[i])
                    zl_obs = set_z_rows(pct_rows(L), M, sizes)
                    zg_obs = set_z_rows(pct_rows(G), M, sizes)
                    ge_l = np.zeros_like(zl_obs)
                    ge_g = np.zeros_like(zg_obs)
                    for s in range(0, len(perms), a.perm_batch):
                        Rp = np.concatenate([contrast_rows(p)[0] for p in perms[s:s + a.perm_batch]])
                        _, Lp, Gp = score_vec(Rp, Pn, pen_all, hub)
                        zl = set_z_rows(pct_rows(Lp), M, sizes).reshape(-1, len(names), len(keys))
                        zg = set_z_rows(pct_rows(Gp), M, sizes).reshape(-1, len(names), len(keys))
                        ge_l += (zl >= zl_obs[None] - 1e-9).sum(0)
                        ge_g += (zg >= zg_obs[None] - 1e-9).sum(0)
                    for i, n in enumerate(names):
                        for j, (fam, sname) in enumerate(keys):
                            tests.append((bank, hub, t, gname, n, fam, sname, int(sizes[j]), zl_obs[i, j], zg_obs[i, j],
                                          ge_l[i, j] / len(perms), ge_g[i, j] / len(perms), len(perms)))
                log.info('[%s] %s %s: %d donors, %d permutations', bank, t, gname, len(ds), len(perms))
                # leave-one-donor-out: which condition centroid (built without the donor) does each donor sit nearest?
                for i, d in enumerate(ds):
                    rest = [e for e in ds if e != d]
                    cen_all = pm[[ds.index(e) for e in rest]].mean(0)
                    v = unit(pm[i] - cen_all)
                    sim = {}
                    for c in conds:
                        mem = [e for e in rest if g[e] == c]
                        if mem:
                            sim[c] = float(v @ unit(pm[[ds.index(e) for e in mem]].mean(0) - cen_all))
                    if bank == 'trained':                              # latent-space, bank-independent
                        loo.append(dict(celltype=t, grouping=gname, donor=d, label=g[d], **{f'cos_{c}': s for c, s in sim.items()},
                                        nearest=max(sim, key=sim.get)))
            # ---- per-donor set tests (analytic z; calibration = the other donors of the same cell type)
            for hub in (True, False):
                for kind in ('donor', 'donor_z'):
                    Ls = np.stack([prof[(kind, d, hub)][1] for d in ds])
                    Gs = np.stack([prof[(kind, d, hub)][2] for d in ds])
                    zl = set_z_rows(pct_rows(Ls), M, sizes)
                    zg = set_z_rows(pct_rows(Gs), M, sizes)
                    for i, d in enumerate(ds):
                        for j, (fam, sname) in enumerate(keys):
                            tests.append((bank, hub, t, kind, d, fam, sname, int(sizes[j]), zl[i, j], zg[i, j],
                                          np.nan, np.nan, 0))
            # ---- top tables + lesion ranks + DE agreement
            for (kind, name, hub), (S, L, G) in prof.items():
                dcon = (f'donor:{name}_vs_rest' if kind in ('donor', 'donor_z')
                        else f'{kind}:{name}_vs_rest' if '_vs_' not in name else f'{kind}:{name}')
                for tail, v in (('loss', L), ('gain', G)):
                    order = np.argsort(-v)
                    rank = np.empty(len(v), int)
                    rank[order] = np.arange(1, len(v) + 1)
                    for r, j in enumerate(order[:a.top_k]):
                        de = (deg_ix.loc[(t, dcon, labels[j])] if deg_ix is not None and (t, dcon, labels[j]) in deg_ix.index
                              else None)
                        tops.append(dict(bank=bank, hub=hub, celltype=t, kind=kind, group=name, tail=tail, rank=r + 1,
                                         gene=labels[j], zero_shot=bool(zs[j]), cos=float(S[j]), score=float(v[j]),
                                         own_logfc=np.nan if de is None else float(de.logfc),
                                         own_padj=np.nan if de is None else float(de.pval_adj)))
                    for les, setn in LESION_SETS.items():
                        for gene in [les] + list(sets.get(('module', setn), [])):
                            if gene in lpos:
                                lesion.append(dict(bank=bank, hub=hub, celltype=t, kind=kind, group=name, tail=tail,
                                                   lesion=les, gene=gene, rank=int(rank[lpos[gene]]), n=len(labels),
                                                   zero_shot=bool(zs[lpos[gene]])))
                if deg_ix is not None and kind != 'donor_z':
                    try:
                        dd = degs[(degs.celltype == t) & (degs.contrast == dcon)].set_index('gene')
                    except KeyError:
                        continue
                    common = [l for l in labels if l in dd.index]
                    if len(common) > 100:
                        from scipy.stats import spearmanr
                        idx = [lpos[l] for l in common]
                        rho.append(dict(bank=bank, hub=hub, celltype=t, kind=kind, group=name, n=len(common),
                                        rho=spearmanr(S[idx], dd.loc[common, 'score']).statistic))
    cols = ['bank', 'hub', 'celltype', 'kind', 'group', 'family', 'set', 'n_set', 'z_loss', 'z_gain',
            'p_perm_loss', 'p_perm_gain', 'n_perm']
    pd.DataFrame(tests, columns=cols).to_csv(out / 'set_tests.csv.gz', index=False)
    pd.DataFrame(tops).to_csv(out / f'top{a.top_k}.csv.gz', index=False)
    pd.DataFrame(lesion).to_csv(out / 'lesion_ranks.csv.gz', index=False)
    pd.DataFrame(loo).to_csv(out / 'loo_assign.csv', index=False)
    pd.DataFrame(rho).to_csv(out / 'deg_rho.csv', index=False)
    log.info('wrote %d tests, %d top rows, %d lesion rows', len(tests), len(tops), len(lesion))


# ------------------------------------------------------------------------------------------- genes
def stage_genes(a):
    """Every prototype's score for the contrasts the gene-level panels show (pair contrasts of both groupings and
    each lesion donor vs rest), both banks, ranked by plain cosine (--hub: cos - stored hub offset), merged with that contrast's own-transcript DE."""
    src = Path(a.out_dir)
    b = np.load(src / 'bank.npz', allow_pickle=True)
    obsp = pd.read_csv(src / 'obs_pbulk.csv')
    Zp = np.load(src / 'z_pbulk.npy').astype(np.float64)
    groups = grouping(dict(zip(obsp.donor, obsp.condition)), a.reassign)
    degs = pd.concat([pd.read_csv(f) for f in sorted((src / 'deg').glob('deg__*.csv.gz'))])
    rows = []
    for bank in ('trained', 'full'):
        keep = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
        labels = np.asarray([str(l) for l in b['labels'][keep]])
        Pn = unit(b['P'][keep].astype(np.float64))
        pen = b['hub_pen'][:, keep].astype(np.float64)
        for t in sorted(obsp.celltype.unique()):
            pt = (obsp.celltype == t).to_numpy()
            ds = sorted(obsp.donor[pt].unique())
            pm = {d: Zp[pt & (obsp.donor == d).to_numpy()].mean(0) for d in ds}
            R = {}
            for gname, g in groups.items():
                C = {c: np.mean([pm[d] for d in ds if g[d] == c], 0) for c in sorted({g[d] for d in ds})}
                for c1, c2 in itertools.combinations(sorted(C), 2):
                    R[f'{gname}:{c1}_vs_{c2}'] = C[c1] - C[c2]
            for d in [d for d in list(LESION) + [kv.split('=')[0] for kv in a.reassign] if d in pm]:
                R[f'donor:{d}_vs_rest'] = pm[d] - np.mean([pm[e] for e in ds if e != d], 0)
            for name, r in R.items():
                s = unit(r[None])[0] @ Pn.T
                sc = s - pen[0] if a.hub else s
                rank = np.empty(len(sc), int)
                rank[np.argsort(-sc)] = np.arange(1, len(sc) + 1)
                dd = degs[(degs.celltype == t) & (degs.contrast == name)].set_index('gene')
                rows.append(pd.DataFrame({'bank': bank, 'celltype': t, 'contrast': name, 'gene': labels,
                                          'zero_shot': ~b['trained'].astype(bool)[keep], 'cos': s, 'score': sc,
                                          'rank': rank, 'de_score': dd.score.reindex(labels).values,
                                          'de_logfc': dd.logfc.reindex(labels).values,
                                          'de_padj': dd.pval_adj.reindex(labels).values}))
        log.info('[%s] genes done', bank)
    pd.concat(rows).to_csv(src / 'score' / 'genes.csv.gz', index=False)


# ------------------------------------------------------------------------------------------- umap
def stage_umap(a):
    import umap
    src = Path(a.out_dir)
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    Z = np.load(src / 'z_cells.npy').astype(np.float32)
    keep = ~np.isnan(Z[:, 0])
    Z = Z[keep] - Z[keep].mean(0, keepdims=True)
    E = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=0).fit_transform(unit(Z))
    o = obs[keep].reset_index(drop=True)
    o['umap1'], o['umap2'] = E[:, 0], E[:, 1]
    o.to_csv(src / 'umap_cells.csv.gz', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'deg', 'score', 'genes', 'umap'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--query', help='Weinberg h5ad (raw counts in X)')
    ap.add_argument('--model-dir')
    ap.add_argument('--bank-npz', help='fullbank/bank.npz harvested from the same model')
    ap.add_argument('--modules', help='mito module tsv')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--task', type=int, default=0, help='deg: cell-type index')
    ap.add_argument('--celltype-col', default='singler_labels_main')
    ap.add_argument('--etc-grouping', action='store_true', help='deg: also the etc grouping (Leigh vs non-PMD, PT67 apart)')
    ap.add_argument('--split-col', default=None, help='in-vitro: stim_condition, unit = cell type x stim')
    ap.add_argument('--donor-col', default='patient_id')
    ap.add_argument('--condition-col', default='group_fine')
    ap.add_argument('--reassign', nargs='*', default=['PT67=mito'], help="regrouped grouping: donor=label")
    ap.add_argument('--min-cells', type=int, default=50)
    ap.add_argument('--n-rep', type=int, default=10)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--hub', action='store_true', help='genes: rank by cos - stored hub offset (default plain cosine)')
    ap.add_argument('--perm-batch', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--n-cores', type=int, default=8)
    a = ap.parse_args()
    {'encode': stage_encode, 'deg': stage_deg, 'score': stage_score, 'genes': stage_genes, 'umap': stage_umap}[a.stage](a)
