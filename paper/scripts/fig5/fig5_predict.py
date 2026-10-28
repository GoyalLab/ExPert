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

"""Fig. 5 readout through the CURRENT predict(): one pseudobulk per donor x cell type (= condition x donor x cell
type, every donor belongs to one condition), the shipped deflated readout, no hubness correction.

stage encode  (model load)
    query      mito_mt mask (MT-* + metallothioneins); per donor x cell type ONE row = mean raw counts of all its
               nuclei (n_cells = group size). The input baseline needs >= 10 control rows per cell type (else
               predict() silently keeps a TRAINING dataset's baseline), so the Normal donors' nuclei also enter as
               --n-rep random pools per donor labelled 'control' -- baseline only, never scored (exactly the
               Fig. 5 pbulk_x baseline).
    predict    ONE call: center_by='celltype', center_mode=0 (constant shifts cancel in the contrast), control_label
               'control', n_cells_key, readout='deflate' and hub_correction=False (both the defaults, set
               explicitly). Saves the group latents, predict()'s own per-row soft predictions / rounds / effect,
               and the readout calibration it used.
stage score   (no model load; one task per bank: trained | full = with the zero-shot f(missing) rows)
    contrast   rrep: within a cell type and donor set {AC1,AH1,Nor1} / {AC2,AH2,Nor2}, q = z_group - mean(z of the
               set's AC, AH, Normal); Normal rows are the null. Depth n_eff = 9 / (4/n_d + 1/n_o1 + 1/n_o2).
    readout    expert.utils.readout.deflate_rows + transplant_scores -- the arithmetic predict() applies to every
               row -- on q, at its n_eff, with the model's calibration; 'cosine' (round 1 only) for comparison.
    unit       'group' (one pseudobulk per donor x cell type, this script) and 'pools' (the cached Fig. 5 pbulk_x
               pools averaged in z, fig5_variants encode) side by side.
    out        tests__predict__<bank>.csv.gz (set z, fig5_variants format: readout = <unit>_<readout>) and
               profiles__predict__<bank>.npz (score vectors per donor x cell type, every unit x readout).

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_predict.py --stage encode ...
"""
from __future__ import annotations

import argparse
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
from fig5_centring_audit import NPC, unit, pct_of, set_z
from fig5_variants import MASKS, PAUSE_BRAKE, mask_query

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig5_predict')
log.setLevel(logging.INFO)


# spliceosome / splicing-regulator families for the splicing probe (curated by complex before looking at ranks)
SPLICE = {
    'U1 snRNP': ['SNRNP70', 'SNRPA', 'SNRPC'],
    'U2 snRNP / SF3': ['SF3A*', 'SF3B*', 'U2AF1', 'U2AF2', 'PHF5A'],
    'Sm core': ['SNRPB', 'SNRPD1', 'SNRPD2', 'SNRPD3', 'SNRPE', 'SNRPF', 'SNRPG'],
    'U4/U6.U5 tri-snRNP': ['PRPF3', 'PRPF4', 'PRPF6', 'PRPF8', 'PRPF31', 'PRPF38A', 'SNRNP200', 'EFTUD2', 'SART1',
                           'USP39', 'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'LSM8'],
    'SR proteins': ['SRSF*', 'TRA2A', 'TRA2B'],
    'hnRNPs': ['HNRNP*'],
    'splicing regulators (ESRP/RBFOX/PTBP/MBNL/CELF)': ['ESRP1', 'ESRP2', 'RBFOX1', 'RBFOX2', 'RBFOX3', 'PTBP1',
                                                        'PTBP2', 'PTBP3', 'MBNL1', 'MBNL2', 'MBNL3', 'CELF1',
                                                        'CELF2', 'QKI', 'RBM47'],
    'exon junction complex': ['EIF4A3', 'MAGOH', 'RBM8A', 'CASC3'],
}


def expand(pats, vocab):
    return list(dict.fromkeys(g for p in pats for g in vocab if (g.startswith(p[:-1]) if p.endswith('*') else g == p)))


def stage_encode(a):
    import anndata as ad
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    if not (hasattr(model.module, 'has_readout_noise') and model.module.has_readout_noise()):
        log.info('no stored readout calibration -> %s', model.add_readout_calibration(a.ctrl_pools))
    noise = model.module.readout_noise.detach().cpu().numpy().astype(np.float64)
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    q, dropped = mask_query(q, MASKS[a.mask])
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    cond = q.obs[a.condition_col].astype(str).to_numpy()
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    cts = a.celltypes.split(',')
    keep = np.isin(ct, cts)
    rng = np.random.default_rng(0)
    X, meta = [], []
    for t in cts:
        for d in sorted(set(donor[keep & (ct == t)])):
            idx = np.flatnonzero(keep & (ct == t) & (donor == d))
            if len(idx) < a.min_cells:
                continue
            c = cond[idx[0]]
            X.append(np.asarray(q.X[idx].mean(0)).ravel())
            meta.append(dict(role='group', donor=d, condition=c, celltype=t, n_cells=len(idx)))
            if c == a.control_condition:                  # input baseline only
                for chunk in np.array_split(rng.permutation(idx), a.n_rep):
                    X.append(np.asarray(q.X[chunk].mean(0)).ravel())
                    meta.append(dict(role='baseline', donor=d, condition=c, celltype=t, n_cells=len(chunk)))
    obs = pd.DataFrame(meta)
    obs.index = [f'r{i}' for i in range(len(obs))]
    obs['perturbation'] = np.where(obs.role == 'baseline', 'control', 'query')
    pb = ad.AnnData(X=sp.csr_matrix(np.stack(X).astype(np.float32)), obs=obs,
                    var=pd.DataFrame(index=pd.Index(q.var_names.astype(str))))
    nb = obs[obs.role == 'baseline'].groupby('celltype').size()
    assert (nb >= 10).all(), f'fewer than 10 baseline rows in a cell type: {nb.to_dict()}'
    n_shared = int(model.module.align_ds_mean.shape[1])
    o = model.predict(pb, return_latents=True, obs_only=True, progress_bar=False, head='A',
                      center_by='celltype', center_mode=np.zeros(n_shared, np.float32), control_label='control',
                      n_cells_key='n_cells', readout='deflate', hub_correction=False)
    g = (obs.role == 'group').to_numpy()
    Z = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)
    P, labels = model._readout_prototypes(o.get(MODULE_KEYS.CLS_PROJ_KEY))
    sp_ = o.get('soft_predictions')
    np.savez(out / 'predict_groups.npz', Z=Z[g], P=P.astype(np.float32), labels=np.asarray(labels), noise=noise,
             soft=np.asarray(sp_)[g].astype(np.float32) if sp_ is not None else np.zeros(0),
             soft_cols=np.asarray(getattr(sp_, 'columns', []), str),
             rounds=np.asarray(o.get('readout_rounds', np.ones(len(obs))))[g],
             effect=np.asarray(o.get('readout_effect', np.full(len(obs), np.nan)))[g])
    obs[g].to_csv(out / 'obs_groups.csv', index=False)
    # the model's own prototype bank: trained rows = what predict() scores; untrained rows = the saved f(missing)
    # fill (zero-shot), exactly as add_hub_correction / predict(obs_only=False) build it
    mod = model.module
    A = mod.effective_atoms()
    if A is not None:
        if mod.has_prior_missing_fill() and 'prior_missing_T' in mod._buffers:
            A = mod._prior_missing_apply(A)
        A = A.detach().float().cpu().numpy()
        n_lab = int(mod.n_labels)
        filled = np.zeros(len(A), bool)
        filled[:n_lab] = True
        if 'prior_missing_T' in mod._buffers:
            filled[n_lab:] = mod.prior_missing_T[n_lab:len(A)].detach().float().cpu().norm(dim=1).numpy() > 0
        lab_all = np.asarray(model.idx_to_label).astype(str)[:len(A)]
        A[:n_lab] = P[:n_lab]                     # trained block = the forward-pass class bank (cls2z)
        trained = np.zeros(len(A), bool)
        trained[:n_lab] = True
        np.savez(out / 'bank.npz', P=A[filled].astype(np.float32), labels=lab_all[filled], trained=trained[filled])
        log.info('bank: %d trained + %d zero-shot rows (f(missing) fill)', n_lab, int(filled.sum()) - n_lab)
    json.dump(dict(mask=a.mask, dropped=dropped, n_groups=int(g.sum()), n_baseline=int((~g).sum()),
                   noise=noise.tolist(), readout='deflate', hub_correction=False,
                   rounds=pd.Series(np.asarray(o.get('readout_rounds', [1]))[g]).value_counts().to_dict()),
              open(out / 'encode.json', 'w'), indent=1, default=str)
    log.info('%d groups + %d baseline pools; rounds %s; median effect %.2f', g.sum(), (~g).sum(),
             pd.Series(np.asarray(o.get('readout_rounds', [1]))[g]).value_counts().to_dict(),
             float(np.nanmedian(np.asarray(o.get('readout_effect', [np.nan]))[g])))


def ref_queries(Zg, obs, a, kind):
    """kind 'ctrl': donor - mean of the Normal donors of its cell type; 'rest': donor - mean of every donor of its
    cell type OUTSIDE its condition. A Normal donor is compared with the other Normal donor in both (the healthy
    donor-vs-donor null). Depth n_eff = 1 / (1/n_d + mean(1/n_ref) / |ref|)."""
    Q, keys, neff = [], [], []
    for t in a.celltypes.split(','):
        m = obs[obs.celltype == t]
        cnd = dict(zip(m.donor, m.condition))
        zz = {d: Zg[i] for d, i in zip(m.donor, m.index)}
        nn = dict(zip(m.donor, m.n_cells))
        nor = [d for d in sorted(cnd) if cnd[d] == a.control_condition]
        for d in sorted(cnd):
            if cnd[d] == a.control_condition:
                ref = [e for e in nor if e != d]
            elif kind == 'ctrl':
                ref = nor
            else:
                ref = [e for e in sorted(cnd) if cnd[e] != cnd[d]]
            if not ref:
                continue
            Q.append(zz[d] - np.mean([zz[e] for e in ref], 0))
            keys.append((d, t))
            neff.append(1.0 / (1.0 / nn[d] + sum(1.0 / nn[e] for e in ref) / len(ref) ** 2))
    return np.stack(Q), keys, np.asarray(neff)


def rrep_queries(Zg, obs, a):
    """rrep contrast per (donor, celltype): q, n_eff. A set missing its Normal donor borrows the other one."""
    Q, keys, neff = [], [], []
    for t in a.celltypes.split(','):
        m = obs[obs.celltype == t]
        cnd = dict(zip(m.donor, m.condition))
        zz = {d: Zg[i] for d, i in zip(m.donor, m.index)}
        nn = dict(zip(m.donor, m.n_cells))
        conds = sorted(set(cnd.values()))
        nor = [d for d in sorted(cnd) if cnd[d] == a.control_condition]
        for k in ('1', '2'):
            trip = {c: [d for d in sorted(cnd) if d.endswith(k) and cnd[d] == c] for c in conds}
            if not trip.get(a.control_condition):
                trip[a.control_condition] = nor[:1]
            if any(not v for v in trip.values()):
                continue
            ds = [trip[c][0] for c in conds]
            mu = np.mean([zz[d] for d in ds], 0)
            for d in ds:
                if not d.endswith(k):                     # a borrowed Normal donor is scored in its own set
                    continue
                Q.append(zz[d] - mu)
                keys.append((d, t))
                o = [e for e in ds if e != d]
                neff.append(9.0 / (4.0 / nn[d] + sum(1.0 / nn[e] for e in o)))
    return np.stack(Q), keys, np.asarray(neff)


def readouts(Q, P, neff, noise, a):
    """{'cosine': S1, 'deflate': transplanted S}, rounds -- predict()'s per-row arithmetic on the contrasts."""
    import expert.utils.readout as ro
    S1 = unit(Q) @ unit(P).T
    dr = ro.deflate_rows(Q, P, neff, noise, rounds=a.rounds, head=a.head, k=a.top)
    S = S1.copy()
    for j, i in enumerate(dr['rows']):
        S[i] = ro.transplant_scores(S1[i][None], dr['loss'][j][None], dr['gain'][j][None])[0]
    return {'cosine': S1, 'deflate': S}, dr['n_rounds'], dr['effect_size']


def stage_score(a):
    import expert.utils.readout as ro
    bank = ['trained', 'full'][a.task]
    src = Path(a.out_dir)
    e = np.load(src / 'predict_groups.npz', allow_pickle=True)
    obs = pd.read_csv(src / 'obs_groups.csv')
    b = np.load(a.bank_npz, allow_pickle=True)
    rows_ = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
    labels = [str(l) for l in b['labels'][rows_]]
    P = b['P'][rows_].astype(np.float64)
    # the bank's trained rows must be the prototypes predict() scores against
    pos = {l: i for i, l in enumerate(labels)}
    mlab = [str(x) for x in e['labels']]
    shared = [i for i, l in enumerate(mlab) if l in pos]
    agree = (unit(e['P'][shared].astype(np.float64)) * unit(P[[pos[mlab[i]] for i in shared]])).sum(1)
    log.info('%s bank: %d rows; predict() prototypes matched %d, min cos %.6f', bank, len(labels), len(shared),
             agree.min())
    noise = e['noise'].astype(np.float64)
    units = {'group': (e['Z'].astype(np.float64), obs)}
    if a.pools_dir:                                    # cached Fig. 5 pools, averaged in z per donor x cell type
        op = pd.read_csv(Path(a.pools_dir) / 'obs_pbulk.csv')
        Zp = np.load(Path(a.pools_dir) / 'z_pbulk.npy').astype(np.float64)
        og = op.groupby(['donor', 'celltype']).agg(condition=('condition', 'first'),
                                                    n_cells=('n_cells', 'sum')).reset_index()
        units['pools'] = (np.stack([Zp[((op.donor == r.donor) & (op.celltype == r.celltype)).to_numpy()].mean(0)
                                    for r in og.itertuples()]), og)
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in pos]
    sets[('custom', 'pause brake')] = [g for g in PAUSE_BRAKE if g in pos]
    keys_ = list(sets)
    members = [np.array([pos[x] for x in sets[k] if x in pos]) for k in keys_]
    rows, prof, info = [], {}, []
    for un, (Zu, ou) in units.items():
        ou = ou.reset_index(drop=True)
        cnd = dict(zip(ou.donor, ou.condition))
        for kind in a.contrasts.split(','):
            # rrep keeps the historic key names (group_deflate ...); other contrasts get a __<kind> suffix
            sfx = '' if kind == 'rrep' else f'__{kind}'
            Q, gkeys, neff = rrep_queries(Zu, ou, a) if kind == 'rrep' else ref_queries(Zu, ou, a, kind)
            R, rounds, eff = readouts(Q, P, neff, noise, a)
            for i, (d, t) in enumerate(gkeys):
                info.append(dict(unit=un, contrast=kind, donor=d, condition=cnd[d], celltype=t, n_eff=neff[i],
                                 rounds=int(rounds[i]), effect_size=float(eff[i])))
            for rd, S in R.items():
                key = f'{un}_{rd}{sfx}'
                prof[key] = S.astype(np.float32)
                prof[f'keys__{key}'] = np.asarray([f'{d}|{t}' for d, t in gkeys])
                for i, (d, t) in enumerate(gkeys):
                    for (fam, sname), (mu, z, _) in zip(keys_, set_z(pct_of(S[i]), members)):
                        rows.append((key, d, t, fam, sname, mu, z))
            if kind == 'rrep':
                prof[f'{un}_keys'] = np.asarray([f'{d}|{t}' for d, t in gkeys])
    tag = f'predict__{bank}'
    t_ = pd.DataFrame(rows, columns=['readout', 'donor', 'celltype', 'family', 'set', 'mean_pct', 'z'])
    t_.insert(0, 'variant', tag)
    t_.to_csv(src / f'tests__{tag}.csv.gz', index=False)
    np.savez(src / f'profiles__{tag}.npz', labels=np.asarray(labels), zero_shot=~b['trained'][rows_].astype(bool),
             **prof)
    inf = pd.DataFrame(info)
    inf.to_csv(src / f'rounds__{tag}.csv', index=False)
    log.info('%s: %d tests; rounds by unit x contrast x condition\n%s', tag, len(t_),
             inf.groupby(['unit', 'contrast', 'condition']).rounds.value_counts().unstack(fill_value=0).to_string())


def stage_ora(a):
    """Over-representation of every set in each group's top-k per tail (the original ALD readout's per-group top
    lists), per donor x cell type and per condition x cell type (union of the condition's donors' top-k)."""
    import expert.utils.readout as ro
    from scipy.stats import hypergeom
    bank = ['trained', 'full'][a.task]
    src = Path(a.out_dir)
    z = np.load(src / f'profiles__predict__{bank}.npz', allow_pickle=True)
    labels = [str(l) for l in z['labels']]
    N = len(labels)
    pos = {l: i for i, l in enumerate(labels)}
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in pos]
    sets[('custom', 'pause brake')] = [g for g in PAUSE_BRAKE if g in pos]
    for k, pats in SPLICE.items():
        sets[('splice', k)] = expand(pats, labels)
    keys_ = [k for k, v in sets.items() if len(v) >= 3]
    M = np.zeros((len(keys_), N), bool)
    for i, k in enumerate(keys_):
        M[i, [pos[g] for g in sets[k]]] = True
    n_set = M.sum(1)
    rows = []
    for rd in profile_keys(z):
        S = z[rd]
        gk = [tuple(x.split('|')) for x in z[f'keys__{rd}']]
        tops = {}
        for i, (d, t) in enumerate(gk):
            o = np.argsort(-S[i], kind='stable')
            tops[(d, t)] = {'loss': o[:a.top], 'gain': o[::-1][:a.top]}
        lists = [('donor', d, t, tail, idx) for (d, t), tt in tops.items() for tail, idx in tt.items()]
        for t in sorted({t for _, t in gk}):
            for c in ('AC', 'AH', 'Nor'):
                ds = [d for d, tt in tops if tt == t and d.startswith(c)]
                for tail in ('loss', 'gain'):
                    if ds:
                        lists.append(('condition', c, t, tail, np.unique(np.concatenate([tops[(d, t)][tail] for d in ds]))))
        for level, g, t, tail, idx in lists:
            K = len(idx)
            hits = M[:, idx].sum(1)
            exp = n_set * K / N
            p = hypergeom.sf(hits - 1, N, n_set, K)
            for (fam, sname), h, e, ns, pp in zip(keys_, hits, exp, n_set, p):
                rows.append((rd, level, g, t, tail, fam, sname, int(ns), K, int(h), e, pp))
    t_ = pd.DataFrame(rows, columns=['readout', 'level', 'group', 'celltype', 'tail', 'family', 'set', 'n_set', 'K',
                                     'hits', 'expected', 'p'])
    t_.to_csv(src / f'ora__predict__{bank}.csv.gz', index=False)
    log.info('ora %s: %d tests over %s', bank, len(t_), sorted(t_.readout.unique()))


def profile_keys(z):
    return [k for k in z.files if k.startswith(('group_', 'pools_')) and not k.endswith('_keys')]


def condition_lists(z, rd, a):
    """(condition, celltype, tail, union of the condition's donors' top-k) for one profile key."""
    S = z[rd]
    gk = [tuple(x.split('|')) for x in z[f'keys__{rd}']]
    out = []
    for t in sorted({t for _, t in gk}):
        for c in ('AC', 'AH', 'Nor'):
            ix = [i for i, (d, tt) in enumerate(gk) if tt == t and d.startswith(c)]
            for tail, sg in (('loss', 1), ('gain', -1)):
                if ix:
                    out.append((c, t, tail, np.unique(np.concatenate(
                        [np.argsort(-sg * S[i], kind='stable')[:a.top] for i in ix]))))
    return out


def stage_unbiased(a):
    """Unbiased enrichment of each condition x cell type's top-k union (both donors' lists, per tail) over the Fig. 1d
    libraries (GO BP / GO CC / KEGG / Reactome / CORUM / Hallmark, 5-300 members in the bank; background = the
    library's annotated prototypes, as fig1d). Normal unions are scored too (the healthy null)."""
    import scipy.sparse as sps
    from scipy.stats import hypergeom
    from fig1d_tree import load_libraries
    bank = ['trained', 'full'][a.task]
    src = Path(a.out_dir)
    z = np.load(src / f'profiles__predict__{bank}.npz', allow_pickle=True)
    labels = [str(l) for l in z['labels']]
    libs = load_libraries(labels, Path(a.lib_cache), 5, 300)
    lists = []
    for rd in [k for k in profile_keys(z) if k.startswith(a.readout)]:
        lists += [(rd, 'condition') + x for x in condition_lists(z, rd, a)]
        S = z[rd]
        for i, (d, t) in enumerate(tuple(k.split('|')) for k in z[f'keys__{rd}']):
            for tail, sg in (('loss', 1), ('gain', -1)):          # donor level: that donor's own top-k
                lists.append((rd, 'donor', d, t, tail, np.argsort(-sg * S[i], kind='stable')[:a.top]))
    rows = []
    for lib, (terms, G, uni) in libs.items():
        G = G.tocsc()
        Ku = np.asarray(G.sum(0)).ravel()
        N = int(uni.sum())
        for rd, lv, c, t, tail, u in lists:
            uu = u[uni[u]]
            hits = np.asarray(G[uu].sum(0)).ravel()
            j = np.flatnonzero(hits >= 2)
            p = hypergeom.sf(hits[j] - 1, N, Ku[j], len(uu))
            for jj, pp in zip(j, p):
                rows.append((rd, lv, c, t, tail, lib, terms[jj], int(Ku[jj]), len(uu), int(hits[jj]),
                             Ku[jj] * len(uu) / N, pp))
    t_ = pd.DataFrame(rows, columns=['readout', 'level', 'group', 'celltype', 'tail', 'library', 'term', 'n_set', 'K', 'hits',
                                     'expected', 'p'])
    # members of every term that is tested anywhere (for redundancy collapsing downstream)
    mem = {}
    for lib, (terms, G, uni) in libs.items():
        G = G.tocsc()
        for term in set(t_.term[t_.library == lib]):
            jj = terms.index(term)
            mem[f'{lib}|{term}'] = ';'.join(labels[i] for i in G[:, jj].indices)
    t_['members'] = (t_.library + '|' + t_.term).map(mem)
    t_.to_csv(src / f'unbiased__predict__{bank}.csv.gz', index=False)
    log.info('unbiased %s: %d tests (hits >= 2) over %d lists', bank, len(t_), len(lists))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'score', 'ora', 'unbiased'])
    ap.add_argument('--task', type=int, default=0, help='score: 0 = trained bank, 1 = full bank')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model-dir')
    ap.add_argument('--ctrl-pools', help='training control pools (fig5_hubness.py ctrlref) if the model lacks '
                                         'a stored readout calibration')
    ap.add_argument('--query', help='alcohol_readout.h5ad (UNMASKED; the mask is applied here)')
    ap.add_argument('--mask', default='mito_mt', choices=list(MASKS))
    ap.add_argument('--bank-npz', help='score: the model\'s fullbank/bank.npz (trained + zero-shot rows)')
    ap.add_argument('--pools-dir', default='', help='score: cached fig5_variants encode for the pools comparison')
    ap.add_argument('--modules')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--condition-col', default='Condition')
    ap.add_argument('--control-condition', default='Normal')
    ap.add_argument('--celltype-col', default='CellTypeSR')
    ap.add_argument('--donor-col', default='sample')
    ap.add_argument('--celltypes', default='HSC,Hepatocyte,Kupffer Cell,LSEC,T cell')
    ap.add_argument('--min-cells', type=int, default=50)
    ap.add_argument('--n-rep', type=int, default=10, help='baseline pools per Normal donor x cell type')
    ap.add_argument('--rounds', type=int, default=3, help='predict() decode_rounds default')
    ap.add_argument('--head', type=int, default=10, help='predict() decode_head default')
    ap.add_argument('--top', type=int, default=200, help='predict() decode_top default; top-k for ora / unbiased')
    ap.add_argument('--readout', default='group_deflate', help='unbiased: profile-key prefix')
    ap.add_argument('--contrasts', default='rrep,ctrl,rest', help='score: rrep (donor set) / ctrl / rest')
    ap.add_argument('--lib-cache', default='', help='unbiased: Fig. 1d gene-set cache (fig1d_tree.load_libraries)')
    a = ap.parse_args()
    {'encode': stage_encode, 'score': stage_score, 'ora': stage_ora, 'unbiased': stage_unbiased}[a.stage](a)
