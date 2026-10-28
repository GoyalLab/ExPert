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

"""ALD disease readout chain, fresh from predict(): input centring x latent centring x grouping x scoring, judged on
what the model predicts as perturbed per condition and how well that matches known biology -- without the signal
being shared with healthy (a shared offset / mis-centring flag).

stage encode  (one task per INPUT arm; one predict() call on single nuclei, mito_mt mask, center_mode = 0)
    ct_normal      center_by cell type, baseline = the Normal nuclei of that cell type (the documented recipe)
    ct_all         center_by cell type, baseline = every nucleus of that cell type (query mean, no control)
    donor_all      center_by donor, baseline = every nucleus of that donor (removes donor-wide offsets)
    donorct_self   center_by donor x cell type, baseline = the group itself     (negative control: erases signal)
    condct_self    center_by condition x cell type, baseline = the group itself (negative control)
    pooled_normal  no center_by, ONE baseline = every Normal nucleus across cell types
    trained        refresh_ctrl_means=False: the trained baseline of the placeholder dataset (no query control)
    (+ pb_ct_normal at score time: the current Fig. 5 encode, one pseudobulk per donor x cell type, Normal pools as
    baseline -- fig5_predict.py predict_groups.npz)
    predict()'s own group readout (decode_perturbations, groups = condition x cell type) is saved for a parity check.
stage score   (one task per arm x bank)
    latent     none | trained (- align_ds_mean of the placeholder) | query_mean | ct_mean | ct_normal | donor_mean;
               exact post-hoc shifts of z (centring is linear); irrelevant to the vsN contrast
    level      condition (all donors of a condition pooled per cell type -- the past best grouping) | donor
    readout    deflate (predict()'s group readout: calibrated_readout + transplant_scores at the group's cell count)
               | cosine
    scheme     abs_top     the group's own top-200 per tail
               unionz_all  union of every group's top-200 (per tail), each gene z-scored ACROSS ALL groups of the
                           level; the group's list = union genes with z >= 1 (loss) / z <= -1 (gain) -- the original
                           plot_perturbation_enrichment readout taken to pathways
               unionz_ct   the same with z across the groups of one cell type
               vsN         contrast: group - Normal of the cell type (Normal donors: the other Normal donor), top-200
    ORA of every list against Reactome 10-500 + readout modules + grouped mRNA export + mRNA splicing (incl. SR
    proteins); background = the bank.
stage summarize  scorecard: target hits (p_emp < .05 vs every Normal-group test of the same config, expected
    direction), opposite calls, hits SHARED with the Normal group of that cell type (same set and direction, p < .05:
    a shared offset), donor replication, and offset diagnostics (correlation / top-200 overlap of disease vs Normal
    absolute reads).

    PYTHONPATH=paper/scripts:. python paper/scripts/fig5/fig5_readout_chain.py --stage encode --task 0 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig5_variants import MASKS, mask_query

CT = ['HSC', 'Hepatocyte', 'Kupffer Cell', 'LSEC', 'T cell']
ARMS = ['ct_normal', 'ct_all', 'donor_all', 'donorct_self', 'condct_self', 'pooled_normal', 'trained',
        'pb_ct_normal']
LATENT = ['none', 'trained', 'query_mean', 'ct_mean', 'ct_normal', 'donor_mean']
SCHEMES = ['abs_top', 'unionz_all', 'unionz_ct', 'vsN']
# (label, set key, conditions, cell types, expected direction, reference) -- condition level
TARGETS = [
    ('mRNA export', 'mRNA export (grouped)', ('AC', 'AH'), ('Hepatocyte',), 'loss', 'RNA FISH (this study)'),
    ('mRNA splicing', 'mRNA splicing', ('AH',), ('Hepatocyte',), 'loss', 'Chembazhi 2025'),
    ('TGF-b signalling', 'Reactome: Signaling by TGFB Family Members', ('AC', 'AH'), tuple(CT), 'any',
     'Chembazhi 2025'),
    ('adult hepatocyte TFs', 'module: adult_hep_tf', ('AC', 'AH'), ('Hepatocyte',), 'loss', 'Chembazhi 2025'),
    ('mitochondrial translation', 'Reactome: Mitochondrial Translation', ('AH',), ('Hepatocyte',), 'gain', 'Hao 2020'),
    ('chromatin modification', 'Reactome: Chromatin Modifying Enzymes', ('AH',), ('Hepatocyte',), 'loss',
     'Ghare 2023'),
    ('SUMOylation', 'Reactome: SUMOylation', ('AC', 'AH'), tuple(CT), 'loss', 'Chandla 2026'),
    ('cytosolic translation', 'Reactome: Cap-dependent Translation Initiation', ('AC', 'AH'), ('Hepatocyte',),
     'loss', 'Rothschild 1971'),
    ('proteasome assembly', 'Reactome: Proteasome Assembly', ('AC', 'AH'), ('Hepatocyte',), 'loss', 'Wang 2016'),
    ('macroautophagy', 'Reactome: Macroautophagy', ('AH',), ('Hepatocyte',), 'loss', 'Chao 2018'),
    ('amino acids -> mTORC1', 'Reactome: Amino Acids Regulate mTORC1', ('AC', 'AH'), ('Hepatocyte',), 'any',
     'Davuluri 2021'),
]


def cond_of(d):
    return 'Normal' if d.startswith('Nor') else d[:2]


# ------------------------------------------------------------------------------------------- encode
def stage_encode(a):
    import anndata as ad
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS, PREDICTION_KEYS
    arm = ARMS[a.task]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    if not (hasattr(model.module, 'has_readout_noise') and model.module.has_readout_noise()):
        model.add_readout_calibration(a.ctrl_pools)
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    q, _ = mask_query(q, MASKS['mito_mt'])
    obs = pd.DataFrame({'donor': q.obs['sample'].astype(str).to_numpy(),
                        'celltype': q.obs['CellTypeSR'].astype(str).to_numpy()}, index=q.obs_names)
    obs['condition'] = obs.donor.map(cond_of)
    keep = obs.celltype.isin(CT).to_numpy()
    q, obs = q[keep].copy(), obs[keep].copy()
    obs['grp'] = obs.donor + '|' + obs.celltype
    obs['cond_ct'] = obs.condition + '|' + obs.celltype
    nor = (obs.condition == 'Normal').to_numpy()
    ctrl = {'ct_normal': nor, 'pooled_normal': nor, 'trained': np.zeros(len(obs), bool)}.get(arm, np.ones(len(obs), bool))
    obs['perturbation'] = np.where(ctrl, 'control', 'query')
    q.obs = obs
    n_shared = int(model.module.align_ds_mean.shape[1])
    kw = dict(return_latents=True, obs_only=True, progress_bar=False, head='A', readout='cosine',
              readout_confidence=False, center_mode=np.zeros(n_shared, np.float32), control_label='control',
              decode_perturbations=True, decode_group_key='cond_ct')
    cb = {'ct_normal': 'celltype', 'ct_all': 'celltype', 'donor_all': 'donor', 'donorct_self': 'grp',
          'condct_self': 'cond_ct'}.get(arm)
    if cb:
        kw['center_by'] = cb
    if arm == 'trained':
        kw['refresh_ctrl_means'] = False
    o = model.predict(q, **kw)
    Z = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)
    ph = str(q.obs['dataset'].iloc[0]) if 'dataset' in q.obs else str(model.idx_to_batch[0])
    bi = list(map(str, model.idx_to_batch)).index(ph)
    dec = o[PREDICTION_KEYS.DECODE_KEY]
    np.savez_compressed(out / f'enc__{arm}.npz', Z=Z.astype(np.float16),
                        align=model.module.align_ds_mean[bi].detach().cpu().numpy(),
                        noise=model.module.readout_noise.detach().cpu().numpy(),
                        dec_scores=dec['scores'].to_numpy(np.float32), dec_groups=np.asarray(dec['groups']),
                        dec_labels=np.asarray(dec['scores'].columns).astype(str))
    obs[['donor', 'celltype', 'condition']].to_csv(out / f'obs__{arm}.csv.gz', index=False)
    print(f'{arm}: {len(obs):,} nuclei, {int(ctrl.sum()):,} baseline, placeholder {ph}, center_by {cb}', flush=True)


# ------------------------------------------------------------------------------------------- score
def gene_sets(a, labels):
    import fig5_rank as fr
    from fig5_data import read_gmt
    sets = fr.gene_sets(a, labels)
    allr = read_gmt(a.reactome_gmt, labels, lo=1, hi=10_000)
    for _, key, *_ in TARGETS:
        if key not in sets and key.startswith('Reactome: '):
            sets[key] = allr.get(key[len('Reactome: '):], [])
    return {k: v for k, v in sets.items() if len(v) >= 5}


def deflate_scores(Q, P, n, noise):
    import expert.utils.readout as ro
    Qn = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    Pn = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)
    S1 = Qn @ Pn.T
    rr = ro.calibrated_readout(Q, P, np.asarray(n, np.float64), noise=noise, rounds=3, head=10, k=200)
    return S1, ro.transplant_scores(S1, rr['loss'], rr['gain'])


def group_latents(Z, obs, level, min_cells):
    key = (obs.condition + '|' + obs.celltype) if level == 'condition' else (obs.donor + '|' + obs.celltype)
    w = obs['w'].to_numpy(np.float64) if 'w' in obs else np.ones(len(obs))
    G, n = {}, {}
    for k, ix in pd.Series(np.arange(len(obs))).groupby(key.to_numpy()).groups.items():
        ix = np.asarray(ix)
        if w[ix].sum() < min_cells:
            continue
        G[k] = (Z[ix] * w[ix, None]).sum(0) / w[ix].sum()
        n[k] = w[ix].sum()
    return G, n


def lists_for(S, keys, scheme, top):
    """{key: (loss_idx_set, gain_idx_set)} for the score rows S (keys aligned)."""
    o = np.argsort(-S, axis=1, kind='stable')
    tl = {k: (set(o[i, :top].tolist()), set(o[i, ::-1][:top].tolist())) for i, k in enumerate(keys)}
    if scheme in ('abs_top', 'vsN'):
        return tl
    out = {}
    ct = np.array([k.split('|')[1] for k in keys])
    blocks = [np.arange(len(keys))] if scheme == 'unionz_all' else [np.flatnonzero(ct == c) for c in np.unique(ct)]
    for b in blocks:
        if len(b) < 3:
            continue
        ul = np.array(sorted(set().union(*[tl[keys[i]][0] for i in b])))
        ug = np.array(sorted(set().union(*[tl[keys[i]][1] for i in b])))
        Zl = (S[np.ix_(b, ul)] - S[np.ix_(b, ul)].mean(0)) / (S[np.ix_(b, ul)].std(0) + 1e-12)
        Zg = (S[np.ix_(b, ug)] - S[np.ix_(b, ug)].mean(0)) / (S[np.ix_(b, ug)].std(0) + 1e-12)
        for r, i in enumerate(b):
            out[keys[i]] = (set(ul[Zl[r] >= 1].tolist()), set(ug[Zg[r] <= -1].tolist()))
    return out


def stage_score(a):
    arm, bank = ARMS[a.task // 2], ['trained', 'full'][a.task % 2]
    src = Path(a.out_dir)
    b = np.load(a.bank_npz, allow_pickle=True)
    rows_ = b['trained'].astype(bool) if bank == 'trained' else np.ones(len(b['labels']), bool)
    labels = [str(l) for l in b['labels'][rows_]]
    P = b['P'][rows_].astype(np.float64)
    N = len(labels)
    sets = gene_sets(a, labels)
    names = list(sets)
    pos = {g: i for i, g in enumerate(labels)}
    M = np.zeros((len(names), N), np.float32)
    for i, k in enumerate(names):
        M[i, [pos[g] for g in sets[k] if g in pos]] = 1
    nset = M.sum(1)
    if arm == 'pb_ct_normal':
        e = np.load(a.predict_npz, allow_pickle=True)
        obs = pd.read_csv(Path(a.predict_npz).parent / 'obs_groups.csv')
        obs = obs.assign(w=obs.n_cells.astype(float))[['donor', 'celltype', 'condition', 'w']]
        obs['condition'] = obs.donor.map(cond_of)
        Z, align, noise = e['Z'].astype(np.float64), None, e['noise'].astype(np.float64)
        min_cells = 30
    else:
        e = np.load(src / f'enc__{arm}.npz', allow_pickle=True)
        obs = pd.read_csv(src / f'obs__{arm}.csv.gz')
        Z, align, noise = e['Z'].astype(np.float64), e['align'].astype(np.float64), e['noise'].astype(np.float64)
        min_cells = 30
    recs, diag = [], []

    def score_lists(L, meta):
        for k, (ll, gl) in L.items():
            for tail, lst in (('loss', ll), ('gain', gl)):
                K = len(lst)
                if K == 0:
                    continue
                v = np.zeros(N, np.float32)
                v[list(lst)] = 1
                h = M @ v
                p = hypergeom.sf(h - 1, N, nset, K)
                recs.append(pd.DataFrame(dict(**meta, group=k, tail=tail, set=names, K=K, hits=h.astype(np.int16),
                                              expected=(nset * K / N).astype(np.float32), p=p.astype(np.float64))))
    # parity: offline condition-level deflate (latent none) vs predict()'s own group readout
    if arm != 'pb_ct_normal' and bank == 'trained':
        G, n = group_latents(Z, obs, 'condition', 1)
        keys = list(e['dec_groups'])
        _, S = deflate_scores(np.stack([G[k] for k in keys]), P, [n[k] for k in keys], noise)
        print(f'parity vs predict() group readout: max |diff| {np.abs(S - e["dec_scores"]).max():.2e}', flush=True)
    for lat in LATENT:
        if lat == 'trained' and align is None:
            continue
        Zl = Z.copy()
        if lat == 'trained':
            Zl -= align
        elif lat == 'query_mean':
            Zl -= np.average(Zl, axis=0, weights=obs['w'] if 'w' in obs else None)
        elif lat in ('ct_mean', 'ct_normal', 'donor_mean'):
            col = 'donor' if lat == 'donor_mean' else 'celltype'
            for g, ix in obs.groupby(col).indices.items():
                ref = ix if lat != 'ct_normal' else ix[(obs.condition.to_numpy()[ix] == 'Normal')]
                if len(ref):
                    Zl[ix] -= Zl[ref].mean(0)
        for level in ('condition', 'donor'):
            G, n = group_latents(Zl, obs, level, min_cells)
            keys = sorted(G)
            Q = np.stack([G[k] for k in keys])
            S1, Sd = deflate_scores(Q, P, [n[k] for k in keys], noise)
            for rd, S in (('cosine', S1), ('deflate', Sd)):
                # offset diagnostics: disease vs Normal absolute reads within a cell type
                for c in CT:
                    nk = [k for k in keys if k.endswith('|' + c) and cond_of(k.split('|')[0]) == 'Normal']
                    for k in keys:
                        if k.endswith('|' + c) and nk and k not in nk:
                            i, j = keys.index(k), keys.index(nk[0])
                            jac = len(set(np.argsort(-S[i])[:200]) & set(np.argsort(-S[j])[:200])) / 200
                            diag.append(dict(arm=arm, bank=bank, latent=lat, level=level, readout=rd, group=k,
                                             r_vs_normal=float(np.corrcoef(S[i], S[j])[0, 1]), top200_jaccard=jac))
                for sch in ('abs_top', 'unionz_all', 'unionz_ct'):
                    score_lists(lists_for(S, keys, sch, a.top),
                                dict(arm=arm, bank=bank, latent=lat, level=level, readout=rd, scheme=sch))
            if lat == 'none':                       # vsN: latent shifts cancel
                Qv, kv, nv = [], [], []
                for k in keys:
                    d, c = k.split('|')
                    if level == 'condition':
                        ref = [x for x in keys if x == f'Normal|{c}']
                        if cond_of(d) == 'Normal':
                            continue
                    else:
                        ref = [x for x in keys if x.endswith('|' + c) and cond_of(x.split('|')[0]) == 'Normal'
                               and x != k]
                    if not ref:
                        continue
                    Qv.append(G[k] - np.mean([G[x] for x in ref], 0))
                    kv.append(k)
                    nv.append(1.0 / (1.0 / n[k] + sum(1.0 / n[x] for x in ref) / len(ref) ** 2))
                S1v, Sdv = deflate_scores(np.stack(Qv), P, nv, noise)
                for rd, S in (('cosine', S1v), ('deflate', Sdv)):
                    score_lists(lists_for(S, kv, 'vsN', a.top),
                                dict(arm=arm, bank=bank, latent='n/a', level=level, readout=rd, scheme='vsN'))
    df = pd.concat(recs, ignore_index=True)
    for c in ('arm', 'bank', 'latent', 'level', 'readout', 'scheme', 'group', 'tail', 'set'):
        df[c] = df[c].astype('category')
    df.to_parquet(src / f'tests__{arm}__{bank}.parquet', index=False)
    pd.DataFrame(diag).to_csv(src / f'diag__{arm}__{bank}.csv', index=False)
    print(f'{arm} {bank}: {len(df):,} tests', flush=True)


# ------------------------------------------------------------------------------------------- summarize
def stage_summarize(a):
    src = Path(a.out_dir)
    res, calls = [], []
    for f in sorted(src.glob('tests__*.parquet')):
        df = pd.read_parquet(f)
        df['donor'] = df.group.astype(str).str.split('|').str[0]
        df['celltype'] = df.group.astype(str).str.split('|').str[1]
        df['condition'] = df.donor.map(cond_of)
        for key, x in df.groupby(['arm', 'bank', 'latent', 'readout', 'scheme'], observed=True):
            cfg = dict(zip(['arm', 'bank', 'latent', 'readout', 'scheme'], key))
            # null: every Normal-group test of this config; for vsN at condition level (no Normal group) the
            # donor-level Normal-vs-other-Normal tests
            nul = x[(x.condition == 'Normal')]
            if not len(nul[nul.level == 'condition']):
                nul_c = nul[nul.level == 'donor']
            else:
                nul_c = nul[nul.level == 'condition']
            row = dict(cfg)
            for level in ('condition', 'donor'):
                y = x[x.level == level]
                if not len(y):
                    continue
                nl = nul_c if level == 'condition' else nul[nul.level == 'donor']
                null = np.sort(nl.p.to_numpy())
                pe = lambda p: (1 + np.searchsorted(null, p, side='right')) / (1 + len(null))
                tk = {t[1] for t in TARGETS}
                y = y[y.set.isin(tk)]
                y = y.assign(pe=pe(y.p.to_numpy()))
                D_ = {(str(g), str(t), str(s_)): (p, q) for g, t, s_, p, q in
                      zip(y.group, y['tail'], y.set, y.p, y.pe)}
                hit = opp = shared = cells = 0
                hit_keys = []
                for lab, sk, conds, cts, dirn, ref in TARGETS:
                    tails = ['loss', 'gain'] if dirn == 'any' else [dirn]
                    other = {'loss': 'gain', 'gain': 'loss'}
                    groups = sorted({g for g in y.group.astype(str).unique()
                                     if cond_of(g.split('|')[0]) in conds and g.split('|')[1] in cts})
                    for g in groups:
                        cells += 1
                        sig = lambda g_, t_: (g_, t_, sk) in D_ and max(D_[(g_, t_, sk)]) < .05
                        ok = [t for t in tails if sig(g, t)]
                        bad = dirn != 'any' and sig(g, other[dirn])
                        if ok:
                            hit += 1
                            hit_keys.append((lab, g))
                            c = g.split('|')[1]
                            ng = [h for h in y.group.astype(str).unique() if h.endswith('|' + c)
                                  and cond_of(h.split('|')[0]) == 'Normal']
                            if any((h, t, sk) in D_ and D_[(h, t, sk)][0] < .05 for h in ng for t in ok):
                                shared += 1
                        opp += int(bad)
                        calls.append(dict(**cfg, level=level, target=lab, group=g,
                                          call='hit' if ok else 'opposite' if bad else ''))
                row.update({f'{level}_hits': hit, f'{level}_of': cells, f'{level}_opposite': opp,
                            f'{level}_shared_with_normal': shared})
                if level == 'condition':
                    row['export_hep'] = ','.join(sorted(g.split('|')[0] for lab, g in hit_keys
                                                        if lab == 'mRNA export' and g.endswith('Hepatocyte')))
                    row['splicing_hep_AH'] = any(lab == 'mRNA splicing' for lab, g in hit_keys)
                if level == 'donor':                    # both donors of a condition hit the same target x cell type
                    hk = pd.DataFrame(hit_keys, columns=['lab', 'g'])
                    if len(hk):
                        hk['cond'] = hk.g.str.split('|').str[0].map(cond_of)
                        hk['ct'] = hk.g.str.split('|').str[1]
                        row['donor_replicated'] = int((hk.groupby(['lab', 'cond', 'ct']).size() == 2).sum())
                    else:
                        row['donor_replicated'] = 0
            res.append(row)
    R = pd.DataFrame(res)
    D = pd.concat([pd.read_csv(f) for f in sorted(src.glob('diag__*.csv'))])
    dg = (D[D.level == 'condition'].groupby(['arm', 'bank', 'latent', 'readout'])[['r_vs_normal', 'top200_jaccard']]
          .mean().reset_index())
    R = R.merge(dg, on=['arm', 'bank', 'latent', 'readout'], how='left')
    R['net'] = R.condition_hits - R.condition_opposite - R.condition_shared_with_normal
    R = R.sort_values(['net', 'donor_replicated'], ascending=False)
    R.to_csv(src / 'chain_scorecard.csv', index=False)
    pd.DataFrame(calls).to_csv(src / 'chain_calls.csv.gz', index=False)
    pd.set_option('display.width', 280, 'display.max_rows', 120, 'display.max_columns', 30)
    print(R.head(60).round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'score', 'summarize'])
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model-dir')
    ap.add_argument('--ctrl-pools', help='training control pools for the readout calibration (fig5_hubness ctrlref)')
    ap.add_argument('--query', help='alcohol_readout.h5ad (UNMASKED; the mito_mt mask is applied here)')
    ap.add_argument('--predict-npz', help='fig5_predict.py predict_groups.npz (the pb_ct_normal arm)')
    ap.add_argument('--bank-npz')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--modules')
    ap.add_argument('--top', type=int, default=200)
    a = ap.parse_args()
    {'encode': stage_encode, 'score': stage_score, 'summarize': stage_summarize}[a.stage](a)
