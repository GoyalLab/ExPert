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

"""Hubness correction (inverted softmax, Smith et al. 2017) on the Fig. 5 ALD readout.

--full-bank: atoms = the full bank (trained + zero-shot f(missing) fills); the penalty is estimated for every row.
Queries Q = the Fig. 5 group vectors (rrep: within one donor set, condition latent minus the mean of the set's
AC / AH / Normal; one per condition x cell type x donor set), atoms A = the trained prototype block.
  S        = unit(Q) @ unit(A).T                                   plain cosine (the figure)
  S'_R,b   = S - lse_r(b * unit(R) @ unit(A).T) / b                per-atom popularity penalty from reference set R
Reference sets (analysis/readout/hubness_readout.py isoft, same formula):
  train    v0.1.0's own training pseudobulks (dataset x perturbation, <=10 cells, z - dataset mean): fixed, deployable
  ctrl     control-only pools from the training control database (random 10-cell pools minus dataset mean)
  trans    the ALD group vectors themselves (diagnostic for the failure mode); trans_loo = leave one cell type out
Per variant: top-200 overlap with plain, Fig. 5b set calls (same set z + Normal-calibrated empirical p), the
export / splicing rows, self-DE rate of the top-200 (model-input genes whose own transcript moves in that group in
the tail's direction), and the raw cosine of atoms promoted into the top-200.

    python paper/scripts/fig5/fig5_hubness.py --stage score --variant-dir ... --bank ... --train-cache ... --ctrl-cache ...
    python paper/scripts/fig5/fig5_hubness.py --stage ctrlref --ctrl-h5ad ... --model-dir ... --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig5_centring_audit import NPC, pct_of, set_z
from fig5_variants import PAUSE_BRAKE

CT = ['HSC', 'Hepatocyte', 'Kupffer Cell', 'LSEC', 'T cell']
ROWS = {'mRNA export to cytoplasm': ('reactome', 'Transport of Mature Transcript to Cytoplasm'),
        'nuclear pore complex': ('npc', 'nuclear pore'),
        'TREX mRNA export': ('module', 'rna_export_trex'),
        'Pol II elongation': ('reactome', 'RNA Polymerase II Transcription Elongation'),
        'Pol II pause brake': ('custom', 'pause brake'),
        'mRNA splicing': ('reactome', 'mRNA Splicing - Major Pathway'),
        'cytosolic translation': ('reactome', 'Cap-dependent Translation Initiation'),
        'mitochondrial translation': ('reactome', 'Mitochondrial Translation'),
        'SUMOylation': ('reactome', 'SUMOylation'),
        'chromatin modification': ('reactome', 'Chromatin Modifying Enzymes'),
        'proteasome assembly': ('reactome', 'Proteasome Assembly')}


def unit(M):
    M = np.asarray(M, np.float64)
    return M / np.clip(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12, None)


def lse_penalty(R, An, beta, block=4096):
    """lse_a / beta with lse_a = log sum_r exp(beta * cos(r, a)), streamed over R."""
    m = np.full(An.shape[0], -np.inf)
    acc = np.zeros(An.shape[0])
    for i in range(0, len(R), block):
        x = beta * (R[i:i + block] @ An.T)
        mx = np.maximum(m, x.max(0))
        acc = acc * np.exp(m - mx) + np.exp(x - mx).sum(0)
        m = mx
    return (m + np.log(acc)) / beta


def rrep_queries(var, cond_ctrl='Normal'):
    """Un-normalised Fig. 5 group vectors (same construction as fig5_data.zrep_profiles(raw=True))."""
    obsp = pd.read_csv(Path(var) / 'obs_pbulk.csv')
    Zp = np.load(Path(var) / 'z_pbulk.npy').astype(np.float64)
    keys, V = [], []
    for t in CT:
        pt = (obsp.celltype == t).to_numpy()
        ds = sorted(obsp.donor[pt].unique())
        cnd = {e: obsp.condition[obsp.donor == e].iloc[0] for e in ds}
        pm = {d: Zp[pt & (obsp.donor == d).to_numpy()].mean(0) for d in ds}
        conds = sorted(set(cnd.values()))
        nor = [e for e in ds if cnd[e] == cond_ctrl]
        for k in ('1', '2'):
            trip = {c: [e for e in ds if e.endswith(k) and cnd[e] == c] for c in conds}
            if not trip.get(cond_ctrl):
                trip[cond_ctrl] = nor[:1]
            if any(not v for v in trip.values()):
                continue
            mu = np.mean([pm[trip[c][0]] for c in conds], 0)
            for c in conds:
                d = trip[c][0]
                if d.endswith(k):
                    keys.append((d, c, t)); V.append(pm[d] - mu)
    return keys, unit(np.stack(V))


def stage_ctrlref(a):
    """Control-only reference: random <=10-cell control pools per training dataset, z minus dataset mean."""
    import anndata as ad
    import scipy.sparse as sp
    from fig5_centring_audit import encode, load_model
    model, _ = load_model(a.model_dir)
    bk = ad.read_h5ad(a.ctrl_h5ad, backed='r')
    ds = bk.obs['dataset'].astype(str).to_numpy()
    genes = np.asarray(bk.var_names).astype(str)
    rng = np.random.default_rng(0)
    Q, D, CZ, CD = [], [], [], []
    for d in np.unique(ds):
        idx = np.sort(rng.choice(np.flatnonzero(ds == d), min(a.ctrl_cells, int((ds == d).sum())), replace=False))
        X = sp.csr_matrix(bk.X[idx])
        z, _ = encode(model, X, genes, np.ones(len(idx), bool), a)  # all cells are the baseline: own controls
        z = np.asarray(z, np.float64)
        base = z.mean(0)
        perm = rng.permutation(len(idx))
        if a.cells_out:                                   # single control cells: the per-cell noise reference
            sc = perm[:a.ctrl_single]
            CZ.append((z[sc] - base).astype(np.float32)), CD.extend([d] * len(sc))
        for s in range(0, len(perm) - 9, 10):
            Q.append(z[perm[s:s + 10]].mean(0) - base)
            D.append(d)
        print(f'{d}: {len(idx)} control cells -> {len(perm) // 10} pools', flush=True)
    np.savez_compressed(a.out, Q=np.stack(Q).astype(np.float32), dataset=np.asarray(D))
    if a.cells_out:
        np.savez_compressed(a.cells_out, Z=np.concatenate(CZ), dataset=np.asarray(CD))
        print(f'ctrl single cells: {len(CD):,} -> {a.cells_out}', flush=True)
    print(f'ctrl reference: {len(Q):,} pools -> {a.out}', flush=True)


def stage_score(a):
    import expert.utils.readout as ro
    b = np.load(a.bank, allow_pickle=True)
    tr = b['trained'].astype(bool)
    n_tr = int(tr.sum())
    rows_ = np.ones(len(tr), bool) if a.full_bank else tr
    labels = [str(l) for l in b['labels'][rows_]]
    zero_shot = ~tr[rows_]
    An = unit(b['P'][rows_])
    lpos = {l: i for i, l in enumerate(labels)}
    keys, Q = rrep_queries(a.variant_dir)
    S = Q @ An.T
    refs = {'train': unit(np.load(a.train_cache, allow_pickle=True)['Q'])}
    tz = np.load(a.train_cache, allow_pickle=True)
    trained_labels = [str(l) for l in b['labels'][tr]]
    assert list(map(str, tz['labels'][:n_tr])) == trained_labels, 'training cache atoms != bank labels'
    if a.ctrl_cache:
        refs['ctrl'] = unit(np.load(a.ctrl_cache)['Q'])
    refs['trans'] = Q
    variants = {'plain': S}
    for rn, R in refs.items():
        if len(R) > a.max_ref:
            R = R[np.random.default_rng(0).choice(len(R), a.max_ref, replace=False)]
        for beta in a.betas:
            variants[f'{rn}_b{int(beta)}'] = S - lse_penalty(R, An, beta)[None, :]
    # leave-one-cell-type-out transductive: the penalty for a group's own cell type comes from the OTHER cell types
    ct_of = np.array([k[2] for k in keys])
    for beta in a.betas:
        M = np.empty_like(S)
        for t in CT:
            m = ct_of == t
            M[m] = S[m] - lse_penalty(Q[~m], An, beta)[None, :]
        variants[f'trans_loo_b{int(beta)}'] = M
    # sets
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in lpos]
    sets[('custom', 'pause brake')] = [g for g in PAUSE_BRAKE if g in lpos]
    row_keys = {lab: fk for lab, fk in ROWS.items() if fk in sets}
    members = {fk: np.array([lpos[g] for g in sets[fk]]) for fk in row_keys.values()}
    all_members = [np.array([lpos[g] for g in v if g in lpos]) for v in sets.values()]
    inputs = set(json.load(open(Path(a.variant_dir) / 'encode.json'))['input_genes'])
    deg = pd.read_csv(a.deg)
    sig = deg[(deg.pval_adj < 0.05) & (deg.logfc.abs() >= 0.25)]
    down = set(zip(sig.group[sig.logfc < 0], sig.gene[sig.logfc < 0]))
    up = set(zip(sig.group[sig.logfc > 0], sig.gene[sig.logfc > 0]))
    is_dis = np.array([k[1] != 'Normal' for k in keys])
    plain_top = {i: set(np.argsort(-S[i])[:a.k]) for i in range(len(keys))}
    plain_top_g = {i: set(np.argsort(S[i])[:a.k]) for i in range(len(keys))}
    summary, calls, promoted = [], [], []
    for vn, M in variants.items():
        # Fig. 5b style set z per group, both tails; null = the Normal groups' |z| of this variant
        zz = {}
        for fk, mem in members.items():
            zz[fk] = np.array([set_z(pct_of(M[i]), [mem])[0][1] for i in range(len(keys))])
        # null = every gene set scored on the Normal groups (as Fig. 5b): healthy-reference variation of set z
        nullz = np.sort(np.abs(np.concatenate([[z_ for _, z_, _ in set_z(pct_of(M[i]), all_members)]
                                               for i in np.flatnonzero(~is_dis)])))
        ov_l, ov_g, sde, pr_cos, zs_l, zs_g = [], [], [], [], [], []
        for i, (d, c, t) in enumerate(keys):
            if c == 'Normal':
                continue
            tl, tg = np.argsort(-M[i])[:a.k], np.argsort(M[i])[:a.k]
            zs_l.append(zero_shot[tl].mean()); zs_g.append(zero_shot[tg].mean())
            ov_l.append(len(set(tl) & plain_top[i]) / a.k)
            ov_g.append(len(set(tg) & plain_top_g[i]) / a.k)
            grp = f'{c}|{t}'
            sde.append(np.mean([labels[j] in inputs and (grp, labels[j]) in down for j in tl] +
                               [labels[j] in inputs and (grp, labels[j]) in up for j in tg]))
            new = [j for j in tl if j not in plain_top[i]]
            pr_cos += [S[i, j] for j in new]
            if vn != 'plain':
                for j in new[:10]:
                    promoted.append(dict(variant=vn, donor=d, celltype=t, gene=labels[j], zero_shot=bool(zero_shot[j]),
                                         raw_cos=S[i, j],
                                         raw_rank=int((S[i] > S[i, j]).sum()) + 1, new_rank=int(np.where(tl == j)[0][0]) + 1))
        n_sig = {}
        for lab, fk in row_keys.items():
            for i, (d, c, t) in enumerate(keys):
                if c == 'Normal':
                    continue
                z_ = zz[fk][i]
                p = (1 + len(nullz) - np.searchsorted(nullz, abs(z_), side='left')) / (1 + len(nullz))
                calls.append(dict(variant=vn, row=lab, donor=d, condition=c, celltype=t, z=z_, p_emp=p))
                n_sig[lab] = n_sig.get(lab, 0) + int(p < 0.01)
        ex = pd.DataFrame([r for r in calls if r['variant'] == vn and r['row'] in
                           ('mRNA export to cytoplasm', 'nuclear pore complex')])
        summary.append(dict(variant=vn, top_overlap_loss=np.mean(ov_l), top_overlap_gain=np.mean(ov_g),
                            zero_shot_share_loss=np.mean(zs_l), zero_shot_share_gain=np.mean(zs_g),
                            self_DE_rate=np.mean(sde), promoted_raw_cos_median=np.median(pr_cos) if pr_cos else np.nan,
                            plain_top_cos_min=np.median([np.sort(S[i])[-a.k] for i in range(len(keys)) if is_dis[i]]),
                            export_AC_loss_z=ex[(ex.condition == 'AC')].z.mean(),
                            export_AH_loss_z=ex[(ex.condition == 'AH')].z.mean(),
                            **{f'sig_{k}': v for k, v in n_sig.items()}))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sm = pd.DataFrame(summary)
    sm.to_csv(out / 'hubness_summary.csv', index=False)
    pd.DataFrame(calls).to_csv(out / 'hubness_calls.csv', index=False)
    pd.DataFrame(promoted).to_csv(out / 'hubness_promoted.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print(sm.round(3).T.to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['score', 'ctrlref'])
    ap.add_argument('--variant-dir')
    ap.add_argument('--bank')
    ap.add_argument('--train-cache', help='cache_queries.py npz of the model\'s capped training pseudobulks')
    ap.add_argument('--ctrl-cache', default='', help='ctrlref output npz')
    ap.add_argument('--modules')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--deg')
    ap.add_argument('--out-dir')
    ap.add_argument('--betas', type=float, nargs='+', default=[10, 30, 100])
    ap.add_argument('--k', type=int, default=200)
    ap.add_argument('--full-bank', action='store_true', help='score all 19,895 rows (trained + zero-shot fills)')
    ap.add_argument('--max-ref', type=int, default=60000)
    ap.add_argument('--model-dir')
    ap.add_argument('--ctrl-h5ad')
    ap.add_argument('--ctrl-cells', type=int, default=2000, help='control cells per training dataset (ctrlref)')
    ap.add_argument('--out')
    ap.add_argument('--cells-out', default='', help='ctrlref: also save single control-cell deltas here')
    ap.add_argument('--ctrl-single', type=int, default=500, help='single control cells saved per dataset')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--n-cores', type=int, default=8)
    a = ap.parse_args()
    {'score': stage_score, 'ctrlref': stage_ctrlref}[a.stage](a)
