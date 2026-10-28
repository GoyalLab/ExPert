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

"""Fig. 6 configuration grid: which readout configuration recovers the mitochondrial-disease signal best?

stage encode (--task = model x mask x baseline): per-cell z, one predict() per cell type.
    mask      none | mt (MT-*) | mt_mt (MT-* + metallothioneins)
    baseline  ctrl (non-PMD donors' cells = encoder input baseline, collaborator protocol) | all (every cell)
stage score: every encode x grouping x aggregation x reference x readout x bank, scored the same way.
    grouping     revised (PMD = PT43/46/54/67) | orig (Leigh = PT43/46/54; PT67 a control, as in the 2026-09-14 artifact)
    aggregation  donor (equal-weight donor centroids; shift = mean of the cases' per-donor shifts) |
                 cell (cell-weighted pooled cases vs pooled reference: the artifact / collaborator 'group' partition)
    reference    ctrl (non-cases) | free (every other donor; group contrast = cases vs all non-cases, same as ctrl)
    readout      cosine | deflate (calibrated_readout on the group shift at its effective depth)
    bank         trained | full
  metrics (monocytes unless noted):
    mito_top20 / mito_top50   OXPHOS/mitochondrial genes in the group profile's top-k (fixed list below, set before
                              looking: the collaborator complex/mito modules + MT-/NDUF/UQCR/COX/ATP5/SDH/MRPL/MRPS)
    p_oxphos                  exact donor-label permutation p of the per-donor OXPHOS score (cases vs non-cases)
    ari / p_ari               label-free 2-cluster cut of the donors' latent shifts vs diagnosis
    lesion_med_pct            median percentile rank of each case's own lesion gene in its own donor profile

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_grid.py --stage encode --task 0 ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
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
log = logging.getLogger('fig6_grid')
log.setLevel(logging.INFO)

DONORS = ['PT43', 'PT46', 'PT54', 'PT67', 'PT38', 'PT50', 'PT55', 'PT41', 'PT42']
GROUPINGS = {'revised': ['PT43', 'PT46', 'PT54', 'PT67'], 'orig': ['PT43', 'PT46', 'PT54']}
NONPMD_BASELINE = ['PT38', 'PT50', 'PT55', 'PT41', 'PT42']
LESION = {'PT43': 'BCS1L', 'PT46': 'NDUFA10', 'PT54': 'NDUFS8', 'PT67': 'PC'}
MASKS = {'none': [], 'mt': ['MT-*'], 'mt_mt': ['MT-*', 'MT1*', 'MT2A', 'MT3', 'MT4']}
CX = ['complex_i', 'complex_iii', 'complex_iv', 'complex_v']
MITO_MODULES = CX + ['complex_ii', 'mitoribosome', 'mt_aminoacyl_trna', 'mtdna_maintenance', 'mtdna_expression',
                     'mito_import', 'coq_biosynthesis', 'iron_sulfur', 'pdh_tca']
MITO_RE = re.compile(r'^(MT-|NDUF|UQCR|COX\d|ATP5|SDH[A-D]|MRPL|MRPS)')


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def grid(a):
    return [(m, k, b) for m in a.models.split(',') for k in MASKS for b in ('ctrl', 'all')]


def stage_encode(a):
    import anndata as ad
    from fig5_centring_audit import encode, load_model
    mtag, mask, base = grid(a)[a.task]
    out = Path(a.out_dir) / f'{mtag}__{mask}__{base}'
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(dict(zip(a.models.split(','), a.model_dirs.split(',')))[mtag])
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(q.n_vars, bool)
    for p in MASKS[mask]:
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    q = q[:, ~drop].copy()
    genes = np.asarray(q.var_names).astype(str)
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    Z = np.full((q.n_obs, int(model.module.align_ds_mean.shape[1])), np.nan, np.float32)
    for t in sorted(set(ct)):
        m = ct == t
        ref = np.isin(donor[m], NONPMD_BASELINE) if base == 'ctrl' else np.ones(m.sum(), bool)
        Z[m], _ = encode(model, q.X[m], genes, ref, a)
    np.save(out / 'z_cells.npy', Z.astype(np.float16))
    pd.DataFrame({'donor': donor, 'celltype': ct}).to_csv(out / 'obs_cells.csv.gz', index=False)
    log.info('%s %s %s: dropped %d genes', mtag, mask, base, int(drop.sum()))


def bank_parts(path, which, modules):
    import expert.utils.readout as ro
    b = np.load(path, allow_pickle=True)
    keep = np.ones(len(b['labels']), bool) if which == 'full' else b['trained'].astype(bool)
    labels = np.asarray([str(l) for l in b['labels'][keep]])
    P = b['P'][keep].astype(np.float64)
    mods = ro.load_modules(modules, list(labels))
    pos = {l: i for i, l in enumerate(labels)}
    mito = {g for m in MITO_MODULES for g in mods.get(m, [])} | {l for l in labels if MITO_RE.match(l)}
    return labels, P, unit(P), {m: np.array([pos[g] for g in mods[m]]) for m in CX}, np.array([l in mito for l in labels]), pos


def oxphos_score(S, cx):
    pct = (-S).argsort(-1).argsort(-1) / S.shape[-1]
    return np.mean([1 - np.median(pct[..., cx[m]], -1) for m in CX], 0)


def perm_p(vals, cases):
    ds = list(vals)
    k = sum(d in cases for d in ds)
    x = np.array([vals[d] for d in ds])
    is_case = np.array([d in cases for d in ds])
    obs = x[is_case].mean() - x[~is_case].mean()
    null = [x[list(c)].mean() - np.delete(x, list(c)).mean() for c in itertools.combinations(range(len(ds)), k)]
    return float(np.mean(np.array(null) >= obs - 1e-12))


def stage_score(a):
    import expert.utils.readout as ro
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score
    out = Path(a.out_dir)
    banks = dict(zip(a.models.split(','), a.banks.split(',')))
    pools = dict(zip(a.models.split(','), a.ctrl_pools.split(',')))
    rows, tops = [], []
    for mtag, mask, base in grid(a):
        src = out / f'{mtag}__{mask}__{base}'
        if not (src / 'z_cells.npy').exists():
            log.warning('missing %s', src)
            continue
        Z = np.load(src / 'z_cells.npy').astype(np.float64)
        obs = pd.read_csv(src / 'obs_cells.csv.gz')
        pl = np.load(pools[mtag], allow_pickle=True)
        BP = {w: bank_parts(banks[mtag], w, a.modules) for w in ('trained', 'full')}
        noise = {}
        for w, (_, P, _, _, _, _) in BP.items():
            basis = ro.span_basis(P, 256)
            noise[w] = (basis, np.array([*ro.fit_noise_curve(pl['Q'], pl['dataset'], None, pool_cells=10),
                                         *ro.fit_noise_curve(pl['Q'], pl['dataset'], basis, pool_cells=10)]))
        for t in a.celltypes.split(','):
            mt = (obs.celltype == t).to_numpy()
            ds = [d for d in DONORS if (mt & (obs.donor == d).to_numpy()).sum() >= a.min_cells]
            cen = {d: Z[mt & (obs.donor == d).to_numpy()].mean(0) for d in ds}
            cnt = {d: int((mt & (obs.donor == d).to_numpy()).sum()) for d in ds}
            free = np.stack([cen[d] - np.mean([cen[e] for e in ds if e != d], 0) for d in ds])
            for gname, cases in GROUPINGS.items():
                cs = [d for d in ds if d in cases]
                rs = [d for d in ds if d not in cases]
                if len(cs) < 2 or len(rs) < 2:
                    continue
                truth = [d in cases for d in ds]
                lab = fcluster(linkage(pdist(free, 'cosine'), 'average'), 2, 'maxclust')
                ari = adjusted_rand_score(truth, lab)
                null = []
                for c in itertools.combinations(range(len(ds)), len(cs)):
                    tt = np.zeros(len(ds), bool)
                    tt[list(c)] = True
                    null.append(adjusted_rand_score(tt, lab))
                p_ari = float(np.mean(np.array(null) >= ari - 1e-12))
                # per-donor shifts for the donor-level tests (ctrl: vs pooled non-cases, LOO for non-cases; free)
                def pooled(mem):
                    w = np.array([cnt[m] for m in mem], float)
                    return (np.stack([cen[m] for m in mem]) * w[:, None]).sum(0) / w.sum()
                donor_sh = {'ctrl': np.stack([cen[d] - pooled([r for r in rs if r != d]) for d in ds]), 'free': free}
                group_sh = {'donor': np.mean([cen[d] for d in cs], 0) - np.mean([cen[d] for d in rs], 0),
                            'cell': pooled(cs) - pooled(rs)}
                n_eff = {'donor': None, 'cell': sum(cnt[d] for d in cs) * sum(cnt[d] for d in rs) /
                         (sum(cnt[d] for d in cs) + sum(cnt[d] for d in rs))}
                n_eff['donor'] = n_eff['cell']
                for w, (labels, P, Pn, cx, ismito, pos) in BP.items():
                    for ref, R in donor_sh.items():
                        S = unit(R) @ Pn.T
                        ox = oxphos_score(S, cx)
                        p_ox = perm_p(dict(zip(ds, ox)), cases)
                        pct = (-S).argsort(-1).argsort(-1) / S.shape[1]
                        les = [pct[ds.index(d), pos[LESION[d]]] for d in cs if LESION[d] in pos]
                        for agg, r in group_sh.items():
                            s1 = unit(r[None]) @ Pn.T
                            for readout in ('cosine', 'deflate'):
                                if readout == 'deflate':
                                    rr = ro.calibrated_readout(r[None], P, n_eff[agg], noise=noise[w][1],
                                                               basis=noise[w][0], k=200)
                                    s = ro.transplant_scores(s1, rr['loss'], rr['gain'])[0]
                                else:
                                    s = s1[0]
                                order = np.argsort(-s)
                                rows.append(dict(model=mtag, mask=mask, baseline=base, celltype=t, grouping=gname,
                                                 aggregation=agg, reference=ref, readout=readout, bank=w,
                                                 n_cases=len(cs), n_ref=len(rs),
                                                 mito_top20=int(ismito[order[:20]].sum()),
                                                 mito_top50=int(ismito[order[:50]].sum()),
                                                 group_oxphos=float(oxphos_score(s[None], cx)[0]),
                                                 p_oxphos=p_ox, ari=ari, p_ari=p_ari,
                                                 lesion_med_pct=float(np.median(les)) if les else np.nan))
                                if t == 'Monocytes':
                                    tops.append(dict(model=mtag, mask=mask, baseline=base, grouping=gname,
                                                     aggregation=agg, reference=ref, readout=readout, bank=w,
                                                     top20=' '.join(labels[j] + ('*' if ismito[j] else '')
                                                                    for j in order[:20])))
        log.info('scored %s %s %s', mtag, mask, base)
    R = pd.DataFrame(rows)
    R.to_csv(out / 'grid_scores.csv', index=False)
    pd.DataFrame(tops).drop_duplicates().to_csv(out / 'grid_top20_monocytes.csv', index=False)
    m = R[(R.celltype == 'Monocytes') & (R.reference == 'ctrl')]
    pd.set_option('display.width', 250, 'display.max_rows', 400, 'display.max_columns', 30)
    log.info('\n%s', m.pivot_table(index=['model', 'mask', 'baseline', 'grouping'],
                                   columns=['bank', 'aggregation', 'readout'], values='mito_top20').to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'score'])
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--query')
    ap.add_argument('--models', default='v0.1.0,EFFGATE3')
    ap.add_argument('--model-dirs')
    ap.add_argument('--banks')
    ap.add_argument('--ctrl-pools', help='comma list aligned with --models (Q + dataset)')
    ap.add_argument('--modules')
    ap.add_argument('--celltypes', default='Monocytes,Dendritic cells,B cells,NK cells,CD4+ T cells,Nonclassical T cells')
    ap.add_argument('--celltype-col', default='singler_labels_main')
    ap.add_argument('--donor-col', default='patient_id')
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--n-cores', type=int, default=8)
    a = ap.parse_args()
    {'encode': stage_encode, 'score': stage_score}[a.stage](a)
