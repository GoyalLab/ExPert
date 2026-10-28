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

"""Fig. 5g follow-up: is the Tahoe GAPDH attractor biology, geometry, or a readout artifact?

The bank has no 'unperturbed' prototype, so the cosine readout names a top gene for ANY Δz,
including one that is pure noise. Four tests on one encoded arm (no model forward pass):

    support     training cells per perturbation for the attractor genes vs the median gene
    hubness     top-1 frequency of each prototype for DMSO cells scored against the DMSO mean of a
                DISJOINT half (pure noise directions) vs for drug cells. A prototype that also wins on
                noise is a hub of the geometry, not a call
    expression  per cell: Spearman between a gene's own normalised log expression and its prototype's
                cosine (a CRISPRi-trained model may read 'this transcript is down' as the knockdown);
                per drug: is the top-1 gene's OWN mRNA reduced in that drug, vs random genes
    calibrated  an 'unperturbed' call: drug cells whose top-1 cosine does not exceed the 95th
                percentile of the DMSO noise distribution; attractor share above vs below

Writes probe.json and per-gene / per-drug tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def unit(M):
    M = np.asarray(M, np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def top1(D, Pn, chunk=8192):
    out_i, out_c = [], []
    for s in range(0, len(D), chunk):
        S = unit(D[s:s + chunk]) @ Pn.T
        i = S.argmax(1)
        out_i.append(i); out_c.append(S[np.arange(len(i)), i])
    return np.concatenate(out_i), np.concatenate(out_c)


def main(a):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    b = np.load(a.bank, allow_pickle=True)
    P, labels, n_lab = b['P'].astype(np.float32), b['labels'].astype(str), int(b['n_labels'])
    Pn = unit(P); pos = {g: i for i, g in enumerate(labels)}
    enc = Path(a.encode_dir)
    obs = pd.read_parquet(enc / f'obs_{a.arm}.parquet')
    Z = np.load(enc / f'z_{a.arm}.npy').astype(np.float32)
    ctrl = obs.is_control.to_numpy(bool)
    res = {}

    # ---- support ------------------------------------------------------------------------------
    import h5py
    from anndata.io import read_elem
    with h5py.File(a.train_h5ad, 'r') as f:
        cnt = pd.Series(np.asarray(read_elem(f['obs'][a.train_label_col])).astype(str)).value_counts()
    res['support'] = {gname: int(cnt.get(gname, 0)) for gname in a.genes}
    res['support_median_trained'] = float(cnt[cnt.index.isin(labels[:n_lab])].median())
    res['support_rank_pct'] = {gname: float((cnt < cnt.get(gname, 0)).mean()) for gname in a.genes}
    print(f'[probe] support done: {res["support"]} (median {res["support_median_trained"]:.0f})', flush=True)

    # ---- hubness: DMSO noise vs drug cells ----------------------------------------------------
    ci = rng.permutation(np.flatnonzero(ctrl))
    half_a, half_b = ci[: len(ci) // 2], ci[len(ci) // 2:]
    mu_b = Z[half_b].mean(0)
    noise = Z[half_a] - mu_b                         # DMSO cells vs the mean of OTHER DMSO cells
    mu_all = Z[ctrl].mean(0)
    di = np.flatnonzero(~ctrl)
    di = rng.choice(di, min(a.n_drug_cells, len(di)), replace=False)
    drug = Z[di] - mu_all
    ti_n, tc_n = top1(noise, Pn)
    ti_d, tc_d = top1(drug, Pn)
    fn = pd.Series(labels[ti_n]).value_counts(normalize=True)
    fd = pd.Series(labels[ti_d]).value_counts(normalize=True)
    hub = pd.DataFrame({'top1_frac_drug_cells': fd, 'top1_frac_dmso_noise': fn}).fillna(0)
    hub = hub.sort_values('top1_frac_drug_cells', ascending=False)
    hub.head(50).to_csv(out / 'hub_top1.csv')
    res['hub'] = {gname: dict(drug=float(fd.get(gname, 0)), noise=float(fn.get(gname, 0))) for gname in a.genes}
    res['norm'] = dict(drug_median=float(np.median(np.linalg.norm(drug, axis=1))),
                       noise_median=float(np.median(np.linalg.norm(noise, axis=1))))
    print(f'[probe] hub done: {res["hub"]}', flush=True)
    # prototype geometry: alignment with the prototype mean (a 'centre' prototype wins on noise)
    pm = unit(Pn.mean(0, keepdims=True))[0]
    cos_mean = Pn @ pm
    res['cos_to_prototype_mean'] = {gname: float(cos_mean[pos[gname]]) for gname in a.genes if gname in pos}
    res['cos_to_prototype_mean_pct'] = {gname: float((cos_mean < cos_mean[pos[gname]]).mean())
                                        for gname in a.genes if gname in pos}
    # alignment with the mean drug response direction (shared drug stress)
    dm = unit(drug.mean(0, keepdims=True))[0]
    cd = Pn @ dm
    res['cos_to_mean_drug_response_rank'] = {gname: int((cd > cd[pos[gname]]).sum() + 1) for gname in a.genes}

    # ---- calibrated 'unperturbed' call ---------------------------------------------------------
    thr = float(np.quantile(tc_n, 0.95))
    called = tc_d > thr
    res['calibrated'] = dict(threshold_cos=thr, frac_drug_cells_called=float(called.mean()),
                             gapdh_share_called=float((labels[ti_d][called] == 'GAPDH').mean()),
                             gapdh_share_uncalled=float((labels[ti_d][~called] == 'GAPDH').mean()),
                             gapdh_share_noise=float((labels[ti_n] == 'GAPDH').mean()))

    # ---- expression cue ----------------------------------------------------------------------
    import anndata as ad
    import scipy.sparse as sp
    q = ad.read_h5ad(a.query)
    X = sp.csr_matrix(q.X, dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    vpos = {g: i for i, g in enumerate(q.var_names.astype(str))}
    qobs = q.obs.reset_index()
    # align query rows to the encoded obs (same order unless controls were replaced)
    key = qobs.iloc[:, 0].astype(str).to_numpy()
    rowmap = pd.Series(np.arange(len(key)), index=key)
    ri = rowmap.reindex(obs['cell'].astype(str)).to_numpy()
    okr = ~np.isnan(ri)
    expr_rows = np.full(len(obs), -1); expr_rows[okr] = ri[okr].astype(int)

    # every gene the tests touch, gathered once into a dense normalised block: slicing the
    # 204k x 62k CSR per gene x drug took > 15 min
    cand = [g for g in labels[:n_lab] if g in vpos]
    rand_pool = list(rng.choice(cand, 200, replace=False))
    Dall = Z - mu_all
    drugs = obs.loc[~ctrl, 'drug_clean'].astype(str)
    groups = {k: np.asarray(v) for k, v in drugs.groupby(drugs).groups.items()}
    top1_of = {}
    for dname, ix in groups.items():
        s1 = unit(Dall[ix].mean(0, keepdims=True)) @ Pn.T
        top1_of[dname] = labels[int(s1.argmax())]
    need = sorted({g for g in list(a.genes) + rand_pool + list(top1_of.values()) if g in vpos})
    cols = np.array([vpos[g] for g in need])
    Xn = X[:, cols].toarray()
    Xn = np.log1p(Xn / np.maximum(lib, 1)[:, None] * 1e4)
    cpos = {g: i for i, g in enumerate(need)}
    print(f'[probe] dense block {Xn.shape} for {len(need)} genes', flush=True)

    def expr(gene, rows):
        if gene not in cpos:
            return np.full(len(rows), np.nan)
        return Xn[rows, cpos[gene]]

    cell_rows = expr_rows[di]
    keep = cell_rows >= 0
    rows_ok = cell_rows[keep]
    Sd = unit(drug[keep]) @ Pn.T
    cue = {}
    for gname in a.genes:
        if gname not in pos:
            continue
        e = expr(gname, rows_ok)
        r = spearmanr(e, Sd[:, pos[gname]])
        cue[gname] = dict(rho=float(r.statistic), p=float(r.pvalue))
    # random-gene reference: same statistic for 200 random trained genes expressed in the query
    ref = []
    for gname in rand_pool:
        e = expr(gname, rows_ok)
        if np.nanstd(e) > 0:
            ref.append(spearmanr(e, Sd[:, pos[gname]]).statistic)
    res['expression_cue'] = dict(genes=cue, random_gene_rho_median=float(np.nanmedian(ref)),
                                 random_gene_rho_q05=float(np.nanquantile(ref, 0.05)),
                                 random_gene_rho_q95=float(np.nanquantile(ref, 0.95)))
    # per drug: own-mRNA change of the drug's top-1 gene vs random genes
    print(f'[probe] expression cue done: {res["expression_cue"]["genes"]}', flush=True)
    crow = expr_rows[ctrl]; crow = crow[crow >= 0]
    rows_out = []
    for dname, ix in groups.items():
        r = expr_rows[ix]; r = r[r >= 0]
        if len(r) < a.min_cells:
            continue
        g1 = top1_of[dname]
        if g1 not in cpos:
            continue
        de = float(np.mean(expr(g1, r)) - np.mean(expr(g1, crow)))
        rg = list(rng.choice(rand_pool, 20, replace=False))
        dr = [float(np.mean(expr(x, r)) - np.mean(expr(x, crow))) for x in rg]
        rows_out.append(dict(drug=dname, top1=g1, own_delta_expr=de, random_delta_median=float(np.median(dr))))
    pdg = pd.DataFrame(rows_out)
    pdg.to_csv(out / 'top1_own_expression.csv', index=False)
    res['top1_own_expression'] = dict(
        n_drugs=int(len(pdg)), frac_top1_own_mrna_down=float((pdg.own_delta_expr < 0).mean()),
        frac_random_gene_down=float((pdg.random_delta_median < 0).mean()),
        median_own_delta=float(pdg.own_delta_expr.median()),
        by_top1=pdg.groupby('top1').own_delta_expr.agg(['size', 'median']).sort_values('size', ascending=False)
        .head(10).round(4).to_dict('index'))
    json.dump(res, open(out / 'probe.json', 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--encode-dir', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--query', required=True, help='the encoded query h5ad (raw counts)')
    ap.add_argument('--train-h5ad', required=True, help='training DB, obs read with h5py only')
    ap.add_argument('--train-label-col', default='perturbation')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--genes', nargs='+', default=['GAPDH', 'PTMA', 'GRWD1', 'SUPT20H', 'ACTB'])
    ap.add_argument('--n-drug-cells', type=int, default=60000)
    ap.add_argument('--min-cells', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
