# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Fig. 4 (combinatorial screens) through the SHIPPED predict() path, on pseudobulks.

STEP=encode  one pseudobulk per condition (mean raw counts, n_cells kept), ONE default predict() per input
             variant: plain | mtmask (MT-*, MT1*, MT2A, MT3, MT4 zeroed) | tmask (every screen target gene
             zeroed; diagnostic only -- v0.1.0 was trained without target masking and reads the knocked-down
             gene). The readout calibration (depth-matched control noise) is added to the loaded model IN
             MEMORY from the model's own training control pools, which is what makes predict()'s default
             readout ('deflate') active; the shipped model.pt is not written. Writes z_<variant>.npy, obs.csv,
             bank.npz (the scored class bank), noise.npy, and the expression pseudobulks for the baselines.
STEP=score   every method on the SAME candidate set (the screen's single-perturbation genes that are trained
             prototypes = 'panel'; ExPert also on all 8,964 trained prototypes = 'trained'):
               expert      calibrated_readout (predict's default: deflation + depth-matched stop + PM-2 head 10)
                           and plain cosine (round 1), on loss / gain / best-of tails
               ceiling     the screen's OWN single-perturbation pseudobulks as the dictionary:
                             gene  log1p expression deltas, every screen target gene masked (free label removed)
                             z     the same singles' ExPert latent deltas (same encode as the queries)
               centroid    training-data centroids (crispr_signature_ref: one log1p delta per perturbed gene,
                           averaged over the training datasets, model input genes), Pearson; target genes
                           masked in query and reference
             Metrics: doubles both@k / per-gene recall@k / any@k, singles recall@k, analytic random baseline.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import fig4_common as fc

MT_MASK = 'MT-*,MT1*,MT2A,MT3,MT4'
KS = (1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)


def unit(M):
    M = np.asarray(M, np.float64)
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def load_query(a):
    import anndata as ad
    q = ad.read_h5ad(a.query)
    X = sp.csr_matrix(q.X)
    var = q.var_names.astype(str).to_numpy()
    if a.drop_features:
        rx = re.compile(a.drop_features)
        keep = np.array([rx.search(v) is None for v in var])
        print(f'[features] dropped {int((~keep).sum())} matching {a.drop_features!r}', flush=True)
        X, var = X[:, keep], var[keep]
    alias = fc.read_alias(a.alias_map)
    groups, targets = fc.build_truth(q.obs, a.pert_col, a.pert_sep, a.gene_cols, a.group_col,
                                     fc.NULL_TOKENS, a.strip_suffix, alias)
    tokens = fc.panel_tokens(q.obs, a.pert_col, a.pert_sep, a.gene_cols, fc.NULL_TOKENS,
                             a.strip_suffix, alias)
    tgt = {}
    for g, t in zip(groups, targets):
        tgt.setdefault(g, set()).add(t)
    bad = [g for g, s in tgt.items() if len(s) > 1]
    if bad:
        raise SystemExit(f'inconsistent truth within conditions: {bad[:5]}')
    tgt = {g: next(iter(s)) for g, s in tgt.items()}
    return X, var, groups, tgt, tokens


def encode(a):
    import anndata as ad
    from expert import ExPert
    import expert.utils.readout as ro
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    X, var, groups, tgt, tokens = load_query(a)
    keys = sorted(k for k in tgt if k != fc.CONTROL)
    n = np.array([int((groups == k).sum()) for k in keys])
    keys = [k for k, m in zip(keys, n) if m >= a.min_cells]
    allk = keys + [fc.CONTROL]
    pos = {k: i for i, k in enumerate(allk)}
    rows = np.array([pos.get(g, -1) for g in groups])
    ok = rows >= 0
    G = sp.csr_matrix((np.ones(ok.sum(), np.float32), (rows[ok], np.flatnonzero(ok))), shape=(len(allk), X.shape[0]))
    cnt = np.asarray(G.sum(1)).ravel()
    PBraw = sp.csr_matrix(sp.diags(1.0 / np.maximum(cnt, 1)) @ G @ X)
    obs = pd.DataFrame({'condition': allk, 'n_cells': cnt.astype(int),
                        'targets': [';'.join(tgt[k]) if k != fc.CONTROL else '' for k in allk],
                        'order': [len(tgt[k]) if k != fc.CONTROL else 0 for k in allk]})
    print(f'[query] {X.shape[0]:,} cells; {len(keys)} conditions >= {a.min_cells} cells '
          f'(singles {int((obs.order == 1).sum())}, doubles {int((obs.order == 2).sum())}, '
          f'higher {int((obs.order > 2).sum())}); control {int(cnt[-1]):,} cells', flush=True)
    obs.to_csv(out / 'obs.csv', index=False)
    # SPLIT-HALF of every condition: correlation between the control-centred responses of two disjoint halves
    # of its cells, in gene space. This is the reproducibility ceiling at that condition's depth -- an
    # additivity R^2 below it is noise, not biology.
    rng_h = np.random.default_rng(0)
    g2 = np.array(groups, dtype=object).copy()
    for k_ in allk:
        ii = np.flatnonzero(groups == k_)
        if len(ii) >= 4:
            h = rng_h.permutation(ii)[: len(ii) // 2]
            g2[h] = str(k_) + '#h2'
            g2[np.setdiff1d(ii, h)] = str(k_) + '#h1'
    kk = [f'{k_}#h{j}' for k_ in allk for j in (1, 2)]
    Lh = fc.log_pseudobulk(X, g2, kk)
    base = Lh[kk.index(f'{fc.CONTROL}#h1')] / 2 + Lh[kk.index(f'{fc.CONTROL}#h2')] / 2
    rows_h = []
    for k_ in keys:
        try:
            d1 = Lh[kk.index(f'{k_}#h1')] - base
            d2 = Lh[kk.index(f'{k_}#h2')] - base
        except ValueError:
            continue
        rows_h.append(dict(condition=k_, splithalf_r=float(np.corrcoef(d1, d2)[0, 1])))
    pd.DataFrame(rows_h).to_csv(out / 'splithalf.csv', index=False)
    print(f'[splithalf] median r {np.median([r["splithalf_r"] for r in rows_h]):.3f} over {len(rows_h)} conditions',
          flush=True)

    # expression pseudobulks for the gene-space ceiling and the centroid baseline (log1p CPM, cell mean)
    L = fc.log_pseudobulk(X, groups, allk)
    np.save(out / 'logpb.npy', L.astype(np.float32))
    np.save(out / 'genes.npy', var)
    tcols = np.flatnonzero(np.isin(var, list(tokens)))
    np.save(out / 'target_cols.npy', tcols)

    model = ExPert.load(a.model_dir, adata=None)
    info = model.add_readout_calibration(a.ctrl_pools)            # in memory only -- never saved
    print(f'[calibration] {info}', flush=True)
    noise = model.module.readout_noise.detach().cpu().numpy().astype(np.float64)
    np.save(out / 'noise.npy', noise)
    variants = {'plain': None, 'mtmask': MT_MASK, 'tmask': 'TARGETS'}
    bank_done = False
    for vname, mask in variants.items():
        Xv = PBraw
        if mask == 'TARGETS':
            Xv = fc.zero_columns(PBraw, tcols)
        elif mask:
            pats = [p for p in mask.split(',') if p]
            hit = np.array([any(fnmatch.fnmatch(v, p) for p in pats) for v in var])
            Xv = sp.csr_matrix(PBraw @ sp.diags(np.where(hit, 0.0, 1.0)))
            print(f'[{vname}] zeroed {int(hit.sum())} genes', flush=True)
        pb = ad.AnnData(Xv.astype(np.float32), obs=obs.assign(perturbation=['control' if k == fc.CONTROL else k
                                                                            for k in allk]).set_index('condition', drop=False),
                        var=pd.DataFrame(index=var))
        pb.obs.index = pb.obs.index.astype(str)
        pb.obs[model.original_batch_key] = a.screen
        model.refresh_ctrl_gene_means(pb, min_ctrl=1)
        model.predict(pb, inplace=True, return_latents=True, soft_predictions=True, head='A',
                      refresh_ctrl_means=False, n_cells_key='n_cells', control_label='control',
                      progress_bar=False, n_cores=a.n_cores)
        Z = np.asarray(pb.obsm[ro.LATENT_KEY], np.float32)
        np.save(out / f'z_{vname}.npy', Z)
        # predict()'s own default soft predictions: top-50 trained labels per row, kept as a sanity record
        sp_key = [k for k in pb.obsm if 'soft' in k.lower()] + [k for k in pb.uns if 'soft' in k.lower()]
        rr = pb.obs[['condition', 'readout_rounds', 'readout_effect', 'readout_confidence']] \
            if 'readout_rounds' in pb.obs else pb.obs[['condition']]
        rr.to_csv(out / f'predict_obs_{vname}.csv', index=False)
        if not bank_done:
            key = [k for k in pb.uns if k.endswith('_latent_cls_proj')]
            P = np.asarray(pb.uns[key[0]], np.float32)
            _, lab_tr, _ = ro.perturbation_prototypes(model)
            labels = np.asarray(model.idx_to_label).astype(str)[:P.shape[0]]
            np.savez(out / 'bank.npz', P=P, labels=labels, n_labels=len(lab_tr))
            bank_done = True
        print(f'[{vname}] z {Z.shape}; rounds used {pb.obs.get("readout_rounds", pd.Series([np.nan])).describe().to_dict()}',
              flush=True)
    json.dump(dict(screen=a.screen, model=a.model_dir, ctrl_pools=a.ctrl_pools, calibration=info),
              open(out / 'meta.json', 'w'), indent=1, default=str)


def rank_lists(orders, truth, k):
    """per condition: ranks (1-based, inf if absent in the top-k list) of each true gene."""
    out = []
    for o, t in zip(orders, truth):
        pos = {int(x): i + 1 for i, x in enumerate(o[:k])}
        out.append([pos.get(int(g), np.inf) for g in t])
    return out


def metrics(R, order, nb):
    rows = {}
    for k in KS:
        rec = np.array([np.mean([r <= k for r in rr]) for rr in R])
        both = np.array([all(r <= k for r in rr) for rr in R])
        anyk = np.array([any(r <= k for r in rr) for rr in R])
        rows[f'recall@{k}'] = rec.mean(); rows[f'both@{k}'] = both.mean(); rows[f'any@{k}'] = anyk.mean()
        # analytic random: per-gene k/nb; both (order 2) k(k-1)/(nb(nb-1))
        rows[f'rand_recall@{k}'] = min(k, nb) / nb
        rows[f'rand_both@{k}'] = (min(k, nb) / nb) if order == 1 else (min(k, nb) * (min(k, nb) - 1)) / (nb * (nb - 1))
    return rows


def score(a):
    from expert.utils.readout import calibrated_readout
    enc = Path(a.encode_dir); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(enc / 'obs.csv', keep_default_na=False)
    b = np.load(enc / 'bank.npz', allow_pickle=True)
    P, L, nl = b['P'].astype(np.float64), b['labels'].astype(str), int(b['n_labels'])
    noise = np.load(enc / 'noise.npy')
    ctrl = int(np.flatnonzero(obs.condition == fc.CONTROL)[0])
    q = obs.index[obs.order >= 1].to_numpy()
    tg = [t.split(';') for t in obs.targets[q]]
    order = obs.order[q].to_numpy()
    ncell = obs.n_cells[q].to_numpy()
    lab_pos = {g: i for i, g in enumerate(L[:nl])}
    # PANEL: genes with a single-perturbation condition in this screen that are trained prototypes
    singles = {tuple(t)[0]: i for i, t, o in zip(q, tg, order) if o == 1}
    panel = sorted(g for g in singles if g in lab_pos)
    scorable = np.array([all(g in set(panel) for g in t) for t in tg])
    print(f'[score] {len(q)} conditions, panel {len(panel)} single genes in the trained vocab; scorable '
          f'singles {int((scorable & (order == 1)).sum())}, doubles {int((scorable & (order == 2)).sum())}',
          flush=True)
    tail_main = a.tail
    res, rank_rows = [], []
    cond_names = obs.condition[q].to_numpy()

    def dump(method, variant, dictname, tail, sel, R, names, nb):
        for i, rr in zip(np.flatnonzero(sel), R):
            for g, r in zip(tg[i], rr):
                rank_rows.append((method, variant, dictname, tail, cond_names[i], int(order[i]), g,
                                  float(r), int(nb)))

    def evaluate(method, variant, dictname, orders_by_tail, names):
        nb = len(names)
        npos = {g: i for i, g in enumerate(names)}
        for tail, ords in orders_by_tail.items():
            for o in (1, 2):
                sel = scorable & (order == o)
                if not sel.any():
                    continue
                truth = [[npos[g] for g in tg[i]] for i in np.flatnonzero(sel)]
                R = rank_lists(ords[sel], truth, ords.shape[1])
                res.append(dict(method=method, variant=variant, dict=dictname, tail=tail, order=o, n=int(sel.sum()),
                                n_cand=nb, **metrics(R, o, nb)))
                dump(method, variant, dictname, tail, sel, R, names, nb)
        if 'loss' in orders_by_tail and 'gain' in orders_by_tail:      # best of the two tails
            for o in (1, 2):
                sel = scorable & (order == o)
                if not sel.any():
                    continue
                truth = [[npos[g] for g in tg[i]] for i in np.flatnonzero(sel)]
                Rl = rank_lists(orders_by_tail['loss'][sel], truth, orders_by_tail['loss'].shape[1])
                Rg = rank_lists(orders_by_tail['gain'][sel], truth, orders_by_tail['gain'].shape[1])
                R = [[min(x, y) for x, y in zip(l, g)] for l, g in zip(Rl, Rg)]
                res.append(dict(method=method, variant=variant, dict=dictname, tail='best', order=o, n=int(sel.sum()),
                                n_cand=nb, **metrics(R, o, nb)))
                dump(method, variant, dictname, 'best', sel, R, names, nb)

    heads = [int(h) for h in a.heads.split(',')]

    def readouts(D, Pd, ncl, use_noise):
        """deflate orders per protected head (dict head -> {loss, gain}), the cosine orders, and the head-10 result."""
        k = min(max(max(KS), a.rank_k), len(Pd))
        dfl, r10 = {}, None
        for h in heads:
            r = calibrated_readout(D, Pd, ncl, noise=noise if use_noise else None, rounds=a.rounds, head=h, k=k,
                                   m=a.deflate_m, rank=a.deflate_rank, select=a.deflate_select,
                                   stop=a.deflate_stop, ratio=a.deflate_ratio, damp=a.deflate_damp)
            dfl[h] = {'loss': r['loss'], 'gain': r['gain']}
            if h == a.head or r10 is None:
                r10 = r
        S = unit(D) @ unit(Pd).T
        return dfl, {'loss': np.argsort(-S, 1)[:, :k], 'gain': np.argsort(S, 1)[:, :k]}, r10

    # ---- ExPert ----------------------------------------------------------------------------------------------
    pidx = np.array([lab_pos[g] for g in panel])
    for variant in ('plain', 'mtmask', 'tmask'):
        f = enc / f'z_{variant}.npy'
        if not f.exists():
            continue
        Z = np.load(f).astype(np.float64)
        D = Z[q] - Z[ctrl]
        for dname, idx, names in (('panel', pidx, panel), ('trained', np.arange(nl), list(L[:nl]))):
            dfl, cos, r = readouts(D, unit(P[idx]), ncell, True)
            for h, o_ in dfl.items():
                evaluate(f'expert_deflate_h{h}', variant, dname, o_, names)
            evaluate('expert_cosine', variant, dname, cos, names)
            if dname == 'panel':
                pd.DataFrame(dict(condition=obs.condition[q].to_numpy(), n_cells=ncell,
                                  rounds=np.ravel(r['n_rounds']),
                                  effect=np.ravel(r.get('effect_size', np.full(len(q), np.nan))))).to_csv(
                    out / f'rounds_{variant}.csv', index=False)
        # latent ceiling: the screen's own singles as the dictionary (same encode)
        if variant == 'plain':
            sidx = np.array([singles[g] for g in panel])
            Ds = Z[sidx] - Z[ctrl]
            dfl, cos, _ = readouts(D, unit(Ds), ncell, True)
            for h, o_ in dfl.items():
                evaluate(f'ceiling_z_h{h}', variant, 'panel', o_, panel)
            evaluate('ceiling_z_cosine', variant, 'panel', cos, panel)

    # ---- expression space: ceiling (own singles) and training centroids, target genes masked -------------------
    Lg = np.load(enc / 'logpb.npy').astype(np.float64)
    genes = np.load(enc / 'genes.npy', allow_pickle=True).astype(str)
    tcols = np.load(enc / 'target_cols.npy')
    keep = np.setdiff1d(np.arange(len(genes)), tcols)
    Dg = Lg[q][:, keep] - Lg[ctrl, keep]
    sidx = np.array([singles[g] for g in panel])
    Ds = Lg[sidx][:, keep] - Lg[ctrl, keep]
    dfl, cos, _ = readouts(Dg, Ds, ncell, False)
    for h, o_ in dfl.items():
        evaluate(f'ceiling_gene_h{h}', 'tmask', 'panel', o_, panel)
    evaluate('ceiling_gene_cosine', 'tmask', 'panel', cos, panel)
    z = np.load(a.centroid_ref, allow_pickle=True)
    Rc, lab_c, g_c = z['D'].astype(np.float64), z['labels'].astype(str), z['genes'].astype(str)
    gpos = {g: i for i, g in enumerate(genes)}
    tset = set(genes[tcols])
    use = [(i, gpos[g]) for i, g in enumerate(g_c) if g in gpos and g not in tset]
    ri, qi = np.array([u[0] for u in use]), np.array([u[1] for u in use])
    cpos = {g: i for i, g in enumerate(lab_c)}
    cpanel = [g for g in panel if g in cpos]
    if len(cpanel) < len(panel):
        print(f'[centroid] {len(panel) - len(cpanel)} panel genes have no training centroid: '
              f'{sorted(set(panel) - set(cpanel))[:10]}', flush=True)
    Rp = Rc[np.ix_([cpos[g] for g in cpanel], ri)]
    Dq = Lg[q][:, qi] - Lg[ctrl, qi]
    zc = lambda M: (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-12)
    Sc = zc(Dq) @ zc(Rp).T / Dq.shape[1]
    k = min(max(KS), len(cpanel))
    # centroid candidate set may be smaller than the panel: score conditions whose genes all have a centroid
    cs = scorable.copy()
    scorable_c = np.array([all(g in set(cpanel) for g in t) for t in tg])
    scorable[:] = scorable_c & cs
    evaluate('centroid_pearson', 'tmask', 'panel', {'loss': np.argsort(-Sc, 1)[:, :k], 'gain': np.argsort(Sc, 1)[:, :k]},
             cpanel)
    dfl_c, _, _ = readouts(zc(Dq), zc(Rp), ncell, False)
    for h, o_ in dfl_c.items():
        evaluate(f'centroid_deflate_h{h}', 'tmask', 'panel', o_, cpanel)
    scorable[:] = cs
    print(f'[centroid] {len(use)} genes shared with the reference (targets masked), {len(cpanel)} panel genes', flush=True)

    # ---- per-double structure: orthogonality of the two singles and additivity of the double ------------------
    spaces = {'gene': (Dg, {g: Ds[j] for j, g in enumerate(panel)})}
    for variant in ('plain', 'mtmask'):
        f = enc / f'z_{variant}.npy'
        if f.exists():
            Z = np.load(f).astype(np.float64)
            spaces[f'z_{variant}'] = (Z[q] - Z[ctrl], {g: Z[singles[g]] - Z[ctrl] for g in panel})
    cosf = lambda x, y: float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
    # SPLIT-HALF reproducibility of each double's own response (written by STEP=encode): the ceiling any
    # additivity statistic can reach at that depth. Without it a low R^2 cannot be told apart from a noisy
    # pseudobulk (the FITTED R^2 tracked n_cells at rho +0.58 on ProPer-seq).
    hp = enc / 'splithalf.csv'
    half = pd.read_csv(hp).set_index('condition')['splithalf_r'].to_dict() if hp.exists() else {}
    st = []
    for i in np.flatnonzero(scorable & (order == 2)):
        ga, gb = tg[i]
        row = dict(screen=a.screen, condition=cond_names[i], gene_a=ga, gene_b=gb, n_cells=int(ncell[i]),
                   splithalf_r=float(half.get(cond_names[i], np.nan)))
        for sp_, (Dq_, S_) in spaces.items():
            y, xa, xb = Dq_[i], S_[ga], S_[gb]
            Xab = np.stack([xa, xb], 1)
            coef, *_ = np.linalg.lstsq(Xab, y, rcond=None)
            res_ = y - Xab @ coef
            # ADDITIVITY, as the degree to which the double IS the sum of its singles. The fitted R^2 below
            # (free coefficients) is NOT that: it rewards a double that collapses onto ONE single, and it
            # tracks cos_sum (rho ~.95) and sequencing depth, i.e. it is mostly an SNR measure. The headline
            # statistic is therefore the SUM model with coefficients FIXED at 1, and the non-additive part is
            # its residual as a fraction of the response. Both are reported; the fitted version is kept for
            # reference and is the one to use only when the question is about the SPAN, not the sum.
            add = xa + xb
            r_sum = y - add
            row.update({f'{sp_}_cos_singles': cosf(xa, xb), f'{sp_}_cos_sum': cosf(y, add),
                        f'{sp_}_r2_sum': 1 - float(r_sum @ r_sum) / max(float(y @ y), 1e-12),
                        f'{sp_}_r2_fitted': 1 - float(res_ @ res_) / max(float(y @ y), 1e-12),
                        f'{sp_}_r2_additive': 1 - float(res_ @ res_) / max(float(y @ y), 1e-12),
                        f'{sp_}_nonadd_frac': float(np.linalg.norm(r_sum) / max(np.linalg.norm(y), 1e-12)),
                        f'{sp_}_coef_a': float(coef[0]), f'{sp_}_coef_b': float(coef[1]),
                        f'{sp_}_coef_sum': float(coef.sum()),
                        f'{sp_}_norm_ratio': float(np.linalg.norm(y) / max(np.linalg.norm(add), 1e-12))})
        st.append(row)
    pd.DataFrame(st).to_csv(out / 'doubles_stats.csv', index=False)
    pr = []
    for sp_, (_, S_) in spaces.items():
        M = unit(np.stack([S_[g] for g in panel]))
        C = M @ M.T
        iu = np.triu_indices(len(panel), 1)
        pr.append(pd.DataFrame(dict(screen=a.screen, space=sp_, gene_a=np.array(panel)[iu[0]],
                                    gene_b=np.array(panel)[iu[1]], cos=C[iu])))
    pd.concat(pr).to_csv(out / 'singles_pairs.csv', index=False)
    pd.DataFrame(rank_rows, columns=['method', 'variant', 'dict', 'tail', 'condition', 'order', 'gene', 'rank',
                                     'n_cand']).assign(screen=a.screen).to_csv(out / 'ranks.csv.gz', index=False)
    print(f'[stats] {len(st)} doubles; {len(rank_rows):,} rank rows', flush=True)

    t = pd.DataFrame(res)
    t.insert(0, 'screen', a.screen)
    t.to_csv(out / 'metrics.csv', index=False)
    pd.set_option('display.width', 250)
    show = t[(t.order == 2) & (t['tail'].isin([tail_main, 'best']))]
    cols = ['method', 'variant', 'dict', 'tail', 'n', 'n_cand', 'both@5', 'both@10', 'both@20', 'recall@10',
            'rand_both@10']
    print(f'\n== {a.screen} DOUBLES (main tail {tail_main}): {len(show)} rows written')
    show1 = t[(t.order == 1) & (t['tail'] == tail_main)]
    print(f'== {a.screen} SINGLES: {len(show1)} rows written')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=('encode', 'score'))
    ap.add_argument('--screen', required=True)
    ap.add_argument('--query', default=None)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--ctrl-pools', default=None, help="model's training control pools (Q, dataset) for calibration")
    ap.add_argument('--centroid-ref', default=None, help='crispr_signature_ref.npz (training centroids)')
    ap.add_argument('--encode-dir', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pert-col', default=None)
    ap.add_argument('--pert-sep', default='_')
    ap.add_argument('--gene-cols', default=None)
    ap.add_argument('--group-col', default=None)
    ap.add_argument('--strip-suffix', default=None)
    ap.add_argument('--drop-features', default=None)
    ap.add_argument('--alias-map', default=None)
    ap.add_argument('--min-cells', type=int, default=25)
    ap.add_argument('--tail', default='loss', choices=('loss', 'gain'), help='CRISPRi -> loss, CRISPRa -> gain')
    ap.add_argument('--ks', default=None, help='comma list overriding the k ladder')
    ap.add_argument('--rank-k', type=int, default=1000, help='readout list length used for per-condition ranks')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--head', type=int, default=10, help='predict() default protected head (rounds table)')
    ap.add_argument('--heads', default='1,2,3,5,10', help='protected-head sweep for every deflate arm')
    ap.add_argument('--deflate-m', type=int, default=25, help='leading-set size per deflation round')
    ap.add_argument('--deflate-rank', type=int, default=3, help='directions removed per round')
    ap.add_argument('--deflate-select', default='fixed', choices=('fixed', 'z', 'zlocal'),
                    help='leading-set selection rule')
    ap.add_argument('--deflate-stop', default='noise', choices=('noise', 'ratio', 'both'),
                    help='stopping rule; ratio is scale-free (RRT-style)')
    ap.add_argument('--deflate-ratio', type=float, default=0.9,
                    help='residual-norm ratio threshold when --deflate-stop uses it')
    ap.add_argument('--deflate-damp', type=float, default=0.0, help='soft-stop blend width (0 = hard rule)')
    ap.add_argument('--n-cores', type=int, default=8)
    a = ap.parse_args()
    if a.ks:
        global KS
        KS = tuple(int(k) for k in a.ks.split(','))
    encode(a) if a.step == 'encode' else score(a)


if __name__ == '__main__':
    main()
