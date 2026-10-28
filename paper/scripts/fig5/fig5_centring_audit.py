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

"""Fig. 5 centring audit: does the INPUT baseline we hand predict() put a disease query in the regime
the encoder was trained in?

The encoder sees only log1p(x) minus a baseline (center_input, center_mode='ctrl'; no covariates),
and predict() sets that baseline to the mean of the cells labelled 'control' IN THE CALL. Every arm
below is therefore just a choice of which cells carry the control label -- the model's own path,
nothing re-implemented. The latent is written raw (center_mode = zeros) and centred afterwards.

stage train  ground truth on training contexts (database_perturbed_200 + database_ctrl):
             own_ctrl   baseline = the context's own controls (the training regime)
             own_mean   baseline = all of the context's cells (controls + perturbed)
             xdonor     baseline = ANOTHER donor's controls, same condition (Marson; the ALD case)
             xcond      baseline = the same donor's controls in another condition (Marson)
             Readout: pseudobulk delta (perturbed - controls) target rank over the trained bank, vs
             own controls AND vs the baseline's controls; per-cell recall@10. Regime statistics:
             per-cell input distance to the baseline, the context's mean offset, latent norms,
             max cosine to the bank.
stage ald    the ALD query under six baselines, one call per stratum:
             pooled (current recipe: per cell type, Normal controls), ref_Nor1, ref_Nor2 (per cell
             type, one healthy donor), self_ctdonor / self_ctcond / self_ct (all cells of the
             cell type x donor / x condition / cell type). Same regime statistics, plus the
             nuclear-pore set per donor x cell type (pseudobulk delta vs the OTHER Normal cells of
             that type; per-cell score), and Spearman between arms of every group's delta profile.

    python paper/scripts/fig5/fig5_centring_audit.py --stage train|ald --model-dir M --bank bank.npz \
        --out-dir D [--train-h5ad ... --ctrl-h5ad ...] [--query alcohol_readout_mtfree.h5ad]
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

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig5_centring_audit')
log.setLevel(logging.INFO)

NPC = ['NUP153', 'NUP188', 'RAE1', 'NUP160', 'NUP54', 'NUP85', 'NUP43', 'NUP50', 'NUP35', 'NUP210',
       'NUP214', 'NUP98', 'NUP205', 'NUP155', 'NUP133', 'NUP107', 'NUP93', 'NUP62', 'NUP58', 'NUP37',
       'SEH1L', 'SEC13', 'AAAS', 'TPR', 'NDC1', 'POM121', 'GLE1', 'NUP88', 'NUP42']


# ------------------------------------------------------------------------------------- encoding
def encode(model, X, genes, ref, args):
    """predict() on raw counts X (cells x genes); cells with ref=True are the 'control' cells whose
    log1p mean becomes the encoder's input baseline. Returns raw z_shared and the baseline buffer."""
    import anndata as ad
    import expert.utils.readout as ro
    a = ad.AnnData(X=X.copy(), var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    a.obs_names = [f'c{i}' for i in range(a.n_obs)]
    a.obs['cond'] = np.where(ref, 'REF', 'Q')
    a.obs['ct'] = 'all'
    ro.build_groups(a, 'cond', 'ct', control_condition='REF')
    ro.mark_query(model, a)
    n_shared = int(model.module.align_ds_mean.shape[1])
    model.predict(a, inplace=True, return_latents=True, soft_predictions=False, head='A', obs_only=True,
                  center_mode=np.zeros(n_shared, np.float32), control_label='control',
                  batch_size=args.batch_size, n_cores=args.n_cores, progress_bar=False)
    buf = model.module.z_encoder.ctrl_gene_means[0].detach().float().cpu().numpy()
    return np.asarray(a.obsm[ro.LATENT_KEY], np.float32), buf


def log1p_dense(X, cols):
    Y = X[:, cols]
    Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
    return np.log1p(Y.astype(np.float32))


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


def regime(xl, base, Z, center, Pn):
    """Input distance to the baseline, latent norm and max cosine to the bank, per cell."""
    d_in = np.linalg.norm(xl - base, axis=1)
    Zc = Z - center
    mc = np.concatenate([(unit(Zc[i:i + 4096]) @ Pn.T).max(1) for i in range(0, len(Zc), 4096)])
    return d_in, np.linalg.norm(Zc, axis=1), mc


def collapse_stats(Zc, Pn, block=4096):
    """Shared-offset collapse on centred latents: ||mean z|| / mean ||z||, mean pairwise cosine, and how
    concentrated the per-cell top-1 prototype is (effective number exp(H), share of the 5 commonest)."""
    U = unit(Zc)
    n = len(U)
    su = U.sum(0)
    pair = float((su @ su - n) / max(n * (n - 1), 1))
    top1 = np.concatenate([(U[i:i + block] @ Pn.T).argmax(1) for i in range(0, n, block)])
    c = np.bincount(top1, minlength=Pn.shape[0]).astype(float)
    p = c[c > 0] / n
    return dict(shared_offset=float(np.linalg.norm(Zc.mean(0)) / np.linalg.norm(Zc, axis=1).mean()),
                mean_pair_cos=pair, top1_eff_n=float(np.exp(-(p * np.log(p)).sum())),
                top1_top5_share=float(np.sort(c)[::-1][:5].sum() / n)), top1


def ranks_of(scores, idx):
    """1-based rank of column idx in each row of scores (descending)."""
    return 1 + (scores > scores[np.arange(len(scores)), idx][:, None]).sum(1)


def load_bank(path):
    b = np.load(path, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = [str(l) for l in b['labels'][tr]]
    return labels, unit(b['P'][tr].astype(np.float32))


def load_model(path):
    from expert import ExPert
    model = ExPert.load(path)
    mod, enc = model.module, model.module.z_encoder
    facts = dict(encode_covariates=bool(getattr(mod, 'encode_covariates', False)),
                 ctrl_pc_mode=str(getattr(enc, 'ctrl_pc_mode', None)),
                 center_input=bool(getattr(enc, 'center_input', False)),
                 center_mode=str(getattr(enc, 'center_mode', None)),
                 invariant_align=bool(getattr(mod, 'invariant_align', False)))
    log.info('model facts %s', facts)
    if facts['encode_covariates'] or facts['ctrl_pc_mode'] not in ('None', 'none', ''):
        log.warning('encoder sees context beyond the centred input -- the audit premise needs revisiting')
    return model, facts


# ------------------------------------------------------------------------------------- training
def read_ctx(path, datasets, max_cells, rng, per_label=None, min_cells=20):
    import anndata as ad
    a = ad.read_h5ad(path, backed='r')
    ds = a.obs['dataset'].astype(str).to_numpy()
    lab = a.obs['perturbation'].astype(str).to_numpy()
    out = {}
    for d in datasets:
        idx = np.flatnonzero(ds == d)
        if per_label:
            keep = []
            for l, c in zip(*np.unique(lab[idx], return_counts=True)):
                if c >= min_cells:
                    ii = idx[lab[idx] == l]
                    keep.append(rng.choice(ii, min(per_label, len(ii)), replace=False))
            idx = np.concatenate(keep) if keep else idx[:0]
        if len(idx) > max_cells:
            idx = rng.choice(idx, max_cells, replace=False)
        idx = np.sort(idx)
        out[d] = (sp.csr_matrix(a.X[idx]), lab[idx])
    return out, np.asarray(a.var_names).astype(str)


def stage_train(a):
    rng = np.random.default_rng(0)
    model, facts = load_model(a.model_dir)
    labels, Pn = load_bank(a.bank)
    lpos = {l: i for i, l in enumerate(labels)}
    genes = np.asarray(model.adata.var_names).astype(str)
    keep_genes = set(pd.read_csv(a.input_genes, header=None)[0].astype(str)) if a.input_genes else set(genes)
    cols = np.array([i for i, g in enumerate(genes) if g in keep_genes])
    marson = [f'Marson2025_D{k}_Rest' for k in range(1, 5)]
    lines = a.line_contexts.split(',')
    all_ctx = lines + marson
    want = all_ctx if a.task is None else [all_ctx[a.task]]
    need_ctrl = set(want)
    for d in want:
        if d.startswith('Marson'):
            k = int(d.split('_D')[1][0])
            need_ctrl |= {f'Marson2025_D{k % 4 + 1}_Rest', f'Marson2025_D{k}_Stim48hr'}
    pert, gp = read_ctx(a.train_h5ad, want, a.max_pert_cells, rng, per_label=a.per_label)
    ctrl, gc = read_ctx(a.ctrl_h5ad, sorted(need_ctrl), a.max_ctrl_cells, rng)
    assert (gp == genes).all() and (gc == genes).all(), 'training files are not on the model gene panel'
    arms = {d: ['own_ctrl', 'own_mean'] for d in lines}
    for d in marson:
        arms[d] = ['own_ctrl', 'own_mean', 'xdonor', 'xcond']
    arms = {d: arms[d] for d in lines + marson}
    contexts = list(arms)
    if a.task is not None:                                     # one SLURM array task = one context
        arms = {contexts[a.task]: arms[contexts[a.task]]}
    rows, prow = [], []
    for d, arm_list in arms.items():
        Xp, lp = pert[d]
        Xc, _ = ctrl[d]
        k = int(d.split('_D')[1][0]) if d.startswith('Marson') else None
        for arm in arm_list:
            t0 = time.time()
            if arm == 'own_ctrl':
                Xr, ref_name = None, d
            elif arm == 'own_mean':
                Xr, ref_name = None, d + ' (all cells)'
            elif arm == 'xdonor':
                ref_name = f'Marson2025_D{k % 4 + 1}_Rest'
                Xr = ctrl[ref_name][0]
            else:
                ref_name = f'Marson2025_D{k}_Stim48hr'
                Xr = ctrl[ref_name][0]
            blocks = [Xp, Xc] + ([Xr] if Xr is not None else [])
            X = sp.vstack(blocks).tocsr()
            n_p, n_c = Xp.shape[0], Xc.shape[0]
            is_p = np.r_[np.ones(n_p, bool), np.zeros(X.shape[0] - n_p, bool)]
            is_own_c = np.r_[np.zeros(n_p, bool), np.ones(n_c, bool), np.zeros(X.shape[0] - n_p - n_c, bool)]
            is_r = ~(is_p | is_own_c)
            ref = {'own_ctrl': is_own_c, 'own_mean': is_p | is_own_c}.get(arm, is_r)
            Z, buf = encode(model, X, genes, ref, a)
            xl = log1p_dense(X, cols)
            base = xl[ref].mean(0)
            q = is_p | is_own_c                                    # the context's own cells
            center = Z[q].mean(0)                                  # training: per-dataset mean of z
            d_in, zn, mc = regime(xl[q], base, Z[q], center, Pn)
            refc = is_r if arm in ('xdonor', 'xcond') else is_own_c
            # pseudobulk delta per perturbation, vs the comparison controls and vs own controls
            res = []
            for l in np.unique(lp):
                if l not in lpos:
                    continue
                m = np.flatnonzero(lp == l)
                zp = Z[m].mean(0)
                for vs, cm in (('ref_ctrl', refc), ('own_ctrl', is_own_c)):
                    s = unit(zp - Z[cm].mean(0)) @ Pn.T
                    res.append(dict(context=d, arm=arm, vs=vs, perturbation=l, n_cells=len(m),
                                    rank=int(1 + (s > s[lpos[l]]).sum()),
                                    delta_norm=float(np.linalg.norm(zp - Z[cm].mean(0)))))
            res = pd.DataFrame(res)
            # per-cell recall@10 on perturbed cells (training-style: z minus the context mean)
            tgt = np.array([lpos.get(l, -1) for l in lp])
            ok = tgt >= 0
            S = unit(Z[:n_p][ok] - center) @ Pn.T
            r_cell = ranks_of(S, tgt[ok])
            off_in = float(np.linalg.norm(xl[q].mean(0) - base))
            cs, _ = collapse_stats(Z[q] - center, Pn)
            rows.append(dict(
                context=d, arm=arm, baseline=ref_name, n_pert_cells=n_p, n_perts=res.perturbation.nunique(),
                input_dist_median=float(np.median(d_in)), input_offset=off_in,
                input_offset_over_dist=off_in / float(np.median(d_in)),
                latent_norm_median=float(np.median(zn)), maxcos_median=float(np.median(mc)),
                maxcos_q10=float(np.quantile(mc, 0.1)),
                ctx_latent_offset=float(np.linalg.norm(Z[q].mean(0) - Z[refc].mean(0))),
                pbulk_median_rank_refctrl=float(res[res.vs == 'ref_ctrl']['rank'].median()),
                pbulk_top10_refctrl=float((res[res.vs == 'ref_ctrl']['rank'] <= 10).mean()),
                pbulk_median_rank_ownctrl=float(res[res.vs == 'own_ctrl']['rank'].median()),
                pbulk_top10_ownctrl=float((res[res.vs == 'own_ctrl']['rank'] <= 10).mean()),
                pert_delta_norm_median=float(res[res.vs == 'own_ctrl'].delta_norm.median()),
                cell_recall10=float((r_cell <= 10).mean()), cell_median_rank=float(np.median(r_cell)),
                buf_matches_ref_mean=float(np.abs(buf[cols] - base).max()), minutes=(time.time() - t0) / 60,
                **cs))
            prow.append(res)
            log.info('%s', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    out = Path(a.out_dir) / 'parts'
    out.mkdir(exist_ok=True)
    tag = 'all' if a.task is None else f'{a.task:02d}'
    pd.DataFrame(rows).to_csv(out / f'train_arms_{tag}.csv', index=False)
    pd.concat(prow).to_csv(out / f'train_perts_{tag}.csv.gz', index=False)
    json.dump(dict(vars(a), model_facts=facts), open(out / f'train_run_{tag}.json', 'w'), indent=1)


# ------------------------------------------------------------------------------------- ALD
def stage_ald(a):
    import anndata as ad
    rng = np.random.default_rng(0)
    model, facts = load_model(a.model_dir)
    labels, Pn = load_bank(a.bank)
    lpos = {l: i for i, l in enumerate(labels)}
    npc = np.array([lpos[g] for g in NPC if g in lpos])
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    genes_q = np.asarray(q.var_names).astype(str)
    model_genes = set(np.asarray(model.adata.var_names).astype(str))
    cols = np.array([i for i, g in enumerate(genes_q) if g in model_genes])
    pd.Series(genes_q[cols]).to_csv(Path(a.out_dir) / 'input_genes.txt', index=False, header=False)
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    cond = q.obs[a.condition_col].astype(str).to_numpy()
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    keep = np.isin(ct, a.celltypes.split(','))
    xl_all = log1p_dense(q.X, cols)
    normal = cond == a.control_condition
    nor_donors = sorted(set(donor[normal]))
    strata = {'pooled': ('ct', lambda m: normal & m), 'self_ct': ('ct', lambda m: m),
              'self_ctcond': ('ctcond', lambda m: m), 'self_ctdonor': ('ctdonor', lambda m: m)}
    for nd in nor_donors:
        strata[f'ref_{nd}'] = ('ct', lambda m, nd=nd: (donor == nd) & m)
    key = {'ct': ct, 'ctcond': np.char.add(np.char.add(ct.astype(str), '|'), cond.astype(str)),
           'ctdonor': np.char.add(np.char.add(ct.astype(str), '|'), donor.astype(str))}
    groups = sorted({(d, t) for d, t in zip(donor[keep], ct[keep])})
    arm_names = list(strata)
    if a.task is not None:                                     # one SLURM array task = one arm
        strata = {arm_names[a.task]: strata[arm_names[a.task]]}
    rows, profiles, cells = [], {}, []
    ref_saved = None
    if a.saved_latents and 'pooled' in strata:
        ref_saved = np.load(a.saved_latents).astype(np.float32)
    for arm, (by, refsel) in strata.items():
        t0 = time.time()
        Z = np.full((q.n_obs, Pn.shape[1]), np.nan, np.float32)
        base_of = np.full((q.n_obs, len(cols)), np.nan, np.float32)
        for s in sorted(set(key[by][keep])):
            m = (key[by] == s) & keep
            r = refsel(m)
            if r.sum() < a.min_ref:
                log.warning('%s %s: %d reference cells < %d, stratum skipped', arm, s, r.sum(), a.min_ref)
                continue
            # one call per stratum: the stratum's cells plus (if outside it) its reference cells
            call = m | r
            z, _ = encode(model, q.X[call], genes_q, r[call], a)
            Z[m] = z[m[call]]
            base_of[m] = xl_all[r].mean(0)
        done = ~np.isnan(Z[:, 0])
        if a.save_latents:
            (Path(a.out_dir) / 'parts').mkdir(exist_ok=True)
            np.save(Path(a.out_dir) / 'parts' / f'z_{arm}.npy', Z.astype(np.float16))
            pd.DataFrame({'donor': donor, 'condition': cond, 'celltype': ct, 'done': done}).to_csv(
                Path(a.out_dir) / 'parts' / 'z_obs.csv.gz', index=False)
        if arm == 'pooled' and ref_saved is not None:
            dmax = float(np.nanmax(np.abs(Z[done] - ref_saved[done])))
            log.info('pooled arm vs saved readout latents: max |dz| = %.2e', dmax)
        center = Z[done].mean(0)                                   # whole-query latent centre (recipe)
        Zc = Z - center
        Sc = np.full((q.n_obs, len(npc)), np.nan, np.float32)
        Sc_all = np.full(q.n_obs, np.nan, np.float32)
        for i in range(0, q.n_obs, 4096):
            sl = slice(i, i + 4096)
            S = unit(np.nan_to_num(Zc[sl])) @ Pn.T
            Sc[sl] = S[:, npc]
            Sc_all[sl] = S.mean(1)
        npc_cell = Sc.mean(1) - Sc_all                             # per-cell nuclear-pore score
        d_in = np.linalg.norm(xl_all - base_of, axis=1)
        mc = np.full(q.n_obs, np.nan, np.float32)
        for i in range(0, q.n_obs, 4096):
            sl = slice(i, i + 4096)
            mc[sl] = (unit(np.nan_to_num(Zc[sl])) @ Pn.T).max(1)
        for d, t in groups:
            m = (donor == d) & (ct == t) & done
            cmp_ = normal & (ct == t) & (donor != d) & done          # the OTHER Normal cells
            if m.sum() < a.min_cells or cmp_.sum() < a.min_cells:
                continue
            delta = Z[m].mean(0) - Z[cmp_].mean(0)
            s = unit(delta) @ Pn.T
            rk = np.empty(len(s), int)
            rk[np.argsort(-s)] = np.arange(1, len(s) + 1)
            pct = 1 - (rk - 1) / (len(s) - 1)
            obs_ = pct[npc].mean()
            null = np.array([pct[rng.choice(len(s), len(npc), replace=False)].mean() for _ in range(a.n_null)])
            profiles[(arm, d, t)] = s
            rows.append(dict(arm=arm, donor=d, condition=cond[m][0], celltype=t, n_cells=int(m.sum()),
                             input_dist_median=float(np.median(d_in[m])),
                             input_offset=float(np.linalg.norm(xl_all[m].mean(0) - base_of[m].mean(0))),
                             latent_norm_median=float(np.median(np.linalg.norm(Zc[m], axis=1))),
                             maxcos_median=float(np.median(mc[m])), maxcos_q10=float(np.quantile(mc[m], 0.1)),
                             delta_norm=float(np.linalg.norm(delta)),
                             npc_pct=float(obs_), npc_p=float((1 + (null >= obs_).sum()) / (1 + a.n_null)),
                             npc_top_genes=','.join(labels[j] for j in npc[np.argsort(rk[npc])][:4]),
                             npc_cell_mean=float(npc_cell[m].mean()),
                             npc_cell_vs_normal=float(npc_cell[m].mean() - npc_cell[cmp_].mean())))
        log.info('arm %-14s %d strata done in %.1f min', arm, len(set(key[by][keep])), (time.time() - t0) / 60)
    out = Path(a.out_dir) / 'parts'
    out.mkdir(exist_ok=True)
    tag = 'all' if a.task is None else list(strata)[0]
    res = pd.DataFrame(rows)
    res.to_csv(out / f'ald_groups_{tag}.csv', index=False)
    np.savez_compressed(out / f'ald_profiles_{tag}.npz', keys=np.array(['@'.join(k) for k in profiles]),
                        S=np.stack(list(profiles.values())).astype(np.float32))
    json.dump(dict(vars(a), model_facts=facts, arms=arm_names), open(out / f'ald_run_{tag}.json', 'w'), indent=1)
    log.info('\n%s', res.pivot_table(index=['donor', 'celltype'], columns='arm', values='npc_pct').round(2))


def stage_merge(a):
    """Concatenate the per-task tables; Spearman between arms of each group's delta profile."""
    from scipy.stats import spearmanr
    out = Path(a.out_dir)
    parts = out / 'parts'
    for pre, ext in (('train_arms', 'csv'), ('train_perts', 'csv.gz'), ('ald_groups', 'csv')):
        fs = sorted(parts.glob(f'{pre}_*.{ext}'))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs]).to_csv(out / f'{pre}.{ext}', index=False)
            log.info('%s: %d parts', pre, len(fs))
    prof = {}
    for f in sorted(parts.glob('ald_profiles_*.npz')):
        z = np.load(f)
        prof.update({tuple(str(k).split('@')): v for k, v in zip(z['keys'], z['S'])})
    arms = sorted({k[0] for k in prof})
    rows = []
    for (arm, d, t), v in prof.items():
        for a2 in arms:
            if a2 > arm and (a2, d, t) in prof:
                rows.append(dict(donor=d, celltype=t, arm1=arm, arm2=a2,
                                 spearman=spearmanr(v, prof[(a2, d, t)]).statistic))
    if rows:
        pd.DataFrame(rows).to_csv(out / 'ald_arm_spearman.csv', index=False)
        log.info('arm Spearman (median over groups):\n%s',
                 pd.DataFrame(rows).groupby(['arm1', 'arm2']).spearman.median().round(3).to_string())


# ------------------------------------------------------------------------------------- scoring
# arm -> (encoding arm whose latents are scored, latent centre). 'query' = whole-query mean of z (the
# readout recipe); 'group' = each donor x cell type's own mean (predict(query_mean) per group call).
SCORE_ARMS = {'pooled_q': ('pooled', 'query'), 'ctmean_q': ('self_ct', 'query'),
              'group_group': ('self_ctdonor', 'group'), 'group_q': ('self_ctdonor', 'query'),
              'pooled_group': ('pooled', 'group')}
# Reactome pathways with an ALD paper behind them (Fig. 5b 'established' rows, PubMed-checked 2026-10-12)
ESTABLISHED = ['SUMOylation', 'Chromatin Modifying Enzymes', 'Mitochondrial Translation', 'Macroautophagy',
               'Amino Acids Regulate mTORC1', 'Proteasome Assembly', 'mRNA Splicing - Major Pathway',
               'Eukaryotic Translation Elongation', 'TNFR1-induced NF-kappa-B Signaling Pathway',
               'Signaling by TGFB Family Members']


def pct_of(v):
    """Tie-averaged percentile, 1 = top of the list."""
    from scipy.stats import rankdata
    return (rankdata(v) - 1) / (len(v) - 1)


def set_z(pct, members):
    """Mean percentile of a set vs random sets of the same size: analytic normal null (finite population,
    variance taken from the profile itself so heavy ties are handled)."""
    from scipy.stats import norm
    N, var = len(pct), pct.var()
    out = []
    for m in members:
        n = len(m)
        sd = np.sqrt(var / n * (N - n) / (N - 1))
        mu = pct[m].mean()
        z = (mu - pct.mean()) / sd
        out.append((mu, z, norm.sf(z)))
    return out


def stage_score(a):
    import expert.utils.readout as ro
    name = list(SCORE_ARMS)[a.task] if a.task is not None else a.score_arm
    enc_arm, centre = SCORE_ARMS[name]
    parts = Path(a.out_dir) / 'parts'
    obs = pd.read_csv(parts / 'z_obs.csv.gz')
    Z = np.load(parts / f'z_{enc_arm}.npy').astype(np.float32)
    labels, Pn = load_bank(a.bank)
    N = len(labels)
    done = obs.done.to_numpy() & np.isin(obs.celltype, a.celltypes.split(','))
    donor, ct, cond = (obs[c].astype(str).to_numpy().astype(str) for c in ('donor', 'celltype', 'condition'))
    gkey = (obs.donor.astype(str) + '|' + obs.celltype.astype(str)).to_numpy().astype(str)
    groups = sorted(set(gkey[done]))
    Zc = np.full_like(Z, np.nan)
    if centre == 'query':
        Zc[done] = Z[done] - Z[done].mean(0)
    else:
        for g in groups:
            m = (gkey == g) & done
            Zc[m] = Z[m] - Z[m].mean(0)
    # per group: top-k membership frequency (both tails), pseudobulk, collapse
    fl, fg, pb, crow = {}, {}, {}, []
    for g in groups:
        m = np.flatnonzero((gkey == g) & done)
        cl, cg = np.zeros(N), np.zeros(N)
        for i in range(0, len(m), 4096):
            S = unit(Zc[m[i:i + 4096]]) @ Pn.T
            np.add.at(cl, np.argpartition(-S, a.top_k, axis=1)[:, :a.top_k].ravel(), 1)
            np.add.at(cg, np.argpartition(S, a.top_k, axis=1)[:, :a.top_k].ravel(), 1)
        fl[g], fg[g] = cl / len(m), cg / len(m)
        pb[g] = Zc[m].mean(0)
        cs, _ = collapse_stats(Zc[m], Pn)
        crow.append(dict(arm=name, group=g, n_cells=len(m), **cs))
    # profiles: each donor vs the OTHER Normal donors of its cell type
    readouts = {'cellfreq': {}, 'pbulk': {}}
    for g in groups:
        d, t = g.split('|')
        others = [h for h in groups if h.endswith('|' + t) and h != g
                  and cond[np.flatnonzero(gkey == h)[0]] == a.control_condition]
        if not others:
            continue
        readouts['cellfreq'][g] = (fl[g] - np.mean([fl[h] for h in others], 0),
                                   fg[g] - np.mean([fg[h] for h in others], 0))
        if centre == 'query':
            delta = pb[g] - np.mean([pb[h] for h in others], 0)
            s = unit(delta) @ Pn.T
            readouts['pbulk'][g] = (s, -s)
    # gene sets
    lpos = {l: i for i, l in enumerate(labels)}
    mods = ro.load_modules(a.modules, labels)
    sets = {('module', k): v for k, v in mods.items()}
    gmt = ro.load_gmt(a.reactome_gmt, vocab=labels, min_size=10, max_size=500)
    sets.update({('reactome', k): v for k, v in gmt.items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in lpos]
    keys = list(sets)
    members = [np.array([lpos[x] for x in sets[k] if x in lpos]) for k in keys]
    rows, prof = [], {}
    for rd, R in readouts.items():
        for g, (vl, vg) in R.items():
            prof[f'{rd}@{g}'] = vl
            for tail, v in (('loss', vl), ('gain', vg)):
                for (fam, sname), (mu, z, p) in zip(keys, set_z(pct_of(v), members)):
                    rows.append((name, rd, g, cond[np.flatnonzero(gkey == g)[0]], tail, fam, sname,
                                 len(sets[(fam, sname)]), mu, z, p))
    t = pd.DataFrame(rows, columns=['arm', 'readout', 'group', 'condition', 'tail', 'family', 'set', 'n',
                                    'mean_pct', 'z', 'p'])
    t.to_csv(parts / f'score_tests_{name}.csv.gz', index=False)
    pd.DataFrame(crow).to_csv(parts / f'score_collapse_{name}.csv', index=False)
    np.savez_compressed(parts / f'score_profiles_{name}.npz', keys=np.array(list(prof)),
                        V=np.stack(list(prof.values())).astype(np.float32))
    log.info('%s: %d groups, %d tests; collapse\n%s', name, len(groups), len(t),
             pd.DataFrame(crow).drop(columns='arm').round(3).to_string(index=False))


def stage_scoremerge(a):
    """One scorecard row per arm x readout."""
    from scipy.stats import spearmanr
    out = Path(a.out_dir)
    parts = out / 'parts'
    t = pd.concat([pd.read_csv(f) for f in sorted(parts.glob('score_tests_*.csv.gz'))], ignore_index=True)
    c = pd.concat([pd.read_csv(f) for f in sorted(parts.glob('score_collapse_*.csv'))], ignore_index=True)
    t.to_csv(out / 'score_tests.csv.gz', index=False)
    c.to_csv(out / 'score_collapse.csv', index=False)
    t['ct'] = t.group.str.split('|').str[1]
    t['donor'] = t.group.str.split('|').str[0]
    t['dis'] = t.condition != a.control_condition
    t['sig'] = t.p < a.alpha
    card = []
    for (arm, rd), x in t.groupby(['arm', 'readout']):
        dis, sg = x.dis.to_numpy(), x.sig.to_numpy()
        ald = ((x.family == 'module') & ~x.set.str.startswith('ctrl_')).to_numpy()
        ctl = ((x.family == 'module') & x.set.str.startswith('ctrl_')).to_numpy()
        est = ((x.family == 'reactome') & x.set.isin(ESTABLISHED)).to_numpy()
        rea = (x.family == 'reactome').to_numpy()
        row = dict(arm=arm, readout=rd,
                   ald_module_rate_disease=sg[ald & dis].mean(), ctrl_panel_rate_disease=sg[ctl & dis].mean(),
                   ald_module_rate_normal=sg[ald & ~dis].mean(),
                   established_rate_disease=sg[est & dis].mean(), established_rate_normal=sg[est & ~dis].mean(),
                   reactome_rate_disease=sg[rea & dis].mean(), reactome_rate_normal=sg[rea & ~dis].mean())
        # replication: same set, same tail, significant in BOTH donors of a condition within a cell type
        for cnd in ('AC', 'AH'):
            y = x[rea & sg & (x.condition == cnd).to_numpy()]
            rep = y.groupby(['ct', 'tail', 'set']).donor.nunique()
            row[f'reactome_replicated_{cnd}'] = int((rep == 2).sum())
        y = x[rea & sg & dis]
        cross = y.assign(k=y.donor.str[-1]).groupby(['ct', 'tail', 'set', 'k']).condition.nunique()
        row['reactome_replicated_cross_AC_AH'] = int((cross == 2).sum())
        npc = x[(x.family == 'npc') & (x.tail == 'loss')].set_index('group')
        for g in npc.index:
            row[f'npc_{g}'] = npc.loc[g, 'mean_pct']
        z = np.load(parts / f'score_profiles_{arm}.npz')
        P = {str(k).split('@', 1)[1]: v for k, v in zip(z['keys'], z['V']) if str(k).startswith(rd + '@')}
        for cnd in ('AC', 'AH'):
            r = [spearmanr(P[f'{cnd}1|{c}'], P[f'{cnd}2|{c}']).statistic for c in a.celltypes.split(',')
                 if f'{cnd}1|{c}' in P and f'{cnd}2|{c}' in P]
            row[f'donor_rho_{cnd}'] = float(np.median(r)) if r else np.nan
        r = [spearmanr(P[f'AC{i}|{c}'], P[f'AH{j}|{c}']).statistic for c in a.celltypes.split(',')
             for i in (1, 2) for j in (1, 2) if f'AC{i}|{c}' in P and f'AH{j}|{c}' in P]
        row['donor_rho_cross_AC_AH'] = float(np.median(r)) if r else np.nan
        card.append(row)
    card = pd.DataFrame(card)
    cc = c.groupby('arm')[['shared_offset', 'mean_pair_cos', 'top1_eff_n', 'top1_top5_share']].median()
    card = card.merge(cc, left_on='arm', right_index=True)
    card.to_csv(out / 'scorecard.csv', index=False)
    pd.set_option('display.width', 250)
    log.info('scorecard\n%s', card.drop(columns=[k for k in card if k.startswith('npc_')]).round(3).T.to_string())


# ------------------------------------------------------------------------------------- pairwise design
# Every contrast is ONE donor vs ONE donor within a cell type, so signal and null pairs are built the same
# way. Null = same-condition pairs (AC1-AC2, AH1-AH2, Nor1-Nor2); signal = disease vs Normal, and AC vs AH.
# Readouts per ordered pair (A query, B reference):
#   pbulk      cos(mean z_A - mean z_B, P)                        (latent centre cancels)
#   cellmean   mean_A cos(z - mean z_B, P) - mean_B cos(z - mean z_B, P)   (control in z + rank adjust)
#   cellfreq   same with top-k membership frequency instead of the mean cosine
PAIR_ENC = ['pooled', 'self_ct', 'self_ctdonor']


def pair_kind(a_, b_):
    ca, cb = a_[:2], b_[:2]
    if ca == cb:
        return 'null_same_condition'
    if 'No' in (ca, cb):
        return f"{ca if ca != 'No' else cb}_vs_Normal" if cb == 'No' else f'Normal_vs_{ca}'
    return f'{ca}_vs_{cb}'


def stage_pairs(a):
    import expert.utils.readout as ro
    enc = PAIR_ENC[a.task] if a.task is not None else a.score_arm
    parts = Path(a.out_dir) / 'parts'
    obs = pd.read_csv(parts / 'z_obs.csv.gz')
    Z = np.load(parts / f'z_{enc}.npy').astype(np.float32)
    labels, Pn = load_bank(a.bank)
    N = len(labels)
    done = obs.done.to_numpy() & np.isin(obs.celltype, a.celltypes.split(','))
    donor = obs.donor.astype(str).to_numpy().astype(str)
    ct = obs.celltype.astype(str).to_numpy().astype(str)
    lpos = {l: i for i, l in enumerate(labels)}
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in lpos]
    keys = list(sets)
    members = [np.array([lpos[x] for x in sets[k] if x in lpos]) for k in keys]
    rows, hubs = [], np.zeros(N)
    for t in a.celltypes.split(','):
        ds = [d for d in sorted(set(donor[done & (ct == t)])) if (done & (ct == t) & (donor == d)).sum() >= a.min_cells]
        idx = {d: np.flatnonzero(done & (ct == t) & (donor == d)) for d in ds}
        mz = {d: Z[idx[d]].mean(0) for d in ds}
        for b in ds:                                            # reference donor b: centre z on it
            stat = {}
            for d in ds:
                ml, cl = np.zeros(N), np.zeros(N)
                m = idx[d]
                for i in range(0, len(m), 4096):
                    S = unit(Z[m[i:i + 4096]] - mz[b]) @ Pn.T
                    ml += S.sum(0)
                    np.add.at(cl, np.argpartition(-S, a.top_k, axis=1)[:, :a.top_k].ravel(), 1)
                    if d == b and b.startswith('Nor'):
                        np.add.at(hubs, S.argmax(1), 1)
                stat[d] = (ml / len(m), cl / len(m))
            for d in ds:
                if d == b:
                    continue
                prof = {'pbulk': unit(mz[d] - mz[b]) @ Pn.T,
                        'cellmean': stat[d][0] - stat[b][0],
                        'cellfreq': stat[d][1] - stat[b][1]}
                for rd, v in prof.items():
                    for tail, w in (('loss', v), ('gain', -v)):
                        for (fam, sname), (mu, z, p) in zip(keys, set_z(pct_of(w), members)):
                            rows.append((enc, rd, t, d, b, pair_kind(d, b), tail, fam, sname, mu, z))
        log.info('%s %s: %d donors', enc, t, len(ds))
    out = pd.DataFrame(rows, columns=['enc', 'readout', 'celltype', 'query', 'ref', 'kind', 'tail', 'family',
                                      'set', 'mean_pct', 'z'])
    out.to_csv(parts / f'pairs_{enc}.csv.gz', index=False)
    top = np.argsort(-hubs)[:25]
    pd.DataFrame({'enc': enc, 'prototype': [labels[j] for j in top],
                  'share_of_normal_cells': hubs[top] / hubs.sum()}).to_csv(parts / f'hubs_{enc}.csv', index=False)


def stage_pairsmerge(a):
    """Threshold each (enc, readout) so same-condition pairs hit at --alpha; report signal hit rates and
    consensus calls (a set, same tail, over threshold in ALL disease-vs-Normal pairs of a cell type)."""
    out = Path(a.out_dir)
    parts = out / 'parts'
    t = pd.concat([pd.read_csv(f) for f in sorted(parts.glob('pairs_*.csv.gz'))], ignore_index=True)
    t.to_csv(out / 'pairs_tests.csv.gz', index=False)
    hubs = pd.concat([pd.read_csv(f) for f in sorted(parts.glob('hubs_*.csv'))], ignore_index=True)
    hubs.to_csv(out / 'pairs_hubs.csv', index=False)
    card, calls = [], []
    for (enc, rd), x in t.groupby(['enc', 'readout']):
        thr = x.loc[x.kind == 'null_same_condition', 'z'].quantile(1 - a.alpha)
        x = x.assign(hit=x.z > thr)
        row = dict(enc=enc, readout=rd, z_threshold=thr)
        ald = (x.family == 'module') & ~x.set.str.startswith('ctrl_')
        ctl = (x.family == 'module') & x.set.str.startswith('ctrl_')
        est = (x.family == 'reactome') & x.set.isin(ESTABLISHED)
        for kind in ('null_same_condition', 'AC_vs_Normal', 'AH_vs_Normal', 'AC_vs_AH'):
            k = x.kind == kind
            row[f'hit_{kind}'] = x.hit[k].mean()
            row[f'ald_module_{kind}'] = x.hit[k & ald].mean()
            row[f'established_{kind}'] = x.hit[k & est].mean()
            row[f'ctrl_panel_{kind}'] = x.hit[k & ctl].mean()
        for cnd in ('AC', 'AH'):
            y = x[x.kind == f'{cnd}_vs_Normal']
            n_pairs = y.groupby('celltype')[['query', 'ref']].apply(lambda q: len(q.drop_duplicates()))
            c = y[y.hit].groupby(['celltype', 'tail', 'family', 'set']).size().reset_index(name='n_hit')
            c = c[c.n_hit == c.celltype.map(n_pairs)]
            row[f'consensus_{cnd}'] = len(c)
            row[f'consensus_{cnd}_ald_modules'] = int(((c.family == 'module') & ~c.set.str.startswith('ctrl_')).sum())
            row[f'consensus_{cnd}_ctrl_panels'] = int(((c.family == 'module') & c.set.str.startswith('ctrl_')).sum())
            calls.append(c.assign(enc=enc, readout=rd, condition=cnd))
            # same-shaped null: 4 pairs between MIXED partitions {cnd1, NorX} -> {cnd2, NorY}; half the pairs
            # are same-condition, so no consistent disease direction runs through them
            nul = []
            for q1, q2 in ((f'{cnd}1', 'Nor1'), (f'{cnd}1', 'Nor2')):
                r1, r2 = ({f'{cnd}1', f'{cnd}2', 'Nor1', 'Nor2'} - {q1}) - {q2}, None
                r1 = sorted(r1)
                pairs = {(q1, r1[0]), (q1, r1[1]), (q2, r1[0]), (q2, r1[1])}
                y = x[[(qq, rr) in pairs for qq, rr in zip(x['query'], x['ref'])]]
                np_ = y.groupby('celltype')[['query', 'ref']].apply(lambda q: len(q.drop_duplicates()))
                cc = y[y.hit].groupby(['celltype', 'tail', 'family', 'set']).size().reset_index(name='n_hit')
                nul.append(int((cc.n_hit == cc.celltype.map(np_)).sum()))
            row[f'consensus_{cnd}_mixed_null'] = float(np.mean(nul))
            npc = c[(c.family == 'npc')]
            row[f'consensus_{cnd}_npc'] = ','.join(f'{r.celltype}:{r.tail}' for r in npc.itertuples()) or '-'
        card.append(row)
    card = pd.DataFrame(card)
    card.to_csv(out / 'pairs_scorecard.csv', index=False)
    pd.concat(calls).to_csv(out / 'pairs_consensus_calls.csv', index=False)
    pd.set_option('display.width', 250)
    log.info('pairwise scorecard\n%s', card.round(3).T.to_string())
    log.info('hub prototypes (share of Normal cells, top-1)\n%s',
             hubs.groupby('enc').head(8).round(3).to_string(index=False))


# ------------------------------------------------------------------------------------- gene-level zoom
def stage_zoom(a):
    """Gene-level consensus for one protocol (encoding --score-arm, pseudobulk, each disease donor vs each
    Normal donor within a cell type). consensus rank = WORST rank over the 4 pairs; null = the same statistic
    over mixed partitions {cnd1, NorX} -> {cnd2, NorY}. Annotated with own-transcript DE and input-gene status."""
    parts = Path(a.out_dir) / 'parts'
    enc = a.score_arm
    obs = pd.read_csv(parts / 'z_obs.csv.gz')
    Z = np.load(parts / f'z_{enc}.npy').astype(np.float32)
    labels, Pn = load_bank(a.bank)
    N = len(labels)
    done = obs.done.to_numpy() & np.isin(obs.celltype, a.celltypes.split(','))
    donor = obs.donor.astype(str).to_numpy().astype(str)
    ct = obs.celltype.astype(str).to_numpy().astype(str)
    inputs = set(pd.read_csv(Path(a.out_dir) / 'input_genes.txt', header=None)[0].astype(str))
    deg = pd.read_csv(a.deg).set_index(['group', 'gene'])

    def rank_of(v):
        r = np.empty(N, int)
        r[np.argsort(-v)] = np.arange(1, N + 1)
        return r

    rows, nulls = [], []
    for t in a.celltypes.split(','):
        ds = [d for d in sorted(set(donor[done & (ct == t)])) if (done & (ct == t) & (donor == d)).sum() >= a.min_cells]
        mz = {d: Z[done & (ct == t) & (donor == d)].mean(0) for d in ds}
        S = {(q, r): unit(mz[q] - mz[r]) @ Pn.T for q in ds for r in ds if q != r}
        for cnd in ('AC', 'AH'):
            qs = [d for d in ds if d.startswith(cnd)]
            rs = [d for d in ds if d.startswith('Nor')]
            if len(qs) < 2 or not rs:
                continue
            pairs = [(q, r) for q in qs for r in rs]
            mixed = []
            for q1, q2 in ((qs[0], rs[0]), (qs[0], rs[-1])):
                rr = sorted(set(qs + rs) - {q1, q2})
                mixed.append([(q, r) for q in (q1, q2) for r in rr])
            for tail, sg in (('loss', 1), ('gain', -1)):
                R = np.stack([rank_of(sg * S[p]) for p in pairs])
                worst = R.max(0)
                for mp in mixed:
                    nulls.append(dict(celltype=t, condition=cnd, tail=tail,
                                      **{f'n_le_{k}': int((np.stack([rank_of(sg * S[p]) for p in mp]).max(0) <= k).sum())
                                         for k in (50, 100, 200, 500)}))
                real = {f'n_le_{k}': int((worst <= k).sum()) for k in (50, 100, 200, 500)}
                nulls.append(dict(celltype=t, condition=cnd, tail=tail, real=True, **real))
                for j in np.argsort(worst)[:a.top_n]:
                    g = labels[j]
                    dk = (f'{cnd}|{t}', g)
                    rows.append(dict(celltype=t, condition=cnd, tail=tail, gene=g, consensus_rank=int(worst[j]),
                                     **{f'rank_{q}>{r}': int(R[i, j]) for i, (q, r) in enumerate(pairs)},
                                     mean_cos=float(np.mean([sg * S[p][j] for p in pairs])),
                                     input_gene=g in inputs,
                                     own_logfc=deg.logfc.get(dk, np.nan), own_padj=deg.pval_adj.get(dk, np.nan)))
        log.info('%s done', t)
    g = pd.DataFrame(rows)
    nl = pd.DataFrame(nulls)
    nl['real'] = nl.get('real', False).fillna(False)
    g.to_csv(Path(a.out_dir) / f'zoom_genes_{enc}.csv', index=False)
    nl.to_csv(Path(a.out_dir) / f'zoom_null_{enc}.csv', index=False)
    summ = nl.groupby(['condition', 'tail', 'celltype', 'real'])[['n_le_100', 'n_le_200']].mean().unstack('real')
    log.info('consensus genes (worst rank <= k): real vs mixed-partition null\n%s', summ.round(1).to_string())


# ------------------------------------------------------------------------------------- group vs rest
EXPORT_SETS = ['rna_export_trex', 'nuclear_pore', 'karyopherin', 'rna_decay_exosome', 'nuclear pore',
               'Transport of Mature Transcript to Cytoplasm',
               'Transport of Mature mRNA Derived From an Intron-Containing Transcript',
               'Nuclear Pore Complex (NPC) Disassembly', 'Rev-mediated Nuclear Export of HIV RNA']


def stage_rest(a):
    """Donor-vs-rest within a cell type. readouts: 'loo' = cos(mean z_d - mean z_{others}, P); 'z' = the same
    profile z-scored per prototype across the cell type's donors. Threshold: Normal donors' set z hit --alpha;
    consensus = both donors of a condition over threshold (same tail); null = cross-condition donor pairs."""
    import itertools
    import expert.utils.readout as ro
    enc = a.score_arm
    parts = Path(a.out_dir) / 'parts'
    obs = pd.read_csv(parts / 'z_obs.csv.gz')
    Z = np.load(parts / f'z_{enc}.npy').astype(np.float32)
    labels, Pn = load_bank(a.bank)
    N = len(labels)
    lpos = {l: i for i, l in enumerate(labels)}
    done = obs.done.to_numpy() & np.isin(obs.celltype, a.celltypes.split(','))
    donor = obs.donor.astype(str).to_numpy().astype(str)
    ct = obs.celltype.astype(str).to_numpy().astype(str)
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, labels).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=labels).items()})
    sets[('npc', 'nuclear pore')] = [g for g in NPC if g in lpos]
    keys = list(sets)
    members = [np.array([lpos[x] for x in sets[k] if x in lpos]) for k in keys]
    rows, gene_rows = [], []
    for t in a.celltypes.split(','):
        ds = [d for d in sorted(set(donor[done & (ct == t)])) if (done & (ct == t) & (donor == d)).sum() >= a.min_cells]
        mz = {d: Z[done & (ct == t) & (donor == d)].mean(0) for d in ds}
        loo = {d: unit(mz[d] - np.mean([mz[e] for e in ds if e != d], 0)) @ Pn.T for d in ds}
        M = np.stack([loo[d] for d in ds])
        zs = (M - M.mean(0)) / (M.std(0) + 1e-9)
        prof = {'loo': loo, 'z': {d: zs[i] for i, d in enumerate(ds)}}
        for rd, P_ in prof.items():
            for d, v in P_.items():
                for tail, w in (('loss', v), ('gain', -v)):
                    for (fam, sname), (mu, z, p) in zip(keys, set_z(pct_of(w), members)):
                        rows.append((enc, rd, t, d, tail, fam, sname, mu, z))
                    r = np.empty(N, int)
                    r[np.argsort(-w)] = np.arange(1, N + 1)
                    for j in np.argsort(r)[:a.top_n]:
                        gene_rows.append((enc, rd, t, d, tail, labels[j], int(r[j])))
        log.info('%s rest %s: %d donors', enc, t, len(ds))
    t = pd.DataFrame(rows, columns=['enc', 'readout', 'celltype', 'donor', 'tail', 'family', 'set', 'mean_pct', 'z'])
    gr = pd.DataFrame(gene_rows, columns=['enc', 'readout', 'celltype', 'donor', 'tail', 'gene', 'rank'])
    t.to_csv(Path(a.out_dir) / f'rest_tests_{enc}.csv.gz', index=False)
    card, calls = [], []
    for rd, x in t.groupby('readout'):
        thr = x.loc[x.donor.str.startswith('Nor'), 'z'].quantile(1 - a.alpha)
        x = x.assign(hit=x.z > thr)
        h = x[x.hit]
        key = ['celltype', 'tail', 'family', 'set']
        def consensus(d1, d2):
            a1 = set(map(tuple, h[h.donor == d1][key].values))
            a2 = set(map(tuple, h[h.donor == d2][key].values))
            return a1 & a2
        row = dict(enc=enc, readout=rd, z_threshold=thr)
        for cnd in ('AC', 'AH'):
            c = consensus(f'{cnd}1', f'{cnd}2')
            row[f'consensus_{cnd}'] = len(c)
            calls += [dict(enc=enc, readout=rd, condition=cnd, celltype=k[0], tail=k[1], family=k[2], set=k[3]) for k in c]
        row['consensus_null_cross'] = float(np.mean([len(consensus(p, q)) for p, q in
                                                     (('AC1', 'AH1'), ('AC1', 'AH2'), ('AC2', 'AH1'), ('AC2', 'AH2'))]))
        row['consensus_null_disease_normal'] = float(np.mean([len(consensus(p, q)) for p, q in
                                                    itertools.product(('AC1', 'AC2', 'AH1', 'AH2'), ('Nor1', 'Nor2'))]))
        row['hit_rate_disease'] = x.hit[~x.donor.str.startswith('Nor')].mean()
        card.append(row)
        ex = x[x.set.isin(EXPORT_SETS) & (x['tail'] == 'loss')].pivot_table(index=['set', 'donor'], columns='celltype',
                                                                             values='z')
        ex.to_csv(Path(a.out_dir) / f'rest_export_{enc}_{rd}.csv')
        log.info('%s %s export-family LOSS z (threshold %.2f)\n%s', enc, rd, thr, ex.round(1).to_string())
    card = pd.DataFrame(card)
    card.to_csv(Path(a.out_dir) / f'rest_scorecard_{enc}.csv', index=False)
    pd.DataFrame(calls).to_csv(Path(a.out_dir) / f'rest_calls_{enc}.csv', index=False)
    # gene consensus: worst rank over the two donors of a condition; null = cross-condition donor pairs
    gene = []
    for (rd, t_, tail), x in gr.groupby(['readout', 'celltype', 'tail']):
        rk = x.pivot_table(index='gene', columns='donor', values='rank')
        for cnd in ('AC', 'AH'):
            cols = [f'{cnd}1', f'{cnd}2']
            if not set(cols) <= set(rk.columns):
                continue
            w = rk[cols].max(1).dropna().sort_values()
            for g_, v in w.head(a.top_n).items():
                gene.append(dict(enc=enc, readout=rd, celltype=t_, tail=tail, condition=cnd, gene=g_, worst_rank=int(v)))
    pd.DataFrame(gene).to_csv(Path(a.out_dir) / f'rest_genes_{enc}.csv', index=False)
    pd.set_option('display.width', 250)
    log.info('rest scorecard\n%s', card.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['train', 'ald', 'merge', 'score', 'scoremerge', 'pairs', 'pairsmerge', 'zoom', 'rest'])
    ap.add_argument('--model-dir')
    ap.add_argument('--bank', help='readout fullbank/bank.npz (P, labels, trained)')
    ap.add_argument('--task', type=int, default=None,
                    help='SLURM array index: one context (train, 0-7) or one arm (ald, 0-5); unset = all')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--train-h5ad', help='database_perturbed_200.h5ad (train stage)')
    ap.add_argument('--ctrl-h5ad', help='database_ctrl.h5ad (train stage)')
    ap.add_argument('--input-genes', default='', help='restrict input statistics to these genes (the ALD '
                                                       'stage writes input_genes.txt) so both stages compare')
    ap.add_argument('--line-contexts', default='ReplogleWeissman2022_K562_essential,ReplogleWeissman2025_HepG2,'
                                               'ReplogleWeissman2022_rpe1,ReplogleWeissman2025_Jurkat')
    ap.add_argument('--per-label', type=int, default=60, help='perturbed cells per perturbation (train)')
    ap.add_argument('--max-pert-cells', type=int, default=8000)
    ap.add_argument('--max-ctrl-cells', type=int, default=3000)
    ap.add_argument('--query', help='ALD readout query h5ad (ald stage)')
    ap.add_argument('--saved-latents', default='', help='readout latents_true/z_raw_celltype.npy (pooled-arm check)')
    ap.add_argument('--condition-col', default='Condition')
    ap.add_argument('--control-condition', default='Normal')
    ap.add_argument('--celltype-col', default='CellTypeSR')
    ap.add_argument('--donor-col', default='sample')
    ap.add_argument('--celltypes', default='HSC,Hepatocyte,Kupffer Cell,LSEC,T cell')
    ap.add_argument('--min-ref', type=int, default=30)
    ap.add_argument('--min-cells', type=int, default=50)
    ap.add_argument('--n-null', type=int, default=1000)
    ap.add_argument('--save-latents', action='store_true', help='ald: write parts/z_<arm>.npy (float16)')
    ap.add_argument('--score-arm', default='group_group', help='score: arm when --task is unset')
    ap.add_argument('--modules', help='score: readout_panels.tsv (ALD modules + ctrl_* panels)')
    ap.add_argument('--reactome-gmt', help='score: Reactome GMT')
    ap.add_argument('--deg', help='zoom: readout_deg/deg.csv.gz (own-transcript DE)')
    ap.add_argument('--top-n', type=int, default=100, help='zoom: genes kept per condition x cell type x tail')
    ap.add_argument('--top-k', type=int, default=20, help='score: per-cell top-k membership')
    ap.add_argument('--alpha', type=float, default=0.01, help='scoremerge: per-test significance')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--n-cores', type=int, default=4)
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    {'train': stage_train, 'ald': stage_ald, 'merge': stage_merge, 'score': stage_score,
     'scoremerge': stage_scoremerge, 'pairs': stage_pairs, 'pairsmerge': stage_pairsmerge,
     'zoom': stage_zoom, 'rest': stage_rest}[a.stage](a)
