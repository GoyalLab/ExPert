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

"""Fig. 6 (v2): primary mitochondrial disease (PMD) PBMC readout following the collaborator protocol
(leigh_pmd_repro_2026-10-15/leigh_v010/README_PMD_figures.md), extended for the figure.

Groups (revised 2026-10-05): PMD = PT43 BCS1L (CIII), PT46 NDUFA10 (CI), PT54 NDUFS8 (CI), PT67 PC (pyruvate
carboxylase, re-sequenced after ExPert flagged it); non-PMD = PT38, PT50, PT55 (undiagnosed) + TBCK twins PT41/PT42.
No donor is healthy, so every readout is a contrast between patients.

stage encode  ONE predict() call on single cells, center_by = cell type, --mask genes dropped (default MT-*), the
              non-PMD donors' cells = each cell type's encoder input baseline (collaborator 'patient' pass), explicit
              zero latent centre; raw z saved.
stage encode_pbulk  the Fig. 5 input instead: one pseudobulk per donor x cell type, ONE predict() call, center_by cell
              type, non-PMD pools as the baseline; written in the single-cell layout (each cell = its pseudobulk latent).
stage score   per donor x cell type (>= --min-cells cells) latent SHIFTS, three references:
                ctrl   PMD vs pooled non-PMD cells; a non-PMD donor vs the pooled other non-PMD (leave-one-out)
                tbck   every donor vs the pooled TBCK cells (a twin vs the other twin)
                free   label-free: donor centroid vs equal-weight mean of every other donor's centroid
              each shift scored by cosine against the trained (8,964) and full (19,895, zero-shot fills) banks:
                complex scores CI/CIII/CIV/CV = 1 - median percentile rank of the complex genes (0.5 = chance),
                oxphos = their mean, isr_on = mean z over EIF2B1-5 + PPP1R15B, set z for the pathway rows,
                own lesion-gene rank (NDUFS8 / NDUFA10 / BCS1L / PC)
              readout: 'cosine' and 'deflate' (the predict() default: calibrated_readout on each shift at its
              effective depth n1 n2 / (n1 + n2), merged order transplanted onto the cosine scale).
              tests: exact donor-label permutation, PMD vs the reference donors scored the same way.
              clustering: label-free shifts, cosine distance, average linkage (latent and prototype-score space).
--cases / --exclude / --tag rerun score + genes on another case set, e.g. the ETC patients only
              (--cases PT43,PT46,PT54 --exclude PT67 --tag etc --de-rest etc:etc_vs_nonpmd --de-tbck orig:leigh_vs_tbck).
stage genes   group profiles PMD vs non-PMD / PMD vs TBCK per cell type, both banks, with own-transcript DE.
stage ora     Fig. 5 top-k recipe: per-donor top-200 per tail, cases' union per cell type, hypergeometric ORA vs the
              bank + exact donor-label permutation of the union hits (cosine and deflate, both banks).
stage cells   per-cell OXPHOS score for the UMAP: cosine of (z - label-free cell-type centre: equal-weight mean of donor centroids) with the
              OXPHOS direction (normalised mean of the unit CI/CIII/CIV/CV knockout prototypes), kNN-smoothed in z;
              UMAP of all cells and of monocytes.

    python paper/scripts/fig6/fig6_pmd.py --stage encode --query <h5ad> --model-dir <model> --out-dir <d>
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

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_pmd')
log.setLevel(logging.INFO)

PMD = ['PT43', 'PT46', 'PT54', 'PT67']
CTRL = ['PT38', 'PT50', 'PT55', 'PT41', 'PT42']
TBCK = ['PT41', 'PT42']
LESION = {'PT43': 'BCS1L', 'PT46': 'NDUFA10', 'PT54': 'NDUFS8', 'PT67': 'PC'}
CX = ['complex_i', 'complex_iii', 'complex_iv', 'complex_v']
ISR = 'EIF2B1 EIF2B2 EIF2B3 EIF2B4 EIF2B5 PPP1R15B'.split()


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)


# ------------------------------------------------------------------------------------------- encode
def stage_encode(a):
    """ONE predict() call: every cell is its own row, obs['perturbation'] = 'control' for the non-PMD donors' cells,
    center_by = cell type (each cell type's input centred on its own non-PMD cells -- the per-cell-type loop in a
    single pass), explicit zero latent centre, cosine readout, no hub correction; raw z_shared saved."""
    import anndata as ad
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(len(v), bool)
    for p in a.mask.split(','):                               # MT-* = collaborator; + metallothioneins = Fig. 4/5
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    if a.stim_col:                                            # cultured cells: group = cell type x stim
        ct = np.char.add(np.char.add(ct.astype(str), '|'), q.obs[a.stim_col].astype(str).to_numpy().astype(str))
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    x = ad.AnnData(X=q.X[:, ~drop], var=pd.DataFrame(index=v[~drop]),
                   obs=pd.DataFrame({'celltype': ct, 'perturbation': np.where(np.isin(donor, CTRL), 'control', 'query')},
                                    index=[f'c{i}' for i in range(q.n_obs)]))
    n_shared = int(model.module.align_ds_mean.shape[1])
    o = model.predict(x, return_latents=True, obs_only=True, soft_predictions=False, progress_bar=False, head='A',
                      center_by='celltype', center_mode=np.zeros(n_shared, np.float32), control_label='control',
                      readout='cosine', hub_correction=False, batch_size=a.batch_size, n_cores=a.n_cores)
    Z = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)
    np.save(out / 'z_cells.npy', Z)
    pd.DataFrame({'donor': donor, 'celltype': ct}).to_csv(out / 'obs_cells.csv.gz', index=False)
    json.dump(dict(model_dir=a.model_dir, query=a.query, n_dropped=int(drop.sum()), baseline_donors=CTRL,
                   predict="one call, center_by='celltype', control = non-PMD cells"),
              open(out / 'encode.json', 'w'), indent=1)
    log.info('%d cells encoded in one predict() call', len(Z))


def stage_encode_pbulk(a):
    """The Fig. 5 input (fig5_predict.py encode): ONE predict() call on one pseudobulk per donor x cell type (mean raw
    counts, n_cells = group size), center_by cell type, the non-PMD donors' cells as --n-rep random pools per donor
    labelled 'control' (input baseline only, never scored). Written in the single-cell layout: every cell carries its
    donor x cell type latent, so every downstream centroid / cell-weighted pool is the pseudobulk latent exactly."""
    import anndata as ad
    from expert import ExPert
    from expert.utils.constants import MODULE_KEYS
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ExPert.load(a.model_dir)
    q = ad.read_h5ad(a.query)
    q.X = q.X.tocsr() if sp.issparse(q.X) else sp.csr_matrix(q.X)
    v = pd.Index(q.var_names).astype(str)
    drop = np.zeros(len(v), bool)
    for p in a.mask.split(','):
        drop |= np.asarray(v.str.startswith(p[:-1]) if p.endswith('*') else (v == p), bool)
    q = q[:, ~drop].copy()
    ct = q.obs[a.celltype_col].astype(str).to_numpy()
    donor = q.obs[a.donor_col].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    X, meta = [], []
    for t in sorted(set(ct)):
        for d in sorted(set(donor[ct == t])):
            idx = np.flatnonzero((ct == t) & (donor == d))
            X.append(np.asarray(q.X[idx].mean(0)).ravel())
            meta.append(dict(role='group', donor=d, celltype=t, n_cells=len(idx)))
            if d in CTRL:                                      # input baseline only
                for chunk in np.array_split(rng.permutation(idx), min(a.n_rep, len(idx))):
                    X.append(np.asarray(q.X[chunk].mean(0)).ravel())
                    meta.append(dict(role='baseline', donor=d, celltype=t, n_cells=len(chunk)))
    pobs = pd.DataFrame(meta)
    pobs.index = [f'r{i}' for i in range(len(pobs))]
    pobs['perturbation'] = np.where(pobs.role == 'baseline', 'control', 'query')
    pb = ad.AnnData(X=sp.csr_matrix(np.stack(X).astype(np.float32)), obs=pobs,
                    var=pd.DataFrame(index=pd.Index(q.var_names.astype(str))))
    nb = pobs[pobs.role == 'baseline'].groupby('celltype').size()
    log.info('baseline rows per cell type: %s', nb.to_dict())
    n_shared = int(model.module.align_ds_mean.shape[1])
    o = model.predict(pb, return_latents=True, obs_only=True, progress_bar=False, head='A', center_by='celltype',
                      center_mode=np.zeros(n_shared, np.float32), control_label='control', n_cells_key='n_cells',
                      readout='cosine', hub_correction=False)
    g = (pobs.role == 'group').to_numpy()
    Zg = np.asarray(o[MODULE_KEYS.Z_SHARED_KEY], np.float32)[g]
    grp = pobs[g].reset_index(drop=True)
    row = {(d, t): i for i, (d, t) in enumerate(zip(grp.donor, grp.celltype))}
    Z = Zg[[row[(d, t)] for d, t in zip(donor, ct)]]
    np.save(out / 'z_cells.npy', Z)
    pd.DataFrame({'donor': donor, 'celltype': ct}).to_csv(out / 'obs_cells.csv.gz', index=False)
    grp.to_csv(out / 'obs_pbulk.csv', index=False)
    np.save(out / 'z_pbulk.npy', Zg)
    json.dump(dict(model_dir=a.model_dir, query=a.query, unit='pseudobulk donor x cell type', n_rep=a.n_rep,
                   n_dropped=int(drop.sum()), baseline_donors=CTRL), open(out / 'encode.json', 'w'), indent=1)
    log.info('%d pseudobulks encoded', len(grp))


# ------------------------------------------------------------------------------------------- shifts
def shifts(Z, obs, min_cells):
    """{(ref, donor, celltype): shift vector} for ref in ctrl / tbck / free, and {key: effective cell count} of each
    difference of means, n1 n2 / (n1 + n2) (the depth the deflate readout's noise stop is judged at)."""
    out, n = {}, {}
    for t in sorted(obs.celltype.unique()):
        mt = (obs.celltype == t).to_numpy()
        ds = [d for d in PMD + CTRL if d in set(obs.donor) and (mt & (obs.donor == d).to_numpy()).sum() >= min_cells]
        cen = {d: Z[mt & (obs.donor == d).to_numpy()].mean(0) for d in ds}
        cnt = {d: int((mt & (obs.donor == d).to_numpy()).sum()) for d in ds}

        def pooled(members):                                    # cell-weighted pooled centroid
            members = [m for m in members if m in cen]
            if not members:
                return None
            w = np.array([cnt[m] for m in members], float)
            return (np.stack([cen[m] for m in members]) * w[:, None]).sum(0) / w.sum()
        neff = lambda n1, n2: n1 * n2 / (n1 + n2)
        for d in ds:
            for ref_name, members in (('ctrl', [c for c in CTRL if c != d]), ('tbck', [c for c in TBCK if c != d])):
                ref = pooled(members)
                if ref is not None:
                    out[(ref_name, d, t)] = cen[d] - ref
                    n[(ref_name, d, t)] = neff(cnt[d], sum(cnt[m] for m in members if m in cnt))
            others = [e for e in ds if e != d]
            out[('free', d, t)] = cen[d] - np.mean([cen[e] for e in others], 0)
            n[('free', d, t)] = neff(cnt[d], sum(cnt[e] for e in others))
    return out, n


def load_sets(a, labels):
    import expert.utils.readout as ro
    sets = {('module', k): v for k, v in ro.load_modules(a.modules, list(labels)).items()}
    sets.update({('reactome', k): v for k, v in ro.load_gmt(a.reactome_gmt, vocab=list(labels)).items()})
    return sets


def donor_scores(S, labels, sets, mods):
    """Collaborator scores on one cosine profile S (n_prototypes,)."""
    N = len(S)
    pos = {l: i for i, l in enumerate(labels)}
    pct = (-S).argsort().argsort() / N                          # 0 = top of the loss tail
    z = (S - S.mean()) / S.std()
    r = {m: 1 - np.median(pct[mods[m]]) for m in CX}
    r['oxphos'] = np.mean([r[m] for m in CX])
    r['isr_on'] = z[[pos[g] for g in ISR if g in pos]].mean()
    rank = pct * N + 1
    for d, g in LESION.items():
        r[f'rank_{g}'] = int(rank[pos[g]]) if g in pos else np.nan
    return r, pct


def set_z(pct, members):
    """Fig. 5 set z on a loss-tail percentile vector (here 0 = top, so flip)."""
    p = 1 - pct
    N, var = len(p), p.var()
    out = []
    for m in members:
        k = len(m)
        out.append((p[m].mean() - p.mean()) / np.sqrt(var / k * (N - k) / (N - 1)))
    return np.array(out)


def perm_p(vals, cases, refs):
    """One-sided exact p for mean(cases) > mean(refs) over every relabelling of the pooled donors."""
    ok = [d for d in cases + refs if d in vals and np.isfinite(vals[d])]
    k = sum(d in cases for d in ok)
    if k < 2 or len(ok) - k < 2:
        return np.nan, np.nan, 0
    x = pd.Series({d: vals[d] for d in ok})
    obs = x[[d for d in ok if d in cases]].mean() - x[[d for d in ok if d not in cases]].mean()
    null = np.array([x[list(c)].mean() - x.drop(list(c)).mean() for c in itertools.combinations(ok, k)])
    return obs, float(np.mean(null >= obs - 1e-12)), len(null)


def load_cells(src, a):
    """z and obs with --exclude donors removed (they are neither cases nor reference)."""
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    Z = np.load(src / 'z_cells.npy').astype(np.float64)
    keep = ~obs.donor.isin(a.exclude).to_numpy()
    return Z[keep], obs[keep].reset_index(drop=True)


def score_dir(a):
    d = Path(a.out_dir) / ('score' if not a.tag else f'score_{a.tag}')
    d.mkdir(exist_ok=True)
    return d


def stage_score(a):
    src = Path(a.out_dir)
    out = score_dir(a)
    b = np.load(a.bank_npz, allow_pickle=True)
    Z, obs = load_cells(src, a)
    SH, ncell = shifts(Z, obs, a.min_cells)
    keys = list(SH)
    R = np.stack([SH[k] for k in keys])
    import expert.utils.readout as ro
    neff = np.array([ncell[k] for k in keys])
    pools = np.load(a.ctrl_pools, allow_pickle=True)
    rows, sets_rows, genes, clus, rstats = [], [], [], [], []
    for bank, readout in itertools.product(('trained', 'full'), a.readouts.split(',')):
        keep = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
        labels = np.asarray([str(l) for l in b['labels'][keep]])
        zs = ~b['trained'].astype(bool)[keep]
        Pn = unit(b['P'][keep].astype(np.float64))
        sets = load_sets(a, labels)
        lpos = {l: i for i, l in enumerate(labels)}
        mods = {m: np.array([lpos[g] for g in sets[('module', m)]]) for m in CX}
        skeys = list(sets)
        smem = [np.array([lpos[g] for g in sets[k]]) for k in skeys]
        S1 = unit(R) @ Pn.T
        S = S1
        if readout == 'deflate':
            # the shipped readout (predict(readout='deflate')): calibrated on this bank's span, stop judged at
            # each shift's effective depth; merged order transplanted onto the row's own cosine scale
            Praw = b['P'][keep].astype(np.float64)
            basis = ro.span_basis(Praw, 256)
            noise = [*ro.fit_noise_curve(pools['Q'], pools['dataset'], None, pool_cells=10),
                     *ro.fit_noise_curve(pools['Q'], pools['dataset'], basis, pool_cells=10)]
            rr = ro.calibrated_readout(R, Praw, neff, noise=np.array(noise), basis=basis, k=a.top_k)
            S = ro.transplant_scores(S1, rr['loss'], rr['gain'])
            for i, k_ in enumerate(keys):
                rstats.append(dict(bank=bank, ref=k_[0], donor=k_[1], celltype=k_[2], n_eff=neff[i],
                                   n_rounds=int(rr['n_rounds'][i]), effect=float(rr['effect_size'][i]),
                                   explainable=float(rr['explainable'][i])))
        for i, (ref, d, t) in enumerate(keys):
            r, pct = donor_scores(S[i], labels, sets, mods)
            rows.append(dict(bank=bank, readout=readout, ref=ref, donor=d, celltype=t, group='PMD' if d in PMD else
                             ('TBCK' if d in TBCK else 'undiagnosed'), n_eff=neff[i],
                             shift_norm=float(np.linalg.norm(R[i])), **r))
            for (fam, sn), z in zip(skeys, set_z(pct, smem)):
                sets_rows.append((bank, readout, ref, d, t, fam, sn, z))
            order = np.argsort(-S[i])
            for rk, j in enumerate(order[:a.top_k]):
                genes.append(dict(bank=bank, readout=readout, ref=ref, donor=d, celltype=t, rank=rk + 1,
                                  gene=labels[j], zero_shot=bool(zs[j]), cos=float(S1[i, j])))
        log.info('[%s/%s] scored %d shifts', bank, readout, len(keys))
    pd.DataFrame(rstats).to_csv(out / 'deflate_rounds.csv', index=False)
    T = pd.DataFrame(rows)
    T.to_csv(out / 'donor_scores.csv', index=False)
    ST = pd.DataFrame(sets_rows, columns=['bank', 'readout', 'ref', 'donor', 'celltype', 'family', 'set', 'z'])
    ST.to_csv(out / 'donor_set_z.csv.gz', index=False)
    pd.DataFrame(genes).to_csv(out / f'donor_top{a.top_k}.csv.gz', index=False)
    # ---- tests: PMD vs the donors scored against the same reference
    tests = []
    design = {'ctrl': [c for c in CTRL], 'tbck': ['PT38', 'PT50', 'PT55'], 'free': list(CTRL)}
    for (bank, readout, ref, t), x in T.groupby(['bank', 'readout', 'ref', 'celltype']):
        v = x.set_index('donor')
        for s in CX + ['oxphos', 'isr_on']:
            eff, p, n = perm_p(v[s].to_dict(), PMD, design[ref])
            tests.append(dict(bank=bank, readout=readout, ref=ref, celltype=t, score=s, effect=eff, p=p, n_perm=n))
    for (bank, readout, ref, t, fam, sn), x in ST.groupby(['bank', 'readout', 'ref', 'celltype', 'family', 'set']):
        if fam != 'reactome' and fam != 'module':
            continue
        eff, p, n = perm_p(x.set_index('donor').z.to_dict(), PMD, design[ref])
        tests.append(dict(bank=bank, readout=readout, ref=ref, celltype=t, score=f'{fam}:{sn}', effect=eff, p=p, n_perm=n))
    pd.DataFrame(tests).to_csv(out / 'tests.csv.gz', index=False)
    # ---- label-free clustering of donors per cell type
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score
    b_tr = b['trained'].astype(bool)
    Ptr = unit(b['P'][b_tr].astype(np.float64))
    for t in sorted(obs.celltype.unique()):
        ds = [d for (r_, d, t_) in keys if r_ == 'free' and t_ == t]
        if len(ds) < 6:
            continue
        V = {'latent': np.stack([SH[('free', d, t)] for d in ds])}
        V['prototype'] = unit(V['latent']) @ Ptr.T
        truth = [d in PMD for d in ds]
        for space, X in V.items():
            L = linkage(pdist(X, 'cosine'), 'average')
            lab = fcluster(L, 2, 'maxclust')
            ari = adjusted_rand_score(truth, lab)
            null = []
            for c in itertools.combinations(range(len(ds)), sum(truth)):
                tt = np.zeros(len(ds), bool)
                tt[list(c)] = True
                null.append(adjusted_rand_score(tt, lab))
            clus.append(dict(celltype=t, space=space, donors=' '.join(ds), clusters=' '.join(map(str, lab)),
                             ari=ari, p=float(np.mean(np.array(null) >= ari - 1e-12)),
                             linkage=json.dumps(L.tolist())))
    pd.DataFrame(clus).to_csv(out / 'clustering.csv', index=False)
    np.savez(out / 'shifts.npz', keys=np.array(['|'.join(k) for k in keys]), R=R.astype(np.float32))
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    m = T[(T.celltype == 'Monocytes') & (T.ref == 'ctrl')]
    log.info('monocytes, ref ctrl\n%s', m[['bank', 'readout', 'donor', 'group'] + CX + ['oxphos', 'isr_on'] +
                                           [f'rank_{g}' for g in LESION.values()]].round(3).to_string(index=False))
    tt = pd.DataFrame(tests)
    log.info('tests (oxphos/isr)\n%s', tt[tt.score.isin(['oxphos', 'isr_on'])].pivot_table(
        index=['bank', 'readout', 'ref', 'score'], columns='celltype', values='p').round(3).to_string())
    log.info('clustering\n%s', pd.DataFrame(clus)[['celltype', 'space', 'donors', 'clusters', 'ari', 'p']].to_string())


# ------------------------------------------------------------------------------------------- genes
def stage_genes(a):
    """Group profiles for the gene panels: PMD vs non-PMD and PMD vs TBCK per cell type, as equal-weight donor
    centroids (contrast name) and cells pooled (name + '_cell'), donors with >= --min-cells cells; cosine against both banks, merged with the own-transcript DE of
    the same contrast (fig6_data.py deg: regrouped:mito_vs_rest / regrouped:mito_vs_tbck)."""
    src = Path(a.out_dir)
    b = np.load(a.bank_npz, allow_pickle=True)
    Z, obs = load_cells(src, a)
    degs = pd.concat([pd.read_csv(f) for f in sorted(Path(a.deg_dir).glob('deg__*.csv.gz'))])
    rows = []
    for t in sorted(obs.celltype.unique()):
        mt = (obs.celltype == t).to_numpy()
        cen = {d: Z[mt & (obs.donor == d).to_numpy()].mean(0) for d in PMD + CTRL
               if (mt & (obs.donor == d).to_numpy()).sum() >= a.min_cells}
        pm = [cen[d] for d in PMD if d in cen]
        if len(pm) < 2:
            continue
        cnt = {d: int((mt & (obs.donor == d).to_numpy()).sum()) for d in cen}

        def pooled(mem):                                        # cell-weighted pooled centroid
            w = np.array([cnt[m] for m in mem], float)
            return (np.stack([cen[m] for m in mem]) * w[:, None]).sum(0) / w.sum()
        for name, refs, de_name in (('pmd_vs_rest', CTRL, a.de_rest), ('pmd_vs_tbck', TBCK, a.de_tbck)):
            rc = [d for d in refs if d in cen]
            if not rc:
                continue
            # 'donor': equal-weight donor centroids; 'cell': cells pooled (the best grid configuration, fig6_grid.py;
            # matches the cell-level DE it is compared against, but weights donors by their cell counts)
            for agg, r in (('donor', np.mean(pm, 0) - np.mean([cen[d] for d in rc], 0)),
                           ('cell', pooled([d for d in PMD if d in cen]) - pooled(rc))):
                dd = degs[(degs.celltype == t) & (degs.contrast == de_name)].set_index('gene')
                for bank in ('trained', 'full'):
                    keep = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
                    labels = np.asarray([str(l) for l in b['labels'][keep]])
                    s = unit(r[None])[0] @ unit(b['P'][keep].astype(np.float64)).T
                    rank = np.empty(len(s), int)
                    rank[np.argsort(-s)] = np.arange(1, len(s) + 1)
                    rows.append(pd.DataFrame({'bank': bank, 'celltype': t, 'gene': labels,
                                              'contrast': name if agg == 'donor' else f'{name}_cell',
                                              'zero_shot': ~b['trained'].astype(bool)[keep], 'cos': s, 'rank': rank,
                                              'de_score': dd.score.reindex(labels).values,
                                              'de_logfc': dd.logfc.reindex(labels).values,
                                              'de_padj': dd.pval_adj.reindex(labels).values}))
    pd.concat(rows).to_csv(score_dir(a) / 'group_genes.csv.gz', index=False)
    log.info('group profiles written')


# ------------------------------------------------------------------------------------------- ora
def stage_ora(a):
    """Fig. 5's top-k recipe: each donor x cell type shift (ref ctrl: cases vs pooled non-cases, a non-case vs the
    other non-cases) read out per tail (cosine and deflate, both banks), top --top-k per tail; the CASES' lists are
    unioned per cell type and tested for over-representation against the bank (hypergeometric, as fig5_predict ora).
    p_perm = exact donor-label permutation of the union statistic (hits): every relabelling of the cell type's
    donors into the same number of cases, union rebuilt each time."""
    import expert.utils.readout as ro
    from scipy.stats import hypergeom
    src = Path(a.out_dir)
    b = np.load(a.bank_npz, allow_pickle=True)
    Z, obs = load_cells(src, a)
    SH, neff = shifts(Z, obs, a.min_cells)
    keys = [k for k in SH if k[0] == 'ctrl']
    R = np.stack([SH[k] for k in keys])
    ne = np.array([neff[k] for k in keys])
    pools = np.load(a.ctrl_pools, allow_pickle=True)
    rows = []
    for bank in ('trained', 'full'):
        keep = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
        labels = [str(l) for l in b['labels'][keep]]
        Praw = b['P'][keep].astype(np.float64)
        N = len(labels)
        pos = {l: i for i, l in enumerate(labels)}
        sets = load_sets(a, labels)
        sk = [k for k, v in sets.items() if len(v) >= 3]
        M = np.zeros((len(sk), N), bool)
        for i, k in enumerate(sk):
            M[i, [pos[g] for g in sets[k]]] = True
        n_set = M.sum(1)
        S1 = unit(R) @ unit(Praw).T
        basis = ro.span_basis(Praw, 256)
        noise = np.array([*ro.fit_noise_curve(pools['Q'], pools['dataset'], None, pool_cells=10),
                          *ro.fit_noise_curve(pools['Q'], pools['dataset'], basis, pool_cells=10)])
        rr = ro.calibrated_readout(R, Praw, ne, noise=noise, basis=basis, k=a.top_k)
        S_def = ro.transplant_scores(S1, rr['loss'], rr['gain'])
        for readout, S in (('cosine', S1), ('deflate', S_def)):
            for t in sorted({k[2] for k in keys}):
                idx = [i for i, k in enumerate(keys) if k[2] == t]
                ds = [keys[i][1] for i in idx]
                cases = [d for d in ds if d in PMD]
                if len(cases) < 2 or len(ds) - len(cases) < 2:
                    continue
                for tail, sgn in (('loss', 1), ('gain', -1)):
                    top = {d: np.argsort(-sgn * S[i], kind='stable')[:a.top_k] for d, i in zip(ds, idx)}
                    TOP = np.zeros((len(ds), N), bool)
                    for j, d in enumerate(ds):
                        TOP[j, top[d]] = True

                    def union_hits(members):
                        u = TOP[members].any(0)
                        return (M[:, u]).sum(1), int(u.sum())
                    obs_h, K = union_hits([ds.index(d) for d in cases])
                    p_hyp = hypergeom.sf(obs_h - 1, N, n_set, K)
                    null = np.stack([union_hits(list(c))[0] for c in itertools.combinations(range(len(ds)),
                                                                                            len(cases))])
                    p_perm = (null >= obs_h[None, :]).mean(0)
                    for (fam, sn), h, ns, ph, pp, nm in zip(sk, obs_h, n_set, p_hyp, p_perm, null.mean(0)):
                        rows.append((bank, readout, t, tail, fam, sn, int(ns), K, int(h), ns * K / N, nm, ph, pp,
                                     len(null)))
            log.info('[ora] %s %s done', bank, readout)
    cols = ['bank', 'readout', 'celltype', 'tail', 'family', 'set', 'n_set', 'K', 'hits', 'expected', 'null_mean_hits',
            'p_hyper', 'p_perm', 'n_perm']
    pd.DataFrame(rows, columns=cols).to_csv(score_dir(a) / 'ora_union.csv.gz', index=False)


# ------------------------------------------------------------------------------------------- cells
def stage_cells_raw(a):
    """Append the UNCENTRED per-cell OXPHOS loss score (cos(z, u), no cohort-mean subtraction, kNN-smoothed as in
    stage cells) to an existing score/cells.csv.gz, leaving the UMAP coordinates untouched."""
    from sklearn.neighbors import NearestNeighbors
    import expert.utils.readout as ro
    src = Path(a.out_dir)
    cells = pd.read_csv(src / 'score' / 'cells.csv.gz')
    Z = np.load(src / 'z_cells.npy').astype(np.float32)
    assert len(cells) == len(Z), 'cells.csv.gz is not row-aligned with z_cells.npy'
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = [str(l) for l in b['labels'][tr]]
    P = unit(b['P'][tr].astype(np.float32))
    mods = ro.load_modules(a.modules, labels)
    pos = {l: i for i, l in enumerate(labels)}
    u = unit(P[[pos[g] for m in CX for g in mods[m]]].mean(0))
    idx = NearestNeighbors(n_neighbors=a.knn).fit(unit(Z - Z.mean(0))).kneighbors(return_distance=False)
    raw = unit(Z) @ u
    cells['oxphos_raw_cell'] = raw
    cells['oxphos_raw_smooth'] = raw[idx].mean(1)
    cells.to_csv(src / 'score' / 'cells.csv.gz', index=False)
    log.info('uncentred OXPHOS loss score by donor (monocytes)\n%s', cells[cells.celltype == 'Monocytes']
             .groupby('donor').oxphos_raw_smooth.mean().round(4).to_string())


def stage_cells(a):
    import umap
    from sklearn.neighbors import NearestNeighbors
    import expert.utils.readout as ro
    src = Path(a.out_dir)
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    Z = np.load(src / 'z_cells.npy').astype(np.float32)
    b = np.load(a.bank_npz, allow_pickle=True)
    tr = b['trained'].astype(bool)
    labels = [str(l) for l in b['labels'][tr]]
    P = unit(b['P'][tr].astype(np.float32))
    mods = ro.load_modules(a.modules, labels)
    pos = {l: i for i, l in enumerate(labels)}
    u = unit(P[[pos[g] for m in CX for g in mods[m]]].mean(0))
    uc = {m: unit(P[[pos[g] for g in mods[m]]].mean(0)) for m in CX}
    D = np.zeros_like(Z)
    for t in obs.celltype.unique():
        mt = (obs.celltype == t).to_numpy()
        # label-free reference: equal-weight mean of every donor's centroid in this cell type
        cen = [Z[mt & (obs.donor == d).to_numpy()].mean(0) for d in obs.donor[mt].unique()
               if (mt & (obs.donor == d).to_numpy()).sum() >= a.min_cells]
        D[mt] = Z[mt] - np.mean(cen, 0)
    score = unit(D) @ u
    nn = NearestNeighbors(n_neighbors=a.knn).fit(unit(Z - Z.mean(0)))
    idx = nn.kneighbors(return_distance=False)
    obs['oxphos_cell'] = score
    obs['oxphos_smooth'] = score[idx].mean(1)
    raw = unit(Z) @ u                                          # uncentred: no cohort-mean subtraction
    obs['oxphos_raw_cell'] = raw
    obs['oxphos_raw_smooth'] = raw[idx].mean(1)
    for m in CX:                                              # the same score per complex (panel B)
        obs[f'{m}_cell'] = unit(D) @ uc[m]
    for name, m in (('all', np.ones(len(obs), bool)), ('mono', (obs.celltype == 'Monocytes').to_numpy())):
        X = unit(Z[m] - Z[m].mean(0))
        E = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=0).fit_transform(X)
        obs.loc[m, f'umap1_{name}'], obs.loc[m, f'umap2_{name}'] = E[:, 0], E[:, 1]
    obs.to_csv(src / 'score' / 'cells.csv.gz', index=False)
    log.info('per-cell oxphos by donor (monocytes)\n%s', obs[obs.celltype == 'Monocytes'].groupby('donor')
             [['oxphos_cell', 'oxphos_smooth']].mean().round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['encode', 'encode_pbulk', 'score', 'genes', 'ora', 'cells', 'cells_raw'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--query')
    ap.add_argument('--model-dir')
    ap.add_argument('--bank-npz', help='full bank (trained + zero-shot fills) of the same model')
    ap.add_argument('--ctrl-pools', help='10-cell training control pools (Q, dataset) for the deflate noise curve')
    ap.add_argument('--modules', help='collaborator modules.tsv (CI/CIII/CIV/CV subunits + assembly factors)')
    ap.add_argument('--reactome-gmt')
    ap.add_argument('--deg-dir', help='fig6_data.py deg/ (own-transcript DE per cell type)')
    ap.add_argument('--mask', default='MT-*', help="encode: input genes dropped, comma list ('MT-*,MT1*,MT2A,MT3,MT4' "
                                                   "= the Fig. 5/6 mask)")
    ap.add_argument('--readouts', default='cosine,deflate')
    ap.add_argument('--cases', default=','.join(PMD), help='case donors (default the 4 PMD patients)')
    ap.add_argument('--exclude', nargs='*', default=[], help='donors dropped from cases and reference')
    ap.add_argument('--tag', default='', help='score/genes output in score_<tag>/')
    ap.add_argument('--de-rest', default='regrouped:mito_vs_rest', help='fig6_data deg contrast for cases vs non-PMD')
    ap.add_argument('--de-tbck', default='regrouped:mito_vs_tbck', help='fig6_data deg contrast for cases vs TBCK')
    ap.add_argument('--celltype-col', default='singler_labels_main')
    ap.add_argument('--donor-col', default='patient_id')
    ap.add_argument('--stim-col', default='', help="encode: obs column appended to the cell type ('Myeloid|Unstim')")
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--top-k', type=int, default=200)
    ap.add_argument('--knn', type=int, default=30)
    ap.add_argument('--n-rep', type=int, default=10, help='encode_pbulk: baseline pools per non-PMD donor x cell type')
    ap.add_argument('--batch-size', type=int, default=2048)
    ap.add_argument('--n-cores', type=int, default=8)
    main_ = ap.parse_args()
    PMD[:] = main_.cases.split(',')
    {'encode': stage_encode, 'encode_pbulk': stage_encode_pbulk, 'score': stage_score, 'genes': stage_genes, 'ora': stage_ora, 'cells': stage_cells, 'cells_raw': stage_cells_raw}[main_.stage](main_)
