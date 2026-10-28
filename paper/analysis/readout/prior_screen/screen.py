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

"""Prior screen: biological quality and zero-shot fill gain for EVERY prior, on EFFGATE3 atoms.

  list    print the prior registry (task index -> prior)
  bio     kNN-to-pathway precision@k / lift for one prior (array task), 3 libraries,
          full coverage AND size-matched (n annotated genes subsampled, several draws)
  zs      f(missing) fill of the untrained atom rows from one prior through ExPert's own PriorAtomPredictor
          (kwargs + fit rows captured from the model, capture.py) + R shuffled twins; scored with the
          mapcache_score.zs() recipe (pooled queries, candidate panel, top-10 x chance) for three query groups:
          all / seen dataset / unseen dataset. Saves the unit fills for the combo stage.
  combos  ExPert's fusion of per-bank unit fills over every subset of the top singles (array chunks),
          with shuffled-twin combos as the null; combos-merge joins the chunks
  parity  the model's own banks through the harness must reproduce its f(missing); offline fusion of
          singles must equal the direct multi-bank fit
  convert GenePT pickles -> panel-restricted npz in --work-dir/banks (run once before the arrays)

Every path is a flag; the sbatch files in this directory set them.

Preprocessing is the harness's: panel genes only, column-centred over covered rows, cosine kNN.
Screen-derived priors (coexpr pseudobulks, response consensus) LEAK the ZS truth -- bio only.
"""
import argparse, glob, json, os, pickle, sys, time, itertools
import numpy as np

LIBS = ('KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023')


def _u(M, eps=1e-12):
    return M / np.clip(np.linalg.norm(M, axis=-1, keepdims=True), eps, None)


# ------------------------------------------------------------------------------------------------
#  registry
# ------------------------------------------------------------------------------------------------
def registry(a):
    """[(name, family, source, leak)] in a FIXED order (array index = position)."""
    reg = []
    for p in sorted(glob.glob(f'{a.emb_dir}/*/**/gene_embeddings.npz', recursive=True)):
        rel = os.path.relpath(os.path.dirname(p), a.emb_dir)
        if rel.startswith(('priors', 'random_init')) or rel.endswith('-shuffled'):
            continue
        reg.append((rel, 'text', p, False))
    reg.append(('GenePT/ada-002', 'text', f'{a.genept_dir}/ada/GenePT_gene_embedding_ada_text.pickle', False))
    reg.append(('GenePT/3-large', 'text', f'{a.genept_dir}/model3/GenePT_gene_protein_embedding_model_3_text.pickle', False))
    ext = [('string', 'PPI (STRING)'), ('string_exp', 'PPI (STRING)'), ('string_coexp', 'PPI (STRING)'),
           ('depmap', 'DepMap'), ('depmap_dstd', 'DepMap'), ('depmap_drop2', 'DepMap'),
           ('l1000', 'L1000'), ('morph_crispr', 'Morphology'), ('morph_orf', 'Morphology'),
           ('proteintalks_global', 'Proteome'), ('proteintalks_within', 'Proteome'),
           ('gff_pos', 'Genomic position')]
    for n, fam in ext:
        reg.append((n, fam, f'{a.prior_dir}/{n}.npz', False))
    for n in ('coexpr_pbulk38k', 'coexpr_pbulk38k_v2', 'resp_consensus'):
        reg.append((n, 'Screen-derived', f'{a.prior_dir}/{n}.npz', True))
    reg.append(('random_gaussian', 'Random', a.random_prior, False))
    return reg


def panel_genes(a):
    return np.asarray(np.load(a.panel, allow_pickle=True)['genes']).astype(str)


def load_prior(source, genes_keep, bank_dir):
    """-> (genes, E float32) restricted to genes_keep. GenePT pickles are converted once into bank_dir."""
    if source.endswith('.pickle'):
        cache = os.path.join(bank_dir, os.path.basename(source).replace('.pickle', '_panel.npz'))
        if not os.path.exists(cache):
            with open(source, 'rb') as fh:
                obj = pickle.load(fh)
            keep = set(genes_keep)
            g = [k for k in obj if str(k) in keep]
            E = np.stack([np.asarray(obj[k], np.float32) for k in g])
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, embeddings=E, genes=np.array([str(x) for x in g], dtype=object))
            print(f'  [genept] {len(obj):,} genes -> {len(g):,} in panel -> {cache}', flush=True)
            del obj
        source = cache
    d = np.load(source, allow_pickle=True)
    g = np.asarray(d['genes']).astype(str); E = np.asarray(d['embeddings'], np.float32)
    keep = np.isin(g, genes_keep)
    g, E = g[keep], E[keep]
    last = np.array(sorted({x: i for i, x in enumerate(g)}.values()), dtype=np.int64)  # duplicates: LAST row wins, as in mapcache_build
    return g[last], E[last]


def to_rows(genes, E, row_gene, perm_seed=None):
    """prior matrix aligned to atom rows; perm_seed permutes WHICH gene owns which vector."""
    if perm_seed is not None:
        E = E[np.random.default_rng(1000 + perm_seed).permutation(len(E))]
    gi = {g: i for i, g in enumerate(genes)}
    idx = np.array([gi.get(g, -1) for g in row_gene])
    v = idx >= 0
    M = np.zeros((len(row_gene), E.shape[1]), np.float32)
    M[v] = E[idx[v]]
    return M, v


# ------------------------------------------------------------------------------------------------
#  bio quality
# ------------------------------------------------------------------------------------------------
def read_gmt(path):
    sets = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 3:
                sets[p[0]] = {x for x in p[2:] if x}
    return sets


def bio_scores(X, genes, sets, ks, rng=None, n_sub=None):
    """precision@k over annotated genes' cosine kNN, exact all-pairs random baseline."""
    from scipy import sparse
    g2s = {}
    for j, (_, mem) in enumerate(sorted(sets.items())):
        for g in mem:
            g2s.setdefault(g, []).append(j)
    ann = np.array([g in g2s for g in genes])
    idx = np.flatnonzero(ann)
    if n_sub is not None:
        if len(idx) < n_sub:
            return None
        idx = np.sort(rng.choice(idx, n_sub, replace=False))
    n = len(idx)
    rr, cc = [], []
    for i, gidx in enumerate(idx):
        js = g2s[genes[gidx]]; rr += [i] * len(js); cc += js
    B = sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(n, len(sets)))
    Xn = _u(X[idx]).astype(np.float32)
    kmax = max(ks)
    flags = np.zeros((n, kmax), np.int8); shared_tot = 0.0
    for s in range(0, n, 1000):
        e = min(s + 1000, n)
        S = Xn[s:e] @ Xn.T
        S[np.arange(e - s), np.arange(s, e)] = -np.inf
        nb = np.argpartition(-S, kmax, axis=1)[:, :kmax]
        o = np.argsort(-np.take_along_axis(S, nb, 1), axis=1)
        nb = np.take_along_axis(nb, o, 1)
        Sh = (B[s:e] @ B.T).toarray() > 0
        shared_tot += Sh.sum() - Sh[np.arange(e - s), np.arange(s, e)].sum()
        flags[s:e] = np.take_along_axis(Sh, nb, 1)
    base = shared_tot / (n * (n - 1))
    cum = np.cumsum(flags, 1)
    prec = {k: float((cum[:, k - 1] / k).mean()) for k in ks}
    return dict(n=int(n), baseline=float(base), **{f'p@{k}': prec[k] for k in ks},
                **{f'lift@{k}': prec[k] / base for k in ks})


def cmd_bio(a):
    reg = registry(a); name, fam, src, leak = reg[a.task]
    t0 = time.time()
    pg = panel_genes(a)
    g, E = load_prior(src, pg, f'{a.work_dir}/banks')
    X = E - E.mean(0, keepdims=True) if a.center else E
    rec = dict(prior=name, family=fam, leak=leak, n_panel_covered=int(len(g)), dim=int(E.shape[1]),
               center=bool(a.center))
    for lib in LIBS:
        sets = read_gmt(f'{a.gene_sets_dir}/{lib}.gmt')
        full = bio_scores(X, g, sets, a.ks)
        rec[lib] = dict(full=full)
        subs = [bio_scores(X, g, sets, a.ks, np.random.default_rng(s), a.n_sub) for s in range(a.draws)]
        if subs[0] is not None:
            rec[lib]['matched'] = {k: float(np.mean([d[k] for d in subs])) for k in subs[0]}
            rec[lib]['matched_sd'] = {k: float(np.std([d[k] for d in subs], ddof=1)) for k in subs[0]}
        else:
            rec[lib]['matched'] = None
        m = rec[lib]['matched']
        print(f'  [bio] {name:<42} {lib:<28} full n {full["n"]:>6} lift@10 {full["lift@10"]:6.2f}'
              + (f'  matched(n={a.n_sub}) lift@10 {m["lift@10"]:6.2f}' if m else '  matched: too few'),
              flush=True)
    os.makedirs(f'{a.out_dir}/bio', exist_ok=True)
    tag = '' if a.center else '_raw'
    with open(f'{a.out_dir}/bio/{a.task:03d}{tag}.json', 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f'[bio] {name} done ({time.time() - t0:.0f}s)')


# ------------------------------------------------------------------------------------------------
#  zero-shot fill
# ------------------------------------------------------------------------------------------------
GROUPS = ('all', 'seen_dataset', 'unseen_dataset')                      # per-query groups (+ ds:<dataset>)
PGROUPS = ('pooled_all', 'pooled_seen_dataset', 'pooled_unseen_dataset')   # pooled-per-candidate groups (main)


def _col(o, name):
    import h5py
    g = o[name]
    return (np.asarray(g['categories'].asstr()[:])[np.asarray(g['codes'][:])] if isinstance(g, h5py.Group)
            else np.asarray(g[:]))


class ZS:
    """Zero-shot readout. Panel = every untrained row with >= 1 query, IDENTICAL for every query, so chance@10 =
    10 / n_panel is shared by every group and x over chance is comparable across datasets.

    POOLED (main): a candidate's queries (dataset-centred cell means) are averaged within a group, then ranked against
    the panel -- `pooled_all` / `pooled_seen_dataset` / `pooled_unseen_dataset`.
    PER-QUERY (per-dataset supplement): every (dataset, perturbation) query ranked on its own -- `all`, `seen_dataset`,
    `unseen_dataset`, `ds:<dataset>`.

    EFFICIENCY WEIGHTS. Query weight = mean obs['efficiency'] of the group's cells in the ZS DB, clipped at 0, all-NaN -> 0
    (the ood_readout_probe w_top10 convention; `wraw_`). Raw efficiency is on different scales per dataset (group
    medians 1.0-4.9), so the MAIN weight (`w_`) is that value's WITHIN-DATASET percentile, all-NaN -> 0. A pooled
    candidate's weight is the mean of its member queries' weights. Weighted x = sum(w * hit) / sum(w) / chance."""

    def __init__(self, path, unseen_datasets, zs_adata, panel='cand', group_map=None):
        import h5py
        import pandas as pd
        z = np.load(path, allow_pickle=True)
        self.Aall = z['atoms'].astype(np.float32); self.ntr = int(z['n_trained'])
        self.rows = np.asarray(z['row_gene']).astype(str)
        Q, t = z['Q'].astype(np.float32), z['truth'].astype(np.int64)
        keys = np.asarray(z['keys']).astype(str)
        self.qds = np.array([k.split('|', 1)[0] for k in keys])
        self.cand = np.unique(t)
        # PANEL: 'cand' = the untrained candidates only (what the screen scores); 'full' = the deployed vocabulary, every
        # atom row incl. the TRAINED block, which is what predict(obs_only=False) ranks against. Chance scales with the
        # panel (10 / n_panel), so x-over-chance is NOT comparable between panels -- compare top10 there.
        self.panel = panel
        self.n_panel = len(self.cand) if panel == 'cand' else len(self.Aall)
        self.ch10 = 10.0 / self.n_panel
        self.pos = np.full(len(self.Aall), -1, np.int64); self.pos[self.cand] = np.arange(len(self.cand))
        self.Qn = _u(Q).astype(np.float32); self.qci = self.pos[t]
        # ---- efficiency weights from ALL cells of each (dataset, perturbation) in the ZS DB
        with h5py.File(zs_adata, 'r') as f:
            o = f['obs']
            df = pd.DataFrame(dict(ds=_col(o, 'dataset').astype(str), lab=_col(o, 'perturbation').astype(str),
                                   eff=np.asarray(_col(o, 'efficiency'), float)))
        g = df.groupby(['ds', 'lab']).eff.agg(lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan)
        raw = np.array([g.get((k.split('|', 1)[0], k.split('|', 1)[1]), np.nan) for k in keys])
        wraw = np.where(np.isfinite(raw), np.clip(raw, 0, None), 0.0)
        wpct = np.zeros(len(keys))
        for d in np.unique(self.qds):
            m = self.qds == d; ok = m & np.isfinite(raw)
            if ok.sum() > 1:
                o_ = np.argsort(np.argsort(np.clip(raw[ok], 0, None), kind='stable'), kind='stable')
                wpct[ok] = o_ / (ok.sum() - 1)
        self.qw = {'w': wpct, 'wraw': wraw}
        print(f'[zs] efficiency weights: {int(np.isfinite(raw).sum()):,}/{len(raw):,} queries with efficiency; '
              f'mean pct-weight {wpct.mean():.3f}', flush=True)
        self.unseen = sorted(unseen_datasets)
        un = np.isin(self.qds, self.unseen)
        self.qgroups = {'all': np.ones(len(t), bool), 'seen_dataset': ~un, 'unseen_dataset': un}
        for d in np.unique(self.qds):
            self.qgroups[f'ds:{d}'] = self.qds == d
        # optional CONTEXT groups (dataset -> label, e.g. the four T-cell donors of one condition
        # pooled): scored per query exactly like the ds: groups, so weights and shuffles stay exact
        if group_map:
            import pandas as pd
            gm = pd.read_csv(group_map, sep='\t')
            for lab, dsets in gm.groupby('label').dataset:
                m = np.isin(self.qds, dsets.to_numpy())
                if m.any():
                    self.qgroups[f'ctx:{lab}'] = m
        # ---- pooled groups: per candidate mean of member queries (and of their weights)
        self.P = {}
        for pg, qg in zip(PGROUPS, GROUPS):
            m = self.qgroups[qg]; tg = t[m]
            order = np.argsort(tg, kind='stable'); ts = tg[order]
            cuts = np.flatnonzero(np.diff(ts)) + 1
            segs = list(zip(np.r_[0, cuts], np.r_[cuts, len(ts)]))
            mi = np.flatnonzero(m)[order]
            Qp = _u(np.stack([Q[mi[b:e]].mean(0) for b, e in segs])).astype(np.float32)
            self.P[pg] = dict(Q=Qp, ci=self.pos[np.unique(ts)],
                              w={k: np.array([v[mi[b:e]].mean() for b, e in segs]) for k, v in self.qw.items()})
        for pg in PGROUPS:
            print(f'[zs] {pg:<22} {len(self.P[pg]["ci"]):>5,} candidates', flush=True)
        for g_ in GROUPS:
            m = self.qgroups[g_]
            print(f'[zs] {g_:<22} {int(m.sum()):>6,} queries, {len(np.unique(self.qds[m]))} datasets', flush=True)
        print(f'[zs] atoms {self.Aall.shape}, {self.ntr:,} trained; panel {len(self.cand):,} candidates, '
              f'chance@10 {self.ch10:.5f}; unseen datasets {self.unseen}', flush=True)

    @staticmethod
    def _rank(Qs, ci, An, chunk=4096):
        r = np.empty(len(ci), np.int32)
        for b in range(0, len(ci), chunk):
            S = Qs[b:b + chunk] @ An.T
            tr = S[np.arange(S.shape[0]), ci[b:b + chunk]]
            r[b:b + chunk] = (S > tr[:, None]).sum(1)
        return r

    def ranks(self, P, ok, query=True):
        """{'query': rank per query, <pooled group>: rank per candidate}. P = unit fills for candidate rows, ok = filled;
        unfilled candidates keep the atom table's random init. On the 'full' panel the TRAINED atoms are distractors too."""
        if self.panel == 'cand':
            Az = self.Aall[self.cand].copy(); Az[ok] = P[ok]
            colof = np.arange(len(self.cand))
        else:
            Az = self.Aall.copy(); Az[self.cand[ok]] = P[ok]
            colof = self.cand
        An = _u(Az).astype(np.float32)
        out = {pg: self._rank(self.P[pg]['Q'], colof[self.P[pg]['ci']], An) for pg in PGROUPS}
        if query:
            out['query'] = self._rank(self.Qn, colof[self.qci], An)
        return out

    def _stats(self, h, w, okg):
        return dict(zs_x=float(h.mean() / self.ch10), top10=float(h.mean()), hits10=int(h.sum()),
                    w_zs_x=float((w['w'] * h).sum() / max(w['w'].sum(), 1e-12) / self.ch10),
                    w_top10=float((w['w'] * h).sum() / max(w['w'].sum(), 1e-12)),
                    wraw_zs_x=float((w['wraw'] * h).sum() / max(w['wraw'].sum(), 1e-12) / self.ch10),
                    cov=float(okg.mean()), zs_x_covered=float(h[okg].mean() / self.ch10) if okg.any() else float('nan'))

    def summary(self, R, ok):
        out = dict(cov=float(ok.mean()))
        for pg in PGROUPS:
            ci = self.P[pg]['ci']; h = R[pg] < 10
            out[pg] = dict(**self._stats(h, self.P[pg]['w'], ok[ci]), n_query=int(len(ci)), n_cand=int(len(ci)),
                           rank_med=float(np.median(R[pg])))
        if 'query' in R:
            h = R['query'] < 10
            for g_, m in self.qgroups.items():
                out[g_] = dict(**self._stats(h[m], {k: v[m] for k, v in self.qw.items()}, ok[self.qci[m]]),
                               n_query=int(m.sum()), n_cand=int(len(np.unique(self.qci[m]))),
                               rank_med=float(np.median(R['query'][m])))
        return out


class Fill:
    """ExPert's OWN f(missing): PriorAtomPredictor with the constructor kwargs, atoms and fit / pull / missing
    rows captured from the model's refit (capture.py). Only the banks change, so every prior (set) goes
    through the same graph map, per-bank normalisation and fusion the model uses."""

    def __init__(self, capture, zs, capture_rows=None):
        import torch
        from expert.modules._prior_atoms import PriorAtomPredictor
        self.torch, self.PAP = torch, PriorAtomPredictor
        c = np.load(capture, allow_pickle=True)
        self.kw = json.loads(str(c['kw_json']))
        if getattr(Fill, 'MISSING_SIM_POW', None) is not None:
            # the shipped zero-shot fill weights graph neighbours by cos^p (prior_atom_missing_sim_pow);
            # the captured recipe predates it (p = 1)
            self.kw['missing_sim_pow'] = float(Fill.MISSING_SIM_POW)
        ge = c['graph_evidence']
        self.kw['graph_evidence'] = torch.from_numpy(ge) if len(ge) else None
        if capture_rows is None:
            assert np.array_equal(c['atoms'], zs.Aall), 'capture atoms != zs-cache atoms (different checkpoint?)'
            self.atoms = torch.from_numpy(c['atoms'])
        else:
            # FOREIGN ATOMS: the fit / pull / missing rows are data-derived (train-split evidence ranks), so the recipe
            # transfers to another model's atom table when its label rows are identical -- asserted, never assumed
            zr = np.load(capture_rows, allow_pickle=True)
            assert np.array_equal(np.asarray(zr['row_gene']).astype(str), zs.rows) and int(zr['n_trained']) == zs.ntr, \
                'capture was taken on a different label/row order'
            assert c['atoms'].shape[0] == zs.Aall.shape[0]
            self.atoms = torch.from_numpy(zs.Aall)
            print(f'[fill] recipe from {capture} applied to FOREIGN atoms {zs.Aall.shape} (rows verified)', flush=True)
        self.strong, self.weak, self.missing = (torch.from_numpy(c[k]).long() for k in ('strong', 'weak', 'missing'))
        self.n_trained = int(c['n_trained']); self.zs = zs
        self.lam = float(self.kw.get('lam', 0.5))
        print(f'[fill] ExPert PriorAtomPredictor {dict((k, v) for k, v in self.kw.items() if k != "graph_evidence")}; '
              f'fit {len(self.strong):,} / pull {len(self.weak):,} / missing {len(self.missing):,}', flush=True)

    def predict(self, banks):
        """banks [(name, M (n_rows, d), valid bool)] -> unit fills for the panel rows + filled mask."""
        t = self.torch
        f = self.PAP(**self.kw)
        f.fit(self.atoms, [(n, t.from_numpy(np.ascontiguousarray(M)), t.from_numpy(v)) for n, M, v in banks],
              self.strong, self.weak, missing=self.missing, n_trained=self.n_trained)
        zs = self.zs
        P = np.zeros((len(zs.cand), self.atoms.shape[1]), np.float32); ok = np.zeros(len(zs.cand), bool)
        if getattr(f, 'missing', None) is not None and len(f.missing_rows):
            p = zs.pos[f.missing_rows.numpy()]; keep = p >= 0
            P[p[keep]] = _u(f.missing.numpy()[keep]); ok[p[keep]] = True
        return P, ok

    def fuse(self, Ps, oks):
        """ExPert's fusion of per-bank unit predictions: (1-lam, lam) for two banks, equal otherwise."""
        acc = np.zeros_like(Ps[0]); w = np.zeros(len(Ps[0]), np.float32)
        for i, (P, ok) in enumerate(zip(Ps, oks)):
            bw = ((1.0 - self.lam) if i == 0 else self.lam) if len(Ps) == 2 else 1.0
            acc[ok] += bw * P[ok]; w[ok] += bw
        good = w > 0
        out = np.zeros_like(acc); out[good] = _u(acc[good])
        return out, good


def unseen_datasets(a):
    md = json.load(open(a.split_meta))
    return [x for f in md['folds'] if int(f['fold']) == a.fold for x in f['held_out_datasets']]


def pred_path(a, task, key):
    return f'{a.work_dir}/{a.pred_tag or a.tag}/pred_expert/{task:03d}_{key}.npz'


def cmd_zs(a):
    reg = registry(a); name, fam, src, leak = reg[a.task]
    t0 = time.time()
    rec = dict(prior=name, family=fam, leak=leak, fill='ExPert PriorAtomPredictor (captured)', model=a.tag,
               panel=a.zs_panel)
    if leak:
        rec['skipped'] = 'screen-derived prior leaks the ZS truth (pull-side only)'
        print(f'[zs] {name}: {rec["skipped"]}')
    else:
        zs = ZS(a.zs_cache, unseen_datasets(a), a.zs_adata, panel=a.zs_panel); fill = Fill(a.capture, zs, a.capture_rows)
        rec.update(n_panel=int(zs.n_panel), chance10=float(zs.ch10))
        keys = ['real'] + [f'shuf{r}' for r in range(a.reps)]
        # the FILL is panel-independent: re-scoring the same atoms on the other panel reuses the cached fills
        cached = a.reuse_fills and all(os.path.exists(pred_path(a, a.task, k)) for k in keys)
        g, E = (None, None) if cached else load_prior(src, zs.rows, f'{a.work_dir}/banks')
        os.makedirs(os.path.dirname(pred_path(a, 0, 'x')), exist_ok=True)
        for rep in [None] + list(range(a.reps)):
            key = 'real' if rep is None else f'shuf{rep}'
            if cached:
                P, ok = load_pred(a, a.task, key)
            else:
                M, v = to_rows(g, E, zs.rows, rep)
                P, ok = fill.predict([(name, M, v)])
            R = zs.ranks(P, ok); s_ = zs.summary(R, ok)
            rec[key] = s_
            if not cached:
                np.savez(pred_path(a, a.task, key), P=P, ok=ok, **{f'r_{k}': v for k, v in R.items()})
            print(f'  [zs] {name:<42} {key:<6} ' + '  '.join(f'{g_} x{s_[g_]["zs_x"]:6.2f} w{s_[g_]["w_zs_x"]:6.2f}'
                                                              for g_ in ('pooled_all', 'pooled_unseen_dataset', 'all'))
                  + f'  cov {s_["cov"]:.3f}  ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(f'{a.out_dir}/{a.tag}/zs', exist_ok=True)
    with open(f'{a.out_dir}/{a.tag}/zs/{a.task:03d}.json', 'w') as fh:
        json.dump(rec, fh, indent=1)


def load_pred(a, task, key):
    d = np.load(pred_path(a, task, key))
    return d['P'], d['ok']


def cmd_combos(a):
    """singles -> pool (top-K by overall ZS, at most --max-text text encoders) -> every subset up to --max-size."""
    reg = registry(a); zs = ZS(a.zs_cache, unseen_datasets(a), a.zs_adata, panel=a.zs_panel)
    fill = Fill(a.capture, zs, a.capture_rows)
    R = []
    for i, (name, fam, _, leak) in enumerate(reg):
        p = f'{a.out_dir}/{a.tag}/zs/{i:03d}.json'
        if leak or fam == 'Random' or not os.path.exists(p):
            continue
        R.append((json.load(open(p))['real'][a.main_group][a.metric], i, name, fam))
    R.sort(reverse=True)
    pool, ntext = [], 0
    for x, i, name, fam in R:
        if fam == 'text':
            if ntext >= a.max_text:
                continue
            ntext += 1
        pool.append((i, name, fam, x))
        if len(pool) == a.pool:
            break
    print('[pool] ' + ', '.join(f'{n} ({x:.2f}x)' for _, n, _, x in pool), flush=True)
    real = {i: load_pred(a, i, 'real') for i, *_ in pool}
    shuf = {i: load_pred(a, i, 'shuf0') for i, *_ in pool}
    half = np.random.default_rng(7).permutation(len(zs.cand)) % 2 == 0      # split-half over candidate GENES
    qhalf = half[zs.qci]                                                    # a gene's queries stay in one half
    subsets = [c for size in range(1, min(a.max_size, len(pool)) + 1) for c in itertools.combinations(range(len(pool)), size)]
    mine = subsets[a.task::a.n_chunks]          # round-robin slice: every chunk gets a mix of sizes
    print(f'[chunk] {a.task}/{a.n_chunks}: {len(mine)} of {len(subsets)} subsets', flush=True)

    def xs(h, w, m):
        return (float(h[m].mean() / zs.ch10), float((w['w'][m] * h[m]).sum() / max(w['w'][m].sum(), 1e-12) / zs.ch10))

    rows = []
    t0 = time.time()
    for c in mine:
        ids = [pool[j][0] for j in c]
        P, ok = fill.fuse([real[i][0] for i in ids], [real[i][1] for i in ids])
        Pn, okn = fill.fuse([shuf[i][0] for i in ids], [shuf[i][1] for i in ids])
        Rr, Rn = zs.ranks(P, ok), zs.ranks(Pn, okn)
        row = dict(size=len(c), priors='+'.join(pool[j][1] for j in c),
                   families='+'.join(sorted({pool[j][2] for j in c})), cov=float(ok.mean()))
        for pg in PGROUPS:
            ci, w = zs.P[pg]['ci'], zs.P[pg]['w']; h, hn = Rr[pg] < 10, Rn[pg] < 10; hm = half[ci]; al = np.ones(len(ci), bool)
            (x, wx), (xa, wxa), (xb, wxb), (xn, wxn) = xs(h, w, al), xs(h, w, hm), xs(h, w, ~hm), xs(hn, w, al)
            row[pg] = dict(zs_x=x, w_zs_x=wx, halfA=xa, halfB=xb, w_halfA=wxa, w_halfB=wxb, null_zs_x=xn, w_null_zs_x=wxn)
        h, hn = Rr['query'] < 10, Rn['query'] < 10
        for g, m in zs.qgroups.items():
            (x, wx), (xn, wxn) = xs(h, zs.qw, m), xs(hn, zs.qw, m)
            rec = dict(zs_x=x, w_zs_x=wx, null_zs_x=xn, w_null_zs_x=wxn)
            if g in GROUPS:
                (xa, wxa), (xb, wxb) = xs(h, zs.qw, m & qhalf), xs(h, zs.qw, m & ~qhalf)
                rec.update(halfA=xa, halfB=xb, w_halfA=wxa, w_halfB=wxb)
            row[g] = rec
        rows.append(row)
    print(f'  [combos] chunk {a.task}: {len(rows)} subsets scored ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(f'{a.out_dir}/{a.tag}/combos_parts', exist_ok=True)
    with open(f'{a.out_dir}/{a.tag}/combos_parts/{a.task:03d}.json', 'w') as fh:
        json.dump(dict(pool=pool, n_subsets=len(subsets), n_chunks=a.n_chunks, rows=rows), fh, indent=1)


def cmd_combos_merge(a):
    parts = sorted(glob.glob(f'{a.out_dir}/{a.tag}/combos_parts/[0-9][0-9][0-9].json'))
    if not parts:
        raise SystemExit(f'no combo chunks under {a.out_dir}/{a.tag}/combos_parts')
    P = [json.load(open(p)) for p in parts]
    n_chunks, n_sub = P[0]['n_chunks'], P[0]['n_subsets']
    assert len(P) == n_chunks and all(x['pool'] == P[0]['pool'] for x in P), 'missing chunk or pool mismatch'
    rows = [r for x in P for r in x['rows']]
    assert len(rows) == n_sub, f'{len(rows)} rows vs {n_sub} subsets'
    rows.sort(key=lambda r: (r['size'], r['priors']))
    for size in sorted({r['size'] for r in rows}):
        b = max((r for r in rows if r['size'] == size), key=lambda r: r[a.main_group][a.metric])
        print(f'  [combos] size {size}: {sum(r["size"] == size for r in rows)} sets, best {a.main_group} {a.metric} '
              f'{b[a.main_group][a.metric]:.2f} {b["priors"]}')
    with open(f'{a.out_dir}/{a.tag}/combos.json', 'w') as fh:
        json.dump(dict(pool=P[0]['pool'], rows=rows), fh, indent=1)
    print(f'[merge] {len(parts)} chunks -> {len(rows)} subsets')


def cmd_parity(a):
    """(1) captured banks through Fill == the model's own f(missing); (2) the same banks rebuilt from the prior
    FILES == it; (3) per-bank singles fused offline == the direct multi-bank fit; then report the model's own set."""
    zs = ZS(a.zs_cache, unseen_datasets(a), a.zs_adata); fill = Fill(a.capture, zs, a.capture_rows)
    c = np.load(a.capture, allow_pickle=True)
    names = [str(x) for x in c['bank_names']]
    ref = np.zeros((len(zs.cand), zs.Aall.shape[1]), np.float32); rok = np.zeros(len(zs.cand), bool)
    p = zs.pos[c['out_rows']]; keep = p >= 0
    ref[p[keep]] = _u(c['out'][keep]); rok[p[keep]] = True
    same = a.capture_rows is None
    if not same:
        print('[parity 1-2] skipped: foreign atoms (the model-fill reference only exists on the capture model\'s atoms)')
    P1, ok1 = (fill.predict([(n, c[f'bank{i}_E'], c[f'bank{i}_valid']) for i, n in enumerate(names)]) if same
               else (ref, rok))
    print(f'[parity 1] captured banks {names}: filled equal {np.array_equal(ok1, rok)}, max|P-model| {np.abs(P1 - ref).max():.2e}')
    assert np.array_equal(ok1, rok) and np.abs(P1 - ref).max() < 1e-4, 'PARITY 1 FAILED: harness call != model f(missing)'
    files = {n: (a.panel if n == 'text' else f'{a.prior_dir}/{n[:-4] if n.endswith(".npz") else n}.npz') for n in names}
    banks = []
    for n in names:
        g, E = load_prior(files[n], zs.rows, f'{a.work_dir}/banks')
        M, v = to_rows(g, E, zs.rows)
        banks.append((n, M, v))
    P2, ok2 = fill.predict(banks)
    print(f'[parity 2] banks from files {list(files.values())}: filled equal {np.array_equal(ok2, rok)}, '
          f'max|P-model| {np.abs(P2 - ref).max():.2e}')
    assert (not same) or (np.array_equal(ok2, rok) and np.abs(P2 - ref).max() < 1e-4), 'PARITY 2 FAILED: file-loaded banks != model banks'
    singles = [fill.predict([b]) for b in banks]
    P3, ok3 = fill.fuse([x[0] for x in singles], [x[1] for x in singles])
    print(f'[parity 3] offline fuse of {len(banks)} singles vs direct fit: filled equal {np.array_equal(ok3, ok2)}, '
          f'max|diff| {np.abs(P3 - P2).max():.2e}')
    assert np.array_equal(ok3, ok2) and np.abs(P3 - P2).max() < 1e-4, 'PARITY 3 FAILED: offline fusion != ExPert fusion'
    for lab, (P, ok) in (('model set ' + '+'.join(names) + ' [coexpr LEAKS ZS]', (P2, ok2)),
                         ('model set without coexpr', fill.fuse([x[0] for n, x in zip(names, singles) if 'coexpr' not in n],
                                                                [x[1] for n, x in zip(names, singles) if 'coexpr' not in n]))):
        s_ = zs.summary(zs.ranks(P, ok), ok)
        print(f'[ref] {lab}: ' + '  '.join(f'{g} x{s_[g]["zs_x"]:.2f} w{s_[g]["w_zs_x"]:.2f}' for g in PGROUPS + GROUPS))


def shipped_fill(a, zs):
    """The fill the model actually SHIPS: the persistent `prior_missing_T` buffer inside model.pt (absolute vocab rows,
    zero row = no prediction). Written by add_prior_missing_fill() from the model's OWN bank set -- for EFFGATE3 that is
    text + depmap_dstd + string + coexpr_v2, i.e. it includes the coexpr bank the screen drops. Falls back to the
    capture's recorded output (verified equal to the buffer) when --model-pt is not given."""
    c = np.load(a.capture, allow_pickle=True)
    rows, out = np.asarray(c['out_rows']).astype(np.int64), np.asarray(c['out'], np.float32)
    lab = 'SHIPPED f(missing) [capture]'
    if a.model_pt:
        import torch
        sd = torch.load(a.model_pt, map_location='cpu', weights_only=False)
        for k in ('model_state_dict', 'state_dict'):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
        keys = [k for k in sd if k.endswith('prior_missing_T')]
        if not keys:
            raise SystemExit(f'no prior_missing_T buffer in {a.model_pt} (keys like {list(sd)[:3]})')
        T = sd[keys[0]].float().numpy()
        filled = np.linalg.norm(T, axis=1) > 0
        d = np.abs(_u(T[rows]) - _u(out)).max()
        print(f'[shipped] {keys[0]} {T.shape}: {int(filled.sum()):,} non-zero rows '
              f'({int(filled[zs.ntr:].sum()):,} untrained); capture rows {len(rows):,}; '
              f'max|unit(buffer) - unit(capture)| {d:.2e}', flush=True)
        rows, out = np.flatnonzero(filled), T[filled]
        lab = 'SHIPPED prior_missing_T [model.pt]'
    P = np.zeros((len(zs.cand), zs.Aall.shape[1]), np.float32); ok = np.zeros(len(zs.cand), bool)
    pos = zs.pos[rows]; keep = pos >= 0
    P[pos[keep]] = _u(out[keep]); ok[pos[keep]] = True
    print(f'[shipped] covers {int(ok.sum()):,}/{len(zs.cand):,} candidates ({ok.mean():.3f})', flush=True)
    return P, ok, lab


def cmd_panels(a):
    """Score the same fused sets on BOTH panels: candidates only vs the full deployed vocabulary (trained atoms as
    distractors, what predict(obs_only=False) does). x-chance is panel-dependent; top10 is the comparable number."""
    reg = registry(a); idx = {r[0]: i for i, r in enumerate(reg)}
    sets = [(sp.split('=', 1)[0], sp.split('=', 1)[1].split('+')) for sp in a.sets] if a.sets else []
    if a.shipped_fill and a.capture_rows is not None and not a.model_pt:
        # without --model-pt the shipped fill is the CAPTURE's output, which only belongs to the capture
        # model; with --model-pt it is that checkpoint's own prior_missing_T, so any model is fine
        raise SystemExit('--shipped-fill without --model-pt is only meaningful on the capture model\'s OWN '
                         'atoms: drop --capture-rows or pass --model-pt')
    print(f'[panels] atoms/queries = {a.tag}; recipe = {a.capture}')
    for panel in ('cand', 'full'):
        zs = ZS(a.zs_cache, unseen_datasets(a), a.zs_adata, panel=panel, group_map=a.group_map)
        fill = Fill(a.capture, zs, a.capture_rows) if sets else None   # recipe only needed for sets
        print(f'\n=== panel {panel}: {zs.n_panel:,} rows, chance@10 {zs.ch10:.5f}')
        print(f'  {"set":<44}{"w_x":>7}{"x":>7}{"w_top10":>9}{"top10":>8}{"hits":>6}   (unseen ds: w_x / top10)')
        if a.shipped_fill:
            Ps, oks, lab = shipped_fill(a, zs)
            sm = zs.summary(zs.ranks(Ps, oks, query=False), oks)
            g, u = sm['pooled_all'], sm['pooled_unseen_dataset']
            print(f'  {lab:<44}{g["w_zs_x"]:>7.2f}{g["zs_x"]:>7.2f}'
                  f'{g["w_top10"]:>9.4f}{g["top10"]:>8.4f}{g["hits10"]:>6}   {u["w_zs_x"]:>6.1f} / {u["top10"]:.4f}')
            if a.shipped_json and panel == a.zs_panel:
                # PER-QUERY / PER-DATASET record of the shipped fill, plus its RANDOM twin: the same filled rows
                # permuted across candidates (same vectors, same hubness, wrong genes) -- the empirical floor
                rec = dict(label=lab, panel=panel, n_panel=int(zs.n_panel), chance10=float(zs.ch10),
                           unseen=zs.unseen, real=zs.summary(zs.ranks(Ps, oks, query=True), oks))
                rng = np.random.default_rng(0)
                rec['shuffled'] = []
                for _ in range(a.reps):
                    idx = np.flatnonzero(oks); Pr = Ps.copy(); Pr[idx] = Ps[rng.permutation(idx)]
                    rec['shuffled'].append(zs.summary(zs.ranks(Pr, oks, query=True), oks))
                os.makedirs(os.path.dirname(a.shipped_json), exist_ok=True)
                with open(a.shipped_json, 'w') as fh:
                    json.dump(rec, fh, indent=1)
                # per-QUERY top-1 (full-vocabulary row index) for an F1 view, same table as the ranks
                if panel == 'cand':
                    Az = zs.Aall[zs.cand].copy(); Az[oks] = Ps[oks]; colof = np.arange(len(zs.cand))
                else:
                    Az = zs.Aall.copy(); Az[zs.cand[oks]] = Ps[oks]; colof = zs.cand
                An = _u(Az).astype(np.float32)
                top1 = np.concatenate([np.argmax(zs.Qn[i:i + 4096] @ An.T, axis=1)
                                       for i in range(0, len(zs.Qn), 4096)])
                gm = {g: m for g, m in zs.qgroups.items() if g.startswith('ctx:')}
                np.savez_compressed(os.path.splitext(a.shipped_json)[0] + '.queries.npz',
                                    truth=colof[zs.qci], top1=top1, w=zs.qw['w'], qds=zs.qds,
                                    ctx_names=np.array(list(gm)), ctx_masks=np.stack(list(gm.values())) if gm
                                    else np.zeros((0, len(zs.qds)), bool))
                print(f'[shipped] per-dataset record -> {a.shipped_json}', flush=True)
        for name, priors in sets:
            miss = [p_ for p_ in priors if not os.path.exists(pred_path(a, idx[p_], 'real'))]
            if miss:
                print(f'  {name:<44} SKIPPED: no cached fill for {miss} (leak-flagged priors are never scored)')
                continue
            L = [load_pred(a, idx[p_], 'real') for p_ in priors]
            P, ok = fill.fuse([x[0] for x in L], [x[1] for x in L])
            sm = zs.summary(zs.ranks(P, ok, query=False), ok)
            g, u = sm['pooled_all'], sm['pooled_unseen_dataset']
            print(f'  {name:<44}{g["w_zs_x"]:>7.2f}{g["zs_x"]:>7.2f}{g["w_top10"]:>9.4f}{g["top10"]:>8.4f}'
                  f'{g["hits10"]:>6}   {u["w_zs_x"]:>6.1f} / {u["top10"]:.4f}')


def cmd_pairs(a):
    """Paired McNemar between fused sets on BOTH panels: same pooled candidates, hit = truth in top 10.

    The unpaired Poisson interval (~+-19 hits on 90) is far wider than the paired one, so 'indistinguishable' has to be
    decided on the discordant pairs, not on the hit counts."""
    from scipy.stats import binomtest
    reg = registry(a); idx = {r[0]: i for i, r in enumerate(reg)}
    sets = [(sp.split('=', 1)[0], sp.split('=', 1)[1].split('+')) for sp in a.sets]
    for panel in ('cand', 'full'):
        zs = ZS(a.zs_cache, unseen_datasets(a), a.zs_adata, panel=panel)
        fill = Fill(a.capture, zs, a.capture_rows)
        H = {}
        for name, priors in sets:
            L = [load_pred(a, idx[p_], 'real') for p_ in priors]
            P, ok = fill.fuse([x[0] for x in L], [x[1] for x in L])
            H[name] = zs.ranks(P, ok, query=False)['pooled_all'] < 10
        print(f'\n=== panel {panel}: {zs.n_panel:,} rows; hits over {len(zs.P["pooled_all"]["ci"]):,} pooled candidates')
        for name in H:
            print(f'  {name:<40} {int(H[name].sum()):>4} hits')
        print(f'  {"pair":<52}{"a-only":>7}{"b-only":>7}{"p":>9}')
        for i, x in enumerate(H):
            for y in list(H)[i + 1:]:
                ab = int((H[x] & ~H[y]).sum()); ba = int((H[y] & ~H[x]).sum())
                pv = binomtest(ab, ab + ba, 0.5).pvalue if ab + ba else 1.0
                print(f'  {x + "  vs  " + y:<52}{ab:>7}{ba:>7}{pv:>9.3f}')


def cmd_convert(a):
    pg = panel_genes(a)
    for name, _, src, _ in registry(a):
        if src.endswith('.pickle'):
            g, _E = load_prior(src, pg, f'{a.work_dir}/banks')
            print(f'[convert] {name}: {len(g):,} panel genes')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('list', 'bio', 'zs', 'combos', 'combos-merge', 'parity', 'panels', 'pairs', 'convert'))
    ap.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    # ---- paths (no defaults: set them in the sbatch file) ----
    ap.add_argument('--emb-dir', required=True, help='text embeddings root: <describer>/<encoder>/gene_embeddings.npz')
    ap.add_argument('--prior-dir', required=True, help='non-text prior banks <name>.npz')
    ap.add_argument('--genept-dir', required=True, help='GenePT_emebdding_v2 (ada/ and model3/ pickles)')
    ap.add_argument('--random-prior', required=True, help='random Gaussian gene embedding npz')
    ap.add_argument('--panel', required=True, help='npz whose genes define the 19,895-gene panel (the model text bank)')
    ap.add_argument('--out-dir', required=True, help='small outputs (JSON/TSV)')
    ap.add_argument('--work-dir', required=True, help='large caches (converted banks, cand predictions)')
    ap.add_argument('--gene-sets-dir', default=None, help='<library>.gmt files (bio)')
    ap.add_argument('--zs-cache', default=None, help='cache_zs_queries.py output (zs / combos / parity)')
    ap.add_argument('--capture', default=None, help='capture.py output: the model\'s own f(missing) fit (zs / combos / parity)')
    ap.add_argument('--split-meta', default=None, help='splits metadata.json (held-out datasets per fold)')
    ap.add_argument('--zs-adata', default=None, help='zero-shot DB h5ad (obs efficiency / perturbation / dataset for the weights)')
    ap.add_argument('--capture-rows', default=None,
                    help='zs cache the capture was taken on; set it to apply the captured recipe to the --zs-cache atoms')
    ap.add_argument('--reuse-fills', action='store_true',
                    help='zs: load cached fills from --pred-tag instead of recomputing (fills do not depend on the panel)')
    ap.add_argument('--model-pt', default=None, help="panels: model.pt whose stored prior_missing_T to score")
    ap.add_argument('--shipped-fill', action='store_true', help="panels: also score the model's stored prior_missing_T")
    ap.add_argument('--group-map', default=None,
                    help='panels: TSV dataset<TAB>label; adds ctx:<label> query groups (pooled datasets)')
    ap.add_argument('--shipped-json', default=None,
                    help='panels: write the shipped fill per query/dataset (+ --reps shuffled twins) on --zs-panel')
    ap.add_argument('--main-group', default='pooled_all', help='group that selects the combo pool')
    ap.add_argument('--metric', default='w_zs_x', choices=('w_zs_x', 'zs_x', 'wraw_zs_x'), help='main readout metric')
    ap.add_argument('--tag', default='EFFGATE3')
    ap.add_argument('--fold', type=int, default=None, help='fold whose held_out_datasets are the unseen datasets')
    ap.add_argument('--reps', type=int, default=3, help='shuffled twins per prior')
    ap.add_argument('--ks', type=int, nargs='+', default=[5, 10, 50])
    ap.add_argument('--n-sub', type=int, default=2000)
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--raw', dest='center', action='store_false', help='bio on uncentred vectors (eval.csv parity)')
    ap.add_argument('--pool', type=int, default=10)
    ap.add_argument('--max-text', type=int, default=3)
    ap.add_argument('--max-size', type=int, default=10)
    ap.add_argument('--n-chunks', type=int, default=1, help='combos: array size; --task picks the chunk')
    ap.add_argument('--sets', nargs='+', default=None, help='panels: "label=prior1+prior2" fused sets to score')
    ap.add_argument('--zs-panel', default='cand', choices=('cand', 'full'),
                    help="what the fills are ranked against: 'cand' = the untrained candidates, 'full' = the whole "
                         'deployed vocabulary incl. trained atoms (predict(obs_only=False)). Chance scales with it.')
    ap.add_argument('--missing-sim-pow', type=float, default=None,
                    help='zero-shot fill neighbour weight cos^p (shipped models: 4); default = captured recipe (1)')
    ap.add_argument('--pred-tag', default=None, help='tag whose cached fills to reuse (default: --tag); fills are panel-independent')
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    Fill.MISSING_SIM_POW = a.missing_sim_pow
    if a.cmd == 'list':
        for i, r in enumerate(registry(a)):
            print(f'{i:>3}  {r[1]:<18} {"LEAK " if r[3] else "     "}{r[0]}')
        return
    zsn = ['zs_cache', 'capture', 'split_meta', 'fold', 'zs_adata']
    need = {'bio': ['gene_sets_dir'], 'zs': zsn, 'combos': zsn, 'parity': zsn, 'panels': zsn, 'pairs': zsn}
    miss = [f'--{k.replace("_", "-")}' for k in need.get(a.cmd, []) if getattr(a, k) is None]
    if miss:
        ap.error(f'{a.cmd} needs {" ".join(miss)}')
    if a.cmd == 'panels' and not a.sets and not a.shipped_fill:
        ap.error('panels needs --sets and/or --shipped-fill')
    if a.cmd == 'pairs' and not a.sets:
        ap.error('pairs needs --sets')
    {'bio': cmd_bio, 'zs': cmd_zs, 'combos': cmd_combos, 'combos-merge': cmd_combos_merge, 'parity': cmd_parity,
     'panels': cmd_panels, 'pairs': cmd_pairs, 'convert': cmd_convert}[a.cmd](a)


if __name__ == '__main__':
    main()
