# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Fig. 3 H/I (formerly Fig. 4 L/M): frozen-regime doubles comparison against OUR OWN full K562 GWPS
fingerprint dictionary (8,347).

Both arms are frozen and neither saw these screens. Unlike the published 4,098-fingerprint dictionary,
this one covers essentially the whole screen, so the shared candidate list is no longer the binding
constraint and the curves measure RANKING rather than dictionary coverage.

  candidates   fingerprints in our dictionary that are ALSO ExPert trained perturbations
               (Supplementary Fig. S5, figS5_data.py: further restricted, per screen, to genes that occur
               in that screen's own doubles -- the ranking question without the rest of the genome)
  scorable     doubles whose BOTH genes are candidates (the head-to-head)
  all          every matched double; a gene outside the candidates is never recovered

ExPert scoring matches fig4_pb.py: signed cosine of the control-centred latent delta vs unit atoms,
loss-tail descending / gain-tail ascending (a learned prototype encodes the loss-of-function direction;
a dictionary built from the screen's own singles would NOT flip -- see project-fingerprint-beats-...).
Fingerprinting ranks by PIP descending, ties by |cor|, exactly as its own pipeline does.
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
import fig4_common as fc

KS = [1, 2, 3, 5, 10, 20, 50, 100, 500]
unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)
zc = lambda M: (M - M.mean(1, keepdims=True)) / np.maximum(M.std(1, keepdims=True), 1e-12)


def ranks_of(S, desc, ia, ib, unrankable=None):
    """(both, any): the rank at which BOTH true genes are recovered, and at which EITHER one is.

    both = max of the two ranks (the harder metric -- it is pinned by whichever gene the response
    represents worse); any = min (credit for naming one of the two).
    """
    S = S.astype(np.float64).copy()
    if unrankable is not None and unrankable.any():
        S[:, unrankable] = -np.inf if desc else np.inf
    o = np.argsort(-S, 1) if desc else np.argsort(S, 1)
    R = np.empty_like(o)
    np.put_along_axis(R, o, np.arange(1, o.shape[1] + 1)[None, :].repeat(o.shape[0], 0), axis=1)
    rr = np.arange(S.shape[0])
    ra, rb = R[rr, ia].astype(float), R[rr, ib].astype(float)
    return np.maximum(ra, rb), np.minimum(ra, rb)


SUBSETS = {'all': lambda x: x,
           'scorable': lambda x: x[x.scorable],
           # in-panel AND scorable: the screen measured both genes alone, and both are candidates
           'in_panel': lambda x: x[x.scorable & x.in_panel]}


def score(a, cand_space='full') -> tuple[pd.DataFrame, dict]:
    """Per double: the rank at which each arm recovers both / either gene, and per-screen counts.
    cand_space 'full' = every shared candidate (Fig. 3 H/I); 'matched' = only genes that occur in the screen's
    own doubles (Supplementary Fig. S4)."""
    pairs_all = pd.read_csv(a.pair_ranks)

    per, meta = [], {}
    for spec in a.screens.split(','):
        s, sub, tail, mod = spec.split(':')
        e = Path(a.encode_root) / s
        pip = pd.read_csv(Path(a.fp_root) / sub / 'pip.csv', index_col=0)
        cor = pd.read_csv(Path(a.fp_root) / sub / 'cor.csv', index_col=0)
        obs = pd.read_csv(e / 'obs.csv', keep_default_na=False)
        b = np.load(e / 'bank.npz', allow_pickle=True)
        P, L, nl = b['P'].astype(np.float64), b['labels'].astype(str), int(b['n_labels'])
        lab_pos = {g: i for i, g in enumerate(L[:nl])}
        ctrl = int(np.flatnonzero(obs.condition == fc.CONTROL)[0])

        cands = [g for g in pip.columns if g in lab_pos]
        cpos_q = {g: j for j, g in enumerate(cands)}
        pr = pairs_all[pairs_all.screen == sub]
        gmap = {r.double: (r.gene_a, r.gene_b) for r in pr.itertuples()}

        q = obs.index[obs.order >= 1].to_numpy()
        tg = [t.split(';') for t in obs.targets[q]]
        order = obs.order[q].to_numpy()
        by_pair = {frozenset(t): i for i, t, o in zip(q, tg, order) if o == 2 and len(t) == 2}
        # the screen's OWN panel: genes it also measured as a single perturbation
        panel = {tuple(t)[0] for t, o in zip(tg, order) if o == 1}

        rows, dbl, ga, gb = [], [], [], []
        for d in pip.index:
            if d not in gmap:
                continue
            k = frozenset(gmap[d])
            if k not in by_pair:
                continue
            rows.append(by_pair[k]); dbl.append(d)
            x, y = sorted(gmap[d])
            ga.append(x); gb.append(y)
        if not rows:
            print(f'[{s}] no matched doubles -- skipped', flush=True)
            continue
        if cand_space == 'matched':
            inds = set(ga) | set(gb)
            cands = [g for g in cands if g in inds]
            cpos_q = {g: j for j, g in enumerate(cands)}
        rows = np.array(rows)
        N = len(cands)
        ia = np.array([cpos_q.get(g, -1) for g in ga])
        ib = np.array([cpos_q.get(g, -1) for g in gb])
        off = (ia < 0) | (ib < 0)
        ia_, ib_ = np.where(ia < 0, 0, ia), np.where(ib < 0, 0, ib)
        desc = tail == 'loss'

        # ---- fingerprinting: PIP desc, ties by |cor| ----------------------------------------------
        pv = pip.loc[dbl, cands].to_numpy(float)
        cv = cor.loc[dbl, cands].to_numpy(float)
        o = np.lexsort((-cv, -pv), axis=1)
        Rf = np.empty_like(o)
        np.put_along_axis(Rf, o, np.arange(1, o.shape[1] + 1)[None, :].repeat(o.shape[0], 0), axis=1)
        rr = np.arange(len(rows))
        fa, fb = Rf[rr, ia_].astype(float), Rf[rr, ib_].astype(float)
        r_fp, r_fp_any = np.maximum(fa, fb), np.minimum(fa, fb)

        # ---- ExPert ---------------------------------------------------------------------------------
        Z = np.load(e / 'z_mtmask.npy').astype(np.float64)
        D = Z[rows] - Z[ctrl]
        Pe = unit(P[np.array([lab_pos[g] for g in cands])])
        r_ex, r_ex_any = ranks_of(unit(D) @ Pe.T, desc, ia_, ib_)

        # ---- training centroids ---------------------------------------------------------------------
        Lg = np.load(e / 'logpb.npy').astype(np.float64)
        genes = np.load(e / 'genes.npy', allow_pickle=True).astype(str)
        tcols = np.load(e / 'target_cols.npy')
        gpos = {g: i for i, g in enumerate(genes)}
        tset = set(genes[tcols])
        z = np.load(a.centroid_ref, allow_pickle=True)
        Rc, lab_c, g_c = z['D'].astype(np.float64), z['labels'].astype(str), z['genes'].astype(str)
        cpos = {g: i for i, g in enumerate(lab_c)}
        use = [(i, gpos[g]) for i, g in enumerate(g_c) if g in gpos and g not in tset]
        ri = np.array([u[0] for u in use]); qi = np.array([u[1] for u in use])
        have = np.array([g in cpos for g in cands])
        Rp = np.zeros((N, len(ri)))
        Rp[have] = Rc[np.ix_([cpos[g] for g in np.array(cands)[have]], ri)]
        Dq = Lg[rows][:, qi] - Lg[ctrl, qi]
        r_ct, r_ct_any = ranks_of(zc(Dq) @ zc(Rp).T / Dq.shape[1], desc, ia_, ib_, unrankable=~have)

        for v in (r_fp, r_ex, r_ct, r_fp_any, r_ex_any, r_ct_any):
            v[off] = np.inf
        d = pd.DataFrame({'screen': s, 'modality': mod, 'condition': obs.condition[rows].to_numpy(),
                          'gene_a': ga, 'gene_b': gb, 'scorable': ~off, 'n_cand': N,
                          'in_panel': [x in panel and y in panel for x, y in zip(ga, gb)],
                          'ExPert': r_ex, 'Fingerprint (our GWPS)': r_fp, 'Centroids': r_ct,
                          'ExPert|any': r_ex_any, 'Fingerprint (our GWPS)|any': r_fp_any,
                          'Centroids|any': r_ct_any})
        per.append(d)
        meta[s] = dict(n_cand=N, n_dict=int(pip.shape[1]), n_doubles=len(d),
                       n_scorable=int((~off).sum()),
                       n_in_panel=int((d.scorable & d.in_panel).sum()),
                       n_centroid=int(have.sum()), modality=mod)
        print(f'[{s}] {len(d)} doubles ({int((~off).sum())} scorable), {N} candidates '
              f'of {pip.shape[1]} fingerprints', flush=True)

    return pd.concat(per, ignore_index=True), meta


def curves(D: pd.DataFrame) -> pd.DataFrame:
    """Recovered fraction at each k, per subset x (screen | modality) x arm x metric (both / any)."""
    rows = []
    for denom, sel in SUBSETS.items():
        src = sel(D)
        for key, g in list(src.groupby('screen')) + list(src.groupby('modality')):
            if not len(g):
                continue
            nr = g.n_cand.to_numpy(float)             # per double: screens differ in matched space
            for k in KS:
                kr = np.minimum(k, nr)
                for m in ('ExPert', 'Fingerprint (our GWPS)', 'Centroids'):
                    rows.append(dict(denom=denom, group=key, method=m, metric='both', k=k,
                                     both=float((g[m] <= k).mean()), n=len(g)))
                    rows.append(dict(denom=denom, group=key, method=m, metric='any', k=k,
                                     both=float((g[m + '|any'] <= k).mean()), n=len(g)))
                # random: both = k(k-1)/(N(N-1)); any = 1 - C(N-2,k)/C(N,k) = 1 - (N-k)(N-k-1)/(N(N-1))
                rows.append(dict(denom=denom, group=key, method='Random', metric='both', k=k,
                                 both=float(np.mean(kr * (kr - 1) / (nr * (nr - 1)))), n=len(g)))
                rows.append(dict(denom=denom, group=key, method='Random', metric='any', k=k,
                                 both=float(np.mean(1 - (nr - kr) * (nr - kr - 1) / (nr * (nr - 1)))),
                                 n=len(g)))
    return pd.DataFrame(rows)


def report(C: pd.DataFrame):
    pd.set_option('display.width', 250)
    for denom in SUBSETS:
        print(f'\n########## both@k, subset = {denom} ##########')
        for met in ('both', 'any'):
            p = C[(C.denom == denom) & (C.metric == met) & (C.k.isin([2, 5, 10, 50, 100]))]
            print(f'-- metric = {met}@k --')
            print(p.pivot_table(index=['group', 'k'], columns='method', values='both').round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encode-root', required=True)
    ap.add_argument('--fp-root', required=True, help='our mapping output (per-screen pip.csv / cor.csv)')
    ap.add_argument('--pair-ranks', required=True, help="published gwps_ranks.csv, for each double's gene pair")
    ap.add_argument('--centroid-ref', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--screens', required=True, help='screen:fp_subdir:tail:modality, comma separated')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    D, meta = score(a)
    D.to_csv(out / 'ourgwps_doubles_ranks.csv', index=False)
    json.dump(meta, open(out / 'ourgwps_meta.json', 'w'), indent=1)
    C = curves(D)
    C.to_csv(out / 'ourgwps_doubles_curves.csv', index=False)
    report(C)


if __name__ == '__main__':
    main()
