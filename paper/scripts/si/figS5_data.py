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

"""Supplementary Fig. S5 plotting data (extended doubles recovery, related to Fig. 3): Fig. 3 H/I's frozen
doubles comparison (ExPert / our full-GWPS fingerprinting / training centroids / random, same doubles), with the
candidates restricted per screen to the genes that occur in that screen's own doubles -- the ranking question
without the rest of the genome. Scoring is fig3_doubles.py's (score / curves), only the candidate space differs.
Panel A (and the aux loss-of-function reference): the same arms on the screens' SINGLE perturbations, ranked against Fig. 3 H/I's full candidate set
(fingerprints that are also ExPert trained perturbations). CRISPRa singles are themselves a transfer for ExPert,
which was trained on loss-of-function screens only; the CRISPRi / Cas13d singles are the in-modality reference.
Writes to --out-dir (paper/figures/si/figS5/data):

    figS5_doubles_ranks.csv     per double: rank at which each arm recovers both / either gene; scorable, in-panel
    figS5_doubles_curves.csv    recovered fraction at each k, per subset x (screen | modality) x arm x metric
    figS5_meta.json             per screen: candidates, dictionary size, doubles, scorable, in-panel, modality
    figS5_singles_ranks.csv     per single perturbation: rank of its target under each arm (inf = not a candidate)
    figS5_singles_curves.csv    target recovered at top k, per (screen | modality) x arm, scorable singles
    figS5_singles_meta.json     per screen: candidates, singles, scorable singles, modality
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
import fig3_doubles as fd  # noqa: E402
import fig4_common as fc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ARMS = ('ExPert', 'Fingerprint (our GWPS)', 'Centroids')
# the fingerprint exports keep the screens' original (outdated) symbols for two singles
FP_ALIAS = {'KIAA1804': 'MAP3K21', 'VPRBP': 'DCAF1'}


def rank_of(S, desc, it):
    """Rank (1 = best) of column it[i] in row i; ties broken against the target."""
    t = S[np.arange(len(S)), it][:, None]
    return ((S > t) if desc else (S < t)).sum(1) + 1 + ((S == t).sum(1) - 1)


def score_singles(a) -> tuple[pd.DataFrame, dict]:
    """Per single perturbation: the rank of its target among Fig. 3 H/I's candidates under each arm."""
    z = np.load(a.centroid_ref, allow_pickle=True)
    Rc, lab_c, g_c = z['D'].astype(np.float64), z['labels'].astype(str), z['genes'].astype(str)
    cpos = {g: i for i, g in enumerate(lab_c)}
    per, meta = [], {}
    for spec in a.screens.split(','):
        s, sub, tail, mod = spec.split(':')
        e = Path(a.encode_root) / s
        f = Path(a.fp_singles_root) / sub
        pip = pd.read_csv(f / 'pip.csv', index_col=0)
        cor = pd.read_csv(f / 'cor.csv', index_col=0)
        pip.index = cor.index = [FP_ALIAS.get(c, c) for c in pip.index]
        obs = pd.read_csv(e / 'obs.csv', keep_default_na=False)
        b = np.load(e / 'bank.npz', allow_pickle=True)
        P, L, nl = b['P'].astype(np.float64), b['labels'].astype(str), int(b['n_labels'])
        lab_pos = {g: i for i, g in enumerate(L[:nl])}
        ctrl = int(np.flatnonzero(obs.condition == fc.CONTROL)[0])
        cands = [g for g in pip.columns if g in lab_pos]
        cp = {g: j for j, g in enumerate(cands)}
        N = len(cands)
        q = obs.index[(obs.order == 1) & obs.condition.isin(pip.index)].to_numpy()
        gene = obs.targets[q].to_numpy()
        it = np.array([cp.get(g, -1) for g in gene])
        off = it < 0
        it_ = np.where(off, 0, it)
        desc = tail == 'loss'

        pv = pip.loc[obs.condition[q], cands].to_numpy(float)
        cv = cor.loc[obs.condition[q], cands].to_numpy(float)
        o = np.lexsort((-cv, -pv), axis=1)
        Rf = np.empty_like(o)
        np.put_along_axis(Rf, o, np.arange(1, N + 1)[None, :].repeat(len(q), 0), axis=1)
        r_fp = Rf[np.arange(len(q)), it_].astype(float)

        Z = np.load(e / 'z_mtmask.npy').astype(np.float64)
        Pe = fd.unit(P[np.array([lab_pos[g] for g in cands])])
        r_ex = rank_of(fd.unit(Z[q] - Z[ctrl]) @ Pe.T, desc, it_).astype(float)

        Lg = np.load(e / 'logpb.npy').astype(np.float64)
        genes = np.load(e / 'genes.npy', allow_pickle=True).astype(str)
        tset = set(genes[np.load(e / 'target_cols.npy')])
        gpos = {g: i for i, g in enumerate(genes)}
        use = [(i, gpos[g]) for i, g in enumerate(g_c) if g in gpos and g not in tset]
        ri = np.array([u[0] for u in use]); qi = np.array([u[1] for u in use])
        have = np.array([g in cpos for g in cands])
        Rp = np.zeros((N, len(ri)))
        Rp[have] = Rc[np.ix_([cpos[g] for g in np.array(cands)[have]], ri)]
        Dq = Lg[q][:, qi] - Lg[ctrl, qi]
        Sc = fd.zc(Dq) @ fd.zc(Rp).T / Dq.shape[1]
        Sc[:, ~have] = -np.inf if desc else np.inf
        r_ct = rank_of(Sc, desc, it_).astype(float)

        for v in (r_ex, r_fp, r_ct):
            v[off] = np.inf
        per.append(pd.DataFrame({'screen': s, 'modality': mod, 'condition': obs.condition[q].to_numpy(),
                                 'gene': gene, 'n_cells': obs.n_cells[q].to_numpy(), 'scorable': ~off,
                                 'n_cand': N, 'ExPert': r_ex, 'Fingerprint (our GWPS)': r_fp,
                                 'Centroids': r_ct}))
        meta[s] = dict(n_cand=N, n_singles=len(q), n_scorable=int((~off).sum()), modality=mod)
        print(f'[{s}] {len(q)} singles ({int((~off).sum())} scorable), {N} candidates', flush=True)
    return pd.concat(per, ignore_index=True), meta


def singles_curves(D: pd.DataFrame) -> pd.DataFrame:
    """Target recovered at each k (scorable singles), in fig3_doubles.curves' schema (metric 'single')."""
    rows = []
    src = D[D.scorable]
    for key, g in list(src.groupby('screen')) + list(src.groupby('modality')):
        nr = g.n_cand.to_numpy(float)
        for k in fd.KS:
            for m in ARMS:
                rows.append(dict(denom='scorable', group=key, method=m, metric='single', k=k,
                                 both=float((g[m] <= k).mean()), n=len(g)))
            rows.append(dict(denom='scorable', group=key, method='Random', metric='single', k=k,
                             both=float(np.mean(np.minimum(k, nr) / nr)), n=len(g)))
    return pd.DataFrame(rows)


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    D, meta = fd.score(a, cand_space='matched')
    D.to_csv(out / 'figS5_doubles_ranks.csv', index=False)
    json.dump(meta, open(out / 'figS5_meta.json', 'w'), indent=1)
    C = fd.curves(D)
    C.to_csv(out / 'figS5_doubles_curves.csv', index=False)
    fd.report(C)
    S, smeta = score_singles(a)
    S.to_csv(out / 'figS5_singles_ranks.csv', index=False)
    json.dump(smeta, open(out / 'figS5_singles_meta.json', 'w'), indent=1)
    Cs = singles_curves(S)
    Cs.to_csv(out / 'figS5_singles_curves.csv', index=False)
    print(Cs[Cs.k.isin([1, 10, 50, 500])].pivot_table(index=['group', 'k'], columns='method', values='both')
          .round(4).to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encode-root', required=True, help='per-screen encodings (fig4 data pb/<model>)')
    ap.add_argument('--fp-root', required=True, help='our mapping output (per-screen pip.csv / cor.csv)')
    ap.add_argument('--fp-singles-root', required=True, help='our mapping of the screens\' singles (pip.csv / cor.csv)')
    ap.add_argument('--pair-ranks', required=True, help="published gwps_ranks.csv, for each double's gene pair")
    ap.add_argument('--centroid-ref', required=True)
    ap.add_argument('--screens', required=True, help='screen:fp_subdir:tail:modality, comma separated')
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
