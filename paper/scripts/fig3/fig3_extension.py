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

"""Fig. 3 text claim: does extending the vocabulary with filled prototypes cost the SCREENED ones?

Panel C measures the PULL (trained prototypes moved toward their graph targets) and ranks every query
against the trained block only, so it says nothing about the EXTENSION: ~10k filled rows joining the
candidate set as plausible distractors. This tests that directly, on held-out queries of TRAINED
perturbations (fold 4, K562 held out; the 10,061 queries of panel C):

    trained     rank of the true prototype among the 8,964 trained ones (reproduces panel C's ranks)
    + fills     the same queries ranked against trained + filled rows for the 9,874 zero-shot genes a
                screen measured (the set of panels A/B). The fill is built for EVERY arm by the same
                graph map (k-NN in each prior bank -> similarity-weighted mean of the arm's own unit
                trained atoms, banks fused) from one shared bank set, so arms differ only in training
    + shipped   EFFGATE3's persistent prior_missing_T, on the same 9,874 rows and on all 10,931
    + random    random unit rows of the same count: a distractor with no plausibility (reference)

    cost        lost top-10 hits (trained-only hit, extended miss), net delta with a paired
                bootstrap CI over queries, and the share of queries with any filled row in top 10

Arms: EFFGATE3 (pull + fill negatives), BASE (no pull; negatives per config), H4 (no prior machinery,
never saw a filled row). Whether an arm's negatives were ACTIVE is read from its training log
(weights/prior_neg), not from the config: before the 2026-09-13 fix the negatives were silently inert
whenever the pull weight was 0.

Writes <out>/fig3_extension.csv (one row per arm x candidate set) and fig3_extension_meta.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from expert.analysis.atom_umap import load_atoms
from expert.modules._prior_atoms import PriorAtomPredictor


def unit(M):
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def ckpt_banks(sd):
    """[(name, E, valid)] over ALL vocabulary rows from a checkpoint's prior buffers (text first)."""
    key = lambda suf: next((k for k in sd if k.endswith('cls_emb.' + suf)), None)
    out = [('text', sd[key('_atom_text')].float(), None)]
    i = 0
    while key(f'_atom_prior_{i}') is not None:
        v = key(f'_atom_prior_valid_{i}')
        out.append((f'prior{i}', sd[key(f'_atom_prior_{i}')].float(), None if v is None else sd[v].bool()))
        i += 1
    return out


def graph_fill(A, banks, rows, k, sim_pow):
    """Unit fills for vocabulary `rows` from trained atoms A (n, d): the shipped map, all trained atoms
    as the neighbour pool; neighbour weight = prior cosine^sim_pow (the model's prior_atom_missing_sim_pow)."""
    n, d = A.shape
    nv = banks[0][1].shape[0]
    At = torch.zeros(nv, d)
    At[:n] = torch.from_numpy(A).float()
    tr = torch.arange(n)
    f = PriorAtomPredictor(map='graph', missing_map='graph', graph_k=k, graph_loo=True,
                           cos_floor=-1.0, min_fit=1, missing_sim_pow=sim_pow)
    f.fit(At, banks, tr, tr, missing=torch.as_tensor(np.sort(rows)), n_trained=n)
    got = f.missing_rows.numpy()
    out = np.zeros((len(rows), d), np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    for r, v in zip(got, f.missing.numpy()):
        out[pos[int(r)]] = v
    ok = np.isin(rows, got)
    return unit(out), ok


def ranks(Qn, truth, D, chunk=2000):
    """0-based rank of the true row (strictly-greater count), and whether any EXTRA row (index >= n_tr)
    sits in the top 10."""
    r = np.empty(len(truth), np.int64)
    for s in range(0, len(truth), chunk):
        S = Qn[s:s + chunk] @ D.T
        t = S[np.arange(S.shape[0]), truth[s:s + chunk]]
        r[s:s + chunk] = (S > t[:, None]).sum(1)
    return r


def intrusion(Qn, D, n_tr, chunk=2000):
    hit = np.zeros(len(Qn), bool)
    for s in range(0, len(Qn), chunk):
        S = Qn[s:s + chunk] @ D.T
        top = np.argpartition(-S, 10, axis=1)[:, :10]
        hit[s:s + chunk] = (top >= n_tr).any(1)
    return hit


def neg_active(model_pt):
    """Was the fill-negative loss applied in training? Read from the run's tfevents."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = Path(model_pt).parent.parent
    ev = sorted(glob.glob(str(run / 'events.out.tfevents.*')))
    if not ev:
        return {'log': None}
    acc = EventAccumulator(ev[-1], size_guidance={'scalars': 0}); acc.Reload()
    tags = [t for t in acc.Tags().get('scalars', []) if 'prior_neg' in t or 'prior_atom/neg' in t]
    out = {'log': os.path.basename(ev[-1]), 'tags': tags}
    for t in tags:
        v = [e.value for e in acc.Scalars(t)]
        out[t] = {'n': len(v), 'max': float(np.max(v)) if v else None, 'last': float(v[-1]) if v else None}
    return out


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    zq = np.load(a.zs_cache, allow_pickle=True)
    zrows = np.unique(np.asarray(zq['truth'], np.int64))
    ref = pd.read_csv(a.ref_ranks, sep='\t').set_index('key')
    sd_b = torch.load(a.banks_from, map_location='cpu', weights_only=False)['model_state_dict']
    banks = ckpt_banks(sd_b)
    print(f'[banks] from {a.banks_from}: ' + ', '.join(
        f'{nm} {tuple(E.shape)}' for nm, E, _ in banks), flush=True)
    rng = np.random.default_rng(0)
    rows, meta = [], {'zs_rows': int(len(zrows)), 'k': a.k, 'sim_pow': a.sim_pow, 'n_boot': a.n_boot, 'arms': {}}
    for spec in a.arm:
        name, model_pt, cache = spec.split('=', 2)
        labels, A, note = load_atoms(model_pt)
        c = np.load(cache, allow_pickle=True)
        assert np.array_equal(np.asarray(c['labels']).astype(str), np.asarray(labels).astype(str)), name
        n = len(labels)
        Qn = unit(np.asarray(c['Q'], np.float64)).astype(np.float32)
        truth = np.asarray(c['truth'], np.int64)
        keys = np.asarray(c['keys']).astype(str)
        An = unit(A).astype(np.float32)
        r0 = ranks(Qn, truth, An)
        col = f'rank_{name}'
        match = (float(np.mean(ref.loc[keys, col].to_numpy() == r0)) if col in ref else None)
        print(f'[{name}] {len(keys):,} queries; trained-only top10 {int((r0 < 10).sum())}; '
              f'matches panel C ranks: {match}', flush=True)
        sets = {}
        F, ok = graph_fill(A, banks, zrows, a.k, a.sim_pow)
        sets['+ fills (graph map, own atoms)'] = F[ok]
        sets['+ random unit rows'] = unit(rng.standard_normal((int(ok.sum()), A.shape[1]))).astype(np.float32)
        sd = torch.load(model_pt, map_location='cpu', weights_only=False)['model_state_dict']
        kf = next((k for k in sd if k.endswith('prior_missing_T')), None)
        if kf is not None:
            T = sd[kf].float().numpy()
            live = np.flatnonzero(np.linalg.norm(T[n:], axis=1) > 0) + n
            sets['+ shipped fill (9,874 measured)'] = unit(T[np.intersect1d(live, zrows)])
            sets[f'+ shipped fill (all {len(live):,})'] = unit(T[live])
        neg = neg_active(model_pt)
        meta['arms'][name] = {'model_pt': model_pt, 'cache': cache, 'atom_note': note,
                              'match_panel_c': match, 'neg_log': neg}
        print(f'[{name}] negatives in training log: {neg}', flush=True)
        h0 = r0 < 10
        rows.append({'arm': name, 'candidates': 'trained only', 'n_candidates': n,
                     'top10': int(h0.sum()), 'delta': 0, 'lo': 0, 'hi': 0, 'lost': 0, 'gained': 0,
                     'top1': int((r0 == 0).sum()), 'rank_median': float(np.median(r0)),
                     'intrusion_frac': 0.0})
        for lab, E in sets.items():
            D = np.concatenate([An, E])
            r1 = ranks(Qn, truth, D)
            h1 = r1 < 10
            d = h1.astype(int) - h0.astype(int)
            bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(a.n_boot)])
            rows.append({'arm': name, 'candidates': lab, 'n_candidates': len(D),
                         'top10': int(h1.sum()), 'delta': int(d.sum()),
                         'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)),
                         'lost': int((h0 & ~h1).sum()), 'gained': int((~h0 & h1).sum()),
                         'top1': int((r1 == 0).sum()), 'rank_median': float(np.median(r1)),
                         'intrusion_frac': float(intrusion(Qn, D, n).mean())})
            print(f'  {lab:<34} top10 {int(h1.sum()):5d}  delta {int(d.sum()):+5d} '
                  f'[{rows[-1]["lo"]:+.0f},{rows[-1]["hi"]:+.0f}]  lost {rows[-1]["lost"]}  '
                  f'intrusion {rows[-1]["intrusion_frac"]:.3f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'fig3_extension.csv', index=False)
    json.dump(meta, open(out / 'fig3_extension_meta.json', 'w'), indent=1)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', action='append', required=True,
                    help='NAME=model.pt=ood_query_cache.npz (repeatable)')
    ap.add_argument('--banks-from', required=True, help='model.pt whose prior banks build every fill')
    ap.add_argument('--zs-cache', required=True, help='zero-shot query cache: the 9,874 measured genes')
    ap.add_argument('--ref-ranks', required=True, help='effgate_decomp per_query.tsv.gz (panel C ranks)')
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--sim-pow', type=float, required=True,
                    help='zero-shot neighbour weight = cos^p (shipped: 4 since 2026-10-23; 1 = the earlier cos fill)')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
