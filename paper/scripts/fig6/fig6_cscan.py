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

"""Fig. 6C scan: is the mitochondrial pathway signal monocyte-specific under every reasonable encoding and scoring?

One task per encode (fig6_grid.py encodes: model x input mask x encoder baseline). Per encode, grouping, bank and cell
type, the donor shifts (each case vs the pooled non-cases; a non-case vs the other non-cases) are scored on the Fig. 6C
pathway rows five ways:
    setz     full ranking, mean percentile of the set vs random sets (Fig. 6C as shipped)
    gsea     full ranking, weighted running-sum enrichment score (classic preranked GSEA ES, weight = cosine)
    topk     per donor, the set's hits in its own top-k (loss tail)
    union    the cases' top-k lists unioned (Fig. 5 recipe); statistic = hits in the union
    pooled   cell-pooled cases minus cell-pooled non-cases (2026-09-14 artifact contrast), set z with the analytic
             random-gene-set null -- a GENE-set p, not a donor test, so it cannot see donor disagreement
setz / gsea / topk: exact donor-label permutation of (mean cases - mean non-cases); union: permutation of the union
statistic; pooled: normal p of the set z. Output: cscan__<encode>.csv.gz.

    PYTHONPATH=paper/scripts python paper/scripts/fig6/fig6_cscan.py --task 0 --grid-dir ... --out-dir ...
"""
from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path += [str(p) for p in Path(__file__).resolve().parents[1].iterdir()  # paper/scripts/<figure>/ folders
             if p.is_dir() and p.name not in ('archive', 'gene_panel_src')]
from fig6_pmd import CTRL, load_sets, shifts, unit

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(name)s | %(message)s', level=logging.WARNING, force=True)
log = logging.getLogger('fig6_cscan')
log.setLevel(logging.INFO)

GROUPINGS = {'revised': ['PT43', 'PT46', 'PT54', 'PT67'], 'orig': ['PT43', 'PT46', 'PT54']}
ROWS = ['module:complex_i', 'module:complex_iii', 'module:complex_iv', 'module:complex_v',
        'reactome:Mitochondrial Translation', 'reactome:Mitochondrial Protein Import',
        'reactome:Mitochondrial Iron-Sulfur Cluster Biogenesis', 'reactome:Cristae Formation', 'module:isr',
        'reactome:Detoxification of Reactive Oxygen Species', 'module:cap_translation',
        'reactome:mTORC1-mediated Signalling', 'reactome:Macroautophagy']


def set_z(S, idx):
    """Mean percentile of the set's loss-tail rank vs random sets (analytic, finite population)."""
    N = S.shape[-1]
    pct = 1 - (-S).argsort(-1).argsort(-1) / (N - 1)          # 1 = top of the loss tail
    k = len(idx)
    var = pct.var(-1)
    return (pct[..., idx].mean(-1) - pct.mean(-1)) / np.sqrt(var / k * (N - k) / (N - 1))


def gsea_es(S, idx):
    """Classic preranked GSEA enrichment score (weight = |score|), signed max deviation of the running sum."""
    order = np.argsort(-S, kind='stable')
    s = S[order]
    hit = np.isin(order, idx)
    w = np.abs(s) * hit
    up = np.cumsum(w) / max(w.sum(), 1e-12)
    down = np.cumsum(~hit) / max((~hit).sum(), 1)
    run = up - down
    return run[np.argmax(np.abs(run))]


def perm_p(vals, cases):
    """One-sided exact p for mean(cases) > mean(others) over every relabelling with the same number of cases."""
    ds = list(vals)
    k = sum(d in cases for d in ds)
    x = np.array([vals[d] for d in ds])
    obs = x[[d in cases for d in ds]].mean() - x[[d not in cases for d in ds]].mean()
    null = np.array([x[list(c)].mean() - np.delete(x, list(c)).mean()
                     for c in itertools.combinations(range(len(ds)), k)])
    return obs, float(np.mean(null >= obs - 1e-12)), len(null)


def main(a):
    from scipy.stats import norm
    encs = sorted(p.name for p in Path(a.grid_dir).iterdir() if (p / 'z_cells.npy').exists())
    enc = encs[a.task]
    src = Path(a.grid_dir) / enc
    model = enc.split('__')[0]
    bank_npz = dict(zip(a.models.split(','), a.banks.split(',')))[model]
    Z = np.load(src / 'z_cells.npy').astype(np.float64)
    obs = pd.read_csv(src / 'obs_cells.csv.gz')
    b = np.load(bank_npz, allow_pickle=True)
    rows = []
    for gname, cases in GROUPINGS.items():
        import fig6_pmd
        fig6_pmd.PMD[:] = cases                               # shifts(): cases vs pooled non-cases (CTRL + PT67)
        ctrl = [d for d in ['PT38', 'PT50', 'PT55', 'PT41', 'PT42', 'PT67'] if d not in cases]
        CTRL[:] = ctrl
        SH, ncell = shifts(Z, obs, a.min_cells)
        for bank in ('trained', 'full'):
            keep = np.ones(len(b['labels']), bool) if bank == 'full' else b['trained'].astype(bool)
            labels = np.asarray([str(l) for l in b['labels'][keep]])
            Pn = unit(b['P'][keep].astype(np.float64))
            sets = load_sets(a, labels)
            pos = {l: i for i, l in enumerate(labels)}
            idx = {k: np.array([pos[g] for g in sets[tuple(k.split(':', 1))]]) for k in ROWS
                   if tuple(k.split(':', 1)) in sets}
            for t in sorted(obs.celltype.unique()):
                keys = [k for k in SH if k[0] == 'ctrl' and k[2] == t]
                ds = [k[1] for k in keys]
                if sum(d in cases for d in ds) < 2 or sum(d not in cases for d in ds) < 2:
                    continue
                S = unit(np.stack([SH[k] for k in keys])) @ Pn.T
                top = np.argsort(-S, axis=1)[:, :a.top_k]
                # artifact contrast: cell-pooled cases minus cell-pooled non-cases (from the raw centroids)
                mt = (obs.celltype == t).to_numpy()
                cen = {d: Z[mt & (obs.donor == d).to_numpy()] for d in ds}
                pool = lambda mem: np.concatenate([cen[d] for d in mem]).mean(0)
                Sp = unit((pool([d for d in ds if d in cases]) - pool([d for d in ds if d not in cases]))[None]) @ Pn.T
                for key, ix in idx.items():
                    out = {}
                    out['setz'] = perm_p(dict(zip(ds, set_z(S, ix))), cases)
                    out['gsea'] = perm_p(dict(zip(ds, [gsea_es(S[i], ix) for i in range(len(ds))])), cases)
                    out['topk'] = perm_p(dict(zip(ds, np.isin(top, ix).sum(1).astype(float))), cases)
                    hits = lambda mem: len(set(np.concatenate([top[ds.index(d)] for d in mem])) & set(ix))
                    obs_h = hits([d for d in ds if d in cases])
                    null = [hits([ds[i] for i in c]) for c in itertools.combinations(range(len(ds)),
                                                                                    sum(d in cases for d in ds))]
                    out['union'] = (obs_h - np.mean(null), float(np.mean(np.array(null) >= obs_h)), len(null))
                    zp = float(set_z(Sp[0], ix))
                    out['pooled'] = (zp, float(norm.sf(zp)), 0)
                    for sc, (eff, p, n) in out.items():
                        rows.append(dict(encode=enc, grouping=gname, bank=bank, celltype=t, set=key, scoring=sc,
                                         effect=eff, p=p, n_perm=n, n_cases=sum(d in cases for d in ds),
                                         n_ref=sum(d not in cases for d in ds)))
            log.info('%s %s %s done', enc, gname, bank)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f'cscan__{enc}.csv.gz', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', type=int, required=True, help='encode index (sorted fig6_grid encode dirs)')
    ap.add_argument('--grid-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--models', default='v0.1.0,EFFGATE3')
    ap.add_argument('--banks', required=True, help='full banks aligned with --models')
    ap.add_argument('--modules', required=True)
    ap.add_argument('--reactome-gmt', required=True)
    ap.add_argument('--min-cells', type=int, default=100)
    ap.add_argument('--top-k', type=int, default=200)
    main(ap.parse_args())
