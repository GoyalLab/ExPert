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

"""Shared pieces of the Fig. 4 (combinatorial decoding) scripts: screen truth, masking, rankings,
and rank metrics with a permutation null.

Truth parsing is carried over from the combinatorics harness (decode_combi.py) so the scored
conditions match the numbers already on record. The decoder is `expert.utils.readout.nnomp`, the
same non-negative OMP the package ships, run with the stopping rule OFF so a pursuit becomes a
RANKING: greedy OMP is prefix-consistent, so the first K picks of a depth-20 run are exactly a
k=K run.

    ranks       for every condition and dictionary, the 1-based position of each true gene in the
                ranking (inf when it is never selected -- a pursuit stops once no candidate has a
                positive correlation with the residual)
    metrics     recall@k (mean over true genes), any@k, both@k (all true genes within k; for a
                single this equals recall@k) and hit1 (the first pick is a true gene)
    null        truth permuted across conditions OF THE SAME ORDER with the rankings held fixed:
                keeps each decoder's popularity bias (a hub atom picked for everything), which an
                analytic 1/n chance would miss
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL = '__control__'
NULL_TOKENS = 'ctrl,control,negative,negctrl,ntc,non-targeting,nontargeting,none,na,nan,'


# ---- truth ---------------------------------------------------------------------------------
def read_alias(path: str | None) -> dict[str, str]:
    """old<TAB>new symbol map ('#' comments), applied to truth tokens before vocabulary lookup."""
    if not path:
        return {}
    out = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        old, new = line.rstrip('\n').split('\t')[:2]
        out[old.strip()] = new.strip()
    return out


def parse_targets(tokens, null_tokens: set, strip_re, alias: dict) -> tuple[str, ...]:
    """Label tokens -> the UNIQUE sorted set of genes targeted (empty == control).

    Guide suffixes are stripped first, so a 'double' of two guides against one gene scores as the
    single it is."""
    out = []
    for t in tokens:
        t = str(t).strip()
        if not t or t.lower() in null_tokens:
            continue
        if strip_re is not None:
            t = strip_re.sub('', t)
        t = alias.get(t, t)
        if t and t.lower() not in null_tokens and t not in out:
            out.append(t)
    return tuple(sorted(out))


def build_truth(obs: pd.DataFrame, pert_col: str | None, pert_sep: str, gene_cols: str | None,
                group_col: str | None, null_tokens: str, strip_suffix: str | None,
                alias: dict) -> tuple[np.ndarray, list[tuple[str, ...]]]:
    """Per cell: (condition label, truth gene set). One condition = one decoded pseudobulk.

    GSE146194 must use `gene_cols`: its label joins genes on '_' and gene names carry '_<n>'
    guide suffixes, so splitting the label would invent genes. All controls collapse into ONE
    condition -- the decode baseline is the control mean."""
    nt = {t.strip().lower() for t in null_tokens.split(',') if t.strip()}
    strip_re = re.compile(strip_suffix) if strip_suffix else None
    if gene_cols:
        cols = [c.strip() for c in gene_cols.split(',') if c.strip()]
        mat = [obs[c].astype(str).to_numpy() for c in cols]
        targets = [parse_targets([m[i] for m in mat], nt, strip_re, alias)
                   for i in range(len(obs))]
    else:
        raw = obs[pert_col].astype(str).to_numpy()
        targets = [parse_targets(v.split(pert_sep), nt, strip_re, alias) for v in raw]
    if group_col and group_col in obs:
        groups = obs[group_col].astype(str).to_numpy()
    else:
        groups = np.array(['+'.join(t) if t else CONTROL for t in targets], dtype=object)
    groups = np.array([g if targets[i] else CONTROL for i, g in enumerate(groups)], dtype=object)
    return groups, targets


def panel_tokens(obs: pd.DataFrame, pert_col: str | None, pert_sep: str, gene_cols: str | None,
                 null_tokens: str, strip_suffix: str | None, alias: dict) -> set[str]:
    """Every symbol a targeted gene can appear under in var: stripped, raw-published AND aliased.

    The mask has to catch the gene however the screen's var spells it; matching only the aliased
    (current HGNC) symbol would silently leave a published-symbol column unmasked."""
    nt = {t.strip().lower() for t in null_tokens.split(',') if t.strip()}
    strip_re = re.compile(strip_suffix) if strip_suffix else None
    if gene_cols:
        vals = np.concatenate([obs[c.strip()].astype(str).unique()
                               for c in gene_cols.split(',') if c.strip()])
    else:
        vals = [t for v in obs[pert_col].astype(str).unique() for t in v.split(pert_sep)]
    out = set()
    for t in vals:
        t = str(t).strip()
        if not t or t.lower() in nt:
            continue
        s = strip_re.sub('', t) if strip_re is not None else t
        out |= {s, alias.get(s, s)}
    inv = {v: k for k, v in alias.items()}
    out |= {inv[t] for t in list(out) if t in inv}
    return {t for t in out if t and t.lower() not in nt}


# ---- expression ----------------------------------------------------------------------------
def zero_columns(X: sp.csr_matrix, cols: np.ndarray) -> sp.csr_matrix:
    """A copy of X with the given columns set to 0 (structural zeros, so nnz drops)."""
    X = X.tocsr(copy=True)
    kill = np.isin(X.indices, np.asarray(cols, dtype=X.indices.dtype))
    X.data[kill] = 0
    X.eliminate_zeros()
    return X


def log_pseudobulk(X: sp.csr_matrix, groups: np.ndarray, keys: list[str],
                   target_sum: float = 1e4) -> np.ndarray:
    """(len(keys), n_genes): per-group mean of log1p(counts normalised to `target_sum` per cell)."""
    lib = np.asarray(X.sum(1)).ravel()
    L = sp.diags(target_sum / np.clip(lib, 1, None)) @ X
    L.data = np.log1p(L.data)
    pos = {k: i for i, k in enumerate(keys)}
    rows = np.array([pos.get(g, -1) for g in groups])
    ok = rows >= 0
    G = sp.csr_matrix((np.ones(ok.sum()), (rows[ok], np.flatnonzero(ok))),
                      shape=(len(keys), X.shape[0]))
    n = np.asarray(G.sum(1)).ravel()
    return np.asarray((sp.diags(1.0 / np.clip(n, 1, None)) @ G @ L).todense())


# ---- rankings ------------------------------------------------------------------------------
def rank_omp(Q: np.ndarray, D: np.ndarray, depth: int) -> list[np.ndarray]:
    """Per query row, the non-negative OMP pick order over D's rows (stopping rule OFF)."""
    from expert.utils.readout import nnomp
    depth = min(int(depth), D.shape[0])
    return [nnomp(q, D, k=depth, stop=None)[0] for q in np.asarray(Q, np.float64)]


def rank_omp_signed(Q: np.ndarray, D: np.ndarray, depth: int) -> list[np.ndarray]:
    """Signed OMP (the disease readout's variant, metabolic_weinberg/signed_omp_check.py): pick
    by |correlation| with the residual, refit ALL chosen coefficients by signed least squares.
    A negative coefficient = the response points AWAY from that knockout, which is how a
    loss-of-function prototype can explain an activation response without negating the table."""
    Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    depth = min(int(depth), D.shape[0])
    out = []
    for q in np.asarray(Q, np.float64):
        r, chosen = q.copy(), []
        for _ in range(depth):
            a = np.abs(Dn @ r)
            if chosen:
                a[chosen] = -np.inf
            j = int(np.argmax(a))
            if not np.isfinite(a[j]) or a[j] <= 0:
                break
            chosen.append(j)
            M = D[chosen].T
            w, *_ = np.linalg.lstsq(M, q, rcond=None)
            r = q - M @ w
        out.append(np.asarray(chosen, int))
    return out


def rank_nnls(Q: np.ndarray, D: np.ndarray, depth: int) -> list[np.ndarray]:
    """The non-greedy alternative to OMP: ONE non-negative least-squares fit over the whole
    dictionary (unit-normalised rows, so coefficients compare across atoms), ranked by coefficient.
    Atoms the fit leaves at exactly zero are unranked."""
    from scipy.optimize import nnls
    Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    out = []
    for q in np.asarray(Q, np.float64):
        w, _ = nnls(Dn.T, q, maxiter=50 * D.shape[0])
        nz = np.flatnonzero(w > 0)
        out.append(nz[np.argsort(-w[nz], kind='stable')][:depth])
    return out


def rank_pairs_additive(Q: np.ndarray, D: np.ndarray, depth: int) -> list[np.ndarray]:
    """Pair matching under the ADDITIVE model (the GEARS-inverted structure with the additive
    predictor GEARS is benchmarked against): predict every double as the sum of two singles, rank
    all pairs by cosine to the query, and read genes off in order of first appearance."""
    n = D.shape[0]
    ia, ib = np.triu_indices(n, 1)
    S = D[ia] + D[ib]
    Sn = S / np.clip(np.linalg.norm(S, axis=1, keepdims=True), 1e-12, None)
    out = []
    for q in np.asarray(Q, np.float64):
        order = np.argsort(-(Sn @ (q / max(np.linalg.norm(q), 1e-12))), kind='stable')
        seen, genes = set(), []
        for p in order:
            for g in (ia[p], ib[p]):
                if g not in seen:
                    seen.add(g); genes.append(g)
            if len(genes) >= min(depth, n):
                break
        out.append(np.asarray(genes[:depth], int))
    return out


def rank_target(T: np.ndarray, depth: int, sign: float) -> list[np.ndarray]:
    """The benchmark's `neg_target` baseline: rank candidates by their OWN transcript's control-
    centred change (sign -1: most knocked down first, CRISPRi; +1: most raised first, CRISPRa).
    T is (queries x candidates). Needs the UNMASKED counts by construction."""
    return [np.argsort(-(sign * t), kind='stable')[:depth] for t in np.asarray(T, np.float64)]


def rank_cos(Q: np.ndarray, D: np.ndarray, depth: int) -> list[np.ndarray]:
    """Per query row, D's rows by descending cosine -- independent scoring, no explaining-away."""
    from expert.utils.readout import prototype_scores
    S = prototype_scores(Q, D)
    depth = min(int(depth), D.shape[0])
    return [np.argsort(-s, kind='stable')[:depth] for s in S]


def true_ranks(ranking: np.ndarray, truth_idx) -> list[float]:
    pos = {int(j): r + 1 for r, j in enumerate(ranking)}
    return [float(pos.get(int(t), np.inf)) for t in truth_idx]


# ---- metrics -------------------------------------------------------------------------------
def _per_condition(ranks: list[list[float]], k: int) -> dict[str, np.ndarray]:
    r = [np.asarray(x) for x in ranks]
    return {'recall': np.array([np.mean(x <= k) for x in r]),
            'any': np.array([np.any(x <= k) for x in r], float),
            'both': np.array([np.all(x <= k) for x in r], float),
            'hit1': np.array([np.any(x == 1) for x in r], float)}


def score(rankings: list[np.ndarray], truths: list[list[int]], orders: np.ndarray, ks,
          n_perm: int, rng) -> pd.DataFrame:
    """Rank metrics per order (1, 2, 'all') and k, each against a within-order permutation null."""
    rows = []
    orders = np.asarray(orders)
    pos = [{int(j): r + 1 for r, j in enumerate(rk)} for rk in rankings]

    def lookup(i, t):
        return [float(pos[i].get(int(g), np.inf)) for g in t]

    for o in sorted(set(orders.tolist())) + ['all']:
        idx = np.flatnonzero(orders == o) if o != 'all' else np.arange(len(orders))
        if not len(idx):
            continue
        obs = [lookup(i, truths[i]) for i in idx]
        perms = []
        for _ in range(n_perm):
            # permute WITHIN order so a single's truth never lands on a double's ranking
            sh = idx.copy()
            for oo in set(orders[idx].tolist()):
                m = orders[idx] == oo
                sh[m] = rng.permutation(idx[m])
            perms.append([lookup(i, truths[j]) for i, j in zip(idx, sh)])
        for k in ks:
            ob = _per_condition(obs, k)
            nl = [_per_condition(p, k) for p in perms]
            for met, v in ob.items():
                if met == 'hit1' and k != ks[0]:
                    continue
                nv = np.array([x[met].mean() for x in nl])
                rows.append({'order': str(o), 'metric': met, 'k': int(k),
                             'value': float(v.mean()), 'null_mean': float(nv.mean()),
                             'null_sd': float(nv.std(ddof=1)) if len(nv) > 1 else np.nan,
                             'p_perm': float((1 + (nv >= v.mean()).sum()) / (1 + len(nv))),
                             'n': int(len(idx))})
    return pd.DataFrame(rows)
