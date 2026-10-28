# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.
"""Combine several relatedness priors into ONE per-pair score for the Proxy Anchor push weight.

WHAT THIS FEEDS. Proxy Anchor's negative term pushes each atom away from every cell that is not
its class. The prior only relaxes that push, per pair:

    delta_pq = delta * (1 - beta * sim_pq)

There is no attractive term anywhere, so a prior can decline to push but can NEVER pull. That is
the property that makes stacking priors safe here, and it is what `struct_rerank` / `atom_struct`
lacked -- both gave a prior an attractive or ordering role and went endogenous (text edges
0.218 -> 0.563 while response edges collapsed 0.185 -> 0.056).

WHY STACK AT ALL. Measured on the pa6/pa3 atom graphs, the modalities are near-DISJOINT: each
backs 7-10% of edges and all three agree on 0.3-0.5%. Unique support was response 0.076, text
0.064, depmap 0.049. Union coverage is close to additive, so a combination genuinely sees more
than any single prior.

THE FOUR RULES, each of which cost something to learn:

1. DOUBLE-CENTRE every matrix before the cosine. Nuisance structure can sit on either index and
   the wrong single choice is catastrophic: column-only left DepMap effectively rank-1
   (participation ratio 1.4 of 1,178, PC1 0.850, 46% of pairs above cosine 0.5) because it
   preserved per-gene essentiality magnitude. Double is optimal or tied on text, DepMap and
   response alike. See reference_double_centre_priors.

2. Z-SCORE each prior against its OWN off-diagonal null before combining. Post-centring scales
   differ a lot (p90: DepMap 0.083, text 0.135, response 0.158), so raw values are not
   comparable and a sum would silently weight by scale.

3. COMBINE BY MAX, not sum. With near-disjoint support a max is close to a union, and it keeps
   the semantics one-sided: any prior may vouch for a pair, none can push it apart.

4. FIXED TOP-K BUDGET over the combined score -- never the union of per-prior top-k lists. The
   relaxation works because it is SELECTIVE; three unioned top-10 lists would take the relaxed
   fraction from ~5% to ~15% of pairs and dilute the repulsion's target.

DO NOT LEARN THE MIXTURE WEIGHTS FROM THIS LOSS. Relaxing a pair always lowers the negative term,
so gradient descent on the weights drives all of them up until every pair is relaxed -- a
degenerate solution that reads as the loss improving. Set them from measured predictive value
instead (response predicts held-out co-response at +0.0684, text at +0.0015 -- a 45x gap).

ALWAYS RUN THE SHUFFLED CONTROL. project_multiprior_struct_loss records a NOISE prior passing a
~0.22 gate in this codebase. Without `shuffle=True` as a comparison arm, "the priors help" cannot
be distinguished from "relaxing k pairs per row helps regardless of which pairs".
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ['load_prior_matrix']


def load_prior_matrix(path, labels, *, double_centre=True, shuffle=False, seed=0):
    """(S, have, reason) -- (L, L) prior cosine over `labels`, or (None, None, reason).

    `have` is (L,) bool: labels absent from the file are excluded rather than scored as
    unrelated, and their rows/cols are set to -inf so they can never enter a top-k.

    shuffle=True permutes the label->row assignment, producing a matrix with IDENTICAL marginal
    structure and NO real correspondence. That is the control arm, not a debugging aid.
    """
    import os
    if not path or not os.path.exists(path):
        return None, None, f'missing prior file: {path}'
    z = np.load(path, allow_pickle=True)
    ek = 'embeddings' if 'embeddings' in z else z.files[0]
    gk = next((k for k in ('genes', 'labels', 'names', 'index') if k in z), z.files[-1])
    E = np.asarray(z[ek], np.float64)
    G = np.asarray(z[gk]).astype(str)
    if E.shape[0] != len(G):
        return None, None, f'{os.path.basename(path)}: {E.shape[0]} rows vs {len(G)} names'
    pos = {g: i for i, g in enumerate(G)}
    labels = [str(x) for x in labels]
    rows = np.array([pos.get(x, -1) for x in labels])
    have = rows >= 0
    if have.sum() < 8:
        return None, None, f'{os.path.basename(path)}: only {int(have.sum())} labels found'
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(have)
        rows[idx] = rows[rng.permutation(idx)]
    M = np.zeros((len(labels), E.shape[1]))
    M[have] = E[rows[have]]
    sub = M[have]
    if double_centre:
        sub = sub - sub.mean(1, keepdims=True)      # per-row: a per-entity magnitude/offset
        sub = sub - sub.mean(0, keepdims=True)      # per-column: a direction shared by all rows
    sub = sub / np.clip(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12, None)
    S = np.full((len(labels), len(labels)), -np.inf)
    S[np.ix_(have, have)] = sub @ sub.T
    np.fill_diagonal(S, -np.inf)                    # a label is never its own neighbour
    return torch.from_numpy(S).float(), torch.from_numpy(have), None


