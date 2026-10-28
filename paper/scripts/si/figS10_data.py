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

"""Supplementary Fig. S10 plotting data (why Proxy Anchor rather than a softmax head: the gradient once the correct
prototype already wins). Controlled geometry at the shipped full-scale shape (--batch cells, --dim latent, --classes
trained prototypes): each cell's latent is its own prototype mixed with noise, z = sep * a_y + (1 - sep) * noise, so
`sep` moves the batch from chance to every cell ranking its own prototype first. Two losses on the same z and
prototypes, at their shipped settings:

    cosine cross-entropy   the classifier head of the CE arms (_xpert._cosine_head_logits, no margin):
                           CE(s * cos(z, w), y), s = --ce-scale
    Proxy Anchor           expert.losses.proxy_anchor.proxy_anchor_loss (alpha, delta as shipped), and its two halves:
                           the negative term (neg_only=True) and the positive term (full minus negative gradient)

One CSV per panel in --out-dir (paper/figures/si/figS10/data):

    figS10a_sweep.csv     per sep x loss term: share of cells whose own prototype ranks first, mean margin (cosine to
                         the own prototype minus the best other), gradient norm on the prototypes and on z, and both
                         relative to the same term at chance geometry (sep = 0)
    figS10_meta.json      shape, settings, seed, the sep used for B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from expert.losses.proxy_anchor import proxy_anchor_loss

TERMS = ('cosine CE', 'Proxy Anchor', 'Proxy Anchor, negative term', 'Proxy Anchor, positive term')


def batch(sep, a, g):
    atoms = F.normalize(torch.randn(a.classes, a.dim, generator=g), dim=-1)
    y = torch.randint(0, a.classes, (a.batch,), generator=g)
    noise = F.normalize(torch.randn(a.batch, a.dim, generator=g), dim=-1)
    return sep * atoms[y] + (1.0 - sep) * noise, atoms, y


def cos_ce(z, w, y, s):
    """the cosine classifier head of the CE arms (no margin, no residual lift)"""
    return F.cross_entropy(F.normalize(z, dim=-1) @ F.normalize(w, dim=-1).T * s, y)


def grads(fn, z, atoms, y):
    zz, aa = z.clone().requires_grad_(True), atoms.clone().requires_grad_(True)
    fn(zz, aa, y).backward()
    return zz.grad.detach(), aa.grad.detach()


def measure(z, atoms, y, a):
    out = {}
    out['cosine CE'] = grads(lambda zz, aa, yy: cos_ce(zz, aa, yy, a.ce_scale), z, atoms, y)
    pa = lambda neg: (lambda zz, aa, yy: proxy_anchor_loss(zz, aa, yy, alpha=a.alpha, delta=a.delta,
                                                           neg_only=neg)[0])
    out['Proxy Anchor'] = grads(pa(False), z, atoms, y)
    out['Proxy Anchor, negative term'] = grads(pa(True), z, atoms, y)
    out['Proxy Anchor, positive term'] = tuple(f - n for f, n in zip(out['Proxy Anchor'],
                                                                      out['Proxy Anchor, negative term']))
    return out


def rank_stats(z, atoms, y):
    s = F.normalize(z, dim=-1) @ F.normalize(atoms, dim=-1).T
    own = s.gather(1, y[:, None]).squeeze(1)
    s.scatter_(1, y[:, None], -2.0)
    margin = own - s.max(1).values
    return margin, float((margin > 0).float().mean())


def main(a):
    torch.manual_seed(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sep in a.seps:
        g = torch.Generator().manual_seed(a.seed)        # same atoms / labels / noise at every sep
        z, atoms, y = batch(sep, a, g)
        margin, top1 = rank_stats(z, atoms, y)
        for term, (gz, ga) in measure(z, atoms, y, a).items():
            rows.append(dict(sep=sep, term=term, top1=top1, margin_mean=float(margin.mean()),
                             grad_atoms=float(ga.norm()), grad_z=float(gz.norm())))
        print(f'sep {sep:.2f}: own prototype first for {top1:.1%}', flush=True)
    A = pd.DataFrame(rows)
    ref = A[A.sep == min(a.seps)].set_index('term')
    for c in ('grad_atoms', 'grad_z'):
        A[f'{c}_rel'] = A[c] / A.term.map(ref[c])
    A.to_csv(out / 'figS10a_sweep.csv', index=False)
    print(A.pivot_table(index='sep', columns='term', values='grad_atoms_rel').round(4).to_string(), flush=True)

    (out / 'figS10_meta.json').write_text(json.dumps(dict(
        batch=a.batch, dim=a.dim, classes=a.classes, ce_scale=a.ce_scale, alpha=a.alpha, delta=a.delta,
        seed=a.seed, seps=a.seps, terms=list(TERMS)), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', type=int, required=True, help='cells per batch (shipped 16384)')
    ap.add_argument('--dim', type=int, required=True, help='latent width (shipped n_latent 512)')
    ap.add_argument('--classes', type=int, required=True, help='trained prototypes (8,964)')
    ap.add_argument('--ce-scale', type=float, required=True, help='cosine head logit scale (cls_head_arc_scale, 30)')
    ap.add_argument('--alpha', type=float, required=True, help='proxy_anchor_alpha (32)')
    ap.add_argument('--delta', type=float, required=True, help='proxy_anchor_delta (0.1)')
    ap.add_argument('--seps', type=float, nargs='+',     # fine where the batch goes from chance to all-correct
                    default=sorted({round(x, 3) for x in [*np.linspace(0, 0.3, 31), *np.linspace(0.3, 1, 15)]}))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    main(ap.parse_args())
