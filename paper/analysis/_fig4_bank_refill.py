# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Refresh a cached bank.npz's zero-shot tail from a refilled checkpoint, without re-encoding.

`prior_missing_T` touches ONLY the untrained rows of the class bank, and `_prior_missing_apply` writes
those rows verbatim (`torch.where(fill, T, A[n:])`). So for a bank whose trained block came from the
same checkpoint, replacing rows [n_labels:] with the checkpoint's T rows reproduces exactly what a
fresh predict() encode would emit -- the latents are untouched by the fill, so re-encoding a 100M-cell
screen to pick up a new tail is wasted compute.

--verify compares an independently RE-ENCODED bank against the patch this script would write, which is
what licenses the shortcut; run it once per release rather than trusting the argument above.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model-pt', required=True)
ap.add_argument('--bank', required=True, help='bank.npz to refresh (or, with --verify, the OLD bank)')
ap.add_argument('--out', help='write the refreshed bank here (default: in place, after a .oldfill copy)')
ap.add_argument('--verify', help='an independently re-encoded bank.npz with the NEW fill')
a = ap.parse_args()

sd = torch.load(a.model_pt, map_location='cpu', weights_only=False)['model_state_dict']
T = sd[next(k for k in sd if k.endswith('prior_missing_T'))].float().numpy()
b = np.load(a.bank, allow_pickle=True)
P, labels, nl = b['P'], b['labels'], int(b['n_labels'])
u = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)

m = min(T.shape[0], P.shape[0])
fill = np.linalg.norm(T[nl:m], axis=1) > 0
new = P.copy()
new[nl:m][fill] = T[nl:m][fill].astype(P.dtype)
print(f'bank {P.shape}, trained {nl}, tail rows refreshed {int(fill.sum()):,} of {P.shape[0] - nl:,}')
print(f'  cos(old tail, new tail) mean {np.nanmean((u(P[nl:m]) * u(new[nl:m])).sum(1)):.4f}')

if a.verify:
    z = np.load(a.verify, allow_pickle=True)
    Q = z['P']
    assert Q.shape == P.shape, f'{Q.shape} vs {P.shape}'
    assert int(z['n_labels']) == nl
    for nm, lo, hi in (('trained block', 0, nl), ('zero-shot tail', nl, m)):
        c = (u(np.asarray(new[lo:hi], np.float64)) * u(np.asarray(Q[lo:hi], np.float64))).sum(1)
        md = float(np.nanmax(np.abs(np.asarray(new[lo:hi], np.float64) - np.asarray(Q[lo:hi], np.float64))))
        ok = np.nanmin(c) > 0.9999
        print(f'  [verify] {nm:15s} cos min {np.nanmin(c):.6f} mean {np.nanmean(c):.6f} '
              f'max|diff| {md:.3e}  {"OK" if ok else "MISMATCH"}')
    raise SystemExit(0)

out = Path(a.out) if a.out else Path(a.bank)
if not a.out:
    bk = out.with_suffix('.oldfill.npz')
    if not bk.exists():
        shutil.copy2(out, bk)
        print(f'  backed up -> {bk}')
np.savez(out, P=new, labels=labels, n_labels=nl)
print(f'  wrote {out}')
