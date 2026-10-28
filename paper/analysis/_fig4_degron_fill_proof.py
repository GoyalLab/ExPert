# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
"""Which f(missing) projection did the degron run actually score against? Settle it from the artefacts.

The degron pipeline saves its scored contrasts (`degron_pb_delta_*.npy`), and the rank it reports is
`calibrated_readout(D, unit(P), n_cells, rounds=3, head=10)`. So the whole readout can be replayed
offline against EITHER bank. If replaying with the NEW bank reproduces the reported ranks and the OLD
bank reproduces the pre-refill ranks, the run demonstrably used the new projection -- no appeal to
file timestamps required.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from expert.utils.readout import calibrated_readout

ap = argparse.ArgumentParser()
ap.add_argument('--delta', required=True)
ap.add_argument('--activity', required=True, help='degron_pb_activity_*.csv (row order of the delta)')
ap.add_argument('--ranks-new', required=True, help='ranks csv from the run under test')
ap.add_argument('--ranks-old', required=True, help='ranks csv from the pre-refill run')
ap.add_argument('--banks', nargs='+', required=True, help='label=bank.npz')
ap.add_argument('--model-pt', required=True, help='for readout_noise')
ap.add_argument('--k', type=int, default=20000)
a = ap.parse_args()

unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
D = np.load(a.delta).astype(np.float64)
act = pd.read_csv(a.activity)
assert len(act) == len(D), f'{len(act)} activity rows vs {len(D)} delta rows'
ncell = act.n_treated.to_numpy()

sd = torch.load(a.model_pt, map_location='cpu', weights_only=False)['model_state_dict']
nk = [k for k in sd if k.endswith('readout_noise')]
noise = sd[nk[0]].float().numpy().astype(np.float64) if nk else None
print(f'delta {D.shape} | readout_noise {"present" if noise is not None else "ABSENT"}')

rec = {}
for nm, f in (('new', a.ranks_new), ('old', a.ranks_old)):
    t = pd.read_csv(f)
    rec[nm] = t[t.bank == 'full'].set_index('clone')[['rank_loss', 'rank_gain']]

for spec in a.banks:
    lab, path = spec.split('=', 1)
    b = np.load(path, allow_pickle=True)
    P, labels = b['P'].astype(np.float64), b['labels'].astype(str)
    Pb, lb = unit(P), labels
    pos = {g: i for i, g in enumerate(lb)}
    r = calibrated_readout(D, Pb, ncell, noise=noise, rounds=3, head=10, k=min(a.k, len(Pb)))
    out = []
    for i, cl in enumerate(act.clone):
        j = pos.get(cl)
        if j is None:
            continue
        row = {'clone': cl}
        for t in ('loss', 'gain'):
            ix = np.asarray(r[t][i], np.int64)
            hit = np.flatnonzero(ix == j)
            row[f'replay_{t}'] = int(hit[0]) + 1 if len(hit) else len(Pb)
        out.append(row)
    rp = pd.DataFrame(out).set_index('clone')
    print(f'\n=== replay with the {lab.upper()} bank ===')
    for nm in ('new', 'old'):
        j = rp.join(rec[nm], how='inner')
        eq = int(((j.replay_loss == j.rank_loss) & (j.replay_gain == j.rank_gain)).sum())
        print(f'  matches the {nm.upper()} run on {eq}/{len(j)} clones')
    j = rp.join(rec['new'], how='inner').join(rec['old'], how='inner', rsuffix='_old')
    show = ['SOX2', 'NUP62', 'MED28', 'CHD1', 'EIF6']
    print(j.loc[[c for c in show if c in j.index],
                ['replay_loss', 'replay_gain', 'rank_loss', 'rank_gain',
                 'rank_loss_old', 'rank_gain_old']].to_string())
