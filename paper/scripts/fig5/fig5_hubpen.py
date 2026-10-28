#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Extract the stored hubness offset (ExPert.add_hub_correction -> module buffer atom_hub_pen, (2, n_vocab)) with
its label order, and check it against a harvested prototype bank, so the Fig. 5 readout can apply it exactly as
predict() does (loss: cos - pen[0]; gain: -cos - pen[1]).

    python paper/scripts/fig5/fig5_hubpen.py --model-pt <model.pt> --bank <bank.npz> --out <npz>
"""
import argparse

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model-pt', required=True)
ap.add_argument('--bank', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ck = torch.load(a.model_pt, map_location='cpu', weights_only=False)
sd = ck['model_state_dict']
key = [k for k in sd if k.endswith('atom_hub_pen')]
assert key, 'checkpoint carries no atom_hub_pen'
pen = sd[key[0]].float().numpy()
from expert import ExPert                                  # idx_to_label = the full vocab order of the offset
import os
model = ExPert.load(os.path.dirname(a.model_pt))
lab = [str(x) for x in model.idx_to_label][:pen.shape[1]]
assert len(lab) == pen.shape[1], (len(lab), pen.shape)
b = np.load(a.bank, allow_pickle=True)
bl = [str(x) for x in b['labels']]
tr = b['trained'].astype(bool)
print('pen', pen.shape, 'labels', len(lab), 'bank', len(bl))
pos = {l: i for i, l in enumerate(lab)}
miss = [l for l in bl if l not in pos]
print('bank labels missing from model vocab:', len(miss))
assert not miss, miss[:5]
P = np.stack([pen[:, pos[l]] if l in pos else np.zeros(2) for l in bl], 1)
for nm, m in (('trained', tr), ('zero-shot', ~tr)):
    print(f'{nm}: n={m.sum()}  loss pen median {np.median(P[0, m]):.4f} [{P[0, m].min():.4f}, {P[0, m].max():.4f}]'
          f'  gain pen median {np.median(P[1, m]):.4f}  zeros {(P[0, m] == 0).sum()}')
np.savez(a.out, pen=P.astype(np.float32), labels=np.array(bl, dtype=object))
print('->', a.out)
