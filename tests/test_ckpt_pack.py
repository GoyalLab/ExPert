# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""expert.utils.ckpt_pack must round-trip a SET of checkpoints to identical object trees.

Synthetic checkpoints with the structure that made the real ones big: a bank shared by both
(state dict AND a DataFrame copy in attr_dict), an all-zero buffer, int64 indices, a defaultdict and
small leaves. The bar is exact equality of every leaf, container type included.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import pytest
import torch

from expert.utils import ckpt_pack


def _ckpt(seed, bank, bank_df):
    g = torch.Generator().manual_seed(seed)
    sd = OrderedDict(weight=torch.randn(400, 1000, generator=g), bank=bank.clone(),
                     centers=torch.zeros(300, 1000), step=torch.tensor(seed))
    dd = defaultdict(list, a=[1, 2])
    return {'model_state_dict': sd, 'var_names': np.array(['g1', 'g2']),
            'attr_dict': {'registry_': {'setup_args': {'ext_df': bank_df.copy()}}, 'counts': dd,
                          'train_indices_': np.arange(seed, 300_000 + seed, dtype=np.int64), 'run_name_': f'r{seed}'}}


@pytest.fixture
def ckpts(tmp_path):
    bank = torch.randn(500, 1000, generator=torch.Generator().manual_seed(7))
    df = pd.DataFrame(bank.numpy(), index=[f'g{i}' for i in range(500)])
    paths = {}
    for s in (1, 2):
        paths[f'm{s}'] = tmp_path / f'm{s}.pt'
        torch.save(_ckpt(s, bank, df), paths[f'm{s}'])
    return paths


def test_round_trip_identical(ckpts, tmp_path):
    meta = ckpt_pack.pack({k: str(v) for k, v in ckpts.items()}, tmp_path / 'packed')
    assert meta['n_shared'] == 2                                   # the bank tensor and its DataFrame copy
    for name, src in ckpts.items():
        r = meta['checkpoints'][name]
        assert r['shared_bytes'] > 0 and r['zeros_bytes'] > 0 and r['zlib_bytes'] > 0
        assert r['slim_bytes'] < r['source_bytes']
        out = ckpt_pack.unpack(tmp_path / 'packed' / f'{name}.slim.pt', tmp_path / 'packed' / 'shared.pt',
                               tmp_path / f'{name}.restored.pt')
        assert ckpt_pack.verify(str(src), str(tmp_path / f'{name}.restored.pt')) == []
        assert type(out['model_state_dict']) is OrderedDict
        assert type(out['attr_dict']['counts']) is defaultdict and out['attr_dict']['counts']['new'] == []


def test_verify_reports_a_change(ckpts, tmp_path):
    ck = torch.load(ckpts['m1'], weights_only=False)
    ck['model_state_dict']['weight'][0, 0] += 1
    assert ckpt_pack.verify(str(ckpts['m1']), ck) == ['model_state_dict.weight']
