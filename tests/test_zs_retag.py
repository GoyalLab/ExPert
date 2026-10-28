# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""The zero-shot subset's arm (id_ood_pert vs ood_ood_pert) is re-derived from the ACTIVE fold's held-out set,
whichever key the split builder used for it."""
from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from expert import ExPert


def _zs(tag):
    obs = pd.DataFrame({"dataset": ["A", "B", "C", "A"], "zs_split": tag}, index=list("wxyz"))
    return ad.AnnData(X=np.zeros((4, 2), np.float32), obs=obs)


@pytest.mark.parametrize("key", ["held_out_contexts", "held_out_datasets"])
def test_retag_reads_either_held_out_key(tmp_path, key):
    (tmp_path / "metadata.json").write_text(json.dumps({"folds": [{key: ["X"]}, {key: ["B", "C"]}]}))
    (tmp_path / "fold_1").mkdir()
    zs = _zs("id_ood_pert")                                       # baked for another fold: everything in-dist
    ExPert._retag_zs_split_for_fold(zs, "zs.h5ad", str(tmp_path / "fold_1"))
    assert list(zs.obs["zs_split"]) == ["id_ood_pert", "ood_ood_pert", "ood_ood_pert", "id_ood_pert"]


def test_missing_context_column_raises(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"folds": [{"held_out_datasets": ["B"]}]}))
    zs = _zs("id_ood_pert")
    zs.obs = zs.obs.drop(columns="dataset")
    with pytest.raises(Exception, match="cannot be derived"):
        ExPert._retag_zs_split_for_fold(zs, "zs.h5ad", str(tmp_path / "fold_0"))
