# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""ExPert._load_class_embedding_df format handling.

Guards the npz support added after the numpy 1.x/2.x pickle break, plus the
dict-pickle and DataFrame paths. All forms must normalize to a DataFrame
indexed by label with dim_<i> columns.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from expert.models._expert import ExPert


def test_npz_roundtrip(tmp_path):
    genes = [f"G{i}" for i in range(20)]
    emb = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    p = tmp_path / "emb.npz"
    np.savez(p, embeddings=emb, genes=np.array(genes, dtype=object))

    df = ExPert._load_class_embedding_df(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 8)
    assert list(df.index) == genes
    assert list(df.columns) == [f"dim_{i}" for i in range(8)]
    np.testing.assert_allclose(df.to_numpy(), emb, rtol=1e-5)


def test_dict_pickle(tmp_path):
    """GenePT-style {gene: vector} pickle (lists or arrays) → DataFrame."""
    d = {f"G{i}": list(np.random.default_rng(i).normal(size=6)) for i in range(15)}
    p = tmp_path / "emb.pkl"
    with open(p, "wb") as f:
        pickle.dump(d, f)
    df = ExPert._load_class_embedding_df(str(p))
    assert df.shape == (15, 6)
    assert set(df.index) == set(d.keys())


def test_dataframe_pickle(tmp_path):
    genes = [f"G{i}" for i in range(10)]
    src = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 5)), index=genes)
    p = tmp_path / "emb_df.pkl"
    src.to_pickle(p)
    df = ExPert._load_class_embedding_df(str(p))
    assert df.shape == (10, 5)
    assert list(df.index) == genes


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "emb.xyz"
    p.write_text("nope")
    with pytest.raises(ValueError):
        ExPert._load_class_embedding_df(str(p))
