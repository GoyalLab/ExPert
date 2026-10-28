# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Shared pytest fixtures.

`tiny_adata` + `tiny_config` are the SHIPPED v0.1.0 configuration at toy scale: learnable atoms trained by the
proxy-anchor loss, the prior-atom pull and its f(missing) negatives (prior banks from `class_emb_extra_paths`),
control-centred encoder input (`control_adata_path`), invariant alignment, the efficiency head and the cosine
head. Only sizes are shrunk, plus the prior-atom thresholds that would never be reached at 12 classes.
Everything is generated under tmp_path -- the suite reads nothing from disk.
"""
from __future__ import annotations

import copy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

N_CELLS = 720
N_GENES = 48
N_OBS_PERTS = 12       # perturbations present in the data (the trained classes)
N_VOCAB = 30           # gene vocabulary of the class bank: 18 genes are never perturbed (zero-shot rows)
N_DATASETS = 3
EMB_DIM = 16


def _gene_names(n):
    return [f"G{i}" for i in range(n)]


def make_adata(n_perts: int = N_OBS_PERTS, n_vocab: int = N_VOCAB, n_cells: int = N_CELLS) -> ad.AnnData:
    """Raw counts, perturbation / dataset / efficiency obs and the gene-vocabulary class bank in uns. Each
    perturbation shifts its own gene block so the classes are learnable."""
    rng = np.random.default_rng(0)
    vocab = _gene_names(n_vocab)
    perts = vocab[:n_perts]
    pert = rng.choice(perts, size=n_cells)
    dataset = rng.choice([f"ds{i}" for i in range(N_DATASETS)], size=n_cells)
    lam = np.full((n_cells, N_GENES), 0.6)
    for j, p in enumerate(perts):
        lam[pert == p, (4 * j) % N_GENES:(4 * j) % N_GENES + 4] *= 4.0
    counts = rng.poisson(lam).astype(np.int16)
    counts[np.arange(n_cells), rng.integers(0, N_GENES, n_cells)] += 1
    obs = pd.DataFrame({
        "perturbation": pd.Categorical(pert),
        "dataset": pd.Categorical(dataset),
        "efficiency": rng.normal(1.0, 1.0, n_cells).astype(np.float32),
    }, index=[f"cell{i}" for i in range(n_cells)])
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(N_GENES)]))
    adata.uns["cls_embedding"] = pd.DataFrame(rng.normal(size=(n_vocab, EMB_DIM)).astype(np.float32), index=vocab)
    return adata


def make_files(tmp_path, adata) -> dict:
    """Control cells for every dataset (ctrl.h5ad) and two partial-coverage prior banks over the vocabulary."""
    rng = np.random.default_rng(1)
    n = 90
    ctrl = ad.AnnData(
        X=sp.csr_matrix(rng.poisson(0.6, size=(n, N_GENES)).astype(np.int16)),
        obs=pd.DataFrame({"dataset": [f"ds{i % N_DATASETS}" for i in range(n)], "perturbation": "NTC"},
                         index=[f"ctrl{i}" for i in range(n)]),
        var=pd.DataFrame(index=adata.var_names))
    ctrl_p = tmp_path / "ctrl.h5ad"
    ctrl.write_h5ad(ctrl_p)
    vocab = np.asarray(adata.uns["cls_embedding"].index)
    priors = []
    for name, keep in (("prior_a", len(vocab)), ("prior_b", len(vocab) - 6)):
        p = tmp_path / f"{name}.npz"
        np.savez(p, embeddings=rng.normal(size=(keep, 8)).astype(np.float32), genes=vocab[:keep])
        priors.append(str(p))
    return {"ctrl": str(ctrl_p), "priors": priors}


def make_config(tmp_path, files) -> dict:
    """v0.1.0's model / plan / schedule settings (translated to the current schema), shrunk to toy sizes."""
    return copy.deepcopy({
        "data": {
            "batch_size": 64, "loader_type": "balanced", "weights_mode": "both", "use_balanced_weights": True,
            "use_contrastive_loader": False, "use_special_for_split": "train", "n_samples": 256,
            "num_workers": 0, "pin_memory": False, "persistent_workers": False, "train_size": 0.8,
            "shuffle": True, "extra_ctx_col": "dataset", "eval_cap_per_group": 5,
        },
        "train": {
            "max_epochs": 3, "limit_train_batches": 4, "limit_val_batches": 4, "accelerator": "cpu", "devices": 1, "precision": 32, "gradient_clip_val": 1.0,
            "check_val_every_n_epoch": 1, "simple_progress_bar": True,
            "checkpoint": True, "checkpoint_every_n_epochs": 1, "checkpoint_mode": "max",
            "checkpoint_monitor": "val_live_macro_top10_recall_A", "checkpoint_save_top_k": 10,
            "early_stopping": False, "early_stopping_start_epoch": 50,
            "plan_kwargs": {
                "lr": 5e-3, "weight_decay": 1e-5, "n_epochs_warmup": 200, "kl_stage_offset": 0,
                "full_log_every_n_epoch": 1, "n_max_cache": 2000, "per_ds_min_cells": 5,
                "plot_umap": False, "plot_cm": False, "plot_cat_emb": False, "save_cache": False,
                "average": "macro", "top_k": 1, "use_posterior_mean": "val", "reduce_lr_on_plateau": False,
                "anneal_schedules": {
                    "kl_weight": {"schedule": "linear", "anneal_k": 8, "min_weight": 0.0, "max_weight": 0.1},
                    "cls_head_weight": {"weight": 0.0},
                    "lambda_eff": {"weight": 0.2},
                    "rl_weight": {"weight": 0.5},
                    "gate_eff": {"weight": 0.0},
                    "T_gamma": {"weight": 0.0},
                },
            },
        },
        "model": {
            "n_latent": 32, "n_hidden": 64, "n_shared": 32, "dropout_rate": 0.1,
            "gene_likelihood": "zinb", "dispersion": "gene", "use_batch_norm": "none", "use_layer_norm": "both",
            "use_adapter": True, "model_efficiency": True, "use_efficiency_pa_weight": True,
            "efficiency_rank_target": True, "decode_covariates": True, "deeply_inject_covariates": True,
            "use_full_cls_emb": True, "batch_representation": "embedding", "invariant_align": True,
            "scale_elbo": False, "min_kl": 0.0, "reduction": "batchmean", "non_elbo_reduction": "mean",
            "use_cls_head": True, "cls_head_cosine": True, "cls_head_dropout": 0.3,
            "class_emb_extra_paths": files["priors"],
            "extra_encoder_kwargs": {"encoder_type": "mixed", "n_layers": 1, "n_encoder_layers": 1,
                                     "center_input": True, "center_mode": "ctrl", "init_emb_pca": True,
                                     "linear_gene_proj": True, "noise_sigma": 0.05, "activation_fn": torch.nn.GELU},
            "extra_decoder_kwargs": {"decoder_type": "mixed", "n_layers": 1, "n_cat_frac": 0.5,
                                     "injection_from_layer": 1, "activation_fn": torch.nn.GELU},
            "extra_aligner_kwargs": {"mode": "shared", "temperature": 0.03, "dropout_rate": 0.1},
            "extra_cls_emb_kwargs": {"random_init_proxies": True, "adapter_mode": "identity", "freeze": False,
                                     "n_prototypes": 1},
            "extra_clip_kwargs": {"proto_data_derived": True, "proto_min_updates": 50, "momentum": 0.1},
            "extra_proxy_anchor_kwargs": {"weight": 200.0, "alpha": 32.0, "delta": 0.1},
            # v0.1.0's pull; min_fit / pull_q / graph_k / neg_k / folds shrunk so 12 classes can fit f()
            "extra_prior_atom_kwargs": {"weight": 3.0, "neg_weight": 1.0, "gate": "eff", "evidence": "nqxnds",
                                        "map": "graph", "missing_map": "graph", "start_epoch": 0,
                                        "ramp_epochs": 1, "refit_every": 1, "cos_floor": -1.0, "coh_max": 0.01,
                                        "pull_q": 0.25, "min_fit": 4, "min_cells": 2, "graph_k": 3,
                                        "neg_k": 8, "folds": 2},
        },
        "model_setup": {
            "batch_key": "dataset", "labels_key": "perturbation", "efficiency_key": "efficiency",
            "class_emb_uns_key": "cls_embedding", "categorical_covariate_keys": ["dataset"],
            "control_adata_path": files["ctrl"],
        },
        "_model_log_dir": str(tmp_path / "model"),
    })


@pytest.fixture
def tiny_adata() -> ad.AnnData:
    return make_adata()


@pytest.fixture
def tiny_files(tmp_path, tiny_adata) -> dict:
    return make_files(tmp_path, tiny_adata)


@pytest.fixture
def tiny_config(tmp_path, tiny_files) -> dict:
    return make_config(tmp_path, tiny_files)


def build_model(adata, cfg, seed: int = 0):
    """setup_anndata + ExPert(...) exactly as the training entry point does, at a fixed seed."""
    import scvi
    from expert import ExPert
    scvi.settings.seed = seed
    ExPert.setup_anndata(adata, **cfg["model_setup"])
    return ExPert(adata, model_log_dir=cfg["_model_log_dir"], **copy.deepcopy(cfg["model"]))


class StepRecorder:
    """Context manager: every Trainer built inside it records `trainer.callback_metrics` at the end of each training
    epoch (`epochs`: the epoch aggregate of every training metric -- the plan logs them on_epoch) and of each
    validation (`val`)."""

    def __init__(self):
        self.epochs: list[dict] = []
        self.val: list[dict] = []

    def series(self, name: str) -> np.ndarray:
        """Per-epoch values of a training metric (epochs that logged it)."""
        return np.array([e[name] for e in self.epochs if name in e], dtype=float)

    def __enter__(self):
        import lightning.pytorch as L
        rec = self

        def scalar(v):
            if torch.is_tensor(v) and v.numel() == 1:
                return float(v.detach().float())
            return float(v) if isinstance(v, (int, float)) else None

        class _Cb(L.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                rec.epochs.append({n.removesuffix('_epoch'): x for n, v in trainer.callback_metrics.items()
                                   if not n.endswith('_step') and (x := scalar(v)) is not None}
                                  | {'epoch': int(trainer.current_epoch)})

            def on_validation_end(self, trainer, pl_module):
                if not trainer.sanity_checking:
                    rec.val.append({n: x for n, v in trainer.callback_metrics.items()
                                    if (x := scalar(v)) is not None} | {'epoch': int(trainer.current_epoch)})

        self._orig = L.Trainer.__init__
        orig = self._orig

        def init(trainer, *a, **k):
            k['callbacks'] = list(k.get('callbacks') or []) + [_Cb()]
            orig(trainer, *a, **k)
        L.Trainer.__init__ = init
        return self

    def __exit__(self, *exc):
        import lightning.pytorch as L
        L.Trainer.__init__ = self._orig
        return False
