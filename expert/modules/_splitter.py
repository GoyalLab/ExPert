# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import lightning.pytorch as pl
import numpy as np
from typing import Literal
from sklearn.model_selection import train_test_split

from expert.data import BalancedAnnDataLoader
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._ann_dataloader import AnnDataLoader

import logging
log = logging.getLogger(__name__)

# Fixed, LOCAL seed for the eval-split per-group subsample ONLY. Deliberately a
# module constant, NOT scvi settings.seed / np.random.seed / torch.manual_seed:
# the subsample must be deterministic (identical eval set every epoch -> the
# checkpoint monitor clip_top10_recall_validation stays comparable) WITHOUT ever
# touching the global RNG, so training stays fully stochastic (seeding training
# overfits, per prior experience). Only used when eval_cap_per_group is set.
_EVAL_SUBSAMPLE_SEED = 0


def _peel_unstratifiable(idx, strat, min_count=2):
    """Remove rows whose stratification group is too small for ``train_test_split``.

    Stratification here is on the CROSS ``label x batch``, and a cross always has a long tail:
    a perturbation with 200 cells overall can still appear exactly once in one dataset. sklearn
    refuses the whole split if a single group has fewer than ``min_count`` members, so one such
    cell aborts a run over millions. Those rows are peeled off and handed back to the caller to
    park in a fixed split -- stratification is variance reduction, not correctness, and dropping
    them instead would silently delete data.

    Only reachable when no ``fold_dir`` supplies cached indices; fold-based runs never split here.

    Returns (idx_kept, strat_kept, idx_peeled).
    """
    idx = np.asarray(idx)
    strat = np.asarray(strat)
    if idx.size == 0:
        return idx, strat, idx[:0]
    _, inv, cnt = np.unique(strat, return_inverse=True, return_counts=True)
    ok = cnt[inv] >= min_count
    return idx[ok], strat[ok], idx[~ok]


def _split_rare(rare, frac, seed=_EVAL_SUBSAMPLE_SEED):
    """Divide peeled rows by ``frac`` WITHOUT stratifying, so split proportions are preserved.

    Parking every peeled row on one side looks harmless at 21 cells but scales badly: the second
    stage sees a much finer cross (the same group count over ~20% of the rows), so most groups are
    singletons there and a one-sided dump lands val/test at 350/46 instead of ~198/198. An
    unstratified proportional split keeps the requested sizes whatever the tail's size.
    """
    rare = np.asarray(rare)
    if rare.size == 0:
        return rare, rare
    perm = np.random.default_rng(seed).permutation(rare.size)
    k = int(round(frac * rare.size))
    return rare[perm[:k]], rare[perm[k:]]


def _stratified_split(idx, strat, train_size, shuffle, seed, rare_seed=_EVAL_SUBSAMPLE_SEED):
    """A stratified split that cannot abort the run.

    sklearn imposes two constraints that a ``label x batch`` cross regularly violates:

      * every group needs >= 2 members -- a long-tail problem, handled exactly by peeling the
        tail off and re-attaching it proportionally, so stratification still applies to the 99.9%
        that can carry it;
      * the smaller side must hold at least as many rows as there are groups -- that means the
        cross is finer than the split can represent AT ALL, and the only honest response is to
        drop stratification for this split and say so loudly.

    Either way the run continues. Losing stratification costs eval-set balance; raising a
    ValueError here costs the whole job, hours in.
    """
    ok, s_ok, rare = _peel_unstratifiable(idx, strat)
    try:
        a, b = train_test_split(ok, train_size=train_size, stratify=s_ok,
                                shuffle=shuffle, random_state=seed)
    except ValueError as e:
        log.warning(f'[Setup] stratified split impossible ({e}); falling back to an '
                    f'UNSTRATIFIED split over {len(idx):,} cells.')
        a, b = train_test_split(np.asarray(idx), train_size=train_size, shuffle=shuffle,
                                random_state=seed)
        return np.asarray(a), np.asarray(b)
    if len(rare):
        log.warning(f'[Setup] {len(rare):,}/{len(idx):,} cells sit in a label x batch group too '
                    f'small to stratify; those are split unstratified.')
    _f = len(a) / max(len(a) + len(b), 1)
    r_a, r_b = _split_rare(rare, _f, seed=rare_seed)
    return np.concatenate([np.asarray(a), r_a]), np.concatenate([np.asarray(b), r_b])


class DataSplitter(pl.LightningDataModule):
    """
    General data splitter supporting both supervised and contrastive training setups.

    loader_type: {'supervised', 'contrastive', 'default'}
        Controls which data loader is used.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        loader_type: Literal['balanced', 'default'] = 'balanced',
        train_size: float | None = 0.9,
        validation_size: float | None = None,
        test_size: float | None = None,
        shuffle_set_split: bool = True,
        n_samples: int | None = 500,
        batch_size: int = 512,
        ctrl_class: str | None = None,
        ctrl_frac: float | None = 1.0,
        last_first: bool = True,
        shuffle: bool = True,
        pin_memory: bool = False,
        use_copy: bool = True,
        use_special_for_split: list[str] = ['train'],
        extra_ctx_col: str | None = None,
        test_context_labels: np.ndarray | str | None = None,
        available_classes: np.ndarray | None = None,
        external_indexing: list[np.ndarray, np.ndarray, np.ndarray] | None = None,
        cache_indices: dict[str, np.ndarray] | None = None,
        drop_last: bool = False,
        use_balanced_weights: bool = True,
        use_balanced_epochs: bool = False,
        use_contrastive_loader: bool = False,
        n_classes_per_batch: int | None = None,
        n_samples_per_class: int | None = None,
        min_contexts_per_class: int | None = None,
        cap_per_group: int | None = None,
        eval_cap_per_group: int | None = None,   # eval splits: max cells per (dataset x pert) group (val/test/ood_test + zs arms); None = off
        min_dataset_fraction: float = 0.5,
        weights_mode: Literal['labels', 'batches', 'both'] = 'both',
        zs_adata_manager: AnnDataManager | None = None,   # 2nd manager: unseen-perturbation subset
        zs_split_col: str = 'zs_split',                   # obs col tagging id_ood_pert / ood_ood_pert
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.loader_type = loader_type
        self.train_size_is_none = not bool(train_size)
        self.train_size = 0.9 if self.train_size_is_none else float(train_size)
        self.validation_size = validation_size
        self.test_size = test_size
        self.shuffle_set_split = shuffle_set_split
        self.drop_last = drop_last
        # Consume shared_mem_x HERE (not per-loader) so it never leaks into any
        # torch DataLoader's kwargs. Flag the manager so create_torch_dataset moves
        # the in-RAM sparse X into shared memory once, before any split captures it.
        if kwargs.pop("shared_mem_x", False):
            adata_manager._want_shared_x = True
        self.data_loader_kwargs = kwargs

        # loader-specific parameters
        self.n_samples = n_samples
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.min_contexts_per_class = min_contexts_per_class
        self.cap_per_group = cap_per_group
        self.eval_cap_per_group = eval_cap_per_group
        self.min_dataset_fraction = min_dataset_fraction
        self.use_balanced_weights = use_balanced_weights
        self.use_balanced_epochs = use_balanced_epochs
        self.use_contrastive_loader = use_contrastive_loader
        self.weights_mode = weights_mode
        
        self.batch_size = batch_size
        self.ctrl_class = ctrl_class
        self.ctrl_frac = ctrl_frac
        self.last_first = last_first
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.use_copy = use_copy
        self.test_context_labels = test_context_labels
        self.external_indexing = external_indexing
        self.cache_indices = cache_indices
        self.use_special_for_split = use_special_for_split
        self.available_classes = available_classes

        # === Label and context extraction ===
        self.indices = np.arange(adata_manager.adata.n_obs)
        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        self.labels = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.labels.attr_name,
            labels_state_registry.original_key,
        ).ravel()
        batch_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        self.batches = get_anndata_attribute(
            adata_manager.adata,
            adata_manager.data_registry.batch.attr_name,
            batch_state_registry.original_key,
        ).ravel()
        # Set extra column if it exists
        if extra_ctx_col is not None and extra_ctx_col in adata_manager.adata.obs:
            self.contexts = adata_manager.adata.obs[extra_ctx_col].values
        # Fall back to batches as context labels
        else:
            self.contexts = None
        # === Zeroshot (unseen-perturbation) eval subset, served as extra val loaders ===
        # A SEPARATE manager (its LABELS_KEY maps to full-bank indices so the model can
        # score held-out genes). We only need the split tags -> arm indices; the two arms
        # use a plain AnnDataLoader, so no balanced-loader labels/batches are required.
        self.zs_adata_manager = zs_adata_manager
        self.zs_split_col = zs_split_col
        if zs_adata_manager is not None and zs_split_col in zs_adata_manager.adata.obs:
            self.zs_split = np.asarray(zs_adata_manager.adata.obs[zs_split_col].astype(str).values)
        else:
            self.zs_adata_manager = None
            self.zs_split = None
        # Per-cell (pert, dataset) for the zs arms (index the zs_adata_manager.adata,
        # NOT the training adata) — used by the eval per-group cap. Guarded: if the zs
        # manager lacks these registries the cap simply no-ops for the zs arms.
        self.zs_labels, self.zs_batches = None, None
        if self.zs_adata_manager is not None:
            try:
                _zl = self.zs_adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
                self.zs_labels = get_anndata_attribute(
                    self.zs_adata_manager.adata,
                    self.zs_adata_manager.data_registry.labels.attr_name, _zl.original_key).ravel()
                _zb = self.zs_adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
                self.zs_batches = get_anndata_attribute(
                    self.zs_adata_manager.adata,
                    self.zs_adata_manager.data_registry.batch.attr_name, _zb.original_key).ravel()
            except Exception as e:
                log.warning(f'[EvalCap] could not read zs labels/batches for per-group cap: {e}; zs arms will not be capped.')
        # Pre-setup() estimates of split sizes
        if self.cache_indices is not None:
            self.n_train = len(self.cache_indices.get('train', []))
            self.n_val = max(1, len(self.cache_indices.get('val', [])))
        else:
            total = self.indices.shape[0]
            if self.validation_size is not None and self.test_size is not None:
                val_ratio = self.validation_size / (self.validation_size + self.test_size)
            else:
                val_ratio = 0.5
            self.n_train = int(total * self.train_size)
            self.n_val = max(1, int(total * (1 - self.train_size) * val_ratio))

    def setup(self, stage: str | None = None):
        """Split indices in train/test/val sets."""
        # === Split logic (identical to contrastive) ===
        if self.cache_indices is None:
            # Define OOD test indices optionally by different label
            split_ctxs = self.contexts if self.contexts is not None else self.batches
            # Always statify based on batch label
            ref_ctxs = self.batches
            if self.test_context_labels is not None:
                # Ensure test contexts is an interable
                test_ctxs = [self.test_context_labels] if isinstance(self.test_context_labels, str) else self.test_context_labels
                test_mask = np.isin(split_ctxs, test_ctxs)
                base_idx = self.indices[~test_mask]
                ood_test_idx = self.indices[test_mask]
            else:
                base_idx = self.indices
                ood_test_idx = []

            # Filter indices for available labels only if given
            if self.available_classes is not None:
                log.info(f'[Setup] Subsetting indices to {len(self.available_classes)}/{len(np.unique(self.labels[base_idx]))} available classes.')
                avail_mask = np.isin(self.labels[base_idx], self.available_classes)
                base_idx = base_idx[avail_mask]

            # Create joint stratification labels
            strat_labels = np.array([
                f"{lbl}_{ctx}" for lbl, ctx in zip(self.labels[base_idx], ref_ctxs[base_idx])
            ])

            # First split: train vs (val + test). Groups too small to stratify go to train.
            train_idx, temp_idx = _stratified_split(
                base_idx, strat_labels, self.train_size, self.shuffle_set_split, settings.seed,
            )
            # Recompute strat labels for remaining set
            temp_strat = np.array([
                f"{lbl}_{ctx}" for lbl, ctx in zip(self.labels[temp_idx], ref_ctxs[temp_idx])
            ])

            # Determine val/test proportions relative to temp set
            if self.validation_size is not None and self.test_size is not None:
                val_ratio = self.validation_size / (self.validation_size + self.test_size)
            # Use 50/50
            else:
                val_ratio = 0.5
            # Split remaining cells into validation and id test. A group with 2-3 members in
            # base leaves exactly one member here, so the same peel is needed a second time.
            val_idx, test_idx = _stratified_split(
                temp_idx, temp_strat, val_ratio, self.shuffle_set_split, settings.seed,
                rare_seed=_EVAL_SUBSAMPLE_SEED + 1,
            )
            # Create split indices
            indices_train = np.array(train_idx)
            indices_val = np.array(val_idx)
            indices_test = np.array(test_idx)
            indices_ood_test = np.array(ood_test_idx)
        else:
            indices_train = np.array(self.cache_indices.get("train"))
            indices_val = np.array(self.cache_indices.get("val"))
            indices_test = np.array(self.cache_indices.get("test"))
            indices_ood_test = np.array(self.cache_indices.get("ood_test"))

        # === Assign splits ===
        self.train_idx = indices_train.astype(int)
        self.val_idx = indices_val.astype(int)
        self.test_idx = indices_test.astype(int)
        self.ood_test_idx = indices_ood_test.astype(int)
        # Setup stats
        self.n_train = self.train_idx.shape[0]
        self.n_val = self.val_idx.shape[0]
        self.n_test = self.test_idx.shape[0]
        self.n_ood_test = self.ood_test_idx.shape[0]

        # === Subset metadata ===
        self.train_labels = self.labels[self.train_idx]
        self.train_batches = self.batches[self.train_idx]
        self.val_labels = self.labels[self.val_idx]
        self.val_batches = self.batches[self.val_idx]
        self.test_labels = self.labels[self.test_idx] if len(self.test_idx) else None
        self.test_batches = self.batches[self.test_idx] if len(self.test_idx) else None
        self.ood_test_labels = self.labels[self.ood_test_idx] if len(self.ood_test_idx) else None
        self.ood_test_batches = self.batches[self.ood_test_idx] if len(self.ood_test_idx) else None

        # === Zeroshot arm indices (into the zs_adata_manager, NOT the training adata) ===
        if self.zs_adata_manager is not None and self.zs_split is not None:
            zi = np.arange(self.zs_adata_manager.adata.n_obs)
            self.id_ood_pert_idx = zi[self.zs_split == 'id_ood_pert'].astype(int)
            self.ood_ood_pert_idx = zi[self.zs_split == 'ood_ood_pert'].astype(int)
        else:
            self.id_ood_pert_idx = np.array([], dtype=int)
            self.ood_ood_pert_idx = np.array([], dtype=int)
        self.n_id_ood_pert = self.id_ood_pert_idx.shape[0]
        self.n_ood_ood_pert = self.ood_ood_pert_idx.shape[0]

        # === Deterministic per-(dataset x pert)-group cap on eval splits ===
        # Cuts validation wall-clock while keeping every label x dataset combo represented;
        # fixed local seed -> identical subset every epoch (checkpoint monitor stays
        # comparable). Canonical *_idx above stay FULL (saved/reproducible split + post-
        # training eval read them); only these _eval_idx views are thinned. No-op when unset.
        self._eval_idx = {}
        if self.eval_cap_per_group:
            for _m, _idx in [('val', self.val_idx), ('test', self.test_idx), ('ood_test', self.ood_test_idx),
                             ('id_ood_pert', self.id_ood_pert_idx), ('ood_ood_pert', self.ood_ood_pert_idx)]:
                _cap = self._cap_eval_indices(_m, _idx)
                self._eval_idx[_m] = _cap
                if _idx is not None and len(_cap) != len(_idx):
                    log.info(f'[EvalCap] {_m}: {len(_idx)} -> {len(_cap)} cells '
                             f'(<= {self.eval_cap_per_group}/(dataset x pert) group)')

        # === Loader selection === use AnnDataLoader as fallback
        self.loader_class_train = AnnDataLoader
        self.loader_class_val = AnnDataLoader
        self.loader_class_test = AnnDataLoader
        self.loader_class_ood_test = AnnDataLoader

        # Assign special loader types
        if "train" in self.use_special_for_split:
            if self.loader_type == "balanced":
                self.loader_class_train = BalancedAnnDataLoader
                
    def _cap_eval_indices(self, mode: str, indices: np.ndarray) -> np.ndarray:
        """Deterministically subsample an eval split to at most ``eval_cap_per_group``
        cells per (dataset x perturbation) group — cuts validation wall-clock while
        keeping every label x dataset combination represented (a flat total cap would
        drop small groups entirely).

        Uses a LOCAL np.random.RandomState with a fixed module seed: never touches the
        global / training RNG (training stays stochastic), yet the evaluated subset is
        identical every epoch so clip_top10_recall_validation is comparable across epochs
        (safe for the checkpoint monitor + early stopping). No-op unless eval_cap_per_group
        is set; the canonical *_idx arrays are never modified.
        """
        cap = self.eval_cap_per_group
        if not cap or cap <= 0 or mode == "train" or indices is None or len(indices) == 0:
            return indices
        if mode in ("id_ood_pert", "ood_ood_pert"):
            labs, bats = self.zs_labels, self.zs_batches
        else:
            labs, bats = self.labels, self.batches
        if labs is None or bats is None:
            return indices
        offset = {"val": 1, "test": 2, "ood_test": 3, "id_ood_pert": 4, "ood_ood_pert": 5}.get(mode, 0)
        rng = np.random.RandomState(_EVAL_SUBSAMPLE_SEED + offset)
        # group by (pert, dataset) via a combined string key; argsort-based grouping is O(n log n)
        keys = np.char.add(np.char.add(labs[indices].astype(str), "|"), bats[indices].astype(str))
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        change = np.where(sk[1:] != sk[:-1])[0] + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(sk)]])
        kept = []
        for s, e in zip(starts, ends):
            grp = order[s:e]
            if e - s > cap:
                grp = rng.choice(grp, size=int(cap), replace=False)
            kept.append(grp)
        sel = np.concatenate(kept)
        return np.sort(indices[sel])  # ascending adata order for read locality; order irrelevant to epoch-mean metrics

    def get_base_kwargs(self, mode: str):
        """Get base data loader keyword arguments for a specific data split."""
        if mode == "train":
            indices = self.train_idx
        elif mode == "val":
            indices = self.val_idx
        elif mode == "test":
            indices = self.test_idx
        elif mode == "ood_test":
            indices = self.ood_test_idx
        elif mode == "id_ood_pert":
            indices = self.id_ood_pert_idx
        elif mode == "ood_ood_pert":
            indices = self.ood_ood_pert_idx
        else:
            raise ValueError(f"Invalid split {mode}")
        # Deterministic per-(dataset x pert)-group eval cap (if configured); train never capped.
        indices = getattr(self, "_eval_idx", {}).get(mode, indices)
        kwargs = {
            "indices": indices,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "pin_memory": self.pin_memory,
            "batch_size": self.batch_size,
            **self.data_loader_kwargs,
        }
        # val/test/ood_test are served as separate dataloaders that Lightning's
        # CombinedLoader iterates *on top of* the train loader's persistent
        # workers. Each split inheriting train's full num_workers forked enough
        # processes off the ~280GB in-memory parent to OOM (train(6)+val(6)+
        # test(6) at first val). One persistent worker per eval split is plenty
        # — eval isn't throughput-critical — and caps the peak at 6 train + 3
        # eval = 9 simultaneous forks, well inside the headroom.
        import sys
        if sys.platform == "darwin":
            # macOS uses spawn (not fork): DataLoader workers re-pickle the
            # in-memory dataset instead of inheriting it cheaply, and require an
            # ``if __name__ == '__main__'`` guard. Force single-process loading
            # for every split to avoid both.
            kwargs["num_workers"] = 0
            kwargs["persistent_workers"] = False
        elif mode != "train":
            # ON GPU: 1 worker, unchanged. The cap exists because eval splits fork off the
            # in-memory parent and train(6)+val(6)+test(6) OOM'd a ~280GB parent at first val.
            # ON CPU: that pressure is different -- a CPU run has spare cores by definition
            # (128 allocated, ~6 used by the train loader) and the dense cache is mmap'd, so
            # forked workers share one copy instead of duplicating it. Lightning suggests
            # num_workers=127 here; that is naive, it counts cores and knows nothing about the
            # fork footprint, so this stays deliberately small.
            # EXPECT LITTLE: val+test is ~27k cells = ~54 batches at batch 512, which one worker
            # reads in seconds. The measured non-training time (~47 s/epoch on a 128-core run) is
            # dominated by METRIC computation, not loading, and no worker count fixes that.
            import torch as _torch
            _cpu_run = not _torch.cuda.is_available()
            kwargs["num_workers"] = 4 if _cpu_run else 1
            kwargs["persistent_workers"] = True
        if mode != "train":
            # PRIVATE GENERATOR for the eval loaders. Creating a DataLoader iterator draws its worker base seed from
            # the given generator, else from the GLOBAL torch RNG. With persistent workers that happens once per
            # process: at epoch 0 in an uninterrupted run, but at the first validation of a RESUMED one -- extra
            # global draws that made a resumed run diverge from the uninterrupted one. Eval loaders never shuffle
            # and the workers draw nothing, so the seed source changes no data; training keeps the global RNG.
            import torch as _torch
            kwargs["generator"] = _torch.Generator()
        return kwargs
         
    def get_special_kwargs(self, mode: str):
        """Get kewword arguments for special dataloaders of a given split."""
        if mode == "train":
            labels = self.train_labels
            batches = self.train_batches
        elif mode == "val":
            labels = self.val_labels
            batches = self.val_batches
        elif mode == "test":
            labels = self.test_labels
            batches = self.test_batches
        elif mode == "ood_test":
            labels = self.ood_test_labels
            batches = self.ood_test_batches
        else:
            raise ValueError(f"Invalid split {mode}")
        return {
            "labels": labels,
            "batches": batches,
            "use_balanced_weights": self.use_balanced_weights,
            "use_balanced_epochs": self.use_balanced_epochs,
            "n_classes_per_batch": self.n_classes_per_batch,
            "n_samples_per_class": self.n_samples_per_class,
            "min_contexts_per_class": self.min_contexts_per_class,
            "use_contrastive_loader": self.use_contrastive_loader if mode == "train" else False,
            "min_dataset_fraction": self.min_dataset_fraction,
            "n_samples": self.n_samples,
            "cap_per_group": self.cap_per_group,
            "weights_mode": self.weights_mode
        }
        
    def _get_dataloader(self, mode: str):
        """Create dataloader of specific split.

        Args:
            mode (str): Data split, one of train, val, test

        Returns:
            DataLoader class
        """
        # Zeroshot arms: plain AnnDataLoader served from the SEPARATE zs manager.
        if mode in ('id_ood_pert', 'ood_ood_pert'):
            kwargs = self.get_base_kwargs(mode)
            kwargs["shuffle"] = False
            return AnnDataLoader(self.zs_adata_manager, **kwargs)
        kwargs = self.get_base_kwargs(mode)
        if mode in self.use_special_for_split:
            if self.loader_type != "default":
                kwargs.update(self.get_special_kwargs(mode))
        # Only shuffle train dataloader
        if mode != "train":
            kwargs["shuffle"] = False
        # Select dataloader class TODO: replace with a dict
        if mode == "train":
            dataloader_cls = self.loader_class_train
        elif mode == "val":
            dataloader_cls = self.loader_class_val
        elif mode == "test":
            dataloader_cls = self.loader_class_test
        elif mode == "ood_test":
            dataloader_cls = self.loader_class_ood_test
        else:
            raise ValueError(f"Invalid split {mode}")
        return dataloader_cls(self.adata_manager, **kwargs)

    # === Dataloaders ===
    def train_dataloader(self):
        return self._get_dataloader('train')

    @property
    def val_split_names(self) -> list[str]:
        """Names of the splits served by ``val_dataloader``, in order.

        Consumed by ``ExPertTrainingPlan._split_for_dataloader_idx`` to map
        Lightning's ``dataloader_idx`` back to a split name.
        """
        names = []
        if len(self.val_idx) > 0:
            names.append('val')
        if len(self.test_idx) > 0:
            names.append('test')
        if len(self.ood_test_idx) > 0:
            names.append('ood_test')
        if getattr(self, 'n_id_ood_pert', 0) > 0:
            names.append('id_ood_pert')
        if getattr(self, 'n_ood_ood_pert', 0) > 0:
            names.append('ood_ood_pert')
        return names

    def val_dataloader(self):
        # Lightning supports a list of dataloaders here and will pass
        # ``dataloader_idx`` into ``validation_step`` accordingly. Order
        # must match ``val_split_names``.
        loaders = [self._get_dataloader(name) for name in self.val_split_names]
        if not loaders:
            return None
        return loaders[0] if len(loaders) == 1 else loaders

    def test_dataloader(self):
        if len(self.test_idx) == 0:
            return None
        return self._get_dataloader('test')

    def ood_test_dataloader(self):
        if len(self.ood_test_idx) == 0:
            return None
        return self._get_dataloader('ood_test')
