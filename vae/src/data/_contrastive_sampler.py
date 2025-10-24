from copy import deepcopy
import numpy as np
import pandas as pd
import math
import logging
from typing import Optional, Literal

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


class RandomContrastiveBatchSampler(Sampler):
    """
    Random sampler that creates batches for contrastive learning.
    Each batch includes multiple classes and contexts (batches).
    """

    def __init__(
        self,
        dataset: Dataset,
        cls_labels: list[str] | list[bool],
        batches: list[str] | list[bool],
        batch_size: int = 512,
        max_cells_per_batch: int = 32,
        max_classes_per_batch: int = 16,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        min_contexts_per_class: int = 2,
        ctrl_class: str | None = None,
        ctrl_frac: float = 1.0,
    ):
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.cls_labels = np.array(cls_labels)
        self.batches = np.array(batches)
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self.min_contexts_per_class = min_contexts_per_class
        self.ctrl_class = ctrl_class
        self.ctrl_frac = ctrl_frac
        self.batch_sampler_cls = ContrastiveBatchSampler
        # Sample initial indices
        self._idc = self._sample_idc()
        # Determine full batch size if control cells were added
        self.full_batch_size = self.get_full_batch_size()

    def get_full_batch_size(self):
        if getattr(self, 'sampler') is None:
            raise ValueError('Sample first before determining full batch size.')
        n_ctrl = getattr(self.sampler, 'n_ctrl', None)
        # Add control cells to batch size
        if n_ctrl is not None:
            return self.batch_size + n_ctrl
        # Return default batch size
        return self.batch_size

    def _sample_idc(self) -> np.ndarray:
        # Shuffle once per epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        labels = self.cls_labels[indices]
        batches = self.batches[indices]

        # Create batches using contrastive logic
        self.sampler = self.batch_sampler_cls(
            indices=indices,
            cls_labels=labels,
            batch_labels=batches,
            batch_size=self.batch_size,
            max_cells_per_batch=self.max_cells_per_batch,
            max_classes_per_batch=self.max_classes_per_batch,
            last_first=True,
            shuffle_classes=self.shuffle,
            min_contexts_per_class=self.min_contexts_per_class,
            ctrl_class=self.ctrl_class,
            ctrl_frac=self.ctrl_frac,
        )
        batch_dict = self.sampler.sample(copy=True, return_details=False)
        # Return sampled class indices if no control cells are given
        return np.array(batch_dict[self.sampler.IDX_KEY])

    def __iter__(self):
        return iter(self._sample_idc())

    def __len__(self):
        return len(self._idc)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedContrastiveBatchSampler(Sampler):
    """
    Distributed version of RandomContrastiveBatchSampler that splits batches across ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        cls_labels: list[int] | list[bool],
        batches: list[int] | list[bool],
        batch_size: int = 512,
        max_cells_per_batch: int = 32,
        max_classes_per_batch: int = 16,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        min_contexts_per_class: int = 2,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, should be in [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self.cls_labels = np.array(cls_labels)
        self.batches = np.array(batches)
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self.min_contexts_per_class = min_contexts_per_class
        self.batch_sampler_cls = ContrastiveBatchSampler

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Shuffle dataset indices per epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        labels = self.cls_labels[indices]
        batches = self.batches[indices]

        # Create batches using contrastive logic
        sampler = self.batch_sampler_cls(
            indices=indices,
            cls_labels=labels,
            batch_labels=batches,
            batch_size=self.batch_size,
            max_cells_per_batch=self.max_cells_per_batch,
            max_classes_per_batch=self.max_classes_per_batch,
            last_first=True,
            shuffle_classes=self.shuffle,
            min_contexts_per_class=self.min_contexts_per_class,
        )
        batch_dict = sampler.sample(copy=False, return_details=False)
        full_batches = batch_dict[sampler.IDX_KEY]

        # Split across distributed replicas
        total_batches = len(full_batches)
        if self.drop_last:
            batches_per_replica = total_batches // self.num_replicas
        else:
            batches_per_replica = math.ceil(total_batches / self.num_replicas)

        total_size = batches_per_replica * self.num_replicas
        indices = indices[:total_size]
        sharded_batches = full_batches[self.rank:total_size:self.num_replicas]

        sampler_iter = iter(sharded_batches)
        while True:
            try:
                batch = [next(sampler_iter) for _ in range(self.batch_size)]
                yield batch
            except StopIteration:
                break

    def __len__(self):
        batches_per_replica = math.ceil(len(self.dataset) / self.batch_size / self.num_replicas)
        return batches_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch


class ContrastiveBatchSampler:
    """
    Builds balanced batches for contrastive training. Supports context-aware sampling.
    Ensures that each class has samples drawn from multiple contexts if possible.
    """
    IDX_KEY = "indices"
    CLS_PER_BATCH_KEY = "classes_per_batch"
    IDC_PER_BATCH_KEY = "idc_per_batch"
    CTRL_IDX_KEY = "ctrl_indices"

    def _check_setup(self):
        if self.max_classes_per_batch is None and self.max_cells_per_batch is None:
            logging.warning("No constraints set; defaulting max_classes_per_batch=8.")
            self.max_classes_per_batch = 8
            self.max_cells_per_batch = self.batch_size // self.max_classes_per_batch
        elif self.max_cells_per_batch is not None and self.max_classes_per_batch is None:
            self.max_classes_per_batch = self.batch_size // self.max_cells_per_batch
        elif self.max_classes_per_batch is not None and self.max_cells_per_batch is None:
            self.max_cells_per_batch = self.batch_size // self.max_classes_per_batch
        if self.batch_size % self.max_classes_per_batch != 0:
            logging.warning(f"Batch size not divisible by {self.max_classes_per_batch}.")

    def __init__(
        self,
        indices: np.ndarray,
        batch_labels: np.ndarray,
        cls_labels: np.ndarray,
        batch_size: int = 512,
        max_cells_per_batch: Optional[int] = 32,
        max_classes_per_batch: Optional[int] = 16,
        shuffle_classes: bool = True,
        strict: bool = True,
        sample_mode: Literal["random", "split", "counter"] = "random",
        split_batches: bool = False,
        min_contexts_per_class: int = 2,
        ctrl_class: str | None = None,
        ctrl_frac: float = 1.0,
        **kwargs
    ):
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self._check_setup()
        self.shuffle_classes = shuffle_classes
        self.strict = strict
        self.sample_mode = sample_mode
        self.split_batches = split_batches
        self.min_contexts_per_class = min_contexts_per_class
        self.ctrl_class = ctrl_class
        self.ctrl_frac = ctrl_frac
        self.kwargs = kwargs

        # Index and label setup
        self.indices = np.array(indices)
        self.cls_labels = pd.Series(cls_labels, name="cls_label")
        self.batch_labels = pd.Series(batch_labels, name="batch")
        self.n_indices = len(indices)
        self.unique_batches = np.unique(self.batch_labels)
        # Split control vs non-control indices
        if self.ctrl_class is not None:
            ctrl_mask = self.cls_labels == self.ctrl_class
            self.ctrl_indices = self.indices[ctrl_mask]
            self.ctrl_pools = {
                b: set(self.ctrl_indices[self.batch_labels[self.cls_labels == self.ctrl_class] == b])
                for b in self.unique_batches
            }
            # remove empty control contexts
            self.ctrl_pools = {b: pool for b, pool in self.ctrl_pools.items() if len(pool) > 0}
            # Set some control specific parameters
            n_ctrl_contexts = len(self.ctrl_pools)
            # Save specifics about control batching if we have have control cells available
            if n_ctrl_contexts > 0:
                n_ctrl = self.batch_size * self.ctrl_frac
                self.n_ctrl_per_ctx = int(np.ceil(n_ctrl / n_ctrl_contexts))
                self.n_ctrl = int(self.n_ctrl_per_ctx * n_ctrl_contexts)
            # Disable control cell batching if none are found
            else:
                self.n_ctrl_per_ctx = 0
                self.n_ctrl = 0

            # remove control indices from normal sampling
            self.indices = self.indices[~ctrl_mask]
            self.cls_labels = self.cls_labels[~ctrl_mask]
            self.batch_labels = self.batch_labels[~ctrl_mask]

        # --- build pools for non-control classes
        self.cells_per_class = self.cls_labels.value_counts()
        self.n_classes = self.cells_per_class.shape[0]
        self.n_batches = int(np.floor(self.n_indices / self.batch_size))
        # Merge cell meta information
        index_meta = pd.concat((self.cls_labels.astype("category"), self.batch_labels.astype("category")), axis=1)
        # Set self.indices as meta indices
        index_meta.set_index(self.indices, inplace=True)
        # Create index pools for each class and context
        self.idx_pools = (
            index_meta.groupby(index_meta.columns.tolist(), observed=True)
            .apply(lambda x: set(x.index))
            .unstack(0)
            .to_dict()
        )

    def _eligible_classes(self, idx_pools: dict[int, dict[int, set[int]]]) -> list[int]:
        eligible = []
        for cls, context_dict in idx_pools.items():
            # Check number of contexts per class that still have cells
            n_contexts = sum(len(v) > 0 for v in context_dict.values() if isinstance(v, set))
            if n_contexts >= self.min_contexts_per_class:
                eligible.append(cls)
        return eligible

    def _fill_batch_split(self, idx_pools):
        batch_space = self.batch_size
        batch_idc = []
        classes_in_batch = []
        cells_per_class = []
        # Get classes that still have available indices
        eligible_classes = self._eligible_classes(idx_pools)
        # Randomly shuffle class pool
        if self.shuffle_classes:
            _class_pool = list(np.random.permutation(eligible_classes))
        else:
            _class_pool = eligible_classes
        # Keep drawing from class pools as long as the batch is not full
        i = 0
        while batch_space > 0 and len(_class_pool) > 0:
            target_class = _class_pool[i]
            current_class_pool = idx_pools[target_class]

            # Get number of cells per context
            available_contexts = [b for b, pool in current_class_pool.items() if isinstance(pool, set) and len(pool) > 0]
            # Remove class from pool if it has less than requires contexts
            if len(available_contexts) < self.min_contexts_per_class:
                _class_pool.remove(target_class)
                if i >= len(_class_pool):
                    break
                continue
            # Randomly choose cells from all possible contexts
            chosen_contexts = np.random.choice(
                available_contexts,
                size=max(len(available_contexts), self.min_contexts_per_class),
                replace=False,
            )

            draw_n_per_ctx = max(1, self.max_cells_per_batch // len(chosen_contexts))
            drawn_total = 0
            for ctx in chosen_contexts:
                pool = current_class_pool[ctx]
                draw_n = min(draw_n_per_ctx, len(pool), batch_space - drawn_total)
                if draw_n <= 0:
                    continue
                chosen = np.random.choice(list(pool), draw_n, replace=False)
                batch_idc.extend(chosen)
                pool -= set(chosen)
                drawn_total += draw_n

            if drawn_total > 0:
                classes_in_batch.append(target_class)
                cells_per_class.append(drawn_total)

            batch_space = self.batch_size - len(batch_idc)
            i = (i + 1) % len(_class_pool)

        return batch_idc, classes_in_batch, cells_per_class
    
    def _fill_control_batch(self, ctrl_pools: dict, keep: bool = True) -> np.ndarray:
        # Create batch list
        ctrl_batch = []
        for b, pool in ctrl_pools.items():
            # Determine how many cells to draw
            n_draw = min(self.n_ctrl_per_ctx, len(pool))
            # Choose cells from available pool
            chosen = np.random.choice(list(pool), n_draw, replace=False)
            ctrl_batch.extend(chosen)
            # Remove chosen cells from pool of available control cells
            if not keep:
                ctrl_pools[b] -= set(chosen)
        # Draw with replacement if needed
        if len(ctrl_batch) < self.n_ctrl:
            # Draw extra random cells if there are not enough control cells for each context
            extra = np.random.choice(self.ctrl_indices, self.n_ctrl - len(ctrl_batch), replace=True)
            ctrl_batch.extend(extra)
        # Shuffle the batch once more
        return np.array(ctrl_batch[: self.n_ctrl])
    
    def sample(self, copy: bool = False, return_details: bool = False) -> dict:
        batches = []
        n_classes_per_batch = []
        n_idc_per_batch = []

        # Create a deepcopy of available pools every sample
        if copy:
            idx_pools = deepcopy(self.idx_pools)
            ctrl_pools = deepcopy(self.ctrl_pools)
        # Use class pool and remove cells that have been drawn
        else:
            idx_pools = self.idx_pools
            ctrl_pools = self.ctrl_pools

        for _ in np.arange(self.n_batches):
            batch_idc, classes_in_batch, cells_per_class = self._fill_batch_split(idx_pools=idx_pools)

            if len(batch_idc) != self.batch_size:
                continue

            if self.strict and (
                len(np.unique(classes_in_batch)) < 1
                or np.any(np.array(cells_per_class) == 0)
            ):
                continue
            # Randomly shuffle the indices
            if self.shuffle_classes:
                batch_idc = np.random.permutation(batch_idc)

            batches.extend(batch_idc)
            n_cls = len(classes_in_batch)
            n_batch = len(batch_idc)
            # Randomly sample control indices for each batch
            if self.ctrl_class is not None and len(ctrl_pools) > 0:
                ctrl_batch = self._fill_control_batch(ctrl_pools=ctrl_pools)
                # Shuffle control indices
                if self.shuffle_classes:
                    ctrl_batch = np.random.permutation(ctrl_batch)
                # Add control batch to class batch and update stats
                batches.extend(ctrl_batch)
                n_cls += 1
                n_batch += 1
            # Add stats
            n_classes_per_batch.append(n_cls)
            n_idc_per_batch.append(n_batch)

        # Create batch output dict
        batch = {self.IDX_KEY: np.array(batches)}
        # Add batch details
        if return_details:
            batch.update({
                self.CLS_PER_BATCH_KEY: n_classes_per_batch,
                self.IDC_PER_BATCH_KEY: np.array(n_idc_per_batch),
            })
        # Return batch
        return batch

