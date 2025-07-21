from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import math
from typing import Optional, Literal
from collections import Counter

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


class RandomContrastiveBatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        cls_labels: list[int] | list[bool],
        batches: list[int] | list[bool],
        batch_size: int = 512,
        max_cells_per_batch: int = 32,
        max_classes_per_batch: int = 16,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
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
        self.batch_sampler_cls = ContrastiveBatchSampler
        # Sample indices
        self._idc = self._sample_idc()

    def _sample_idc(self) -> np.ndarray:
        # Shuffle at epoch level
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
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
        )
        batch_dict = sampler.sample(copy=True, return_details=False)
        return np.array(batch_dict[sampler.IDX_KEY])

    def __iter__(self):
        # Return iter over batched indices
        return iter(self._sample_idc())

    def __len__(self):
        return len(self._idc)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedContrastiveBatchSampler(Sampler):
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
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.cls_labels = np.array(cls_labels)
        self.batches = np.array(batches)
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self.batch_sampler_cls = ContrastiveBatchSampler

    def __iter__(self):
        # Shuffle at epoch level
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
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
        )
        batch_dict = sampler.sample(copy=False, return_details=False)
        full_batches = batch_dict[sampler.IDX_KEY]

        # Evenly divide batches among replicas
        total_batches = len(full_batches)
        if self.drop_last:
            batches_per_replica = total_batches // self.num_replicas
        else:
            batches_per_replica = math.ceil(total_batches / self.num_replicas)

        total_size = batches_per_replica * self.num_replicas
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: total_size]
        assert len(indices) == total_size

        # Shard for this replica
        sharded_batches = full_batches[self.rank : total_size : self.num_replicas]

        # Create sparse efficient generator
        sampler_iter = iter(sharded_batches)
        if self.drop_last:
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in sampler_iter:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self):
        batches_per_replica = math.ceil(len(self.dataset) / self.batch_size / self.num_replicas)
        return batches_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch


class ContrastiveBatchSampler:
    IDX_KEY: str = 'indices'
    CLASSES_PER_BATCH_KEY: str = 'cbp'
    IDC_PER_BATCH_KEY: str = 'ipb'
    CLASS_POOL_INFO_KEY: str = 'cpi'

    def _check_setup(self) -> None:
        # None of the parameters are set, fall back to default split
        if self.max_classes_per_batch is None and self.max_cells_per_batch is None:
            logging.warning(f'Falling back to default max_classes_per_batch=8.')
            self.max_classes_per_batch = 8
            self.max_cells_per_batch = self.batch_size // self.max_classes_per_batch
        # One of the parameters is set, adjust other
        elif self.max_cells_per_batch is not None and self.max_classes_per_batch is None:
            self.max_classes_per_batch = self.batch_size // self.max_cells_per_batch
        elif self.max_classes_per_batch is not None and self.max_cells_per_batch is None:
            self.max_cells_per_batch = self.batch_size // self.max_classes_per_batch
        # Both are set, change nothing
        if self.batch_size % self.max_classes_per_batch != 0:
            logging.warning(f'Batch size is not directly divisible by {self.max_classes_per_batch}. This could lead to performance issues.')

    def __init__(
        self,
        indices: np.ndarray,
        batch_labels: np.ndarray,
        cls_labels: np.ndarray,
        batch_size: int = 480,
        max_cells_per_batch: int | None = 24,
        max_classes_per_batch: int | None = 20,
        last_first: bool = True,
        shuffle_classes: bool = True,
        strict: bool = True,
        sample_mode: Literal['random', 'split', 'counter'] = 'random',
        split_batches: bool = False,
    ):
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self._check_setup()
        self.class_limit = self.max_classes_per_batch if self.max_classes_per_batch is not None else 0
        self.shuffle_classes = shuffle_classes
        self.strict = strict
        self.sample_mode = sample_mode
        self.split_batches = split_batches
        # Setup indices and labels
        self.indices = np.array(indices)
        self.n_indices = len(indices)
        self.cls_labels = pd.Series(cls_labels, name='cls_label')
        self.batch_labels = pd.Series(batch_labels, name='batch')
        # Count number of cells per class
        self.cells_per_class = self.cls_labels.value_counts()
        # Define number of unique classes in data
        self.n_classes = self.cells_per_class.shape[0]
        # Determine number of batches to pick from indices with this batch size
        self.n_batches = np.floor(self.n_indices / self.batch_size)
        # Define pool of all indices per class to choose from
        self.idx_pools = [set(self.indices[self.cls_labels==label]) for label in self.cells_per_class.index.values]
        # Define pool of all possible class indices to choose from
        self.class_pool = list(np.arange(self.n_classes))
        # Sample from smallest classes first to ensure that they will be included in training
        if last_first:
            self.class_pool = self.class_pool[::-1]
        # Draw equal number of cells per batch label
        if self.split_batches:
            self.n_batches = np.floor(self.n_indices / self.batch_size)
            self.unique_batches = self.batch_labels.unique()
            self.n_cells_per_batch = int(self.max_cells_per_batch / len(self.unique_batches))
            self.idx_pools = {
                cls: {
                    b: set(self.indices[(self.cls_labels == cls) & (self.batch_labels == b)])
                    for b in self.unique_batches
                } for cls in self.cells_per_class.index.values
            }

    def _fill_batch_split(self, class_pool: list[int], idx_pools: dict[int, dict[int, set[int]]]) -> tuple[list[int], list[int], list[int]]:
        # Set batch size
        batch_space = self.batch_size
        # Collect indices from random classes for a batch
        batch_idc = []
        # Keep drawing cells from classes until batch bin is full
        i = 0
        classes_in_batch = []
        cells_per_class = []
        # Randomly walk through the available classes
        if self.shuffle_classes:
            _class_pool = list(np.random.permutation(class_pool))
        else:
            _class_pool = list(class_pool)
        
        while batch_space > 0 and len(_class_pool) >= self.class_limit:
            target_class = _class_pool[i]
            # Chcek available space in batch
            batch_space = self.batch_size - len(batch_idc)
            # Get all available indices of that class
            current_class_pool = idx_pools[target_class]
            # Go through every batch label and draw cells
            n_drawn_total = 0
            # Check total number of items in class
            n_min = np.min([len(current_class_pool[bl]) for bl in self.unique_batches])
            if self.n_cells_per_batch > n_min:
                # Remove class from available classes in pool
                _class_pool.remove(target_class)
                # If we don't want strict sampling, just draw as many as there are left in class
                draw_n = n_min
            else:
                draw_n = self.n_cells_per_batch
            # Collect equal amounts of samples from each batch
            for batch_label in self.unique_batches:
                target_pool = current_class_pool[batch_label]

                # Check if we can add enough cells to batch
                if draw_n > batch_space:
                    draw_n = batch_space
                # Draw samples
                if draw_n > 0:
                    # Randomly draw as many cells as we can from that class and add to batch
                    target_idc = np.random.choice(list(target_pool), draw_n, replace=False)
                    batch_idc.extend(target_idc)
                    # Remove those indices from class pool
                    target_pool -= set(target_idc)
                n_drawn_total += draw_n
            # Update total statistics of batch
            if n_drawn_total > 0:
                classes_in_batch.append(i)
                cells_per_class.append(draw_n)
            # Move class indexer, because we just looked at a class pool
            i += 1
            # Cycle through all classes repeatedly until either batch is full or classes are empty
            if i >= len(_class_pool):
                i = 0
            
        return batch_idc, classes_in_batch, cells_per_class

    def _fill_batch(self, class_pool: list[int], idx_pools: list[set[int]]) -> tuple[list[int], list[int], list[int]]:
        # Set batch size
        batch_space = self.batch_size
        # Collect indices from random classes for a batch
        batch_idc = []
        # Keep drawing cells from classes until batch bin is full
        i = 0
        classes_in_batch = []
        cells_per_class = []
        # Randomly walk through the available classes
        if self.shuffle_classes:
            _class_pool = list(np.random.permutation(class_pool))
        else:
            _class_pool = list(class_pool)
        
        while batch_space > 0 and len(_class_pool) >= self.class_limit:
            target_class = _class_pool[i]
            cls_idx = i
            batch_space = self.batch_size - len(batch_idc)
            # Get all available indices of that class
            target_pool = idx_pools[target_class]
            # Check number of available indices in class
            n_target_idc = len(target_pool)
            # Try to draw max_cells_per_class if we have that many
            if self.max_cells_per_batch <= n_target_idc:
                draw_n = self.max_cells_per_batch
            else:
                # Remove class from available classes in pool
                _class_pool.remove(target_class)
                # If we don't want strict sampling, just draw as many as there are left in class
                draw_n = n_target_idc
            # Move class indexer, because we just looked at a class pool
            i += 1
            # Check if we can add enough cells to batch
            if draw_n > batch_space:
                draw_n = batch_space
            # Cycle through all classes repeatedly until either batch is full or classes are empty
            if i >= len(_class_pool):
                i = 0
            if draw_n > 0:
                # Randomly draw as many cells as we can from that class and add to batch
                target_idc = np.random.choice(list(target_pool), draw_n, replace=False)
                batch_idc.extend(target_idc)
                # Remove those indices from class pool
                target_pool -= set(target_idc)
                classes_in_batch.append(cls_idx)
                cells_per_class.append(draw_n)
        return batch_idc, classes_in_batch, cells_per_class

    def _fill_batch_undersampled(self, class_pool: list[int], idx_pools: list[set[int]]) -> tuple[list[int], list[int], list[int]]:
        batch_idc = []
        classes_in_batch = []
        cells_per_class = []
        batch_space = self.batch_size
        usage_counter = getattr(self, 'class_usage_counter', Counter())

        # Filter out classes with no remaining indices
        class_pool = [cls for cls in class_pool if len(idx_pools[cls]) > 0]

        # Prioritize under-sampled classes
        if self.shuffle_classes:
            class_pool = sorted(class_pool, key=lambda c: (usage_counter[c], np.random.rand()))
        else:
            class_pool = sorted(class_pool, key=lambda c: usage_counter[c])

        i = 0
        while batch_space > 0 and len(class_pool) >= self.class_limit:
            target_class = class_pool[i]
            pool = idx_pools[target_class]

            # Skip if exhausted
            if len(pool) == 0:
                i += 1
                if i >= len(class_pool):
                    break
                continue

            draw_n = min(self.max_cells_per_batch, len(pool), batch_space)
            target_idc = np.random.choice(list(pool), draw_n, replace=False)
            pool -= set(target_idc)

            batch_idc.extend(target_idc)
            classes_in_batch.append(target_class)
            cells_per_class.append(draw_n)
            usage_counter[target_class] += 1

            batch_space = self.batch_size - len(batch_idc)
            i += 1
            if i >= len(class_pool):
                i = 0

        self.class_usage_counter = usage_counter
        return batch_idc, classes_in_batch, cells_per_class


    def sample(self, copy: bool = False, return_details: bool = False) -> dict[str, np.ndarray | list]:
        batches = []
        n_classes_per_batch = []
        n_idc_per_batch = []
        classes_in_batches = []

        # Make copies of classes and indices
        if copy:
            class_pool = deepcopy(self.class_pool)
            idx_pools = deepcopy(self.idx_pools)
        else:
            class_pool = self.class_pool
            idx_pools = self.idx_pools
        # Determine sample function
        if self.sample_mode == 'split':
            sample_fn = self._fill_batch_split
        elif self.sample_mode == 'counter':
            sample_fn = self._fill_batch_undersampled
        else:
            sample_fn = self._fill_batch
        # Draw cells for each batch
        for _ in np.arange(self.n_batches):
            # Fill batch
            batch_idc, classes_in_batch, cells_per_class = sample_fn(class_pool=class_pool, idx_pools=idx_pools)
            # Check for invalid batches
            if len(batch_idc) != self.batch_size:
                continue
            # If strict, only include batch, if the composition is perfect
            if self.strict and (len(np.unique(classes_in_batch)) != self.max_classes_per_batch or np.any(np.array(cells_per_class) != self.max_cells_per_batch)):
                continue
            if self.shuffle_classes:
                batch_idc = np.random.permutation(batch_idc)
            batches.extend(batch_idc)
            n_classes_per_batch.append(cells_per_class)
            n_idc_per_batch.append(len(batch_idc))
            classes_in_batches.append(classes_in_batch)
        # Count how many cells are left in each class after drawing
        class_pool_space = [len(c) for c in idx_pools]
        if return_details:
            return {
                self.IDX_KEY: np.array(batches),
                self.CLASSES_PER_BATCH_KEY: n_classes_per_batch,
                self.IDC_PER_BATCH_KEY: np.array(n_idc_per_batch),
                self.CLASS_POOL_INFO_KEY: np.array(class_pool_space)
            }
        else:
            return {
                self.IDX_KEY: np.array(batches)
            }
        

        