from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import math
from typing import Optional
from rdflib import Dataset
import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class RandomContrastiveBatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        cls_labels: list[int] | list[bool],
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

        # Create batches using contrastive logic
        sampler = self.batch_sampler_cls(
            indices=indices,
            cls_labels=labels,
            batch_size=self.batch_size,
            max_cells_per_batch=self.max_cells_per_batch,
            max_classes_per_batch=self.max_classes_per_batch,
            last_first=True,
            shuffle_classes=self.shuffle,
        )
        batch_dict = sampler.sample(copy=False, return_details=False)
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

        # Create batches using contrastive logic
        sampler = self.batch_sampler_cls(
            indices=indices,
            cls_labels=labels,
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
        cls_labels: np.ndarray,
        batch_size: int = 480,
        max_cells_per_batch: int | None = 24,
        max_classes_per_batch: int | None = 20,
        last_first: bool = True,
        shuffle_classes: bool = True,
    ):
        self.batch_size = batch_size
        self.max_cells_per_batch = max_cells_per_batch
        self.max_classes_per_batch = max_classes_per_batch
        self._check_setup()
        self.shuffle_classes = shuffle_classes
        # Setup indices and labels
        self.indices = np.array(indices)
        self.n_indices = len(indices)
        self.cls_labels = pd.Series(cls_labels)
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
        
        while batch_space > 0 and len(_class_pool) > 0:
            target_class = _class_pool[i]
            batch_space = self.batch_size - len(batch_idc)
            # Get all available indices of that class
            target_pool = idx_pools[target_class]
            # Check number of available indices in class
            n_target_idc = len(target_pool)
            # Try to draw max_cells_per_class if we have that many
            if self.max_cells_per_batch <= n_target_idc:
                draw_n = self.max_cells_per_batch
            else:
                # Just draw as many as there are left in class
                draw_n = n_target_idc
                # Remove class from class pool
                _class_pool.remove(target_class)
                
            # Check if we can add enough cells to batch
            if draw_n > batch_space:
                draw_n = batch_space
            i += 1
            if draw_n > 0:
                # Randomly draw as many cells as we can from that class and add to batch
                target_idc = np.random.choice(list(target_pool), draw_n, replace=False)
                batch_idc.extend(target_idc)
                # Remove those indices from class pool
                target_pool -= set(target_idc)
                classes_in_batch.append(i)
                cells_per_class.append(draw_n)
            # Cycle through all classes repeatedly until either batch is full or classes are empty
            if i >= len(_class_pool):
                i = 0
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
        # Draw cells for each batch
        for _ in np.arange(self.n_batches):
            batch_idc, classes_in_batch, cells_per_class = self._fill_batch(class_pool=class_pool, idx_pools=idx_pools)
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
        

        