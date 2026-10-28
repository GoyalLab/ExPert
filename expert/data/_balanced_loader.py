# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler, Sampler
from copy import deepcopy
from collections import Counter
from typing import List, Iterator, Union, Iterable, Literal
import logging
from sklearn.utils.class_weight import compute_class_weight

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute

logger = logging.getLogger(__name__)


class _LargeWeightedRandomSampler(Sampler):
    """Drop-in replacement for ``WeightedRandomSampler`` that supports
    weight tensors with more than 2**24 entries.

    PyTorch's ``WeightedRandomSampler`` calls ``torch.multinomial``, which
    caps category count at ``2**24 = 16_777_216`` (PyTorch limitation; see
    pytorch/pytorch#2576). Beyond that, training on datasets with > 16.7M
    rows crashes with ``RuntimeError: number of categories cannot exceed
    2^24``. For those sizes we precompute the CDF once and draw via
    inverse-transform sampling (``searchsorted``) — same algorithm
    ``np.random.choice`` uses internally, but without redoing the cumsum
    on every epoch. Below the cap we keep PyTorch's native sampler so
    existing generator / worker-seeding semantics are preserved.
    """

    _MULTINOMIAL_CAP = 1 << 24  # 16_777_216

    def __init__(self, weights, num_samples, replacement=True, generator=None):
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().double().numpy()
        else:
            weights_np = np.asarray(weights, dtype=np.float64)
        self._num_samples = int(num_samples)
        self._replacement = bool(replacement)
        self._n = int(weights_np.shape[0])
        if self._n > self._MULTINOMIAL_CAP:
            if not self._replacement:
                raise NotImplementedError(
                    f'{type(self).__name__} only supports replacement=True for '
                    f'n > 2**24 (got n={self._n}).'
                )
            total = float(weights_np.sum())
            if not (total > 0 and np.isfinite(total)):
                raise ValueError(
                    f'{type(self).__name__}: weights must sum to a positive '
                    f'finite value (got sum={total}, n={self._n}).'
                )
            # Cache the CDF once; each epoch only pays O(K) for sampling +
            # O(K log N) for searchsorted, ~50-100x faster than
            # np.random.choice which redoes the cumsum each call.
            self._cdf = np.cumsum(weights_np / total, dtype=np.float64)
            # Clamp the endpoint to 1.0 so a uniform u→1.0 can't searchsorted
            # past the last category.
            self._cdf[-1] = 1.0
            self._fallback = True
            self._inner = None
        else:
            self._inner = WeightedRandomSampler(
                weights=torch.from_numpy(weights_np),
                num_samples=self._num_samples,
                replacement=self._replacement,
                generator=generator,
            )
            self._fallback = False
            self._cdf = None

    def __iter__(self):
        if self._fallback:
            # Process-local RNG. Per-worker reseeding (DataLoader's
            # worker_init_fn) already handles reproducibility across workers.
            u = np.random.random(self._num_samples)
            idx = np.searchsorted(self._cdf, u, side='right')
            # Defensive clip on rare FP edge cases.
            np.clip(idx, 0, self._n - 1, out=idx)
            return iter(idx.tolist())
        return iter(self._inner)

    def __len__(self):
        return self._num_samples


class _IOSortedBatchSampler(Sampler):
    """Wraps a BatchSampler to reorder batches by disk locality.

    Pre-generates all batches for the epoch, then sorts them by median
    index so consecutive batches read from nearby file regions. This
    turns random I/O into roughly sequential I/O on backed AnnData.
    """

    def __init__(self, inner: BatchSampler):
        self.inner = inner
        self.batch_size = getattr(inner, 'batch_size', None)

    def __iter__(self):
        # Materialize all batches, sort by median index for disk locality
        batches = list(self.inner)
        batches.sort(key=lambda b: np.median(b))
        yield from batches

    def __len__(self):
        return len(self.inner)


class BalancedAnnDataLoader(DataLoader):
    """
    DataLoader that applies class-balanced sampling via WeightedRandomSampler.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        labels: np.ndarray | None = None,
        batches: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        n_samples: int = 500,
        batch_size: int = 512,
        sampler: Sampler | None = None,
        drop_last: bool = False,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        iter_ndarray: bool = False,
        distributed_sampler: bool = False,
        load_sparse_tensor: bool = False,
        use_balanced_weights: bool = True,
        # The contrastive batch sampler was removed. These are still ACCEPTED (DataSplitter forwards
        # them) and ignored; use_contrastive_loader=True is refused instead of silently changing
        # the sampler.
        use_contrastive_loader: bool = False,
        n_classes_per_batch: int | None = None,
        n_samples_per_class: int | None = None,
        min_contexts_per_class: int | None = None,
        cap_per_group: int | None = None,
        min_dataset_fraction: float = 0.5,
        # The epoch-balancing sampler was removed; these are still ACCEPTED and ignored, and
        # use_balanced_epochs=True is refused.
        use_balanced_epochs: bool = False,
        target_per_group: int | None = None,
        max_ratio: float = 3.0,
        min_group_samples: int = 4,
        weights_mode: Literal['labels', 'batches', 'both'] = 'both',
        **kwargs,
    ):
        if use_contrastive_loader:
            raise ValueError('use_contrastive_loader=True is no longer supported (the contrastive '
                             'batch sampler was removed); set data.use_contrastive_loader: false.')
        if use_balanced_epochs:
            raise ValueError('use_balanced_epochs=True is no longer supported (the epoch-balancing '
                             'sampler was removed); set data.use_balanced_epochs: false.')
        # --- prepare indices
        if indices is None:
            indices = np.arange(adata_manager.adata.shape[0])
        else:
            indices = np.asarray(indices)
            if indices.dtype == bool:
                indices = np.where(indices)[0]
        self.indices = indices

        # Opt-in: relocate the in-RAM sparse X to shared memory so forked workers
        # mmap one copy (avoids COW duplication). Set on the manager BEFORE the
        # dataset captures adata.X; idempotent across the train/val/test loaders.
        if kwargs.pop("shared_mem_x", False):
            adata_manager._want_shared_x = True

        # --- create dataset
        self.dataset = adata_manager.create_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )

        # use balanced weights
        self.use_balanced_weights = use_balanced_weights
        self.weights_mode = weights_mode

        # --- labels (read from obs registry to avoid loading entire dataset)
        if labels is None:
            labels_state = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
            labels = get_anndata_attribute(
                adata_manager.adata,
                adata_manager.data_registry.labels.attr_name,
                labels_state.original_key,
            ).ravel()[indices]
        labels = np.asarray(labels)
        self.labels = labels
        # contexts (read from obs registry to avoid loading entire dataset)
        if batches is None:
            batch_state = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
            batches = get_anndata_attribute(
                adata_manager.adata,
                adata_manager.data_registry.batch.attr_name,
                batch_state.original_key,
            ).ravel()[indices]
        batches = np.asarray(batches)
        self.batches = batches
        
        # set number of samples
        self.n_samples = n_samples * batch_size

        if "num_workers" not in kwargs:
            kwargs["num_workers"] = settings.dl_num_workers
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = settings.dl_persistent_workers

        self.kwargs = deepcopy(kwargs)

        if sampler is not None and distributed_sampler:
            raise ValueError("Cannot specify both `sampler` and `distributed_sampler`.")

        self.use_balanced_epochs = use_balanced_epochs
        self.use_contrastive_loader = use_contrastive_loader

        # --- create sampler
        if sampler is None:
            if not distributed_sampler:
                # Get sample weights
                sample_weights = self._get_sample_weights()
                # Create sampler
                sampler = _LargeWeightedRandomSampler(
                    weights=torch.DoubleTensor(sample_weights),
                    num_samples=self.n_samples,
                    replacement=True,
                )
                batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
                # Sort batches by disk location for sequential I/O on backed data
                if adata_manager.adata.isbacked:
                    self._sampler = _IOSortedBatchSampler(batch_sampler)
                else:
                    self._sampler = batch_sampler

                self.kwargs.update({"sampler": self._sampler, "batch_size": None, "shuffle": False})
            else:
                raise NotImplementedError("Distributed sampler not implemented for weighted sampling.")
            self.kwargs.update({"batch_size": None, "shuffle": False})
        else:
            self.kwargs.update({"sampler": sampler})

        if iter_ndarray:
            self.kwargs.update({"collate_fn": lambda x: x})
        # Save parameters
        self._batch_size = batch_size
        self._drop_last = drop_last

        super().__init__(self.dataset, **self.kwargs)

        logger.info(
            f"Initialized BalancedAnnDataLoader with {len(self.dataset)} samples, "
            f"{len(np.unique(labels))} classes, batch size {batch_size}, "
            f"Weights: {self.weights_mode}, "
            f"Contrastive: {use_contrastive_loader}, "
            f"Balanced epochs: {self.use_balanced_epochs}."
        )

    def _get_sample_weights(self):
        if self.weights_mode == 'both':
            # Group by dataset x class
            group_keys = list(zip(self.labels, self.batches))
            group_counts = Counter(group_keys)
            
            if self.use_balanced_weights:
                # Sqrt-balanced: reduces imbalance without extreme upsampling
                sample_weights = np.array([
                    1.0 / np.sqrt(group_counts[(l, b)] + 1)
                    for l, b in group_keys
                ])
            else:
                sample_weights = np.array([
                    1.0 / group_counts[(l, b)]
                    for l, b in group_keys
                ])
            # Normalize to mean 1
            sample_weights /= sample_weights.mean()
            return sample_weights
        else:
            # Weight by labels
            if self.weights_mode == 'labels':
                labels = self.labels
            # Weight by batches/datasets
            elif self.weights_mode == 'batches':
                labels = self.batches
            else:
                raise ValueError(f"Unknown weights_mode: {self.weights_mode}")
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            if self.use_balanced_weights:
                class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
            else:
                class_weights = 1.0 / counts
            return np.array([class_weights[np.where(unique_labels == l)[0][0]] for l in labels])

    def set_epoch(self, epoch: int):
        # Forward to the active batch sampler too (if it reseeds per epoch).
        sampler = getattr(self, '_sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

