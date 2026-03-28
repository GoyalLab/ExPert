import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler, Sampler
from copy import deepcopy
from collections import defaultdict, Counter
from typing import List, Iterator, Optional, Union, Iterable, Literal
import logging
from sklearn.utils.class_weight import compute_class_weight

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager

logger = logging.getLogger(__name__)


class _MutableBatchSampler(Sampler):
    """Wrapper that allows swapping the inner sampler after DataLoader init."""

    def __init__(self, inner):
        self.inner = inner
        self.batch_size = getattr(inner, 'batch_size', None)

    def __iter__(self):
        yield from self.inner

    def __len__(self):
        return len(self.inner)

    def swap(self, new_inner):
        self.inner = new_inner
        self.batch_size = getattr(new_inner, 'batch_size', None)


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


class BalancedEpochSampler(Sampler):
    """Fast epoch-level balancing. Pre-computes group arrays, just shuffles and slices each epoch."""

    def __init__(
        self,
        labels: np.ndarray,
        dataset_ids: np.ndarray,
        target_per_group: int | None = None,
        max_ratio: float = 3.0,
        min_samples: int = 4,
        seed: int = 0,
    ):
        self.seed = seed
        self.epoch = 0
        self.min_samples = min_samples

        # Pre-compute group arrays once
        labels = np.asarray(labels)
        dataset_ids = np.asarray(dataset_ids)
        
        self.group_keys = []
        self.group_arrays = []
        
        # Build groups using pandas for speed
        df = pd.DataFrame({'label': labels, 'dataset': dataset_ids, 'idx': np.arange(len(labels))})
        for (label, ds), group in df.groupby(['label', 'dataset']):
            arr = group['idx'].values
            if len(arr) >= min_samples:
                self.group_keys.append((label, ds))
                self.group_arrays.append(arr)

        # Compute target
        group_sizes = np.array([len(a) for a in self.group_arrays])
        if target_per_group is None:
            self.target = int(np.median(group_sizes) * max_ratio)
        else:
            self.target = target_per_group

        # Pre-compute length
        self._len = sum(min(len(a), self.target) for a in self.group_arrays)

    def _resample(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Pre-allocate output
        chunks = []
        for arr in self.group_arrays:
            n = len(arr)
            if n >= self.target:
                # Fast partial shuffle: pick target random indices
                chosen = rng.choice(arr, size=self.target, replace=False)
            else:
                # Upsample with replacement
                chosen = rng.choice(arr, size=self.target, replace=True)
            chunks.append(chosen)
        
        indices = np.concatenate(chunks)
        rng.shuffle(indices)
        return indices

    def __iter__(self):
        return iter(self._resample().tolist())

    def __len__(self):
        return self._len

    def set_epoch(self, epoch):
        self.epoch = epoch


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
        use_contrastive_loader: bool = True,
        n_classes_per_batch: int | None = None,
        n_samples_per_class: int | None = None,
        min_contexts_per_class: int | None = None,
        cap_per_group: int | None = None,
        min_dataset_fraction: float = 0.5,
        use_balanced_epochs: bool = False,
        target_per_group: int | None = None,
        max_ratio: float = 3.0,
        min_group_samples: int = 4,
        weights_mode: Literal['labels', 'batches', 'both'] = 'both',
        **kwargs,
    ):
        # --- prepare indices
        if indices is None:
            indices = np.arange(adata_manager.adata.shape[0])
        else:
            indices = np.asarray(indices)
            if indices.dtype == bool:
                indices = np.where(indices)[0]
        self.indices = indices

        # --- create dataset
        self.dataset = adata_manager.create_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )

        # use balanced weights
        self.use_balanced_weights = use_balanced_weights
        self.weights_mode = weights_mode

        # --- labels
        if labels is None:
            labels = self.dataset[:][REGISTRY_KEYS.LABELS_KEY].flatten()
        labels = np.asarray(labels)
        self.labels = labels
        # contexts
        if batches is None:
            batches = self.dataset[:][REGISTRY_KEYS.BATCH_KEY].flatten()
        batches = np.asarray(batches)
        self.batches = batches
        
        # set number of samples
        self.n_samples = n_samples * batch_size

        # Use fast dataloader defaults when dense cache is active (mmap, multi-worker safe)
        has_dense_cache = getattr(adata_manager, '_dense_cache', None) is not None
        if has_dense_cache:
            if "num_workers" not in kwargs:
                kwargs["num_workers"] = min(4, os.cpu_count() or 1)
            if "persistent_workers" not in kwargs:
                kwargs["persistent_workers"] = kwargs.get("num_workers", 0) > 0
            if "pin_memory" not in kwargs:
                kwargs["pin_memory"] = torch.cuda.is_available()
            if "prefetch_factor" not in kwargs and kwargs.get("num_workers", 0) > 0:
                kwargs["prefetch_factor"] = 3
        else:
            if "num_workers" not in kwargs:
                kwargs["num_workers"] = settings.dl_num_workers
            if "persistent_workers" not in kwargs:
                kwargs["persistent_workers"] = settings.dl_persistent_workers

        self.kwargs = deepcopy(kwargs)

        if sampler is not None and distributed_sampler:
            raise ValueError("Cannot specify both `sampler` and `distributed_sampler`.")

        # Store epoch balancing config — independent of contrastive
        self.use_balanced_epochs = use_balanced_epochs
        self.use_contrastive_loader = use_contrastive_loader
        self._epoch_balancer = None

        if self.use_balanced_epochs:
            self._epoch_balancer = BalancedEpochSampler(
                labels=labels,
                dataset_ids=batches,
                target_per_group=target_per_group,
                max_ratio=max_ratio,
                min_samples=min_group_samples,
            )
            
        # Store contrastive params for rebuilding each epoch
        self._contrastive_params = dict(
            n_classes_per_batch=n_classes_per_batch,
            n_samples_per_class=n_samples_per_class,
            min_contexts_per_class=min_contexts_per_class,
            cap_per_group=cap_per_group,
            min_dataset_fraction=min_dataset_fraction,
            use_class_weights=self.use_balanced_weights,
        )

        # --- create sampler
        if sampler is None:
            if not distributed_sampler:
                if use_balanced_epochs:
                    self._sampler = _MutableBatchSampler(
                        BatchSampler(self._epoch_balancer, batch_size=batch_size, drop_last=drop_last)
                    )
                elif use_contrastive_loader:
                    self._sampler = self._build_contrastive_sampler()
                else:
                    # Get sample weights
                    sample_weights = self._get_sample_weights()
                    # Create sampler
                    sampler = WeightedRandomSampler(
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

    def _build_weighted_sampler(self, batch_size, drop_last):
        """Build weighted random sampler, optionally from balanced subset."""
        if self._epoch_balancer is not None:
            balanced_idx = self._epoch_balancer._resample()
            labels = self.labels[balanced_idx]
            sample_indices = balanced_idx
        else:
            labels = self.labels
            sample_indices = self.indices

        unique_labels, counts = np.unique(labels, return_counts=True)
        if self.use_balanced_weights:
            class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
        else:
            class_weights = 1.0 / counts
        
        sample_weights = np.array([
            class_weights[np.where(unique_labels == l)[0][0]] for l in labels
        ])

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_indices),
            replacement=True,
        )
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

        if self._epoch_balancer is not None:
            return _RemappedBatchSampler(batch_sampler, sample_indices)
        return batch_sampler

    def _build_contrastive_sampler(self):
        """Build contrastive batch sampler, optionally from balanced subset."""
        if self._epoch_balancer is not None:
            balanced_idx = self._epoch_balancer._resample()
            sampler = WeightedContrastiveBatchSampler(
                indices=balanced_idx,
                class_labels=self.labels[balanced_idx],
                context_labels=self.batches[balanced_idx],
                **self._contrastive_params,
            )
            return _RemappedBatchSampler(sampler, balanced_idx)
        else:
            return WeightedContrastiveBatchSampler(
                indices=self.indices,
                class_labels=self.labels,
                context_labels=self.batches,
                **self._contrastive_params,
            )

    def set_epoch(self, epoch: int):
        if self._epoch_balancer is not None:
            self._epoch_balancer.set_epoch(epoch)

        #if self.use_balanced_epochs:
        #    if self.use_contrastive_loader:
        #        new_sampler = self._build_contrastive_sampler()
        #    else:
        #        new_sampler = self._build_weighted_sampler(
        #            self._batch_size, self._drop_last
        #        )
        #    self._mutable_sampler.swap(new_sampler)


class _RemappedBatchSampler(Sampler):
    """Wraps a batch sampler and remaps indices back to original dataset indices."""

    def __init__(self, base_sampler, index_map: np.ndarray):
        self.base_sampler = base_sampler
        self.index_map = index_map
        # Forward batch_size attribute
        self.batch_size = getattr(base_sampler, 'batch_size', None)

    def __iter__(self):
        for batch in self.base_sampler:
            yield [int(self.index_map[i]) for i in batch]

    def __len__(self):
        return len(self.base_sampler)

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


class WeightedContrastiveBatchSamplerOld(Sampler[List[int]]):
    """
    Weighted contrastive sampler for multi-class datasets with optional context diversity.

    Each batch:
      - Samples `n_classes_per_batch` unique classes (weighted by inverse frequency)
      - Samples `n_samples_per_class` samples per class
      - Optionally enforces `min_contexts_per_class` distinct contexts per class

    Parameters
    ----------
    indices : np.ndarray
        Dataset indices (e.g. np.arange(N)).
    class_labels : np.ndarray
        Class label for each sample.
    context_labels : np.ndarray | None
        Optional context/group labels (e.g. batch, donor, dataset).
    n_classes_per_batch : int
        Number of distinct classes per batch.
    n_samples_per_class : int
        Number of samples per class per batch.
    min_contexts_per_class : int
        Minimum number of distinct contexts for a class in a batch.
    replacement : bool
        Whether to sample within each class pool with replacement.
    shuffle_classes : bool
        Whether to shuffle class order before sampling.
    use_class_weights : bool
        If True, rare classes are upweighted (inverse-frequency weighting).
    strict : bool
        Whether to enforce exact batch structure (drop incomplete).
    """

    def __init__(
        self,
        indices: np.ndarray,
        class_labels: np.ndarray,
        context_labels: Optional[np.ndarray] = None,
        n_classes_per_batch: int = 8,
        n_samples_per_class: int = 4,
        min_contexts_per_class: int = 1,
        replacement: bool = True,
        shuffle: bool = True,
        use_class_weights: bool = True,
        strict: bool = True,
        seed: int | None = None,
        **kwargs
    ):
        super().__init__(None)
        self.indices = np.asarray(indices)
        self.class_labels = np.asarray(class_labels)
        self.context_labels = (
            np.asarray(context_labels) if context_labels is not None else None
        )
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.replacement = replacement
        self.shuffle = shuffle
        self.use_class_weights = use_class_weights
        self.strict = strict
        self.seed = seed

        # --- build per-class (+context) pools
        self.class_pools = defaultdict(list)
        if self.context_labels is not None:
            for i, (cls, ctx) in enumerate(zip(self.class_labels, self.context_labels)):
                self.class_pools[(cls, ctx)].append(i)
        else:
            for i, cls in enumerate(self.class_labels):
                self.class_pools[(cls, None)].append(i)
        n_ctx = len(np.unique(self.context_labels))
        self.min_contexts_per_class = min_contexts_per_class if min_contexts_per_class > 0 else n_ctx

        # --- compute class-level info
        self.unique_classes, counts = np.unique(self.class_labels, return_counts=True)
        self.class_counts = dict(zip(self.unique_classes, counts))

        # --- build class sampling weights (inverse frequency)
        if self.use_class_weights:
            inv_freq = 1.0 / (counts + 1e-8)
            self.class_weights = inv_freq / inv_freq.sum()
        else:
            self.class_weights = np.ones_like(counts, dtype=float) / len(counts)
        # Calculate batch size
        self.batch_size = n_classes_per_batch * n_samples_per_class
        self.num_batches = len(self.indices) // self.batch_size

    def _sample_class_indices(self, cls, rng):
        """Sample indices for one class (respecting context diversity if available)."""
        if self.context_labels is not None:
            ctxs = [ctx for (c, ctx) in self.class_pools.keys() if c == cls]
            rng.shuffle(ctxs)
            chosen = []
            # Do not include this class if it is not included in at least min contexts
            if len(ctxs) < self.min_contexts_per_class:
                return []
            # Sample class indices from each context
            for ctx in ctxs:
                pool = self.class_pools[(cls, ctx)]
                if len(pool) == 0:
                    continue
                # Draw more than needed and subset later
                draw_n = int(max(1, np.ceil(self.n_samples_per_class / len(ctxs))))
                chosen.extend(
                    rng.choice(pool, draw_n, replace=True)
                )
            return chosen[: self.n_samples_per_class]
        # Sample random class indices
        else:
            pool = [i for (c, ctx), ids in self.class_pools.items() if c == cls for i in ids]
            if len(pool) == 0:
                return []
            return rng.choice(pool, self.n_samples_per_class, replace=self.replacement).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        # Initiate random generator
        rng = np.random.default_rng(seed=self.seed)

        # Sample batches
        for _ in range(self.num_batches):
            # --- sample classes (weighted)
            class_idx = rng.choice(
                len(self.unique_classes),
                size=min(self.n_classes_per_batch, len(self.unique_classes)),
                replace=False,
                p=self.class_weights,
            )
            classes = self.unique_classes[class_idx]

            # --- gather samples for every class
            batch = []
            for cls in classes:
                cls_indices = self._sample_class_indices(cls, rng)
                # Only add subbatch if it has enough samples
                if len(cls_indices) < self.n_samples_per_class and self.strict:
                    break
                batch.extend(cls_indices)
            # Shuffle batch indices
            if self.shuffle:
                rng.shuffle(batch)
            # Ensure batch size is valid
            if len(batch) == self.n_classes_per_batch * self.n_samples_per_class:
                yield batch

    def __len__(self) -> int:
        return self.num_batches
    

class WeightedContrastiveBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        indices: np.ndarray,
        class_labels: np.ndarray,
        context_labels: Optional[np.ndarray] = None,
        n_classes_per_batch: int = 8,
        n_samples_per_class: int = 4,
        min_contexts_per_class: int = 1,
        cap_per_group: int | None = None,
        min_dataset_fraction: float = 0.5,
        replacement: bool = True,
        shuffle: bool = True,
        use_class_weights: bool = True,
        strict: bool = True,
        seed: int | None = None,
    ):
        super().__init__(None)
        self.indices = np.asarray(indices)
        self.class_labels = np.asarray(class_labels)
        self.context_labels = (
            np.asarray(context_labels) if context_labels is not None else None
        )
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.cap_per_group = cap_per_group
        self.min_dataset_fraction = min_dataset_fraction
        self.replacement = replacement
        self.shuffle = shuffle
        self.use_class_weights = use_class_weights
        self.strict = strict
        self.seed = seed
        self.has_contexts = context_labels is not None

        # --- build pools as contiguous arrays
        pool_map = defaultdict(list)
        if self.has_contexts:
            for i, (cls, ctx) in enumerate(zip(self.class_labels, self.context_labels)):
                pool_map[(cls, ctx)].append(i)
        else:
            for i, cls in enumerate(self.class_labels):
                pool_map[(cls, None)].append(i)

        self.class_pools = {k: np.array(v, dtype=np.int64) for k, v in pool_map.items()}

        # --- precompute per-class structures as arrays for fast iteration
        self.unique_classes, counts = np.unique(self.class_labels, return_counts=True)
        self.n_unique_classes = len(self.unique_classes)
        self.class_counts = dict(zip(self.unique_classes, counts))

        # Per-class: ordered context arrays and their pools
        # Avoids dict lookups in hot path
        self._cls_contexts = {}    # cls -> np.array of contexts
        self._cls_pools = {}       # cls -> list of np.array (aligned with _cls_contexts)
        self._cls_pool_sizes = {}  # cls -> np.array of pool sizes
        self._cls_weights = {}     # cls -> np.array of sampling weights per context

        if self.has_contexts:
            self.unique_contexts = np.unique(self.context_labels)
            n_ctx = len(self.unique_contexts)
            self.min_contexts_per_class = min(
                min_contexts_per_class if min_contexts_per_class > 0 else n_ctx, n_ctx
            )
            self.min_n_datasets = max(2, int(n_ctx * min_dataset_fraction))

            # Group sizes for weight computation
            sizes = np.array([len(v) for v in self.class_pools.values()])
            median_size = np.median(sizes) if len(sizes) > 0 else 1.0

            for cls in self.unique_classes:
                ctxs = []
                pools = []
                pool_sizes = []
                weights = []
                for (c, ctx), pool in self.class_pools.items():
                    if c == cls and len(pool) > 0:
                        ctxs.append(ctx)
                        pools.append(pool)
                        pool_sizes.append(len(pool))
                        weights.append(median_size / max(len(pool), 1))

                self._cls_contexts[cls] = np.array(ctxs)
                self._cls_pools[cls] = pools
                self._cls_pool_sizes[cls] = np.array(pool_sizes, dtype=np.int64)
                w = np.array(weights, dtype=np.float64)
                w /= w.sum() + 1e-12
                self._cls_weights[cls] = w
        else:
            self.unique_contexts = None
            self.min_contexts_per_class = 1
            self.min_n_datasets = 0
            # Flat pools per class
            for cls in self.unique_classes:
                pools = [v for (c, _), v in self.class_pools.items() if c == cls]
                flat = np.concatenate(pools) if pools else np.array([], dtype=np.int64)
                self._cls_pools[cls] = flat

        # --- class sampling weights
        if self.use_class_weights:
            inv_freq = 1.0 / (counts + 1e-8)
            self.class_weights = inv_freq / inv_freq.sum()
        else:
            self.class_weights = np.ones(len(counts), dtype=np.float64) / len(counts)

        self.batch_size = n_classes_per_batch * n_samples_per_class
        self.num_batches = len(self.indices) // self.batch_size

    def _sample_class_indices(self, cls, rng, uncovered=None):
        """Sample indices for one class with weighted context selection."""
        if not self.has_contexts:
            pool = self._cls_pools[cls]
            if len(pool) == 0:
                return None, None
            return rng.choice(pool, self.n_samples_per_class, replace=self.replacement), None

        contexts = self._cls_contexts[cls]
        n_available = len(contexts)

        if n_available < self.min_contexts_per_class:
            return None, None

        weights = self._cls_weights[cls]
        n_ctx = max(self.min_contexts_per_class, min(n_available, self.n_samples_per_class))

        # Boost weights for uncovered contexts
        if uncovered and len(uncovered) > 0:
            boosted = weights.copy()
            for j in range(n_available):
                if contexts[j] in uncovered:
                    boosted[j] *= 10.0  # strong preference, not hard constraint
            boosted /= boosted.sum()
            pick_weights = boosted
        else:
            pick_weights = weights

        # Choose contexts
        pick_n = min(n_ctx, n_available)
        ctx_idx = rng.choice(n_available, size=pick_n, replace=False, p=pick_weights)

        # Compute draws per context (proportional to weight)
        sel_weights = weights[ctx_idx]
        sel_weights /= sel_weights.sum() + 1e-12
        draws = np.maximum(np.floor(sel_weights * self.n_samples_per_class).astype(np.int64), 1)

        if self.cap_per_group is not None:
            np.minimum(draws, self.cap_per_group, out=draws)

        # Distribute leftover
        leftover = self.n_samples_per_class - draws.sum()
        if leftover > 0:
            order = np.argsort(-sel_weights)
            for idx in order:
                cap = (self.cap_per_group - draws[idx]) if self.cap_per_group else leftover
                add = min(leftover, max(cap, 0))
                draws[idx] += add
                leftover -= add
                if leftover <= 0:
                    break

        # Draw samples — single pre-allocated array
        result = np.empty(self.n_samples_per_class, dtype=np.int64)
        pos = 0

        for j in range(pick_n):
            ci = ctx_idx[j]
            pool = self._cls_pools[cls][ci]
            n = min(int(draws[j]), len(pool)) if not self.replacement else int(draws[j])
            if n <= 0:
                continue
            drawn = rng.choice(pool, n, replace=self.replacement)
            end = min(pos + len(drawn), self.n_samples_per_class)
            result[pos:end] = drawn[:end - pos]
            pos = end
            if pos >= self.n_samples_per_class:
                break

        return result[:pos], set(contexts[ctx_idx])

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(seed=self.seed)
        batch_buf = np.empty(self.batch_size, dtype=np.int64)

        for _ in range(self.num_batches):
            # Dataset coverage tracking
            if self.has_contexts:
                uncovered = set(rng.choice(
                    self.unique_contexts,
                    size=min(self.min_n_datasets, len(self.unique_contexts)),
                    replace=False,
                ))
            else:
                uncovered = None

            # Sample classes
            class_idx = rng.choice(
                self.n_unique_classes,
                size=min(self.n_classes_per_batch, self.n_unique_classes),
                replace=False,
                p=self.class_weights,
            )

            pos = 0
            for ci in class_idx:
                cls = self.unique_classes[ci]
                cls_indices, datasets_used = self._sample_class_indices(cls, rng, uncovered)

                if cls_indices is None or (self.strict and len(cls_indices) < self.n_samples_per_class):
                    continue

                if uncovered and datasets_used:
                    uncovered -= datasets_used

                n = min(len(cls_indices), self.batch_size - pos)
                batch_buf[pos:pos + n] = cls_indices[:n]
                pos += n

                if pos >= self.batch_size:
                    break

            if pos >= self.batch_size:
                if self.shuffle:
                    rng.shuffle(batch_buf)
                yield batch_buf.tolist()

    def __len__(self) -> int:
        return self.num_batches
