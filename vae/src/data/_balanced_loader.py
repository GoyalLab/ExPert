import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler, Sampler
from copy import deepcopy
from collections import defaultdict
from typing import List, Iterator, Optional, Union, Iterable
import logging
from sklearn.utils.class_weight import compute_class_weight

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager

logger = logging.getLogger(__name__)


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
        

        if "num_workers" not in kwargs:
            kwargs["num_workers"] = settings.dl_num_workers
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = settings.dl_persistent_workers

        self.kwargs = deepcopy(kwargs)

        if sampler is not None and distributed_sampler:
            raise ValueError("Cannot specify both `sampler` and `distributed_sampler`.")

        # --- create weighted sampler
        if sampler is None:
            if not distributed_sampler:
                if use_contrastive_loader:
                    batch_sampler = WeightedContrastiveBatchSampler(
                        indices=indices,
                        class_labels=labels,
                        context_labels=batches,
                        n_classes_per_batch=n_classes_per_batch,
                        n_samples_per_class=n_samples_per_class,
                        min_contexts_per_class=min_contexts_per_class,
                        use_class_weights=self.use_balanced_weights,
                    )
                    # Update batch size based on contrastive sampler
                    batch_size = batch_sampler.batch_size
                else:
                    # compute inverse-frequency weights per class
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    class_weights = 1.0 / counts
                    # Flip the frequencies to favour low-representation classes
                    if self.use_balanced_weights:
                        class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
                    sample_weights = np.array([class_weights[np.where(unique_labels == l)[0][0]] for l in labels])

                    # create weighted sampler
                    sampler = WeightedRandomSampler(
                        weights=torch.DoubleTensor(sample_weights),
                        num_samples=n_samples,
                        replacement=True,
                    )

                    # wrap into batch sampler
                    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

                self.kwargs.update({"sampler": batch_sampler, "batch_size": None, "shuffle": False})
            else:
                raise NotImplementedError("Distributed sampler not implemented for weighted sampling.")
            # do not touch batch size here, sampler gives batched indices
            # This disables PyTorch automatic batching, which is necessary
            # for fast access to sparse matrices
            self.kwargs.update({"batch_size": None, "shuffle": False})
        else:
            self.kwargs.update({"sampler": sampler})

        if iter_ndarray:
            self.kwargs.update({"collate_fn": lambda x: x})

        super().__init__(self.dataset, **self.kwargs)

        logger.info(f"Initialized BalancedAnnDataLoader with {len(self.dataset)} samples, "
                    f"{len(np.unique(labels))} classes, batch size {batch_size}.")

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


class WeightedContrastiveBatchSampler(Sampler[List[int]]):
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
        shuffle_classes: bool = True,
        use_class_weights: bool = True,
        strict: bool = True,
    ):
        super().__init__(None)
        self.indices = np.asarray(indices)
        self.class_labels = np.asarray(class_labels)
        self.context_labels = (
            np.asarray(context_labels) if context_labels is not None else None
        )
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.min_contexts_per_class = min_contexts_per_class
        self.replacement = replacement
        self.shuffle_classes = shuffle_classes
        self.use_class_weights = use_class_weights
        self.strict = strict

        # --- build per-class (+context) pools
        self.class_pools = defaultdict(list)
        if self.context_labels is not None:
            for i, (cls, ctx) in enumerate(zip(self.class_labels, self.context_labels)):
                self.class_pools[(cls, ctx)].append(i)
        else:
            for i, cls in enumerate(self.class_labels):
                self.class_pools[(cls, None)].append(i)

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
                draw_n = max(1, self.n_samples_per_class // len(ctxs))
                chosen.extend(
                    rng.choice(pool, min(draw_n, len(pool)), replace=self.replacement)
                )
            return chosen[: self.n_samples_per_class]
        # Sample random class indices
        else:
            pool = [i for (c, ctx), ids in self.class_pools.items() if c == cls for i in ids]
            if len(pool) == 0:
                return []
            return rng.choice(pool, self.n_samples_per_class, replace=self.replacement).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng()

        for _ in range(self.num_batches):
            # --- sample classes (weighted)
            class_idx = rng.choice(
                len(self.unique_classes),
                size=min(self.n_classes_per_batch, len(self.unique_classes)),
                replace=False,
                p=self.class_weights,
            )
            classes = self.unique_classes[class_idx]

            if self.shuffle_classes:
                rng.shuffle(classes)

            # --- gather samples
            batch = []
            for cls in classes:
                cls_indices = self._sample_class_indices(cls, rng)
                if len(cls_indices) < self.n_samples_per_class and self.strict:
                    break
                batch.extend(cls_indices)
            if len(batch) == self.n_classes_per_batch * self.n_samples_per_class:
                yield batch

    def __len__(self) -> int:
        return self.num_batches
