from cProfile import label
import copy
import logging

import numpy as np
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import get_anndata_attribute
from scvi.dataloaders._samplers import BatchDistributedSampler

from src.data._contrastive_sampler import DistributedContrastiveBatchSampler, RandomContrastiveBatchSampler

logger = logging.getLogger(__name__)


class ContrastiveAnnDataLoader(DataLoader):
    """DataLoader for loading tensors from AnnData objects.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    indices
        The indices of the observations in `adata_manager.adata` to load.
    batch_size
        Minibatch size to load each iteration. If `distributed_sampler` is `True`,
        refers to the minibatch size per replica. Thus, the effective minibatch
        size is `batch_size` * `num_replicas`.
    shuffle
        Whether the dataset should be shuffled prior to sampling.
    sampler
        Defines the strategy to draw samples from the dataset. Can be any Iterable with __len__
        implemented. If specified, shuffle must not be specified. By default, we use a custom
        sampler that is designed to get a minibatch of data with one call to __getitem__.
    drop_last
        If `True` and the dataset is not evenly divisible by `batch_size`, the last
        incomplete batch is dropped. If `False` and the dataset is not evenly divisible
        by `batch_size`, then the last batch will be smaller than `batch_size`.
    drop_dataset_tail
        Only used if `distributed_sampler` is `True`. If `True` the sampler will drop
        the tail of the dataset to make it evenly divisible by the number of replicas.
        If `False`, then the sampler will add extra indices to make the dataset evenly
        divisible by the number of replicas.
    data_and_attributes
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor) or list of
        such keys. A list can be used to subset to certain keys in the event that more tensors than
        needed have been registered. If ``None``, defaults to all registered data.
    iter_ndarray
        Whether to iterate over numpy arrays instead of torch tensors
    distributed_sampler
        ``EXPERIMENTAL`` Whether to use :class:`~scvi.dataloaders.BatchDistributedSampler` as the
        sampler. If `True`, `sampler` must be `None`.
    load_sparse_tensor
        ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
        :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
        GPUs, depending on the sparsity of the data.
    **kwargs
        Additional keyword arguments passed into :class:`~torch.utils.data.DataLoader`.

    Notes
    -----
    If `sampler` is not specified, a :class:`~torch.utils.data.BatchSampler` instance is
    passed in as the sampler, which retrieves a minibatch of data with one call to
    :meth:`~scvi.data.AnnTorchDataset.__getitem__`. This is useful for fast access to
    sparse matrices as retrieving single observations and then collating is inefficient.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        labels: list[int] | list[bool],
        indices: list[int] | list[bool] | None = None,
        batch_size: int = 512,
        max_cells_per_batch: int = 32,
        max_classes_per_batch: int = 16,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        drop_last: bool = False,
        drop_dataset_tail: bool = False,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        iter_ndarray: bool = False,
        distributed_sampler: bool = False,
        load_sparse_tensor: bool = False,
        **kwargs,
    ):
        # Set or get indices from adata manager
        if indices is None:
            indices = np.arange(adata_manager.adata.shape[0])
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
        self.indices = indices
        # Create dataset
        self.dataset = adata_manager.create_torch_dataset(
            indices=indices,
            data_and_attributes=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )
        # Set labels
        if labels is None:
            labels = self.dataset[REGISTRY_KEYS.LABELS_KEY]
        else:
            if hasattr(labels, "dtype") and labels.dtype is np.dtype("bool"):
                labels = np.where(labels)[0].ravel()
            labels = np.asarray(labels)

        if "num_workers" not in kwargs:
            kwargs["num_workers"] = settings.dl_num_workers
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = settings.dl_persistent_workers

        self.kwargs = copy.deepcopy(kwargs)

        if sampler is not None and distributed_sampler:
            raise ValueError("Cannot specify both `sampler` and `distributed_sampler`.")

        # custom sampler for efficient minibatching on sparse matrices
        if sampler is None:
            if not distributed_sampler:
                # Replace RandomSampler with ContrastiveSampler
                sampler_cls = RandomContrastiveBatchSampler(
                    self.dataset,
                    labels,
                    batch_size,
                    max_cells_per_batch,
                    max_classes_per_batch,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
                sampler = BatchSampler(
                    sampler=sampler_cls,
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            else:
                sampler = DistributedContrastiveBatchSampler(
                    self.dataset,
                    labels,
                    batch_size,
                    max_cells_per_batch,
                    max_classes_per_batch,
                    drop_last=drop_last,
                    shuffle=shuffle,
                )
            # do not touch batch size here, sampler gives batched indices
            # This disables PyTorch automatic batching, which is necessary
            # for fast access to sparse matrices
            self.kwargs.update({"batch_size": None, "shuffle": False})

        self.kwargs.update({"sampler": sampler})

        if iter_ndarray:
            self.kwargs.update({"collate_fn": lambda x: x})

        super().__init__(self.dataset, **self.kwargs)
