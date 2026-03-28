from scvi.data._manager import AnnDataManager, AnnDataManagerValidationCheck
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data._utils import scipy_to_torch_sparse
from torch.utils.data import Subset
import os
import torch
import h5py
from scipy.sparse import issparse
from anndata.abc import CSCDataset, CSRDataset
import pandas as pd
import numpy as np

from src.utils.constants import REGISTRY_KEYS
from collections.abc import Sequence
from scvi.data.fields import AnnDataField
from src.data._cache import DenseCache, try_migrate_cache_to_ssd

SparseDataset = (CSRDataset, CSCDataset)

import logging


class EmbAnnTorchDataset(AnnTorchDataset):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
        ignore_types: list = ['uns', 'varm'],
        dense_cache: DenseCache | None = None,
    ):
        super().__init__(adata_manager, getitem_tensors, load_sparse_tensor)
        self.ignore_types = ignore_types
        self.dense_cache = dense_cache

    def __getitem__(
        self, indexes: int | list[int] | slice
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Fetch data from the :class:`~anndata.AnnData` object.

        Parameters
        ----------
        indexes
            Indexes of the observations to fetch. Can be a single index, a list of indexes, or a
            slice.

        Returns
        -------
        Mapping of data registry keys to arrays of shape ``(n_obs, ...)``.
        """
        if isinstance(indexes, int):
            indexes = [indexes]  # force batched single observations

        if self.adata_manager.adata.isbacked and self.dense_cache is None and isinstance(indexes, list | np.ndarray):
            # need to sort indexes for h5py datasets (only when not using dense cache)
            indexes = np.sort(indexes)

        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]
            idx_slice = indexes
            if self.adata_manager.data_registry[key].attr_name in self.ignore_types:
                # Ignore .uns and .varm
                continue
            # Use dense mmap cache for X if available
            if self.dense_cache is not None and key == REGISTRY_KEYS.X_KEY:
                sliced_data = self.dense_cache[idx_slice].astype(dtype, copy=False)
            elif isinstance(data, np.ndarray | h5py.Dataset):
                sliced_data = data[idx_slice].astype(dtype, copy=False)
            elif isinstance(data, pd.DataFrame):
                sliced_data = data.iloc[idx_slice, :].to_numpy().astype(dtype, copy=False)
            elif issparse(data) or isinstance(data, SparseDataset):
                sliced_data = data[idx_slice].astype(dtype, copy=False)
                if self.load_sparse_tensor:
                    sliced_data = scipy_to_torch_sparse(sliced_data)
                else:
                    sliced_data = sliced_data.toarray()
            elif isinstance(data, str) and key == REGISTRY_KEYS.MINIFY_TYPE_KEY:
                # for minified anndata, we can have a string for `data`,
                # which is the value of the MINIFY_TYPE_KEY in adata.uns,
                # used to record the type data minification
                continue
            else:
                raise TypeError(f"{key} is not a supported type")

            data_map[key] = sliced_data

        return data_map


class EmbAnnDataManager(AnnDataManager):

    _dense_cache: DenseCache | None = None

    def __init__(
        self,
        fields: list[AnnDataField] | None = None,
        setup_method_args: dict | None = None,
        validation_checks: AnnDataManagerValidationCheck | None = None,
    ) -> None:
        super().__init__(fields, setup_method_args, validation_checks)

    def _get_or_create_dense_cache(
        self,
        cache_path: str | None = None,
        chunk_size: int = 50_000,
    ) -> DenseCache | None:
        """Load existing dense cache or create one if adata is backed."""
        if self._dense_cache is not None:
            return self._dense_cache

        adata = self.adata
        if not adata.isbacked:
            return None

        # Default cache path: <adata_dir>/<.cache>/X_dense.npy
        if cache_path is None:
            cache_dir = os.path.join(os.path.dirname(adata.filename), '.cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, 'X_dense.npy')

        meta_path = cache_path + '.meta.json'
        if os.path.exists(meta_path) and os.path.exists(cache_path):
            logging.getLogger(__name__).info(f"Loading dense cache from {cache_path}")
            self._dense_cache = DenseCache(cache_path)
        else:
            logging.getLogger(__name__).info(
                f"No dense cache found — converting backed adata to dense at {cache_path}"
            )
            self._dense_cache = DenseCache.create(
                adata, cache_path=cache_path, chunk_size=chunk_size,
            )

        # Try to migrate cache to local SSD for faster random access
        ssd_path = try_migrate_cache_to_ssd(self._dense_cache.cache_path)
        if ssd_path != self._dense_cache.cache_path:
            self._dense_cache = DenseCache(ssd_path)

        return self._dense_cache

    def create_torch_dataset(
        self,
        indices: Sequence[int] | Sequence[bool] = None,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
        cache_path: str | None = None,
    ) -> AnnTorchDataset:
        """
        Creates a torch dataset from the AnnData object registered with this instance.

        Parameters
        ----------
        indices
            The indices of the observations in the adata to use
        data_and_attributes
            Dictionary with keys representing keys in data registry
            (``adata_manager.data_registry``) and value equal to desired numpy loading type (later
            made into torch tensor) or list of such keys. A list can be used to subset to certain
            keys in the event that more tensors than needed have been registered. If ``None``,
            defaults to all registered data.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data.
        cache_path
            Path for dense .npy cache. Only used when adata is backed.
            Defaults to ``<adata_dir>/X_dense.npy``.

        Returns
        -------
        :class:`~scvi.data.AnnTorchDataset`
        """
        dense_cache = self._get_or_create_dense_cache(cache_path)
        if dense_cache is not None:
            logging.getLogger(__name__).info(
                f"Using dense cache: {dense_cache.n_obs:,} x {dense_cache.n_vars:,}, "
                f"mmap'd from {dense_cache.cache_path}"
            )

        dataset = EmbAnnTorchDataset(
            self,
            getitem_tensors=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
            dense_cache=dense_cache,
        )
        if indices is not None:
            # This is a lazy subset, it just remaps indices
            dataset = Subset(dataset, indices)
        return dataset
