from scvi.data._manager import AnnDataManager, AnnDataManagerValidationCheck
from scvi.data._anntorchdataset import AnnTorchDataset
from scvi.data._utils import scipy_to_torch_sparse
from torch.utils.data import Subset
import torch
import h5py
from scipy.sparse import issparse
from anndata.abc import CSCDataset, CSRDataset
import pandas as pd
import numpy as np

from src.utils.constants import REGISTRY_KEYS
from collections.abc import Sequence
from scvi.data.fields import AnnDataField

SparseDataset = (CSRDataset, CSCDataset)

import logging


class EmbAnnTorchDataset(AnnTorchDataset):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
        ignore_types: list = ['uns', 'varm']
    ):
        super().__init__(adata_manager, getitem_tensors, load_sparse_tensor)
        self.ignore_types = ignore_types

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

        if self.adata_manager.adata.isbacked and isinstance(indexes, list | np.ndarray):
            # need to sort indexes for h5py datasets
            indexes = np.sort(indexes)

        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]
            idx_slice = indexes
            if self.adata_manager.data_registry[key].attr_name in self.ignore_types:
                # Ignore .uns and .varm
                continue
            if isinstance(data, np.ndarray | h5py.Dataset):
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
                # for minified  anndata, we need this because we can have a string
                # for `data``, which is the value of the MINIFY_TYPE_KEY in adata.uns,
                # used to record the type data minification
                # TODO: Adata manager should have a list of which fields it will load
                continue
            else:
                raise TypeError(f"{key} is not a supported type")

            data_map[key] = sliced_data

        return data_map


class EmbAnnDataManager(AnnDataManager):
    def __init__(
        self,
        fields: list[AnnDataField] | None = None,
        setup_method_args: dict | None = None,
        validation_checks: AnnDataManagerValidationCheck | None = None,
    ) -> None:
        super().__init__(fields, setup_method_args, validation_checks)

    def create_torch_dataset(
        self,
        indices: Sequence[int] | Sequence[bool] = None,
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
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

        Returns
        -------
        :class:`~scvi.data.AnnTorchDataset`
        """
        dataset = EmbAnnTorchDataset(
            self,
            getitem_tensors=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )
        if indices is not None:
            # This is a lazy subset, it just remaps indices
            dataset = Subset(dataset, indices)
        return dataset
