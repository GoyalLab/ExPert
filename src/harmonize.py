import os

import pandas as pd
import logging
import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch
import dask.array as da
from pathlib import Path

from typing import List, Literal
from collections.abc import Iterable as IterableClass
from collections.abc import Sequence
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi._types import Number

from scvi.distributions._utils import DistributionConcatenator
from scvi.model._utils import _get_batch_code_from_category
from scvi import REGISTRY_KEYS

from src.utils import log_decorator, log_memory_usage


def _train_scANVI(adata, model_dir='./scanvi'):
    # build scVI model
    scvi_model = scvi.model.SCVI(adata, )
    # train model
    logging.info('Training scVI model')
    scvi_model.train(early_stopping=True)
    logging.info('Finished training scVI model')
    # run scANVI that additionally incorporated cell labels
    scvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, unlabeled_category="unlabelled"
    )
    logging.info('Training scANVI model')
    scvi_model.train()
    logging.info('Finished training scANVI model')
    logging.info(f'Saving scANVI model in {model_dir}')
    scvi_model.save(model_dir, overwrite=True, save_anndata=False)
    # add trained latent space
    adata.obsm["X_scANVI"] = scvi_model.get_latent_representation()
    logging.info('Finished SCANVI training')
    return scvi_model


def _get_batch_code_from_category(adata_manager: AnnDataManager, category: Sequence[Number | str]):
    if not isinstance(category, IterableClass) or isinstance(category, str):
        category = [category]

    batch_mappings = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
    batch_code = []
    for cat in category:
        if cat is None:
            batch_code.append(None)
        elif cat not in batch_mappings:
            raise ValueError(f'"{cat}" not a valid batch category.')
        else:
            batch_loc = np.where(batch_mappings == cat)[0][0]
            batch_code.append(batch_loc)
    return batch_code


@torch.inference_mode()
def normalize_expression(
    model,
    adata: AnnData | None = None,
    indices: list[int] | None = None,
    gene_list: list[str] | None = None,
    transform_batch: list[Number | str] | None = None,
    n_samples: float = 1,
    library_size: float | Literal["latent"] = 10_000,
    weights: Literal["uniform", "importance"] | None = None,
    batch_size: int | None = None,
    as_dask: bool = True
):
    # validate adata for model
    adata = model._validate_anndata(adata)

    if indices is None:
        indices = np.arange(adata.n_obs)
    scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

    transform_batch = _get_batch_code_from_category(
        model.get_anndata_manager(adata, required=True), transform_batch
    )

    gene_mask = slice(None) if gene_list is None else adata.var_names.isin(gene_list)

    if library_size == "latent":
        generative_output_key = "mu"
        scaling = 1
    else:
        generative_output_key = "scale"
        scaling = library_size

    store_distributions = weights == "importance"
    if store_distributions and len(transform_batch) > 1:
        raise NotImplementedError(
            "Importance weights cannot be computed when expression levels are averaged across "
            "batches."
        )
    print(f'Library_size={library_size}, n_samples={n_samples}, transform_batch={transform_batch}')

    exprs = []
    zs = []
    qz_store = DistributionConcatenator()
    px_store = DistributionConcatenator()
    for tensors in scdl:
        per_batch_exprs = []
        for batch in transform_batch:
            generative_kwargs = model._get_transform_batch_gen_kwargs(batch)
            inference_kwargs = {"n_samples": n_samples}
            inference_outputs, generative_outputs = model.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                generative_kwargs=generative_kwargs,
                compute_loss=False
            )
            exp_ = generative_outputs["px"].get_normalized(generative_output_key)
            exp_ = exp_[..., gene_mask]
            exp_ *= scaling
            per_batch_exprs.append(exp_[None].cpu())
            if store_distributions:
                qz_store.store_distribution(inference_outputs["qz"])
                px_store.store_distribution(generative_outputs["px"])

        zs.append(inference_outputs["z"].cpu())
        per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
        
        # convert to dask array
        if as_dask:
            per_batch_exprs = da.from_array(per_batch_exprs)
        exprs.append(per_batch_exprs)
    # stack batches
    if as_dask:
        exprs = da.vstack(exprs)
    else:
        exprs = np.concatenate(exprs, axis=0)
    zs = torch.concat(zs, dim=0)

    return exprs


@log_decorator
def _scANVI(adata: AnnData, batch_key: str = 'dataset', labels_key: str = 'celltype', model_dir: str = './scanvi') -> None:
    if adata.isbacked:
        logging.info('Loading .X into memory to train faster')
        adata.X = adata.X.to_memory()
    if not isinstance(adata.X, sp.csr_matrix):
        logging.info('Converting .X to CSR matrix to increase training speed')
        adata.X = sp.csr_matrix(adata.X)
    logging.info('Running SCANVI')
    scvi.settings.verbosity = 2
    # Check if CUDA (GPU) is available
    use_gpu = torch.cuda.is_available()
    logging.info(f'GPU available: {"yes" if use_gpu else "no"}')
    # Initialize scvi adata, specify dataset and cell type
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, labels_key=labels_key)
    # train model or fall back to pre-calculated model
    if os.path.exists(os.path.join(model_dir, 'model.pt')):
        logging.info(f'Caching model from {model_dir}')
        model_scanvi = scvi.model.SCANVI.load(model_dir, adata=adata)
    else:
        model_scanvi = _train_scANVI(adata, model_dir=model_dir)
    # set dataset to correct for (ideally select largest dataset)
    batch_key = adata.obs['dataset'].value_counts().index[0]
    # reconstruct normalized gene expression counts and set them as default to avoid adding layers
    logging.info(f'Reconstructing gene expression corrected for reference: {batch_key}')
    adata.X = normalize_expression(model=model_scanvi, 
                                   adata=adata,
                                   library_size=10_000,
                                   transform_batch=batch_key,
                                   as_dask=True)


def _filter(d, dataset_name, hvg_pool, zero_pad=True):
    # filter for pool of highly variable genes
    matching_hvgs = d.var_names.intersection(hvg_pool)
    logging.info(f'Found {len(matching_hvgs)} pool genes in {dataset_name}')
    d = d[:, matching_hvgs]
    # check for missing genes
    missing_hvgs = set(hvg_pool) - set(matching_hvgs)
    if zero_pad and len(missing_hvgs) > 0:
        logging.info(f'{len(missing_hvgs)} hvgs missing from pool; padding with zero values')
        # build zero matrix for missing genes
        zero_matrix = np.zeros((d.n_obs, len(missing_hvgs)))
        # define AnnData for missing genes
        ds_zeros = ad.AnnData(X=zero_matrix, var=pd.DataFrame(index=list(missing_hvgs)), obs=d.obs)
        # merge original and zero AnnData, keep original meta data for samples
        d = ad.concat([d, ds_zeros], axis=1, merge='first')
    else:
        logging.info(f'{len(missing_hvgs)} hvgs lost from pool')
    return d


def _read_datasets(data_files, pool, hvg_filter=True, zero_pad=True):
    ds_dict = {}
    for file in data_files:
        if file.endswith('.h5ad'):
            logging.info('Reading {}'.format(file))
            name = Path(file).stem
            adata = sc.read(file)
            # make cell barcodes unique to dataset
            adata.obs_names = adata.obs_names + ';' + name
            # apply hvg pool and zero-pad filters to dataset
            if hvg_filter:
                adata = _filter(adata, name, pool, zero_pad=zero_pad)
            logging.info(f'Adding {adata.n_obs} cells to merged dataset')
            # save dataset in dictionary
            ds_dict[name] = adata
    logging.info(f'Finished reading {len(ds_dict)} datasets')
    return ds_dict

def reduce_to_common_genes(ds_list):
    common_genes = list(set.intersection(*(set(adata.var_names) for adata in ds_list)))
    logging.info(f'Found {len(common_genes)} common genes between datasets while merging')
    return [d[:, common_genes] for d in ds_list]


def merge_datasets(ds_list, ds_keys, label='dataset'):
    logging.info('Merging datasets')
    merged = ad.concat(ds_list, label=label, keys=ds_keys)
    logging.info('Finished merge')
    return merged


def merge_datasets_dict(ds_dict, label='dataset'):
    logging.info('Merging datasets')
    merged = ad.concat(ds_dict, label=label)
    logging.info('Finished merge')
    return merged


@log_decorator
def _pca(adata, **kwargs):
    sc.pp.pca(adata, **kwargs)

@log_decorator
def _neighbors(adata, **kwargs):
    sc.pp.neighbors(adata, **kwargs)

@log_decorator
def _umap(adata,**kwargs):
    sc.tl.umap(adata, **kwargs)

def _read_dataset(file, backed=None):
    logging.info(f'Reading {file}...')
    d = sc.read(file, backed=backed)
    logging.info(f'Finished reading {file}')
    return d


class Harmonizer:
    """
    Class to Harmonize a merged AnnData object
    """
    metaset: AnnData = None
    method: str = 'scANVI'

    _mem_log_dir: str = './'
    _is_harmonized: bool = False
    _harmonized_pca: str = None
    _methods: List[str] = ['scANVI', 'scanorama', 'harmonypy', 'skip']


    def __init__(self, metaset_file: str, method: str = None, mem_log_dir: str = None) -> None:
        self._init_method(method)
        self._init_dataset(metaset_file)
        self._init_mem_log_dir(mem_log_dir)
        

    def _init_mem_log_dir(self, mem_log_dir: str) -> None:
        if mem_log_dir:
            os.makedirs(mem_log_dir, exist_ok=True)
            self._mem_log_dir = mem_log_dir

    def _init_method(self, m: str) -> None:
        if m not in self._methods:
            raise ValueError(f'Method has to be one of: {self._methods}')
        self.method = m
        self._harmonized_pca = f'X_{m}'

    def _init_dataset(self, file: str) -> None:
        self.metaset = _read_dataset(file)
    
    def _run_scANVI(self, dataset_key: str, cell_type_key: str, model_dir: str):
        _scANVI(self.metaset, batch_key=dataset_key, labels_key=cell_type_key, model_dir=model_dir)
    
    @log_decorator
    def harmonize(self, dataset_key: str = 'dataset', cell_type_key: str = 'celltype', model_dir:str = './', method: str = None):
        # Update method if given
        if method:
            self.method = self._init_method(method)
        # launch specific method
        if self.method == 'scANVI':
            self._run_scANVI(dataset_key, cell_type_key, model_dir)
            self._is_harmonized = True
        elif self.method == 'skip':
            logging.info('No harmonization applied to registered metaset')

    def _run_umap(self):
        @log_memory_usage(self._mem_log_dir)
        def monitor_umap():
            _pca(self.metaset)
            _neighbors(self.metaset)
            _umap(self.metaset)
        monitor_umap()

    def calculate_harm_umap(self):
        if self._is_harmonized:
            self._run_umap()
        else:
            logging.info("Can't calculate harmonized UMAP on row data")
    
    def calculate_raw_umap(self):
        if not self._is_harmonized:
            self._run_umap()
        else:
            logging.info("Can't calculate raw UMAP on harmonized data")

    def _save(self, *args, **kwargs):
        self.metaset.write_h5ad(*args, **kwargs)

    def save_normalized_adata(self, path: str, **kwargs):
        if self._is_harmonized:
            self._save(path, **kwargs)
    
