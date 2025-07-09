import logging

import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path
import numpy as np
from src.utils import read_ad
import dask.array as da
from typing import Iterable


def _filter(d: ad.AnnData, dataset_name: str, hvg_pool: Iterable[str], zero_pad: bool = True):
    # extract .X, obs and var from dataset and delete input
    X, obs, var = d.X, d.obs, d.var
    # ensure .X is in csc format
    if isinstance(X, sp.csr_matrix):
        logging.debug('Changing matrix format to CSC')
        X = X.tocsc()
    # Filter for pool of genes in dataset
    hvg_pool_set = set(hvg_pool)
    matching_hvgs = var.index.intersection(hvg_pool_set)
    logging.info(f'Found {len(matching_hvgs)} pool genes in {dataset_name}')
    
    # Subset dataset for matching genes and convert to dask array
    X = da.from_array(X[:, var.index.isin(matching_hvgs)])
    
    # Determine pool genes that are missing from this dataset
    missing_hvgs = hvg_pool_set - set(matching_hvgs)
    # Add missing genes to dataset
    if zero_pad and missing_hvgs:
        logging.info(f'{len(missing_hvgs)} hvgs missing from pool; padding with zero values')
        # Construct 0-pad
        zero_matrix = da.from_array(sp.csc_matrix((obs.shape[0], len(missing_hvgs)), dtype=np.float32))

        logging.debug(f'Adding 0-pad ({obs.shape[0]}, {len(missing_hvgs)})')
        # Combine existing data with zero matrix
        X = da.hstack([X, zero_matrix])
        
        # Update d.var with padded genes
        var = pd.DataFrame(index=pd.Index(list(matching_hvgs) + list(missing_hvgs)))
    else:
        logging.info(f'{len(missing_hvgs)} hvgs lost from pool')
        var = pd.DataFrame(index=list(matching_hvgs))
    
    # order the AnnData.X based on the pool index
    logging.debug('Ordering AnnData based on pool order')
    sorted_by_pool_idx = var.index.get_indexer(hvg_pool)
    # reorder based on chunks
    dask_order = da.from_array(sorted_by_pool_idx, chunks=X.chunks[1])
    X = X[:, dask_order]
    var = var.iloc[sorted_by_pool_idx]
    # Create update AnnData
    return ad.AnnData(X=X, obs=obs, var=var)


def prepare_merge(input_pth: str, pool_genes: Iterable[str], out: str, zero_pad: bool = True):
    if input_pth.endswith('.h5ad'):
        logging.info(f'Preparing file: {input_pth}')
        name = Path(input_pth).stem
        adata = read_ad(input_pth)
        # make cell barcodes unique to dataset
        adata.obs_names = adata.obs_names + ';' + name
        adata.obs['dataset'] = name
        # subset to pre-calculated gene pool
        adata = _filter(adata, name, pool_genes, zero_pad=zero_pad)
        # save prepared adata
        adata.write_h5ad(out, compression='gzip')
        return adata.obs
    else:
        raise FileNotFoundError(f'File has to be .h5ad format, got {input_pth}')
