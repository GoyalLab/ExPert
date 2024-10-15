import logging

import pandas as pd
import scipy.sparse as sp
import anndata as ad
from pathlib import Path
import numpy as np
import scanpy as sc


def _filter(d, dataset_name, hvg_pool, zero_pad=True):
    # filter for pool of highly variable genes
    matching_hvgs = d.var_names.intersection(hvg_pool)
    logging.info(f'Found {len(matching_hvgs)} pool genes in {dataset_name}')
    d = d[:, matching_hvgs].copy()
    # ensure appropriate typing
    if not sp.isspmatrix_csr(d.X):
        logging.debug(f'Changing .X from {d.X.__class__} to scipy.sparse._csr.csr_matrix')
        d.X = sp.csr_matrix(d.X, dtype=np.float32)
    # check for missing genes
    missing_hvgs = set(hvg_pool) - set(matching_hvgs)
    if zero_pad and len(missing_hvgs) > 0:
        logging.info(f'{len(missing_hvgs)} hvgs missing from pool; padding with zero values')
        # build empty matrix for padded matrix
        zero_matrix = sp.csr_matrix((d.n_obs, len(missing_hvgs)), dtype=np.float32)
        # stack matrices horizontally
        logging.debug(f'Adding sparse 0-pad to .X ({d.n_obs}, {len(missing_hvgs)})')
        X = sp.hstack((d.X, zero_matrix))
        # define proper gene assignments, missing genes is unordered, but does not matter (all are 0)
        gene_list = list(matching_hvgs) + (list(missing_hvgs))
        # build final AnnData object
        logging.debug('Updating padded AnnData')
        d = sc.AnnData(X=X, obs=d.obs, var=pd.DataFrame(index=gene_list))
        logging.debug('Ordering genes based on gene pool')
        d = d[:, hvg_pool].copy()
    else:
        logging.info(f'{len(missing_hvgs)} hvgs lost from pool')
    return d


def prepare_merge(input_pth, pool_genes, out, hvg=True, zero_pad=True):
    if input_pth.endswith('.h5ad'):
        logging.info(f'Preparing file: {input_pth}')
        name = Path(input_pth).stem
        adata = sc.read(input_pth)
        # make cell barcodes unique to dataset
        adata.obs_names = adata.obs_names + ';' + name
        adata.obs['dataset'] = name
        # subset to pre-calculated gene pool
        if hvg:
            adata = _filter(adata, name, pool_genes, zero_pad=zero_pad)
        # save prepared adata
        adata.write_h5ad(out, compression='gzip')
        return adata.obs
    else:
        raise FileNotFoundError('File has to be .h5ad format')


