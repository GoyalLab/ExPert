import logging

import pandas as pd
import os
import anndata as ad
from pathlib import Path
import numpy as np
import scanpy as sc


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


def prepare_merge(input_pth, pool_genes, out, hvg=True, zero_pad=True):
    if input_pth.endswith('.h5ad'):
        logging.info(f'Preparing file: {input_pth}')
        name = Path(input_pth).stem
        adata = sc.read(input_pth)
        # make cell barcodes unique to dataset
        adata.obs_names = adata.obs_names + ';' + name
        adata.obs['dataset'] = name
        # subset to precalculated gene pool
        if hvg:
            adata = _filter(adata, name, pool_genes, zero_pad=zero_pad)
        # save prepared adata
        adata.write_h5ad(out)
        return adata.obs
    else:
        raise FileNotFoundError('File has to be .h5ad format')


