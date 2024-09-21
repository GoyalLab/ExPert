import os

import pandas as pd
import scanpy as sc
import harmonypy as hm
import bbknn
import logging
import anndata as ad
import numpy as np


def _harmony(d):
    if 'X_pca' not in d.obsm.keys():
        logging.info('Performing PCA...')
        sc.pp.pca(d)
    logging.info('Starting Harmony...')
    ho = hm.run_harmony(d.obsm['X_pca'], d.obs, 'dataset')
    # Replace the PCA coordinates with Harmony-corrected ones
    d.obsm['X_pca_harmony'] = ho.Z_corr.T
    sc.pp.neighbors(d, use_rep='X_pca_harmony')
    logging.info('Finished Harmony.')


def _read_datasets(data_files, pool):
    ds_dict = {}
    for file in data_files:
        if file.endswith('.h5ad'):
            logging.info('Reading {}'.format(file))
            name = ''.join(os.path.basename(file).split('.')[:-1])
            adata = ad.read(file)
            # make cell barcodes unique to dataset
            adata.obs_names = adata.obs_names + ';' + name
            # filter for pool of highly variable genes
            pool_idx = adata.var_names.intersection(pool)
            ds_dict[name] = adata[:, pool_idx]
    logging.info(f'Finished reading {len(ds_dict)} datasets')
    return ds_dict

def _check_methods(m):
    if m not in ['harmony', 'bbknn']:
        raise ValueError('Method must be "harmony" or "bbknn"')


def harmonize(data_files, hvg_pool, method='harmony'):
    _check_methods(method)
    # read datasets to one dict
    ds_dict = _read_datasets(data_files, hvg_pool)
    # collapse to a merged AnnData
    merged = ad.concat(ds_dict, label='dataset')
    # calculate statistics
    mean_vars = np.mean([v.n_vars for v in ds_dict.values()])
    # observe merged dataset vs. originals
    logging.info(f'merged dataset has: {merged.shape[1]} hvgs; mean hvgs over all datasets: {mean_vars}')
    # Perform PCA on merged set
    sc.pp.pca(merged)
    # Calculate neighbors
    sc.pp.neighbors(merged, use_rep='X_pca')

    # harmonize datasets
    if method == 'harmony':
        _harmony(merged)
    if method == 'bbknn':
        bbknn.bbknn(merged, batch_key='dataset')
    return merged
