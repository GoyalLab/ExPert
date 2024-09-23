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
            name = ''.join(os.path.basename(file).split('.')[:-1])
            adata = ad.read(file)
            # make cell barcodes unique to dataset
            adata.obs_names = adata.obs_names + ';' + name
            # apply hvg pool and zero-pad filters to dataset
            if hvg_filter:
                adata = _filter(adata, name, pool, zero_pad=zero_pad)
            # save dataset in dictionary
            ds_dict[name] = adata
    logging.info(f'Finished reading {len(ds_dict)} datasets')
    return ds_dict

def _check_methods(m):
    if m not in ['harmony', 'bbknn', 'skip']:
        raise ValueError('Method must be "harmony", "bbknn", or "skip"')


def harmonize(data_files, hvg_pool, method='skip', hvg=True, zero_pad=True):
    _check_methods(method)
    # read datasets to one dict
    ds_dict = _read_datasets(data_files, hvg_pool.index, hvg_filter=hvg, zero_pad=zero_pad)
    # collapse to a merged AnnData
    merged = ad.concat(ds_dict, label='dataset')
    # add var data (highly variable gene information)
    merged.var = pd.concat([merged.var, hvg_pool], axis=1, join='inner')
    # apply batch effect normalization on different datasets in PCA space
    if method != 'skip':
        # Perform PCA on merged set
        sc.pp.pca(merged)
        # Calculate neighbors
        sc.pp.neighbors(merged, use_rep='X_pca')
        # harmonize datasets
        if method == 'harmony':
            _harmony(merged)
        elif method == 'bbknn':
            bbknn.bbknn(merged, batch_key='dataset')
    return merged
