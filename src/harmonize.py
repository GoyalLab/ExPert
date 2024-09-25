import os

import pandas as pd
import logging
import anndata as ad
import numpy as np
import scanpy as sc
import scanorama


def _scanorama(ds_list):
    # prepare data
    ds_list = [d.copy() for d in ds_list]
    for ds in ds_list:
        ds.X = ds.X.tocsr()
    # Perform batch correction using Scanorama
    logging.info('Running scanorama')
    corrected = scanorama.correct_scanpy(ds_list)
    logging.info('Finished scanorama')
    return corrected


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

def correction_methods():
    return ['scanorama', 'skip']

def _check_methods(m):
    methods = correction_methods()
    if m not in methods:
        raise ValueError(f'Method must be {methods}; got {m}')


def harmonize(data_files, hvg_pool, method='skip', hvg=True, zero_pad=True, scale=True, cores=-1):
    _check_methods(method)
    # read datasets to one dict
    ds_dict = _read_datasets(data_files, hvg_pool.index, hvg_filter=hvg, zero_pad=zero_pad)
    # extract names and according AnnDatas
    ds_list = [*ds_dict.values()]
    ds_keys = [*ds_dict.keys()]
    if method != 'skip':
        # reduce datasets to common genes
        common_genes = list(set.intersection(*(set(adata.var_names) for adata in ds_list)))
        logging.info(f'Found {len(common_genes)} between datasets while merging')
        ds_list = [d[:, common_genes] for d in ds_list]
        # apply batch effect normalization over different datasets and adjust original counts
        if method == 'scanorama':
            ds_list = _scanorama(ds_list)
        else:
            raise ValueError(f'Method must be one of {correction_methods()}; got {method}')
        # concatenate corrected datasets
        logging.info('Merging datasets')
        merged = ad.concat(ds_list, label='dataset', keys=ds_keys)
        logging.info('Finished merge')
        # scale merged dataset
        if scale:
            # scale data after merge
            sc.pp.scale(merged)
    else:
        # select scaled data if available
        if scale:
            logging.info('Setting scaled data as default in datasets')
            for ds in ds_list:
                logging.info(f'Scaling dataset {ds.uns["dataset_name"]} before merge')
                sc.pp.scale(ds)
        # collapse to a merged AnnData object
        logging.info('Merging datasets')
        merged = ad.concat(ds_dict, label='dataset')
        logging.info('Finished merge')
    # add summarized var data (highly variable gene information)
    merged.var = pd.concat([merged.var, hvg_pool], axis=1, join='inner')
    logging.info(f'Resulting metaset spans: {merged.shape[0]} combined cells and {merged.shape[1]} common genes')
    return merged
