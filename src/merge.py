import scanpy as sc
import scipy.sparse as sp
import logging
import anndata as ad
import pandas as pd
from pathlib import Path


def merge_methods():
    return ['on_disk', 'in_memory']


def _read_datasets(data_files, pool, hvg_filter=True, zero_pad=True):
    ds_dict = {}
    for file in data_files:
        if file.endswith('.h5ad'):
            logging.info('Reading {}'.format(file))
            name = Path(file).stem
            adata = sc.read(file)
            logging.info(f'Adding {adata.n_obs} cells to merged dataset')
            # save dataset in dictionary
            ds_dict[name] = adata
    logging.info(f'Finished reading {len(ds_dict)} datasets')
    return ds_dict


def _merge_datasets(ds_list, ds_keys, label='dataset'):
    logging.info('Merging datasets')
    merged = ad.concat(ds_list, label=label, keys=ds_keys)
    logging.info('Finished merge')
    return merged


def _merge_datasets_dict(ds_dict, label='dataset'):
    logging.info('Merging datasets')
    merged = ad.concat(ds_dict, label=label)
    logging.info('Finished merge')
    return merged


def _in_memory(input_pths, out_pth):
    ds_dict = _read_datasets(input_pths)
    _merge_datasets_dict(ds_dict).write_h5ad(out_pth, compression='gzip')


def collapse_obs(obs_files, join='inner'):
    obs = None
    # read all .obs and combine them
    for file in obs_files:
        logging.info(f'Collecting .obs from {file}')
        ds_obs = pd.read_csv(file, index_col=0)
        if obs is None:
            obs = ds_obs
        else:
            obs = pd.concat([obs, ds_obs], axis=0, join=join)
    return obs


def _on_disk(input_pths, out_pth, obs, var, key='dataset'):
    # create meta-set with first input AnnData, keep file connection open to update
    X = sp.csr_matrix((obs.shape[0], var.shape[0]), dtype="float32")

    sc.AnnData(X=X, obs=obs, var=var).write_h5ad(out_pth, compression='gzip')
    # keep track of index batches to insert into
    idx_map = obs[key].value_counts()
    # open empty meta-set and keep updating .X
    try:
        meta_set = sc.read(out_pth, backed='r+')
        # insert every dataset .X
        start, end_idx = 0, 0
        for idx, file in enumerate(input_pths):
            logging.info(f'Inserting {file} into meta-set')
            # get target slice of meta set .X
            end_idx = idx_map.iloc[idx]
            logging.debug(f'Target slice : {start}:{end_idx}')
            # read dataset
            ds = sc.read(file)
            # define update data
            new_data = ds.X.tocoo()
            row_idx = new_data.row + start
            new_data = sp.csr_matrix((new_data.data, (row_idx, new_data.col)), shape=meta_set.X.shape)
            # update slice
            meta_set.X = meta_set.X.to_memory() + new_data
            # write changes to disk
            meta_set.write()
            # update next insert location
            start = end_idx
    finally:
        meta_set.file.close()


def merge(input_pths, out_pth, obs, var, method='on_disk'):
    if method=='on_disk':
        _on_disk(input_pths, out_pth, obs=obs, var=var)
    elif method=='in_memory':
        _in_memory(input_pths, out_pth)
