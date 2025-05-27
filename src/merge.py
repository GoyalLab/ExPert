import scanpy as sc
import scipy.sparse as sp
import logging
import anndata as ad
import pandas as pd
from pathlib import Path
import numpy as np
import dask.array as da


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


def collapse_obs(obs_files, join='inner', cell_key='celltype') -> pd.DataFrame:
    obs = None
    # read all .obs and combine them
    for file in obs_files:
        logging.info(f'Collecting .obs from {file}')
        ds_obs = pd.read_csv(file, index_col=0)
        if cell_key not in ds_obs.columns:
            ds_obs[cell_key] = ds_obs.get('cell_line', 'Unknown')
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
    idx_map = obs[key].value_counts().sort_index()
    # open empty meta-set and keep updating .X
    try:
        meta_set = sc.read(out_pth, backed='r+')
        # insert every dataset .X
        start, end_idx = 0, 0
        for idx, file in enumerate(input_pths):
            logging.info(f'Inserting {file} into meta-set')
            # get target slice of meta set .X
            end_idx = start + idx_map.iloc[idx]
            logging.debug(f'Target slice : {start}:{end_idx}')
            # read dataset
            logging.debug(f'Reading : {file}')
            ds = sc.read(file)
            # define update data
            logging.debug('Building update slice')
            new_data = ds.X.tocoo()
            row_idx = new_data.row + start
            new_data = sp.csr_matrix((new_data.data, (row_idx, new_data.col)), shape=meta_set.X.shape)
            # update slice
            logging.debug(f'Updating slice : {start}:{end_idx}')
            meta_set.X = meta_set.X.to_memory() + new_data
            # write changes to disk
            logging.debug('Writing updated .X')
            meta_set.write()
            # update next insert location
            start = end_idx
            logging.info(f'Updated {np.round(end_idx/obs.shape[0]*100, 2)}% of meta-set .X matrix')
    finally:
        meta_set.file.close()


def _dask_vstack(input_pths, obs, var):
    X = []
    for file in input_pths:
        logging.info(f'Adding {file} .X to meta-set')
        X.append(da.from_array(sc.read(file).X))
    logging.info('Stacking .X matrices')
    X = da.vstack(X)
    return sc.AnnData(X=X, obs=obs, var=var)


def _pca_dask(adata, dense_chunk_size=10_000):
    logging.info('Computing PCA for metaset')
    adata.layers["dense"] = adata.X.rechunk((dense_chunk_size, -1)).map_blocks(
        lambda x: x.toarray(), dtype=adata.X.dtype, meta=np.array([])
    )
    sc.pp.pca(adata, layer='dense')
    logging.debug('Computing last step of PCA')
    adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
    logging.info('Computing neighbors')
    sc.pp.neighbors(adata)


def _umap(adata):
    logging.info('Computing UMAP')
    sc.tl.umap(adata)


def merge(input_pths, out_pth, obs, var, method='dask'):
    if method == 'dask':
        metaset = _dask_vstack(input_pths, obs=obs, var=var)
        logging.info(f'Saving metaset AnnData to {out_pth}')
        metaset.write_h5ad(out_pth, compression='gzip')
    elif method=='on_disk':
        _on_disk(input_pths, out_pth, obs=obs, var=var)
    elif method=='in_memory':
        _in_memory(input_pths, out_pth)
