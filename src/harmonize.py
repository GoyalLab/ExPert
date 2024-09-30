import os

import pandas as pd
import logging
import anndata as ad
import numpy as np
import scanpy as sc
import scanorama
import harmonypy as hm
import scipy.sparse as sp
from MulticoreTSNE import MulticoreTSNE as tsne
import scvi


def _pca_reconstruction(adata, n_comps=50, pca_key='X_pca'):
    logging.info(f'Computing PCA reconstruction with {n_comps} components')
    # Get PCA coordinates and loadings
    pca_coords = adata.obsm[pca_key][:, :n_comps]
    pca_loadings = adata.varm['PCs'][:, :n_comps]
    # Reconstruct the data
    reconstructed = np.dot(pca_coords, pca_loadings.T)
    # If the data was scaled before PCA, we need to reverse the scaling
    if 'mean' in adata.var and 'std' in adata.var:
        reconstructed = reconstructed * adata.var['std'].values + adata.var['mean'].values
    logging.info(f'Finished PCA reconstruction with {n_comps} components')
    return reconstructed


def _harmony(adata):
    if 'scaled' not in adata.uns:
        logging.info('Scaling dataset before harmonizing')
        sc.pp.scale(adata)
    if 'X_pca' not in adata.obsm:
        logging.info('Performing PCA')
        sc.tl.pca(adata)
    logging.info('Starting harmony')
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'dataset')
    key = 'X_pca_harmony'
    adata.obsm[key] = ho.Z_corr.T
    logging.info('Finished harmony')
    adata.X = _pca_reconstruction(adata, pca_key=key)


def _scANVI(adata, batch_key='dataset', labels_key='celltype'):
    logging.info('Running SCANVI')
    scvi.settings.verbosity = 2
    # Initialize scvi adata, specify dataset and cell type
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, labels_key=labels_key)
    # build scVI model
    scvi_model = scvi.model.SCVI(adata)
    # train model
    scvi_model.train(early_stopping=True)
    # run scANVI that additionally incorporated cell labels
    model_scanvi = scvi.model.SCANVI.from_scvi_model(
        scvi_model, labels_key=labels_key, unlabeled_category="unlabelled"
    )
    model_scanvi.train()
    # add trained latent space
    adata.obsm["X_scANVI"] = model_scanvi.get_latent_representation()
    logging.info('Finished SCANVI training')
    # reconstruct normalized gene expression counts and set them as default to avoid adding layers
    adata.X = _pca_reconstruction(adata)


def _scanorama(ds_list):
    # prepare data
    ds_list = [d.copy() for d in ds_list]
    for ds in ds_list:
        if sp.issparse(ds.X) and not isinstance(ds.X, sp.csr_matrix):
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
            logging.info(f'Adding {adata.n_obs} cells to merged dataset')
            # save dataset in dictionary
            ds_dict[name] = adata
    logging.info(f'Finished reading {len(ds_dict)} datasets')
    return ds_dict

def correction_methods():
    return ['scANVI', 'scanorama', 'harmonypy', 'skip']

def requires_raw_data():
    return ['scANVI']

def requires_processed_data():
    return ['scanorama', 'harmonypy']

def check_method(m, conf):
    methods = correction_methods()
    if m not in methods:
        raise ValueError(f'Method must be {methods}; got {m}')
    if m in requires_raw_data():
        logging.info(f'Method {m} requires raw data, setting preprocess to exclude normalization and log1p')
        conf['norm'] = False
        conf['log_norm'] = False
    if m in requires_processed_data():
        logging.info(f'Method {m} requires preprocessed data, setting preprocess to include normalization and log1p')
        conf['norm'] = True
        conf['log_norm'] = True


def harmonize(data_files, hvg_pool, method='skip', hvg=True, zero_pad=True, scale=True, cores=1, include_raw_tsne=False):
    # read datasets to one dict
    ds_dict = _read_datasets(data_files, hvg_pool.index, hvg_filter=hvg, zero_pad=zero_pad)
    # extract names and according AnnDatas
    ds_list = [*ds_dict.values()]
    ds_keys = [*ds_dict.keys()]
    pca_key = 'X_pca'

    if method != 'skip':
        # reduce datasets to common genes
        common_genes = list(set.intersection(*(set(adata.var_names) for adata in ds_list)))
        logging.info(f'Found {len(common_genes)} common genes between datasets while merging')
        ds_list = [d[:, common_genes] for d in ds_list]
        # apply batch effect normalization over different datasets and adjust original counts
        if method == 'scanorama':
            ds_list = _scanorama(ds_list)
        # concatenate corrected datasets
        logging.info('Merging datasets')
        merged = ad.concat(ds_list, label='dataset', keys=ds_keys)
        logging.info('Finished merge')
        logging.info('Computing PCA')
        sc.pp.pca(merged)
        # normalize expression based on merged dataset
        if method == 'scANVI':
            pca_key = 'X_scANVI'
            _scANVI(merged)
        elif method == 'harmonypy':
            pca_key = 'X_pca_harmony'
            _harmony(merged)
        # scale merged dataset
        if scale:
            # scale data after merge
            logging.info('Scaling and centering merged dataset')
            sc.pp.scale(merged)
    else:
        # select scaled data if available
        if scale:
            logging.info('Setting scaled data as default in datasets')
            for i, ds in enumerate(ds_list):
                logging.info(f'Scaling dataset {ds_keys[i]} before merge')
                sc.pp.scale(ds)
        # collapse to a merged AnnData object
        logging.info('Merging datasets')
        merged = ad.concat(ds_dict, label='dataset')
        logging.info('Finished merge')
        logging.info('Computing PCA')
        sc.pp.pca(merged)
    # add summarized var data (highly variable gene information)
    # merged.var = pd.concat([merged.var, hvg_pool], axis=1, join='inner')
    logging.info(f'Resulting metaset spans: {merged.shape[0]} combined cells and {merged.shape[1]} common genes')
    logging.info('Calculating neighbors')
    sc.pp.neighbors(merged, use_rep=pca_key)
    # Calculate tSNE for entire dataset (also add raw tSNE if correction method is given and option is true)
    if include_raw_tsne and pca_key!='X_pca':
        logging.info('Computing raw tsne')
        merged.obsm['X_tsne_raw'] = tsne(n_jobs=cores).fit_transform(merged.obsm['X_pca'])
    logging.info('Computing tsne')
    merged.obsm['X_tsne'] = tsne(n_jobs=cores).fit_transform(merged.obsm[pca_key])
    logging.info('Done')
    return merged
