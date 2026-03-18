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
    hvg_pool = list(hvg_pool)
    matching = d.var.index.intersection(hvg_pool)
    logging.info(f'Found {len(matching)} pool genes in {dataset_name}')
    
    if zero_pad:
        missing = set(hvg_pool) - set(matching)
        if missing:
            logging.info(f'{len(missing)} hvgs missing; padding with zeros')
        
        # Subset to matching, then add empty columns and reorder
        d_sub = d[:, matching].copy()
        
        # Add zero columns for missing genes
        if missing:
            zero_block = sp.csr_matrix((d.shape[0], len(missing)), dtype=np.float32)
            missing_var = pd.DataFrame(index=pd.Index(list(missing)))
            d_zero = ad.AnnData(X=zero_block, obs=d_sub.obs, var=missing_var)
            d_sub = ad.concat([d_sub, d_zero], axis=1, merge='first')
        
        # Reorder to pool order
        d_sub = d_sub[:, hvg_pool]
        return d_sub
    else:
        pool_in_data = [g for g in hvg_pool if g in set(d.var.index)]
        return d[:, pool_in_data].copy()

def prepare_merge(input_pth: str, pool_genes: Iterable[str], out: str, zero_pad: bool = True, **kwargs):
    if input_pth.endswith('.h5ad'):
        logging.info(f'Preparing file: {input_pth}')
        name = Path(input_pth).stem
        adata = read_ad(input_pth)
        # make cell barcodes unique to dataset
        adata.obs_names = adata.obs_names + ';' + name
        adata.obs['dataset'] = name
        # subset to pre-calculated gene pool
        adata = _filter(adata, name, pool_genes)
        # Convert adata.X to csr
        if not isinstance(adata.X, sp.csr_matrix):
            logging.info('Converting to CSR matrix.')
            adata.X = sp.csr_matrix(adata.X)
        # Filter for a minimum number of cells per perturbation
        mcpp = kwargs.get('min_cells_per_perturbation', 0)
        if mcpp > 0:
            p_col = kwargs.get('perturbation_col', 'perturbation')
            cpp = adata.obs[p_col].value_counts()
            filtered_perturbations = cpp[cpp >= mcpp].index
            filter_mask = adata.obs[p_col].isin(filtered_perturbations)
            adata._inplace_subset_obs(filter_mask)
            logging.info(f'Filtering for a minimum of {mcpp} cells per perturbation. N perturbations: {filtered_perturbations.shape[0]}/{cpp.shape[0]}')
        # save prepared adata
        adata.write_h5ad(out, compression='gzip')
        return adata.obs
    else:
        raise FileNotFoundError(f'File has to be .h5ad format, got {input_pth}')
