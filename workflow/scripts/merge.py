import logging

import pandas as pd

from src.harmonize import harmonize
from src.utils import setup_logger, make_obs_names_unique


def get_hvg_pool(files, agg='mean'):
    pool = None
    # for each csv file, update the total mapping of highly variable genes
    for file in files:
        hvg = pd.read_csv(file, index_col=0)
        if pool is None:
            pool = hvg
        else:
            pool = pd.concat([pool, hvg], axis=1)
    # clean up duplicate values
    pool = make_obs_names_unique(pool, agg=agg)
    logging.info(f'HVG pool shape: {pool.shape[0]} genes, {pool.shape[1]} var columns')
    return pool


def merge_datasets(dataset_files, hvg_files, merged_set_output, pool_output, method='harmony', hvg=True, zero_pad=True, cores=-1, plot=True, do_umap=True, do_tsne=False):
    # collect pool of highly variable genes over all datasets
    pool = get_hvg_pool(hvg_files)
    pool.to_csv(pool_output)
    # read all preprocessed datasets, combine, and correct for batch effects
    merged = harmonize(dataset_files, pool, method=method, hvg=hvg, zero_pad=zero_pad, cores=cores, plot=plot, do_umap=do_umap, do_tsne=do_tsne)
    # write final dataset to file
    merged.write_h5ad(merged_set_output)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        merge_datasets(
            dataset_files=snakemake.input.dataset_files,
            hvg_files=snakemake.input.hvg_files,
            merged_set_output=snakemake.output.merged_set,
            pool_output=snakemake.output.pool,
            method=snakemake.params.method,
            hvg=snakemake.params.hvg,
            zero_pad=snakemake.params.zero_pad,
            cores=snakemake.params.cores,
            plot=snakemake.params.plot,
            do_umap=snakemake.params.do_umap,
            do_tsne=snakemake.params.do_tsne
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
