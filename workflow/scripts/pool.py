import logging

import pandas as pd

from src.utils import setup_logger, make_obs_names_unique


def get_pool(files, agg='mean'):
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


def build_pool(hvg_files, pool_output):
    # Determine overall gene pool to use for meta-set
    pool = get_pool(hvg_files)
    pool.to_csv(pool_output)
    


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        build_pool(
            hvg_files=snakemake.input.hvg_files,
            pool_output=snakemake.output.pool,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")