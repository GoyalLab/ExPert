import logging

import pandas as pd

from src.utils import setup_logger
from src.merge import collapse_obs, merge



def merge_datasets(dataset_files, merged_set_output, obs_files, pool_file, merge_method='on_disk'):
    # read pool
    pool = pd.read_csv(pool_file, index_col=0)
    # collect all .obs data from the datasets
    obs = collapse_obs(obs_files)
    logging.info(f'Prepared meta-set for merge with: {obs.shape[0]} cells, {pool.shape[0]} genes')
    # merge datasets and write to file
    merge(dataset_files, merged_set_output, obs=obs, var=pool, method=merge_method)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        merge_datasets(
            dataset_files=snakemake.input.dataset_files,
            pool_file=snakemake.input.pool,
            obs_files=snakemake.input.obs_files,
            merged_set_output=snakemake.output.merged_set,
            merge_method=snakemake.params.merge_method
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
