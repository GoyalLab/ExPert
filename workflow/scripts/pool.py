import logging

import pandas as pd
from typing import Iterable, Literal

from src.utils import setup_logger, make_obs_names_unique


def get_pool(files: Iterable[str], method: Literal['intersection', 'union'] = 'intersection'):
    # Read all .var annotations from datasets
    pool = [pd.read_csv(file, index_col=0) for file in files]
    # Determine union and intersection of indices (genes)
    all_genes = [set(v.index) for v in pool]
    union_genes = set.union(*all_genes)
    intersection_genes = set.intersection(*all_genes)
    logging.info(f'Union unique genes: {len(union_genes)}, intersection: {len(intersection_genes)}')
    # When selecting shared genes, take annotation from first dataset and mark the dataset
    if method == 'intersection':
        pool = pool[0].loc[sorted(list(intersection_genes)),:]
    # For union, combine all annotations 
    else:
        shared_cols = list(set.intersection(*[set(v.columns) for v in pool]))
        pool = pd.concat([v.loc[:,shared_cols] for v in pool], axis=1)
    logging.info(f'HVG pool shape: {pool.shape[0]} genes, {pool.shape[1]} var columns')
    return pool


def build_pool(hvg_files: Iterable[str], pool_output: str, method: Literal['intersection', 'union'] = 'intersection'):
    # Determine overall gene pool to use for meta-set
    pool = get_pool(hvg_files, method=method)
    pool.to_csv(pool_output)
    


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        build_pool(
            hvg_files=snakemake.input.hvg_files,
            pool_output=snakemake.output[0],
            method=snakemake.params.var_merge
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")