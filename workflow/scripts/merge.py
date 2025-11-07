import logging
from pathlib import Path
import pandas as pd

from src.utils import setup_logger
from src.merge import collapse_obs, merge
from src.statics import DATA_SHEET_KEYS, OBS_KEYS
from typing import List



def get_basenames(fs: List[str]) -> List[str]:
    return [Path(f).stem for f in fs]


def check_files(fs1: List[str], fs2: List[str]) -> None:
    assert get_basenames(fs1) == get_basenames(fs2)


def merge_datasets(
        dataset_files: List[str], 
        merged_set_output: str, 
        obs_files: List[str], 
        pool_file: str, 
        meta_sheet: pd.DataFrame, 
        merge_method: str = 'dask'
    ) -> None:
    # ensure order of datasets of insertion and .obs is the same
    dataset_files = sorted(dataset_files)
    obs_files = sorted(obs_files)
    check_files(dataset_files, obs_files)
    # read pool
    pool = pd.read_csv(pool_file, index_col=0)
    # collect all .obs data from the datasets
    obs = collapse_obs(obs_files)
    # Add top level meta sheet information to obs
    meta = pd.DataFrame({
        OBS_KEYS.DATASET_KEY: meta_sheet.index.values,
        OBS_KEYS.PERTURBATION_TYPE_KEY: meta_sheet[DATA_SHEET_KEYS.PERTURBATION_TYPE].values,
        OBS_KEYS.CELL_TYPE_KEY: meta_sheet[DATA_SHEET_KEYS.CELL_TYPE].values,
        OBS_KEYS.CONTEXT_KEY: meta_sheet[DATA_SHEET_KEYS.CONTEXT].values,
    })
    obs = obs.merge(meta, on=OBS_KEYS.DATASET_KEY)
    # reduce .var to the index
    var = pd.DataFrame(index=list(pool.index))
    logging.info(f'Prepared meta-set for merge with: {obs.shape[0]} cells, {pool.shape[0]} genes')
    # merge datasets and write to file
    merge(dataset_files, merged_set_output, obs=obs, var=var, method=merge_method)


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
            meta_sheet=snakemake.params.meta_sheet,
            merge_method=snakemake.params.merge_method
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
