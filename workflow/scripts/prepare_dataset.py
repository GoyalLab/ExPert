import pandas as pd

from src.utils import setup_logger
from src.prepare import prepare_merge


def prepare_dataset(dataset_file: str, pool_file: str, prepared_path: str, obs_path: str, zero_pad: bool, **kwargs):
    # read gene pool
    pool = pd.read_csv(pool_file, index_col=0)
    # subset dataset, make obs names unique, and save obs to file
    obs = prepare_merge(dataset_file, pool.index, prepared_path, zero_pad=zero_pad, **kwargs)
    obs.to_csv(obs_path)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        prepare_dataset(
            dataset_file=snakemake.input.dataset_file,
            pool_file=snakemake.input.pool,
            prepared_path=snakemake.output.prepared,
            obs_path=snakemake.output.obs,
            zero_pad=snakemake.params.zero_pad,
            **snakemake.params.kwargs,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")