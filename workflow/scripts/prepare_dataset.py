import pandas as pd

from src.utils import setup_logger
from src.prepare import prepare_merge


def prepare_dataset(dataset_file, pool_file, prepared_path, obs_path, hvg=True, zero_pad=True):
    # read gene pool
    pool = pd.read_csv(pool_file, index_col=0)
    # subset dataset, make obs names unique, and save obs to file
    obs = prepare_merge(dataset_file, pool.index, prepared_path, hvg=hvg, zero_pad=zero_pad)
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
            hvg=snakemake.params.hvg,
            zero_pad=snakemake.params.zero_pad,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")