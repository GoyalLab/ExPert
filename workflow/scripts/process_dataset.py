from src.utils import get_dataset, setup_logger
from src.preprocess import prepare_dataset
import logging


def process_dataset(dataset_link, dataset_name, output_path, cache=True, qc=True, norm=True, log=True, n_hvg=2000, subset=False):
    # download/cache dataset
    logging.info(f"Downloading/caching dataset {dataset_name}")
    ds = get_dataset(dataset_link, dataset_name, output_path, cache)
    # prepare dataset
    logging.info(f"Preparing dataset {dataset_name}")
    ds = prepare_dataset(ds, dataset_name, qc=qc, norm=norm, log=log, n_hvg=n_hvg, subset=subset)
    logging.info(f"Finished preparing dataset {dataset_name}")
    ds.write_h5ad(output_path)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        process_dataset(
            dataset_link=snakemake.params.url,
            dataset_name=snakemake.params.name,
            output_path=snakemake.output[0],
            cache=snakemake.params.cache,
            qc=snakemake.params.qc,
            norm=snakemake.params.norm,
            log=snakemake.params.log_norm,
            n_hvg=snakemake.params.n_hvg
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
        # Optionally, add fallback behavior for direct execution
