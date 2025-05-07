from src.utils import download_file, setup_logger
import logging


def download_dataset(dataset_link, dataset_name, output_path, cache=True):
    # download/cache dataset
    logging.info(f"Downloading/caching dataset {dataset_name}")
    msg = download_file(dataset_link, output_path, cache=cache)
    logging.info(msg)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        download_dataset(
            dataset_link=snakemake.params.url,
            dataset_name=snakemake.params.name,
            output_path=snakemake.output.raw,
            cache=snakemake.params.cache
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
