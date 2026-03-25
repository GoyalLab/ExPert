import logging

from src.utils import setup_logger
from src.merge import merge


def merge_datasets(
        dataset_files: list[str],
        merged_set_output: str,
        merge_method: str = 'concat_on_disk'
    ) -> None:
    dataset_files = sorted(dataset_files)
    merge(dataset_files, merged_set_output, method=merge_method)


if __name__ == "__main__":
    log_file = snakemake.log[0]
    setup_logger(log_file)
    try:
        merge_datasets(
            dataset_files=snakemake.input.dataset_files,
            merged_set_output=snakemake.output.merged_set,
            merge_method=snakemake.params.merge_method
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
