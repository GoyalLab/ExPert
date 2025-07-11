from src.utils import setup_logger
from src.info import create_perturbation_pool


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        create_perturbation_pool(
            input_files=snakemake.input.input_files,
            out_file=snakemake.output.perturbation_pool_file,
            plt_dir=snakemake.params.plt_dir,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
