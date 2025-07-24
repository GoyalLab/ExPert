from src.utils import setup_logger
from src.info import create_meta_summary


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        create_meta_summary(
            input_files=snakemake.input.input_files,
            perturbation_pool_file=snakemake.output.perturbation_pool_file,
            feature_pool_file=snakemake.output.feature_pool_file,
            plt_dir=snakemake.params.plt_dir,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
