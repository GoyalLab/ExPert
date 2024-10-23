
import logging

from src.utils import setup_logger
from src.harmonize import harmonize_metaset



def harmonize(metaset_path, harmonized_output, method='scANVI', umap=True, model_dir='./scanvi'):
    harmonize_metaset(metaset_path, harmonized_output, method=method, umap=umap, model_dir=model_dir)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        harmonize(
            metaset_path=snakemake.input.merged,
            harmonized_output=snakemake.output.harmonized,
            method=snakemake.params.method,
            umap=snakemake.params.umap,
            model_dir=snakemake.params.model_dir
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
