
import os

from src.utils import setup_logger
from src.harmonize import Harmonizer



def harmonize(metaset_path, harmonized_output, method='scANVI', model_dir='./scanvi', mem_dir='./'):
    # set up harmonizer
    harmonizer = Harmonizer(metaset_file=metaset_path, method=method, mem_log_dir=mem_dir)
    # harmonize datasets
    harmonizer.harmonize(model_dir=model_dir)
    # save normalized data
    harmonizer.save_normalized_adata(harmonized_output)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    mem_log_dir = os.path.join(os.path.dirname(log_file), 'harmonize_mem')
    # execute dataset preparation
    try:
        harmonize(
            metaset_path=snakemake.input.merged,
            harmonized_output=snakemake.output.harmonized,
            method=snakemake.params.method,
            model_dir=snakemake.params.model_dir,
            mem_dir=mem_log_dir
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
