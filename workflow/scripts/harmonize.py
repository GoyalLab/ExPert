
from src.utils import setup_logger
from src.harmonize import harmonize_metaset



def harmonize(metaset_path, merged_set_output, method='scANVI', umap=True):
    harmonize_metaset(metaset_path, method=method, umap=umap).write_h5ad(merged_set_output, compression='gzip')


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        harmonize(
            metaset_path=snakemake.input.merged,
            harmonized_output=snakemake.output.harmonized,
            method=snakemake.params.correction_method,
            umap=snakemake.params.umap
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
