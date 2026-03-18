import scanpy as sc

from src.utils import setup_logger
from src.efficiency import compute_perturbation_scores


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        # Read input adata file
        adata = sc.read(snakemake.input.dataset_file)
        compute_perturbation_scores(
            adata,
            perturbation_key=snakemake.params.perturbation_col,
            ctrl_key=snakemake.params.ctrl_key,
            min_de_genes=snakemake.params.min_deg,
            n_jobs=snakemake.resources.cpus_per_task,
        )
        # Save updated adata to output file
        adata.write_h5ad(snakemake.output.filtered_file)
    except NameError:
        print("This script is meant to be run through Snakemake.")
