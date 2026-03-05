from src.utils import setup_logger
from src.preprocess import preprocess_dataset
import scanpy as sc


def process_dataset(dataset_file, output_path, **kwargs):
    # prepare dataset
    ds = preprocess_dataset(dataset_file, **kwargs)
    # write prepared dataset to output file
    ds.write_h5ad(output_path)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        process_dataset(
            dataset_file=snakemake.input.dataset_file,
            output_path=snakemake.output.processed,
            perturbation_pool_file=snakemake.input.perturbation_pool_file,
            feature_pool_file=snakemake.input.feature_pool_file,
            cancer=snakemake.params.is_cancer,
            name=snakemake.params.name,
            qc=snakemake.params.qc,
            norm=snakemake.params.norm,
            log=snakemake.params.log_norm,
            scale=snakemake.params.scale,
            hvg=snakemake.params.hvg,
            n_hvg=snakemake.params.n_hvg,
            subset=snakemake.params.subset,
            n_ctrl=snakemake.params.n_ctrl,
            use_perturbation_pool=snakemake.params.use_perturbation_pool,
            use_feature_pool=snakemake.params.use_feature_pool,
            z_score_filter=snakemake.params.z_score_filter,
            control_neighbor_threshold=snakemake.params.control_neighbor_threshold,
            min_cells_per_perturbation=snakemake.params.min_cells_per_perturbation,
            single_perturbations_only=snakemake.params.single_perturbations_only,
            p_col=snakemake.params.p_col,
            ctrl_key=snakemake.params.ctrl_key,
            seed=snakemake.params.seed
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
        # Optionally, add fallback behavior for direct execution
