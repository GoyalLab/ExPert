from src.utils import setup_logger
from src.preprocess import preprocess_dataset
import logging
import scanpy as sc
from pathlib import Path


def process_dataset(dataset_file, output_path, **kwargs):
    # read dataset from downloaded file
    ds = sc.read(dataset_file)
    # prepare dataset
    ds = preprocess_dataset(ds, **kwargs)
    ds.write_h5ad(output_path, compression='gzip')


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        process_dataset(
            dataset_file=snakemake.input.dataset_file,
            output_path=snakemake.output.processed,
            cancer=snakemake.params.is_cancer,
            name=snakemake.params.name,
            qc=snakemake.params.qc,
            norm=snakemake.params.norm,
            log=snakemake.params.log_norm,
            scale=snakemake.params.scale,
            n_hvg=snakemake.params.n_hvg,
            subset=snakemake.params.subset,
            n_ctrl=snakemake.params.n_ctrl,
            single_perturbations_only=snakemake.params.single_perturbations_only,
            p_col=snakemake.params.p_col,
            ctrl_key=snakemake.params.ctrl_key,
            seed=snakemake.params.seed
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
        # Optionally, add fallback behavior for direct execution
