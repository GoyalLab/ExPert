from src.utils import setup_logger
from src.preprocess import preprocess_dataset
import logging
import scanpy as sc
from pathlib import Path


def process_dataset(dataset_file, output_path, qc=True, norm=True, log=True, scale=True, n_hvg=2000, subset=False):
    # read dataset from downloaded file
    ds = sc.read(dataset_file)
    # get dataset_file name
    ds_name = Path(dataset_file).stem
    # prepare dataset
    logging.info(f"Preparing dataset {ds_name}")
    ds = preprocess_dataset(ds, ds_name, qc=qc, norm=norm, log=log, scale=scale, n_hvg=n_hvg, subset=subset)
    logging.info(f"Finished preparing dataset {ds_name}")
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
            qc=snakemake.params.qc,
            norm=snakemake.params.norm,
            log=snakemake.params.log_norm,
            scale=snakemake.params.scale,
            n_hvg=snakemake.params.n_hvg,
            subset=snakemake.params.subset
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
        # Optionally, add fallback behavior for direct execution
