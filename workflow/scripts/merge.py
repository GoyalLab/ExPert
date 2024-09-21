from src.harmonize import harmonize
from src.utils import setup_logger


def get_hvg_pool(files):
    pool = set()
    for file in files:
        with open(file, 'r') as f:
            hvg = set(line.strip() for line in f)
        pool.update(hvg)
    return pool


def save_pool(pool, file):
    with open(file, 'w') as f:
        f.write('\n'.join(pool))

def merge_datasets(dataset_files, hvg_files, merged_set_output, pool_output, method='harmony'):
    # collect pool of highly variable genes over all datasets
    pool = get_hvg_pool(hvg_files)
    save_pool(pool, pool_output)
    # read all preprocessed datasets, combine, and correct for batch effects
    merged = harmonize(dataset_files, pool, method=method)
    # write final dataset to file
    merged.write_h5ad(merged_set_output)


if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # execute dataset preparation
    try:
        merge_datasets(
            dataset_files=snakemake.input.dataset_files,
            hvg_files=snakemake.input.hvg_files,
            merged_set_output=snakemake.output.merged_set,
            pool_output=snakemake.output.pool,
            method=snakemake.params.method
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
