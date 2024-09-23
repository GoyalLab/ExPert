import anndata as ad


def hvg_pool(dataset_file, output_file):
    if dataset_file.endswith('.h5ad'):
        adata = ad.read(dataset_file)
        # save hvg gene info
        adata[:, adata.var.highly_variable].var.to_csv(output_file, index=True)
    else:
        raise FileNotFoundError(f'{dataset_file} is not a .h5ad file.')


if __name__ == '__main__':
    try:
        hvg_pool(
            dataset_file=snakemake.input[0],
            output_file=snakemake.output[0]
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")