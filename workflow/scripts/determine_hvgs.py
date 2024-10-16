import anndata as ad


def hvg_pool(dataset_file, output_file, hvg=True):
    if dataset_file.endswith('.h5ad'):
        adata = ad.read(dataset_file, backed='r')
        # save hvg gene info
        if hvg:
            pool_var = adata[:, adata.var.highly_variable].var
        else:
            pool_var = adata.var
        pool_var.to_csv(output_file, index=True)
    else:
        raise FileNotFoundError(f'{dataset_file} is not a .h5ad file.')


if __name__ == '__main__':
    try:
        hvg_pool(
            dataset_file=snakemake.input[0],
            output_file=snakemake.output[0],
            hvg=snakemake.params.hvg
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")