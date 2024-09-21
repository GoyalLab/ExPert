import anndata as ad


def hvg_pool(dataset_file, output_file):
    if dataset_file.endswith('.h5ad'):
        adata = ad.read(dataset_file)
        # collect gene names
        hvg = adata[:, adata.var.highly_variable].var_names
        with open(output_file, 'w') as f:
            f.write('\n'.join(hvg))
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