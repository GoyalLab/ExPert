import logging
import anndata as ad


def merge(input_pths, out_pth: str, method: str = 'concat_on_disk'):
    logging.info(f'Merging {len(input_pths)} files to {out_pth} using {method}')
    # Load everything into memory and concat (expensive)
    if method == 'in_memory':
        ds_list = [ad.read_h5ad(f) for f in input_pths]
        ad.concat(ds_list).write_h5ad(out_pth, compression='gzip')
    # Use anndata on-disk merge function
    else:
        # TODO: check if that makes prepare step unnecessary when using join='outer'
        ad.experimental.concat_on_disk(input_pths, out_pth, join='inner')
    logging.info(f'Merge complete: {out_pth}')
