import logging

import scanpy as sc

from src.utils import setup_logger
    
from src.cls_embedding import EmbeddingProcessor

def _add_emb(adata_p: str, emb_p: str, output_path: str, **kwargs):
    ep = EmbeddingProcessor(emb_p, **kwargs)
    logging.info(f'Reading adata from: {adata_p}')
    adata = sc.read(adata_p)
    ep.process(adata)
    logging.info(f'Writing adata with gene embedding to {output_path}')
    adata.write_h5ad(output_path, compression='gzip')

if __name__ == "__main__":
    # handle logging
    log_file = snakemake.log[0]
    setup_logger(log_file)
    # Add gene embedding to adata
    try:
        _add_emb(
            adata_p=snakemake.input.input_file,
            emb_p=snakemake.params.embedding_file,
            output_path=snakemake.output.output_file,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")