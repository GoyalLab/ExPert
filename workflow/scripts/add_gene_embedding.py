import os
import logging
import pandas as pd
import scanpy as sc

from src.utils import setup_logger
from src.cls_embedding import EmbeddingProcessor


def _add_emb(adata_p: str, gene_emb_p: str, ctx_emb_p: str, output_path: str, **kwargs):
    ep = EmbeddingProcessor(gene_emb_p, **kwargs)
    logging.info(f'Reading adata from: {adata_p}')
    adata = sc.read(adata_p)
    ep.process(adata)
    # Add context embedding if path exists
    if os.path.exists(ctx_emb_p):
        logging.info(f'Adding context embeddings from {ctx_emb_p}.')
        # Add context embedding
        adata.uns['ctx_embedding'] = pd.read_csv(ctx_emb_p, index_col=0)

    logging.info(f'Writing adata with gene embedding to {output_path}')
    adata.write_h5ad(output_path, compression='gzip')

if __name__ == "__main__":
    # Add gene embedding to adata
    try:
        # handle logging
        log_file = snakemake.log[0]
        setup_logger(log_file)
        _add_emb(
            adata_p=snakemake.input.input_file,
            gene_emb_p=snakemake.params.gene_embedding_file,
            ctx_emb_p=snakemake.params.ctx_embedding_file,
            output_path=snakemake.output.output_file,
            add_emb_for_features=snakemake.params.add_emb_for_features,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")