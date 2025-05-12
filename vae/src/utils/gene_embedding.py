# TODO: handle expansion of gene embedding in this class

from typing import Iterable
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
from scvi.module.base import auto_move_data
import logging


class GeneEmbedding(torch.nn.Module):

    def _set_embedding(self) -> None:
        # Extract gene names and dimensions from embedding
        emb_obs = pd.DataFrame({'gene': self.embedding.index.tolist()})
        emb_var = pd.DataFrame({'gene': self.embedding.columns.tolist()}, index=self.embedding.columns.tolist())

        # Determine missing genes
        missing_genes = set(self.perturbations).difference(self.embedding.index)
        n_missing = len(missing_genes)
        if n_missing > 0:
            logging.info(f'{n_missing} are missing from embedding. Initializing with {self.fill_na}.')
            missing_emb = np.full((n_missing, self.embedding.shape[1]), self.fill_na)
            missing_obs = pd.DataFrame({'gene': list(missing_genes)})
        # Add control class to embedding
        emb = np.array(self.embedding)
        # Add missing genes to 
        emb = np.concatenate([emb, missing_emb], axis=0)
        obs = pd.concat([emb_obs, missing_obs], axis=0).reset_index(drop=True)
        # Set control value
        if self.ctrl_val != self.fill_na:
            logging.info(f'Initializing {self.ctrl_key} embedding with {self.ctrl_val}.')
            emb[np.where(obs.gene==self.ctrl_key),:] = self.ctrl_val
        # Sort data by gene
        obs.sort_values(by='gene', inplace=True)
        # Arrange data to match sorted genes
        emb = emb[obs.index].copy()
        self.adata = ad.AnnData(emb, obs=obs, var=emb_var)
        self.X = torch.tensor(self.adata.X, dtype=torch.float)

    def __init__(
        self,
        embedding: pd.DataFrame,                                          # Actual embedding data 
        perturbations: pd.Series[str],                                            # Perturbed genes in adata
        ctrl_key: str = 'control',
        ctrl_val: float = 0,
        fill_na: float = 0
    ):
        super().__init__()
        self.embedding = embedding
        self.perturbations = perturbations.sort_values()                # Ensure perturbed genes are sorted
        self.ctrl_key = ctrl_key
        self.ctrl_val = ctrl_val
        self.fill_na = fill_na
        # Init class embedding
        self._set_embedding()

    @auto_move_data
    # Assumes labels are factorized and based on sorted genes
    def forward(self, label_idc: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Expand gene embedding for batch
        x = self.X[label_idc]
        return x * scores
    