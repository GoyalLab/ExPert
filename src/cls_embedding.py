import numpy as np
import pandas as pd
import pickle
import logging
import anndata as ad
from typing import Literal
import logging

from src.statics import OBS_KEYS


class EmbeddingProcessor:
    SUPPORTED_TYPES = ['.pickle', '.csv', '.tsv']

    def __init__(
            self, 
            emb_p: str, 
            p_col: str = OBS_KEYS.PERTURBATION_KEY, 
            p_type_col: str = OBS_KEYS.PERTURBATION_TYPE_KEY,
            ctrl_key: str = OBS_KEYS.CTRL_KEY,
            unknown_key: str = 'unknown', 
            scaling_factor: int = 1,
            misc_method: Literal['mean', 'gaussian', 'zeros'] = 'mean',
            std: float = 1e-3,
            filter_embedding: bool = True,
        ):
        # Init class settings
        self.emb_p = emb_p
        self.p_col = p_col
        self.p_type_col = p_type_col
        self.ctrl_key = ctrl_key
        self.unknown_key = unknown_key
        self.scaling_factor = scaling_factor
        self.emb = None
        self.misc_method = misc_method
        self.std = std
        self.filter_embedding = filter_embedding

    def _read_embedding(self) -> pd.DataFrame:
        if self.emb_p.endswith('.pickle'):
            with open(self.emb_p, 'rb') as file:
                emb = pd.DataFrame(pickle.load(file)).T
        elif self.emb_p.endswith('.csv'):
            emb = pd.read_csv(self.emb_p, index_col=0)
        elif self.emb_p.endswith('.tsv'):
            emb = pd.read_csv(self.emb_p, sep='\t', index_col=0)
        else:
            raise ValueError(f'Unsupported embedding file format provided.')
        return emb

    def _filter_emb(self, observed_genes: list[str]) -> pd.DataFrame:
        available_targets = set(observed_genes).intersection(self.emb.index)
        logging.info(f'Found {len(available_targets)}/{len(observed_genes)} perturbations in resource')
        return self.emb.loc[list(available_targets)]
    
    def _get_misc_row(self, emb: pd.DataFrame) -> np.ndarray:
        shape = (1, emb.shape[1])
        if self.misc_method == 'zeros':
            return np.zeros(shape=shape)
        elif self.misc_method == 'mean':
            return np.matrix(emb.mean(axis=0))
        elif self.misc_method == 'gaussian':
            return np.random.normal(loc=0, scale=self.std, size=shape)
        else:
            raise ValueError(f"misc_method has to be one of 'mean', 'gaussian', 'zeros', got {self.misc_method}")
    
    def _add_misc_rows(self) -> pd.DataFrame:
        if self.emb is None:
            raise ValueError('First initialize self.emb before adding rows.')
        ctrl_row = self._get_misc_row(self.emb)
        unknown_row = self._get_misc_row(self.emb)
        zero_rows = pd.DataFrame(np.concatenate([ctrl_row, unknown_row], axis=0), index=[self.ctrl_key, self.unknown_key])
        return pd.concat([self.emb, zero_rows], axis=0)

    def _add_direction_to_emb(self, pos_key: str | None = 'pos', neg_key: str | None = 'neg', sep: str = ';') -> pd.DataFrame:
        if pos_key is None and neg_key is None:
            logging.info('Saving embedding to adata without direction.')
            return self.emb
        emb_list = []
        if pos_key is not None:
            logging.info(f'Adding {pos_key} direction to embedding.')
            emb_pos = self.emb.copy()
            emb_pos.index = f'{pos_key}{sep}' + emb_pos.index.astype(str)
            emb_list.append(emb_pos)
        if neg_key is not None:
            logging.info(f'Adding {neg_key} direction to embedding.')
            emb_neg = self.emb * -1
            emb_neg.index = f'{neg_key}{sep}' + emb_neg.index.astype(str)
            emb_list.append(emb_neg)
        return pd.concat(emb_list)

    def _add_emb_to_adata(self, adata: ad.AnnData, pos_key: str = 'pos', neg_key: str = 'neg', sep: str = ';',
                          direction_col_key: str = 'perturbation_direction', cls_emb_uns_key: str = 'cls_embedding') -> None:
        adata.obs[direction_col_key] = adata.obs[self.p_type_col].str.startswith('CRISPRa').apply(
            lambda x: pos_key if x else neg_key)
        cls_labels = (adata.obs[direction_col_key].astype(str) + ';' + adata.obs[self.p_col].astype(str)).unique()
        self.emb.columns = 'dim_' + self.emb.columns.astype(str)
        if self.filter_embedding:
            adata.uns[cls_emb_uns_key] = self.emb.loc[list(set(cls_labels).intersection(set(self.emb.index))), :]
        else:
            adata.uns[cls_emb_uns_key] = self.emb

    def process(self, adata: ad.AnnData, pos_key: str | None = 'pos', neg_key: str | None = 'neg') -> None:
        """Process and add class embedding key to adata."""
        observed_genes = adata.obs[self.p_col].unique().tolist()
        logging.info('Reading embedding.')
        self.emb = self._read_embedding()
        if self.filter_embedding:
            logging.info(f'Filtering embedding for perturbed genes ({len(observed_genes)}).')
            self.emb = self._filter_emb(observed_genes)
        self.emb = self._add_misc_rows()
        self.emb *= self.scaling_factor
        self.emb = self._add_direction_to_emb(pos_key=pos_key, neg_key=neg_key)
        logging.info(f'Adding embedding to adata.')
        self._add_emb_to_adata(adata)
