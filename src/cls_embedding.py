import numpy as np
import pandas as pd
import pickle
import logging
import anndata as ad
import scanpy as sc
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
            scaling_factor: float = 1.0,
            misc_method: Literal['mean', 'gaussian', 'zeros'] = 'mean',
            std: float = 1e-3,
            filter_embedding: bool = True,
            add_emb_for_features: bool = False,
            cls_emb_uns_key: str = 'cls_embedding',
            gene_embedding_varm_key: str = 'gene_embedding',
            pos_key: str | None = 'pos', 
            neg_key: str | None = 'neg',
            pos_type: str = 'crispra',
            additional_targets: np.ndarray | None = None,
            cls_label: str = 'cls_label',
            sim_cutoff: float = 0.8,
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
        self.add_emb_for_features = add_emb_for_features
        self.cls_emb_uns_key = cls_emb_uns_key
        self.gene_embedding_varm_key = gene_embedding_varm_key
        self.additional_targets = additional_targets
        self.cls_label = cls_label
        self.sim_cutoff = sim_cutoff
        self.pos_key = pos_key
        self.neg_key = neg_key
        self.pos_type = pos_type
        self.is_directional = False

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

    def _filter_emb(self, observed_genes: list[str], inplace: bool = True) -> None | pd.DataFrame:
        """Filter external embedding based on observed genes and additional targets."""
        # Check if embedding is not directional
        self.assert_not_directional()
        # Always include all observed classes
        targets = set(observed_genes)
        # Include additional targets if given
        if self.additional_targets is not None:
            targets |= set(self.additional_targets)
        # Subset classes in embedding according to targets
        available_targets = sorted(list(targets & set(self.emb.index)))
        logging.info(f'Found {len(available_targets)}/{len(observed_genes)} perturbations in resource')
        # Filter embedding
        filtered_emb = self.emb.loc[list(available_targets)].copy()
        # Inplace subset embedding
        if inplace:
            self.emb = filtered_emb
            return
        # Return filtered embedding
        else:
            return filtered_emb
    
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
        # Check if embedding is not directional
        self.assert_not_directional()
        if self.emb is None:
            raise ValueError('First initialize self.emb before adding rows.')
        ctrl_row = self._get_misc_row(self.emb)
        unknown_row = self._get_misc_row(self.emb)
        zero_rows = pd.DataFrame(np.concatenate([ctrl_row, unknown_row], axis=0), index=[self.ctrl_key, self.unknown_key])
        return pd.concat([self.emb, zero_rows], axis=0)

    def _add_direction_to_emb(self, adata: ad.AnnData, sep: str = ';') -> None:
        # Check if adata has multiple directions or no keys are given
        if self.pos_key is None and self.neg_key is None or adata.obs[self.p_type_col].nunique() < 2:
            logging.info('Saving embedding to adata without direction.')
            return
        # Get adata's directions
        has_pos = (adata.obs[self.p_type_col].str.lower()==self.pos_type).any()
        has_neg = (adata.obs[self.p_type_col].str.lower()!=self.pos_type).any()
        # Collect directional embeddings
        emb_list = []
        if self.neg_key is not None and has_neg:
            logging.info(f'Adding negative({self.neg_key}) direction to embedding.')
            emb_neg = self.emb.copy()
            emb_neg.index = f'{self.neg_key}{sep}' + emb_neg.index.astype(str)
            emb_list.append(emb_neg)
        if self.pos_key is not None and has_pos:
            logging.info(f'Adding positive({self.pos_key}) direction to embedding.')
            emb_pos = self.emb * -1
            emb_pos.index = f'{self.pos_key}{sep}' + emb_pos.index.astype(str)
            emb_list.append(emb_pos)
        # Concat embeddings
        self.emb = pd.concat(emb_list)
        # Set directional as true
        self.is_directional = True

    def _add_emb_to_uns(self, adata: ad.AnnData, sep: str = ';', direction_col_key: str = 'perturbation_direction') -> None:
        # Determine class labels
        if self.is_directional:
            # Determine directionality
            adata.obs[direction_col_key] = adata.obs[self.p_type_col].str.lower().str.startswith('crispra').apply(
                lambda x: self.pos_key if x else self.neg_key
            )
            # Add direction to adata class labels
            adata.obs[self.cls_label] = adata.obs[direction_col_key].astype(str) + sep + adata.obs[self.p_col].astype(str)
        else:
            adata.obs[self.cls_label] = adata.obs[self.p_col]
        # Transform indices to dimensions
        self.emb.columns = 'dim_' + self.emb.columns.astype(str)
        # Add embedding to uns
        adata.uns[self.cls_emb_uns_key] = self.emb
            
    def assert_not_directional(self):
        """Helper method to check if embedding is not directional"""
        if self.is_directional:
            raise ValueError('Embedding has already been directionalized. This operation must be performed before adding directionality.')
        
    def _add_emb_to_varm(self, adata: ad.AnnData, raw_emb: pd.DataFrame):
        """Add embedding to .varm for feature annotation"""
        # Check if embedding is not directional
        self.assert_not_directional()
        """Add embedding to .varm for feature annotation"""
        # Filter genes for embeddings genes
        adata = adata[:,adata.var_names.intersection(raw_emb.index)].copy()
        # Add gene embedding to .obsm
        adata.varm[self.gene_embedding_varm_key] = raw_emb.loc[adata.var_names,:]
        # Add names to dimensions
        adata.varm[self.gene_embedding_varm_key].columns = 'dim_' + adata.varm[self.gene_embedding_varm_key].columns.astype(str)

    def _remove_classes_without_emb(self, adata: ad.AnnData):
        """Remove perturbation classes that do not have an embedding representation."""
        self.assert_not_directional()
        # Remove non-zero embeddings
        non_zero_emb_mask = self.emb.sum(axis=1)!=0
        # Filter classes for intersection with embedding
        embedding_mask = adata.obs[self.p_col].isin(self.emb[non_zero_emb_mask].index)
        # Create control mask for data to not remove them
        ctrl_mask = adata.obs[self.p_col]==self.ctrl_key
        # Subset adata to perturbation with embeddings or control cells
        mask = (embedding_mask | ctrl_mask)
        if not mask.all():
            adata._inplace_subset_obs(mask)
        # Check number of unique perturbations to classify
        logging.info(f'Initializing dataset with {adata.obs[self.p_col].nunique()} classes.')

    def _filter_classes_by_similarity(self, adata: ad.AnnData, sim_cutoff: float = 0.8, return_graph: bool = False):
        """Filter classes by similarity cutoff.""" 
        # Check if embedding is not directional
        self.assert_not_directional()
        import networkx as nx
        # Calculate class supports
        class_support = adata.obs[self.p_col].value_counts()
        # Filter class support for available classes in embedding
        class_support = class_support[class_support.index.isin(self.emb.index)].copy()
        ps = class_support.index.astype(str)
        # Subset embedding to observable classes
        obs_emb = self.emb.loc[ps]
        # Calculate class similarities
        cls_sim = (obs_emb @ obs_emb.T).values
        # Exclude similarity to self
        np.fill_diagonal(cls_sim, 0)
        n = cls_sim.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        # Add edges for classes above cutoff
        for i in range(n):
            for j in range(i + 1, n):
                if cls_sim[i, j] >= sim_cutoff:
                    G.add_edge(i, j, weight=cls_sim[i, j])
        # Pick one node out of every connected component
        keep_idx = []
        for comp in nx.connected_components(G):
            comp = list(comp)
            # Pick index with maximum support
            rep = class_support.iloc[comp].idxmax()
            keep_idx.append(rep)
        # Subset adata
        adata._inplace_subset_obs(adata.obs[self.p_col].isin(keep_idx))
        logging.info(f'Adata has {adata.obs[self.p_col].nunique()} classes after similarity filtering.')
        if return_graph:
            return G
        
    def log_adata(self, adata: ad.AnnData):
        """Log adata stats."""
        # Basic adata information
        out = [
            f'Adata:',
            f' - shape: {adata.shape}',
            f' - n_cls: {adata.obs[self.p_col].nunique()}'
        ]
        # Add cls label if it exists
        if self.cls_label in adata.obs:
            out.extend([f' - n_cls_label: {adata.obs[self.cls_label].nunique()}'])
        # Add class embedding information
        if self.cls_emb_uns_key in adata.uns:
            emb: pd.DataFrame = adata.uns[self.cls_emb_uns_key]
            out.extend([
                f'Class emb:',
                f' - shape: {emb.shape}',
            ])
        # Log full adata information
        logging.info('\n'.join(out))

    def process(self, adata: ad.AnnData) -> None:
        """Process and add class embedding key to adata."""
        logging.info('Reading embedding.')
        self.emb = self._read_embedding()
        # Annotate adata features with embedding
        if self.add_emb_for_features:
            self._add_emb_to_varm(adata, raw_emb=self.emb)
        logging.info(f'Removing classes without embeddings.')
        self._remove_classes_without_emb(adata)
        # Filter embedding for observed genes
        if self.filter_embedding:
            observed_genes = adata.obs[self.p_col].unique().tolist()
            logging.info(f'Filtering embedding for perturbed genes ({len(observed_genes)}).')
            self._filter_emb(observed_genes)
            if self.sim_cutoff > 0:
                logging.info('Filtering adata for low interclass similarities.')
                self._filter_classes_by_similarity(adata, sim_cutoff=self.sim_cutoff)
        # Add control and unknown embeddings
        self.emb = self._add_misc_rows()
        # Scale embedding values
        if self.scaling_factor > 0:
            self.emb *= self.scaling_factor
        # Add directionality to embeddings 
        self._add_direction_to_emb(adata)
        logging.info(f'Adding embedding to adata.')
        self._add_emb_to_uns(adata)
        logging.info(f'Removin empty cells or genes from adata.')
        # Remove genes with no counts (zero-padded)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        self.log_adata(adata)
