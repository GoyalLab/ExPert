import scanpy as sc
import pandas as pd
import logging
import numpy as np
import scipy.sparse as sp
import anndata as ad
from typing import List, Literal


class MetasetCompressor:
    def __init__(self, adata: ad.AnnData, min_cells: int = 50, group_labels: List[str] = ['celltype', 'perturbation'], remove_control: bool = False, output_file: str = 'gpp.h5ad'):
        """
        Initialize the MetasetCompressor.

        Parameters:
        adata (ad.AnnData): The annotated data matrix.
        min_cells (int): Minimum number of cells required for a perturbation to be kept.
        group_labels (List[str]): List of labels to group by.
        remove_control (bool): Whether to remove control samples.
        output_file (str): Path to the output file.
        """
        self.adata = adata
        self.min_cells = min_cells
        self.group_labels = group_labels
        self.exact_label = 'exact_perturbation'
        self.remove_control = remove_control
        self.output_file = output_file

    def loose_ctrl(self, dataset_label: str = 'dataset', perturbation_label: str = 'perturbation', ctrl_pattern: str = 'control') -> None:
        """
        Remove control samples and center the dataset.

        Parameters:
        dataset_label (str): Label for the dataset.
        perturbation_label (str): Label for the perturbation.
        ctrl_pattern (str): Pattern to identify control samples.
        """
        if sp.issparse(self.adata.X):
            self.adata.X = self.adata.X.todense()
        for dataset in self.adata.obs[dataset_label].unique():
            logging.info(f'Centering {dataset} in meta-set')
            ds = self.adata.obs[self.adata.obs[dataset_label] == dataset]
            cs = ds[ds[perturbation_label].str.startswith(ctrl_pattern)]
            control_anchor = self.adata[cs.index].X.mean(axis=0)
            self.adata[ds.index].X -= np.round(control_anchor)
        control_mask = self.adata.obs[perturbation_label].str.startswith(ctrl_pattern)
        logging.info(f'Removing {np.sum(control_mask)} control samples from meta-set')
        self.adata._inplace_subset_obs(~control_mask)

    def aggregate_per_perturbation(self, label: str = 'perturbation', func: Literal['mean', 'sum', 'var', 'count_nonzero'] = 'mean', eps: float = 1e-9) -> ad.AnnData:
        """
        Aggregate data per perturbation.

        Parameters:
        label (str): Label to aggregate by.
        func (Literal['mean', 'sum', 'var', 'count_nonzero']): Aggregation function.
        eps (float): Epsilon value to enforce true zeros.

        Returns:
        ad.AnnData: Aggregated annotated data matrix.
        """
        tmp = sc.get.aggregate(self.adata, by=label, func=func)
        # Enforce some true 0s
        tmp.layers[func][tmp.layers[func] <= eps] = 0
        # Set .X to aggregated layer and convert to sparse matrix
        tmp.X = sp.csr_matrix(tmp.layers[func])
        return tmp

    def filter_adata_by_label(self, label: str = 'exact_perturbation') -> None:
        """
        Filter annotated data by label.

        Parameters:
        label (str): Label to filter by.
        """
        if len(self.group_labels) > 1:
            self.adata.obs[label] = self.adata.obs[self.group_labels].apply(lambda row: ';'.join(row.values.astype(str)), axis=1)
        else:
            self.adata.obs[label] = self.adata.obs[self.group_labels[0]]
        pert_counts = self.adata.obs[label].value_counts()
        valid_perturbations = pert_counts[pert_counts >= self.min_cells].index
        self.adata = self.adata[self.adata.obs[label].isin(valid_perturbations) & (~pd.isna(self.adata.obs[label]))]

    def _ensure_sparse_matrix(self) -> None:
        """
        Ensure that the data matrix is in CSR sparse format.
        """
        if not sp.isspmatrix_csr(self.adata.X):
            logging.info(f'Converting .X to csr')
            self.adata.X = sp.csr_matrix(self.adata.X)

    def process(self):
        """
        Process the annotated data matrix by filtering, removing controls, and aggregating.
        """
        self._ensure_sparse_matrix()                                                
        self.filter_adata_by_label(label=self.exact_label)
        if self.remove_control:
            self.loose_ctrl(dataset_label='dataset', perturbation_label=self.exact_label)
        agg_adata = self.aggregate_per_perturbation(label=self.exact_label)
        logging.info(f'Saving compressed AnnData to {self.output_file}')
        agg_adata.write_h5ad(self.output_file, compression='gzip')
