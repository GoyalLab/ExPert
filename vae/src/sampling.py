import warnings
import anndata as ad
import numpy as np
import scipy.sparse as sp
import pandas as pd
import logging
from tqdm import tqdm
from typing import Iterable, Tuple, Literal, List
from joblib import Parallel, delayed
import os
import scvi
import scanpy as sc


def sample_from_ctrl_cells(
        adata: ad.AnnData,                  # AnnData object
        cls_labels: List[str],              # Labels to group by
        ctrl_key: str = 'control',          # Key for control group in last cls label
        n_ctrl: int | None = None,          # Fixed number of cells to draw from each control
        x_ctrl: float = 2.5,                # x_ctrl = n_cells_ctrl / n_cells_perturbed (within each group)
    ) -> ad.AnnData:
    
    if adata.obs[cls_labels[:-1]].value_counts().shape[0] == 1:
        logging.info('Got only one group, sampling from one control pool.')
        cls_label = cls_labels[-1]
        # Manual subet for only one group, DON'T DO FOR OTHER DATASETS
        ctrl_mask = adata.obs[cls_label]==ctrl_key
        ctrl_codes = adata[ctrl_mask].obs.sample(n_ctrl, replace=False).index
        ctrl_idc = np.where(adata.obs.index.isin(ctrl_codes))[0]
        p_idc = np.where(~ctrl_mask)[0]
        return adata[np.concatenate([ctrl_idc, p_idc]).astype(int)]
    # Randomly sample N control cells from each control group
    ctrl_mask = adata.obs[cls_labels[-1]]==ctrl_key
    # Split .obs in control and perturbed
    ctrl_obs = adata.obs[ctrl_mask]
    p_obs = adata.obs[~ctrl_mask]
    # Determine N based on average number of cells per perturbation in group (cell type, etc.)
    cpp = p_obs.groupby(cls_labels[:-1], observed=True)[cls_labels[-1]].value_counts().reset_index()
    scpp = cpp.groupby(cls_labels[:-1], observed=True)['count'].mean().reset_index()
    # sample X times number of perturbed cells
    ctrl_cell_idc = (
        ctrl_obs
        .reset_index(names='cell_idx_reset')
        .merge(scpp, on=cls_labels[:-1])
        .groupby(cls_labels[:-1], observed=True)
        .apply(lambda group: group.sample(
            n=int(group['count'].unique()[0] * x_ctrl) if n_ctrl is None else n_ctrl, 
            replace=False
        )['cell_idx_reset'])
    ).reset_index()['cell_idx_reset']
    # subset adata to only have the sampled cells in control
    idc = p_obs.index.tolist()
    idc.extend(ctrl_cell_idc.tolist())
    logging.info(f'Resulting set has {len(idc)} cells.')
    return adata[idc]


def upsample_from_scvi(scvi_model: scvi.model.SCVI, cls_label: str, n_cells: int | None = None, n_control_cells: int | None = None, ctrl_pattern: str = 'control'):
    """
    Upsample cells from a trained scVI model to balance the number of cells per class.

    This function takes a trained scVI model and performs up-sampling to generate additional cells for each class,
    balancing the number of cells across classes. The generated cells are sampled from the posterior predictive
    distribution of the scVI model.

    Parameters:
    - scvi_model (scvi.model.SCVI): A trained scVI model.
    - cls_label (str): The column name in the adata.obs dataframe that contains the class labels.
    - n_cells (int, optional): The target number of cells per class. If None, the number of cells will be scaled to match
                             the class with the most cells. Default is None.
    - n_control_cells (int, optional): The target number of cells per control group. Determined by class labels that contain ctrl_pattern. Default is None
    - ctrl_pattern (str): Pattern of control labels. Default is 'control'

    Returns:
    ad.AnnData: An AnnData object containing the original and up-sampled cells with updated observations.
    """
    if not scvi_model.is_trained:
        raise ValueError('Please first train the given scvi model to use up-sampling.')
    obs = scvi_model.adata.obs.copy()
    
    cells_per_class = pd.DataFrame(obs[cls_label].value_counts())    # get number of cells for each class in adata of trained model
    if n_cells is None:                                                 # scale each class to the number of cells from the class with the most cells
        n_cells = cells_per_class['count'].max()
    n_control_cells = n_control_cells if n_control_cells is not None else n_cells
    # determine number of cells for each class depending on control pattern or not
    cells_per_class['n_cells'] = n_cells
    cells_per_class['n_cells'][cells_per_class.index.str.contains(ctrl_pattern)] = n_control_cells
    n_sim_per_class = cells_per_class['n_cells'] - cells_per_class['count']             # determine number of cells to change per class
    up_sample_rates = n_sim_per_class / cells_per_class['count']                             # determine up-sampling rates for each class

    if np.abs(up_sample_rates.max()) > 2:
        warnings.warn(f'Generating more samples than original class had for some of the classes provided, top 3: {up_sample_rates.sort_values(ascending=False).head(3)}')
    
    all_sim_pool = []                                                 # collect all indices of existing data to be up-sampled
    all_original_pool = set(obs.index)                                     # include all observations by default
    # iterate over all classes
    for ck in n_sim_per_class.index:
        ns = n_sim_per_class[ck].astype(int)               # get number of cells to change
        idx_pool = obs[obs[cls_label]==ck].index            # get available indices of original data
        
        if ns == 0:
            continue
        elif ns < 0:                                     # randomly remove cells from data without replacement
            sample_pool = pd.Series(idx_pool).sample(int(ns*-1), replace=False)
            all_original_pool -= set(sample_pool)
        else:                                           # randomly add cells to data with replacement
            sample_pool = pd.Series(idx_pool).sample(ns, replace=True)
            all_sim_pool.extend(sample_pool.tolist())

    if all_sim_pool is None or len(all_sim_pool)+len(all_original_pool) != cells_per_class.n_cells.sum():      # sanity check
        raise ValueError(f'Number of indices to be up-sampled does not match the given data.')
    # Perform actual up-sampling
    obs['n_idx'] = np.arange(obs.shape[0])
    # get obs of generated cells and mark as generated
    sim_obs = obs.loc[all_sim_pool].copy()
    sim_obs['generated'] = True
    # subset obs to original cells and mark those as original
    obs = obs.loc[list(all_original_pool)].copy()
    obs['generated'] = False
    logging.info('Sampling data from scvi posterior distribution')
    simulated = scvi_model.posterior_predictive_sample(indices=sim_obs.n_idx)
    # stack .X gene expression matrices of original and generated counts
    logging.info('Creating balanced dataset')
    X = sp.vstack((scvi_model.adata[obs.index].X, simulated.to_scipy_sparse()))
    obs = pd.concat((obs, sim_obs), axis=0)
    balanced_set = ad.AnnData(X=X, var=scvi_model.adata.var, obs=obs)
    balanced_set.uns['up_sampling_info'] = {
        'cls_label': cls_label, 'n_cells': n_cells, 'up_sample_rates': up_sample_rates
    }
    return balanced_set


def duplicate_from_scvi(scvi_model: scvi.model.SCVI):
    obs = scvi_model.adata.obs.copy()
    obs['generated'] = False
    logging.info('Sampling data from scvi posterior distribution')
    simulated = scvi_model.posterior_predictive_sample()
    sim_obs = obs.copy()
    sim_obs['generated'] = True
    sim_obs.set_index(sim_obs.index + '_sim', inplace=True)
    # stack .X gene expression matrices of original and generated counts
    logging.info('Creating duplicated dataset')
    X = sp.vstack((scvi_model.adata.X, simulated.to_scipy_sparse()))
    obs = pd.concat((obs, sim_obs), axis=0)
    upsampled_set = ad.AnnData(X=X, var=scvi_model.adata.var, obs=obs)
    upsampled_set.uns['up_sampling_info'] = {
        'mode': 'all'
    }
    return upsampled_set


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, pb_args={}, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._pb_args = pb_args
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, position=0, **self._pb_args) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def aggregate_cells(adata, n_cells: int, seed: int = 42, method: Literal['random'] | Literal['var'] = 'var') -> sp.csr_matrix:
        np.random.seed(seed)
        if method == 'random':
            indices = np.random.permutation(adata.n_obs)
        if method == 'var':
            X = adata.X
            row_means = X.mean(axis=1).A1  # Get row means (A1 to convert to 1D array)
            row_squared_diffs = (X.multiply(X)).mean(axis=1).A1 - row_means**2
            indices = pd.Series(row_squared_diffs).sort_values().index.tolist()
        else:
            raise ValueError(f'Method has to be one either random or var, got {method}')
        bins = np.array_split(indices, n_cells)
        X = np.vstack([adata[b].X.mean(axis=0) for b in bins])
        return sp.csr_matrix(X)


class CellCompressor:
    COMPR_KEY: str = 'compress_args'
    CLUSTER_KEY: str = 'cluster'
    compress_args: dict

    def _calculate_distance_matrix(self):
        sc.pp.pca(self.adata)
        sc.pp.neighbors(self.adata)

    def _calculate_variance_distance(self, adata: ad.AnnData) -> sp.csr_matrix:
        # compute variance over genes per cell
        if sp.issparse(adata.X):
            row_means = adata.X.mean(axis=1).A1  # Get row means (A1 to convert to 1D array)
            row_squared_diffs = (adata.X.multiply(adata.X)).mean(axis=1).A1 - row_means**2
        else:
            row_means = adata.X.mean(axis=1)
            row_squared_diffs = (adata.X.multiply(adata.X)).mean(axis=1) - row_means**2
        # compute pairwise distances of each cell
        pairwise_distances_squared = (row_squared_diffs[:, np.newaxis] - row_squared_diffs) ** 2
        # return variance-based distances
        return pairwise_distances_squared

    def _pre_process(self, methods: List[str]):
        import scanpy as sc
        if 'norm' in methods:
            logging.info('Normalizing adata')
            sc.pp.normalize_total(self.adata)
        if 'log1p' in methods:
            logging.info('Transforming adata with log1p')
            sc.pp.log1p(self.adata)
        if 'scale' in methods:
            logging.info('Scaling and centering adata')
            sc.pp.scale(self.adata)

    """
    Initialize the compression process for single-cell data.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.
    ct_key : str
        Key for cell type annotations in `adata.obs`.
    donor_key : str
        Key for donor annotations in `adata.obs`.
    method : Literal['random', 'var', 'cluster'], optional
        Method to use for compression. Options are 'random', 'var', 'distance', and 'pca'. Default is 'var'.
    subsample : bool, optional
        Whether to subsample the data instead of aggregating. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    max_worker : int, optional
        Maximum number of workers for parallel processing. Default is 20.
    copy : bool, optional
        Whether to copy the input `adata` or modify it in place. Default is True.
    """
    def __init__(self,
            adata: ad.AnnData, 
            groups: Iterable[str], 
            condition: str, 
            method: Literal['random', 'var', 'pca', 'distance'] = 'var',
            pre_process: List[str] | None = ['norm', 'log1p', 'scale'],
            subsample: bool = False, 
            agg_func: Literal['mean', 'sum', 'median'] = 'mean',
            seed: int = 42, 
            max_worker: int = 1,
            copy: bool = True
        ):
        
        self.adata: ad.AnnData = adata.copy() if copy else adata
        self.groups: Iterable[str] = groups
        self.condition: str = condition
        self.method: Literal['random', 'var', 'pca', 'distance'] = method
        self.subsample: bool = subsample
        self.agg_func = agg_func
        self.seed: int = seed
        self.max_worker: int = max_worker
        
        # Apply pre-processing steps to the given adata
        if pre_process is not None:
            self._pre_process(pre_process)
            self.pre_process: List[str] = pre_process
        # slots in adata.obsm/p
        self.pca_slot: str = 'X_pca'
        self.dist_slot: str = 'distances'
        # check if provided adata has PCA or distance slot and calculate if missing
        if self.method == 'pca' and not self.pca_slot in adata.obsm:
            logging.info('PCA not found in adata.obsm, calculating PCA')
            sc.pp.pca(self.adata)
        elif self.method == 'distance' and not self.dist_slot in adata.obsp:
            logging.info('Distance matrix not found in adata.uns, calculating distance matrix')
            self._calculate_distance_matrix()
        elif not self.method in ['var', 'random']:
            raise ValueError(f'Method has to be one of "pca", "distance", "var", or "random", got {self.method}')

    def _cluster_variance_idx(self, adata: ad.AnnData, n_cells: int) -> pd.Series:
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform

        # get distances based on variance per cell
        X = self._calculate_variance_distance(adata)
        condensed_dist = squareform(X)
        # Compute linkage
        Z = sch.linkage(condensed_dist, method='ward')
        # Cut the tree to get exactly n_clusters
        clusters = sch.fcluster(Z, n_cells, criterion='maxclust')
        # Add to AnnData object
        return pd.Series(clusters, name=self.CLUSTER_KEY)

    def _cluster_collapse_idx(self, adata: ad.AnnData, n_cells: int) -> pd.Series:
        from sklearn.cluster import KMeans, AgglomerativeClustering

        # get matrix to cluster on according to method
        cluster_class = KMeans(n_clusters=n_cells, random_state=self.seed)
        if self.method == 'distance':
            X = adata.obsp[self.dist_slot]
        elif self.method == 'pca':
            X = adata.obsm[self.pca_slot]
        elif self.method == 'var':
            return self._cluster_variance_idx(adata, n_cells)
        else:
            raise ValueError(f'{self.method} cannot be used for clustering')

        clusters = cluster_class.fit_predict(X)
        # check if number of clusters is the same as n_cells and adjust if not
        if len(set(clusters)) != n_cells:
            logging.info(f'Number of clusters ({adata.obs["cluster"].nunique()}) is not equal to number of cells ({n_cells}), forcing clustering')

            # Apply Agglomerative Clustering to split the clusters further into N groups
            agglom = AgglomerativeClustering(n_clusters=n_cells)
            agglom_labels = agglom.fit_predict(X.toarray())

            # Use the agglomerative clustering labels to ensure exactly N clusters
            clusters = agglom_labels
        return pd.Series(clusters, name=self.CLUSTER_KEY)

    def _collapse_cells_idx(self, adata: ad.AnnData, n_cells: int) -> pd.Series:
        np.random.seed(self.seed)
        if self.method != 'random':
            # cluster cells based on given method
            return self._cluster_collapse_idx(adata, n_cells)
        else:
            # randomly select cells
            indices = np.random.permutation(adata.n_obs)

        bins = np.array_split(indices, n_cells)                             # group indices into n_cells bins
        clusters = []                                                       # collect cluster labels
        for i, b in enumerate(bins):                                        # iterate over each bin and collect label
            clusters.extend(np.repeat(i, len(b)))
        cluster_df = pd.DataFrame({'idx': indices, 'cluster': clusters})    # convert information to map to indices
        cluster_labels = cluster_df.sort_values('idx')['cluster'].rename(self.CLUSTER_KEY)        # order cluster labels back to original adata indices
        return cluster_labels.reset_index(drop=True)

    def filter_patients(self, min_cells: int = 50, min_conditions: int = 50) -> pd.DataFrame:
        from functools import reduce
        vp = self.adata.obs.groupby(self.groups, observed=True)[self.condition].value_counts()
        vp = vp[vp >= min_cells].reset_index()
        vct = vp.groupby(self.groups, observed=True)[self.condition].nunique() >= min_conditions
        vct = vct[vct].reset_index()
        vp = reduce(
            lambda df, group: df[df[group].isin(vct[group])],
            self.groups,
            vp
        )
        if vp.shape[0] == 0:
            return vp
        classes_per_condition = vp.groupby(self.groups, observed=True)[self.condition].apply(lambda x: set(x))
        shared_conditions = set.intersection(*classes_per_condition)
        vp = vp[vp[self.condition].isin(shared_conditions)]
        logging.info(f'Found {len(shared_conditions)} shared conditions (out of {self.adata.obs[self.condition].nunique()}) between {vct.shape[0]} groups (out of {self.adata.obs.groupby(self.groups, observed=True).size().shape[0]})')
        return vp

    def _get_group_data(self, row) -> ad.AnnData:
        mask = pd.DataFrame(self.adata.obs[self.condition] == row[self.condition])
        for group in self.groups:
            mask[group] = (self.adata.obs[group] == row[group])
        mask = mask.sum(axis=1) == len(self.groups)+1
        return self.adata[mask]
    
    def __call__(
            self, 
            min_cells: int, 
            min_conditions: int, 
            n_samples: int | None = None, 
            n_ctrl_samples: int | None = None,
            ctrl_key: str = 'control',
            raw: bool = False
        ) -> ad.AnnData | None:
        p_info = self.filter_patients(min_cells, min_conditions)
        if p_info.shape[0] == 0:
            logging.info('No shared conditions found')
            return None
        # check n_samples and n_ctrl_samples, assign default value if none
        if n_samples is None:
            n_samples = p_info['count'].min()
        if n_ctrl_samples is None:
            n_ctrl_samples = n_samples
        logging.info(f'Compressing dataset to {n_samples} cells per condition and {n_ctrl_samples} for control groups')
        # Assign number of samples to draw for each condition based on ctrl_key
        p_info['n_samples'] = n_samples
        p_info.loc[p_info[self.condition]==ctrl_key, 'n_samples'] = n_ctrl_samples
        
        def process_row(row_idx: int) -> pd.DataFrame:
            row = p_info.iloc[row_idx]
            tmp = self._get_group_data(row)
            N = int(row['n_samples'])
            if N < tmp.n_obs:
                cluster_labels = self._collapse_cells_idx(tmp, n_cells=N)           # collapse cells of donor to minimum required number of cells
            else:
                cluster_labels = pd.Series(np.arange(tmp.n_obs), name=self.CLUSTER_KEY)      # treat all datapoints as individual clusters
            row_key = pd.DataFrame(row[[*self.groups, self.condition]]).agg(';'.join, axis=0).values[0]        # add row-specific information to cluster label for later aggregation
            cluster_labels = row_key + ';' + cluster_labels.astype(str)
            return pd.DataFrame(cluster_labels.set_axis(tmp.obs.index))

        # Perform calculation iterative or parallel
        if self.max_worker < 2:
            idx_cluster_map = []
            for i in tqdm(np.arange(p_info.shape[0]), desc='Compressing patient cells', unit='condition'):
                idx_cluster_map.append(process_row(i))
        else:
            idx_cluster_map = ProgressParallel(total=p_info.shape[0], n_jobs=self.max_worker)(
                delayed(process_row)(i) for i in np.arange(p_info.shape[0])
            )
        # Concatenate index to cluster map
        idx_cluster_map = pd.concat(idx_cluster_map, axis=0).reset_index(names='index')
        # Subset self.adata to shared patient data
        self.adata._inplace_subset_obs(idx_cluster_map['index'])
        # Add cluster information to adata
        self.adata.obs = self.adata.obs.reset_index(names='index').merge(idx_cluster_map, on='index', how='left')
        # Compress data for each cluster
        if self.subsample:
            logging.info('Subsampling cells from each condition')
            # TODO: sample different number of cells for control group
            # instead of aggreagting the data, sample fixed number of cells from each calculated cluster
            subsample_idx = self.adata.obs.groupby(self.CLUSTER_KEY, observed=True).apply(lambda x: x.sample(n_samples).reset_index()).reset_index(drop=True)['index']
            _adata = self.adata[subsample_idx].copy()
        else:
            logging.info('Aggregating cells from each condition')
            # take a represenative observation annotation from every cluster to add to future adata
            agg_obs = self.adata.obs.groupby(self.CLUSTER_KEY, observed=True).apply(
                    lambda x: x.head(1).reset_index()
                ).reset_index(drop=True).set_index('index')
            # aggregate each calculated cluster
            _adata = sc.get.aggregate(self.adata, by=self.CLUSTER_KEY, func=self.agg_func)
            _adata.X = sp.csr_matrix(_adata.layers[self.agg_func])
            del _adata.layers[self.agg_func]
            _adata.obs = agg_obs
            if raw:
                logging.info('Rounding compressed adata.X')
                _adata.X = np.round(_adata.X) 
        # save compression args to new adata and class
        _adata.uns[self.COMPR_KEY] = {
            'groups': self.groups, 'condition': self.condition,
            'min_cells': min_cells, 'min_patients': min_conditions,
            'method': self.method, 'subsample': self.subsample,
            'agg_func': self.agg_func, 'is_raw': raw
        }
        self.compress_args = _adata.uns[self.COMPR_KEY]
        logging.info(f'Resulting dataset dimensions: {_adata.shape}')
        return _adata
    
    def get_dir(self):
        if self.compress_args is not None:
            ca = self.compress_args
            return os.path.join(*[f'{k}:{v}' for k,v in ca.items() if not k.endswith('key')])
        else:
            raise ValueError("Compression arguments not found in adata.uns")


def get_zero_inflation_rate(adata:ad.AnnData) -> float:
    """
    Compute the zero-inflation rate of the data.
    This function computes the zero-inflation rate of the data by counting the number of zeros in the data matrix.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    Returns:
    --------
    float
        The zero-inflation rate of the data.
    """
    if sp.issparse(adata.X):
        n = adata.X.shape[0] * adata.X.shape[1]
        return 1 - adata.X.nnz / n
    else:
        return (adata.X == 0).sum() / adata.X.size

def upsample(original_adata, m, adj_n_genes: bool = False):
    if adj_n_genes:
        sX = original_adata.X / np.array(original_adata.obs.ngenes).reshape(-1, 1)       # Adjust gene counts for number of genes in cell
    else:
        sX = original_adata.X
    lambda_noise = sX.mean(axis=0)    # Compute the mean of the gene expression counts per gene

    mc = original_adata.obs.sample(n=m, replace=False).index      # draw missing cells from population
    md = original_adata[mc].copy()
    poisson_noise = np.random.poisson(lam=lambda_noise, size=md.shape)           # create background noise
    
    # Add background noise to original data
    md.X = sp.csr_matrix(poisson_noise)
    md.obs['upsampled'] = True
    return md

def draw_samples(
        adata: ad.AnnData, 
        label: str = 'exact_perturbation',
        n_samples: int = 2000, 
        n_control: int = 4000, 
        seed: int = 42, 
        max_sampling_rate: float = 0.25,
        verbose: bool = False,
        subset: bool = True,
    ) -> ad.AnnData:
    """
    Draw samples from the dataset with optional zero-inflation noise.
    This function samples cells from an AnnData object based on a specified label. 
    It can draw a specified number of samples for each condition, adding zero-inflation noise 
    to the data if the number of required samples exceeds the available cells.
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    label : str, optional (default: 'exact_perturbation')
        The column name in `adata.obs` to group by for sampling.
    n_samples : int, optional (default: 2000)
        Number of samples to draw for each non-control condition.
    n_control : int, optional (default: 4000)
        Number of samples to draw for each control condition.
    Returns:
    --------
    AnnData
        A new AnnData object containing the sampled cells.
    """
    import tqdm
    import warnings
    import random
    import scipy.sparse as sp
    random.seed(seed)

    groups = adata.obs[label].unique()
    X = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for cond in tqdm.tqdm(groups, desc=f"Processing condition", unit='condition'):
            N = n_samples
            # define subset of class
            c = adata[adata.obs[label]==cond]
            if cond.endswith('control'):
                N = n_control
        
            # draw without replacement from existing cells
            if N <= c.n_obs:
                if subset:
                    # TODO: ensure that we keep diversity of data intact
                    target_cells = c.obs.sample(n=N, replace=False).index
                    sd = c[target_cells].copy()
                else:
                    sd = c.copy()
                sd.obs['upsampled'] = False
                X.append(sd)
            # draw with replacement from existing cells
            else:
                c.obs['upsampled'] = False
                X.append(c)                # select all available cells and add .X to data
                m = N - c.n_obs                    # determine missing cells

                # if m is much larger than c.n_obs
                max_samples = max_sampling_rate*c.n_obs
                if m <= max_samples:
                    md = upsample(c, m)
                    X.append(md)
                else:
                    # determine how often we need to sample
                    rounds = int(np.round(m / (max_samples)))
                    m_round = int(np.round(max_samples))
                    d = c
                    if verbose:
                        logging.info(f'Sampling {rounds} rounds')
                    for r in range(rounds):
                        us = upsample(d, m_round)       # draw samples
                        d = ad.concat([d, us])  
                        d.obs_names_make_unique()
                        if verbose:
                            logging.info(f'Round: {r}, cells: {d.shape[0]}')
                    X.append(d)
        # collapse data list to a uniform dataset
        print(f'Collapsing list to uniform dataset')
        var = adata.var.copy()
        obs = pd.concat([d.obs for d in X])
        X = sp.vstack([d.X for d in X])
        X = sp.csr_matrix(X)
        return ad.AnnData(X=X, var=var, obs=obs)
