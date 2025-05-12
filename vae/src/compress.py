import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp
import logging
from tqdm import tqdm
from typing import Iterable, Tuple, Literal, List
from joblib import Parallel, delayed
import os


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


class CellCompressor:
    COMPR_KEY: str = 'compress_args'
    compress_args: dict

    def calculate_distance_matrix(self):
        import scanpy as sc
        sc.pp.pca(self.adata)
        sc.pp.neighbors(self.adata)

    def _pp(self, methods: List[str]):
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


    def __init__(self, adata: ad.AnnData, ct_key: str, donor_key: str, pre_process: List[str] | None, method: Literal['random', 'var', 'subsample', 'distance'] = 'var', seed: int = 42, max_worker: int = 20):
        self.adata: ad.AnnData = adata
        self.ct_key: str = ct_key
        self.donor_key: str = donor_key
        self.method: Literal['random', 'var', 'subsample', 'distance'] = method
        self.seed: int = seed
        self.max_worker: int = max_worker
        if pre_process is not None:
            self._pp(pre_process)
        if self.method == 'distance' and not 'distances' in adata.obsp:
            logging.info('Distance matrix not found in adata.uns, calculating distance matrix')
            self.calculate_distance_matrix()

    def _distance_collapse(self, adata, n_cells: int) -> sp.csr_matrix:
        from sklearn.cluster import KMeans

        distances = adata.obsp['distances']
        kmeans = KMeans(n_clusters=n_cells, random_state=self.seed)
        adata.obs['cluster'] = kmeans.fit_predict(distances)
        X = sc.get.aggregate(adata, by='cluster', func='mean').layers['mean']
        return sp.csr_matrix(X)
            
    def collapse_cells(self, adata, n_cells: int) -> sp.csr_matrix:
        np.random.seed(self.seed)
        if self.method == 'subsample':
            indices = adata.obs.reset_index().sample(n_cells).index
            return sp.csr_matrix(adata[indices].X)
        elif self.method == 'distance':
            return self._distance_collapse(adata, n_cells)
        elif self.method == 'var':
            X = adata.X
            row_means = X.mean(axis=1).A1  # Get row means (A1 to convert to 1D array)
            row_squared_diffs = (X.multiply(X)).mean(axis=1).A1 - row_means**2
            indices = pd.Series(row_squared_diffs).sort_values().index.tolist()
        elif self.method == 'random':
            indices = np.random.permutation(adata.n_obs)
        else:
            raise ValueError(f'Method has to be one either random, var, or subsample, got {self.method}')
        bins = np.array_split(indices, n_cells)
        X = np.vstack([adata[b].X.mean(axis=0) for b in bins])
        return sp.csr_matrix(X)

    def filter_patients(self, min_cells: int = 50, min_patients: int = 50) -> pd.DataFrame:
        vp = self.adata.obs.groupby(self.ct_key)[self.donor_key].value_counts()
        vp = vp[vp >= min_cells].reset_index()
        vct = vp.groupby(self.ct_key)[self.donor_key].nunique() >= min_patients
        vct = vct[vct].index
        vp = vp[vp[self.ct_key].isin(vct)]
        if vp.shape[0] == 0:
            return vp
        shared_patients = set.intersection(*vp.groupby(self.ct_key)[self.donor_key].apply(lambda x: set(x)))
        vp = vp[vp[self.donor_key].isin(shared_patients)]
        logging.info(f'Found {len(shared_patients)} shared patients (out of {self.adata.obs[self.donor_key].nunique()}) between {vct.shape[0]} cell types (out of {self.adata.obs[self.ct_key].nunique()})')
        return vp

    def dynamic_parameters(self, mc_space: Iterable[int], mp_space: Iterable[int], return_results: bool = True) -> dict[str, int] | Tuple[dict[str, int], pd.DataFrame]:
        logging.info('Determining optimal parameters for run')
        results = []
        n_cts = self.adata.obs[self.ct_key].nunique()
        n_pts = self.adata.obs[self.donor_key].nunique()
        
        for mc in mc_space:
            for mp in mp_space:
                p_info = self.filter_patients(mc, mp)
                pt = p_info[self.donor_key].nunique()
                ct = p_info[self.ct_key].nunique()
                ratio = ((pt / n_pts) + (ct / n_cts)) / 2
                results.append((mc, mp, ct, pt, ratio))
        # create result data frame and determine best parameters
        results_df = pd.DataFrame(results, columns=['min_cells', 'min_patients', 'n_cell_types', 'n_patients', 'ratio'])
        best_params = results_df.sort_values('ratio', ascending=False).iloc[0]
        # create output dict
        r = {
            'min_cells': int(best_params['min_cells']),
            'min_patients': int(best_params['min_patients'])
        }
        if return_results:
            return r, results_df
        return r


    def compress_dataset(self, min_cells: int | None = 50, min_patients: int | None = 50, raw: bool = False) -> ad.AnnData:
        if min_cells is None or min_patients is None:
            space = np.arange(10, 101, 10)
            params: dict[str, int] = self.dynamic_parameters(space, space, return_results=False)
            min_cells = params.get('min_cells', 10)
            min_patients = params.get('min_patients', 10)
        p_info = self.filter_patients(min_cells, min_patients)
        n_samples = p_info['count'].min()
        X_l = []
        obs_l = []
        
        def process_row(row_idx):
            row = p_info.iloc[row_idx]
            tmp = self.adata[(self.adata.obs[self.ct_key] == row[self.ct_key]) & (self.adata.obs[self.donor_key] == row[self.donor_key])]
            if n_samples < tmp.n_obs:
                X = self.collapse_cells(tmp, n_cells=n_samples)  # collapse cells of donor to minimum required number of cells
            else:
                X = tmp.X
            labels = row[self.ct_key] + ';' + row[self.donor_key] + ';mean_bin_' + np.arange(n_samples).astype(str)  # create labels for each mean bin
            obs = tmp.obs.iloc[:n_samples].copy().set_index(labels)  # copy obs of patient and add mean index labels
            return X, obs

        results = ProgressParallel(total=p_info.shape[0], n_jobs=self.max_worker)(
            delayed(process_row)(i) for i in np.arange(p_info.shape[0])
        )
        # collect results
        for X, obs in results:
            X_l.append(X)
            obs_l.append(obs)
        # stack results to one object
        X = sp.vstack(X_l)
        obs = pd.concat(obs_l, axis=0)
        obs['bin'] = obs.index.str.split('_').str[-1]
        _adata = ad.AnnData(X=X, obs=obs, var=self.adata.var)
        _adata.uns[self.COMPR_KEY] = {
            'ct_key': self.ct_key, 'donor_key': self.donor_key,
            'min_cells': min_cells, 'min_patients': min_patients,
            'is_raw': raw
        }
        if raw:
            logging.info('Rounding compressed adata.X')
            _adata.X = np.round(_adata.X)    
        self.compress_args = _adata.uns[self.COMPR_KEY]
        return _adata
    
    def get_dir(self):
        if self.compress_args is not None:
            ca = self.compress_args
            return os.path.join(*[f'{k}:{v}' for k,v in ca.items() if not k.endswith('key')])
        else:
            raise ValueError("Compression arguments not found in adata.uns")
