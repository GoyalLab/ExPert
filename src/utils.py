import requests
import scanpy as sc
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np
import os
import anndata as ad
import scipy.sparse as sp
from memory_profiler import memory_usage
import functools
import hashlib
import yaml


def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def hash(s: str, d: int = 4):
    return hashlib.shake_128(s.encode('utf-8')).hexdigest(d)


def get_param_hash(p: dict, d: int = 8):
    blacklist = {'log', 'datasets'}
    ps = p.copy()
    # sort dict
    ps = dict(sorted(p.items()))
    s = ';'.join([f'{k}:{v}' for k,v in ps.items() if k not in blacklist])
    s += 'datasets:' + ','.join(sorted(ps.get('datasets')))
    return hash(s, d)


def save_config(c: dict, o: str):
    d = os.path.dirname(o)
    os.makedirs(d, exist_ok=True)
    with open(o, 'w') as f:
        yaml.dump(c, f, default_flow_style=False)


def log_decorator(func):
    def wrapper(*args, **kwargs):
        logging.info(f'Computing {func.__name__}')
        result = func(*args, **kwargs)
        logging.info(f'Finished {func.__name__}')
        return result
    return wrapper


def log_memory_usage(mem_dir="./mem"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mem_file = os.path.join(mem_dir, f'{func.__name__}.txt')
            with open(mem_file, "a") as log_file:
                log_file.write(f"\nMemory usage for {func.__name__}:\n")
                
                def track_memory():
                    mem_usage = memory_usage((func, args, kwargs), interval=0.1)
                    return mem_usage

                # Run memory tracking and execute the function
                mem_usage = track_memory()

                # Log memory usage per line
                for line_num, mem in enumerate(mem_usage):
                    log_file.write(f"Line {line_num + 1}: {mem} MiB\n")
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator


def convert_size(size_bytes: int) -> str:
    if size_bytes == -1:
        return "Unknown size"
    elif size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int((len(str(size_bytes)) - 1) // 3)
    p = 1024 ** i
    size = round(size_bytes / p, 2)
    return f"{size} {size_name[i]}"


def get_file_size(url, verbose=False):
    try:
        response = requests.head(url, allow_redirects=True)
        file_size = response.headers.get('Content-Length')
        if verbose:
            print(f"Checking file size of {url}: {file_size}")
        if file_size is not None:
            return int(file_size)
        else:
            return -1
    except requests.RequestException as e:
        return f"Error: {e}"


def download_file(url, output_path, cache=True) -> str:
    if not os.path.exists(output_path) or not cache:
        try:
            # Send a GET request to download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for errors

            # Get the total file size from the response headers (in bytes)
            total_size_in_bytes = int(response.headers.get('content-length', 0))

            # Initialize the progress bar with the total file size
            progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc=output_path)

            # Write the file to the output path in chunks
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update the progress bar
            progress_bar.close()

            return f"Successfully downloaded file: {output_path}"
        except requests.exceptions.RequestException as e:
            return f"Error downloading file: {e}"


def get_dataset(url, dataset_name, output_path, cache=True):
    print(f"Reading dataset {dataset_name}")
    data = sc.read(output_path, backup_url=url, cache=cache)
    data.uns["dataset_name"] = dataset_name
    return data


def make_obs_names_unique(df, agg='mean'):
    # check number of datasets each hvg is detected
    df['hv_in'] = df['highly_variable'].sum(axis=1)
    numeric_cols = df.dtypes[df.dtypes != 'object'].index
    obj_cols = df.dtypes[df.dtypes == 'object'].index
    # collapse all numeric cols with an aggregation method (e.g. mean)
    if agg == 'mean':
        numerics = df[numeric_cols].groupby(level=0, axis=1).mean()
    elif agg == 'median':
        numerics = df[numeric_cols].groupby(level=0, axis=1).median()
    else:
        raise ValueError(f"Unknown aggregation {agg}, choose between 'mean', 'median'")
    # choose the first non-nan value for duplicated object columns
    meta = df[obj_cols].groupby(df[obj_cols].columns, axis=1).apply(lambda x: x.bfill(axis=1).iloc[:, 0])
    return pd.concat([meta, numerics], axis=1)

def create_config_file(base_conf, datasets, config_path='config/config.yaml'):
    with open(config_path, 'w') as f:
        # Add parameters
        f.write('# Base parameters for ExPert datasets merge\n')
        for k,v in base_conf.items():
            f.write(f"{k}: {v}\n")
        # Add datasets
        f.write(f'\n# Datasets to merge (total of {datasets.shape[0]} sets)\n')
        f.write('datasets:\n')
        for i, row in datasets[['index', 'download link']].iterrows():
            f.write(f" \"{row['index']}\": \"{row['download link']}\"\n")


def ref_per_method():
    return {
        'skip': {
            'cpu': {'mem': 2.25, 'time': 5}
        },
        'scanorama': {
            'cpu': {'mem': 3.64, 'time': 5}
        },
        'scANVI': {
            'cpu': {'mem': 2.86, 'time': 180},
            'gpu': {'mem': 2.86, 'time': 5}
        }
    }
            

def estimate_time(ds, n, method='skip', min_h=1, device='cpu'):
    ref_min = ref_per_method()[method][device]['time']
    x = ds.sort_values('bytes').iloc[:4,].bytes.sum()
    y = ds.sort_values('bytes').iloc[:n,].bytes.sum()
    h = int(y/x * ref_min / 60)
    if h < min_h:
        h = min_h
    return int(h)


# estimate RAM based on test set
def estimate_RAM(ds, n, method='skip', round=True, factor=10, min_mem=20, device='cpu'):
    ref_ram = ref_per_method()[method][device]['mem']
    x = ds.sort_values('bytes').iloc[:4,].bytes.sum()
    y = ds.sort_values('bytes').iloc[:n,].bytes.sum()
    mem = y/x * ref_ram
    if round:
        mem = np.round(mem / factor)*factor
    if mem < min_mem:
        mem = min_mem
    return int(mem)


# get appropriate partition
def get_partition(ram=100, use_gpu=False):
    if use_gpu:
        return 'genomics-gpu'
    if ram > 240:
        return 'genomics-himem'
    else:
        return 'genomics'
    

def requires_gpu(m):
    return m=='scANVI'
    

# create slurm script
def create_sbatch_script(base_conf, ds, conda_env='harmonize', script='main_quest.sh', config_path='config/config.yaml', time_str=None):
    # check for GPU requirement
    method = base_conf['correction_method'].strip("''")
    gpu = requires_gpu(method)
    device = 'gpu' if gpu else 'cpu'
    # estimate RAM usage
    ram = estimate_RAM(ds, n=ds.shape[0], method=method, device=device)
    # estimate time
    if time_str is None:
        time_str = f'{estimate_time(ds, n=ds.shape[0], method=method, device=device)}:00:00'
    # check partition
    partition = get_partition(ram, use_gpu=gpu)
    # define logs (create log dir for each run)
    log_dir = 'logs/quest/%j'
    log_file = f'{log_dir}/snakemake.log'
    # define sbatch parameters
    sbatch_params = {
        'account': 'b1042',
        'partition': partition,
        'job-name': 'ExPert',
        'nodes': 1,
        'ntasks-per-node': ds.shape[0],
        'mem': f'{ram}GB',
        'time': time_str,
        'output': log_file,
        'verbose': True
    }
    # write sbatch file
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n')
        # add sbatch parameters
        for k,v in sbatch_params.items():
            if isinstance(v, bool):
                # flags
                p = f'#SBATCH --{k}\n'
            else:
                p = f'#SBATCH --{k} {v}\n'
            f.write(p)
        # add run parameters
        f.write('# Define log directory\n')
        f.write('LOG="logs/quest/${SLURM_JOB_ID}"\n\n')
        # load modules
        f.write('echo "Setting up pipeline..."\n')
        f.write('module purge\n')
        f.write('module load anaconda3\n')
        f.write('source ~/.bashrc\n')
        f.write(f'conda activate {conda_env}\n')
        # execute actual pipeline
        f.write('echo "Starting pipeline..."\n')
        f.write(f'snakemake --cores {ds.shape[0]} --verbose --configfile "{config_path}" --config log="$LOG"\n')
        f.write('echo "Finished pipeline"\n')
        f.write('conda deactivate\n')


def _is_sparse(f):
    # Check if the matrix is sparse or not
    adata = sc.read_h5ad(f, backed='r')
    is_sparse = isinstance(adata.X, (ad._core.sparse_dataset.CSCDataset, ad._core.sparse_dataset.CSRDataset))
    # Close the file to free resources if needed
    adata.file.close()
    return is_sparse

def read_ad(f):
    kws = {}
    if not _is_sparse(f):
        logging.info('Converting dense .X matrix to sparse.csc_matrix')
        kws = {'as_sparse':['X'], 'as_sparse_fmt':sp.csc_matrix}
    return sc.read_h5ad(f, **kws)
