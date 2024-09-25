import requests
import scanpy as sc
from tqdm import tqdm
import os
import logging
import pandas as pd


def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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


def download_file(url, output_path):
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

        print(f"Downloaded successfully: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def get_dataset(url, dataset_name, output_path, cache=True):
    if os.path.exists(output_path) and cache:
        print(f"Using cached file {output_path}")
    else:
        print(f"Downloading {url}")
        download_file(url, output_path)
    print(f"Reading dataset {dataset_name}")
    data = sc.read_h5ad(output_path)
    data.uns["dataset_name"] = dataset_name
    return data


def make_obs_names_unique(df, agg='mean'):
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
