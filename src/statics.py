from typing import NamedTuple


class _DATA_SHEET_NT(NamedTuple):
    URL: str = 'download link'
    INDEX: str = 'index'
    P_INDEX: str = 'publication index'
    D_INDEX: str = 'dataset index'
    CANCER: str = 'cancer'
    BYTES: str = 'bytes'
    MEM: str = 'memory'

def booleans():
    return [
        'qc', 
        'norm', 
        'log_norm', 
        'scale', 
        'hvg', 
        'subset_hvg', 
        'single_perturbations_only',
        'mixscale_filter', 
        'zero_padding', 
        'cache', 
        'plot', 
        'do_umap', 
        'do_tsne'
    ]

def strings():
    return [
        'datasheet', 
        'output_dir', 
        'data_dir', 
        'cache_dir', 
        'log_dir', 
        'perturbation_col', 
        'ctrl_key', 
        'correction_method', 
        'merge_method'
    ]

def ints():
    return [
        'n_hvg',
        'n_ctrl',
        'min_deg',
        'min_cells_per_perturbation',
        'cores',
        'seed',
        'min_mem',
        'max_mem'
    ]

def floats():
    return [
        'ctrl_dev'
    ]

DATA_SHEET_KEYS = _DATA_SHEET_NT()
STRINGS = strings()
BOOLEANS = booleans()
INTS = ints()
FLOATS = floats()
