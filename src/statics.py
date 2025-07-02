from typing import NamedTuple


class _DATA_SHEET_NT(NamedTuple):
    URL: str = 'download link'
    INDEX: str = 'index'
    P_INDEX: str = 'publication index'
    D_INDEX: str = 'dataset index'
    CANCER: str = 'cancer'
    PERTURBATION_TYPE: str = 'perturbation'
    CELL_TYPE: str = 'cell type'
    BYTES: str = 'bytes'
    MEM: str = 'memory'
    MAX_MEM: str = 'max_memory'
    PARTITION: str = 'partition'

class _OBS_KEYS_NT(NamedTuple):
    DATASET_KEY: str = 'dataset'
    PERTURBATION_TYPE_KEY: str = 'perturbation'
    CELL_TYPE_KEY: str = 'celltype_broad'

class _SETTINGS(NamedTuple):
    MT_PERCENT_CANCER: int = 20
    MT_PERCENT_NORMAL: int = 12

def _perturbation_cols():
    return [
        'perturbation',
        'gene',
        'perturbation_1',
        'target',
        'gene_target'
    ]

def _ctrl_keys():
    return [
        'control',
        'ctrl',
        'non-targeting',
        'nt'
    ]

def _booleans():
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

def _strings():
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

def _ints():
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

def _floats():
    return [
        'ctrl_dev'
    ]

DATA_SHEET_KEYS = _DATA_SHEET_NT()
OBS_KEYS = _OBS_KEYS_NT()
SETTINGS = _SETTINGS()
STRINGS = _strings()
BOOLEANS = _booleans()
INTS = _ints()
FLOATS = _floats()
P_COLS = _perturbation_cols()
CTRL_KEYS = _ctrl_keys()
