from snakemake.io import load_configfile
import hashlib
import os
import yaml
import numpy as np
import pandas as pd
from src.statics import DATA_SHEET_KEYS, STRINGS, BOOLEANS, INTS, FLOATS
from typing import List


def load_configs(wf, default_config="config/defaults.yaml"):
    # Load default config
    config = load_configfile(default_config)
    
    # Update with any additional config files (from --configfile)
    for cf in wf.overwrite_configfiles:
        update_config = load_configfile(cf)
        config.update(update_config)

    # Update with command-line options (from --config)
    config.update(wf.overwrite_config)
    # TODO: Check config for invalid arguments
    # check_inputs(config)
    return config

def hash(s: str, d: int = 4):
    return hashlib.shake_128(s.encode('utf-8')).hexdigest(d)


def get_param_hash(p: dict, dataset_indices: list[str], d: int = 8):
    blacklist = {'datasets', 'log_dir', 'data_dir', 'cache_dir'}
    ps = p.copy()
    # sort dict
    ps = dict(sorted(p.items()))
    s = ';'.join([f'{k}:{v}' for k,v in ps.items() if k not in blacklist])
    # add all dataset indices
    s += 'datasets:' + ','.join(sorted(dataset_indices))
    return hash(s, d)

def save_config(c: dict, o: str):
    d = os.path.dirname(o)
    os.makedirs(d, exist_ok=True)
    with open(o, 'w') as f:
        yaml.dump(c, f, default_flow_style=False)

def read_data_sheet(p: str):
    d = pd.read_csv(p)
    if DATA_SHEET_KEYS.INDEX not in d.columns:
        d[DATA_SHEET_KEYS.INDEX] = d[DATA_SHEET_KEYS.P_INDEX].astype(str) + d[DATA_SHEET_KEYS.D_INDEX].fillna('').apply(lambda x: f"_{x}" if x else '')
    # set as index
    d.set_index(DATA_SHEET_KEYS.INDEX, inplace=True)
    if DATA_SHEET_KEYS.CANCER in d.columns:
        # convert y/n to boolean
        d[DATA_SHEET_KEYS.CANCER] = d[DATA_SHEET_KEYS.CANCER].map({'y': True, 'n': False})
    return d


def correction_methods():
    return ['scANVI', 'scanorama', 'harmonypy', 'skip']

def requires_raw_data():
    return ['scANVI']

def requires_processed_data():
    return ['scanorama', 'harmonypy']

def requires_gpu():
    return ['scANVI']

def check_method(conf):
    import logging
    import warnings

    methods = correction_methods()
    m = conf['correction_method']
    if m not in methods:
        raise ValueError(f'Method must be {methods}; got {m}')
    if m in requires_raw_data():
        logging.info(f'Method {m} requires raw data, setting preprocess to exclude normalization and log1p')
        conf['norm'] = False
        conf['log_norm'] = False
        conf['scale'] = False
    if m in requires_processed_data():
        logging.info(f'Method {m} requires preprocessed data, setting preprocess to include normalization and log1p')
        conf['norm'] = True
        conf['log_norm'] = True
    if m in requires_gpu():
        import torch
        if not torch.cuda.is_available():
            # raise SystemError('scANVI requires a GPU to effectively harmonize.')
            warnings.warn('scANVI requires a GPU to effectively harmonize. Running on CPU will significantly increase the runtime.')

def check_types(conf):
    type_checks = {
        str: STRINGS,
        bool: BOOLEANS,
        int: INTS,
        float: FLOATS
    }
    # Check every param for correct type
    for param, value in conf.items():
        for expected_type, keys in type_checks.items():
            if param in keys:
                assert isinstance(value, expected_type), f"Invalid value for {expected_type.__name__} parameter '{param}': {value}"

def determine_resources(config: dict, data_sheet: pd.DataFrame):
    # Update configs resource requirements based on the size of the input file
    min_mem = config['min_mem']
    max_mem = config['max_mem']
    # Only overwrite configs if memory is not given
    if DATA_SHEET_KEYS.MEM not in data_sheet.columns:
        # Fill memory column based on bytes given in metadata
        if DATA_SHEET_KEYS.BYTES in data_sheet.columns:
            mem_per_ds = np.maximum((data_sheet[DATA_SHEET_KEYS.BYTES] / data_sheet[DATA_SHEET_KEYS.BYTES].max() * max_mem).astype(int), min_mem)
            data_sheet[DATA_SHEET_KEYS.MEM] = mem_per_ds.astype(str) + 'GB'
        else:
            # Fall back to minimum memory settings
            data_sheet[DATA_SHEET_KEYS.MEM] = min_mem
    else:
        # Check if memory is given in correct format
        assert data_sheet[DATA_SHEET_KEYS.MEM].str.endswith('GB').all()

def check_config(conf: dict):
    # validate config types
    check_types(conf)
    # validate correction method
    check_method(conf)
    # add individual resource requirements
