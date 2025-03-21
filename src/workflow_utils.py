from snakemake.io import load_configfile
import hashlib
import os
import yaml
import pandas as pd
from src.statics import DATA_SHEET_KEYS


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
        d[DATA_SHEET_KEYS.INDEX] = d[DATA_SHEET_KEYS.P_INDEX].astype(str) + '_' + d[DATA_SHEET_KEYS.D_INDEX].astype(str)
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

def check_method(m, conf):
    import logging
    import warnings

    methods = correction_methods()
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
