from snakemake.io import load_configfile
import hashlib
import os
import yaml
import numpy as np
import pandas as pd
from src.statics import DATA_SHEET_KEYS, STRINGS, BOOLEANS, INTS, FLOATS
import logging


def recursive_update(d1, d2):
    for k, v in d2.items():
        if (
            k in d1 and isinstance(d1[k], dict) and isinstance(v, dict)
        ):
            recursive_update(d1[k], v)
        else:
            d1[k] = v

def load_configs(wf, default_config="config/defaults.yaml"):
    # Load default config
    config = load_configfile(default_config)
    
    # Update with any additional config files (from --configfile)
    for cf in wf.overwrite_configfiles:
        update_config = load_configfile(cf)
        recursive_update(config, update_config)

    # Update with command-line options (from --config)
    recursive_update(config, wf.overwrite_config)
    # Check config parameters
    check_config(config)
    return config

def hash(s: str, d: int = 4):
    return hashlib.shake_128(s.encode('utf-8')).hexdigest(d)

def get_param_hash(p: dict, dataset_indices: list[str], d: int = 8):
    blacklist = {'log_dir', 'output_dir', 'cache_dir', 'plot_dir', 'resources'}
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

def get_default_resources(resource_config: dict, default_key: str = 'default') -> dict[str, str]:
    o = {
        'time': resource_config['time'][default_key],
        'mem': resource_config['memory'][default_key],
    }
    partition = resource_config['partitions'][default_key]
    if partition != '':
        o['partition'] = partition
    return o

def get_job_resources(resource_config: dict, job_name: str, partition_prio: str | None = None) -> dict[str, str]:
    # Check if there are resource specification for that job name
    if job_name not in resource_config['jobs']:
        logging.warning(f'Could not resource config for job {job_name}, falling back to defaults.')
        return get_default_resources()
    # Get resource configs for job from defaults.yaml with updates by user
    time: str = resource_config['jobs'][job_name]['time']
    mem: str = resource_config['jobs'][job_name]['mem']
    partition: str = resource_config['partitions']['default']
    # Get maximum memory cap of default partition
    max_default_mem: int = resource_config['memory']['max_mem']
    # Try to set to high memory partition if given
    if int(mem.rstrip('GB')) > max_default_mem:
        hi_mem_partition = resource_config['partitions']['high_mem']
        if hi_mem_partition != '':
            partition = hi_mem_partition
        # Else set memory to default partition limit
        else:
            mem = str(max_default_mem)+'GB'
    # Try to set partition to given prio if found in config, fall back to previous partition if not
    if partition_prio is not None:
        partition = resource_config['partitions'].get(partition_prio, partition)
    o = {'time': time, 'mem': mem}
    if partition != '':
        o['partition'] = partition
    return o


def determine_resources(config: dict, data_sheet: pd.DataFrame):
    # Update configs resource requirements based on the size of the input file
    resource_config: dict = config['resources']
    # Determine overall memory range
    mem_config: dict = resource_config['memory']
    min_mem = mem_config['min_mem']
    max_mem = mem_config['max_mem']
    # Determine partitions
    partition_config: dict = resource_config['partitions']
    default_partition: str = partition_config['default']
    gpu_partition: str = partition_config.get('gpu', None)
    # Check if there is a high memory allocation
    himem_alloc: str = partition_config.get('high_mem', default_partition)
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
    # Set max memory to 2x MEM or max_mem if that is less
    def calc_max_mem(mem_str):
        mem_val = int(mem_str.rstrip('GB'))
        doubled = 2 * mem_val
        return f"{min(doubled, max_mem)}GB"
    data_sheet[DATA_SHEET_KEYS.MAX_MEM] = data_sheet[DATA_SHEET_KEYS.MEM].apply(calc_max_mem)
    # Set partition for each dataset based on memory assigned to it
    data_sheet[DATA_SHEET_KEYS.PARTITION] = default_partition
    himem_mask = data_sheet[DATA_SHEET_KEYS.MEM].str.rstrip('GB').astype(int) > max_mem
    highest_mem = data_sheet[DATA_SHEET_KEYS.MEM].str.rstrip('GB').astype(int).max()
    if himem_mask.sum() > 0 and himem_alloc == default_partition:
        logging.warning(f'Some datasets allocate more memory than the set max. memory ({max_mem}) and no high-memory partition is given. Defaulting to partition {default_partition} and max. allocation of {highest_mem}GB.')
    data_sheet.loc[himem_mask,DATA_SHEET_KEYS.PARTITION] = himem_alloc

def read_data_sheet(config: dict):
    d = pd.read_csv(str(config['datasheet']))
    if DATA_SHEET_KEYS.INDEX not in d.columns:
        d[DATA_SHEET_KEYS.INDEX] = d[DATA_SHEET_KEYS.P_INDEX].astype(str) + d[DATA_SHEET_KEYS.D_INDEX].fillna('').apply(lambda x: f"_{x}" if x else '')
    # set as index
    d.set_index(DATA_SHEET_KEYS.INDEX, inplace=True)
    if DATA_SHEET_KEYS.CANCER in d.columns:
        # convert y/n to boolean
        d[DATA_SHEET_KEYS.CANCER] = d[DATA_SHEET_KEYS.CANCER].map({'y': True, 'n': False})
    # determine resources for each dataset
    determine_resources(config, d)
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

def check_config(conf: dict):
    # validate config types
    check_types(conf)
    # validate correction method
    check_method(conf)
    # add individual resource requirements
