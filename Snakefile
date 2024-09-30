import os
from src.harmonize import correction_methods
from snakemake.io import load_configfile


# Get the absolute path to the directory containing the Snakefile
workflow.basedir = os.path.abspath(os.path.dirname(workflow.snakefile))

# Add the project root to PYTHONPATH
os.environ["PYTHONPATH"] = f"{workflow.basedir}:{os.environ.get('PYTHONPATH', '')}"

def load_configs(default_config="config/config.yaml"):
    # Load default config
    config = load_configfile(default_config)
    
    # Update with any additional config files
    for cf in workflow.overwrite_configfiles:
        update_config = load_configfile(cf)
        config.update(update_config)
    
    return config

# Load and merge configs
config = load_configs()

# get basic parameters
DATA = config.get('data', 'resources/datasets/data')
LOG = config.get('log', 'logs')
# List of datasets to process
DATASETS = config['datasets']
DATASET_NAMES = list(DATASETS.keys())
print(f'Got {len(DATASET_NAMES)} datasets')
DATASET_URLS = DATASETS
# save used params for execution
params = ['qc', 'n_hvg', 'subset_hvg', 'hvg', 'zero_padding', 'scale', 'correction_method']
CONFIG_STR = os.path.sep.join(f"{k}/{v}" for k, v in config.items() if k in params)
OUTPUT_DIR = os.path.join(str(config['output_dir']), CONFIG_STR)
# define output file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset.h5ad')
# HVG pool output
HVG_DIR = os.path.join(OUTPUT_DIR, 'hvg')
HVG_POOL = os.path.join(OUTPUT_DIR, 'hvg_pool.csv')

## PARAMETERS (or defaults)
cache = config.get('cache', True)
qc = config.get('qc', True)
n_hvg = config.get('n_hvg', 10000)
subset_hvg = config.get('subset_hvg', False)
hvg = config.get('hvg', True)
zero_padding = config.get('zero_padding', False)
scale = config.get('scale', True)
correction_method = config.get('correction_method', 'scanorama')
if correction_method not in correction_methods():
    raise ValueError(f'"correction_method" has to be one of {correction_methods()}')


## START OF PIPELINE

# define pipeline output, i.e. merged dataset
rule all:
    input:
        HVG_POOL,
        OUTPUT_FILE

# 1. Download and preprocess each dataset
rule process_dataset:
    output:
        os.path.join(DATA, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'processing', "{dataset}.log")
    params:
        url = lambda wildcards: DATASET_URLS[wildcards.dataset],
        name = lambda wildcards: wildcards.dataset,
        cache = cache,
        qc = qc,
        n_hvg = n_hvg,
        subset = subset_hvg
    script:
        "workflow/scripts/process_dataset.py"

# 2. Determine pool of highly variable genes to include in meta set
rule determine_hvg:
    input:
        os.path.join(DATA, "{dataset}.h5ad")
    output:
        os.path.join(HVG_DIR, "{dataset}_hvgs.csv")
    script:
        "workflow/scripts/determine_hvgs.py"

# 3. Combine hvg pool and harmonize datasets
rule merge_datasets:
    input:
        hvg_files = expand(os.path.join(HVG_DIR, "{dataset}_hvgs.csv"), dataset=DATASET_NAMES),
        dataset_files = expand(os.path.join(DATA, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    log:
        os.path.join(LOG, "merge.log")
    params:
        method = correction_method,
        hvg = hvg,
        zero_pad = zero_padding,
        scale = scale,
        cores = config['cores']
    output:
        merged_set = OUTPUT_FILE,
        pool = HVG_POOL
    script:
        "workflow/scripts/merge.py"
