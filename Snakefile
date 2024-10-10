import os
from src.harmonize import check_method
from snakemake.io import load_configfile


# Get the absolute path to the directory containing the Snakefile
workflow.basedir = os.path.abspath(os.path.dirname(workflow.snakefile))

# Add the project root to PYTHONPATH
os.environ["PYTHONPATH"] = f"{workflow.basedir}:{os.environ.get('PYTHONPATH', '')}"

def load_configs(default_config="config/config.yaml"):
    # Load default config
    config = load_configfile(default_config)
    
    # Update with any additional config files (from --configfile)
    for cf in workflow.overwrite_configfiles:
        update_config = load_configfile(cf)
        config.update(update_config)

    # Update with command-line options (from --config)
    config.update(workflow.overwrite_config)
    
    return config

# Load and merge configs
config = load_configs()

# get basic parameters
DATA = config.get('data', 'resources/datasets/data')
CACHE_DIR = config.get('cache_dir', 'resources/datasets/data/raw')
LOG = config.get('log', 'logs')
# List of datasets to process
DATASETS = config['datasets']
DATASET_NAMES = list(DATASETS.keys())
print(f'Got {len(DATASET_NAMES)} datasets')
DATASET_URLS = DATASETS
# save used params for execution
params = ['qc', 'norm', 'log_norm', 'n_hvg', 'subset_hvg', 'hvg', 'zero_padding', 'scale', 'correction_method']
CONFIG_STR = os.path.sep.join(f"{k}/{v}" for k, v in config.items() if k in params)
OUTPUT_DIR = os.path.join(str(config['output_dir']), CONFIG_STR)
# define output files
MERGED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset.h5ad')
HARMONIZED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset_harmonized.h5ad')
# process_dataset
PROCESS_DIR = os.path.join(DATA, 'processed')
# prepare_dataset
PREPARE_DIR = os.path.join(DATA, 'prepared')
# HVG outputs
HVG_DIR = os.path.join(OUTPUT_DIR, 'hvg')
HVG_POOL = os.path.join(OUTPUT_DIR, 'hvg_pool.csv')
# obs outputs
OBS_DIR = os.path.join(OUTPUT_DIR, 'obs')

## PARAMETERS (or defaults)
correction_method = config.get('correction_method', 'scANVI')
check_method(correction_method, config)

cache = config.get('cache', True)                       # Cache datasets if already downloaded
qc = config.get('qc', True)                             # Perform QC on cells
norm = config.get('norm', False)                        # Normalize gene expression (total sum)
log_norm = config.get('log_norm', False)                # Log normalize gene expression (log1p)
n_hvg = config.get('n_hvg', 2000)                       # Number of highly variable genes to include for each dataset
subset_hvg = config.get('subset_hvg', False)            # Only include highly variable genes of each dataset
hvg = config.get('hvg', True)                           # Filter metaset genes for high variance
zero_padding = config.get('zero_padding', True)         # Fill missing genes with 0s to include all genes across the merged metaset
scale = config.get('scale', False)                      # Center and scale each dataset
plot = config.get('plot', True)                         # Whether to run plotting options such as tSNE or UMAP, if true, UMAP is default (can be plotted with final object using sc.pl.*)
do_tsne = config.get('do_tsne', False)                      # Calculate tSNE for merged dataset (can take some time)
do_umap = config.get('do_umap', True)                       # Calculate UMAP for merged dataset
merge_method = config.get('merge_method', 'on_disk')    # How to merge datasets into meta-set


## START OF PIPELINE

# define final pipeline endpoint, i.e. merged dataset or harmonized dataset
OUTPUT_FILE = MERGED_OUTPUT_FILE if correction_method=='skip' else HARMONIZED_OUTPUT_FILE
rule all:
    input:
        OUTPUT_FILE


# 1. Download each dataset
rule download_dataset:
    output:
        raw = os.path.join(CACHE_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'downloads', "{dataset}.log") 
    params:
        url = lambda wildcards: DATASET_URLS[wildcards.dataset],
        name = lambda wildcards: wildcards.dataset,
        cache = cache
    script:
        "workflow/scripts/download_dataset.py"


# 2. Preprocess each dataset
rule process_dataset:
    input:
        dataset_file = os.path.join(CACHE_DIR, "{dataset}.h5ad")
    output:
        processed = os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'processing', "{dataset}.log")
    params:
        qc = qc,
        norm = norm,
        log_norm = log_norm,
        scale = scale,
        n_hvg = n_hvg,
        subset = subset_hvg
    script:
        "workflow/scripts/process_dataset.py"


# 3. Determine highly variable genes in each dataset
rule determine_hvg:
    input:
        os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    output:
        os.path.join(HVG_DIR, "{dataset}_hvgs.csv")
    script:
        "workflow/scripts/determine_hvgs.py"


# 4. Determine pool of genes to include in meta set
rule build_gene_pool:
    input:
       hvg_files = expand(os.path.join(HVG_DIR, "{dataset}_hvgs.csv"), dataset=DATASET_NAMES)
    output:
        pool = HVG_POOL
    log:
        os.path.join(LOG, 'pool.log')
    script:
        "workflow/scripts/pool.py"


# 5. Prepare each dataset for the merge
rule prepare_datasets:
    input:
        pool = HVG_POOL,
        dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    output:
        prepared = os.path.join(PREPARE_DIR, "{dataset}.h5ad"),
        obs = os.path.join(OBS_DIR, "{dataset}.csv")
    log:
        os.path.join(LOG, 'prepare', "{dataset}.log")
    params:
        hvg=hvg,
        zero_pad=zero_padding
    script:
        "workflow/scripts/prepare_dataset.py"


# 6. Merge datasets into meta-set
rule merge_datasets:
    input:
        obs_files = expand(os.path.join(OBS_DIR, "{dataset}.csv"), dataset=DATASET_NAMES),
        pool = HVG_POOL,
        dataset_files = expand(os.path.join(PREPARE_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    log:
        os.path.join(LOG, "merge.log")
    params:
        merge_method = merge_method
    output:
        merged_set = MERGED_OUTPUT_FILE
    script:
        "workflow/scripts/merge.py"


# 7. Harmonize merged set
rule harmonize:
    input:
        merged = MERGED_OUTPUT_FILE
    log:
        os.path.join(LOG, "harmonize.log")
    output:
        harmonized = HARMONIZED_OUTPUT_FILE
    params:
        method = correction_method
    script:
        "workflow/scripts/harmonize.py"
