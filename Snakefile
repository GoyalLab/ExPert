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
correction_method = config.get('correction_method', 'scanorama')
check_method(correction_method, config)

cache = config.get('cache', True)
qc = config.get('qc', True)                             # Perform QC on cells
norm = config.get('norm', True)                         # Normalize gene expression (total sum)
log_norm = config.get('log_norm', True)                 # Log normalize gene expression (log1p)
n_hvg = config.get('n_hvg', 2000)                       # Number of highly variable genes to include for each dataset
subset_hvg = config.get('subset_hvg', False)            # Only include highly variable genes of each dataset
hvg = config.get('hvg', True)                           # Filter metaset genes for high variance
zero_padding = config.get('zero_padding', False)        # Fill missing genes with 0s to include all genes across the merged metaset
scale = config.get('scale', True)                       # Center and scale each dataset
plot = config.get('plot', True)                         # Whether to run plotting options such as tSNE or UMAP, if true, UMAP is default
do_tsne = config.get('do_tsne', False)                      # Calculate tSNE for merged dataset (can take some time)
do_umap = config.get('do_umap', True)                       # Calculate UMAP for merged dataset


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
        norm = norm,
        log_norm = log_norm,
        scale = scale,
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
        cores = config['cores'],
        plot = plot,
        do_umap = do_umap,
        do_tsne = do_tsne
    output:
        merged_set = OUTPUT_FILE,
        pool = HVG_POOL
    script:
        "workflow/scripts/merge.py"
