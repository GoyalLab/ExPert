import os

# Get the absolute path to the directory containing the Snakefile
workflow.basedir = os.path.abspath(os.path.dirname(workflow.snakefile))

# Add the project root to PYTHONPATH
os.environ["PYTHONPATH"] = f"{workflow.basedir}:{os.environ.get('PYTHONPATH', '')}"

# specify config file
configfile: "config/test.yaml"
# get basic parameters
DATA = config['data']
LOG = config['log']
# List of datasets to process
DATASETS = config['datasets']
DATASET_NAMES = list(DATASETS.keys())
DATASET_URLS = DATASETS
# output path
OUTPUT_DIR = config['output_dir']
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset.h5ad')
# HVG pool output
HVG_DIR = os.path.join(OUTPUT_DIR, 'hvg')
HVG_POOL = os.path.join(OUTPUT_DIR, 'hvg_pool.txt')

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
        os.path.join(LOG,"{dataset}.log")
    params:
        url = lambda wildcards: DATASET_URLS[wildcards.dataset],
        name = lambda wildcards: wildcards.dataset,
        cache = config['cache'],
        qc = config['qc'],
        hvg = config['hvg']
    script:
        "workflow/scripts/process_dataset.py"

# 2. Determine pool of highly variable genes to include in meta set
rule determine_hvg:
    input:
        os.path.join(DATA, "{dataset}.h5ad")
    output:
        os.path.join(HVG_DIR, "{dataset}_hvgs.txt")
    script:
        "workflow/scripts/determine_hvgs.py"

# 3. Combine hvg pool and harmonize datasets
rule merge_datasets:
    input:
        hvg_files = expand(os.path.join(HVG_DIR, "{dataset}_hvgs.txt"), dataset=DATASET_NAMES),
        dataset_files = expand(os.path.join(DATA, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    log:
        os.path.join(LOG, "merge.log")
    params:
        method = config['correction_method']
    output:
        merged_set = OUTPUT_FILE,
        pool = HVG_POOL
    script:
        "workflow/scripts/merge.py"
