import os
from src.workflow_utils import load_configs, get_param_hash, save_config, read_data_sheet, check_config, get_job_resources
from src.statics import DATA_SHEET_KEYS
import numpy as np


# Get the absolute path to the directory containing the Snakefile
workflow.basedir = os.path.abspath(os.path.dirname(workflow.snakefile))

# Add the project root to PYTHONPATH
os.environ["PYTHONPATH"] = f"{workflow.basedir}:{os.environ.get('PYTHONPATH', '')}"

# Load and merge configs
config = load_configs(wf=workflow)

## PARAMETERS: I/O
CACHE_DIR = str(config.get('cache_dir'))
LOG = config.get('log_dir')
# List of datasets to process
DATASET_SHEET = read_data_sheet(config)
DATASET_NAMES = DATASET_SHEET.index.tolist()

## BUILD I/O PATHS
# Generate hash code for each run config and use as output directory
CONFIG_HASH = get_param_hash(config, DATASET_NAMES)
OUTPUT_DIR = os.path.join(str(config['output_dir']), CONFIG_HASH)
PLT_DIR = os.path.join(OUTPUT_DIR, config['plot_dir']) if config['plot_dir'] != '' else None
# save used params in output directory
save_config(config, os.path.join(OUTPUT_DIR, 'config.yaml'))

# Output files
PERTURBATION_POOL_FILE = os.path.join(OUTPUT_DIR, 'perturbation_pool.csv')
FEATURE_POOL_FILE = os.path.join(OUTPUT_DIR, 'feature_pool.csv')
MERGED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset.h5ad')
HARMONIZED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset_harmonized.h5ad')
# download_datasets
DWNL_DIR = os.path.join(CACHE_DIR, 'raw')
# process_dataset
PROCESS_DIR = os.path.join(CACHE_DIR, 'processed')
# filter_cells
FILTER_DIR = os.path.join(CACHE_DIR, 'filtered')
# prepare_dataset
PREPARE_DIR = os.path.join(CACHE_DIR, 'prepared')
# HVG outputs
HVG_DIR = os.path.join(OUTPUT_DIR, 'hvg')
HVG_POOL = os.path.join(OUTPUT_DIR, 'hvg_pool.csv')
# obs outputs
OBS_DIR = os.path.join(OUTPUT_DIR, 'obs')
# model output
MODEL_DIR = os.path.join(OUTPUT_DIR, 'scanvi')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pt')
# define final pipeline endpoint, i.e. merged dataset or harmonized dataset
OUTPUT_FILE = MERGED_OUTPUT_FILE if config['correction_method']=='skip' else HARMONIZED_OUTPUT_FILE
# add gene embedding to output file
if os.path.exists(config['gene_embedding']):
    OUTPUT_FILE_W_EMB = f"{OUTPUT_FILE.rstrip('.h5ad')}_w_emb.h5ad"
    ENPOINT = OUTPUT_FILE_W_EMB
else:
    ENPOINT = OUTPUT_FILE


## PIPELINE START

rule all:
    input:
        ENPOINT


# 0. Download each dataset
rule download_dataset:
    output:
        raw = os.path.join(DWNL_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'downloads', "{dataset}.log") 
    params:
        url = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.URL],
        name = lambda wildcards: wildcards.dataset,
        cache = config['cache']
    resources:
        **get_job_resources(config['resources'], job_name='download_dataset')
    script:
        "workflow/scripts/download_dataset.py"

# 1. Create meta data overview and build perturbation pool for filtering
rule meta_info:
    input:
        input_files = expand(os.path.join(DWNL_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    output:
        perturbation_pool_file = PERTURBATION_POOL_FILE,
        feature_pool_file = FEATURE_POOL_FILE,
    log:
        os.path.join(LOG, 'meta_info.log')
    params:
        plt_dir = PLT_DIR
    resources:
        **get_job_resources(config['resources'], job_name='meta_info')
    script:
        "workflow/scripts/info.py"

# 2. Preprocess each dataset
rule process_dataset:
    input:
        dataset_file = os.path.join(DWNL_DIR, "{dataset}.h5ad"),
        perturbation_pool_file = PERTURBATION_POOL_FILE,
        feature_pool_file = FEATURE_POOL_FILE,
    output:
        processed = os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'processing', "{dataset}.log")
    params:
        is_cancer = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.CANCER],
        name = lambda wildcards: wildcards.dataset,
        qc = config['qc'],
        norm = config['norm'],
        log_norm = config['log_norm'],
        scale = config['scale'],
        n_hvg = config['n_hvg'],
        subset = config['subset_hvg'],
        n_ctrl = config['n_ctrl'],
        use_perturbation_pool = config['use_perturbation_pool'],
        use_feature_pool = config['use_feature_pool'],
        z_score_filter = config['z_score_filter'],
        control_neighbor_threshold = config['control_neighbor_threshold'],
        min_cells_per_class = config['min_cells_per_class'],
        single_perturbations_only = config['single_perturbations_only'],
        p_col = config['perturbation_col'],
        ctrl_key = config['ctrl_key'],
        seed = config['seed']
    resources:
        time = config['resources']['jobs']['process_dataset']['time'],
        mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
        partition = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.PARTITION],
    script:
        "workflow/scripts/process_dataset.py"


# 2.1 Filter cells based on perturbation efficiency
if config['mixscale_filter']:
    # Setup mixscale environment for mixscale runs
    rule setup_mixscale:
        output:
            setup_status = os.path.join(LOG, ".mixscale_setup.txt")
        conda:
            "workflow/envs/mixscale.yaml"
        resources:
            **get_job_resources(config['resources'], job_name='setup_mixscale')
        shell:
            """
            echo 'Settin up mixscale env'
            bash workflow/envs/setup_mixscale.sh > {output.setup_status}
            """
    # Run mixscale for each dataset
    rule filter_cells_by_efficiency:
        input:
            dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
            setup_status = os.path.join(LOG, ".mixscale_setup.txt")
        output:
            filtered_file = os.path.join(FILTER_DIR, "{dataset}.h5ad")
        conda:
            "workflow/envs/mixscale.yaml"
        params:
            ctrl_dev = config['ctrl_dev'],
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            min_deg = config['min_deg'],
        resources:
            time = config['resources']['jobs']['filter_cells_by_efficiency']['time'],
            mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
            partition = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.PARTITION],
            cpus_per_task = int(config['resources']['jobs']['filter_cells_by_efficiency'].get('threads', 1)),
        shell:
            """
            Rscript workflow/scripts/mixscale.R \\
            -i {input.dataset_file} \\
            -o {output.filtered_file} \\
            -d {params.min_deg} \\
            -t {params.ctrl_dev} \\
            --pcol {params.perturbation_col} \\
            -c {params.ctrl_key} \\
            -n {resources.cpus_per_task}
            """
else:
    # Skip this step and set output to previous step
    FILTER_DIR = PROCESS_DIR


# 3. Determine highly variable genes in each dataset
rule determine_hvg:
    input:
        os.path.join(FILTER_DIR, "{dataset}.h5ad")
    params:
        hvg = config['hvg']
    output:
        os.path.join(HVG_DIR, "{dataset}_hvgs.csv")
    resources:
        **get_job_resources(config['resources'], job_name='determine_hvg')
    script:
        "workflow/scripts/determine_hvgs.py"


# 4. Determine pool of genes to include in meta set
rule build_gene_pool:
    input:
        hvg_files = expand(os.path.join(HVG_DIR, "{dataset}_hvgs.csv"), dataset=DATASET_NAMES)
    output:
        HVG_POOL
    log:
        os.path.join(LOG, 'pool.log')
    params:
        var_merge = config['var_merge']
    resources:
        **get_job_resources(config['resources'], job_name='build_gene_pool')
    script:
        "workflow/scripts/pool.py"


# 5. Prepare each dataset for the merge
rule prepare_dataset:
    input:
        pool = HVG_POOL,
        dataset_file = os.path.join(FILTER_DIR, "{dataset}.h5ad"),
    output:
        prepared = os.path.join(PREPARE_DIR, "{dataset}.h5ad"),
        obs = os.path.join(OBS_DIR, "{dataset}.csv")
    log:
        os.path.join(LOG, 'prepare', "{dataset}.log")
    params:
        zero_pad = config['zero_padding'],
        kwargs = config,
    resources:
        time = config['resources']['jobs']['prepare_dataset']['time'],
        mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
        partition = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.PARTITION],
    script:
        "workflow/scripts/prepare_dataset.py"


# 6. Merge datasets into meta-set

rule merge_datasets:
    input:
        obs_files = expand(os.path.join(OBS_DIR, "{dataset}.csv"), dataset=DATASET_NAMES),
        pool = HVG_POOL,
        dataset_files = expand(os.path.join(PREPARE_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    output:
        merged_set = MERGED_OUTPUT_FILE
    log:
        os.path.join(LOG, "merge.log")
    params:
        meta_sheet = DATASET_SHEET,
        merge_method = config['merge_method']
    resources:
        **get_job_resources(config['resources'], job_name='merge_datasets')
    script:
        "workflow/scripts/merge.py"


# 7. Harmonize merged set (optional)
# define output files of rule based on correction method
HARMONIZED_OUTPUT_FILES = {'harmonized': HARMONIZED_OUTPUT_FILE}
# Check if we want a gpu
partition_prio = 'gpu' if config['correction_method'] == 'scanvi' else None
rule harmonize:
    input:
        merged = MERGED_OUTPUT_FILE
    output:
        **HARMONIZED_OUTPUT_FILES
    log:
        os.path.join(LOG, "harmonize.log")
    params:
        method = config['correction_method'],
        model_dir = MODEL_DIR
    resources:
        **get_job_resources(config['resources'], job_name='harmonize', partition_prio=partition_prio)
    script:
        "workflow/scripts/harmonize.py"

# 8. Add gene embedding to merged dataset if that was given in params (optional)
rule add_gene_embedding:
    input:
        input_file = OUTPUT_FILE
    output:
        output_file = OUTPUT_FILE_W_EMB
    log:
        os.path.join(LOG, "add_emb.log")
    params:
        embedding_file = config['gene_embedding'],
    resources:
        **get_job_resources(config['resources'], job_name='add_gene_embedding')
    script:
        "workflow/scripts/add_gene_embedding.py"
