import os
from src.workflow_utils import load_configs, get_param_hash, save_config, read_data_sheet, check_config
from src.statics import DATA_SHEET_KEYS


# Get the absolute path to the directory containing the Snakefile
workflow.basedir = os.path.abspath(os.path.dirname(workflow.snakefile))

# Add the project root to PYTHONPATH
os.environ["PYTHONPATH"] = f"{workflow.basedir}:{os.environ.get('PYTHONPATH', '')}"

# Load and merge configs
config = load_configs(wf=workflow)

## PARAMETERS: I/O
DATASHEET_PATH = str(config.get('datasheet'))
DATA = str(config.get('data_dir'))
CACHE_DIR = str(config.get('cache_dir'))
LOG = config.get('log_dir')
# List of datasets to process
DATASET_SHEET = read_data_sheet(DATASHEET_PATH)
DATASET_NAMES = DATASET_SHEET.index.tolist()
sep = '\n\t - '
dataset_info = sep + sep.join(DATASET_NAMES)
print(f'Datasets:\n{dataset_info}')


## CHECK PARAMS & BUILD I/O PATHS
check_config(config)
# Generate hash code for each run config and use as output directory
CONFIG_HASH = get_param_hash(config, DATASET_NAMES)
OUTPUT_DIR = os.path.join(str(config['output_dir']), CONFIG_HASH)
# save used params in output directory
save_config(config, os.path.join(OUTPUT_DIR, 'config.yaml'))

# Output files
MERGED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset.h5ad')
HARMONIZED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'perturb_metaset_harmonized.h5ad')
# process_dataset
PROCESS_DIR = os.path.join(DATA, 'processed')
# filter_cells
FILTER_DIR = os.path.join(DATA, 'filtered')
# prepare_dataset
PREPARE_DIR = os.path.join(DATA, 'prepared')
# HVG outputs
HVG_DIR = os.path.join(OUTPUT_DIR, 'hvg')
HVG_POOL = os.path.join(OUTPUT_DIR, 'hvg_pool.csv')
# obs outputs
OBS_DIR = os.path.join(OUTPUT_DIR, 'obs')
# model output
MODEL_DIR = os.path.join(OUTPUT_DIR, 'scanvi')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pt')


## START OF PIPELINE

# define final pipeline endpoint, i.e. merged dataset or harmonized dataset, and other outputs
OUTPUT_FILES = [MERGED_OUTPUT_FILE]
correction_method = config['correction_method']
if correction_method!='skip':
    OUTPUT_FILES.append(HARMONIZED_OUTPUT_FILE)
    if correction_method=='scANVI':
        print('Caching trained models')
        OUTPUT_FILES.append(MODEL_FILE)

rule all:
    input:
        *OUTPUT_FILES


# 1. Download each dataset
rule download_dataset:
    output:
        raw = os.path.join(CACHE_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'downloads', "{dataset}.log") 
    params:
        url = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.URL],
        name = lambda wildcards: wildcards.dataset,
        cache = config['cache']
    resources:
        time = config['dwl_t'],
        mem = config['dwl_m'],
        partition = config['dwl_p']
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
        is_cancer = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.CANCER],
        name = lambda wildcards: wildcards.dataset,
        qc = config['qc'],
        norm = config['norm'],
        log_norm = config['log_norm'],
        scale = config['scale'],
        n_hvg = config['n_hvg'],
        subset = config['subset_hvg'],
        n_ctrl = config['n_ctrl'],
        single_perturbations_only = config['single_perturbations_only'],
        p_col = config['perturbation_col'],
        ctrl_key = config['ctrl_key'],
        seed = config['seed']
    resources:
        time = config['pp_t'],
        mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
        partition = config['pp_p']
    script:
        "workflow/scripts/process_dataset.py"


# 2.1 Filter cells based on perturbation efficiency
if config['mixscale_filter']:
    # Setup mixscale environment for mixscale runs
    rule setup_mixscale:
        output:
            setup_status = os.path.join(FILTER_DIR, ".setup.txt")
        conda:
            "workflow/envs/mixscale.yaml"
        resources:
            time = '04:00:00',
            mem = '20GB',
            partition = config['fc_p']
        shell:
            """
            echo 'Settin up mixscale env'
            bash workflow/envs/setup_mixscale.sh > {output.setup_status}
            """
    # Run mixscale for each dataset
    rule filter_cells_by_efficiency:
        input:
            dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
            setup_status = os.path.join(FILTER_DIR, ".setup.txt")
        output:
            filtered_file = os.path.join(FILTER_DIR, "{dataset}.h5ad")
        conda:
            "workflow/envs/mixscale.yaml"
        params:
            ctrl_dev = config['ctrl_dev'],
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            min_cells_per_perturbation = config['min_cells_per_perturbation']
        resources:
            time = config['fc_t'],
            mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
            partition = config['fc_p']
        shell:
            """
            Rscript workflow/scripts/mixscale.R \\
            -i {input.dataset_file} \\
            -o {output.filtered_file} \\
            -t {params.ctrl_dev} \\
            -m {params.min_cells_per_perturbation} \\
            -p {params.perturbation_col} \\
            -c {params.ctrl_key}
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
        time = config['hvg_t'],
        mem = config['hvg_m'],
        partition = config['hvg_p']
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
    resources:
        time = config['pool_t'],
        mem = config['pool_m'],
        partition = config['pool_p']
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
        zero_pad = config['zero_padding']
    resources:
        time = config['prep_t'],
        mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
        partition = config['prep_p']
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
        merge_method = config['merge_method']
    resources:
        time = config['merge_t'],
        mem = config['merge_m'],
        partition = config['merge_p']
    script:
        "workflow/scripts/merge.py"


# 7. Harmonize merged set (optional)

# define output files of rule based on correction method
HARMONIZED_OUTPUT_FILES = {'harmonized': HARMONIZED_OUTPUT_FILE}
if correction_method == 'scANVI':
    print('Caching trained models')
    # HARMONIZED_OUTPUT_FILES.update({'model_file': MODEL_FILE})

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
        time = config['harm_t'],
        mem = config['harm_m'],
        partition = config['harm_p']
    script:
        "workflow/scripts/harmonize.py"
