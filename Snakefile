import os
from src.workflow_utils import (
    load_configs,
    get_param_hash,
    save_config,
    read_data_sheet,
    get_job_resources,
    estimate_resources
)
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
# List of datasets to process
DATASET_SHEET = read_data_sheet(config)
DATASET_NAMES = DATASET_SHEET.index.tolist()

## BUILD I/O PATHS
# Generate hash code for each run config and use as output directory
CONFIG_HASH = get_param_hash(config, DATASET_NAMES)
OUTPUT_DIR = os.path.join(str(config['output_dir']), CONFIG_HASH)
PLT_DIR = os.path.join(OUTPUT_DIR, config['plot_dir']) if config['plot_dir'] != '' else None
LOG = os.path.join(OUTPUT_DIR, config['log_dir'])
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
EFF_DIR = os.path.join(CACHE_DIR, 'efficiencies')
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
if config['gene_embedding'] is not None and config['gene_embedding'] != '' and os.path.exists(config['gene_embedding']):
    OUTPUT_FILE_W_EMB = f"{OUTPUT_FILE.rstrip('.h5ad')}_w_emb.h5ad"
    ENPOINT = OUTPUT_FILE_W_EMB
else:
    OUTPUT_FILE_W_EMB = None
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
        **get_job_resources(
            config['resources'], 
            job_name='download_dataset',
            output=os.path.join(LOG, 'downloads', "{dataset}.slurm.log")
        )
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
        min_cells_per_class = config['min_cells_per_class'],
        min_dataset_frac = config['min_dataset_frac'],
        dataset_sheet = DATASET_SHEET,
        plt_dir = PLT_DIR
    resources:
        **get_job_resources(config['resources'], job_name='meta_info', output=os.path.join(LOG, 'meta_info.slurm.log'))
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
        hvg = config['hvg'],
        n_hvg = config['n_hvg'],
        subset = config['subset_hvg'],
        n_ctrl = config['n_ctrl'],
        use_perturbation_pool = config['use_perturbation_pool'],
        use_feature_pool = config['use_feature_pool'],
        z_score_filter = config['z_score_filter'],
        control_neighbor_threshold = config['control_neighbor_threshold'],
        min_cells_per_perturbation = config['min_cells_per_perturbation'],
        single_perturbations_only = config['single_perturbations_only'],
        p_col = config['perturbation_col'],
        ctrl_key = config['ctrl_key'],
        seed = config['seed']
    resources:
        time = config['resources']['jobs']['process_dataset']['time'],
        mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
        partition = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.PARTITION],
        output = lambda wildcards: os.path.join(LOG, 'processing', f"{wildcards.dataset}.slurm.log")
    script:
        "workflow/scripts/process_dataset.py"


# 2.1 Filter cells based on perturbation efficiency
PREPARE_INPUT = {
    # Pre-processed dataset file
    'dataset_file': os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
    # HVG pool
    'pool': HVG_POOL,
}
# Calculate mixscale scores
if config['mixscale_filter']:
    # Add input
    PREPARE_INPUT['scores_file'] = os.path.join(EFF_DIR, "{dataset}.csv")
    # Setup mixscale environment for mixscale runs
    rule setup_mixscale:
        output:
            setup_status = os.path.join(".snakemake/conda", ".mixscale_setup.txt")
        conda:
            "workflow/envs/mixscale.yaml"
        resources:
            **get_job_resources(config['resources'], job_name='setup_mixscale')
        shell:
            """
            echo 'Settin up mixscale env'
            bash workflow/envs/setup_mixscale.sh > {output.setup_status}
            """
    # TODO: change output to be a csv file (scores series only), Run mixscale for each dataset
    rule mixscale:
        input:
            dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
            setup_status = os.path.join(".snakemake/conda", ".mixscale_setup.txt")
        output:
            scores_file = os.path.join(EFF_DIR, "{dataset}.csv")
        conda:
            "workflow/envs/mixscale.yaml"
        log:
            os.path.join(LOG, 'mixscale', "{dataset}.log")
        params:
            ctrl_dev = config['ctrl_dev'],
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            min_deg = config['min_deg'],
        resources:
            time = config['resources']['jobs']['mixscale']['time'],
            mem = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=6,
                job_name='mixscale'
            )['mem'],
            partition = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=6,
                job_name='mixscale'
            )['partition'],
            cpus_per_task = int(config['resources']['jobs']['mixscale'].get('threads', 1)),
            output = lambda wildcards: os.path.join(LOG, 'mixscale', f"{wildcards.dataset}.slurm.log")
        shell:
            """
            Rscript workflow/scripts/mixscale.R \\
            -i {input.dataset_file} \\
            -o {output.scores_file} \\
            -d {params.min_deg} \\
            -t {params.ctrl_dev} \\
            --pcol {params.perturbation_col} \\
            -c {params.ctrl_key} \\
            -n {resources.cpus_per_task}
            """
# Calculate custom mixscale scores
elif config['efficiency_filter']:
    # Add input
    PREPARE_INPUT['scores_file'] = os.path.join(EFF_DIR, "{dataset}.csv")
    # Run custom python mixscale implementation for each dataset
    rule filter_cells_by_efficiency:
        input:
            dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
        output:
            scores_file = os.path.join(EFF_DIR, "{dataset}.csv")
        log:
            os.path.join(LOG, 'filter_cells_by_efficiency', "{dataset}.log")
        params:
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            min_deg = config['min_deg'],
        resources:
            time = config['resources']['jobs']['filter_cells_by_efficiency']['time'],
            mem = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=2,
                job_name='filter_cells_by_efficiency'
            )['mem'],
            partition = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=2,
                job_name='filter_cells_by_efficiency'
            )['partition'],
            cpus_per_task = int(config['resources']['jobs']['filter_cells_by_efficiency'].get('threads', 1)),
            output = lambda wildcards: os.path.join(LOG, 'filter_cells_by_efficiency', f"{wildcards.dataset}.slurm.log")
        script:
            "workflow/scripts/efficiency_filter.py"
# Skip efficiency step as input for prepare
else:
    PREPARE_INPUT['scores_file'] = None


# 3. Determine highly variable genes in each dataset
rule determine_hvg:
    input:
        os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    output:
        os.path.join(HVG_DIR, "{dataset}_hvgs.csv")
    log:
        os.path.join(LOG, 'determine_hvg', "{dataset}.log")
    resources:
        **get_job_resources(
            config['resources'], 
            job_name='determine_hvg',
            output = lambda wildcards: os.path.join(LOG, 'determine_hvg', f"{wildcards.dataset}.slurm.log")
         )
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
        **get_job_resources(config['resources'], job_name='build_gene_pool', output=os.path.join(LOG, 'pool.slurm.log'))
    script:
        "workflow/scripts/pool.py"


# 5. Prepare each dataset for the merge
rule prepare_dataset:
    input:
        **PREPARE_INPUT
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
        mem = lambda wc: estimate_resources(
            os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
            resource_config=config["resources"],
            factor=4,
            job_name='prepare_dataset'
        )['mem'],
        partition = lambda wc: estimate_resources(
            os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
            resource_config=config["resources"],
            factor=4,
            job_name='prepare_dataset'
        )['partition'],
        output = lambda wildcards: os.path.join(LOG, 'prepare', f"{wildcards.dataset}.slurm.log")
    script:
        "workflow/scripts/prepare_dataset.py"


# 6. Merge datasets into meta-set
rule merge_datasets:
    input:
        dataset_files = expand(os.path.join(PREPARE_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    output:
        merged_set = MERGED_OUTPUT_FILE
    log:
        os.path.join(LOG, "merge.log")
    params:
        merge_method = config['merge_method']
    resources:
        **get_job_resources(config['resources'], job_name='merge_datasets', output=os.path.join(LOG, 'merge.slurm.log'))
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
if OUTPUT_FILE_W_EMB is not None:
    rule add_gene_embedding:
        input:
            input_file = OUTPUT_FILE
        output:
            output_file = OUTPUT_FILE_W_EMB
        log:
            os.path.join(LOG, "add_emb.log")
        params:
            gene_embedding_file = config['gene_embedding'],
            ctx_embedding_file = config['context_embedding'],
            add_emb_for_features = config['add_emb_for_features'],
        resources:
            **get_job_resources(config['resources'], job_name='add_gene_embedding', output=os.path.join(LOG, 'add_emb.slurm.log'))
        script:
            "workflow/scripts/add_gene_embedding.py"
