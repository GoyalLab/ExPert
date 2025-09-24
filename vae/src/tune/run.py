import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import itertools, random
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import pytorch_lightning as pl

import scanpy as sc
import anndata as ad

from tqdm import tqdm

from src.utils.constants import REGISTRY_KEYS
from ._statics import CONF_KEYS
from ..models._jedvi import JEDVI
from ..plotting import get_model_results, get_classification_report, get_latest_tensor_dir, plot_confusion
from ..performance import get_library, performance_metric


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run vae model')
    parser.add_argument('--input', type=str, required=True, help='Path to h5ad input file (train/val split)')
    parser.add_argument('--test', type=str, default=None, help='Path to h5ad test file (test split)')
    parser.add_argument('--config', type=str, required=True, help='config.yaml file that specififes hyperparameters for training')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory, default is config file directory')
    return parser.parse_args()

def generate_random_configs(space: dict, n_samples: int | None = None, fixed_weights: bool = True, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    # Helper to create valid schedule combos
    def valid_schedule_combos(subspace):
        keys = list(subspace.keys())
        vals = [subspace[k] for k in keys]
        for combo in itertools.product(*vals):
            combo_dict = dict(zip(keys, combo))
            # Check schedule weights
            if 'min_weight' in combo_dict and 'max_weight' in combo_dict:
                # Enforce constant weights
                if fixed_weights and combo_dict['max_weight'] != combo_dict['min_weight']:
                    continue
                # Filter for ascending schedules only
                if combo_dict['max_weight'] < combo_dict['min_weight']:
                    continue
            yield combo_dict

    # Build precomputed lists for schedules (valid only)
    schedule_spaces = {name: list(valid_schedule_combos(params)) for name, params in space['schedules'].items()}

    # Flatten other subspaces
    data_items = list(itertools.product(*space['data'].values()))
    encoder_items = list(itertools.product(*space['encoder'].values()))
    decoder_items = list(itertools.product(*space['decoder'].values()))
    plan_items = list(itertools.product(*space['plan'].values()))
    model_items = list(itertools.product(*space['model'].values()))
    schedules_items = list(itertools.product(*schedule_spaces.values()))

    # Keys for nesting
    data_keys = list(space['data'].keys())
    encoder_keys = list(space['encoder'].keys())
    decoder_keys = list(space['decoder'].keys())
    plan_keys = list(space['plan'].keys())
    model_keys = list(space['model'].keys())
    schedule_keys = list(schedule_spaces.keys())

    total_combos = (len(data_items) * len(encoder_items) * len(decoder_items) *
                    len(plan_items) * len(model_items) * len(schedules_items))
    logging.info(f"Total possible configs: {total_combos}")

    def random_config():
        return {
            'data': dict(zip(data_keys, random.choice(data_items))),
            'encoder': dict(zip(encoder_keys, random.choice(encoder_items))),
            'decoder': dict(zip(decoder_keys, random.choice(decoder_items))),
            'plan': dict(zip(plan_keys, random.choice(plan_items))),
            'model': dict(zip(model_keys, random.choice(model_items))),
            'schedules': {
                schedule_keys[i]: random.choice(schedule_spaces[schedule_keys[i]])
                for i in range(len(schedule_keys))
            }
        }

    # Generator: yield n_samples (or infinite if n_samples=None)
    count = 0
    while n_samples is None or count < n_samples:
        yield random_config()
        count += 1

def recursive_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def sample_configs(space: dict, src_config: dict, N: int = 10, base_dir: str | None = None):
    # Sample configs
    space_configs, config_paths = [], []
    for i, sample in enumerate(generate_random_configs(space, n_samples=N, seed=42)):
        # Merge sample into a deep copy of src_config
        merged_config = recursive_update(deepcopy(src_config), sample)
        space_configs.append(merged_config)

        if base_dir is None:
            continue
        
        # Create run directory & save merged config
        run_dir = os.path.join(base_dir, f'run_{str(i)}')
        os.makedirs(run_dir, exist_ok=True)
        conf_out = os.path.join(run_dir, 'config.yaml')
        config_paths.append(conf_out)
        logging.info(f'Saving run config to {conf_out}')
        with open(conf_out, 'w') as f:
            yaml.dump(merged_config, f, sort_keys=False)

    # Return configs or configs + saved paths
    if base_dir is None:
        return space_configs
    else:
        return space_configs, pd.DataFrame({'config_path': config_paths})

# Recursive function to replace "nn.*" strings with actual torch.nn classes
def replace_nn_modules(d):
    if isinstance(d, dict):
        return {k: replace_nn_modules(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_nn_modules(v) for v in d]
    elif isinstance(d, str) and d.startswith("nn."):
        attr = d.split("nn.")[-1]
        return getattr(nn, attr)
    elif isinstance(d, str) and hasattr(nn, d):
        return getattr(nn, d)
    else:
        return d

def read_config(config_p: str) -> dict:
    """Read hyperparameter yaml file"""
    logging.info(f'Loading config file: {config_p}')
    with open(config_p, 'r') as f:
        config: dict = yaml.safe_load(f)
    # Check for config keys
    expected_keys = set(CONF_KEYS._asdict().values())
    assert expected_keys.issubset(config.keys()), f"Missing keys: {expected_keys - set(config.keys())}"
    # Convert nn modules to actual classes
    config = replace_nn_modules(config)
    return config

def train(
        adata_p: str, 
        step_model_dir: str, 
        config: dict, 
        cls_label: str = 'cls_label',
        batch_key: str = 'dataset',
        verbose: bool = True,
    ) -> tuple[nn.Module, pd.DataFrame, ad.AnnData]:
    logging.info(f'Reading training data from: {adata_p}')
    model_set = sc.read(adata_p)
    # Check if dataset is compatible
    assert cls_label in model_set.obs.columns and batch_key in model_set.obs.columns
    
    # Define all labels to classify on
    if 'perturbation_direction' in model_set.obs.columns:
        logging.info('Using perturbation direction to classify')
        cls_labels = ['perturbation_direction', 'perturbation']
    else:
        cls_labels = ['celltype', 'perturbation_type', 'perturbation']
    if verbose:
        # Check number of unique perturbations to classify
        logging.info(f'Initializing dataset with {model_set.obs.cls_label.nunique()} classes')
        logging.info(f'{model_set.obs[cls_labels[1:]].drop_duplicates().shape[0]} unique perturbations')
        logging.info(f'{model_set.obs[cls_labels[-1]].nunique()} unique gene-perturbations')
        logging.info(f'{model_set.obs["celltype"].nunique()} unique cell types')
        logging.info(f'{model_set.obs["dataset"].nunique()} datasets')
        logging.info(f'Mean number of cells / perturbation {model_set.obs.cls_label.value_counts().mean()}')
        logging.info(f'Class embedding shape: {model_set.uns["cls_embedding"].shape}')
        logging.info(f'Adata shape: {model_set.shape}')

    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Add schedule params to plan
    config[CONF_KEYS.PLAN]['anneal_schedules'] = config[CONF_KEYS.SCHEDULES]
    # Add plan to train
    config[CONF_KEYS.TRAIN]['plan_kwargs'] = config[CONF_KEYS.PLAN]
    # Add encoder and decoder args to model
    config[CONF_KEYS.MODEL]['extra_encoder_kwargs'] = config[CONF_KEYS.ENCODER]
    config[CONF_KEYS.MODEL]['extra_decoder_kwargs'] = config[CONF_KEYS.DECODER]
    # Add classifier args to model
    config[CONF_KEYS.MODEL]['classifier_parameters'] = config[CONF_KEYS.CLS]

    # Setup model
    logging.info('Setting up model.')
    # Setup anndata with model
    JEDVI.setup_anndata(
        model_set,
        batch_key=batch_key,
        labels_key=cls_label
    )
    model = JEDVI(model_set, **config[CONF_KEYS.MODEL].copy())
    logging.info(model)
    # Set training logger
    config[CONF_KEYS.TRAIN]['logger'] = pl.loggers.TensorBoardLogger(step_model_dir)
    # Train the model
    logging.info(f'Running at: {step_model_dir}')
    model.train(
        data_params=config[CONF_KEYS.DATA].copy(), 
        model_params=config[CONF_KEYS.MODEL].copy(), 
        train_params=config[CONF_KEYS.TRAIN].copy(), 
        return_runner=False
    )
    # Save results to lightning directory
    results, latent = get_model_results(
        model=model, 
        cls_labels=cls_labels, 
        log_dir=step_model_dir, 
        plot=True, 
        max_classes=100
    )
    return model, results, latent

def load_test_data(test_adata_p: str, model: nn.Module, incl_unseen: bool = True):
    # Read adata from file
    test_ad = sc.read(test_adata_p)
    # Convert gene names if needed
    if test_ad.var_names.str.lower().str.startswith('ens').all():
        logging.info(f'Dataset .var indices are ensembl ids, attempting transfer to gene symbols using internal adata.var.')
        test_ad = ens_to_symbol(test_ad).copy()
    adata = model.adata

    train_perturbations = set(adata.obs.perturbation.unique())
    test_perturbations = set(test_ad.obs.perturbation.unique())
    embedding_perturbations = set(
        adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['labels'].str.replace('[neg;|pos;]', '', regex=True)
    )

    shared = test_perturbations.intersection(train_perturbations)
    unseen = test_perturbations.difference(train_perturbations)
    test_embedding = test_perturbations.intersection(embedding_perturbations)

    logging.info(f"Found {len(train_perturbations)} trained class embeddings in model.")
    logging.info(f"Found {len(embedding_perturbations)} total class embeddings in model.")
    logging.info(f"Found {len(test_perturbations)} test perturbations.")
    logging.info(f"Found {len(shared)} shared perturbations between training and testing.")
    logging.info(f"Found {len(unseen)} unseen perturbations between training and testing.")
    logging.info(f"Found {len(test_embedding)} perturbations in model's class embedding.")

    if incl_unseen:
        test_ad._inplace_subset_obs(test_ad.obs.perturbation.isin(test_embedding))
    else:
        test_ad._inplace_subset_obs(test_ad.obs.perturbation.isin(shared))
        logging.info(f"Subsetting to training perturbations only, got {len(shared)}")

    return test_ad


def setup_labels(test_ad, test_adata_p: str, model, use_fixed_dataset_label: bool, cls_label: str = 'cls_label', batch_key: str = 'dataset', seed: int = 42):
    test_ad.obs['perturbation_direction'] = 'neg'
    if 'perturbation_direction' in test_ad.obs.columns:
        cls_labels = ['perturbation_direction', 'perturbation']
    else:
        cls_labels = ['celltype', 'perturbation_type', 'perturbation']

    test_ad.obs[cls_label] = test_ad.obs[cls_labels].agg(';'.join, axis=1)

    ds_name = os.path.basename(test_adata_p).split('.h5ad')[0]
    # Save original batch keys in adata
    orig_batch_key = f'orig_{batch_key}'
    test_ad.obs[orig_batch_key] = test_ad.obs[batch_key].values
    # Set dataset labels for classification
    if not use_fixed_dataset_label:
        logging.info("Randomly drawing dataset labels from training data.")
        training_datasets = model.adata.obs.dataset.unique()
        np.random.seed(seed)
        ds_names = np.random.choice(training_datasets, test_ad.shape[0])
        test_ad.obs[batch_key] = pd.Categorical(ds_names)
    else:
        logging.info(f"Added {ds_name} as dataset key.")
        test_ad.obs[batch_key] = ds_name

    return cls_label, batch_key, orig_batch_key


def filter_test_data(test_ad, min_ms: float, min_cpp: int):
    if 'mixscale_score' in test_ad.obs and min_ms is not None and min_ms > 0:
        ms_mask = test_ad.obs.mixscale_score >= min_ms
        test_ad._inplace_subset_obs(ms_mask)
        logging.info(f"Filtered for minimum mixscale score of {min_ms}, got {test_ad.shape[0]} cells.")

    cpp = test_ad.obs.perturbation.value_counts()
    valid = cpp[cpp >= min_cpp].index
    test_ad._inplace_subset_obs(test_ad.obs.perturbation.isin(valid))
    logging.info(f"Found {valid.shape[0]} perturbations with at least {min_cpp} cells.")

    return test_ad


def run_model_predictions(model, test_ad, cls_label, batch_key, incl_unseen: bool = True):
    # Always use interal gene embeddings if used in model
    if REGISTRY_KEYS.GENE_EMB_KEY in model.adata.varm:
        logging.info("Using model's gene embeddings for inference.")
        test_ad.varm[REGISTRY_KEYS.GENE_EMB_KEY] = model.adata.varm[REGISTRY_KEYS.GENE_EMB_KEY]
    model.setup_anndata(test_ad, labels_key=cls_label, batch_key=batch_key)
    test_ad = model.create_test_data(test_ad)

    # Get latent representation of model
    test_ad.obsm['latent_z'] = model.get_latent_representation(adata=test_ad)
    predictions, cz = model.predict(adata=test_ad, return_latent=True, soft=True, use_full_cls_emb=incl_unseen)
    
    test_ad.obsm['latent_cz'] = cz
    test_ad.obs['cls_prediction'] = predictions.columns[predictions.values.argmax(axis=-1)]
    test_ad.obs['cls_score'] = predictions.max(axis=1) - predictions.mean(axis=1)

    return test_ad, predictions


def evaluate_predictions(test_ad, predictions, train_perturbations, output_dir: str, plot: bool, groupby: str, use_pathway: bool = True):
    _, report = get_classification_report(test_ad, cls_label='cls_label', mode='test')
    report.sort_values('f1-score', ascending=False, inplace=True)
    report['perturbation'] = report.index.str.split(';').str[1]
    report['is_training_perturbation'] = report.perturbation.isin(train_perturbations)
    # Save predictions to adata
    test_ad.obsm['soft_predictions'] = predictions
    # Setup pathway library
    lib = get_library(library='KEGG_2019_Human', organism='Human') if use_pathway else None
    # Calculate top n predictions
    top_n_predictions = compute_top_n_predictions(test_ad, predictions, train_perturbations, lib=lib, groupby=groupby)
    top_n_predictions.to_csv(os.path.join(output_dir, 'top_n_predictions_report.csv'))
    # Plot metrics results
    if plot:
        # Create plot dir if not already there
        plt_dir = os.path.join(output_dir, 'plots', 'test')
        os.makedirs(plt_dir, exist_ok=True)
        # Plot confusion matrix
        # Filter out random predictions
        no_random = top_n_predictions[top_n_predictions['mode'] != 'random'].copy()
        n_classes = test_ad.obs.cls_label.nunique()
        # Plot difference in F1 between trained and unseen classes
        plot_f1_zero_shot(no_random=no_random, n_classes=n_classes, plt_dir=plt_dir, use_pathway=False)
        plot_f1_zero_shot(no_random=no_random, n_classes=len(lib), plt_dir=plt_dir, use_pathway=True)
        # Plot F1 distribution over groups (cell types)
        plot_f1_groups(no_random=no_random, plt_dir=plt_dir, groupby=groupby, train_cls_only=False, use_pathway=False)
        plot_f1_groups(no_random=no_random, plt_dir=plt_dir, groupby=groupby, train_cls_only=True, use_pathway=False)
        plot_f1_groups(no_random=no_random, plt_dir=plt_dir, groupby=groupby, train_cls_only=False, use_pathway=True)
        plot_f1_groups(no_random=no_random, plt_dir=plt_dir, groupby=groupby, train_cls_only=True, use_pathway=True)
        # Plot all random vs. normal predictions
        plot_f1_random_vs_normal(top_n_predictions, n_classes=n_classes, plt_dir=plt_dir, use_pathway=False, train_only=False)
        plot_f1_random_vs_normal(top_n_predictions, n_classes=n_classes, plt_dir=plt_dir, use_pathway=True, train_only=False)
        plot_f1_random_vs_normal(top_n_predictions, n_classes=n_classes, plt_dir=plt_dir, use_pathway=False, train_only=True)
        plot_f1_random_vs_normal(top_n_predictions, n_classes=n_classes, plt_dir=plt_dir, use_pathway=True, train_only=True)


def compute_top_n_predictions(test_ad, predictions, train_perturbations, n: int = 20, groupby: str = '_dataset', lib = None):
    # Calculate label index in prediction columns per cell
    test_ad.obs['cls_prediction_idx'] = (test_ad.obs.cls_label.values == predictions.columns.values.reshape(-1, 1)).argmax(axis=0)
    labels = test_ad.obs.cls_label.values
    max_idx = np.argsort(predictions, axis=1)

    # Look at top N predictions (can be useful for pathways etc.)
    top_n_predictions = []
    # Calculate F1-scores separately for each group
    for ct in test_ad.obs[groupby].unique():
        tmp_mask = (test_ad.obs[groupby]==ct)
        idx = np.where(tmp_mask)[0]
        tmp = test_ad[tmp_mask]
        y = tmp.obs['cls_prediction_idx'].values
        labels = tmp.obs.cls_label.values
        for top_n in tqdm(np.arange(n) + 1):
            top_predictions = max_idx[idx,-top_n:]
            hit_mask = top_predictions == np.array(y)[:, np.newaxis]
            hit_idx = np.argmax(hit_mask, axis=1)
            is_hit = np.any(hit_mask, axis=1).astype(int)

            stats_per_label = pd.DataFrame({
                'y': np.concatenate([y[is_hit == 1], y[is_hit == 0]]),
                'idx': np.concatenate([(top_n + 1 - hit_idx[is_hit == 1]) / (top_n + 1), np.repeat(0, np.sum(is_hit == 0))]),
                'label': np.concatenate([labels[is_hit == 1], labels[is_hit == 0]]),
                'prediction': np.concatenate([y[is_hit == 1], top_predictions[is_hit == 0][:, -1]])
            })
            stats_per_label['pred_label'] = predictions.columns[stats_per_label.prediction.values.astype(int)]
            random_pred = pd.Series(np.random.choice(predictions.columns, stats_per_label.shape[0], replace=True)).str.split(';').str[1]

            actual = stats_per_label.label.str.split(';').str[1]
            pred = stats_per_label.pred_label.str.split(';').str[1]
            l = predictions.columns.str.split(';').str[1]
            # Calculate performance metrices for predictions
            test_metrics = performance_metric(actual, pred, l, lib=None, mode='test')
            # Calculate random performance
            rand_metrics = performance_metric(actual, random_pred, l, lib=None, mode='random')
            # Concatenate outputs
            metrics = pd.concat((test_metrics, rand_metrics), axis=0)
            metrics['use_pathway'] = False
            # Calculate pathway predictions and add to overall metrics
            if lib is not None:
                test_pw_metrics = performance_metric(actual, pred, l, lib=lib, mode='test')
                rand_pw_metrics = performance_metric(actual, random_pred, l, lib=lib, mode='random')
                pw_metrics = pd.concat((test_pw_metrics, rand_pw_metrics), axis=0)
                pw_metrics['use_pathway'] = True
                metrics = pd.concat((metrics, pw_metrics), axis=0)
            # Kick predictions with no support in test data
            metrics = metrics[metrics.support > 0].copy()
            # Add iteration information
            metrics['top_n'] = top_n
            metrics[groupby] = ct
            # Add to list of predictions
            top_n_predictions.append(metrics)
    # Concat results
    top_n_predictions = pd.concat(top_n_predictions, axis=0)
    top_n_predictions['perturbation'] = top_n_predictions.index
    top_n_predictions['is_training_perturbation'] = top_n_predictions.perturbation.isin(train_perturbations)

    return top_n_predictions

def plot_f1_random_vs_normal(top_n_predictions: pd.DataFrame, n_classes: int, plt_dir: str, use_pathway: bool = False, train_only: bool = True):
    tmp = top_n_predictions[(top_n_predictions['use_pathway']==use_pathway) & (top_n_predictions.is_training_perturbation==train_only)]
    zs_title = ' (Unseen Perturbations)' if not train_only else ''
    add_title = ' (Pathway-based)' if use_pathway else ''
    plt.figure(dpi=300, figsize=(10, 5))
    ax = sns.boxenplot(tmp, x='top_n', y='f1', hue='mode',
                  palette=['#7900d7', '#3274a1'])
    plt.xlabel('Number of top predictions')
    plt.ylim([0.0, 1.0])
    plt.ylabel('F1-score per class')
    plt.title(f'Test F1-score distribution over top predictions (N={n_classes}){zs_title}{add_title}', pad=20)
    # Build legend labels with counts
    handles, labels = ax.get_legend_handles_labels()
    counts = tmp.groupby('mode')['perturbation'].nunique()
    new_labels = [
        f"{label} (N={counts[label]})"
        for label in labels
    ]

    # Place legend outside plot
    plt.legend(
        handles, new_labels,
        title="Training Perturbation",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )
    plt.tight_layout()
    _o = "_pw" if use_pathway else ""
    _z = "_zero_shot" if not train_only else ""
    o = f"test_f1_vs_random{_o}{_z}.svg"
    plt.savefig(os.path.join(plt_dir, o))
    plt.close()

def plot_f1_zero_shot(no_random: pd.DataFrame, n_classes: int, plt_dir: str, use_pathway: bool = False):
    tmp = no_random[no_random['use_pathway']==use_pathway]
    add_title = ' (Pathway-based)' if use_pathway else ''
    plt.figure(dpi=300, figsize=(10, 5))
    ax = sns.boxenplot(tmp, x='top_n', y='f1', hue='is_training_perturbation',
                  palette=['#7900d7', '#3274a1'])
    plt.xlabel('Number of top predictions')
    plt.ylim([0.0, 1.0])
    plt.ylabel('F1-score per class')
    plt.title(f'Test F1-score distribution over top predictions (N={n_classes}){add_title}', pad=20)
    # Build legend labels with counts
    handles, labels = ax.get_legend_handles_labels()
    counts = tmp.groupby('is_training_perturbation')['perturbation'].nunique()
    new_labels = [
        f"{label} (N={counts[True] if label == 'True' else counts[False]})"
        for label in labels
    ]

    # Place legend outside plot
    plt.legend(
        handles, new_labels,
        title="Training Perturbation",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )
    plt.tight_layout()
    _o = "_pw" if use_pathway else ""
    o = f"test_f1_train_cls{_o}.svg"
    plt.savefig(os.path.join(plt_dir, o))
    plt.close()

def plot_f1_groups(no_random: pd.DataFrame, plt_dir: str, groupby: str = 'celltype', top_n: int = 10, train_cls_only: bool = True, use_pathway: bool = True):
    # Plot test cell types
    mask = (no_random.top_n<=top_n) & (no_random['mode']=='test') & (no_random['is_training_perturbation']==train_cls_only) & (no_random['use_pathway']==use_pathway)
    top_10 = no_random[mask].copy()
    zs_title = ' (Unseen Perturbations)' if not train_cls_only else ''
    add_title = ' (Pathway-based)' if use_pathway else ''
    plt.figure(dpi=300, figsize=(10, 5))
    ax = sns.boxenplot(top_10, x='top_n', y='f1', hue=groupby, flier_kws={'s': 10}, legend=True)
    plt.xlabel('Number of top predictions')
    plt.ylim([0.0, 1.0])
    plt.ylabel('F1-score per class')
    plt.title(f'F1-score distribution over top predictions (N={top_10.perturbation.nunique()}){zs_title}{add_title}', pad=20)
    # Add counts to legend labels
    handles, labels = ax.get_legend_handles_labels()
    counts = top_10.groupby(groupby)['perturbation'].nunique()
    new_labels = [
        f"{label} (N={counts[label]})"
        for label in labels
    ]

    # Place legend outside plot
    plt.legend(
        handles, new_labels,
        title="Cell type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0
    )
    plt.tight_layout()
    _o = "_pw" if use_pathway else ""
    _z = "_zero_shot" if not train_cls_only else ""
    o = f"test_f1_per_ct{_o}{_z}.svg"
    plt.savefig(os.path.join(plt_dir, o))
    plt.close()

def ens_to_symbol(adata: ad.AnnData, gene_symbol_keys: list[str] = ['gene_symbol', 'gene_name', 'gene', 'gene symbol', 'gene name']) -> ad.AnnData:
    # Look for possible gene symbol columns
    gscl = adata.var.columns.intersection(set(gene_symbol_keys)).values
    if len(gscl) == 0:
        raise ValueError(f'Could not find a column that describes gene symbol mappings in adata.var, looked for {gene_symbol_keys}')
    # Choose first hit if multiple
    gsh = list(gscl)[0]
    # Convert index
    adata.var.reset_index(names='ensembl_id', inplace=True)
    adata.var.set_index(gsh, inplace=True)
    # Check for duplicate index conflicts
    if adata.var_names.nunique() != adata.shape[0]:
        logging.info(f'Found duplicate indices for ensembl to symbol mapping, highest number of conflicts: {adata.var_names.value_counts().max()}')
        # Fix conflicts by choosing the gene with the higher harmonic mean of mean expression and normalized variance out of pool
        if len(set(['means', 'variances_norm']).intersection(adata.var.columns)) == 2:
            adata.var['hm_var'] = (2 * adata.var.means * adata.var.variances_norm) / (adata.var.means + adata.var.variances_norm)
        else:
            adata.var['hm_var'] = np.arange(adata.n_vars)
        idx = adata.var.reset_index().groupby(gsh, observed=True).hm_var.idxmax().values
        adata = adata[:,idx]
    return adata

def update_latent(model: nn.Module, test_ad: ad.AnnData, version_dir: str, plot: bool = True, incl_unseen: bool = True, batch_key: str = 'dataset', orig_group_key: str = 'orig_dataset'):
    # I/O
    plt_dir = os.path.join(version_dir, 'plots')
    latent = sc.read(os.path.join(version_dir, 'latent.h5ad'))
    # Plot combination of latent spaces
    _, model_cz = model.predict(return_latent=True, soft=True, use_full_cls_emb=incl_unseen)
    idx_map = (
        latent.obs
        .reset_index(names='idx')
        .reset_index(names='latent_index')[['latent_index', 'idx']]
        .merge(
            model.adata.obs
            .reset_index(names='idx')
            .reset_index(names='model_index')[['model_index', 'idx']],
            on='idx')
    )
    # Add batch label to latent
    latent.obs[orig_group_key] = latent.obs[batch_key].values
    latent.obsm['cz'] = model_cz[idx_map.model_index]
    test_ad.obs['mode'] = 'test'
    test_z = test_ad.obsm['latent_z']
    test_z = ad.AnnData(test_z, obs=test_ad.obs)
    test_z.obsm['cz'] = test_ad.obsm['latent_cz']
    test_z.obsm['soft_predictions'] = test_ad.obsm['soft_predictions']
    # Combine latent spaces
    latent = ad.concat([latent, test_z], axis=0)
    # TODO: save cz umap as individual .obsm and not overwrite existing latent umap
    latent.obsm['cz_pca'] = sc.pp.pca(latent.obsm['cz'])
    sc.pp.neighbors(latent, use_rep='cz_pca')
    sc.tl.umap(latent)
    # Write updated latent space to file
    latent.write_h5ad(os.path.join(version_dir, 'full_latent.h5ad'))
    # Plot updated umap with new testing data
    if plot:
        logging.info(f'Plotting merged umap.')
        sc.pl.umap(latent, color=['celltype', 'mode'], ncols=1, return_fig=True, show=False)
        plt.savefig(os.path.join(plt_dir, f'full_umap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
def test(
        model, 
        test_adata_p: str, 
        output_dir: str,
        incl_unseen: bool = True,
        use_fixed_dataset_label: bool = True,
        min_ms: float = 4.0,
        min_cpp: int = 10,
        plot: bool = False,
        return_adata: bool = False
    ):
    test_ad = load_test_data(test_adata_p, model, incl_unseen=incl_unseen)
    cls_label, batch_key, orig_batch_key = setup_labels(test_ad, test_adata_p, model, use_fixed_dataset_label)
    test_ad = filter_test_data(test_ad, min_ms, min_cpp)
    test_ad, predictions = run_model_predictions(model, test_ad, cls_label, batch_key, incl_unseen=incl_unseen)
    # Evaluate predictions, plot and update latent space
    train_perturbations = set(model.adata.obs.perturbation.unique())
    # Basic confusion matrix

    evaluate_predictions(test_ad, predictions, train_perturbations, output_dir=output_dir, plot=plot, groupby=orig_batch_key)
    # Update latent representation and calculate new umaps
    logging.info(f'Updating latent space with test data.')
    update_latent(model, test_ad=test_ad, version_dir=output_dir, incl_unseen=incl_unseen, batch_key=batch_key, orig_group_key=orig_batch_key)
    if return_adata:
        return test_ad


def get_ouptut_dir(config_p: str, output_base_dir: str | None = None) -> str:
    # Create directory based on the input config name
    output_dir = os.path.dirname(config_p) if output_base_dir is None else output_base_dir
    logging.info(f'Run output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == '__main__':
    # Parse cmd line args
    args = parse_args()
    # Load config file
    config = read_config(args.config)
    step_model_dir = get_ouptut_dir(args.config, output_base_dir=args.outdir)
    # Train the model
    logging.info(f'Training config: {args.config}')
    model, results, latent = train(
        adata_p=args.input, 
        step_model_dir=step_model_dir, 
        config=config
    )
    # Get latest lightning directory
    version_dir = get_latest_tensor_dir(step_model_dir)
    # Test model performance if path is given
    if args.test is not None and os.path.exists(args.test) and args.test.endswith('.h5ad'):
        logging.info(f'Testing with: {args.test}')
        test(model, test_adata_p=args.test, output_dir=version_dir, plot=True)
