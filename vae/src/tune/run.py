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

from src.utils.constants import REGISTRY_KEYS, TRAINING_KEYS, MODULE_KEYS
from src.utils.io import read_config, read_adata
from src.utils.preprocess import neighborhood_purity
from ._statics import CONF_KEYS, NESTED_CONF_KEYS
from ..models._jedvi import JEDVI
from ..plotting import get_model_results, get_classification_report, get_latest_tensor_dir
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
    parser.add_argument('--test_unseen', action='store_true', help='Test model on unseen perturbations')
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

def sample_configs(space: dict, src_config: dict, N: int = 10, base_dir: str | None = None, verbose: bool = False):
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
        if verbose:
            logging.info(f'Saving run config to {conf_out}')
        with open(conf_out, 'w') as f:
            yaml.dump(merged_config, f, sort_keys=False)

    # Return configs or configs + saved paths
    if base_dir is None:
        return space_configs
    else:
        return space_configs, pd.DataFrame({'config_path': config_paths})

def _train(
        adata_p: str, 
        step_model_dir: str, 
        config: dict, 
        cls_label: str = 'cls_label',
        batch_key: str = 'dataset',
        verbose: bool = False,
        **train_kwargs
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
    config[CONF_KEYS.PLAN][NESTED_CONF_KEYS.SCHEDULES_KEY] = config[CONF_KEYS.SCHEDULES]
    # Add plan to train
    config[CONF_KEYS.TRAIN][NESTED_CONF_KEYS.PLAN_KEY] = config[CONF_KEYS.PLAN]
    # Add encoder and decoder args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.ENCODER_KEY] = config[CONF_KEYS.ENCODER]
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.DECODER_KEY] = config[CONF_KEYS.DECODER]
    # Add classifier args to model
    config[CONF_KEYS.MODEL][NESTED_CONF_KEYS.CLS_KEY] = config[CONF_KEYS.CLS]

    # Setup model
    logging.info('Setting up model.')
    # Setup anndata with model
    setup_kwargs = {'batch_key': batch_key, 'labels_key': cls_label}
    setup_kwargs.update(config.get(CONF_KEYS.MODEL_SETUP, {}))
    JEDVI.setup_anndata(
        model_set,
        **setup_kwargs
    )
    model = JEDVI(model_set, **config[CONF_KEYS.MODEL].copy())
    if verbose:
        print(model.module)
    # Set training logger
    config[CONF_KEYS.TRAIN]['logger'] = pl.loggers.TensorBoardLogger(step_model_dir)
    # Train the model
    logging.info(f'Running at: {step_model_dir}')
    model.train(
        data_params=config[CONF_KEYS.DATA].copy(),
        model_params=config[CONF_KEYS.MODEL].copy(),
        train_params=config[CONF_KEYS.TRAIN].copy(),
        **train_kwargs
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

def load_test_data(
        test_adata_p: str, 
        model: nn.Module, 
        incl_unseen: bool = True, 
        verbose: bool = True, 
        label_key: str = 'perturbation', 
        ctrl_key: str = 'control', 
        control_neighbor_threshold: float = 0.0
    ) -> ad.AnnData:
    # Read adata from file
    test_ad = read_adata(test_adata_p)
    # Filter adata
    if control_neighbor_threshold > 0:
        logging.info(f'Filtering test data for min class neighbor proportion of {control_neighbor_threshold}.')
        neighborhood_purity(
            test_ad, 
            label_key=label_key, 
            ignore_class=ctrl_key, 
            new_label_key=label_key, 
            threshold=control_neighbor_threshold
        )
    # Get training adata
    adata = model.adata
    # Get training and testing labels
    train_perturbations = set(adata.obs.perturbation.unique())
    test_perturbations = set(test_ad.obs.perturbation.unique())
    if REGISTRY_KEYS.CLS_EMB_INIT in adata.uns:
        # Set total available embeddings to class embedding keys
        embedding_perturbations = set(
            adata.uns[REGISTRY_KEYS.CLS_EMB_INIT]['labels'].str.replace('[neg;|pos;]', '', regex=True)
        )
    else:
        # Fall back to internal training embeddings
        embedding_perturbations = train_perturbations

    shared = test_perturbations.intersection(train_perturbations)
    unseen = test_perturbations.difference(train_perturbations)
    test_embedding = test_perturbations.intersection(embedding_perturbations)
    if verbose:
        logging.info(f"Found {len(train_perturbations)} trained class embeddings in model.")
        logging.info(f"Found {len(embedding_perturbations)} total class embeddings in model.")
        logging.info(f"Found {len(test_perturbations)} test perturbation(s).")
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


def filter_test_data(test_ad: ad.AnnData, min_ms: float, min_cpp: int):
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
    use_full = None if not incl_unseen else True
    pred_out = model.predict(adata=test_ad, return_latent=True, soft=True, use_full_cls_emb=use_full)
    # Check if model has an embedding classifier or not
    if isinstance(pred_out, tuple):
        predictions, cz, cls_emb_z = pred_out
        # Store in adata
        if cz is not None:
            test_ad.obsm['latent_cz'] = cz
        if cls_emb_z is not None:
            test_ad.uns['cls_emb_z'] = cls_emb_z
    else:
        predictions = pred_out
    test_ad.obs['cls_prediction'] = predictions.columns[predictions.values.argmax(axis=-1)]
    test_ad.obs['cls_score'] = predictions.max(axis=1) - predictions.mean(axis=1)

    return test_ad, predictions


def plot_test_confusion(test_ad: ad.AnnData, plt_dir: str) -> None:
    from src.plotting import plot_confusion

    actual = test_ad.obs.perturbation.values.tolist()
    pred = test_ad.obs.cls_prediction.str.split(';').str[1].values.tolist()
    # Output file
    o = os.path.join(plt_dir, 'test_confusion_cls_label.png')
    plot_confusion(actual, pred, plt_file=o)


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
        plt_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plt_dir, exist_ok=True)
        # Plot confusion matrix
        plot_test_confusion(test_ad, plt_dir=plt_dir)
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
    # Return results
    return top_n_predictions


def compute_top_n_predictions(test_ad, predictions: pd.DataFrame, train_perturbations, n: int = 20, groupby: str = '_dataset', lib = None):
    # Calculate label index in prediction columns per cell
    test_ad.obs['cls_prediction_idx'] = (test_ad.obs.cls_label.values == predictions.columns.values.reshape(-1, 1)).argmax(axis=0)
    labels = test_ad.obs.cls_label.values
    max_idx = np.argsort(predictions, axis=1)
    # Include random predictions
    random_predictions = pd.DataFrame(np.random.random(predictions.shape), columns=predictions.columns, index=predictions.index)
    random_max_idx = np.argsort(random_predictions, axis=1)

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
            # Get top predictions
            top_predictions = max_idx[idx,-top_n:]
            hit_mask = top_predictions == np.array(y)[:, np.newaxis]
            hit_idx = np.argmax(hit_mask, axis=1)
            is_hit = np.any(hit_mask, axis=1).astype(int)
            # Get top random predictions
            top_random_predictions = random_max_idx[idx,-top_n:]
            random_hit_mask = top_random_predictions == np.array(y)[:, np.newaxis]
            is_random_hit = np.any(random_hit_mask, axis=1).astype(int)
            # Summarize results
            stats_per_label = pd.DataFrame({
                'y': np.concatenate([y[is_hit == 1], y[is_hit == 0]]),
                'idx': np.concatenate([(top_n + 1 - hit_idx[is_hit == 1]) / (top_n + 1), np.repeat(0, np.sum(is_hit == 0))]),
                'label': np.concatenate([labels[is_hit == 1], labels[is_hit == 0]]),
                'prediction': np.concatenate([y[is_hit == 1], top_predictions[is_hit == 0][:, -1]]),
                'random_prediction': np.concatenate([y[is_random_hit == 1], top_random_predictions[is_random_hit == 0][:, -1]])
            })
            # Pick actual labels from indices
            stats_per_label['pred_label'] = predictions.columns[stats_per_label.prediction.values.astype(int)]
            stats_per_label['random_pred_label'] = predictions.columns[stats_per_label.random_prediction.values.astype(int)]
            # Transform labels to strings
            actual = stats_per_label.label.str.split(';').str[1].astype(str)
            pred = stats_per_label.pred_label.str.split(';').str[1].astype(str)
            random_pred = stats_per_label.random_pred_label.str.split(';').str[1].astype(str)
            l = predictions.columns.str.split(';').str[1].astype(str)
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
    ax = sns.boxenplot(tmp, x='top_n', y='f1-score', hue='mode',
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
    ax = sns.boxenplot(tmp, x='top_n', y='f1-score', hue='is_training_perturbation',
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
    ax = sns.boxenplot(top_10, x='top_n', y='f1-score', hue=groupby, flier_kws={'s': 10}, legend=True)
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

def update_latent(
        model: nn.Module, 
        test_ad: ad.AnnData, 
        version_dir: str, 
        output_dir: str | None = None,
        plot: bool = True, 
        incl_unseen: bool = True, 
        batch_key: str = 'dataset', 
        orig_group_key: str = 'orig_dataset'
    ):
    # Set output directory
    output_dir = version_dir if output_dir is None else output_dir
    # I/O
    plt_dir = os.path.join(output_dir, 'plots')
    latent = sc.read(os.path.join(version_dir, 'latent.h5ad'))
    # Plot combination of latent spaces
    use_full = None if not incl_unseen else True
    pred_out = model.predict(return_latent=True, soft=True, use_full_cls_emb=use_full)
    # Build index map
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
    
    # Add test set
    test_ad.obs['mode'] = 'test'
    test_z = test_ad.obsm['latent_z']
    test_z = ad.AnnData(test_z, obs=test_ad.obs)
    # Add latent class projection if it exists
    if 'latent_cz' in test_ad.obsm:
        test_z.obsm['cz'] = test_ad.obsm['latent_cz']
    test_z.obsm['soft_predictions'] = test_ad.obsm['soft_predictions']
    # Model has embedding projections
    use_rep = 'X'
    field = 'layer'
    # Add model
    if isinstance(pred_out, tuple):
        _, model_cz, _ = pred_out
        latent.obsm['cz'] = model_cz[idx_map.model_index]
        latent.uns['cls_emb_z'] = test_ad.uns['cls_emb_z']
        use_rep = 'cz'
        field = 'obsm'
    # Combine latent spaces
    latent = ad.concat([latent, test_z], axis=0)
    # TODO: save cz umap as individual .obsm and not overwrite existing latent umap
    if field == 'layer':
        pca_input = latent.X if use_rep == 'X' else latent.layers[use_rep]
    elif field == 'obsm':
        pca_input = latent.obsm[use_rep]
    else:
        raise ValueError(f'Field in adata has to be "layer" or "obsm", got: {field}')
    latent.obsm['cz_pca'] = sc.pp.pca(pca_input)
    sc.pp.neighbors(latent, use_rep='cz_pca')
    sc.tl.umap(latent)
    # Write updated latent space to file
    latent.write_h5ad(os.path.join(output_dir, 'full_latent.h5ad'))
    # Plot updated umap with new testing data
    if plot:
        logging.info(f'Plotting merged umaps.')
        sc.pl.umap(latent, color=[orig_group_key], ncols=1, return_fig=True, show=False)
        plt.savefig(os.path.join(plt_dir, f'full_umap_orig_batch_key.png'), dpi=300, bbox_inches='tight')
        plt.close()
        sc.pl.umap(latent, color=['celltype'], ncols=1, return_fig=True, show=False)
        plt.savefig(os.path.join(plt_dir, f'full_umap_ct.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Plot class labels if less than maximum color palette
        if latent.obs.perturbation.nunique() < 102:
            sc.pl.umap(latent, color=['perturbation'], ncols=1, return_fig=True, show=False)
            plt.savefig(os.path.join(plt_dir, f'full_umap_perturbation.png'), dpi=300, bbox_inches='tight')
            plt.close()

def _get_test_output_dir(**kwargs) -> str:
    blacklist = set(['output_dir', 'plot'])
    kw_strs = [f'{k}:{v}' for k,v in kwargs.items() if k not in blacklist]
    return os.path.join('test', '_'.join(kw_strs))
        
def test(
        model, 
        test_adata_p: str, 
        output_dir: str,
        incl_unseen: bool = True,
        use_fixed_dataset_label: bool = True,
        label_key: str = 'perturbation',
        ctrl_key: str = 'control',
        control_neighbor_threshold: float = 0.0,
        min_ms: float = 4.0,
        min_cpp: int = 10,
        plot: bool = True,
        return_results: bool = False,
        update_latent: bool = False,
        verbose: bool = True
    ):
    logging.info(f'Testing model with: {test_adata_p}')
    # Load test set
    test_ad = load_test_data(test_adata_p, model, incl_unseen=incl_unseen, label_key=label_key, ctrl_key=ctrl_key, control_neighbor_threshold=control_neighbor_threshold)
    # Setup test set labels
    cls_label, batch_key, orig_batch_key = setup_labels(test_ad, test_adata_p, model, use_fixed_dataset_label)
    # Filter test set for labels in class embedding
    test_ad = filter_test_data(test_ad, min_ms, min_cpp)
    # Predict
    test_ad, predictions = run_model_predictions(model, test_ad, cls_label, batch_key, incl_unseen=incl_unseen)
    # Evaluate predictions, plot and update latent space
    train_perturbations = set(model.adata.obs.perturbation.unique())
    # Generate output directory for specific test case
    test_out_dirname = _get_test_output_dir(incl_unseen=incl_unseen, use_fixed_dataset_label=use_fixed_dataset_label, min_ms=min_ms, min_cpp=min_cpp)
    test_out_dir = os.path.join(output_dir, test_out_dirname)
    os.makedirs(test_out_dir, exist_ok=True)
    logging.info(f'Evaluating results for test set. Test output dir: {test_out_dir}')
    top_n_predictions = evaluate_predictions(test_ad, predictions, train_perturbations, output_dir=test_out_dir, plot=plot, groupby=orig_batch_key)
    # Update latent representation and calculate new umaps
    if update_latent:
        logging.info(f'Updating latent space with test data.')
        update_latent(model, test_ad=test_ad, output_dir=test_out_dir, version_dir=output_dir, incl_unseen=incl_unseen, batch_key=batch_key, orig_group_key=orig_batch_key)
    if return_results:
        return top_n_predictions, test_ad

def full_run(
        config_p: str,
        train_p: str,
        model_dir: str,
        test_p: str,
        test_unseen: bool = False,
        load_best: bool = True,
        **kwargs
    ) -> dict:
    # Get config name
    config_name = os.path.basename(config_p).replace('.yaml', '')
    # Set default model output to config directory
    model_dir = model_dir if model_dir is not None else os.path.dirname(config_p)
    # Train model with loaded config file
    train_output = train(adata_p=train_p, config_p=config_p, out_dir=model_dir, **kwargs)
    # Try to load best checkpoint
    if load_best:
        model = JEDVI.load_checkpoint(
            train_output[TRAINING_KEYS.OUTPUT_KEY],
            adata=sc.read(train_p)
        )
    else:
        model = train_output[TRAINING_KEYS.MODEL_KEY]
    # Do not test model
    if test_p is None or not os.path.exists(test_p):
        logging.info(f'Skipping model test since no valid test path is provided.')
        return None
    # Test model with specified test set
    top_n_predictions, _ = test(
        model=model,
        test_adata_p=test_p,
        output_dir=train_output[TRAINING_KEYS.OUTPUT_KEY],
        incl_unseen=False,
        plot=True,
        return_results=True
    )
    top_n_predictions['incl_unseen'] = False
    # Test model with unseen perturbations
    if test_unseen:
        top_n_predictions_unseen, _ = test(
            model=model,
            test_adata_p=test_p,
            output_dir=train_output[TRAINING_KEYS.OUTPUT_KEY],
            incl_unseen=True,
            plot=True,
            return_results=True
        )
        top_n_predictions_unseen['incl_unseen'] = True
        # Concat both predictions
        top_n_predictions = pd.concat((top_n_predictions, top_n_predictions_unseen), axis=0)
    # Add config name to predictions
    top_n_predictions['config_name'] = config_name
    # Save overall testing performance to file
    top_n_predictions.to_csv(os.path.join(train_output[TRAINING_KEYS.OUTPUT_KEY], 'test_top_n_predictions.csv'))
    return top_n_predictions

def get_ouptut_dir(config_p: str, output_base_dir: str | None = None) -> str:
    # Create directory based on the input config name
    if output_base_dir is None:
        output_dir = os.path.dirname(config_p)
    else:
        output_dir = os.path.join(output_base_dir, os.path.basename(config_p).replace('.yaml', ''))
    logging.info(f'Run output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def train(adata_p: str, config_p: str, out_dir: str, **kwargs) -> dict[str: nn.Module | pd.DataFrame | ad.AnnData | str]:
    # Load run config
    config = read_config(config_p)
    # Init run output dir
    step_model_dir = get_ouptut_dir(config_p, output_base_dir=out_dir)
    # Train the model
    model, results, latent = _train(
        adata_p=adata_p, 
        step_model_dir=step_model_dir, 
        config=config,
        **kwargs
    )
    # Get latest lightning directory TODO: replace this by looking up the version via tensorboard logger object
    version_dir = get_latest_tensor_dir(step_model_dir)
    return {
        TRAINING_KEYS.MODEL_KEY: model,
        TRAINING_KEYS.RESULTS_KEY: results,
        TRAINING_KEYS.LATENT_KEY: latent,
        TRAINING_KEYS.OUTPUT_KEY: version_dir
    }

if __name__ == '__main__':
    # Parse cmd line args
    args = parse_args()
    # Full model run
    full_run(config_p=args.config, train_p=args.input, model_dir=args.outdir, test_p=args.test, test_unseen=args.test_unseen)
