import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import itertools, random
from copy import deepcopy

import torch
import torch.nn as nn
import pytorch_lightning as pl

import scanpy as sc
import anndata as ad

from tqdm import tqdm

from ._statics import CONF_KEYS
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

import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_test_data(test_adata_p: str, model, incl_unseen: bool):
    test_ad = sc.read(test_adata_p)
    adata = model.adata

    train_perturbations = set(adata.obs.perturbation.unique())
    test_perturbations = set(test_ad.obs.perturbation.unique())
    embedding_perturbations = set(
        adata.uns['CLS_EMB_INIT']['labels'].str.replace('[neg;|pos;]', '', regex=True)
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
    if not use_fixed_dataset_label:
        logging.info("Randomly drawing dataset labels from training data.")
        training_datasets = model.adata.obs.dataset.unique()
        np.random.seed(seed)
        ds_names = np.random.choice(training_datasets, test_ad.shape[0])
        test_ad.obs[batch_key] = pd.Categorical(ds_names)
    else:
        logging.info(f"Added {ds_name} as dataset key.")
        test_ad.obs[batch_key] = ds_name

    return cls_label, batch_key


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


def run_model_predictions(model, test_ad, cls_label, batch_key):
    model.setup_anndata(test_ad, labels_key=cls_label, batch_key=batch_key)
    test_ad = model.create_test_data(test_ad)

    if 'cls_embedding' not in test_ad.uns:
        logging.info("Adding class embedding to test set")
        test_ad.uns['cls_embedding'] = model.adata.uns['cls_embedding']

    test_ad.obsm['latent_z'] = model.get_latent_representation(adata=test_ad)
    predictions, cz = model.predict(adata=test_ad, return_latent=True, soft=True)

    test_ad.obs['cls_prediction'] = predictions.columns[predictions.values.argmax(axis=-1)]
    test_ad.obs['cls_score'] = predictions.max(axis=1) - predictions.mean(axis=1)

    return test_ad, predictions


def evaluate_predictions(test_ad, predictions, train_perturbations, output_dir: str, plot: bool):
    _, report = get_classification_report(test_ad, cls_label='cls_label', mode='test')
    report.sort_values('f1-score', ascending=False, inplace=True)
    report['perturbation'] = report.index.str.split(';').str[1]
    report['is_training_perturbation'] = report.perturbation.isin(train_perturbations)

    top_n_predictions = compute_top_n_predictions(test_ad, predictions, train_perturbations)
    top_n_predictions.to_csv(os.path.join(output_dir, 'top_n_predictions_report.csv'))

    if plot:
        plot_f1_distribution(top_n_predictions, test_ad, output_dir)


def compute_top_n_predictions(test_ad, predictions, train_perturbations, n: int = 20):
    y = (test_ad.obs.cls_label.values == predictions.columns.values.reshape(-1, 1)).argmax(axis=0)
    labels = test_ad.obs.cls_label.values
    max_idx = np.argsort(predictions, axis=1)

    top_n_predictions = []
    for top_n in tqdm(np.arange(n) + 1):
        top_predictions = max_idx[:, -top_n:]
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
        random_pred = pd.Series(np.random.choice(predictions.columns, stats_per_label.shape[0])).str.split(';').str[1]

        actual = stats_per_label.label.str.split(';').str[1]
        pred = stats_per_label.pred_label.str.split(';').str[1]
        l = predictions.columns.str.split(';').str[1]

        test_metrics = performance_metric(actual, pred, l, None, mode='test')
        rand_metrics = performance_metric(actual, random_pred, l, None, mode='random')
        metrics = pd.concat((test_metrics, rand_metrics), axis=0)
        metrics = metrics[metrics.support > 0].copy()
        metrics['top_n'] = top_n
        top_n_predictions.append(metrics)

    top_n_predictions = pd.concat(top_n_predictions, axis=0)
    top_n_predictions['perturbation'] = top_n_predictions.index
    top_n_predictions['is_training_perturbation'] = top_n_predictions.perturbation.isin(train_perturbations)

    return top_n_predictions


def plot_f1_distribution(top_n_predictions, test_ad, output_dir: str):
    no_random = top_n_predictions[top_n_predictions['mode'] != 'random'].copy()
    n_classes = test_ad.obs.cls_label.nunique()

    plt.figure(dpi=300, figsize=(10, 5))
    sns.boxenplot(no_random, x='top_n', y='f1', hue='is_training_perturbation',
                  palette=['#7900d7', '#3274a1'])
    plt.xlabel('Number of top predictions')
    plt.ylim([0.0, 1.0])
    plt.ylabel('F1-score per class')
    plt.title(f'Test F1-score distribution over top predictions (N={n_classes})', pad=20)
    plt.legend(title="Training Perturbation", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plots", "test_f1.svg"))
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
    ):
    test_ad = load_test_data(test_adata_p, model, incl_unseen)
    cls_label, batch_key = setup_labels(test_ad, test_adata_p, model, use_fixed_dataset_label)
    test_ad = filter_test_data(test_ad, min_ms, min_cpp)
    test_ad, predictions = run_model_predictions(model, test_ad, cls_label, batch_key)

    train_perturbations = set(model.adata.obs.perturbation.unique())
    top_n_predictions = evaluate_predictions(test_ad, predictions, train_perturbations, output_dir, plot)
    # Save results to file
    top_n_predictions.to_csv(os.path.join(output_dir, 'top_n_predictions_report.csv'))


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
