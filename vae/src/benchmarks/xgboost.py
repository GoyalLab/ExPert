import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import scipy.sparse as sp
import logging
import argparse


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_and_prepare(
        adata_path, 
        labels_key='perturbation', 
        split_key='split',
        cpm=False,
        log1p=False,
        scale=False
    ):
    """Load adata and extract splits."""
    log.info(f'Loading data from {adata_path}')
    adata = sc.read_h5ad(adata_path)
    if cpm:
        log.info(f'Applying CPM transformation')
        sc.pp.normalize_total(adata, target_sum=1e6)
    if log1p:
        log.info(f'Applying Log1p transformation')
        sc.pp.log1p(adata)
    if scale:
        log.info(f'Applying scale transformation')
        sc.pp.scale(adata)
    
    # Extract X
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(adata.obs[labels_key].values)
    
    # Extract splits
    splits = adata.obs[split_key].values
    
    train_idx = np.where(splits == 'train')[0]
    val_idx = np.where(splits == 'val')[0]
    test_idx = np.where(splits == 'test')[0]
    ood_test_idx = np.where(splits == 'ood_test')[0]
    
    log.info(f'Train: {len(train_idx)}, Val: {len(val_idx)}, ID-Test: {len(test_idx)}, OOD-Test: {len(ood_test_idx)}')
    log.info(f'Classes: {len(le.classes_)}')
    
    return {
        'X': X, 
        'y': y, 
        'train_idx': train_idx, 
        'val_idx': val_idx, 
        'test_idx': test_idx, 
        'ood_test_idx': ood_test_idx, 
        'le': le, 
        'adata': adata
    }

def mask_inactive_genes(X, fill_value=np.nan):
    """Set genes that are 0 across all cells to fill_value."""
    inactive = (X == 0).all(axis=0)
    X = X.copy()
    X[:, inactive] = fill_value
    return X

def train_and_evaluate(
    input_dict,
    dataset_key='dataset',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=20,
    n_jobs=-1,
):
    """Train XGBoost and evaluate on val/test."""
    # Unpack input dictionary
    adata = input_dict['adata']
    X = input_dict['X']
    y = input_dict['y']
    le = input_dict['le']
    train_idx = input_dict['train_idx']
    val_idx = input_dict['val_idx']
    test_idx = input_dict['test_idx']
    ood_test_idx = input_dict['ood_test_idx']
    # Get data splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_ood_test, y_ood_test = X[ood_test_idx], y[ood_test_idx]
    # Mask inactive genes per split
    X_train = mask_inactive_genes(X_train)
    X_val = mask_inactive_genes(X_val)
    X_test = mask_inactive_genes(X_test)
    X_ood_test = mask_inactive_genes(X_ood_test)
    
    n_classes = len(le.classes_)
    
    log.info(f'Training XGBoost with {n_jobs} jobs...')
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softprob',
        num_class=n_classes,
        tree_method='hist',
        n_jobs=n_jobs,
        eval_metric='mlogloss',
        early_stopping_rounds=early_stopping_rounds,
        verbosity=1,
    )
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    
    log.info(f'Best iteration: {model.best_iteration}')
    
    # Evaluate
    results = {}
    for split_name, X_split, y_split, split_idx in [
        ('val', X_val, y_val, val_idx),
        ('test', X_test, y_test, test_idx),
        ('ood_test', X_ood_test, y_ood_test, ood_test_idx),
    ]:
        # Skip empty splits
        if len(split_idx) == 0:
            continue
        # Predict split performance
        y_pred = model.predict(X_split)
        
        metrics = {
            'accuracy': accuracy_score(y_split, y_pred),
            'f1_macro': f1_score(y_split, y_pred, average='macro'),
            'f1_weighted': f1_score(y_split, y_pred, average='weighted'),
        }
        
        log.info(f'\n--- {split_name.upper()} ---')
        log.info(f'Accuracy: {metrics["accuracy"]:.4f}')
        log.info(f'F1 macro: {metrics["f1_macro"]:.4f}')
        log.info(f'F1 weighted: {metrics["f1_weighted"]:.4f}')
        
        # Per-dataset breakdown
        if dataset_key in adata.obs.columns:
            datasets = adata.obs[dataset_key].values[split_idx]
            log.info(f'\nPer-dataset {split_name} F1:')
            for ds in np.unique(datasets):
                ds_mask = datasets == ds
                if ds_mask.sum() < 5:
                    continue
                ds_f1 = f1_score(y_split[ds_mask], y_pred[ds_mask], average='macro')
                metrics[f'f1_{ds}'] = ds_f1
                log.info(f'  {ds}: {ds_f1:.4f} (n={ds_mask.sum()})')
        
        results[split_name] = metrics
    
    return model, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata', type=str, required=True)
    parser.add_argument('--labels_key', type=str, default='perturbation')
    parser.add_argument('--split_key', type=str, default='split')
    parser.add_argument('--dataset_key', type=str, default='ds_label')
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--cpm', action='store_true')
    parser.add_argument('--log1p', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    # Collect input data for training and splits
    input_dict = load_and_prepare(
        args.adata, args.labels_key, args.split_key,
        cpm=args.cpm, log1p=args.log1p, scale=args.scale
    )
    # Train model and evaluate different data splits
    model, results = train_and_evaluate(
        input_dict,
        dataset_key=args.dataset_key,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        n_jobs=args.n_jobs
    )
    # Save results to file if given
    if args.output:
        if os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
            of = os.path.join(args.output, 'results.csv')
        else:
            pard = os.path.dirname(args.output)
            os.makedirs(pard, exist_ok=True)
            of = args.output
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(of)
        log.info(f'Results saved to {args.output}')
    # Return results
    return model, results


if __name__ == '__main__':
    main()
