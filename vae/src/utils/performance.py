import numpy as np
import pandas as pd
import anndata as ad
import gseapy as gp

from typing import Any

from sklearn.metrics import precision_recall_fscore_support


def performance_metric(actual, pred, labels, lib=None, mode='test'):
    # Add pathway overlap
    if lib is not None:
        universe = set(map(str.upper, set(actual) | set(pred)))
        gene2p = build_gene2pathways(universe, lib)
        df_cmp = compare_gene_pairs(actual, pred, gene2p)
        actual = df_cmp.actual
        pred = df_cmp.pathway_predicted
    # Calculate metrics for new predictions
    cls_report = precision_recall_fscore_support(
        actual.values.astype(str),
        pred.values.astype(str),
        labels=labels.values.astype(str),
        average=None, 
        zero_division=np.nan)
    m = pd.DataFrame(cls_report, index=['precision', 'recall', 'f1-score', 'support'], columns=labels).T
    # Set label
    m['mode'] = mode
    return m

def get_classification_report(
        adata: ad.AnnData, 
        y_test: Any, 
        y_train: Any, 
        cls_label: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import classification_report

    class_names = adata.obs[cls_label].unique()
    report = classification_report(y_test, y_train, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    actual_support_map = adata.obs[cls_label].value_counts().reset_index()
    summary = report_df[report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    report_data = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    report_data = report_data.reset_index().merge(actual_support_map, left_on='index', right_on=cls_label)
    report_data['log_count'] = np.log(report_data['count'])
    return summary, report_data


def get_library(library="KEGG_2019_Human", organism="Human") -> dict | None:
    # Load gene pathway library
    try:
        return gp.get_library(name=library, organism=organism)
    except Exception as e:
        print(f"Error loading library '{library}' for organism '{organism}': {e}")
        return None

def build_gene2pathways(genes, library_obj=None):
    """Return dict: GENE -> set({pathways}) using a library from gseapy."""
    if library_obj is None:
        library_obj = get_library()
    wanted = set(g.upper() for g in genes)
    gene2p = {}
    for pathway, members in library_obj.items():
        for g in members:
            gU = g.upper()
            if gU in wanted:
                gene2p.setdefault(gU, set()).add(pathway)
    return gene2p

def compare_gene_pairs(actual_genes, predicted_genes, gene2pathways):
    """Compare per-position pairs: is predicted in any same pathway as actual?"""
    rows = []
    for a, p in zip(actual_genes, predicted_genes):
        aU, pU = a.upper(), p.upper()
        A = gene2pathways.get(aU, set())
        P = gene2pathways.get(pU, set())
        shared = sorted(A & P)
        rows.append({
            "actual": aU,
            "predicted": pU,
            "pathway_predicted": aU if bool(shared) else pU,
            "same_pathway": bool(shared),
            "n_shared": len(shared),
            "shared_pathways": "; ".join(shared),
            "actual_pathways": "; ".join(A),
            "predicted_pathways": "; ".join(P),
        })
    return pd.DataFrame(rows)

def calculate_prediction_stats(
        y: np.ndarray, 
        is_hit: np.ndarray, 
        hit_idx: np.ndarray, 
        top_n: int, 
        labels: np.ndarray, 
        top_predictions: np.ndarray, 
        top_random_predictions: np.ndarray, 
        is_random_hit: np.ndarray, 
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
    """Calculate basic statistics for predictions."""
    stats_per_label = pd.DataFrame({
        'y': np.concatenate([y[is_hit == 1], y[is_hit == 0]]),
        'idx': np.concatenate([(top_n + 1 - hit_idx[is_hit == 1]) / (top_n + 1), np.repeat(0, np.sum(is_hit == 0))]),
        'label': np.concatenate([labels[is_hit == 1], labels[is_hit == 0]]),
        'prediction': np.concatenate([y[is_hit == 1], top_predictions[is_hit == 0][:, -1]]),
        'random_prediction': np.concatenate([y[is_random_hit == 1], top_random_predictions[is_random_hit == 0][:, -1]])
    })
    stats_per_label['pred_label'] = predictions.columns[stats_per_label.prediction.values.astype(int)]
    stats_per_label['random_pred_label'] = predictions.columns[stats_per_label.random_prediction.values.astype(int)]
    return stats_per_label

def calculate_metrics_for_group(
        adata: ad.AnnData,
        tmp_mask: np.ndarray, 
        y: np.ndarray, 
        max_idx: np.ndarray, 
        random_max_idx: np.ndarray, 
        predictions: pd.DataFrame, 
        top_n: int, 
        lib: dict | None
    ) -> pd.DataFrame:
    """Calculate metrics for a specific group and top_n value."""
    idx = np.where(tmp_mask)[0]
    tmp = adata[tmp_mask]
    y = y[idx]
    labels = tmp.obs.cls_label.values
    
    # Get top predictions and hits
    top_predictions = max_idx[idx,-top_n:]
    hit_mask = top_predictions == np.array(y)[:, np.newaxis]
    hit_idx = np.argmax(hit_mask, axis=1)
    is_hit = np.any(hit_mask, axis=1).astype(int)
    
    # Get random predictions and hits
    top_random_predictions = random_max_idx[idx,-top_n:]
    random_hit_mask = top_random_predictions == np.array(y)[:, np.newaxis]
    is_random_hit = np.any(random_hit_mask, axis=1).astype(int)
    
    stats_per_label = calculate_prediction_stats(
        y, is_hit, hit_idx, top_n, labels, top_predictions, 
        top_random_predictions, is_random_hit, predictions
    )
    
    # Transform labels
    actual = stats_per_label.label.astype(str)
    pred = stats_per_label.pred_label.astype(str)
    random_pred = stats_per_label.random_pred_label.astype(str)
    l = predictions.columns.astype(str)
    
    # Calculate metrics
    metrics = pd.concat([
        performance_metric(actual, pred, l, lib=None, mode='test'),
        performance_metric(actual, random_pred, l, lib=None, mode='random')
    ])
    metrics['use_pathway'] = False
    
    if lib is not None:
        pw_metrics = pd.concat([
            performance_metric(actual, pred, l, lib=lib, mode='test'),
            performance_metric(actual, random_pred, l, lib=lib, mode='random')
        ])
        pw_metrics['use_pathway'] = True
        metrics = pd.concat([metrics, pw_metrics])
    
    return metrics[metrics.support > 0].copy()

def compute_top_n_predictions(
        adata: ad.AnnData, 
        split_key: str | None = None,
        context_key: str = 'dataset',
        labels_key: str = 'cls_label',
        ctrl_key: str | None = None,
        n: int = 20, 
        lib: dict | None = None,
        train_perturbations: list[str] | None = None, 
        predictions_key: str = 'soft_predictions'
    ) -> pd.DataFrame:
    """Compute top N predictions with metrics across groups and splits."""
    # Subset adata to ignore control cells in classification
    if ctrl_key is not None and ctrl_key in adata.obs[labels_key].unique():
        _adata = adata[adata.obs[labels_key]!=ctrl_key]
    else:
        _adata = adata
    # Extract soft predictions from adata
    predictions = _adata.obsm[predictions_key]
    # Match actual to prediction label indices
    label_to_idx = {label: idx for idx, label in enumerate(predictions.columns)}
    # Get prediction ranks
    max_idx = np.argsort(predictions, axis=1)
    # Get random prediction ranks
    random_predictions = pd.DataFrame(np.random.random(predictions.shape), columns=predictions.columns, index=predictions.index)
    random_max_idx = np.argsort(random_predictions, axis=1)
    # Collect top n predictions for all groups
    top_n_predictions = []
    splits = [None] if split_key is None else _adata.obs[split_key].unique()
    # Process each data split individually
    for split in splits:
        split_mask = slice(None) if split is None else (_adata.obs[split_key] == split)
        split_ad = _adata[split_mask]
        # Process each context individually
        for ct in split_ad.obs[context_key].unique():
            group_mask = (split_ad.obs[context_key] == ct)
            # Calculate metrics for each top N prediction to consider
            group_y = split_ad.obs[labels_key].map(label_to_idx).values
            for top_n in np.arange(n) + 1:
                metrics = calculate_metrics_for_group(
                    split_ad, group_mask, group_y, max_idx[split_mask], 
                    random_max_idx[split_mask], predictions, top_n, lib
                )
                metrics['top_n'] = top_n
                metrics[context_key] = ct
                if split_key is not None:
                    metrics[split_key] = split
                top_n_predictions.append(metrics)
    # Combine prediction metrics
    top_n_predictions = pd.concat(top_n_predictions, axis=0)
    top_n_predictions['label'] = top_n_predictions.index
    # Add informations about zero-shot predictions
    if train_perturbations is not None:
        top_n_predictions['is_training_perturbation'] = top_n_predictions.label.isin(train_perturbations)
    
    return top_n_predictions
