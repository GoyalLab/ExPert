import os
import glob
import yaml
import numpy as np
import pandas as pd
import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA

from tqdm import tqdm


class PermissiveLoader(yaml.SafeLoader):
    pass

def construct_any(loader, tag_suffix, node):
    try:
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        elif isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        else:
            return loader.construct_scalar(node)
    except Exception:
        return None
    
PermissiveLoader.add_multi_constructor("", construct_any)


def performance_metric(actual, pred, labels, lib=None, mode='test'):
    # Add pathway overlap
    if lib is not None:
        universe = set(map(str.upper, set(actual) | set(pred)))
        gene2p = build_gene2pathways(universe, lib)
        df_cmp = compare_gene_pairs(actual, pred, gene2p)
        actual = df_cmp.actual
        pred = df_cmp.pathway_predicted
    # Transform to string series
    actual = pd.Series(actual, dtype=str)
    pred = pd.Series(pred, dtype=str)
    labels = pd.Series(labels, dtype=str)
    # Calculate metrics for new predictions
    cls_report = precision_recall_fscore_support(
        actual,
        pred,
        labels=labels,
        average=None, 
        zero_division=np.nan)
    m = pd.DataFrame(cls_report, index=['precision', 'recall', 'f1-score', 'support'], columns=labels).T
    # Set label
    m['mode'] = mode
    return m

def get_topk_predictions(
        soft_preds: np.ndarray,
        y_true: np.ndarray | list,
        k: int = 5,
        return_scores: bool = False
    ):
    """
    Convert soft predictions into flattened top-k predictions suitable for sklearn.

    Parameters
    ----------
    soft_preds : np.ndarray, shape (N, C)
        Soft logits/probabilities.
    y_true : array-like, shape (N,)
        Ground-truth class labels.
    k : int
        Number of top-k predictions per cell.
    return_scores : bool
        Whether to return flattened top-k scores.

    Returns
    -------
    y_true_flat : np.ndarray, shape (N*k,)
        True labels, repeated k times.
    y_pred_flat : np.ndarray, shape (N*k,)
        Top-k predicted class indices, flattened.
    y_scores_flat : np.ndarray, shape (N*k,), optional
        Top-k scores aligned to y_pred_flat.
    """

    soft_preds = np.asarray(soft_preds)
    y_true = np.asarray(y_true)

    N, C = soft_preds.shape

    # -------------------------
    # Top-k class indices
    # -------------------------
    topk_idx = np.argsort(soft_preds, axis=1)[:, -k:][:, ::-1]   # (N, k)
    topk_scores = np.take_along_axis(soft_preds, topk_idx, axis=1)  # (N, k)

    # -------------------------
    # Flatten outputs
    # -------------------------
    y_pred_flat = topk_idx.reshape(-1)          # shape (N*k,)
    y_scores_flat = topk_scores.reshape(-1)     # shape (N*k,)
    y_true_flat = np.repeat(y_true, k)          # shape (N*k,)

    if return_scores:
        return y_true_flat, y_pred_flat, y_scores_flat

    return y_true_flat, y_pred_flat

def get_classification_report(
        adata: ad.AnnData, 
        cls_label: str,
        pred_label: str,
        class_names: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import classification_report

    y = adata.obs[cls_label].values.astype(str)
    y_test = adata.obs[pred_label].values.astype(str)
    # Get classification report
    report = classification_report(
        y_true=y, 
        y_pred=y_test,
        labels=y,
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
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

def compute_top_n_predictions(
        adata: ad.AnnData, 
        split_key: str = 'split',
        context_key: str = 'dataset',
        labels_key: str = 'cls_label',
        ctrl_key: str | None = None,
        n: int = 10,
        use_pathways: bool = False,
        train_perturbations: list[str] | None = None, 
        predictions_key: str = 'soft_predictions',
        n_random_samples: int = 10,
        verbose: bool = True,
    ) -> pd.DataFrame:
    from tqdm.auto import tqdm
    """Compute top N predictions with metrics across groups and splits."""
    if ctrl_key is not None and ctrl_key in adata.obs[labels_key].unique():
        _adata = adata[adata.obs[labels_key] != ctrl_key]
    else:
        _adata = adata
    
    predictions = _adata.obsm[predictions_key]
    if use_pathways:
        lib = build_gene2pathways(genes=predictions.columns.values)
    else:
        lib = None
    
    label_to_idx = {label: idx for idx, label in enumerate(predictions.columns)}
    n_cells, n_classes = predictions.shape
    
    # Argsort model predictions once
    max_idx = np.argsort(predictions.values, axis=1)
    
    # Pre-generate all random orderings once
    random_max_idxs = []
    for _ in range(n_random_samples):
        rand = np.random.random((n_cells, n_classes))
        random_max_idxs.append(np.argsort(rand, axis=1))
    
    top_n_predictions = []
    splits = [None] if split_key is None else _adata.obs[split_key].unique()
    if verbose:
        splits = tqdm(splits, desc='Splits', leave=True)
    
    for split in splits:
        split_mask = slice(None) if split is None else (_adata.obs[split_key] == split)
        split_ad = _adata[split_mask]
        
        # Pre-slice for this split
        split_max_idx = max_idx[split_mask]
        split_random_max_idxs = [r[split_mask] for r in random_max_idxs]
        split_y = split_ad.obs[labels_key].map(label_to_idx).values
        
        contexts = split_ad.obs[context_key].unique()
        if verbose:
            contexts = tqdm(contexts, desc=f'Contexts ({split})', leave=False)
        
        for ct in contexts:
            context_mask = (split_ad.obs[context_key] == ct)
            ctx_idx = np.where(context_mask)[0]
            ctx_y = split_y[ctx_idx]
            ctx_labels = split_ad[context_mask].obs[labels_key].values
            
            # Pre-slice model and random predictions for this context
            ctx_max_idx = split_max_idx[ctx_idx]
            ctx_random_max_idxs = [r[ctx_idx] for r in split_random_max_idxs]
            
            top_n_range = np.arange(n) + 1
            if verbose:
                top_n_range = tqdm(top_n_range, desc=f'Top-N ({ct})', leave=False)
            
            for top_n in top_n_range:
                # --- Model predictions ---
                metrics = _compute_metrics_from_sliced(
                    ctx_y, ctx_labels, ctx_max_idx, top_n,
                    predictions, lib, mode='test',
                )
                metrics['top_n'] = top_n
                metrics[context_key] = ct
                if split_key is not None:
                    metrics[split_key] = split
                top_n_predictions.append(metrics)
                
                # --- Random baseline: batch over pre-generated orderings ---
                random_metrics_list = []
                for ctx_rand in ctx_random_max_idxs:
                    rm = _compute_metrics_from_sliced(
                        ctx_y, ctx_labels, ctx_rand, top_n,
                        predictions, lib, mode='random',
                    )
                    random_metrics_list.append(rm)
                
                # Average random metrics
                random_metrics = pd.concat(random_metrics_list)
                numeric_cols = random_metrics.select_dtypes(include=np.number).columns
                random_avg = random_metrics.groupby(random_metrics.index)[numeric_cols].mean()
                for col in random_metrics.columns:
                    if col not in numeric_cols:
                        random_avg[col] = random_metrics_list[0][col]
                
                random_avg['top_n'] = top_n
                random_avg[context_key] = ct
                if split_key is not None:
                    random_avg[split_key] = split
                top_n_predictions.append(random_avg)
    
    top_n_predictions = pd.concat(top_n_predictions, axis=0)
    top_n_predictions['label'] = top_n_predictions.index
    
    if train_perturbations is not None:
        top_n_predictions['is_training_perturbation'] = top_n_predictions.label.isin(train_perturbations)
    
    random_mask = top_n_predictions['mode'] == 'random'
    top_n_predictions.loc[random_mask, split_key] = 'random'
    return top_n_predictions


def _compute_metrics_from_sliced(
    y: np.ndarray,
    labels: np.ndarray,
    max_idx: np.ndarray,
    top_n: int,
    predictions: pd.DataFrame,
    lib: dict | None,
    mode: str = 'test',
) -> pd.DataFrame:
    """Compute metrics from pre-sliced arrays. No adata indexing needed."""
    top_predictions = max_idx[:, -top_n:]
    hit_mask = top_predictions == y[:, np.newaxis]
    is_hit = np.any(hit_mask, axis=1).astype(int)
    
    pred_labels = np.where(is_hit == 1, y, top_predictions[:, -1])
    
    actual = labels.astype(str)
    pred = predictions.columns[pred_labels.astype(int)].astype(str)
    l = predictions.columns.astype(str)
    
    metrics = performance_metric(actual, pred, l, lib=None, mode=mode)
    metrics['use_pathway'] = False
    
    if lib is not None:
        pw_metrics = performance_metric(actual, pred, l, lib=lib, mode=mode)
        pw_metrics['use_pathway'] = True
        metrics = pd.concat([metrics, pw_metrics])
    
    return metrics[metrics.support > 0].copy()

def collect_runs(base_dir: str):
    runs = []
    #for root, _, files in os.walk(base_dir):
    summary_ps = glob.glob(f'{base_dir}/**/*_eval_summary.csv', recursive=True)
    # Add all run versions
    for summ_path in summary_ps:
        # Look for hparams.yaml
        run_dir = os.path.dirname(summ_path)
        _cfg_path = os.path.join(run_dir, 'hparams.yaml')
        if os.path.exists(_cfg_path):
            runs.append((_cfg_path, summ_path))
    print(f"Found {len(runs)} runs.")
    return runs

def parse_run(cfg_path, summ_path, split="test", metric="f1-score"):
    # Parse config.yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=PermissiveLoader)

    # Flatten nested dicts
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_cfg = flatten_dict(cfg)

    # Parse summary.csv
    df = pd.read_csv(summ_path)
    # Select validation F1-score
    val_row = df[(df["split"] == split) & (df.index == 4)]  # macro avg row (index may vary)
    # safer alternative:
    val_row = df[(df["split"] == split) & (df.iloc[:,0].astype(str).str.contains("macro avg", case=False))]
    f1_val = float(val_row[metric].values[0]) if not val_row.empty else np.nan

    split_metric_key = f"{metric}_{split}"
    flat_cfg[split_metric_key] = f1_val
    return flat_cfg

def build_dataframe(base_dir: str, split: str = "test", metric: str = "f1-score"):
    data = []
    idx = []
    metric_key = f"{metric}_{split}"
    for cfg_path, summ_path in tqdm(collect_runs(base_dir)):
        try:
            row = parse_run(cfg_path, summ_path, split=split, metric=metric)
            data.append(row)
            config_id = summ_path.replace(base_dir, '').strip('/').split('/')[0]
            version_id = os.path.basename(os.path.dirname(summ_path))
            run_id = str(config_id) + '_' + str(version_id)
            idx.extend([run_id])
        except Exception as e:
            print(f"Error parsing {cfg_path}: {e}")
    df = pd.DataFrame(data, index=idx)
    df = df.dropna(subset=[metric_key])
    print(f"Final dataset shape: {df.shape}")
    return df

def prepare_features(df, target="f1_test"):
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # --- Step 1: Normalize data types ---
    def serialize_value(v):
        """Convert any value to a hashable string or numeric form."""
        if isinstance(v, (list, dict)):
            return str(v)           # e.g. "[64, 128]" → "64_128"
        elif isinstance(v, bool):
            return int(v)           # True/False → 1/0
        elif v is None:
            return "None"
        else:
            return v

    X = X.applymap(serialize_value)

    # --- Step 2: Separate numeric and categorical columns ---
    cat_cols, num_cols = [], []
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
            num_cols.append(col)
        except (ValueError, TypeError):
            cat_cols.append(col)

    # --- Step 3: Encode ---
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(X[cat_cols]) if cat_cols else np.empty((len(X), 0))
    X_num = scaler.fit_transform(X[num_cols]) if num_cols else np.empty((len(X), 0))

    X_combined = np.nan_to_num(np.hstack([X_num, X_cat]))
    feature_names = (
        list(num_cols) +
        list(encoder.get_feature_names_out(cat_cols)) if cat_cols else num_cols
    )

    print(f"Encoded {len(num_cols)} numeric and {len(cat_cols)} categorical features.")
    return X_combined, y, feature_names

def ridge_importance(X, y, feature_names, plot_dir: str | None = None):
    model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X, y)
    coefs = pd.Series(model.coef_, index=feature_names).sort_values(key=abs, ascending=False)
    coefs = coefs.head(15)[coefs.abs() > 0]
    sns.barplot(x=coefs.values, y=coefs.index)
    plt.title("Ridge Coefficients (Top 15)")
    plt.tight_layout()
    if plot_dir is not None and os.path.exists(plot_dir):
        plot_p = f'{plot_dir}/ridge_importance.png'
        plt.savefig(plot_p, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()

def rf_importance(X, y, feature_names, plot_dir: str | None = None):
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    # --- 1. Built-in feature importance ---
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    fi = fi.head(15)[fi > 0]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title("Random Forest Feature Importances (Top 15)")
    plt.tight_layout()
    if plot_dir is not None and os.path.exists(plot_dir):
        plot_p = f'{plot_dir}/rf_importance.png'
        plt.savefig(plot_p, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()

    # --- 2. Permutation importance (more robust) ---
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    perm_importances = perm_importances.head(15)[perm_importances > 0]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=perm_importances.values, y=perm_importances.index)
    plt.title("Permutation Importances (Top 15)")
    plt.tight_layout()
    if plot_dir is not None and os.path.exists(plot_dir):
        plot_p = f'{plot_dir}/permutation_importance.png'
        plt.savefig(plot_p, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()

    return rf

def pca_plot(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=30)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Validation F1")
    plt.title("Hyperparameter Landscape (PCA projection)")
    plt.tight_layout()
    plt.show()

def analyse_grid_run(
        run_dir: str, 
        pca: bool = False, 
        split: str = "test", 
        metric: str = "f1-score",
        plot_dir: str | None = None
    ) -> dict:
    """Analyse main performance drivers in grid runs."""
    df = build_dataframe(run_dir, split=split, metric=metric)
    metric_key = f"{metric}_{split}"
    X, y, feature_names = prepare_features(df, target=metric_key)
    # Sort by performance
    df.sort_values(metric_key, ascending=False, inplace=True)

    # correlation screening
    corr = pd.DataFrame({"feature": feature_names, "corr": [np.corrcoef(X[:,i], y)[0,1] for i in range(X.shape[1])]})
    corr["abs_corr"] = corr["corr"].abs()
    # Plot ridge importance
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
    ridge_importance(X, y, feature_names, plot_dir=plot_dir)
    rf = rf_importance(X, y, feature_names, plot_dir=plot_dir)
    if pca:
        pca_plot(X, y)
    return {
        'df': df, 'corr': corr, 'rf': rf
    }
