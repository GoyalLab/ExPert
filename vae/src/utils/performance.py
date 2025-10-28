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
        cls_label: str,
        pred_label: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import classification_report

    class_names = adata.obs[cls_label].unique()
    y = adata.obs[cls_label].values.astype(str)
    y_test = adata.obs[pred_label].values.astype(str)
    report = classification_report(
        y_true=y_test, 
        y_pred=y,
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
        split_key: str = 'split',
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
            context_mask = (split_ad.obs[context_key] == ct)
            # Calculate metrics for each top N prediction to consider
            split_y = split_ad.obs[labels_key].map(label_to_idx).values
            for top_n in np.arange(n) + 1:
                metrics = calculate_metrics_for_group(
                    adata=split_ad, 
                    tmp_mask=context_mask, 
                    y=split_y,
                    max_idx=max_idx[split_mask], 
                    random_max_idx=random_max_idx[split_mask], 
                    predictions=predictions, 
                    top_n=top_n,
                    lib=lib
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
    # Overwrite split labels as random
    random_mask = top_n_predictions['mode']=='random'
    top_n_predictions.loc[random_mask,split_key] = 'random'
    return top_n_predictions

def collect_runs(base_dir: str):
    runs = []
    for root, dirs, files in os.walk(base_dir):
        summary_ps = glob.glob(f'{root}/**/JEDVI_eval_summary.csv', recursive=True)
        if "config.yaml" in files and len(summary_ps) > 0:
            cfg_path = os.path.join(root, "config.yaml")
            summ_path = summary_ps[0]
            runs.append((cfg_path, summ_path))
    print(f"Found {len(runs)} runs.")
    return runs

def parse_run(cfg_path, summ_path):
    # Parse config.yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

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
    val_row = df[(df["split"] == "val") & (df.index == 4)]  # macro avg row (index may vary)
    # safer alternative:
    val_row = df[(df["split"] == "val") & (df.iloc[:,0].astype(str).str.contains("macro avg", case=False))]
    f1_val = float(val_row["f1-score"].values[0]) if not val_row.empty else np.nan

    flat_cfg["f1_val"] = f1_val
    return flat_cfg

def build_dataframe(base_dir: str):
    data = []
    idx = []
    for cfg_path, summ_path in tqdm(collect_runs(base_dir)):
        try:
            row = parse_run(cfg_path, summ_path)
            data.append(row)
            idx.extend([os.path.basename(os.path.dirname(cfg_path))])
        except Exception as e:
            print(f"Error parsing {cfg_path}: {e}")
    df = pd.DataFrame(data, index=idx)
    df = df.dropna(subset=["f1_val"])
    print(f"Final dataset shape: {df.shape}")
    return df

def prepare_features(df, target="f1_val"):
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

    X_combined = np.hstack([X_num, X_cat])
    feature_names = (
        list(num_cols) +
        list(encoder.get_feature_names_out(cat_cols)) if cat_cols else num_cols
    )

    print(f"Encoded {len(num_cols)} numeric and {len(cat_cols)} categorical features.")
    return X_combined, y, feature_names

def ridge_importance(X, y, feature_names):
    model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X, y)
    coefs = pd.Series(model.coef_, index=feature_names).sort_values(key=abs, ascending=False)
    print("Top linear effects:")
    print(coefs.head(10))
    sns.barplot(x=coefs.head(15).values, y=coefs.head(15).index)
    plt.title("Ridge Coefficients (Top 15)")
    plt.tight_layout()
    plt.show()


def rf_importance(X, y, feature_names):
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    # --- 1. Built-in feature importance ---
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(6, 6))
    sns.barplot(x=fi.values[:15], y=fi.index[:15])
    plt.title("Random Forest Feature Importances (Top 15)")
    plt.tight_layout()
    plt.show()

    # --- 2. Permutation importance (more robust) ---
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(6, 6))
    sns.barplot(x=perm_importances.values[:15], y=perm_importances.index[:15])
    plt.title("Permutation Importances (Top 15)")
    plt.tight_layout()
    plt.show()

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

def analyse_grid_run(run_dir: str, pca: bool = False) -> dict:
    """Analyse main performance drivers in grid runs."""
    df = build_dataframe(run_dir)
    X, y, feature_names = prepare_features(df)
    # Sort by performance
    df.sort_values("f1_val", ascending=False, inplace=True)

    # correlation screening
    corr = pd.DataFrame({"feature": feature_names, "corr": [np.corrcoef(X[:,i], y)[0,1] for i in range(X.shape[1])]})
    corr["abs_corr"] = corr["corr"].abs()

    ridge_importance(X, y, feature_names)
    rf = rf_importance(X, y, feature_names)
    if pca:
        pca_plot(X, y)
    return {
        'df': df, 'corr': corr, 'rf': rf
    }
