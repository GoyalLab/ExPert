import scvi
import pandas as pd
import os
import logging
import anndata as ad
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from src.performance import get_classification_report
from sklearn.metrics import accuracy_score


def run_scvi(adata: ad.AnnData, n_latent: int, batch_label: str = 'dataset') -> scvi.model.SCVI:
    logging.info(f'Only one dataset, falling back to scvi')
    scvi.model.SCVI.setup_anndata(
        adata, batch_key=batch_label
    )
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(early_stopping=True, max_epochs=100)
    return model

def run_scanvi(
        adata: ad.AnnData, 
        n_latent: int, 
        ct_label: str = 'celltype', 
        batch_label: str = 'dataset',
        scanvi_params: dict = {},
        data_params: dict = {},
        train_params: dict = {}

    ) -> scvi.model.SCANVI:
    from src.train import prepare_scanvi

    scvi.model.SCANVI.setup_anndata(
        adata,
        labels_key=ct_label,
        unlabeled_category='unknown',
        batch_key=batch_label
    )
    # same training parameters
    model = scvi.model.SCANVI(adata, **scanvi_params)
    # get runner and data splitter
    runner, data_splitter = prepare_scanvi(model, data_params, train_params)
    runner()
    return model

def get_xg_data_split(model: scvi.model.SCVI, adata: ad.AnnData, cls_label: str, seed: int = 42) -> dict:
    X = model.get_latent_representation()
    y = adata.obs[cls_label]  # Class labels

    # Perform stratified train-test split
    train_idx, test_idx = train_test_split(
        range(adata.n_obs),
        test_size=0.1,
        stratify=y,           # Preserve class distribution
        random_state=seed
    )
    adata.obs['initial_split'] = None
    adata.obs.iloc[train_idx, (adata.obs.shape[1]-1)] = 'train'
    adata.obs.iloc[test_idx, (adata.obs.shape[1]-1)] = 'test'
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, y_train = X[train_idx], y_encoded[train_idx]
    X_test, y_test = X[test_idx], y_encoded[test_idx]
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'train_idx': train_idx, 'test_idx': test_idx
    }

def get_xg(n_classes: int):
    return xgb.XGBClassifier(
        tree_method='hist',
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        reg_alpha=0.5,        # Increase L1 regularization
        reg_lambda=2.0,       # Increase L2 regularization
        min_child_weight=10,   # Require more observations per leaf
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=10,
    )
    
def run_xgboost(adata: ad.AnnData, n_latent: int, n_classes: int, batch_label: str, cls_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = run_scvi(adata, n_latent, batch_label=batch_label)         # run scvi with latent dimension
    data_split = get_xg_data_split(model, adata, cls_label=cls_label)  # get data split
    xgc = get_xg(n_classes)                                            # create xgboost model
    # Train model
    xgc.fit(
        data_split['X_train'], data_split['y_train'],
        eval_set=[
            (data_split['X_train'], data_split['y_train']),
            (data_split['X_test'], data_split['y_test'])
        ],
        verbose=False
    )
    # Evaluate
    y_pred = xgc.predict(data_split['X_test'])
    accuracy = accuracy_score(data_split['y_test'], y_pred)

    # Detailed classification report
    test_set = adata[data_split['test_idx']]
    summary, report = get_classification_report(test_set, data_split['y_test'], y_pred, cls_label)   
    report['n_latent'] = n_latent
    summary['n_latent'] = n_latent
    return summary, report


def xgboost_scvi_latent_run(
        adata: ad.AnnData, 
        cls_label: str,
        n_classes: int,
        latent_dims: list[int] = [10, 20, 30], 
        batch_label: str = 'dataset',
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    reports = []
    for n_latent in latent_dims:
        logging.info(f'Running scvi w. {n_latent} latent dimensions')
        summary, report = run_xgboost(adata, n_latent, n_classes, batch_label, cls_label)
        
        summaries.append(summary)
        reports.append(report)
    summaries = pd.concat(summaries, axis=0)
    reports = pd.concat(reports, axis=0)
    return summaries, reports
