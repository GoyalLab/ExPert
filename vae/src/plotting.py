import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import anndata as ad
import logging
import scanpy as sc
from typing import List
import scipy.sparse as sp

from src.utils.constants import MODULE_KEYS


def get_latest_tensor_dir(d: str) -> str:
    ll_key = 'lightning_logs'
    if not os.path.exists(d):
        raise ValueError(f'Directory {d} does not exist.')
    if not os.path.isdir(d):
        raise ValueError(f'Path {d} is not a directory.')
    if ll_key not in os.listdir(d):
        raise ValueError(f'Directory is not a lighning log directory.')
    if len(os.path.join(d, ll_key)) == 0:
        raise ValueError(f'Directory is empty')
    version_dirs = os.listdir(os.path.join(d, ll_key))
    version_dir_df = pd.DataFrame(version_dirs, columns=['version_dir'])
    version_dir_df = version_dir_df[~version_dir_df['version_dir'].str.startswith('.')].copy()          # exclude hidden directories
    version_dir_df['version'] = version_dir_df.version_dir.str.split('_').str[-1].astype(int)
    latest_version_dir = version_dir_df.sort_values('version', ascending=False).head(1).version_dir.values[0]
    return os.path.join(d, ll_key, latest_version_dir)


def plot_confusion(y_true, y_pred, figsize=(10, 8), hm_kwargs={'annot': False}, verbose=False, plt_file=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Get class labels (for multiclass classification, this will correspond to unique labels)
    class_labels = np.unique(y_true)
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Print the results
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cm_percentage, xticklabels=class_labels, yticklabels=class_labels, **hm_kwargs)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if plt_file is None:
        plt.show()
    else:
        plt.savefig(plt_file, dpi=300, bbox_inches='tight')
    plt.close()

def get_model_latent(model, mode) -> ad.AnnData:
    if mode == 'train':
        idx = model.train_indices
    elif mode == 'val':
        idx = model.validation_indices
    elif mode == 'test':
        idx = model.test_indices
    else:
        raise ValueError(f'Model does not have indices for mode: {mode}, has to be one of [train, val, test]')
    lr = model.get_latent_representation(indices=idx)
    if isinstance(lr, dict):
        z = lr[MODULE_KEYS.Z_KEY]
        zg = lr[MODULE_KEYS.ZG_KEY]
    else:
        z = lr
        zg = None
    latent = ad.AnnData(z)
    latent.obs = model.adata.obs.iloc[idx,:].copy()
    latent.obs[MODULE_KEYS.PREDICTION_KEY] = model.predict(indices=idx)
    if zg is not None:
        latent.obsm[MODULE_KEYS.ZG_KEY] = zg
    return latent

def calc_umap(adata: ad.AnnData, rep='X') -> None:
    logging.info(f'Calculating latent neighbors')
    sc.pp.neighbors(adata, use_rep=rep)
    logging.info(f'Calculating latent umap')
    sc.tl.umap(adata)

def add_labels(summary: pd.DataFrame, cls_labels: List[str]) -> None:
    # Add labels to results
    summary.reset_index(names='cls_label', inplace=True)
    label_df = summary.cls_label.str.split(';', expand=True)
    label_df.columns = cls_labels
    summary.loc[:,cls_labels] = label_df

def get_classification_report(latent: ad.AnnData, cls_label: str, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import classification_report

    report = classification_report(
        latent.obs[cls_label],
        latent.obs[MODULE_KEYS.PREDICTION_KEY],
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()

    summary = report_df[report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
    report_data = report_df[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
    report_data['log_count'] = np.log(report_data['support'])
    report_data['mode'] = mode
    summary['mode'] = mode
    return summary, report_data

def plot_performance_support_corr(summary: pd.DataFrame, o: str):
    sns.kdeplot(summary, x='log_count', y='f1-score', hue='mode')
    plt.xlabel('Class support (log)')
    plt.ylabel('Macro f1-score')
    plt.savefig(o, dpi=300, bbox_inches='tight')

def plot_model_results_mode(latent: ad.AnnData, mode: str, batch_key: str, cls_labels: list[str], plt_dir: str, cm: bool = True, add_key: str = '') -> None:
    for i, cl in enumerate([*cls_labels, batch_key]):
        logging.info(f'Plotting {mode} for label: {cl}')
        sc.pl.umap(latent, color=cl, return_fig=True, show=False)
        plt.savefig(os.path.join(plt_dir, f'{mode}_umap{add_key}_{cl}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # plot confusion matrix
        if cl != batch_key and cm:
            cm_p = os.path.join(plt_dir, f'{mode}_cm_{cl}.png')
            yt = latent.obs[cl]
            yp = latent.obs[MODULE_KEYS.PREDICTION_KEY].str.split(';').str[i]
            plot_confusion(yt, yp, plt_file=cm_p)
    if cm:
        # plot confusion matrix for exact label
        cm_p = os.path.join(plt_dir, f'{mode}_cm_cls_label.png')
        yt = latent.obs['cls_label']
        yp = latent.obs[MODULE_KEYS.PREDICTION_KEY]
        plot_confusion(yt, yp, plt_file=cm_p)

def predict_novel(model, adata: ad.AnnData, cls_label: str, out_dir: str | None = None, batch_key: str = 'dataset', key: str = 'report', mode: str = 'test', emb_key: str = 'gene_embedding', soft: bool = False):
    adata.X = sp.csr_matrix(adata.X)
    model.setup_anndata(adata, batch_key=batch_key, labels_key=cls_label, unlabeled_category='unknown', gene_emb_obsm_key=emb_key)
    adata.obsm['latent_z'] = model.get_latent_representation(adata=adata, ignore_embedding=emb_key is None)
    sc.pp.neighbors(adata, use_rep='latent_z')
    sc.tl.umap(adata)
    # run the trained model with novel data
    adata.obs[MODULE_KEYS.PREDICTION_KEY] = model.predict(adata=adata, ignore_embedding=emb_key is None, soft=soft)
    test_summary, test_report = get_classification_report(adata, cls_label=cls_label, mode='test')
    # add reports to adata
    adata.uns[key] = {
        'summary': test_summary, 'report': test_report
    }
    if out_dir is not None:
        # Save to file
        test_summary.to_csv(os.path.join(out_dir, f'{mode}_summary.csv'))
        test_report.to_csv(os.path.join(out_dir, f'{mode}_report.csv'))
        adata.write_h5ad(os.path.join(out_dir, f'{mode}_adata.h5ad'))

def get_model_results(
        model, 
        cls_labels: list[str], 
        log_dir: str, 
        modes: list[str] = ['train', 'val'], 
        test_adata: ad.AnnData | None = None,
        save: bool = True, 
        plot: bool = False
    ) -> tuple[pd.DataFrame, ad.AnnData]:
    version_dir = get_latest_tensor_dir(log_dir)
    plt_dir = os.path.join(version_dir, 'plots')
    os.makedirs(plt_dir, exist_ok=True)
    # plot for each mode
    summaries = []
    class_reports = []
    latents = []
    for mode in modes:
        logging.info(f'Processing {mode} set')
        latent = get_model_latent(model, mode).copy()
        batch_key = model.registry_['setup_args']['batch_key']
        cls_label = model.registry_['setup_args']['labels_key']
        summary, class_report = get_classification_report(latent, cls_label, mode)
        summaries.append(summary)
        class_reports.append(class_report)
        # Add classification labels to class performance report
        add_labels(class_report, cls_labels)
        if plot:
            calc_umap(latent)
            plot_model_results_mode(latent, mode, batch_key, cls_labels, plt_dir)
            plot_performance_support_corr(class_report, os.path.join(plt_dir, f'{mode}_support_corr.png'))
            if 'zg' in latent.obsm:
                calc_umap(latent, rep='zg')
                plot_model_results_mode(latent, mode, batch_key, cls_labels, plt_dir, cm=False, add_key='zg')
        latent.obs['mode'] = mode
        latents.append(latent)
    summaries = pd.concat(summaries, axis=0)
    class_reports = pd.concat(class_reports, axis=0)
    latents = ad.concat(latents, axis=0)
    if save:
        logging.info(f'Saving results to: {version_dir}')
        latents.write_h5ad(os.path.join(version_dir, 'latent.h5ad'))
        model.save(dir_path=os.path.join(version_dir, 'model'), save_anndata=False, overwrite=True)
        summaries.to_csv(os.path.join(version_dir, 'summary.csv'))
        class_reports.to_csv(os.path.join(version_dir, 'report.csv'))
    if test_adata is not None:
        # Predict on novel data
        logging.info('Running model with novel test data.')
        predict_novel(model, test_adata, cls_label, cls_labels, out_dir=version_dir)
        plot_model_results_mode(test_adata, mode='test', batch_key='dataset', cls_labels=cls_labels, plt_dir=plt_dir)
    return class_reports, latents
