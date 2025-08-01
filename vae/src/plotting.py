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
import torch.nn as nn

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
    plt.close()

def plot_soft_predictions(model: nn.Module, adata: ad.AnnData | None = None, mode: str = 'val', plt_dir: str | None = None, n: int = 10) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support
    from tqdm import tqdm

    # Split into sets
    if mode == 'train':
        idx = model.train_indices
    elif mode == 'val':
        idx = model.validation_indices
    elif mode == 'test':
        idx = None

    if adata is None:
        # Fall back to model's adata
        adata = model.adata
    else:
        # Validate new adata
        model._validate_anndata(adata)
        idx = np.arange(adata.shape[0])

    data = adata[idx].copy() if idx is not None else adata
    data.obs['idx'] = idx
    data = data[data.obs.perturbation!='control']
    soft_predictions = model.predict(indices=data.obs['idx'].values, soft=True)
    # Plot wo ctrl
    n_labels = adata.obs.perturbation.nunique()
    y = data.obs._scvi_labels.values
    labels = data.obs.cls_label.values
    max_idx = np.argsort(soft_predictions, axis=1)
    # Look at top N predictions (can be useful for pathways etc.)
    top_n_predictions = []
    for top_n in tqdm(np.arange(1, n+1)):
        top_predictions = max_idx[:,-top_n:]
        hit_mask = top_predictions == np.array(y)[:, np.newaxis]
        hit_idx = np.argmax(hit_mask, axis=1)
        is_hit = np.any(hit_mask, axis=1).astype(int)
        # Default to actual prediction if label is not in top n predictions
        stats_per_label = pd.DataFrame(
            {
                'y': np.concatenate([y[is_hit==1], y[is_hit==0]]),
                'idx': np.concatenate([(top_n+1-hit_idx[is_hit==1])/(top_n+1), np.repeat(0, np.sum(is_hit==0))]),
                'label': np.concatenate([labels[is_hit==1], labels[is_hit==0]]),
                'prediction': np.concatenate([y[is_hit==1], top_predictions[is_hit==0][:,-1]])
            }
        )
        # Calculate metrics for new predictions
        precision, recall, f1, support = precision_recall_fscore_support(stats_per_label.y, stats_per_label.prediction, average=None)
        
        # Add to dataframe
        metrics = pd.DataFrame(
            {
                'y': stats_per_label.y.unique(), 'cls_label': stats_per_label.label.unique(),
                'precision': precision[:n_labels], 'recall': recall[:n_labels], 'f1': f1[:n_labels], 'support': support[:n_labels], 'top_n': top_n
            }
        )
        metrics = metrics.merge(stats_per_label.groupby('y')['idx'].mean(), left_on='y', right_index=True)
        top_n_predictions.append(metrics)
    top_n_predictions = pd.concat(top_n_predictions, axis=0)
    top_n_predictions['mode'] = mode
    # Return results without plotting them
    if plt_dir is None:
        return top_n_predictions
    # Plot results
    f1_p = os.path.join(plt_dir, f'{mode}_f1_top_{n}_predictions_wo_ctrl.svg')

    plt.figure(dpi=120)
    top_n_predictions_no_ctrl = top_n_predictions[~top_n_predictions.cls_label.str.endswith('control')]
    ax = sns.boxplot(top_n_predictions_no_ctrl, x='top_n', y='f1', hue='top_n', legend=False)
    # Display means on top of boxes
    means = top_n_predictions_no_ctrl.groupby('top_n')['f1'].mean()
    for i, top_n_value in enumerate(sorted(top_n_predictions['top_n'].unique())):
        ax.text(i, 1.0, f'{means[top_n_value]:.2f}', 
                horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    plt.xlabel('Number of top predictions')
    plt.ylim([0.0, 1.0])
    plt.ylabel('F1-score per class')
    plt.title(f'Validation F1-score distribution over top predictions (N={model.summary_stats.n_labels})', pad=20)
    plt.savefig(f1_p, dpi=300, bbox_inches='tight')
    return top_n_predictions

def plot_model_results_mode(latent: ad.AnnData, mode: str, batch_key: str, cls_labels: list[str], plt_dir: str, cm: bool = True, add_key: str = '') -> None:
    for i, cl in enumerate([*cls_labels, batch_key]):
        logging.info(f'Plotting {mode} for label: {cl}')
        sc.pl.umap(latent, color=cl, return_fig=True, show=False)
        plt.savefig(os.path.join(plt_dir, f'{mode}_umap{add_key}_{cl}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # plot confusion matrix
        if cl != batch_key and cm:
            cm_p = os.path.join(plt_dir, f'{mode}_cm_{cl}.svg')
            yt = latent.obs[cl]
            yp = latent.obs[MODULE_KEYS.PREDICTION_KEY].str.split(';').str[i]
            plot_confusion(yt, yp, plt_file=cm_p)
    if cm:
        # plot confusion matrix for exact label
        cm_p = os.path.join(plt_dir, f'{mode}_cm_cls_label.svg')
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

def plot_cls_masks(latents: ad.AnnData, plt_dir: str, max_classes: int = 100):
    pp_plt_dir = os.path.join(plt_dir, 'per_perturbation')
    os.makedirs(pp_plt_dir, exist_ok=True)
    pal = ['#919aa1', '#94c47d', '#ff7f0f']
    # Plot train and val for every perturbation
    for p in latents.obs.cls_label.unique()[:max_classes]:
        latents.obs['mask'] = latents.obs['mode'].values.tolist()
        latents.obs.loc[latents.obs.cls_label!=p, 'mask'] = 'other'
        latents.obs['mask'] = pd.Categorical(latents.obs['mask'])
        sc.pl.umap(latents, title=p, color='mask', palette=pal, return_fig=True, show=False)
        plt.savefig(os.path.join(pp_plt_dir, f'{p}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_interactive_latent(latent: ad.AnnData, report: pd.DataFrame | None = None, color_options: list[str] = ['perturbation', 'dataset', 'mode'], deep_mode: bool = True):
    import plotly.graph_objects as go
    import pandas as pd
    import plotly.io as pio
    pio.renderers.default = "notebook"

    # Extract umap data from latent
    umap = pd.DataFrame(latent.obsm['X_umap'], columns=['UMAP_1', 'UMAP_2'])
    umap = pd.concat([umap, latent.obs.reset_index()], axis=1)
    # Define color options
    default_color = color_options[0]
    hover_info_cols = ['perturbation', 'celltype', 'mode', 'dataset', 'mixscale_score']
    # Add classification validation performance to plot
    if report is not None and 'val-f1-score' not in umap.columns:
        tmp = report.loc[report['mode']=='val',['cls_label', 'f1-score']].copy()
        tmp['val-f1-score'] = tmp['f1-score'].values
        umap = umap.merge(tmp[['cls_label', 'val-f1-score']], on='cls_label', how='left')
        hover_info_cols.append('val-f1-score')

    # Generate hovertemplate
    hovertemplate = "<br>".join(
        f"{col}: "+"%{customdata[" + f"{i}" + "]}"
        for i, col in enumerate(hover_info_cols)
    ) + "<extra></extra>"

    # Create figure with traces for each color option (one per label)
    fig = go.Figure()

    for i, label in enumerate(color_options):
        for val in umap[label].unique():
            df_subset = umap[umap[label] == val]
            # Add trace for each coloring option
            fig.add_trace(go.Scattergl(
                x=df_subset['UMAP_1'],
                y=df_subset['UMAP_2'],
                mode='markers',
                marker=dict(size=4, opacity=0.5),
                name=str(val),
                visible=(i == 0),  # show only first color group at start
                hovertext=df_subset[label],
                legendgroup=val,
                hovertemplate=hovertemplate,
                customdata=df_subset[hover_info_cols].values
            ))

    # Create dropdown menu for color switch
    n_labels = [len(umap[label].unique()) for label in color_options]
    visibility_blocks = []
    i = 0
    for n in n_labels:
        vis = [False] * sum(n_labels)
        vis[i:i + n] = [True] * n
        visibility_blocks.append(vis)
        i += n

    # Add dropdown
    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=[
                dict(label=label,
                    method="update",
                    args=[{"visible": visibility_blocks[i]},
                        {"title": f"Color by {label}"}])
                for i, label in enumerate(color_options)
            ],
            x=1.5,
            y=1.0
        )],
        title=f"Color by {default_color}",
        xaxis_title="UMAP_1",
        yaxis_title="UMAP_2"
    )
    fig.show()


def get_model_results(
        model, 
        cls_labels: list[str], 
        log_dir: str, 
        modes: list[str] = ['train', 'val'], 
        test_adata: ad.AnnData | None = None,
        save: bool = True, 
        plot: bool = False,
        max_classes: int = 200,
        save_ad: bool = False,
    ) -> tuple[pd.DataFrame, ad.AnnData]:
    version_dir = get_latest_tensor_dir(log_dir)
    plt_dir = os.path.join(version_dir, 'plots')
    os.makedirs(plt_dir, exist_ok=True)
    # plot for each mode
    summaries = []
    class_reports = []
    latents = []
    soft_predictions = []
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
            plot_performance_support_corr(class_report, os.path.join(plt_dir, f'{mode}_support_corr.svg'))
            soft_predictions_mode = plot_soft_predictions(model, mode=mode, plt_dir=plt_dir)
            soft_predictions.append(soft_predictions_mode)
            if 'zg' in latent.obsm:
                calc_umap(latent, rep='zg')
                plot_model_results_mode(latent, mode, batch_key, cls_labels, plt_dir, cm=False, add_key='zg')
        latent.obs['mode'] = mode
        latents.append(latent)
    summaries = pd.concat(summaries, axis=0)
    class_reports = pd.concat(class_reports, axis=0)
    latents = ad.concat(latents, axis=0)
    soft_predictions = pd.concat(soft_predictions, axis=0)
    # Calculate overall latent
    sc.pp.neighbors(latents, use_rep='X')
    sc.tl.umap(latents)
    if plot and max_classes > 0:
        logging.info('Plotting class masks for run.')
        plot_cls_masks(latents, plt_dir=plt_dir, max_classes=max_classes)
    if save:
        logging.info(f'Saving results to: {version_dir}')
        latents.write_h5ad(os.path.join(version_dir, 'latent.h5ad'))
        model.save(dir_path=os.path.join(version_dir, 'model'), save_anndata=save_ad, overwrite=True)
        summaries.to_csv(os.path.join(version_dir, 'summary.csv'))
        class_reports.to_csv(os.path.join(version_dir, 'report.csv'))
        soft_predictions.to_csv(os.path.join(version_dir, 'soft_predictions.csv'))
    if test_adata is not None:
        # Predict on novel data
        logging.info('Running model with novel test data.')
        predict_novel(model, test_adata, cls_label, cls_labels, out_dir=version_dir)
        plot_model_results_mode(test_adata, mode='test', batch_key='dataset', cls_labels=cls_labels, plt_dir=plt_dir)
    return class_reports, latents
