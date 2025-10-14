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
