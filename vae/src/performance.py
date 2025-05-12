import numpy as np
import pandas as pd
import anndata as ad

from typing import Iterable, Any


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
