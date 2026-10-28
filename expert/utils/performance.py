# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

import yaml
import functools
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
from typing import Literal
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm
import logging


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

def weighted_f1_per_class(
        y_true,
        y_pred,
        weights,
        n_classes: int,
        ignore_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Per-class weighted F1 and weighted support, computed via sklearn.

    Classes with no weighted support get NaN (via `zero_division=np.nan`),
    so downstream macro averaging with `np.nanmean` skips them. Matches the
    semantics used in `_get_model_classification_report`.

    Parameters
    ----------
    y_true, y_pred : array-like of int class ids (torch tensors also accepted).
    weights : array-like of unsigned sample weights.
    n_classes : number of classes; labels used are range(n_classes).
    ignore_index : drop samples whose true label equals this.

    Returns
    -------
    f1_per_class : np.ndarray shape (n_classes,), NaN for empty classes.
    weighted_support : np.ndarray shape (n_classes,) — Σ w_i · 1[y_i = c].
    """
    from sklearn.metrics import precision_recall_fscore_support
    # Accept torch or numpy inputs uniformly
    def _to_np(x, dtype):
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype).ravel()

    y_true = _to_np(y_true, np.int64)
    y_pred = _to_np(y_pred, np.int64)
    weights = np.abs(_to_np(weights, np.float64))

    if ignore_index is not None:
        keep = y_true != ignore_index
        y_true = y_true[keep]
        y_pred = y_pred[keep]
        weights = weights[keep]

    labels = np.arange(n_classes)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None,
        sample_weight=weights, zero_division=np.nan,
    )
    # Weighted support: Σ w_i · 1[y_i = c] (sklearn's support is unweighted count)
    support = np.zeros(n_classes, dtype=np.float64)
    for c in labels:
        support[c] = weights[y_true == c].sum()
    return f1, support


def weighted_f1_macro(
        y_true,
        y_pred,
        weights,
        n_classes: int,
        ignore_index: int | None = None,
    ) -> float:
    """Class-weighted F1: per-class F1 (computed with sample weights on cells)
    aggregated by weighted average using each class's total |w| as the class
    weight. High-effect classes dominate the score — a plain ``nanmean`` over
    classes would dilute the cell-weighting because every class gets one vote
    regardless of how much real perturbation effect it carries.

    Classes with NaN F1 or zero weighted support are skipped (no information
    about them in the cell-weighted view). If every class is skipped, returns
    0.0 rather than NaN to keep this safe as a logged scalar metric.
    """
    f1, support = weighted_f1_per_class(y_true, y_pred, weights, n_classes, ignore_index)
    keep = ~np.isnan(f1) & (support > 0)
    if not keep.any():
        return 0.0
    return float(np.average(f1[keep], weights=support[keep]))


def class_similarity_to_trained(
    cls_emb,
    cls_labels,
    trained_labels,
    aggregator: Literal['max', 'mean', 'topk_mean'] = 'max',
    topk: int = 5,
) -> pd.DataFrame:
    """Per-class cosine similarity from each class to the trained class set.

    Use the **raw** class embedding here (the LLM space the model was trained
    against), not the adapter projection — the adapter is a linear map that
    can collapse semantically distinct classes onto similar points, washing
    out the "is this novel?" signal we want to quantify.

    Parameters
    ----------
    cls_emb
        ``(n_classes, dim)`` embedding matrix. Numpy / torch / DataFrame ok.
    cls_labels
        ``(n_classes,)`` label per row of ``cls_emb`` in the same order.
    trained_labels
        Labels the model has seen during training (``self.idx_to_label``).
    aggregator
        How to reduce the per-trained-class similarities to a single score:
          - ``max``         : nearest-neighbor sim (least conservative; usual choice).
          - ``mean``        : average sim across the whole trained set.
          - ``topk_mean``   : mean over top-k nearest trained classes (smooths
                              outliers — useful when the trained class set
                              has dense clusters).

    Returns
    -------
    pd.DataFrame indexed by class label with columns:
      - ``is_trained``           : True if label was seen during training.
      - ``similarity_to_trained``: aggregated cosine similarity (NaN for
                                   trained classes — their score is trivially
                                   1.0 and would confuse downstream binning).
      - ``nearest_trained``      : label of the nearest trained class.
    """
    # Accept torch / DataFrame inputs
    if hasattr(cls_emb, 'detach'):
        cls_emb = cls_emb.detach().cpu().numpy()
    if isinstance(cls_emb, pd.DataFrame):
        # If user passed a DataFrame, the row index is authoritative; use it
        # as labels unless they explicitly provided cls_labels.
        if cls_labels is None:
            cls_labels = cls_emb.index.values
        cls_emb = cls_emb.values
    cls_emb = np.asarray(cls_emb, dtype=np.float64)
    cls_labels = np.asarray(cls_labels).astype(str)
    if cls_emb.shape[0] != len(cls_labels):
        raise ValueError(
            f'cls_emb has {cls_emb.shape[0]} rows but cls_labels has {len(cls_labels)} entries.'
        )

    trained_set = set(np.asarray(trained_labels).astype(str))
    is_trained = np.array([lbl in trained_set for lbl in cls_labels], dtype=bool)
    if not is_trained.any():
        raise ValueError(
            'No overlap between cls_labels and trained_labels — '
            'cannot compute similarity to a trained set of size 0.'
        )

    # Normalize for cosine
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    cls_norm = cls_emb / norms
    trained_norm = cls_norm[is_trained]
    trained_arr = cls_labels[is_trained]

    sim_to_trained = cls_norm @ trained_norm.T  # (n_classes, n_trained)
    if aggregator == 'max':
        sims = sim_to_trained.max(axis=1)
    elif aggregator == 'mean':
        sims = sim_to_trained.mean(axis=1)
    elif aggregator == 'topk_mean':
        k = max(1, min(int(topk), sim_to_trained.shape[1]))
        top_idx = np.argsort(-sim_to_trained, axis=1)[:, :k]
        sims = np.take_along_axis(sim_to_trained, top_idx, axis=1).mean(axis=1)
    else:
        raise ValueError(f'Unknown aggregator: {aggregator!r}')

    nearest_idx = sim_to_trained.argmax(axis=1)
    nearest = trained_arr[nearest_idx].astype(object)

    sims_out = sims.astype(float)
    sims_out[is_trained] = np.nan         # trained classes: self-similarity is trivial
    nearest_out = nearest.copy()
    nearest_out[is_trained] = ''

    return pd.DataFrame({
        'is_trained': is_trained,
        'similarity_to_trained': sims_out,
        'nearest_trained': nearest_out,
    }, index=pd.Index(cls_labels, name='label'))


def random_cosine_baseline(
    cls_emb,
    n_samples: int = 10_000,
    exclude_self: bool = True,
    seed: int = 0,
    center: bool = False,
) -> dict:
    """Expected cosine similarity between two randomly-drawn class anchors.

    The null hypothesis for any per-cell soft-cosine analysis: if the
    model predicted a *random* class from the candidate set, what cosine
    to the true class would we expect? In LLM embedding spaces this is
    rarely 0 — embeddings are anisotropic and pile near a low-dim
    subspace, so the random-pair baseline is often 0.2-0.5 even for
    semantically unrelated pairs. Without this, "model achieves 0.66" is
    uninterpretable.

    Sampled (not exhaustive) because exhaustive is ``n²`` cosines, which
    is fine for 7K classes but wasteful — 10K samples is enough for a
    standard error around 0.005.

    Parameters
    ----------
    cls_emb
        ``(n_classes, dim)`` anchor matrix. Numpy / torch / DataFrame ok.
    n_samples
        Number of random pairs to draw.
    exclude_self
        If True, only pairs with i != j contribute (default — diagonal is
        trivially 1.0 and inflates the mean).
    seed
        For reproducibility.
    center
        If True, mean-center the embedding (subtract the per-dim global
        mean) before normalizing. Strips the DC offset that anisotropic
        LLM spaces have — random-pair cosine becomes ~0 in the centered
        view, so "model achieves X" is interpretable as a direct
        signal-over-noise ratio. Compare paired runs (raw + centered)
        to understand how much of your model's cosine score is real
        semantic structure vs. just LLM-space anisotropy.

    Returns
    -------
    dict with ``mean``, ``median``, ``std``, ``n_samples`` — the baseline
    distribution stats for use in plot annotations or significance tests.
    """
    if hasattr(cls_emb, 'detach'):
        cls_emb = cls_emb.detach().cpu().numpy()
    if isinstance(cls_emb, pd.DataFrame):
        cls_emb = cls_emb.values
    cls_emb = np.asarray(cls_emb, dtype=np.float64)
    n = cls_emb.shape[0]
    if n < 2:
        return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'), 'n_samples': 0}
    # Optional mean-centering: subtract the per-dim mean before
    # normalizing. LLM embedding spaces are typically anisotropic — they
    # pile near a low-dim subspace, so raw mean pairwise cosine is often
    # 0.5-0.8 even for semantically unrelated pairs. Centering removes
    # this DC offset so random-pair cosine becomes ~0 and the model's
    # actual "lift over random" is directly visible in the metric.
    if center:
        cls_emb = cls_emb - cls_emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    cls_n = cls_emb / norms
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_samples)
    j = rng.integers(0, n, size=n_samples)
    if exclude_self:
        mask = i != j
        if not mask.all():
            i, j = i[mask], j[mask]
    sims = np.einsum('ij,ij->i', cls_n[i], cls_n[j])
    return {
        'mean': float(sims.mean()),
        'median': float(np.median(sims)),
        'std': float(sims.std()),
        'n_samples': int(len(sims)),
    }


def per_cell_topk_recall(
    y_true,
    soft_predictions,
    cls_columns,
    ks: tuple[int, ...] = (5, 50, 100),
) -> pd.DataFrame:
    """Per-cell top-K hit indicators at multiple K values.

    For each cell and each K, returns ``True`` if the cell's true class
    is among the K highest-logit predictions. The natural soft-but-
    discrete companion to strict F1: instead of "did you guess the
    exact gene as #1", you ask "is the truth in your top-K shortlist?".

    Computes the top max(ks) once and derives smaller K's by slicing,
    so the cost is one argpartition + one local sort per cell — not
    one full sort per K.

    Parameters
    ----------
    y_true
        Per-cell true class labels.
    soft_predictions
        ``(n_cells, n_classes)`` logits matrix. Accepts numpy ndarray,
        pandas DataFrame (columns = class labels), or scipy CSR.
    cls_columns
        Array of class labels matching soft_predictions columns. For
        DataFrames, ignored if absent (uses the DataFrame's columns).
    ks
        Tuple of K values to evaluate at. Capped at n_classes per cell.

    Returns
    -------
    pd.DataFrame with one row per cell and one ``hit@{K}`` bool column
    per K. Cells whose y_true isn't in ``cls_columns`` get NaN at all K.
    """
    if isinstance(soft_predictions, pd.DataFrame):
        if cls_columns is None:
            cls_columns = soft_predictions.columns.values
        soft_predictions = soft_predictions.values
    cls_columns = np.asarray(cls_columns).astype(str)
    y_true = np.asarray(y_true).astype(str)
    col_to_idx = {c: i for i, c in enumerate(cls_columns)}
    true_col = np.array([col_to_idx.get(l, -1) for l in y_true])

    n_cells = len(y_true)
    out = pd.DataFrame(index=range(n_cells))
    if not (true_col >= 0).any():
        # No valid lookups — return NaN columns
        for k in ks:
            out[f'hit@{k}'] = np.nan
        return out

    max_k = max(ks)
    if issparse(soft_predictions):
        # CSR rows already carry only top-K_stored entries (when produced
        # by predict(top_k_logits=…)). Take top of those per row.
        soft_predictions = soft_predictions.tocsr()
        n_cells_, n_classes = soft_predictions.shape
        eff_max_k = min(max_k, n_classes)
        # For each row, gather column indices sorted by value (descending).
        # This is O(nnz) which is small for top-K-stored CSR.
        topk_idx_sorted = np.full((n_cells_, eff_max_k), -1, dtype=np.int64)
        indptr = soft_predictions.indptr
        indices = soft_predictions.indices
        data = soft_predictions.data
        for i in range(n_cells_):
            s, e = indptr[i], indptr[i + 1]
            n_nz = e - s
            if n_nz == 0:
                continue
            row_vals = data[s:e]
            row_cols = indices[s:e]
            take = min(eff_max_k, n_nz)
            # Partial sort: top ``take`` by value, descending
            order = np.argpartition(-row_vals, take - 1)[:take]
            order = order[np.argsort(-row_vals[order])]
            topk_idx_sorted[i, :take] = row_cols[order]
    else:
        sp_arr = np.asarray(soft_predictions)
        eff_max_k = min(max_k, sp_arr.shape[1])
        topk_idx = np.argpartition(sp_arr, -eff_max_k, axis=1)[:, -eff_max_k:]
        topk_vals = np.take_along_axis(sp_arr, topk_idx, axis=1)
        order = np.argsort(-topk_vals, axis=1)
        topk_idx_sorted = np.take_along_axis(topk_idx, order, axis=1)

    # Membership check per K — slice from the sorted top-max_K.
    for k in ks:
        kk = min(k, eff_max_k)
        hit = (topk_idx_sorted[:, :kk] == true_col[:, None]).any(axis=1)
        # Mark NaN for cells whose y_true couldn't be looked up.
        hit_f = hit.astype(np.float32)
        hit_f[true_col < 0] = np.nan
        out[f'hit@{k}'] = hit_f
    return out


def per_cell_prediction_cosine(
    y_true,
    y_pred,
    cls_emb,
    cls_labels,
    trained_labels=None,
    cls_similarities: pd.DataFrame | None = None,
    center: bool = False,
) -> pd.DataFrame:
    """Per-cell cosine similarity between predicted and true class proxies.

    One row per cell — no per-class aggregation. The result is what you
    want for plots like "distribution of soft-quality across cells,
    grouped by zero-shot bin", where each cell is an independent data
    point binned by its **true class's** similarity to the trained set.

    Parameters
    ----------
    y_true, y_pred
        Length-N per-cell label arrays.
    cls_emb
        ``(n_classes, dim)`` proxy matrix (raw or projection space).
    cls_labels
        Length-``n_classes`` label array matching ``cls_emb`` rows.
    trained_labels
        Optional. Used to add ``is_zero_shot`` based on y_true.
    cls_similarities
        Optional output of ``class_similarity_to_trained``. Used to add
        per-cell ``similarity_to_trained`` (of y_true), so each cell can
        be binned in a downstream plot.

    Returns
    -------
    pd.DataFrame with one row per input cell. Columns:
        ``y_true, y_pred, cosine``
        ``is_zero_shot``                (when ``trained_labels`` given)
        ``similarity_to_trained``       (when ``cls_similarities`` given)
    Rows where y_true or y_pred isn't in ``cls_labels`` get NaN cosine.
    """
    if hasattr(cls_emb, 'detach'):
        cls_emb = cls_emb.detach().cpu().numpy()
    if isinstance(cls_emb, pd.DataFrame):
        if cls_labels is None:
            cls_labels = cls_emb.index.values
        cls_emb = cls_emb.values
    cls_emb = np.asarray(cls_emb, dtype=np.float64)
    cls_labels = np.asarray(cls_labels).astype(str)
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)

    # Optional mean-centering — see random_cosine_baseline docstring.
    # Strips the LLM-space DC offset so the per-cell cosine reflects the
    # *centered* similarity (random pair ≈ 0). Apply before normalizing.
    if center:
        cls_emb = cls_emb - cls_emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_n = cls_emb / norms

    label_to_idx = {lbl: i for i, lbl in enumerate(cls_labels)}
    true_idx = np.array([label_to_idx.get(l, -1) for l in y_true])
    pred_idx = np.array([label_to_idx.get(l, -1) for l in y_pred])
    valid = (true_idx >= 0) & (pred_idx >= 0)
    cosine = np.full(len(y_true), np.nan, dtype=np.float64)
    if valid.any():
        cosine[valid] = np.einsum(
            'ij,ij->i', emb_n[true_idx[valid]], emb_n[pred_idx[valid]],
        )

    out = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'cosine': cosine,
    })
    if trained_labels is not None:
        trained_set = set(np.asarray(trained_labels).astype(str))
        out['is_zero_shot'] = ~np.isin(y_true, list(trained_set))
    if cls_similarities is not None:
        # similarity_to_trained is NaN for trained classes in the
        # cls_similarities DataFrame — that's intentional and the
        # downstream binning treats NaN-similarity cells as 'seen'.
        sim_map = cls_similarities['similarity_to_trained'].to_dict()
        out['similarity_to_trained'] = [sim_map.get(l, np.nan) for l in y_true]
    return out


def embedding_prediction_quality(
    y_true,
    y_pred,
    cls_emb,
    cls_labels,
    trained_labels=None,
) -> tuple[dict, pd.DataFrame]:
    """Per-cell cosine similarity between predicted and true class **in the
    given embedding space**. Soft alternative to strict label match — wrong
    predictions that land near the true class still score above zero.

    Usually called with the **projection space** (``cls2z``) since that's
    where the CLIP loss operates and where "close" actually means "the
    model considered them similar". Can also be called with the raw cls
    embedding to measure semantic, model-free closeness.

    Parameters
    ----------
    y_true, y_pred
        Length-N cell label arrays.
    cls_emb
        ``(n_classes, dim)`` matrix of class anchors. Rows match ``cls_labels``.
    cls_labels
        Length-``n_classes`` label array, in the same order as ``cls_emb``.
    trained_labels
        Optional. If given, the summary also splits by seen vs zero-shot,
        so you can read "soft quality on seen" vs "soft quality on novel".

    Returns
    -------
    summary : dict
        ``mean_cosine``, ``median_cosine``, ``frac_above_0.5``,
        ``frac_above_0.9``, and the same split into ``seen`` / ``zero_shot``
        buckets when ``trained_labels`` is provided.
    per_class : pd.DataFrame
        For each true-class label: ``support``, ``mean_cosine``,
        ``median_cosine``, ``frac_above_0.5``, plus ``is_zero_shot`` when
        ``trained_labels`` is provided.
    """
    if hasattr(cls_emb, 'detach'):
        cls_emb = cls_emb.detach().cpu().numpy()
    if isinstance(cls_emb, pd.DataFrame):
        if cls_labels is None:
            cls_labels = cls_emb.index.values
        cls_emb = cls_emb.values
    cls_emb = np.asarray(cls_emb, dtype=np.float64)
    cls_labels = np.asarray(cls_labels).astype(str)
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    if len(cls_emb) != len(cls_labels):
        raise ValueError(
            f'cls_emb has {len(cls_emb)} rows but cls_labels has {len(cls_labels)} entries.'
        )

    # Normalize once
    norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_n = cls_emb / norms

    # Per-cell lookup. Cells whose y_true or y_pred isn't in cls_labels
    # (shouldn't happen for in-distribution predictions, but guard anyway)
    # get NaN similarity and are skipped in the aggregation.
    label_to_idx = {lbl: i for i, lbl in enumerate(cls_labels)}
    true_idx = np.array([label_to_idx.get(l, -1) for l in y_true])
    pred_idx = np.array([label_to_idx.get(l, -1) for l in y_pred])
    valid = (true_idx >= 0) & (pred_idx >= 0)
    cell_sim = np.full(len(y_true), np.nan, dtype=np.float64)
    if valid.any():
        cell_sim[valid] = np.einsum(
            'ij,ij->i', emb_n[true_idx[valid]], emb_n[pred_idx[valid]],
        )

    # Per-class breakdowns: BOTH directions.
    #   per_true: rows = unique y_true; "for cells of class X, how close
    #             are predictions to X?". Limited to classes in y_true.
    #   per_pred: rows = unique y_pred; "when model predicts X, how close
    #             is X to whatever the cell's true label was?". Crucial
    #             for the "model predicts close-but-novel" use case —
    #             includes held-out classes the model picked under
    #             obs_only=False that never appear in y_true.
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'cosine': cell_sim})
    df = df.dropna(subset=['cosine'])
    by_true = df.groupby('y_true')['cosine'].agg(['size', 'mean', 'median'])
    by_true.columns = ['support_true', 'mean_cosine_when_true', 'median_cosine_when_true']
    by_pred = df.groupby('y_pred')['cosine'].agg(['size', 'mean', 'median'])
    by_pred.columns = ['support_pred', 'mean_cosine_when_pred', 'median_cosine_when_pred']
    # Outer-join so every class that appears in either direction shows
    # up; columns for the missing direction stay NaN.
    per_class = by_true.join(by_pred, how='outer').rename_axis('label')
    # Back-compat: keep ``support`` / ``mean_cosine`` aliases pointing
    # to the per-true variant (the original semantics). frac_above_0.5
    # also stays per-true since that was the documented "soft credit on
    # cells of class X" reading.
    per_class['support'] = per_class['support_true']
    per_class['mean_cosine'] = per_class['mean_cosine_when_true']
    per_class['median_cosine'] = per_class['median_cosine_when_true']
    above_05_per_true = df.groupby('y_true')['cosine'].apply(lambda s: float((s > 0.5).mean()))
    above_05_per_pred = df.groupby('y_pred')['cosine'].apply(lambda s: float((s > 0.5).mean()))
    per_class['frac_above_0.5'] = above_05_per_true
    per_class['frac_above_0.5_when_pred'] = above_05_per_pred

    if trained_labels is not None:
        trained_set = set(np.asarray(trained_labels).astype(str))
        per_class['is_zero_shot'] = ~per_class.index.isin(trained_set)

    # Summary scalars (all-cell and optional split)
    def _agg(mask):
        v = cell_sim[mask & ~np.isnan(cell_sim)]
        if v.size == 0:
            return {'n_cells': 0, 'mean_cosine': float('nan'), 'median_cosine': float('nan'),
                    'frac_above_0.5': float('nan'), 'frac_above_0.9': float('nan')}
        return {
            'n_cells': int(v.size),
            'mean_cosine': float(v.mean()),
            'median_cosine': float(np.median(v)),
            'frac_above_0.5': float((v > 0.5).mean()),
            'frac_above_0.9': float((v > 0.9).mean()),
        }

    all_mask = np.ones(len(cell_sim), dtype=bool)
    summary = {'all': _agg(all_mask)}
    if trained_labels is not None:
        trained_set = set(np.asarray(trained_labels).astype(str))
        is_seen = np.array([l in trained_set for l in y_true], dtype=bool)
        summary['seen'] = _agg(is_seen)
        summary['zero_shot'] = _agg(~is_seen)
    return summary, per_class


def adapter_fidelity(
    raw_emb,
    proj_emb,
    cls_labels,
    trained_labels,
    raw_trained_labels=None,
) -> tuple[dict, pd.DataFrame]:
    """Quantify how much the adapter distorts the held-out class geometry.

    Compares similarity structure of held-out (zero-shot) classes to the
    trained class set in two spaces:
      - ``raw_emb``  : input to the adapter (the LLM embedding).
      - ``proj_emb`` : output of adapter (or further-projected). What the
                       model actually classifies against.

    Two per-class checks for each held-out class:
      1. **Nearest-trained agreement** — is the closest trained class the
         same in raw vs projected space?
      2. **Local geometry preservation** — Pearson correlation between the
         vector of (held-out → each trained) cosine similarities in raw
         and projected spaces. High r = adapter preserves; low r =
         geometry has been warped.

    Plus the absolute drift in nearest-similarity (``proj_sim - raw_sim``):
    negative means the held-out class moved *farther* from its raw
    sibling under projection (worse), positive means closer.

    Parameters
    ----------
    raw_emb, proj_emb
        ``(n_classes, dim_*)`` matrices. Rows aligned to ``cls_labels``.
        ``raw_emb`` rows should be ordered identically to ``proj_emb``;
        no internal reindexing is performed.
    cls_labels
        Length-``n_classes`` label array for both ``raw_emb`` and
        ``proj_emb`` rows.
    trained_labels
        Labels considered "trained" / seen during model training.
    raw_trained_labels
        Optional. If the raw embedding was supplied with a different label
        ordering than ``cls_labels`` (e.g. only the trained subset), pass
        the matching labels. By default we re-use ``trained_labels``.

    Returns
    -------
    summary : dict
        Scalars: ``frac_nearest_agree``, ``mean_pearson``, ``median_pearson``,
        ``mean_proj_minus_raw_sim``.
    per_class : pd.DataFrame
        One row per held-out class: ``raw_nearest``, ``raw_sim``,
        ``proj_nearest``, ``proj_sim``, ``nearest_agree`` (bool),
        ``pearson_r`` (over similarities to all trained), ``proj_minus_raw``.
    """
    def _to_np(a):
        if hasattr(a, 'detach'):
            a = a.detach().cpu().numpy()
        if isinstance(a, pd.DataFrame):
            return a.values
        return np.asarray(a, dtype=np.float64)
    raw_emb = _to_np(raw_emb)
    proj_emb = _to_np(proj_emb)
    cls_labels = np.asarray(cls_labels).astype(str)
    trained_arr = np.asarray(trained_labels).astype(str)

    if len(raw_emb) != len(cls_labels) or len(proj_emb) != len(cls_labels):
        raise ValueError(
            f'raw_emb / proj_emb row counts ({len(raw_emb)}, {len(proj_emb)}) '
            f'must match cls_labels length ({len(cls_labels)}).'
        )

    is_trained = np.isin(cls_labels, trained_arr)
    is_held_out = ~is_trained
    if not is_held_out.any():
        return {'frac_nearest_agree': float('nan'), 'mean_pearson': float('nan'),
                'median_pearson': float('nan'), 'mean_proj_minus_raw_sim': float('nan')}, \
               pd.DataFrame(columns=[
                   'raw_nearest', 'raw_sim', 'proj_nearest', 'proj_sim',
                   'nearest_agree', 'pearson_r', 'proj_minus_raw',
               ])

    # Normalize for cosine in both spaces
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(n < 1e-12, 1.0, n)
    raw_n = _norm(raw_emb)
    proj_n = _norm(proj_emb)

    trained_labels_arr = cls_labels[is_trained]
    held_idx = np.where(is_held_out)[0]
    held_labels = cls_labels[is_held_out]

    raw_sim = raw_n[held_idx] @ raw_n[is_trained].T     # (n_held, n_trained)
    proj_sim = proj_n[held_idx] @ proj_n[is_trained].T  # (n_held, n_trained)

    raw_nearest_idx = raw_sim.argmax(axis=1)
    proj_nearest_idx = proj_sim.argmax(axis=1)
    raw_nearest = trained_labels_arr[raw_nearest_idx]
    proj_nearest = trained_labels_arr[proj_nearest_idx]
    raw_max = raw_sim.max(axis=1)
    proj_max = proj_sim.max(axis=1)
    nearest_agree = raw_nearest == proj_nearest

    # Per-row Pearson between raw and proj similarity vectors (over trained set)
    raw_mean = raw_sim.mean(axis=1, keepdims=True)
    proj_mean = proj_sim.mean(axis=1, keepdims=True)
    raw_c = raw_sim - raw_mean
    proj_c = proj_sim - proj_mean
    raw_std = np.sqrt((raw_c ** 2).sum(axis=1))
    proj_std = np.sqrt((proj_c ** 2).sum(axis=1))
    denom = raw_std * proj_std
    pearson = np.where(denom > 1e-12, (raw_c * proj_c).sum(axis=1) / denom, np.nan)

    per_class = pd.DataFrame({
        'raw_nearest': raw_nearest,
        'raw_sim': raw_max,
        'proj_nearest': proj_nearest,
        'proj_sim': proj_max,
        'nearest_agree': nearest_agree,
        'pearson_r': pearson,
        'proj_minus_raw': proj_max - raw_max,
    }, index=pd.Index(held_labels, name='label'))

    summary = {
        'n_held_out': int(is_held_out.sum()),
        'frac_nearest_agree': float(nearest_agree.mean()),
        'mean_pearson': float(np.nanmean(pearson)),
        'median_pearson': float(np.nanmedian(pearson)),
        'mean_proj_minus_raw_sim': float((proj_max - raw_max).mean()),
    }
    return summary, per_class


def zero_shot_classification_report(
    y_true,
    y_pred,
    trained_labels,
    sample_weight=None,
    cls_similarities: pd.DataFrame | None = None,
    similarity_bins: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 1.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split classification metrics into seen-class vs zero-shot cells.

    Cells are bucketed by whether their **true** label was in
    ``trained_labels``:
      - ``seen``      : retained-performance check on the trained class set.
      - ``zero_shot`` : generalization to never-seen perturbations.

    When ``cls_similarities`` is supplied (output of
    :func:`class_similarity_to_trained`), zero-shot cells are further split
    into similarity bins so you can separate "guessed a sibling" wins from
    actual novel-class hits.

    Returns
    -------
    summary : pd.DataFrame
        One row per bucket. Columns: ``n_cells``, ``n_classes``,
        ``macro_f1``, ``weighted_f1``, ``accuracy``.
    per_class : pd.DataFrame
        One row per class. Columns: ``f1``, ``precision``, ``recall``,
        ``support``, ``is_zero_shot``, ``similarity_to_trained``,
        ``nearest_trained``.
    """
    from sklearn.metrics import classification_report, accuracy_score

    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    trained_set = set(np.asarray(trained_labels).astype(str))
    is_seen = np.array([lbl in trained_set for lbl in y_true], dtype=bool)

    def _bucket_row(name, mask):
        if not mask.any():
            return None
        yt, yp = y_true[mask], y_pred[mask]
        sw = sample_weight[mask] if sample_weight is not None else None
        rep = classification_report(
            yt, yp, output_dict=True, zero_division=np.nan, sample_weight=sw,
        )
        return {
            'bucket': name,
            'n_cells': int(mask.sum()),
            'n_classes': int(len(np.unique(yt))),
            'macro_f1': rep.get('macro avg', {}).get('f1-score'),
            'weighted_f1': rep.get('weighted avg', {}).get('f1-score'),
            'accuracy': accuracy_score(yt, yp, sample_weight=sw),
        }

    summary_rows = [
        _bucket_row('seen', is_seen),
        _bucket_row('zero_shot', ~is_seen),
        _bucket_row('all', np.ones_like(is_seen)),
    ]
    summary_rows = [r for r in summary_rows if r is not None]

    # Optional similarity-bin breakdown for zero-shot cells
    if cls_similarities is not None and (~is_seen).any():
        sim_map = cls_similarities['similarity_to_trained'].to_dict()
        cell_sim = np.array([sim_map.get(lbl, np.nan) for lbl in y_true], dtype=float)
        bins = np.asarray(similarity_bins, dtype=float)
        for lo, hi in zip(bins[:-1], bins[1:]):
            # Half-open [lo, hi) except the last bin which is closed
            top_closed = np.isclose(hi, bins[-1])
            mask = (~is_seen) & (cell_sim >= lo) & (
                (cell_sim < hi) | (top_closed & np.isclose(cell_sim, hi))
            )
            row = _bucket_row(f'zero_shot[sim={lo:.2f}-{hi:.2f}]', mask)
            if row is not None:
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index('bucket')

    # Per-class breakdown
    full_rep = classification_report(
        y_true, y_pred, output_dict=True, zero_division=np.nan, sample_weight=sample_weight,
    )
    pc_rows = []
    skip_keys = {'accuracy', 'macro avg', 'weighted avg'}
    for label, vals in full_rep.items():
        if label in skip_keys or not isinstance(vals, dict):
            continue
        # Keep classes from sklearn's union(y_true, y_pred) — including
        # ones with support=0 in y_true (model predicted them but they
        # never appeared as ground truth). The embedding-prediction-
        # quality join below will populate ``mean_cosine_when_pred`` for
        # those, so they're not just noise rows.
        pc_rows.append({
            'label': label,
            'f1': vals.get('f1-score'),
            'precision': vals.get('precision'),
            'recall': vals.get('recall'),
            'support': vals.get('support'),
            'is_zero_shot': label not in trained_set,
        })
    per_class = pd.DataFrame(pc_rows).set_index('label')
    if cls_similarities is not None:
        per_class = per_class.join(
            cls_similarities[['similarity_to_trained', 'nearest_trained']],
            how='left',
        )
    return summary, per_class


_ENRICHR_HOST = {
    'Human': 'https://maayanlab.cloud/Enrichr',
    'Mouse': 'https://maayanlab.cloud/modEnrichr/Mouse',
    'Rat': 'https://maayanlab.cloud/modEnrichr/Rat',
    'Fly': 'https://maayanlab.cloud/modEnrichr/Fly',
    'Yeast': 'https://maayanlab.cloud/modEnrichr/Yeast',
    'Worm': 'https://maayanlab.cloud/modEnrichr/Worm',
    'Fish': 'https://maayanlab.cloud/modEnrichr/Fish',
}


def _fetch_enrichr_library(
    library: str,
    organism: str = 'Human',
    n_retries: int = 3,
    retry_backoff: float = 2.0,
) -> dict[str, list[str]]:
    """Fetch an Enrichr pathway library directly via HTTP and parse it.

    Bypasses gseapy's parser, which has version-dependent encoding bugs
    (``TypeError: a bytes-like object is required, not 'str'``) on certain
    library/Python combinations. The Enrichr text endpoint returns one
    line per pathway: ``<name>\\t\\t<gene1>\\t<gene2>\\t...``.

    Retries transient 4xx/5xx with exponential backoff — Enrichr will
    occasionally return 400 under load even for valid library names, so
    a single failed request shouldn't kill the call.
    """
    import time
    import requests
    host = _ENRICHR_HOST.get(organism)
    if host is None:
        raise ValueError(
            f"Unknown organism {organism!r}. "
            f"Supported: {sorted(_ENRICHR_HOST.keys())}"
        )
    url = f'{host}/geneSetLibrary'
    params = {'mode': 'text', 'libraryName': library}
    headers = {'User-Agent': 'expert-pathways/1.0 (+https://github.com)'}

    last_err: Exception | None = None
    text: str | None = None
    for attempt in range(n_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 200 and resp.text.strip():
                text = resp.text
                break
            # Surface server-side body for diagnostics on 4xx/5xx.
            body = (resp.text or '').strip().splitlines()[:1]
            last_err = requests.HTTPError(
                f'{resp.status_code} {resp.reason} for {resp.url} '
                f'(body: {body[0] if body else "<empty>"})'
            )
        except requests.RequestException as e:
            last_err = e
        if attempt < n_retries - 1:
            time.sleep(retry_backoff * (2 ** attempt))

    if text is None:
        raise RuntimeError(
            f'Enrichr fetch failed after {n_retries} attempts for '
            f'{library!r} ({organism}). Last error: {last_err!r}'
        ) from last_err

    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        name = parts[0].strip()
        # Enrichr format has empty placeholder column 1; genes start at index 1
        # but with a blank first entry. Filter empties.
        genes = [g.strip().split(',')[0] for g in parts[1:] if g and g.strip()]
        if name and genes:
            out[name] = genes
    if not out:
        raise RuntimeError(
            f"Empty library returned from Enrichr for {library!r} ({organism}). "
            f"URL: {url}?mode=text&libraryName={library}"
        )
    return out


@functools.lru_cache(maxsize=16)
def get_library(library="KEGG_2019_Human", organism="Human") -> dict | None:
    """Load a gseapy pathway library with HTTP fallback + helpful diagnostics.

    Tries gseapy first; on any failure, retries via direct Enrichr HTTP
    fetch (bypasses gseapy parser bugs). On total failure, surfaces the
    underlying exceptions and prints a list of similarly-named libraries
    available for ``organism``.

    Cached via ``lru_cache`` (size 16) so repeated calls with the same
    ``(library, organism)`` return the same dict without re-fetching —
    matters when multiple analysis functions all want the same library.
    Callers should treat the result as read-only.
    """
    try:
        import gseapy as gp
        return gp.get_library(name=library, organism=organism)
    except Exception as e:
        gseapy_err = repr(e)
    # gseapy failed — try direct HTTP. Common workaround for the
    # 'bytes-like / str' TypeError seen on some gseapy versions.
    try:
        result = _fetch_enrichr_library(library, organism=organism)
        print(f"[get_library] gseapy failed ({gseapy_err}); fell back to direct Enrichr fetch.")
        return result
    except Exception as http_err:
        print(
            f"Error loading library '{library}' for organism '{organism}'.\n"
            f"  gseapy error:    {gseapy_err}\n"
            f"  HTTP fallback:   {http_err!r}"
        )
        try:
            import gseapy as gp
            available = gp.get_library_name(organism=organism)
        except Exception:
            available = []
        token = library.split('_')[0].lower()
        related = [n for n in available if token in n.lower()]
        if related:
            print(f"Available {token!r}-like libraries for {organism!r}:")
            for n in related:
                print(f"  - {n}")
        return None

def mask_top_degs(
    adata: ad.AnnData,
    condition_col: str,
    n_top: int = 200,
    method: str = 'wilcoxon',
    fill: str = 'mean',
    pairwise: bool = True,
) -> tuple[ad.AnnData, list[str]]:
    """Return a copy of ``adata`` with the top-N DEGs per condition masked.

    Control experiment for the question: "Is the model's predicted
    perturbation enrichment merely a fancy DEG readout?" If predict +
    enrichment looks the same on a DEG-masked adata as on the original,
    the model is using distributed pathway-level signal rather than
    direct expression of a few high-LFC genes.

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[condition_col]`` defined.
    condition_col
        Column with the groups to compute DEGs across.
    n_top
        Top-N genes per group (or pairwise comparison) to mask. The
        masked set is the *union* across all groups / pairs.
    method
        Passed to ``sc.tl.rank_genes_groups``. Wilcoxon is robust to
        non-normal scRNA-seq distributions.
    fill
        How to replace the masked entries. Options:
        - ``'mean'``   per-gene mean across all cells (default)
        - ``'median'`` per-gene median across all cells
        - ``'zero'``   simulate "absent" (risky — model may interpret as KO)
        - ``'shuffle'`` shuffle each masked column independently across cells
          (destroys cross-cell variation but preserves the marginal).
    pairwise
        If True, compute DEGs for every pair of conditions (A vs B,
        A vs C, B vs C, ...) and union the top-N per pair. If False,
        compute one-vs-rest per condition. ``pairwise=True`` catches
        more genes — recommended for the strongest control.

    Returns
    -------
    masked_adata
        Copy of ``adata`` with the masked columns replaced.
    masked_gene_names
        Sorted list of gene names that were masked.
    """
    import scanpy as sc

    if condition_col not in adata.obs:
        raise KeyError(f'condition_col {condition_col!r} not in adata.obs.')
    groups = list(pd.unique(adata.obs[condition_col].astype(str)))
    if len(groups) < 2:
        raise ValueError(f'Need ≥2 conditions in {condition_col!r}, got {groups}.')

    def _log_normalize_for_ranking(sub: ad.AnnData) -> ad.AnnData:
        """Return a copy of ``sub`` with X normalized + log1p'd so
        rank_genes_groups (Wilcoxon / t-test) operates on a homoscedastic
        scale rather than raw counts (which biases Wilcoxon by library
        size). The masked output we return to the caller stays on raw X
        — only the DEG ranking uses the logged copy."""
        sub = sub.copy()
        # Heuristic: if X looks integer-ish (raw counts), normalize + log.
        # If max < 50 and dtype is float, assume already normalized — skip.
        x_sample = sub.X
        if issparse(x_sample):
            x_max = x_sample.max() if x_sample.nnz > 0 else 0
        else:
            x_max = float(np.asarray(x_sample).max()) if x_sample.size > 0 else 0
        looks_raw = (
            x_max > 50
            or np.issubdtype(getattr(x_sample, 'dtype', np.float32), np.integer)
        )
        if looks_raw:
            sc.pp.normalize_total(sub, target_sum=1e4)
            sc.pp.log1p(sub)
        sub.obs[condition_col] = sub.obs[condition_col].astype(str).astype('category')
        return sub

    # Gather top-N gene names from each comparison.
    top_set: set[str] = set()
    if pairwise:
        for i, g_i in enumerate(groups):
            for g_j in groups[i + 1:]:
                mask = adata.obs[condition_col].astype(str).isin([g_i, g_j])
                tmp = _log_normalize_for_ranking(adata[mask])
                sc.tl.rank_genes_groups(
                    tmp, condition_col, method=method, n_genes=n_top, use_raw=False,
                )
                names = tmp.uns['rank_genes_groups']['names']
                for g in (g_i, g_j):
                    if g in names.dtype.names:
                        top_set.update(np.asarray(names[g]).flatten().tolist())
    else:
        tmp = _log_normalize_for_ranking(adata)
        sc.tl.rank_genes_groups(
            tmp, condition_col, method=method, n_genes=n_top, use_raw=False,
        )
        names = tmp.uns['rank_genes_groups']['names']
        for g in groups:
            if g in names.dtype.names:
                top_set.update(np.asarray(names[g]).flatten().tolist())

    # Map gene names → column indices in adata.var_names.
    gene_idx = adata.var_names.get_indexer(sorted(top_set))
    gene_idx = gene_idx[gene_idx >= 0]
    masked_gene_names = adata.var_names[gene_idx].tolist()
    n_masked = len(masked_gene_names)
    print(
        f'[mask_top_degs] Masking {n_masked} / {adata.n_vars} genes '
        f'(top-{n_top} per comparison, fill={fill}, pairwise={pairwise}).'
    )

    masked = adata.copy()
    _fill_masked_columns(masked, gene_idx, fill=fill)
    return masked, masked_gene_names


def _fill_masked_columns(masked: ad.AnnData, gene_idx: np.ndarray, fill: str = 'mean', seed: int = 0) -> None:
    """Replace masked-column values in ``masked.X`` according to ``fill``.

    Shared by ``mask_top_degs`` and ``random_mask``. Densifies only the
    masked columns (not the whole matrix) so the memory footprint is
    ``n_cells * n_masked * 4`` bytes regardless of whether X is sparse.
    """
    X = masked.X
    if fill == 'zero':
        if issparse(X):
            X = X.tolil()
            X[:, gene_idx] = 0
            X = X.tocsr()
        else:
            X[:, gene_idx] = 0
    elif fill in ('mean', 'median'):
        if issparse(X):
            cols_dense = np.asarray(X[:, gene_idx].todense())
        else:
            cols_dense = np.asarray(X[:, gene_idx], dtype=np.float32)
        stat = np.median(cols_dense, axis=0) if fill == 'median' else cols_dense.mean(axis=0)
        replacement = np.broadcast_to(stat, cols_dense.shape).copy()
        if issparse(X):
            X = X.tolil()
            X[:, gene_idx] = replacement
            X = X.tocsr()
        else:
            X[:, gene_idx] = replacement
    elif fill == 'shuffle':
        rng = np.random.default_rng(seed)
        if issparse(X):
            cols_dense = np.asarray(X[:, gene_idx].todense())
        else:
            cols_dense = np.asarray(X[:, gene_idx], dtype=np.float32)
        for j in range(cols_dense.shape[1]):
            cols_dense[:, j] = rng.permutation(cols_dense[:, j])
        if issparse(X):
            X = X.tolil()
            X[:, gene_idx] = cols_dense
            X = X.tocsr()
        else:
            X[:, gene_idx] = cols_dense
    else:
        raise ValueError(f'Unknown fill: {fill!r}. Use mean / median / zero / shuffle.')
    masked.X = X


def random_mask(
    adata: ad.AnnData,
    n_genes: int,
    fill: str = 'mean',
    exclude_genes: list[str] | None = None,
    seed: int = 0,
) -> tuple[ad.AnnData, list[str]]:
    """Mask ``n_genes`` random gene columns (optionally excluding a set).

    Matched control for ``mask_top_degs``: pass the same ``n_genes`` and
    ``fill``, and exclude the DEG list so the random set is disjoint
    from the DEG set. The differential ``r_deg − r_random`` from the
    downstream predictions tells you whether masking is gene-specific
    (DEG-dependent model) or just generally disruptive (DEG-agnostic).
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(adata.n_vars)
    if exclude_genes:
        ex_idx = adata.var_names.get_indexer(list(exclude_genes))
        ex_idx = ex_idx[ex_idx >= 0]
        all_idx = np.setdiff1d(all_idx, ex_idx, assume_unique=False)
    n = min(int(n_genes), len(all_idx))
    gene_idx = np.sort(rng.choice(all_idx, size=n, replace=False))
    masked_gene_names = adata.var_names[gene_idx].tolist()
    masked = adata.copy()
    _fill_masked_columns(masked, gene_idx, fill=fill, seed=seed)
    return masked, masked_gene_names


def _pathway_scores_from_z(
    z_df: pd.DataFrame,
    library_obj: dict,
    min_pathway_size: int = 1,
) -> pd.DataFrame:
    """Per-(group, pathway) mean z-score over pathway members present in z_df.

    Returns rows=groups, cols=pathways (only those with ≥min_pathway_size
    members in z_df's perturbation set). Same aggregation as
    ``plot_pathway_enrichment``'s ``score='mean_z'`` mode.
    """
    pert_upper_to_idx = {str(p).upper(): i for i, p in enumerate(z_df.columns)}
    out: dict[str, np.ndarray] = {}
    z_vals = z_df.values
    for pw, members in library_obj.items():
        idx = [pert_upper_to_idx[str(m).upper()]
               for m in members if str(m).upper() in pert_upper_to_idx]
        if len(idx) < min_pathway_size:
            continue
        out[pw] = z_vals[:, idx].mean(axis=1)
    return pd.DataFrame(out, index=z_df.index)


def evaluate_deg_robustness(
    model,
    query: ad.AnnData,
    condition_col: str,
    n_top_list: list[int] | tuple[int, ...] = (50, 200, 1000),
    fill: str = 'mean',
    method: str = 'wilcoxon',
    pairwise: bool = True,
    top_perts: int | None = 100,
    top_k_classify: int = 50,
    jaccard_k: int = 20,
    pathway_library: str | None = None,
    pathway_organism: str = 'Human',
    pathway_min_size: int = 5,
    random_baseline: bool = True,
    random_seed: int = 0,
    obs_only: bool = False,
    soft_predictions_key: str = 'soft_predictions',
    verbose: bool = True,
) -> pd.DataFrame:
    """DEG-masking robustness control for a queried adata.

    Three-pronged test of whether the model's predictions are merely a
    DEG readout, or carry information beyond what classical DEG analysis
    can detect (driver-burst / post-transcriptional / signaling-cascade
    biology that's no longer visible at sequencing time):

    1. **Per-pert robustness**: Pearson / MAD / Jaccard between original
       and DEG-masked per-group z-profiles. Survives masking → not a
       DEG wrapper.
    2. **Random-mask baseline**: same metrics after masking the same
       number of *random* (non-DEG) genes. If
       ``r_deg ≈ r_random`` → model is DEG-*agnostic* (the strongest
       evidence). If ``r_deg ≪ r_random`` → DEG-dependent.
    3. **Pathway-level conservation**: same Pearson/Jaccard on
       per-(group, pathway) mean-z scores. The most biologically
       meaningful claim — *"pathway-level enrichments survive even after
       removing the top DE genes that defined them"*. Enabled by
       passing ``pathway_library``.

    Each (n_top, condition) is also classified at the perturbation level:

    - ``DEG_occult``    — in top-K of original AND DEG-masked profiles,
                          and the perturbation gene is **not** in the
                          DEG list → driver-burst candidate (model finds
                          it without DEG evidence)
    - ``DEG_confirmed`` — in top-K of both, gene **is** in the DEG list
                          → consensus with classical DEG
    - ``DEG_dependent`` — in top-K of original, drops out of masked
                          top-K → DEG echo
    - ``gained_after_mask`` — in masked top-K but not original (suspect;
                              usually artifact)

    Parameters
    ----------
    model, query, condition_col, n_top_list, fill, method, pairwise,
    top_perts, obs_only, soft_predictions_key, verbose
        See previous version — unchanged.
    top_k_classify
        Top-K per (group, mask) used for the per-pert classification.
    jaccard_k
        K for the top-K Jaccard summary metric (replaces hardcoded 20).
    pathway_library
        gseapy / Enrichr library name (e.g. ``'MSigDB_Hallmark_2020'``).
        When provided, ``pearson_pathway`` and ``jaccard_pathway`` are
        populated and full per-pathway scores are stored on
        ``df.attrs['pathway_scores']``.
    pathway_organism, pathway_min_size
        Forwarded to ``get_library`` / ``_pathway_scores_from_z``.
    random_baseline
        Run a matched random-mask iteration per ``n_top`` (default True).
        Doubles prediction cost but is necessary to make the "not a DEG
        wrapper" claim quantitatively.
    random_seed
        Seed for random-mask gene selection. The actual seed per
        iteration is ``random_seed + n_top`` so iterations differ.

    Returns
    -------
    pd.DataFrame
        Long-form per ``(n_top, group, mask_type)``. Columns:
        ``n_top, n_masked, group, mask_type, pearson, mean_abs_delta,
        top_k_jaccard, pearson_pathway, jaccard_pathway``.

        Attached via ``df.attrs``:

        - ``pert_classification`` — per-(n_top, group, perturbation)
          classification: ``[n_top, group, perturbation, z_orig,
          z_masked, rank_orig, rank_masked, in_deg_list, class]``.
        - ``pathway_scores`` — per-(n_top, group, pathway, mask_type)
          score values for downstream pathway-level inspection. The
          ``mask_type='orig'`` rows carry ``n_top=NaN``.
    """

    def _z_profile(adata: ad.AnnData) -> pd.DataFrame:
        """Per-group × per-pert z-score (full, no top-K filter)."""
        soft = adata.obsm[soft_predictions_key]
        if hasattr(soft, 'columns') and hasattr(soft, 'values'):
            pert_names = np.asarray(soft.columns).astype(str)
            soft_arr = np.asarray(soft.values)
        else:
            soft_arr = np.asarray(soft)
            pert_names = np.arange(soft_arr.shape[1]).astype(str)
        groups = adata.obs[condition_col].astype(str).values
        unique_groups = pd.unique(groups)
        profiles = np.zeros((len(unique_groups), soft_arr.shape[1]), dtype=np.float32)
        for i, g in enumerate(unique_groups):
            mask = groups == g
            if not mask.any():
                continue
            profiles[i] = soft_arr[mask].astype(np.float32).mean(axis=0)
        mu = profiles.mean(axis=0)
        sd = profiles.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        z = (profiles - mu) / sd
        return pd.DataFrame(z, index=unique_groups, columns=pert_names)

    def _compare(v_full: np.ndarray, v_alt: np.ndarray) -> tuple[float, float, float]:
        """Return (pearson, mad, top_k_jaccard) of two same-length z-vectors."""
        if np.std(v_full) > 1e-9 and np.std(v_alt) > 1e-9:
            r = float(np.corrcoef(v_full, v_alt)[0, 1])
        else:
            r = float('nan')
        mad = float(np.mean(np.abs(v_full - v_alt)))
        t_full = set(np.argsort(-np.abs(v_full))[:jaccard_k])
        t_alt = set(np.argsort(-np.abs(v_alt))[:jaccard_k])
        jac = len(t_full & t_alt) / max(len(t_full | t_alt), 1)
        return r, mad, jac

    # --- Seed original profile ---
    if soft_predictions_key not in query.obsm:
        if verbose:
            print(f'[deg-robustness] {soft_predictions_key!r} missing — predicting first.')
        model.predict(adata=query, inplace=True, obs_only=obs_only)
    z_orig = _z_profile(query)

    if top_perts is not None and top_perts < z_orig.shape[1]:
        focus_cols = z_orig.abs().max(axis=0).nlargest(top_perts).index
    else:
        focus_cols = z_orig.columns
    if verbose:
        print(f'[deg-robustness] Comparing on {len(focus_cols)} perts.')

    # --- Pathway library (optional) ---
    lib_obj = None
    pw_orig = None
    if pathway_library is not None:
        if verbose:
            print(f'[deg-robustness] Loading pathway library {pathway_library!r}.')
        lib_obj = get_library(library=pathway_library, organism=pathway_organism)
        if lib_obj is None:
            logging.warning(
                f'[deg-robustness] Could not load {pathway_library}; '
                'pathway-level checks disabled.'
            )
        else:
            pw_orig = _pathway_scores_from_z(z_orig, lib_obj, pathway_min_size)
            if verbose:
                print(
                    f'[deg-robustness] {pw_orig.shape[1]} pathways pass '
                    f'min_size={pathway_min_size}.'
                )

    rows: list[dict] = []
    class_rows: list[dict] = []
    pathway_rows: list[dict] = []

    # Record original pathway scores once (mask_type='orig')
    if pw_orig is not None:
        for g in pw_orig.index:
            for pw in pw_orig.columns:
                pathway_rows.append({
                    'n_top': np.nan,
                    'group': g,
                    'pathway': pw,
                    'mask_type': 'orig',
                    'score': float(pw_orig.loc[g, pw]),
                })

    for n in n_top_list:
        # === DEG mask ===
        deg_adata, masked_genes = mask_top_degs(
            query, condition_col=condition_col,
            n_top=n, fill=fill, method=method, pairwise=pairwise,
        )
        n_masked = len(masked_genes)
        if verbose:
            print(f'[deg-robustness] n_top={n}: {n_masked} DEGs masked.')
        model.predict(adata=deg_adata, inplace=True, obs_only=obs_only)
        z_deg = _z_profile(deg_adata)
        pw_deg = _pathway_scores_from_z(z_deg, lib_obj, pathway_min_size) if lib_obj else None
        del deg_adata

        # === Random mask (matched size, disjoint from DEG list) ===
        z_rand = None
        pw_rand = None
        if random_baseline:
            rand_adata, rand_genes = random_mask(
                query, n_genes=n_masked, fill=fill,
                exclude_genes=masked_genes, seed=random_seed + int(n),
            )
            if verbose:
                print(f'[deg-robustness] n_top={n}: {len(rand_genes)} RANDOM genes masked.')
            model.predict(adata=rand_adata, inplace=True, obs_only=obs_only)
            z_rand = _z_profile(rand_adata)
            pw_rand = _pathway_scores_from_z(z_rand, lib_obj, pathway_min_size) if lib_obj else None
            del rand_adata

        # === Per-group comparison ===
        common_cols = focus_cols.intersection(z_deg.columns)
        cols_arr = list(common_cols)
        deg_set_upper = {str(g).upper() for g in masked_genes}

        for g in z_orig.index:
            if g not in z_deg.index:
                continue
            v_orig = z_orig.loc[g, common_cols].values.astype(np.float32)
            v_deg = z_deg.loc[g, common_cols].values.astype(np.float32)
            r_deg, mad_deg, jac_deg = _compare(v_orig, v_deg)

            # Per-pert classification using top_k_classify
            order_orig = np.argsort(-np.abs(v_orig))
            order_deg = np.argsort(-np.abs(v_deg))
            rank_pos_orig = np.empty_like(order_orig)
            rank_pos_orig[order_orig] = np.arange(len(order_orig)) + 1
            rank_pos_deg = np.empty_like(order_deg)
            rank_pos_deg[order_deg] = np.arange(len(order_deg)) + 1
            top_orig = set(order_orig[:top_k_classify].tolist())
            top_deg = set(order_deg[:top_k_classify].tolist())

            for j, pert in enumerate(cols_arr):
                in_orig = j in top_orig
                in_deg_top = j in top_deg
                in_deg_list = str(pert).upper() in deg_set_upper
                if in_orig and in_deg_top:
                    cls = 'DEG_confirmed' if in_deg_list else 'DEG_occult'
                elif in_orig and not in_deg_top:
                    cls = 'DEG_dependent'
                elif (not in_orig) and in_deg_top:
                    cls = 'gained_after_mask'
                else:
                    continue  # not_top in either — skip to keep table compact
                class_rows.append({
                    'n_top': n,
                    'group': g,
                    'perturbation': pert,
                    'z_orig': float(v_orig[j]),
                    'z_masked': float(v_deg[j]),
                    'rank_orig': int(rank_pos_orig[j]),
                    'rank_masked': int(rank_pos_deg[j]),
                    'in_deg_list': bool(in_deg_list),
                    'class': cls,
                })

            # Pathway-level metrics (DEG mask)
            r_pw_deg, jac_pw_deg = float('nan'), float('nan')
            if pw_orig is not None and pw_deg is not None and g in pw_deg.index:
                common_pw = pw_orig.columns.intersection(pw_deg.columns)
                if len(common_pw) > 0:
                    p_orig = pw_orig.loc[g, common_pw].values.astype(np.float32)
                    p_deg = pw_deg.loc[g, common_pw].values.astype(np.float32)
                    r_pw_deg, _, jac_pw_deg = _compare(p_orig, p_deg)

            rows.append({
                'n_top': n,
                'n_masked': n_masked,
                'group': g,
                'mask_type': 'deg',
                'pearson': r_deg,
                'mean_abs_delta': mad_deg,
                'top_k_jaccard': jac_deg,
                'pearson_pathway': r_pw_deg,
                'jaccard_pathway': jac_pw_deg,
            })

            # Random-mask row (matched)
            r_rand = mad_rand = jac_rand = float('nan')
            r_pw_rand = jac_pw_rand = float('nan')
            if z_rand is not None and g in z_rand.index:
                v_rand = z_rand.loc[g, common_cols].values.astype(np.float32)
                r_rand, mad_rand, jac_rand = _compare(v_orig, v_rand)
                if pw_orig is not None and pw_rand is not None and g in pw_rand.index:
                    common_pw_r = pw_orig.columns.intersection(pw_rand.columns)
                    if len(common_pw_r) > 0:
                        p_orig_r = pw_orig.loc[g, common_pw_r].values.astype(np.float32)
                        p_rand_g = pw_rand.loc[g, common_pw_r].values.astype(np.float32)
                        r_pw_rand, _, jac_pw_rand = _compare(p_orig_r, p_rand_g)
                rows.append({
                    'n_top': n,
                    'n_masked': n_masked,
                    'group': g,
                    'mask_type': 'random',
                    'pearson': r_rand,
                    'mean_abs_delta': mad_rand,
                    'top_k_jaccard': jac_rand,
                    'pearson_pathway': r_pw_rand,
                    'jaccard_pathway': jac_pw_rand,
                })

            if verbose:
                msg = f'  {g}: pearson_deg={r_deg:.3f} jac_deg={jac_deg:.2f}'
                if z_rand is not None:
                    msg += f' | pearson_rand={r_rand:.3f} jac_rand={jac_rand:.2f}'
                if not np.isnan(r_pw_deg):
                    msg += f' | pw_pearson_deg={r_pw_deg:.3f}'
                    if z_rand is not None and not np.isnan(r_pw_rand):
                        msg += f' pw_pearson_rand={r_pw_rand:.3f}'
                print(msg)

        # Collect pathway-score values for the attrs table.
        if pw_deg is not None:
            for g in pw_deg.index:
                for pw in pw_deg.columns:
                    pathway_rows.append({
                        'n_top': n,
                        'group': g,
                        'pathway': pw,
                        'mask_type': 'deg',
                        'score': float(pw_deg.loc[g, pw]),
                    })
        if pw_rand is not None:
            for g in pw_rand.index:
                for pw in pw_rand.columns:
                    pathway_rows.append({
                        'n_top': n,
                        'group': g,
                        'pathway': pw,
                        'mask_type': 'random',
                        'score': float(pw_rand.loc[g, pw]),
                    })

        del z_deg
        if pw_deg is not None:
            del pw_deg
        if z_rand is not None:
            del z_rand
        if pw_rand is not None:
            del pw_rand

    out = pd.DataFrame(rows, columns=[
        'n_top', 'n_masked', 'group', 'mask_type',
        'pearson', 'mean_abs_delta', 'top_k_jaccard',
        'pearson_pathway', 'jaccard_pathway',
    ])
    if class_rows:
        out.attrs['pert_classification'] = pd.DataFrame(class_rows)
    if pathway_rows:
        out.attrs['pathway_scores'] = pd.DataFrame(pathway_rows)
    return out


def compute_pert_significance(
    adata: ad.AnnData,
    condition_col: str,
    soft_predictions_key: str = 'soft_predictions',
    baseline: Literal['other', 'all'] = 'other',
    method: Literal['wilcoxon', 'ttest'] = 'wilcoxon',
    fdr_method: Literal['bh', 'bonferroni'] | None = 'bh',
    n_chunks: int = 16,
) -> pd.DataFrame:
    """Per-(condition, perturbation) significance test on soft predictions.

    For each ``(condition, perturbation)`` cell, tests whether the
    per-cell soft predictions of cells *in* this condition are
    systematically higher than per-cell soft predictions of a
    baseline set. Answers the question: *"Is the model reliably more
    confident in this perturbation for cells in this condition than
    for the baseline?"* — exactly what you want to filter top-N
    predictions for confidence rather than just rank.

    Vectorised over perturbations via ``scipy.stats.rankdata``
    
    Parameters
    ----------
    adata
        Queried AnnData with ``obsm[soft_predictions_key]`` populated.
    condition_col
        Column in ``adata.obs`` defining the groups.
    baseline
        - ``'other'`` (default) — for each condition, compare against
          cells in all *other* conditions. Classic one-vs-rest.
        - ``'all'`` — compare against all cells including own condition
          (slightly more conservative; useful if you only have 1 group).
    method
        ``'wilcoxon'`` (Mann-Whitney rank-sum, recommended — robust
        to non-Gaussian softmax distributions) or ``'ttest'`` (Welch's
        t-test on per-cell soft probs).
    fdr_method
        ``'bh'`` (Benjamini-Hochberg, recommended), ``'bonferroni'``
        (very conservative), or ``None`` (no correction). FDR is
        computed per condition.
    n_chunks
        Chunks of perturbations processed at once when ranking. Lower
        if RAM-bound on very large matrices.

    Returns
    -------
    pd.DataFrame
        Long-form ``[condition, perturbation, mean_in, mean_out,
        fold, log2_fold, statistic, pvalue, qvalue]`` with one row per
        ``(condition, perturbation)``. Sorted by ``(condition, qvalue
        ascending)`` so the highest-confidence calls per condition are
        at the top.
    """
    from scipy.stats import rankdata, norm, ttest_ind

    if condition_col not in adata.obs:
        raise KeyError(f'condition_col {condition_col!r} not in adata.obs.')
    if soft_predictions_key not in adata.obsm:
        raise KeyError(
            f'{soft_predictions_key!r} not in adata.obsm — adata does not look '
            f'like the output of model.predict(). '
            f'Available obsm keys: {list(adata.obsm.keys())}'
        )

    soft = adata.obsm[soft_predictions_key]
    if hasattr(soft, 'columns'):
        pert_names = np.asarray(soft.columns).astype(str)
        soft_arr = np.asarray(soft.values, dtype=np.float32)
    else:
        soft_arr = np.asarray(soft, dtype=np.float32)
        pert_names = np.arange(soft_arr.shape[1]).astype(str)
    n_cells, n_perts = soft_arr.shape

    groups = adata.obs[condition_col].astype(str).values
    unique_groups = list(pd.unique(groups))
    n_groups = len(unique_groups)

    rows = []
    if method == 'wilcoxon':
        # Rank within each pert column (across cells) — single vectorized
        # pass. ``rankdata`` along axis=0 handles ties via 'average'.
        # Process in pert chunks to bound peak memory.
        chunk_size = max(1, n_perts // max(1, n_chunks))
        ranks_chunks = []
        for start in range(0, n_perts, chunk_size):
            end = min(start + chunk_size, n_perts)
            ranks_chunks.append(rankdata(soft_arr[:, start:end], axis=0))
        ranks = np.concatenate(ranks_chunks, axis=1).astype(np.float64)
        del ranks_chunks
        # Precompute mean profiles per group for fold-change.
        for cond in unique_groups:
            in_mask = groups == cond
            out_mask = ~in_mask if baseline == 'other' else np.ones_like(in_mask, dtype=bool)
            n_in = int(in_mask.sum())
            n_out = int(out_mask.sum())
            if n_in == 0 or n_out == 0:
                continue
            # Sum of ranks for in-condition cells, per pert.
            R1 = ranks[in_mask].sum(axis=0)
            # U statistic and asymptotic z.
            U1 = R1 - n_in * (n_in + 1) / 2.0
            mu_U = n_in * n_out / 2.0
            # Tie correction skipped — softmax distributions are
            # continuous enough that the uncorrected variance is fine.
            sigma_U = np.sqrt(n_in * n_out * (n_cells + 1) / 12.0)
            z = (U1 - mu_U) / sigma_U
            # Two-sided p-value via normal CDF.
            pvals = 2.0 * norm.sf(np.abs(z))
            mean_in = soft_arr[in_mask].mean(axis=0)
            mean_out = soft_arr[out_mask].mean(axis=0)
            fold = mean_in / np.where(mean_out > 0, mean_out, 1e-12)
            log2_fold = np.log2(np.where(fold > 0, fold, 1e-12))
            for j in range(n_perts):
                rows.append({
                    'condition': str(cond),
                    'perturbation': str(pert_names[j]),
                    'mean_in': float(mean_in[j]),
                    'mean_out': float(mean_out[j]),
                    'fold': float(fold[j]),
                    'log2_fold': float(log2_fold[j]),
                    'statistic': float(z[j]),
                    'pvalue': float(pvals[j]),
                })
    elif method == 'ttest':
        # Welch's t-test per (condition, pert). Less robust to non-normal
        # softmax but useful when you want a parametric output.
        for cond in unique_groups:
            in_mask = groups == cond
            out_mask = ~in_mask if baseline == 'other' else np.ones_like(in_mask, dtype=bool)
            if in_mask.sum() == 0 or out_mask.sum() == 0:
                continue
            x_in = soft_arr[in_mask]
            x_out = soft_arr[out_mask]
            stat, pvals = ttest_ind(x_in, x_out, equal_var=False, axis=0)
            mean_in = x_in.mean(axis=0)
            mean_out = x_out.mean(axis=0)
            fold = mean_in / np.where(mean_out > 0, mean_out, 1e-12)
            log2_fold = np.log2(np.where(fold > 0, fold, 1e-12))
            for j in range(n_perts):
                rows.append({
                    'condition': str(cond),
                    'perturbation': str(pert_names[j]),
                    'mean_in': float(mean_in[j]),
                    'mean_out': float(mean_out[j]),
                    'fold': float(fold[j]),
                    'log2_fold': float(log2_fold[j]),
                    'statistic': float(stat[j]),
                    'pvalue': float(pvals[j]),
                })
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'wilcoxon' or 'ttest'.")

    out = pd.DataFrame(rows)

    # FDR per condition (most natural per-condition multiple-testing scope).
    if fdr_method == 'bh' and not out.empty:
        # Benjamini-Hochberg within each condition.
        def _bh(pvals: np.ndarray) -> np.ndarray:
            m = len(pvals)
            order = np.argsort(pvals)
            ranks_p = np.arange(1, m + 1)
            q_sorted = pvals[order] * m / ranks_p
            # Enforce monotonicity (cummin from the right).
            q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
            q = np.empty(m)
            q[order] = np.minimum(q_sorted, 1.0)
            return q
        out['qvalue'] = np.concatenate([
            _bh(out.loc[out['condition'] == c, 'pvalue'].values)
            for c in unique_groups if (out['condition'] == c).any()
        ])
    elif fdr_method == 'bonferroni' and not out.empty:
        out['qvalue'] = np.minimum(out['pvalue'] * n_perts, 1.0)
    else:
        out['qvalue'] = out['pvalue']

    out = out.sort_values(['condition', 'qvalue'], ascending=[True, True]) \
             .reset_index(drop=True)
    return out


def compute_pairwise_degs(
    adata: ad.AnnData,
    condition_col: str,
    top_n_per_pair: int = 200,
    method: str = 'wilcoxon',
    pairwise: bool = True,
    baseline_group: str | None = None,
    return_per_group: bool = False,
) -> set[str] | dict[str, set[str]]:
    """Compute the union of (pairwise or vs-baseline or one-vs-rest) top-N DEG names.

    Shared building block for ``compute_deg_pathway_set`` and
    ``compare_pert_deg_overlap``. Lazy-imports scanpy. Auto log-normalises
    raw-count X before ranking (same heuristic as ``mask_top_degs``)
    so Wilcoxon isn't biased by library size.

    Parameters
    ----------
    baseline_group
        If provided (e.g. ``'Normal'``), only run ``baseline vs other``
        pairs — not all-vs-all. Produces a *narrower, disease-relevant*
        DEG set: genes that differ between the baseline and any
        non-baseline condition. The "what's wrong here vs healthy"
        framing rather than "anything that differs across conditions".
        Requires ``pairwise=True``.
    return_per_group
        When ``True`` (and ``pairwise=True``), return a dict
        ``{group: deg_set}`` mapping each non-baseline condition to
        its own ``baseline-vs-group`` DEG set. Lets downstream callers
        (e.g. the Venn per-condition test) use a condition-specific
        DEG baseline instead of the union. Falls back to single union
        when ``False`` (default).

    Returns
    -------
    set[str] or dict[str, set[str]]
        Union DEG name set, or per-group dict when
        ``return_per_group=True``.
    """
    import scanpy as sc

    if condition_col not in adata.obs:
        raise KeyError(f'condition_col {condition_col!r} not in adata.obs.')
    groups = list(pd.unique(adata.obs[condition_col].astype(str)))
    if len(groups) < 2:
        raise ValueError(f'Need ≥2 conditions in {condition_col!r}, got {groups}.')

    if baseline_group is not None:
        if not pairwise:
            raise ValueError(
                'baseline_group requires pairwise=True; one-vs-rest does '
                'not have a baseline anchor.'
            )
        if baseline_group not in groups:
            raise ValueError(
                f'baseline_group {baseline_group!r} not in {groups}.'
            )

    def _gather(sub: ad.AnnData) -> set[str]:
        sub = sub.copy()
        x = sub.X
        if issparse(x):
            x_max = x.max() if x.nnz > 0 else 0
        else:
            x_max = float(np.asarray(x).max()) if x.size > 0 else 0
        looks_raw = (
            x_max > 50
            or np.issubdtype(getattr(x, 'dtype', np.float32), np.integer)
        )
        if looks_raw:
            sc.pp.normalize_total(sub, target_sum=1e4)
            sc.pp.log1p(sub)
        sub.obs[condition_col] = sub.obs[condition_col].astype(str).astype('category')
        sc.tl.rank_genes_groups(sub, condition_col, method=method,
                                n_genes=top_n_per_pair, use_raw=False)
        names = sub.uns['rank_genes_groups']['names']
        out: set[str] = set()
        for g in names.dtype.names:
            out.update(np.asarray(names[g]).flatten().tolist())
        return out

    if not pairwise:
        return _gather(adata)

    # Pair generation depends on whether we're anchored to a baseline.
    if baseline_group is not None:
        pairs = [(baseline_group, g) for g in groups if g != baseline_group]
    else:
        pairs = [(groups[i], groups[j])
                 for i in range(len(groups))
                 for j in range(i + 1, len(groups))]

    if return_per_group:
        per_group: dict[str, set[str]] = {}
        for g_i, g_j in pairs:
            mask = adata.obs[condition_col].astype(str).isin([g_i, g_j])
            d = _gather(adata[mask])
            # Key on the *non-baseline* member when baseline is set;
            # otherwise key on the pair as "g_i_vs_g_j".
            if baseline_group is not None:
                key = g_j if g_i == baseline_group else g_i
            else:
                key = f'{g_i}_vs_{g_j}'
            per_group[key] = d
        return per_group

    deg_names: set[str] = set()
    for g_i, g_j in pairs:
        mask = adata.obs[condition_col].astype(str).isin([g_i, g_j])
        deg_names.update(_gather(adata[mask]))
    return deg_names


def compute_deg_pathway_set(
    adata: ad.AnnData,
    condition_col: str,
    library: str = 'MSigDB_Hallmark_2020',
    organism: str = 'Human',
    top_n_per_pair: int = 200,
    method: str = 'wilcoxon',
    pairwise: bool = True,
    baseline_group: str | None = None,
    min_pathway_size: int = 1,
    fdr_threshold: float = 0.05,
) -> tuple[set[str], pd.DataFrame]:
    """DEG-derived pathway baseline for comparing enrichment methods.

    Computes pairwise (or one-vs-rest) classical DEGs on the data, then
    asks which pathways are significantly over-represented in the union
    DEG list via hypergeometric test. The returned set is the "what
    standard DE would have given you" answer at the pathway level —
    useful as the baseline for evaluating how DEG-driven other methods
    (z-score, mean, frequency) are.

    Parameters
    ----------
    adata, condition_col
        Same as ``mask_top_degs``.
    library, organism
        Pathway library for the overlap test.
    top_n_per_pair
        Top-N DEGs per pairwise comparison (union taken).
    method
        DEG ranking method passed to ``sc.tl.rank_genes_groups``.
    pairwise
        If True, pairwise DEGs (more sensitive). If False, one-vs-rest.
    baseline_group
        If provided (e.g. ``'Normal'``), only run ``baseline vs
        other_group`` pairs — not all-vs-all. Restricts the DEG set
        to "what's different between disease and the chosen control"
        instead of "anything differential between any pair". Cleaner
        biological framing when you have a true control condition.
        Requires ``pairwise=True``.
    min_pathway_size
        Minimum pathway-member count (in the dataset's gene background).
    fdr_threshold
        BH-corrected p-value cutoff for "this pathway is DEG-enriched".

    Returns
    -------
    qualifying : set[str]
        Pathway names with BH-FDR ≤ ``fdr_threshold`` overlap.
    details : pd.DataFrame
        Per-pathway: ``[pathway, n_overlap, n_pathway, n_degs, pvalue, qvalue]``.
        Sorted by pvalue ascending.

        ``details.attrs['deg_genes']`` carries the underlying DEG name
        set (a ``set[str]``) computed by ``compute_pairwise_degs`` —
        reuse it directly via
        ``compare_pert_deg_overlap(..., deg_genes=details.attrs['deg_genes'])``
        to avoid a redundant DEG pass over the same adata.
    """
    from scipy.stats import hypergeom

    deg_names = compute_pairwise_degs(
        adata, condition_col,
        top_n_per_pair=top_n_per_pair,
        method=method,
        pairwise=pairwise,
        baseline_group=baseline_group,
    )

    # Load library and run hypergeometric per pathway.
    lib_obj = get_library(library=library, organism=organism)
    if lib_obj is None:
        raise RuntimeError(f'Could not load library {library!r}.')

    bg_upper = {str(v).upper() for v in adata.var_names}
    n_bg = len(bg_upper)
    deg_upper = {str(d).upper() for d in deg_names} & bg_upper
    n_degs = len(deg_upper)

    rows = []
    for pw, members in lib_obj.items():
        members_in_bg = {str(m).upper() for m in members} & bg_upper
        n_pw_genes = len(members_in_bg)
        if n_pw_genes < min_pathway_size:
            continue
        overlap = members_in_bg & deg_upper
        n_overlap = len(overlap)
        # hypergeom.sf(k-1, M, n, N): P(X >= k) with M=bg, n=pathway, N=DEGs
        pval = float(hypergeom.sf(n_overlap - 1, n_bg, n_pw_genes, n_degs))
        rows.append({
            'pathway': pw,
            'n_overlap': n_overlap,
            'n_pathway': n_pw_genes,
            'n_degs': n_degs,
            'pvalue': pval,
        })
    details = pd.DataFrame(rows).sort_values('pvalue').reset_index(drop=True)

    # Benjamini-Hochberg FDR.
    if not details.empty:
        m = len(details)
        ranks = np.arange(1, m + 1)
        qvals = (details['pvalue'].values * m / ranks)
        # Enforce monotonicity from the back
        qvals = np.minimum.accumulate(qvals[::-1])[::-1]
        details['qvalue'] = np.minimum(qvals, 1.0)
    else:
        details['qvalue'] = []

    qualifying = set(details.loc[details['qvalue'] <= fdr_threshold, 'pathway'])
    # Attach the underlying DEG gene set so downstream callers
    # (e.g. compare_pert_deg_overlap) can reuse it without another
    # rank_genes_groups pass over the same data.
    details.attrs['deg_genes'] = deg_names
    return qualifying, details


def compare_pathway_enrichments(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = 'A',
    label_b: str = 'B',
    top_k: int = 15,
    deg_pathways: set[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side comparison of two ``plot_pathway_enrichment`` outputs.

    Quantifies how much two scoring modes agree, and (optionally) how
    much each one looks like a classical DEG-based pathway analysis.
    Typical use: compare ``score='mean_z'`` vs ``score='frequency'``
    on the same query, then pass the DEG-derived pathway baseline from
    ``compute_deg_pathway_set`` to test the "frequency is less
    DEG-dependent" claim.

    Per condition, computes:

    - Top-K Jaccard between the two top-pathway lists
    - Pearson correlation of pathway scores on the union of top-Ks
    - When ``deg_pathways`` is supplied, fraction of each mode's top-K
      that overlaps the DEG-baseline pathway set

    Parameters
    ----------
    df_a, df_b
        Outputs of ``plot_pathway_enrichment`` (rows=groups, cols=top-K
        pathways with the truncated ``"name (n)"`` labels).
    label_a, label_b
        Display labels for the two modes (e.g., ``'mean_z'`` /
        ``'frequency'``); used as column-name suffixes.
    top_k
        Number of top pathways per mode and condition to compare.
    deg_pathways
        Optional set of pathway names — typically returned by
        ``compute_deg_pathway_set`` — to gate DEG-overlap fractions.
        Matching is case-insensitive and ignores the ``" (n)"`` suffix.

    Returns
    -------
    pd.DataFrame
        One row per condition. Columns:

        ``[group, top_k_jaccard, pearson_common, n_top_{label_a},
        n_top_{label_b}, n_intersection, n_union]``.

        With ``deg_pathways`` set, two extra columns:
        ``[deg_overlap_{label_a}, deg_overlap_{label_b}]``.

        ``df.attrs['pathway_classification']`` holds the long-form
        per-pathway agreement (which mode flagged each pathway, in
        which conditions). Helpful for surfacing the "method-unique"
        hits for the poster.
    """
    if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
        raise TypeError('df_a and df_b must be DataFrames from plot_pathway_enrichment.')

    def _strip_size_suffix(name: str) -> str:
        """'TNF-alpha Signaling via NF-kB (92)' -> 'TNF-alpha Signaling via NF-kB'."""
        s = str(name)
        if s.endswith(')') and '(' in s:
            return s.rsplit('(', 1)[0].rstrip()
        return s

    def _normalize_orientation(df: pd.DataFrame) -> pd.DataFrame:
        """Return df oriented as ``rows=groups, cols=pathways``.

        ``plot_pathway_enrichment(transpose=True)`` returns
        ``rows=pathways, cols=groups`` with ``df.index.name='pathway'``;
        the non-transposed version uses ``df.index.name=condition_col``.
        We accept either orientation and transpose if needed.
        """
        if str(getattr(df.index, 'name', '') or '') == 'pathway':
            return df.T
        # Heuristic fallback when index.name isn't set: column-count
        # looks like pathway-count (usually ≫ group-count).
        if len(df.index) > len(df.columns):
            # likely rows=pathways → transpose
            return df.T
        return df

    df_a = _normalize_orientation(df_a)
    df_b = _normalize_orientation(df_b)

    # Sanity warning when the requested top_k exhausts the input pool —
    # in that case per-group "top-K" collapses to "all columns" and the
    # Jaccard / DEG-overlap will be identical across conditions.
    n_pw_a, n_pw_b = df_a.shape[1], df_b.shape[1]
    if top_k >= min(n_pw_a, n_pw_b):
        logging.warning(
            f'compare_pathway_enrichments: top_k={top_k} >= number of '
            f'pathway columns in df_a ({n_pw_a}) / df_b ({n_pw_b}). '
            f'Per-group top-K will return all columns and be identical '
            f'across conditions. Re-run plot_pathway_enrichment with '
            f'top_k_pathways much greater than {top_k} (e.g. the full '
            f'library size) for meaningful per-group comparisons.'
        )

    # Build (original_label, plain_name) maps for both dataframes.
    a_cols = list(df_a.columns)
    b_cols = list(df_b.columns)
    a_plain = {c: _strip_size_suffix(c).upper() for c in a_cols}
    b_plain = {c: _strip_size_suffix(c).upper() for c in b_cols}
    deg_plain = (
        {_strip_size_suffix(p).upper() for p in deg_pathways}
        if deg_pathways is not None else None
    )

    common_groups = [g for g in df_a.index if g in df_b.index]
    if not common_groups:
        raise ValueError(
            f'No common condition groups between the two DataFrames after '
            f'orientation normalization. df_a rows: {list(df_a.index)}, '
            f'df_b rows: {list(df_b.index)}.'
        )
    rows = []
    classification: list[dict] = []
    for g in common_groups:
        # Top-K pathways per mode (ranked by |score|).
        a_top = (df_a.loc[g].abs().nlargest(top_k))
        b_top = (df_b.loc[g].abs().nlargest(top_k))
        a_top_plain = {a_plain[c] for c in a_top.index}
        b_top_plain = {b_plain[c] for c in b_top.index}

        inter = a_top_plain & b_top_plain
        union = a_top_plain | b_top_plain
        jacc = len(inter) / max(len(union), 1)

        # Pearson on scores over the union (using whatever name the
        # column has in each df; resolve via the plain map).
        a_plain_to_orig = {v: k for k, v in a_plain.items()}
        b_plain_to_orig = {v: k for k, v in b_plain.items()}
        a_vec, b_vec = [], []
        for pw in sorted(union):
            if pw in a_plain_to_orig and pw in b_plain_to_orig:
                a_vec.append(df_a.loc[g, a_plain_to_orig[pw]])
                b_vec.append(df_b.loc[g, b_plain_to_orig[pw]])
        if len(a_vec) >= 2 and np.std(a_vec) > 1e-9 and np.std(b_vec) > 1e-9:
            r = float(np.corrcoef(a_vec, b_vec)[0, 1])
        else:
            r = float('nan')

        row = {
            'group': g,
            'top_k_jaccard': jacc,
            'pearson_common': r,
            f'n_top_{label_a}': len(a_top_plain),
            f'n_top_{label_b}': len(b_top_plain),
            'n_intersection': len(inter),
            'n_union': len(union),
        }
        if deg_plain is not None:
            a_in_deg = len(a_top_plain & deg_plain)
            b_in_deg = len(b_top_plain & deg_plain)
            row[f'deg_overlap_{label_a}'] = a_in_deg / max(len(a_top_plain), 1)
            row[f'deg_overlap_{label_b}'] = b_in_deg / max(len(b_top_plain), 1)
        rows.append(row)

        # Per-pathway classification within this condition.
        for pw in sorted(union):
            classification.append({
                'group': g,
                'pathway': pw,
                f'in_{label_a}': pw in a_top_plain,
                f'in_{label_b}': pw in b_top_plain,
                f'score_{label_a}': (
                    float(df_a.loc[g, a_plain_to_orig[pw]])
                    if pw in a_plain_to_orig else float('nan')
                ),
                f'score_{label_b}': (
                    float(df_b.loc[g, b_plain_to_orig[pw]])
                    if pw in b_plain_to_orig else float('nan')
                ),
                'in_deg': (pw in deg_plain) if deg_plain is not None else None,
            })

    out = pd.DataFrame(rows)
    if classification:
        out.attrs['pathway_classification'] = pd.DataFrame(classification)
    return out


def compare_pert_deg_overlap(
    adata: ad.AnnData,
    condition_col: str,
    pert_dfs: dict[str, pd.DataFrame],
    top_k: int = 50,
    deg_genes: set[str] | None = None,
    pert_vocab: set[str] | list[str] | None = None,
    pert_vocab_restrict: bool = False,
    top_n_per_pair: int = 200,
    method: str = 'wilcoxon',
    pairwise: bool = True,
    score_axis: Literal['abs', 'signed'] = 'abs',
) -> pd.DataFrame:
    """Per-mode, per-condition gene-level DEG overlap of top-K perturbations.

    Computes ``"what fraction of each mode's top-K predicted
    perturbations overlap the classical pairwise DEG list?"`` —
    the gene-level analogue of the pathway-level overlap in
    ``compare_pathway_enrichments``. Useful for the headline claim:
    z-score predictions sit close to classical DE, frequency / mean
    predictions sit further away.

    Parameters
    ----------
    adata, condition_col
        Source data + grouping column. Used to compute the DEG list
        when ``deg_genes`` isn't supplied.
    pert_dfs
        Mapping from mode name (used as a label) to the per-group
        perturbation z-score / mean / frequency DataFrame returned by
        ``plot_perturbation_enrichment``. Rows=conditions, columns=
        perturbations (gene names). All values are treated as scores;
        ranking is done by the chosen ``score_axis``.

        Example: ``{'zscore': df_z, 'frequency': df_f, 'mean': df_m}``.
    top_k
        Per (mode, condition), take the top-K perturbations.
    deg_genes
        Optional pre-computed DEG name set. When ``None``, computed
        via ``compute_pairwise_degs`` (top_n_per_pair / method /
        pairwise control the call).
    score_axis
        ``'abs'`` (default) — rank by |score|; works for z-score
        (where negatives mean "depleted") as well as mean / frequency.
        ``'signed'`` — rank by raw score descending; appropriate when
        you want "top by enrichment" only.

    pert_vocab
        Full perturbation vocabulary the model can predict (typically
        ``model.idx_to_label`` or
        ``query.obsm['soft_predictions'].columns``). Used as the
        *background population* for hypergeometric enrichment of
        observed top-K ∩ DEG overlap. When ``None``, falls back to
        the union of all input ``pert_dfs`` columns — only correct
        when those dfs carry the *full* vocab rather than a top-K
        filtered subset.

    Returns
    -------
    pd.DataFrame
        Long-form with one row per ``(mode, condition)``. Columns:

        ``[mode, condition, top_k, n_overlap, fraction_in_deg,
        deg_set_size, vocab_size, deg_in_vocab, expected_overlap,
        fold_enrichment, pvalue]``

        The hypergeometric test compares observed ``n_overlap``
        against ``Hypergeom(M=vocab_size, n=deg_in_vocab, N=top_k)``
        and reports right-tail ``P(X ≥ n_overlap)``.

        Attached via ``df.attrs``:

        - ``deg_genes`` — the DEG gene set used (so callers can
          inspect / reuse it).
        - ``top_perts`` — long-form ``[mode, condition, perturbation,
          score, rank, in_deg]`` for the top-K perts that went into
          the overlap counting.
    """
    from scipy.stats import hypergeom
    if not isinstance(pert_dfs, dict) or not pert_dfs:
        raise TypeError('pert_dfs must be a non-empty dict of mode_name -> DataFrame.')

    # Sanity warning: when top_k matches (or exceeds) the column count
    # of an input df, the per-condition top-K collapses to "all
    # columns" and the resulting sets are identical across conditions.
    min_cols = min(d.shape[1] for d in pert_dfs.values())
    if top_k >= min_cols:
        logging.warning(
            f'compare_pert_deg_overlap: top_k={top_k} >= min(pert_dfs cols)={min_cols}. '
            f'Per-condition top-K will return ALL columns and be identical '
            f'across conditions. Re-run plot_perturbation_enrichment with '
            f'top_k much greater than {top_k} (e.g. 500) so per-condition '
            f'ranking has room.'
        )

    if deg_genes is None:
        deg_genes = compute_pairwise_degs(
            adata, condition_col,
            top_n_per_pair=top_n_per_pair,
            method=method,
            pairwise=pairwise,
        )
    deg_upper = {str(g).upper() for g in deg_genes}

    # Resolve background vocabulary for hypergeometric enrichment.
    # Prefer caller-supplied pert_vocab; fall back to the union of all
    # input columns (only correct when the dfs carry the full vocab).
    if pert_vocab is None:
        union_cols: set[str] = set()
        for d in pert_dfs.values():
            union_cols.update(str(c) for c in d.columns)
        pert_vocab = union_cols
    vocab_upper = {str(v).upper() for v in pert_vocab}

    # Optional restriction: keep only perts that are *also* in
    # adata.var_names. This makes the test fair — non-gene perts (drugs,
    # controls) can never be DEGs by construction, so leaving them in
    # the background inflates ``vocab_size`` and depresses expected
    # overlap. Restricting also filters the input pert_dfs so the top-K
    # is taken from the same restricted space.
    if pert_vocab_restrict:
        var_upper = {str(v).upper() for v in adata.var_names}
        before = len(vocab_upper)
        vocab_upper = vocab_upper & var_upper
        logging.info(
            f'[compare_pert_deg_overlap] pert_vocab_restrict=True: '
            f'vocab {before} → {len(vocab_upper)} (kept perts in var_names).'
        )
        # Filter each df's columns to those in the restricted vocab.
        restricted_dfs = {}
        for mode, df in pert_dfs.items():
            keep_cols = [c for c in df.columns
                         if str(c).upper() in vocab_upper]
            restricted_dfs[mode] = df[keep_cols]
        pert_dfs = restricted_dfs

    vocab_size = len(vocab_upper)
    deg_in_vocab = len(deg_upper & vocab_upper)

    rows = []
    top_pert_rows = []
    for mode, df in pert_dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'pert_dfs[{mode!r}] must be a DataFrame.')
        for g in df.index:
            scores = df.loc[g]
            if score_axis == 'abs':
                ranked = scores.abs().sort_values(ascending=False)
            else:
                ranked = scores.sort_values(ascending=False)
            top = ranked.head(top_k)
            top_names = [str(c).upper() for c in top.index]
            n_overlap = sum(1 for n in top_names if n in deg_upper)
            n_top = len(top)
            # Hypergeometric enrichment: P(X >= n_overlap) when drawing
            # n_top items from a pool of vocab_size, of which
            # deg_in_vocab are "successes" (DEGs).
            if vocab_size > 0 and n_top > 0:
                expected = n_top * deg_in_vocab / vocab_size
                fold = (n_overlap / n_top) / (deg_in_vocab / vocab_size) \
                    if deg_in_vocab > 0 else float('nan')
                # Right-tail p-value (enrichment direction).
                pval = float(hypergeom.sf(
                    n_overlap - 1, vocab_size, deg_in_vocab, n_top
                ))
            else:
                expected = float('nan')
                fold = float('nan')
                pval = float('nan')
            rows.append({
                'mode': mode,
                'condition': str(g),
                'top_k': n_top,
                'n_overlap': n_overlap,
                'fraction_in_deg': n_overlap / max(n_top, 1),
                'deg_set_size': len(deg_upper),
                'vocab_size': vocab_size,
                'deg_in_vocab': deg_in_vocab,
                'expected_overlap': expected,
                'fold_enrichment': fold,
                'pvalue': pval,
            })
            for rank, (name, _score) in enumerate(top.items(), start=1):
                top_pert_rows.append({
                    'mode': mode,
                    'condition': str(g),
                    'perturbation': str(name),
                    'score': float(scores.loc[name]),
                    'rank': rank,
                    'in_deg': str(name).upper() in deg_upper,
                })

    out = pd.DataFrame(rows)
    out.attrs['deg_genes'] = deg_genes
    out.attrs['top_perts'] = pd.DataFrame(top_pert_rows)
    return out


def _build_pathway_dfs_from_perts(
    pert_dfs: dict[str, pd.DataFrame],
    pert_vocab: list[str] | set[str],
    library: str | dict,
    organism: str = 'Human',
    top_k_perts: int = 100,
    top_k_pathways: int = 40,
    score_axis: Literal['abs', 'signed'] = 'abs',
    min_pathway_size: int = 1,
) -> dict[str, pd.DataFrame]:
    """Build per-mode pathway-score dfs from pert-level top-N via membership.

    For each (mode, condition):
      - Take the model's top-N perturbations (by score_axis)
      - For each pathway, count how many of those perts are members
      - Take the top_k_pathways with highest counts
      - Build a per-condition row in the synthetic df

    The resulting per-mode dfs are shaped like real
    ``plot_pathway_enrichment`` outputs (rows=conditions,
    columns=pathway labels), so downstream code in
    ``compare_pathway_deg_overlap`` works unchanged. The pathway "score"
    in this synthetic df is the member-count itself.
    """
    if isinstance(library, str):
        lib_obj = get_library(library=library, organism=organism)
        if lib_obj is None:
            raise RuntimeError(f'Could not load library {library!r}.')
    else:
        lib_obj = library

    pert_upper = [str(p).upper() for p in pert_vocab]
    pert_upper_set = set(pert_upper)

    # Pre-compute pathway → set of pert names (upper) restricted to vocab.
    pathway_to_pert_names: dict[str, set[str]] = {}
    pathway_sizes: dict[str, int] = {}
    for pw, members in lib_obj.items():
        pset = {str(m).upper() for m in members} & pert_upper_set
        if len(pset) >= min_pathway_size:
            pathway_to_pert_names[pw] = pset
            pathway_sizes[pw] = len(pset)
    if not pathway_to_pert_names:
        raise RuntimeError(
            f'No pathways with ≥{min_pathway_size} members in pert_vocab.'
        )

    out: dict[str, pd.DataFrame] = {}
    for mode, df in pert_dfs.items():
        per_cond_rows: dict[str, dict[str, float]] = {}
        for cond in df.index:
            scores = df.loc[cond]
            ranked = (
                scores.abs() if score_axis == 'abs' else scores
            ).sort_values(ascending=False)
            top_perts_upper = {str(p).upper() for p in ranked.head(top_k_perts).index}

            # Count member intersections per pathway.
            counts = {
                pw: len(pset & top_perts_upper)
                for pw, pset in pathway_to_pert_names.items()
            }
            # Top-K pathways by count (with pathway-size suffix label for
            # downstream display consistency).
            top_pws = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k_pathways]
            row = {f'{pw} ({pathway_sizes[pw]})': float(c) for pw, c in top_pws}
            per_cond_rows[str(cond)] = row

        # Per-condition unions of column names → build the df with all
        # columns; missing entries get 0.
        all_cols = sorted({c for d in per_cond_rows.values() for c in d})
        rows_data = {
            cond: [per_cond_rows[cond].get(c, 0.0) for c in all_cols]
            for cond in per_cond_rows
        }
        synth = pd.DataFrame.from_dict(rows_data, orient='index',
                                       columns=all_cols)
        synth.index.name = list(df.index.names)[0] if df.index.name else 'condition'
        out[mode] = synth
    return out


def compare_pathway_deg_overlap(
    pathway_dfs: dict[str, pd.DataFrame] | None = None,
    deg_pathways: set[str] | None = None,
    top_k: int = 15,
    score_axis: Literal['abs', 'signed'] = 'abs',
    pathway_universe_size: int | None = None,
    library: str | None = None,
    organism: str = 'Human',
    pvalue_method: Literal['hypergeom', 'pert_sampling'] = 'hypergeom',
    pert_dfs: dict[str, pd.DataFrame] | None = None,
    pert_vocab: list[str] | set[str] | None = None,
    top_k_perts: int = 100,
    n_permutations: int = 200,
    seed: int = 0,
    derive_pathways_from_perts: bool = False,
    min_pathway_size: int = 5,
) -> pd.DataFrame:
    """Per-mode, per-condition pathway-level DEG overlap.

    Pathway-level analogue of ``compare_pert_deg_overlap``: takes one
    pathway-enrichment DataFrame per scoring mode, computes the top-K
    pathways per ``(mode, condition)``, and reports how many of them
    are in the DEG-derived pathway baseline.

    Expected pattern (paired with the gene-level Venn):

    - Gene-level overlap with DEGs is *small* — model predictions
      identify perturbations whose phenotype matches the cell state,
      which is a different question than which genes are
      differentially expressed.
    - Pathway-level overlap *should be substantially larger* — both
      methods land on the same biological pathway structure, even when
      the gene-level effectors differ.

    The Venn-diagram contrast between gene-level (small intersection)
    and pathway-level (large intersection) is the cleanest visual
    statement of "different upstream effectors, same downstream biology."

    Parameters
    ----------
    pathway_dfs
        Mapping ``{mode_name: pathway_enrichment_df}``. Each df should
        be the return of ``plot_pathway_enrichment``. Accepts either
        orientation — index=conditions (default) or index=pathways
        (``transpose=True`` output) — and normalizes internally.
    deg_pathways
        Set of pathway names from ``compute_deg_pathway_set``. Matching
        ignores the ``" (n)"`` size suffix the plot adds to column
        labels.
    top_k
        Top-K pathways per ``(mode, condition)`` to compare.
    score_axis
        ``'abs'`` (default) — rank by ``|score|``. Use this for
        ``mean_z``/``wilcoxon`` modes where negatives mean "depleted".
        ``'signed'`` — rank by raw score descending. Use this when
        you only care about positively-enriched pathways
        (``mean``/``frequency`` modes).
    min_pathway_size
        Drop pathways with fewer than this many member perts in the
        prediction vocabulary before any analysis runs. Applies
        consistently to ``_build_pathway_dfs_from_perts`` (when
        ``derive_pathways_from_perts=True``) and the sampling null.
        Default 5 — filters singleton / near-empty pathways that
        dominate rank metrics for trivial reasons. Set to 1 to keep
        all pathways that have any vocab member.

    Returns
    -------
    pd.DataFrame
        Long-form ``[mode, condition, top_k, n_overlap,
        fraction_in_deg, deg_set_size]``. One row per ``(mode,
        condition)``.

        Attached to ``df.attrs``:

        - ``deg_pathways`` — the input pathway set (reusable).
        - ``top_pathways`` — per ``(mode, condition, pathway, score,
          rank, in_deg)``, the top-K pathways. Drop-in input for the
          ``plot_pathway_deg_venn`` helper.
    """
    if deg_pathways is None:
        raise ValueError('deg_pathways is required.')

    # When `derive_pathways_from_perts=True`, the displayed pathway sets
    # come from membership counting on the model's top-N perts (same perts
    # the pert-level test uses). This makes the Venn circles and the
    # p-value annotation coherent — both reflect the same underlying
    # predictions. Falls through to a synthetic pathway_dfs built from
    # the pert-derived top-K so the rest of the function works unchanged.
    if derive_pathways_from_perts:
        if pert_dfs is None or pert_vocab is None or library is None:
            raise ValueError(
                "derive_pathways_from_perts=True requires pert_dfs, "
                "pert_vocab, and library."
            )
        pathway_dfs = _build_pathway_dfs_from_perts(
            pert_dfs=pert_dfs,
            pert_vocab=pert_vocab,
            library=library,
            organism=organism,
            top_k_perts=top_k_perts,
            top_k_pathways=top_k,
            score_axis=score_axis,
            min_pathway_size=min_pathway_size,
        )

    if not isinstance(pathway_dfs, dict) or not pathway_dfs:
        raise TypeError(
            'pathway_dfs must be a non-empty dict of mode_name -> DataFrame, '
            'OR pass derive_pathways_from_perts=True with pert_dfs.'
        )

    def _strip_size_suffix(name: str) -> str:
        s = str(name)
        if s.endswith(')') and '(' in s:
            return s.rsplit('(', 1)[0].rstrip()
        return s

    def _normalize_orientation(df: pd.DataFrame) -> pd.DataFrame:
        # plot_pathway_enrichment(transpose=True) returns rows=pathways.
        # Normalize to rows=conditions for uniform downstream handling.
        if str(getattr(df.index, 'name', '') or '') == 'pathway':
            return df.T
        if len(df.index) > len(df.columns):
            return df.T
        return df

    normalized = {m: _normalize_orientation(d) for m, d in pathway_dfs.items()}

    # Same per-condition-collapse warning as the pert version.
    min_cols = min(d.shape[1] for d in normalized.values())
    if top_k >= min_cols:
        logging.warning(
            f'compare_pathway_deg_overlap: top_k={top_k} >= min(pathway_dfs cols)={min_cols}. '
            f'Per-condition top-K will return ALL columns and be identical '
            f'across conditions. Re-run plot_pathway_enrichment with '
            f'top_k_pathways much greater than {top_k} (e.g. the full '
            f'library size) so per-condition ranking has room.'
        )

    from scipy.stats import hypergeom

    deg_plain = {_strip_size_suffix(p).upper() for p in deg_pathways}

    # Resolve pathway universe size for the hypergeometric test.
    # Priority: explicit ``pathway_universe_size`` → load ``library`` →
    # fall back to the union of df columns (conservative — only the
    # displayed top-K survived the upstream filter, so this overstates
    # significance and emits a warning).
    if pathway_universe_size is None and library is not None:
        lib_obj = get_library(library=library, organism=organism)
        if lib_obj is not None:
            pathway_universe_size = len(lib_obj)
    if pathway_universe_size is None:
        union_cols: set[str] = set()
        for d in normalized.values():
            union_cols.update(_strip_size_suffix(c).upper() for c in d.columns)
        pathway_universe_size = len(union_cols)
        logging.warning(
            f'compare_pathway_deg_overlap: pathway_universe_size derived from '
            f'displayed df columns (={pathway_universe_size}); this overstates '
            f'significance because the input dfs are already top-K filtered. '
            f'Pass `library=...` or `pathway_universe_size=...` for a proper '
            f'background.'
        )
    logging.info(f'Pathway library size: {pathway_universe_size}')

    deg_in_universe = min(len(deg_plain), pathway_universe_size)

    rows = []
    top_path_rows = []
    for mode, df in normalized.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'pathway_dfs[{mode!r}] must be a DataFrame.')
        for g in df.index:
            scores = df.loc[g]
            if score_axis == 'abs':
                ranked = scores.abs().sort_values(ascending=False)
            else:
                ranked = scores.sort_values(ascending=False)
            top = ranked.head(top_k)
            top_plain = [_strip_size_suffix(c).upper() for c in top.index]
            n_overlap = sum(1 for n in top_plain if n in deg_plain)
            n_top = len(top)
            # Hypergeometric: P(X >= n_overlap | universe, DEG-pathways, top_k)
            if pathway_universe_size > 0 and n_top > 0 and deg_in_universe > 0:
                expected = n_top * deg_in_universe / pathway_universe_size
                fold = (n_overlap / n_top) / (deg_in_universe / pathway_universe_size)
                pval = float(hypergeom.sf(
                    n_overlap - 1, pathway_universe_size,
                    deg_in_universe, n_top,
                ))
            else:
                expected = float('nan')
                fold = float('nan')
                pval = float('nan')
            rows.append({
                'mode': mode,
                'condition': str(g),
                'top_k': n_top,
                'n_overlap': n_overlap,
                'fraction_in_deg': n_overlap / max(n_top, 1),
                'deg_set_size': len(deg_plain),
                'universe_size': pathway_universe_size,
                'deg_in_universe': deg_in_universe,
                'expected_overlap': expected,
                'fold_enrichment': fold,
                'pvalue': pval,
            })
            for rank, (name, _score) in enumerate(top.items(), start=1):
                plain = _strip_size_suffix(name).upper()
                top_path_rows.append({
                    'mode': mode,
                    'condition': str(g),
                    'pathway': _strip_size_suffix(name),
                    'pathway_label': str(name),    # original "(n)"-suffixed label
                    'score': float(scores.loc[name]),
                    'rank': rank,
                    'in_deg': plain in deg_plain,
                })

    out = pd.DataFrame(rows)
    out.attrs['deg_pathways'] = deg_pathways
    out.attrs['top_pathways'] = pd.DataFrame(top_path_rows)

    # --- Optional: replace hypergeometric pvalues with sampling-based ones ---
    # The pert-level sampling test ("top-N perts ∩ DEG-pathway-member-union"
    # vs random N perts) — gives the meaningful enrichment p-value the
    # user kept. The Venn diagram circles meanwhile display the
    # pathway-level overlap (from ``attrs['top_pathways']``); the two
    # measure related but distinct quantities on the same input perts.
    # The Venn subtitle labels its numbers as ``pert-level`` to avoid
    # confusion with the visible circle counts.
    if pvalue_method == 'pert_sampling':
        if pert_dfs is None or pert_vocab is None:
            raise ValueError(
                "pvalue_method='pert_sampling' requires both pert_dfs (per-pert "
                "score dfs) and pert_vocab (full perturbation vocabulary)."
            )
        if library is None:
            raise ValueError(
                "pvalue_method='pert_sampling' requires `library` so the "
                "sampling test can build pathway memberships from the same "
                "library used for the displayed pathway analysis."
            )
        sampling = pathway_overlap_pert_sampling_test(
            pert_dfs=pert_dfs,
            pert_vocab=pert_vocab,
            deg_pathways=deg_pathways,
            library=library,
            organism=organism,
            top_k_perts=top_k_perts,
            top_k_pathways=top_k,
            n_permutations=n_permutations,
            seed=seed,
            score_axis=score_axis,
            min_pathway_size=min_pathway_size,
        )
        key_cols = ['mode', 'condition']
        sampling_indexed = sampling.set_index(key_cols)[
            ['observed_overlap', 'null_mean', 'null_std', 'p_emp',
             'deg_pathway_size', 'universe_size']
        ]
        # Keep the pathway-level overlap as ``pathway_n_overlap`` for
        # cross-reference with the Venn circles; ``n_overlap`` itself
        # is replaced by the pert-level test's observed value.
        pathway_n_overlap = out.set_index(key_cols)['n_overlap'].rename('pathway_n_overlap')
        out_indexed = out.set_index(key_cols)
        out_indexed['n_overlap'] = sampling_indexed['observed_overlap']
        out_indexed['expected_overlap'] = sampling_indexed['null_mean']
        out_indexed['null_std'] = sampling_indexed['null_std']
        out_indexed['pvalue'] = sampling_indexed['p_emp']
        out_indexed['deg_set_size'] = sampling_indexed['deg_pathway_size']
        out_indexed['universe_size'] = sampling_indexed['universe_size']
        with np.errstate(divide='ignore', invalid='ignore'):
            out_indexed['fold_enrichment'] = (
                out_indexed['n_overlap'].astype(float)
                / out_indexed['expected_overlap'].astype(float)
            )
        out_indexed['pathway_n_overlap'] = pathway_n_overlap
        out = out_indexed.reset_index()
        out.attrs['deg_pathways'] = deg_pathways
        out.attrs['top_pathways'] = pd.DataFrame(top_path_rows)
        out.attrs['null_overlaps'] = sampling.attrs.get('null_overlaps')
        # Forward pert-level Venn inputs so plot_pathway_deg_venn can
        # render the second-row pert panel when ``pert_panel=True``.
        out.attrs['top_perts'] = sampling.attrs.get('top_perts')
        out.attrs['deg_pw_members'] = sampling.attrs.get('deg_pw_members')
        out.attrs['pvalue_method'] = 'pert_sampling'
    else:
        out.attrs['pvalue_method'] = 'hypergeom'

    return out


def pathway_overlap_pert_sampling_test(
    pert_dfs: dict[str, pd.DataFrame],
    pert_vocab: list[str] | set[str],
    deg_pathways: set[str],
    *,
    library: str | dict = 'MSigDB_Hallmark_2020',
    organism: str = 'Human',
    top_k_perts: int = 100,
    top_k_pathways: int = 15,
    min_pathway_size: int = 5,
    n_permutations: int = 200,
    seed: int = 0,
    score_axis: Literal['abs', 'signed'] = 'abs',
) -> pd.DataFrame:
    """Empirical null for pathway-DEG overlap via random pert sampling.

    The "fair" null for testing whether the model's pathway-level
    convergence with DEG-pathways is more than what *any* random
    selection of K perts would produce. Unlike the analytical
    hypergeometric on the Venn (which assumes uniform pathway
    sampling), this respects the fact that:

    - Perturbations have non-uniform pathway membership counts
    - Pathways have non-uniform sizes (some absorb random samples more
      easily than others)
    - Not all perts even map to a pathway

    For each permutation we sample ``top_k_perts`` random perts from
    ``pert_vocab``, rank pathways by how many of the sample are
    members, take the top ``top_k_pathways``, and count overlap with
    ``deg_pathways``. The observed value for each ``(mode, condition)``
    uses the model's actual top-N perts through the same membership
    pipeline, so the test is apples-to-apples.

    Parameters
    ----------
    pert_dfs
        ``{mode_name: per-pert df}`` — same shape as
        ``compare_pert_deg_overlap``. Top-N per (mode, condition) is
        taken from each.
    pert_vocab
        Full perturbation vocabulary the model can predict (e.g.
        ``list(model.idx_to_label)`` or
        ``query.obsm['soft_predictions'].columns``).
    deg_pathways
        DEG-derived pathway set (from ``compute_deg_pathway_set``).
    library
        gseapy library name OR a preloaded dict (``{pathway: members}``).
    top_k_perts
        Per (mode, condition), top-N perts to use as "predictions".
    top_k_pathways
        Number of top pathways to extract via membership ranking.
    n_permutations
        Number of random pert samples (200 is enough for ~0.01 p-value
        resolution).
    seed
        RNG seed.

    Returns
    -------
    pd.DataFrame
        One row per ``(mode, condition)``. Columns:
        ``[mode, condition, top_k_perts, top_k_pathways,
        observed_overlap, null_mean, null_std, p_emp,
        deg_pathway_size, universe_size]``.

        ``out.attrs['null_overlaps']`` = ``(n_permutations,)`` array
        (the null is condition-independent — same random sampling
        applies to every condition's test).
    """
    if not isinstance(pert_dfs, dict) or not pert_dfs:
        raise TypeError('pert_dfs must be a non-empty dict of mode_name -> DataFrame.')

    # Load library if a name was passed.
    if isinstance(library, str):
        lib_obj = get_library(library=library, organism=organism)
        if lib_obj is None:
            raise RuntimeError(f'Could not load library {library!r}.')
    else:
        lib_obj = library

    pert_vocab = list(pert_vocab)
    pert_upper = [str(p).upper() for p in pert_vocab]
    pert_to_idx = {p: i for i, p in enumerate(pert_upper)}
    n_vocab = len(pert_vocab)

    # Build pathway membership in the pert-index space.
    def _strip_size(name: str) -> str:
        s = str(name)
        if s.endswith(')') and '(' in s:
            return s.rsplit('(', 1)[0].rstrip()
        return s

    pathway_to_pert_idx: dict[str, set[int]] = {}
    for pw, members in lib_obj.items():
        idx_set = {
            pert_to_idx[str(m).upper()]
            for m in members
            if str(m).upper() in pert_to_idx
        }
        if len(idx_set) >= min_pathway_size:
            pathway_to_pert_idx[pw] = idx_set
    if not pathway_to_pert_idx:
        raise RuntimeError(
            f'No pathways with ≥{min_pathway_size} members in the pert vocab.'
        )
    pathway_names = list(pathway_to_pert_idx.keys())
    n_pathways = len(pathway_names)

    deg_plain = {_strip_size(p).upper() for p in deg_pathways}
    # Build the *union* of perts that belong to any DEG pathway — this is
    # the actual "success" set against which we count overlap. Counting
    # "DEG pathways hit" instead would be biased against coherent model
    # predictions (they hit fewer distinct pathways than spread-out random
    # samples) — using the pert union removes that spread bias entirely.
    deg_pw_member_set: set[int] = set()
    n_deg_pathways_in_universe = 0
    for pw, idx_set in pathway_to_pert_idx.items():
        if _strip_size(pw).upper() in deg_plain:
            deg_pw_member_set.update(idx_set)
            n_deg_pathways_in_universe += 1
    deg_in_universe = len(deg_pw_member_set)

    def _topk_overlap(pert_idx_set: set[int]) -> int:
        """Number of perts in pert_idx_set that are members of ANY DEG pathway.

        Pert-level enrichment against the union of DEG-pathway members.
        Unbiased by pathway spread — each pert contributes once
        regardless of how many DEG pathways it's in. Bounded by
        ``min(len(pert_idx_set), len(deg_pw_member_set))``.

        This is equivalent to the hypergeometric test on the gene
        level with ``deg_genes = ∪ DEG-pathway members``.
        """
        return len(pert_idx_set & deg_pw_member_set)

    # --- Null distribution (one shared, condition-invariant) ---
    rng = np.random.default_rng(seed)
    sample_size = min(top_k_perts, n_vocab)
    null_overlaps = np.zeros(n_permutations, dtype=np.int32)
    for p in range(n_permutations):
        sample_idx = set(rng.choice(n_vocab, size=sample_size, replace=False).tolist())
        null_overlaps[p] = _topk_overlap(sample_idx)
    null_mean = float(null_overlaps.mean())
    null_std = float(null_overlaps.std())

    # --- Observed per (mode, condition) ---
    # Also track the top-K perts (long-form) and the DEG-pathway-member
    # union as pert NAMES so downstream Venn plotters can visualize the
    # gene-level overlap that produces the empirical p-value.
    rows = []
    top_pert_rows = []
    # Build the DEG-pathway-member union as a name set for downstream.
    deg_pw_member_names = {
        pert_upper[i] for i in deg_pw_member_set
    }
    for mode, df in pert_dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'pert_dfs[{mode!r}] must be a DataFrame.')
        for g in df.index:
            scores = df.loc[g]
            if score_axis == 'abs':
                ranked = scores.abs().sort_values(ascending=False)
            else:
                ranked = scores.sort_values(ascending=False)
            top_perts = list(ranked.head(top_k_perts).index)
            top_idx_set = {
                pert_to_idx[str(p).upper()]
                for p in top_perts
                if str(p).upper() in pert_to_idx
            }
            observed = _topk_overlap(top_idx_set)
            # +1/+1 smoothing prevents empirical p=0 when no null
            # surpasses the observed.
            p_emp = (
                int(np.sum(null_overlaps >= observed)) + 1
            ) / (n_permutations + 1)
            rows.append({
                'mode': mode,
                'condition': str(g),
                'top_k_perts': top_k_perts,
                'top_k_pathways': top_k_pathways,
                'observed_overlap': observed,
                'null_mean': null_mean,
                'null_std': null_std,
                'p_emp': float(p_emp),
                # ``deg_pathway_size`` (legacy column name) now reports the
                # number of perts in the union-of-DEG-pathway-members
                # success set — that's what the test counts against.
                # ``n_deg_pathways`` reports the original pathway count.
                'deg_pathway_size': deg_in_universe,
                'n_deg_pathways': n_deg_pathways_in_universe,
                'universe_size': n_vocab,
            })
            for rank, pert in enumerate(top_perts, start=1):
                p_upper = str(pert).upper()
                top_pert_rows.append({
                    'mode': mode,
                    'condition': str(g),
                    'perturbation': str(pert),
                    'rank': rank,
                    'in_deg_pw_members': p_upper in deg_pw_member_names,
                })

    out = pd.DataFrame(rows)
    out.attrs['null_overlaps'] = null_overlaps
    out.attrs['library'] = library if isinstance(library, str) else 'custom_dict'
    out.attrs['top_perts'] = pd.DataFrame(top_pert_rows)
    out.attrs['deg_pw_members'] = deg_pw_member_names
    return out


def pathway_overlap_permutation_test(
    adata: ad.AnnData,
    condition_col: str,
    deg_pathways: set[str],
    *,
    library: str = 'MSigDB_Hallmark_2020',
    organism: str = 'Human',
    soft_predictions_key: str = 'soft_predictions',
    score: Literal['mean_z', 'mean', 'frequency'] = 'mean',
    top_k_metric: Literal['max_abs', 'std', 'range', 'vs_baseline'] = 'max_abs',
    baseline_group: str | None = None,
    top_k_pathways: int = 15,
    top_per_cell: int = 1,
    min_pathway_size: int = 1,
    n_permutations: int = 200,
    seed: int = 0,
    pert_vocab_restrict: bool = False,
) -> pd.DataFrame:
    """Permutation + hypergeometric test for pathway-DEG overlap.

    Addresses the "is the pathway-level convergence real?" question
    with two complementary null models:

    1. **Permutation null** — for ``n_permutations`` iterations, each
       pathway is reassigned a random set of perturbations (same size
       as the real pathway), the model's per-pert scores stay fixed,
       and the pathway-overlap analysis re-runs. Asks: *"would any
       plausible pathway structure give the same DEG-pathway overlap
       given these per-pert scores?"* — directly tests whether the
       *actual* pert→pathway mapping is what carries the signal.

    2. **Hypergeometric** — analytical p-value for observed top-K ∩
       DEG given the pathway universe. Asks: *"is this enrichment
       above what random sampling of K pathways would give?"*

    Parameters mirror ``plot_pathway_enrichment`` so the test runs
    the *same* pathway-scoring logic that produced your Venn.

    Parameters
    ----------
    deg_pathways
        DEG-derived pathway set from ``compute_deg_pathway_set``.
    score / top_k_metric / baseline_group / top_k_pathways /
    top_per_cell / min_pathway_size
        Match the call you used in ``plot_pathway_enrichment``.
    n_permutations
        Number of pathway-membership shuffles for the empirical null
        (200 is usually enough for 0.01-resolution p-values).
    pert_vocab_restrict
        If True, restrict the pert score matrix and the pathway-member
        universe to perturbations that are *also* in ``adata.var_names``.
        Sanity check that the signal isn't an artifact of perts the
        model could never have linked to genes.

    Returns
    -------
    pd.DataFrame
        One row per condition. Columns:
        ``[condition, top_k, observed_overlap,
        null_mean, null_std, p_emp,
        deg_pathway_size, universe_size, p_hypergeom, fold_hypergeom]``

        ``out.attrs['null_overlaps']`` = ``(n_permutations, n_groups)``
        array of per-condition null overlap counts (useful for plotting
        the null distribution alongside the observed value).
    """
    from scipy.stats import hypergeom

    if condition_col not in adata.obs:
        raise KeyError(f'condition_col {condition_col!r} not in adata.obs.')
    if soft_predictions_key not in adata.obsm:
        raise KeyError(f'{soft_predictions_key!r} not in adata.obsm.')

    soft = adata.obsm[soft_predictions_key]
    if hasattr(soft, 'columns'):
        pert_names = np.asarray(soft.columns).astype(str)
        soft_arr = np.asarray(soft.values)
    else:
        soft_arr = np.asarray(soft)
        pert_names = np.arange(soft_arr.shape[1]).astype(str)

    # Optional restriction to perts that exist as genes in adata.var_names.
    if pert_vocab_restrict:
        var_upper = {str(v).upper() for v in adata.var_names}
        keep_idx = np.array([
            i for i, p in enumerate(pert_names)
            if str(p).upper() in var_upper
        ])
        if len(keep_idx) < min_pathway_size:
            raise ValueError(
                f'After pert_vocab_restrict, only {len(keep_idx)} perts '
                f'remain — below min_pathway_size={min_pathway_size}.'
            )
        pert_names = pert_names[keep_idx]
        soft_arr = soft_arr[:, keep_idx]

    groups = adata.obs[condition_col].astype(str).values
    unique_groups = pd.unique(groups)
    n_groups = len(unique_groups)
    n_perts = soft_arr.shape[1]

    # Per-pert per-group score matrix matching plot_pathway_enrichment.
    profiles = np.zeros((n_groups, n_perts), dtype=np.float32)
    freq = None
    if score == 'frequency':
        freq = np.zeros((n_groups, n_perts), dtype=np.float32)
    for i, g in enumerate(unique_groups):
        mask = groups == g
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        sub = soft_arr[mask]
        profiles[i] = sub.astype(np.float32).mean(axis=0)
        if freq is not None:
            tpc = max(1, min(int(top_per_cell), n_perts))
            if tpc == 1:
                idx = sub.argmax(axis=1)
            else:
                idx = np.argpartition(sub, -tpc, axis=1)[:, -tpc:]
            counts = np.bincount(np.asarray(idx).ravel(), minlength=n_perts)
            freq[i] = counts.astype(np.float32) / float(n_in)

    if score == 'mean_z':
        mu = profiles.mean(axis=0); sd = profiles.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        pert_score_matrix = (profiles - mu) / sd
    elif score == 'mean':
        pert_score_matrix = profiles
    elif score == 'frequency':
        pert_score_matrix = freq
    else:
        raise ValueError(f"Unknown score: {score!r}.")

    # Pathway library and member indices (restricted to current pert pool).
    lib_obj = get_library(library=library, organism=organism)
    if lib_obj is None:
        raise RuntimeError(f'Could not load library {library!r}.')

    pert_upper_to_idx = {str(p).upper(): i for i, p in enumerate(pert_names)}
    pathway_members: dict[str, np.ndarray] = {}
    for pw, members in lib_obj.items():
        idx = [pert_upper_to_idx[str(m).upper()] for m in members
               if str(m).upper() in pert_upper_to_idx]
        if len(idx) >= min_pathway_size:
            pathway_members[pw] = np.asarray(idx, dtype=np.int64)
    if not pathway_members:
        raise RuntimeError(
            f'No pathways with ≥{min_pathway_size} members. Lower threshold.'
        )

    pathway_names = list(pathway_members.keys())
    n_pw = len(pathway_names)

    # --- Helper to compute pathway_score → top-K → overlap ---
    def _strip_size(name: str) -> str:
        s = str(name)
        if s.endswith(')') and '(' in s:
            return s.rsplit('(', 1)[0].rstrip()
        return s
    deg_plain = {_strip_size(p).upper() for p in deg_pathways}
    pathway_in_deg = np.array(
        [_strip_size(p).upper() in deg_plain for p in pathway_names]
    )

    def _topk_per_group(pw_scores: np.ndarray) -> np.ndarray:
        """Return (n_groups, top_k_pathways) array of pathway indices."""
        out = np.zeros((n_groups, top_k_pathways), dtype=np.int64)
        for i in range(n_groups):
            # max_abs ranking per group is what compare_pathway_deg_overlap
            # uses; the test is consistent with the displayed Venn.
            order = np.argsort(-np.abs(pw_scores[i]))
            out[i] = order[:top_k_pathways]
        return out

    def _pw_scores(members_dict: dict[str, np.ndarray]) -> np.ndarray:
        # Aggregate per-pathway mean of pert scores (same as the visible mode).
        pws = np.zeros((n_groups, n_pw), dtype=np.float32)
        for j, pw in enumerate(pathway_names):
            members = members_dict[pw]
            pws[:, j] = pert_score_matrix[:, members].mean(axis=1)
        return pws

    # --- Observed ---
    observed_scores = _pw_scores(pathway_members)
    observed_topk = _topk_per_group(observed_scores)
    observed_overlap = np.array([
        int(pathway_in_deg[observed_topk[i]].sum())
        for i in range(n_groups)
    ])

    # --- Permutations: shuffle each pathway's members ---
    rng = np.random.default_rng(seed)
    perm_overlaps = np.zeros((n_permutations, n_groups), dtype=np.int32)
    for p in range(n_permutations):
        perm_members = {
            pw: rng.choice(n_perts, size=len(members), replace=False)
            for pw, members in pathway_members.items()
        }
        perm_scores = _pw_scores(perm_members)
        perm_topk = _topk_per_group(perm_scores)
        for i in range(n_groups):
            perm_overlaps[p, i] = int(pathway_in_deg[perm_topk[i]].sum())

    null_mean = perm_overlaps.mean(axis=0)
    null_std = perm_overlaps.std(axis=0)
    # Empirical right-tail p (with +1 / +1 smoothing to avoid p=0).
    p_emp = (np.sum(perm_overlaps >= observed_overlap[None, :], axis=0) + 1) \
        / (n_permutations + 1)

    # --- Hypergeometric vs the pathway universe ---
    deg_in_universe = int(pathway_in_deg.sum())
    rows = []
    for i, g in enumerate(unique_groups):
        p_hg = float(hypergeom.sf(
            observed_overlap[i] - 1, n_pw, deg_in_universe, top_k_pathways
        ))
        fold = (
            (observed_overlap[i] / max(top_k_pathways, 1))
            / (deg_in_universe / max(n_pw, 1))
            if deg_in_universe > 0 else float('nan')
        )
        rows.append({
            'condition': str(g),
            'top_k': top_k_pathways,
            'observed_overlap': int(observed_overlap[i]),
            'null_mean': float(null_mean[i]),
            'null_std': float(null_std[i]),
            'p_emp': float(p_emp[i]),
            'deg_pathway_size': len(deg_plain),
            'universe_size': n_pw,
            'p_hypergeom': p_hg,
            'fold_hypergeom': fold,
        })

    out = pd.DataFrame(rows)
    out.attrs['null_overlaps'] = perm_overlaps
    out.attrs['conditions'] = list(unique_groups)
    return out


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
        verbose: bool = False,
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

    # Top-K only: a full argsort over (n_cells, n_classes) is (n_cells, n_classes)
    # int64 = ~222GB at our scale (4M cells × 7K classes × 8 bytes), which OOMs
    # the box. We only ever read `max_idx[:, -top_n:]` for top_n in 1..n, so
    # store only the K largest indices per row, ordered ascending so the last
    # column is the maximum (matches the original `max_idx[:, -top_n:]` convention).
    K = min(int(n), n_classes)
    pred_vals = predictions.values
    # argpartition: K largest values' indices land at positions [-K:] (no order within).
    partitioned = np.argpartition(pred_vals, -K, axis=1)[:, -K:]            # (N, K)
    top_K_vals = np.take_along_axis(pred_vals, partitioned, axis=1)         # (N, K)
    order_within = np.argsort(top_K_vals, axis=1)                            # (N, K) ascending
    max_idx = np.take_along_axis(partitioned, order_within, axis=1)          # (N, K)
    del partitioned, top_K_vals, order_within

    # Random baseline for top-N: original code argsort'd a full
    # (n_cells, n_classes) random matrix per sample (~222GB each × 10 samples).
    # We only need K random class indices per cell — uniform sampling with
    # replacement is fine (collision rate K(K-1)/(2*n_classes) ≈ 0.6% at our
    # scale, negligible for chance-baseline metrics).
    random_max_idxs = [
        np.random.randint(0, n_classes, size=(n_cells, K), dtype=np.int32)
        for _ in range(n_random_samples)
    ]
    
    top_n_predictions = []
    splits = (
        [None] if split_key is None
        else [s for s in _adata.obs[split_key].unique() if pd.notna(s)]
    )
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

