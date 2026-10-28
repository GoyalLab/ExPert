# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

"""Disease readout: turning `predict()` into a scored statement about a disease query.

Applying a CRISPR-LoF-trained model to a disease atlas is not a prediction problem -- nobody
knocked anything out, so there is no recall to report. What can be reported is which trained
perturbations the disease response points at, and whether that is more than chance would give.
Every dataset we have done this for (alcoholic liver disease, the van Galen AML atlas) re-derived
the same scaffold by hand, and each re-derivation reintroduced the same handful of mistakes. This
module is that scaffold, with the mistakes encoded as defaults.

Everything is written back onto the query AnnData, the way `model.predict(query, inplace=True)`
and scanpy do it -- `obs` for per-cell labels, `uns[uns_key]` for the group-level results -- so a
readout travels with the object and survives a `write_h5ad`. Every step returns `None` unless
`copy=True`.

    import scanpy as sc
    import expert.utils.readout as ro
    from expert import ExPert

    model = ExPert.load(model_dir)
    query = sc.read('disease.h5ad')

    ro.build_groups(query, 'Condition', 'CellType', control_condition='Normal')
    ro.mark_query(model, query)
    model.predict(query, inplace=True, return_latents=True)
    ro.group_response(query)
    ro.score_prototypes(model, query, contrast='delta')
    ro.module_test(query, modules)
    ro.text_enrichment(query, descriptions)

    ro.get_scores(query, 'delta')          # groups x perturbations, as a DataFrame
    query.uns['expert_readout']['module_test']

PERTURBATION PROTOTYPES are the vectors the model scores a response against -- one per trained
perturbation, in the shared latent space. Which concrete object that is depends on how the
checkpoint was trained, and reading the wrong one reads an UNTRAINED head, so it is resolved by
the model rather than rebuilt here:

    atoms     `random_init_proxies` checkpoints put the loss weight on a learnable prototype table.
    adapter   text-alignment checkpoints train the aligner's class projection (cls2z). Only a
              forward pass produces it, so it is read from `obsm['cls_proj']`, which
              `predict(inplace=True)` writes.
    ema       `class_centers`, a running mean that NO loss shapes. Almost never what you want;
              kept because older readouts used it.

FOUR THINGS ARE DEFAULTS HERE BECAUSE GETTING THEM WRONG COST US REAL RESULTS

    1  Under `delta` a control group is baselined against itself, so its response is exactly the
       origin -- not a group with a small response, the ZERO vector. Cosines against it are
       meaningless and a pursuit loop selects nothing. Controls are dropped for delta structurally
       rather than special-cased downstream; the same bug was independently reintroduced three
       times before that.
    2  Prototypes come from the model, never from a local reconstruction. A reconstruction that
       drifts produces a full set of plausible numbers attributed to prototypes the model does
       not have.
    3  The default profile is the top-K cosine prototypes weighted BY COSINE, not OMP
       coefficients. Measured on ALD and AML across two checkpoints: OMP explains far more
       variance at matched K (~0.52 vs ~0.35) but its text enrichment scores at the
       shuffled-description null (vs_null ~1.0 against cosine's 2.0-2.8) and it loses module hit
       rate in 5 of 6 configurations. Orthogonalisation drops the correlated partners that a set
       statistic counts. Use `nnomp` for the support question -- which prototypes JOINTLY explain
       a response -- not for scoring.
    4  `specificity` is only interpretable when `vs_null` exceeds 1. Sparse, near-random profiles
       decorrelate across groups and so score HIGH on specificity while carrying nothing;
       `text_enrichment` reports the two together and warns when specificity is being read off a
       readout that is at the null.
    5  BOTH SIGNED TAILS are profiled. Prototypes are loss-of-function directions and every
       selection rule here is non-negative, so a readout keeping only positive cosines cannot say
       "this pathway is MORE active" -- it can only fail to mention it. Measured on Weinberg PBMC
       the gain tail carries MORE text signal than the loss tail (vs_null 2.11 vs 1.23), is not its
       mirror, and is where the respiratory-chain signal sits in T/NK. Every consumer takes `tail=`
       and defaults to 'loss', so adding the gain profile changes nothing already written.

RANKING IS ABUNDANCE-STRATIFIED once you ask for it. The cosine barely tracks training support
(|rho| ~ 0.05 on ALD and psoriasis) but the TOP of the list does: top-40 lists average the 66th
support percentile and the best-supported decile is named ~30x more often than the worst.
`attach_prototype_covariates` stores support and hubness on the query; `score_prototypes` and
`top_perturbations` then default to `stratify='auto'`, which z-scores every prototype within its
own covariate decile (`stratified_z`) so a hit must be an outlier among equally-represented genes.
Nothing is attached unless the caller asks, so this is a no-op otherwise, and the raw cosines are
always stored alongside.

Only `score_prototypes` needs the model. The remaining steps read what is already on the AnnData,
so they re-run without a GPU or a second forward pass.

CENTERING, MEASURED (Weinberg Leigh PBMC, GRAPH0 + EFFGATE3, 2026-09-14). For checkpoints trained on
control-centred input (encoder `center_mode='ctrl'`), the forward pass is the place to get right:

    build_groups(...) and mark_query(...) on the FULL query, then per cell type (per batch x cell
    type when every batch has that type's controls):
        model.predict(sub, inplace=True, return_latents=True, obs_only=True,
                      center_mode=np.zeros(n_shared), control_label='control')
    concatenate the z_shared back onto the full query, subtract its WHOLE-query mean, and only then
    group_response / score_prototypes.

One pooled call made the `mean` contrast identity-dominated (disease vs control Spearman 0.96) and
blind to the known mTORC1 / Complex III signal; per cell type it recovered both, with `delta`
unchanged either way. The latent centre never changes `delta` (a per-cell-type constant cancels);
it only moves `mean`. Never centre z_shared on a per-cell-type all-cell mean, which is what
`center_mode='query_mean'` does if left on inside per-cell-type calls. Why and the numbers:
`ExPert.predict` Notes; harness data/inference/metabolic_weinberg/centering_audit.py.
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CONTROL_PREFIX = "__control__"
CONTRASTS = ("mean", "delta", "group_specific")
PROFILE_METHODS = ("cosine_topk", "cosine_nnls", "omp")
TAILS = ("loss", "gain")
UNS_KEY = "expert_readout"
COV_KEY = "prototype_covariates"
GROUP_KEY = "expert_group"
LATENT_KEY = "ExPert_latent_z_shared"
CLS_PROJ_KEY = "cls_proj"


def _ns(adata, uns_key: str = UNS_KEY, create: bool = False) -> dict:
    """The readout's namespace in `adata.uns`."""
    if uns_key not in adata.uns:
        if not create:
            raise KeyError(
                f"adata.uns[{uns_key!r}] not found -- run readout.group_response(adata) first "
                f"(and build_groups / mark_query / predict before that)."
            )
        adata.uns[uns_key] = {}
    return adata.uns[uns_key]


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def _celltype(key: str) -> str:
    return key.split("|", 1)[1] if "|" in key else "all"


def _check_tail(tail: str) -> str:
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}, got {tail!r}")
    return tail


def _pkey(contrast: str, tail: str = "loss") -> str:
    """Storage key for one (contrast, tail).

    The loss tail keeps the bare contrast as its key, so every result written before signed tails
    existed reads back unchanged.
    """
    return contrast if _check_tail(tail) == "loss" else f"{contrast}|{tail}"


def _resolve_covariate(adata, stratify, uns_key: str = UNS_KEY):
    """`(name, values)` for the covariate a ranking is stratified within, or `(None, None)`.

    'auto' prefers training support over hubness -- support is the bias that was measured to move
    the top of the list, hubness the one that was measured not to.
    """
    if stratify is None or stratify is False:
        return None, None
    if not isinstance(stratify, str):
        return "custom", np.asarray(stratify, np.float64)
    covs = (_ns(adata, uns_key).get(COV_KEY) or {}) if uns_key in getattr(adata, "uns", {}) else {}
    if stratify == "auto":
        for name in ("support", "hubness"):
            if name in covs:
                return name, np.asarray(covs[name], np.float64)
        return None, None
    if stratify not in covs:
        raise KeyError(f"no prototype covariate {stratify!r} -- call "
                       f"attach_prototype_covariates(...) first; attached: {sorted(covs)}")
    return stratify, np.asarray(covs[stratify], np.float64)


# ======================================================================================
# groups
# ======================================================================================
def build_groups(
    adata,
    condition_key: str,
    celltype_key: str | None = None,
    control_condition: str | None = None,
    key_added: str = GROUP_KEY,
    copy: bool = False,
):
    """Label cells `'<condition>|<celltype>'`, controls `'__control__|<celltype>'`.

    The rename is what makes the baseline cell-type-matched: a disease group is later compared
    against the control group carrying the SAME suffix. Without it the only available baseline is
    pooled over cell types, and in every atlas we have looked at the disease shifts composition,
    so a pooled baseline measures the shift in cell-type mix rather than the disease.

    Writes
    ------
    adata.obs[key_added]            per-cell group label
    adata.obs[key_added + '_flat']  the same with every control collapsed to the literal
                                    `'__control__'`. `mark_query` matches the control group by
                                    exact string while the matched baseline needs the suffix, so
                                    both spellings are stored rather than recomputed per call.
    """
    adata = adata.copy() if copy else adata
    cond = adata.obs[condition_key].astype(str).to_numpy()
    if celltype_key is not None and celltype_key in adata.obs:
        ct = adata.obs[celltype_key].astype(str).to_numpy()
    else:
        if celltype_key is not None:
            log.warning("celltype_key=%r not in obs; all cells share one cell type", celltype_key)
        ct = np.full(adata.n_obs, "all", dtype=object)

    groups = np.array([
        f"{CONTROL_PREFIX}|{t}" if (control_condition is not None and c == control_condition)
        else f"{c}|{t}"
        for c, t in zip(cond, ct)
    ], dtype=object)
    n_ctrl = int(np.char.startswith(groups.astype(str), CONTROL_PREFIX).sum())
    if control_condition is not None and n_ctrl == 0:
        raise ValueError(
            f"control_condition={control_condition!r} matched no cells in obs[{condition_key!r}]; "
            f"present values: {sorted(set(cond))[:12]}"
        )
    adata.obs[key_added] = pd.Categorical(groups)
    adata.obs[f"{key_added}_flat"] = pd.Categorical(
        np.where(np.char.startswith(groups.astype(str), CONTROL_PREFIX), CONTROL_PREFIX,
                 groups.astype(str))
    )
    log.info("build_groups: %d cells -> %d groups (%d control cells) in obs[%r]",
             adata.n_obs, len(set(groups)), n_ctrl, key_added)
    return adata if copy else None


def mark_query(model, adata, group_key: str = GROUP_KEY, uns_key: str = UNS_KEY,
               copy: bool = False):
    """Write the model's own label/batch columns onto the query, carrying ONLY control identity.

    Two separate mechanisms need to know which cells are controls: an encoder trained with
    `center_mode='ctrl'` subtracts a control mean from its input, and the decode subtracts the
    control mean of the shared latent. If no control cells are marked both silently fall back to
    a baseline derived from TRAINING data -- a different biological state -- which removes most of
    the disease signature while still returning plausible numbers.

    Every non-control cell gets a filler label, so no real gene name can reach the model through
    `obs`: the query must be decoded from expression alone.

    The batch column is DROPPED rather than set to the query's name. `predict()` fills a missing
    categorical with the first known training category so the dataloader gets a valid index, then
    refreshes that row's control statistics from this query's own controls. Registering a novel
    category instead sends `predict()` down a buffer-extension path that grows the encoder's
    per-dataset buffers but not the module's alignment means, and centering then indexes a row
    that does not exist.
    """
    adata = adata.copy() if copy else adata
    flat_key = f"{group_key}_flat"
    if flat_key not in adata.obs:
        raise KeyError(f"obs[{flat_key!r}] not found -- call build_groups(...) first")
    groups = adata.obs[flat_key].astype(str).to_numpy()

    cls_col = model.adata_manager.get_state_registry("labels").original_key
    known = set(map(str, np.asarray(model.idx_to_label)))
    filler = str(model.idx_to_label[0])
    lab = np.array([filler] * adata.n_obs, dtype=object)
    is_ctrl = groups == CONTROL_PREFIX
    ctrl_label = "control"
    if ctrl_label not in known:
        ctrl_label = str(getattr(model, "ctrl_class", None) or ctrl_label)
    lab[is_ctrl] = ctrl_label
    adata.obs[cls_col] = pd.Categorical(lab)

    enc = getattr(model.module, "z_encoder", None)
    centers = (getattr(enc, "center_input", False)
               and str(getattr(enc, "center_mode", "")).startswith("ctrl"))
    log.info("mark_query: labels -> obs[%r]: %d control (%r), %d filler; encoder centers=%s",
             cls_col, int(is_ctrl.sum()), ctrl_label, int((~is_ctrl).sum()), centers)
    if int(is_ctrl.sum()) == 0:
        raise ValueError(
            "no control cells marked -- the decode baseline and the encoder centering both need "
            "them. Pass control_condition= to build_groups()."
        )
    try:
        batch_col = model.adata_manager.get_state_registry("batch").original_key
    except Exception:
        batch_col = None
    if batch_col and batch_col in adata.obs:
        adata.obs.drop(columns=[batch_col], inplace=True)
        log.info("mark_query: dropped obs[%r] so predict() assigns its own placeholder batch",
                 batch_col)
    ns = _ns(adata, uns_key, create=True)
    ns["control_label"] = ctrl_label
    ns["label_column"] = cls_col
    return adata if copy else None


# ======================================================================================
# response
# ======================================================================================
def group_response(
    adata,
    group_key: str = GROUP_KEY,
    latent_key: str = LATENT_KEY,
    latent: "np.ndarray | None" = None,
    min_cells: int = 100,
    uns_key: str = UNS_KEY,
    copy: bool = False,
):
    """Aggregate the per-cell shared latent into per-group means.

    Reads the latent `model.predict(adata, inplace=True, return_latents=True)` already wrote, so
    this and everything after it re-run without a GPU or a second forward pass.

    Groups below `min_cells` are dropped -- a pseudobulk over a handful of cells is mostly
    sampling noise and would otherwise pass silently into every downstream test. Control groups
    are kept regardless of size, because dropping a cell type's control quietly converts every
    disease group of that type to the pooled baseline.

    Writes into `adata.uns[uns_key]`
    -------------------------------
    groups, response                  (n_groups,) labels and (n_groups, d) means
    control_groups, control_response  the same for the control arm, per cell type
    n_cells                           cells behind each group
    """
    adata = adata.copy() if copy else adata
    if latent is None and latent_key not in adata.obsm:
        raise KeyError(
            f"obsm[{latent_key!r}] not found -- run model.predict(adata, inplace=True, "
            f"return_latents=True) first (it writes the shared latent there), or pass "
            f"latent=<array> if you called predict(inplace=False) and hold the array yourself."
        )
    if group_key not in adata.obs:
        raise KeyError(f"obs[{group_key!r}] not found -- call build_groups(...) first")

    Z = np.asarray(adata.obsm[latent_key] if latent is None else latent, np.float64)
    g = adata.obs[group_key].astype(str).to_numpy()
    counts = pd.Series(g).value_counts()
    keys = sorted(k for k in counts.index
                  if not k.startswith(CONTROL_PREFIX) and counts[k] >= min_cells)
    dropped = sorted(k for k in counts.index
                     if not k.startswith(CONTROL_PREFIX) and counts[k] < min_cells)
    ctrl_keys = sorted(k for k in counts.index if k.startswith(CONTROL_PREFIX))
    if dropped:
        log.info("group_response: dropped %d group(s) under min_cells=%d: %s", len(dropped),
                 min_cells, ", ".join(f"{k}({counts[k]})" for k in dropped[:8]))
    if not keys:
        raise ValueError(f"no group reached min_cells={min_cells}; largest is {counts.max()}")

    ns = _ns(adata, uns_key, create=True)
    ns["groups"] = np.array(keys)
    ns["response"] = np.stack([Z[g == k].mean(0) for k in keys]).astype(np.float32)
    ns["control_groups"] = np.array(ctrl_keys)
    ns["control_response"] = (np.stack([Z[g == k].mean(0) for k in ctrl_keys]).astype(np.float32)
                              if ctrl_keys else np.zeros((0, Z.shape[1]), np.float32))
    ns["n_cells"] = np.array([int((g == k).sum()) for k in keys])
    ns["min_cells"] = int(min_cells)
    log.info("group_response: %d groups, %d control groups, latent dim %d",
             len(keys), len(ctrl_keys), Z.shape[1])
    return adata if copy else None


def baseline(adata, keys=None, uns_key: str = UNS_KEY) -> np.ndarray:
    """The cell-type-matched control mean for each group, falling back to the pooled control."""
    ns = _ns(adata, uns_key)
    cQ = np.asarray(ns["control_response"], np.float64)
    if not len(cQ):
        raise ValueError("no control groups: a delta contrast needs a control_condition")
    keys = [str(k) for k in ns["groups"]] if keys is None else [str(k) for k in keys]
    by_ct = {_celltype(str(k)): cQ[i] for i, k in enumerate(ns["control_groups"])}
    pooled = cQ.mean(0)
    missing = sorted({_celltype(k) for k in keys if _celltype(k) not in by_ct})
    if missing:
        log.warning("no matched control for cell types %s -- pooled baseline used there", missing)
    return np.stack([by_ct.get(_celltype(k), pooled) for k in keys])


def get_response(adata, contrast: str = "delta", uns_key: str = UNS_KEY):
    """`(keys, R)` for one contrast of the stored group responses.

    mean            the response as-is. "What does this group look like."
    delta           minus the cell-type-matched control. "What is different about this group's
                    disease state." Controls are EXCLUDED: baselined against themselves they are
                    the origin, not a group, and every downstream cosine and pursuit step is
                    undefined on a zero vector.
    group_specific  minus the mean of the OTHER groups. Shared structure cancels. A trade-off,
                    not an improvement -- it sharpens gene-level readouts and weakens module
                    tests, because module statistics are set-based over correlated prototypes and
                    part of that correlation is the shared axis being removed. Measure both ways
                    before relying on it.
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, got {contrast!r}")
    ns = _ns(adata, uns_key)
    keys = [str(k) for k in ns["groups"]]
    Q = np.asarray(ns["response"], np.float64)
    if contrast == "mean":
        return keys, Q
    if contrast == "delta":
        return keys, Q - baseline(adata, keys, uns_key)
    other = (Q.sum(0, keepdims=True) - Q) / max(len(keys) - 1, 1)
    return keys, Q - other


# ======================================================================================
# perturbation prototypes
# ======================================================================================
def perturbation_prototypes(model, adata=None, which: str = "auto",
                            cls_proj_key: str = CLS_PROJ_KEY):
    """`(source, labels, P)` -- the vectors the model scores a response against.

    One row per trained perturbation, in the shared latent space. Resolved through the model so
    it cannot drift from what the model's own decode uses; see the module docstring for why the
    two checkpoint families expose different objects and what reading the wrong one costs.

    `which='adapter'` needs the aligner's class projection, which only a forward pass produces --
    it is read from `adata.obsm[cls_proj_key]`, written by `predict(inplace=True)`.
    """
    mod = model.module
    ce = getattr(mod, "cls_emb", None)
    has_atoms = bool(getattr(ce, "random_init_proxies", False))
    if which == "auto":
        which = "atoms" if has_atoms else "adapter"

    if which == "adapter":
        if adata is None or cls_proj_key not in getattr(adata, "obsm", {}):
            raise ValueError(
                f"prototypes 'adapter' need the aligner's class projection, which only a forward "
                f"pass produces. Run model.predict(adata, inplace=True) and it lands in "
                f"obsm[{cls_proj_key!r}]."
            )
        P = np.asarray(adata.obsm[cls_proj_key], np.float64)
    elif which == "atoms":
        if not has_atoms:
            raise ValueError("no learnable prototype table on this checkpoint; use 'adapter'")
        P = ce.encode().detach().float().cpu().numpy().astype(np.float64)
    elif which == "ema":
        log.warning("prototypes 'ema' (class_centers) is a running mean that no loss shapes; "
                    "the trained object is 'atoms' or 'adapter'")
        P = mod.class_centers.detach().float().cpu().numpy().astype(np.float64)
    else:
        raise ValueError(f"which must be auto/atoms/adapter/ema, got {which!r}")

    n_lab = int(getattr(mod, "n_labels", P.shape[0]))
    if 0 < n_lab < P.shape[0]:
        log.info("prototypes %r: truncating %d -> %d rows (trained block only)",
                 which, P.shape[0], n_lab)
        P = P[:n_lab]
    labels = np.asarray(model.idx_to_label).astype(str)[:P.shape[0]]
    return which, labels, np.asarray(P, np.float64)


def prototype_scores(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """`(n_groups, n_prototypes)` cosine between each group response and each prototype."""
    return _unit(np.asarray(R, np.float64)) @ _unit(np.asarray(P, np.float64)).T


def hub_penalty(model, n: int):
    """`(2, n)` stored hubness offset (row 0: +prototype / 'loss', row 1: -prototype / 'gain'), or None.

    Written by `ExPert.add_hub_correction`; ranking uses `cos - pen` for the tail being read. The
    cosines themselves stay raw everywhere they are stored."""
    mod = getattr(model, "module", model)
    if not (hasattr(mod, "has_hub_correction") and mod.has_hub_correction()):
        return None
    pen = mod.atom_hub_pen.detach().float().cpu().numpy()
    return pen[:, :n] if pen.shape[1] >= n else None


def _tail_offset(hub_pen, tail: str):
    return None if hub_pen is None else np.asarray(hub_pen[0 if tail == "loss" else 1], np.float64)


def explained(q: np.ndarray, P: np.ndarray, idx, solver: str = "nnls") -> tuple[np.ndarray, float]:
    """Non-negative coefficients on a FIXED prototype set, and the variance they explain.

    Refitting makes the number a property of the SELECTION rather than of whatever weights
    produced it, which is the only way to compare two selection rules at matched size.

    solver='nnls'  THE DEFAULT: the true constrained solve (`scipy.optimize.nnls`).

    solver='clip'  solve unconstrained, then clip negatives to zero. NOT the constrained optimum.
                   Safe ONLY for a pursuit refitting a handful of near-orthogonal prototypes -- and
                   there it is a no-op anyway, because a pursuit selects against the current
                   residual so its unconstrained fit is already non-negative (measured: 0 clipped
                   coefficients across 3,400 depth-200 OMP refits). Pass it to reproduce a legacy
                   OMP number exactly; never for a cosine-top-K set or a curated module.
    Returns the fraction of VARIANCE explained (1 - ||r||^2/||q||^2). A fraction of the NORM
    (1 - ||r||/||q||) is a different, also-legitimate quantity and is what some existing
    module-power readouts report -- do not mix the two in one table.

    solver='nnls'  the true constrained solve (`scipy.optimize.nnls`). REQUIRED whenever `idx` is
                   a large correlated block, e.g. a whole curated module: clipping an
                   unconstrained fit there can leave a residual LARGER than the response (a
                   measured -0.33 on a 206-prototype module), which is not a weak explanation but
                   an invalid one. True NNLS is bounded below by 0 because w=0 is always feasible,
                   so the fraction stays interpretable and modules of different sizes stay
                   comparable against a size-matched null.
    """
    idx = np.asarray(idx, int)
    q = np.asarray(q, np.float64)
    if idx.size == 0:
        return np.zeros(0), 0.0
    M = np.asarray(P, np.float64)[idx].T
    if solver == "nnls":
        from scipy.optimize import nnls
        w, _ = nnls(M, q)
    elif solver == "clip":
        w, *_ = np.linalg.lstsq(M, q, rcond=None)
        w = np.clip(w, 0, None)
    else:
        raise ValueError(f"solver must be 'clip' or 'nnls', got {solver!r}")
    r = q - M @ w
    return w, float(1 - np.linalg.norm(r) ** 2 / max(np.linalg.norm(q) ** 2, 1e-12))


def nnomp(q: np.ndarray, P: np.ndarray, k: int = 8, stop: str | None = "corr_gumbel",
          stop_t: float = 1.1):
    """Non-negative orthogonal matching pursuit: `(selected_indices, coefficients)`.

    Correlate the residual, take the best prototype, refit all chosen coefficients by NNLS,
    repeat on the new residual. `stop='corr_gumbel'` keeps a prototype only while its correlation
    is an outlier against the whole table (threshold scaled by sqrt(2 ln n)), which is what stops
    the loop running to `k` on noise.

    Answers which prototypes JOINTLY explain a response. It is NOT the scoring profile -- see the
    module docstring, point 3.
    """
    P = np.asarray(P, np.float64)
    Pn = _unit(P)
    q = np.asarray(q, np.float64)
    r, chosen, w = q.copy(), [], np.zeros(0)
    use_gumbel = stop == "corr_gumbel"
    for _ in range(int(k)):
        c = Pn @ r
        if chosen:
            c[chosen] = -np.inf
        j = int(np.argmax(c))
        if not np.isfinite(c[j]) or c[j] <= 0:
            break
        if use_gumbel and chosen:
            # Must match ExPert.decode_combinations exactly: a ROBUST statistic (median/MAD, so the
            # right tail of genuinely-correlated prototypes does not inflate the spread the way
            # mean/std does) tested against a Gumbel LOCATION + SCALE threshold. Using mean/std or
            # multiplying the location gives a different algorithm at the same stop_t, and results
            # from the two get compared as if they were the same one.
            fin = c[np.isfinite(c)]
            med = float(np.median(fin))
            mad = 1.4826 * float(np.median(np.abs(fin - med)))
            z = (float(c[j]) - med) / max(mad, 1e-12)
            cal, n_eff = 35.0, max(len(fin), 2)
            L = float(np.sqrt(2 * np.log(n_eff)))
            Lc = float(np.sqrt(2 * np.log(cal)))
            bc = Lc - (np.log(np.log(cal)) + np.log(4 * np.pi)) / (2 * Lc)
            g = (float(stop_t) * Lc - bc) * Lc
            b = L - (np.log(np.log(n_eff)) + np.log(4 * np.pi)) / (2 * L)
            if z <= b + g / L:
                break
        chosen.append(j)
        w, _ = explained(q, P, chosen)
        r = q - P[chosen].T @ w
    return np.asarray(chosen, int), np.asarray(w, np.float64)


DEFLATE_MODES = ("atom", "soft", "subspace", "program")
MERGE_METHODS = ("pm2", "rrf", "max")


def symnmf_programs(P: np.ndarray, k: int = 150, min_cos: float = 0.0, n_iter: int = 300,
                    seed: int = 0, knn: int = 30, center: bool = True) -> np.ndarray:
    """`(n_prototypes, k)` soft program memberships: symmetric NMF of a sparse cosine kernel.

    Prototypes are signed latent vectors, so NMF cannot factor them directly; the kernel
    `max(0, cos(P, P))` is non-negative and symmetric, and `K ~ H H^T` (Ding's multiplicative
    update, beta 0.5) gives each prototype a non-negative loading on each program. Depends only on
    the table, so compute once per checkpoint and cache.

    `knn` keeps each prototype's `knn` nearest neighbours (symmetrised); `center` removes the mean
    prototype first. Keep `min_cos` at 0: most prototypes' 30th neighbour sits below cos 0.2, so a
    0.2 floor leaves their rows identity-only, they get ZERO loading everywhere, and argmax dumps
    them into one 'program' (5,952 of 8,964 v0.1.0 atoms; knn 30 / min_cos 0: median 29, max 848).
    """
    P = np.asarray(P, np.float64)
    if center:
        P = P - P.mean(0, keepdims=True)
    Pn = _unit(P).astype(np.float32)
    K = Pn @ Pn.T
    np.fill_diagonal(K, 0.0)
    if knn:
        kth = np.partition(K, -int(knn), axis=1)[:, -int(knn)][:, None]
        K = np.where(K >= kth, K, 0.0)
        K = np.maximum(K, K.T)
    K[K < min_cos] = 0.0
    np.fill_diagonal(K, 1.0)
    rng = np.random.default_rng(seed)
    H = (rng.random((K.shape[0], int(k))) * np.sqrt(K.mean() / k)).astype(np.float32)
    for _ in range(int(n_iter)):
        KH = K @ H
        HHH = H @ (H.T @ H)
        H *= 0.5 + 0.5 * KH / np.maximum(HHH, 1e-12)
    return H.astype(np.float64)


def _outlier_z(c: np.ndarray) -> tuple[float, float]:
    """(robust z of max |c|, its n-scaled Gumbel location) over the SIGNED table.

    Same statistic as `nnomp`'s corr_gumbel rule (median/MAD, location sqrt(2 ln n)), applied to
    max |c| with both tails pooled, since a residual may point along a negated prototype."""
    fin = np.concatenate([c, -c])
    med = float(np.median(fin))
    mad = 1.4826 * float(np.median(np.abs(fin - med)))
    z = (float(np.abs(c).max()) - med) / max(mad, 1e-12)
    n_eff = max(len(fin), 2)
    L = float(np.sqrt(2 * np.log(n_eff)))
    b = L - (np.log(np.log(n_eff)) + np.log(4 * np.pi)) / (2 * L)
    return z, b


def _gumbel_gap(stop_t: float, n_eff: int) -> float:
    """Excess over the Gumbel location required to continue -- `nnomp`'s `g / L` term."""
    cal = 35.0
    Lc = float(np.sqrt(2 * np.log(cal)))
    bc = Lc - (np.log(np.log(cal)) + np.log(4 * np.pi)) / (2 * Lc)
    return (float(stop_t) * Lc - bc) * Lc / float(np.sqrt(2 * np.log(max(n_eff, 2))))


def _leading_set(c, m, select, z_keep, band):
    """Indices of the prototypes a round deflates.

    `select='fixed'` takes the top `m` -- a hard cutoff that is nearly the whole table for a panel of a
    few dozen prototypes but an arbitrary 25 out of thousands. The adaptive rules instead keep only the
    prototypes whose score stands out from a null, so a query with one clean hit deflates one direction
    and a query with a tight family deflates the family:
      'z'      robust z over the SIGNED table (median / MAD, as `_outlier_z`) -- the global null
      'zlocal' the same statistic with the null taken from a BAND of the sorted scores (default ranks
               10-110), i.e. significance against the near-misses rather than against the whole table
    Both are capped at `m` and always return at least one prototype.
    """
    a = np.abs(c)
    m = min(int(m), a.shape[0])
    if select == "fixed":
        return np.arange(a.shape[0]) if m >= a.shape[0] else np.argpartition(-a, m)[:m]
    if select == "z":
        fin = np.concatenate([c, -c])
        med = float(np.median(fin))
        mad = 1.4826 * float(np.median(np.abs(fin - med)))
    elif select == "zlocal":
        lo, hi = band
        srt = np.sort(a)[::-1]
        ref = srt[lo:hi] if srt.shape[0] > lo else srt
        if ref.size < 4:
            ref = srt
        med = float(np.median(ref))
        mad = 1.4826 * float(np.median(np.abs(ref - med)))
    else:
        raise ValueError(f"unknown select={select!r}")
    keep = np.flatnonzero((a - med) / max(mad, 1e-12) > z_keep)
    if keep.size == 0:
        return np.array([int(np.argmax(a))])
    if keep.size > m:
        keep = keep[np.argsort(-a[keep])[:m]]
    return keep


def _deflate_basis(c, Pn, mode, m, rank, programs, select="fixed", z_keep=3.0, band=(10, 110)):
    """Orthonormal `(d, r)` basis of what one round explains, from the round's signed scores."""
    m = min(int(m), int(c.shape[0]))        # a table smaller than the leading set deflates all of it
    if mode == "atom":
        j = int(np.argmax(np.abs(c)))
        return Pn[j][:, None]
    if mode == "program":
        j = int(np.argmax(np.abs(c)))
        p = int(np.argmax(programs[j]))
        idx = np.argsort(-programs[:, p])[:m]
        M = programs[idx, p][:, None] * Pn[idx]
    else:
        idx = _leading_set(c, m, select, z_keep, band)
        # signed weights: a round led by the GAIN tail deflates the negated prototypes
        M = c[idx][:, None] * Pn[idx]
    if mode == "soft":
        v = M.sum(0)
        return (v / max(np.linalg.norm(v), 1e-12))[:, None]
    U, s, _ = np.linalg.svd(M.T, full_matrices=False)
    return U[:, :int(min(rank, U.shape[1]))]


def deflate_rounds(Q: np.ndarray, P: np.ndarray, rounds: int = 3, mode: str = "subspace",
                   m: int = 25, rank: int = 3, programs: np.ndarray | None = None, k: int = 200,
                   chunk: int = 2048, effect=None, select: str = "fixed", z_keep: float = 3.0,
                   band: tuple = (10, 110)) -> dict:
    """Full cosine readout, repeated on what the previous rounds left unexplained.

    Round 1 is the plain cosine readout of each query against every prototype. Each round then
    projects out the subspace its leading prototypes span (`mode`) and scores the residual again
    against the WHOLE table, so round 2 is a complete readout of the part of the response round 1
    could not account for. Correlated partners of a hit stay together inside a round (they are
    what a set statistic counts -- the reason OMP profiles score at the null), while the next
    round is free to name a different mechanism.

    mode   'atom'      the single top prototype (classic matching pursuit)
           'soft'      one direction: the score-weighted sum of the top-m prototypes
           'subspace'  rank-`rank` span of the score-weighted top-m prototypes (default)
           'program'   rank-`rank` span of the top prototype's program (`symnmf_programs`)

    All `rounds` are computed; `stop_rounds` and `merge_rounds` decide afterwards how many are
    used and how they combine, so a stopping or merge rule can be swept without recomputing.
    Returns, per query: `top_loss` / `top_gain` `(n, rounds, k)` prototype indices,
    `val_loss` / `val_gain` the matching cosines scaled by the residual's share of the query
    norm (`<p_hat, residual> / ||q||`), `energy` `(n, rounds)` the fraction of ||q||^2 each round
    removed, and `z` / `zloc` `(n, rounds)` the outlier statistic of each round's scores.
    `effect`, a callable `(m, d) -> (m,)`, is applied to the query and to the residual after every
    round: `effect` `(n, rounds + 1)`, column 0 = the query's own effect size.
    """
    if mode not in DEFLATE_MODES:
        raise ValueError(f"mode must be one of {DEFLATE_MODES}, got {mode!r}")
    if mode == "program" and programs is None:
        raise ValueError("mode='program' needs programs= (symnmf_programs(P))")
    Q = np.asarray(Q, np.float64)
    Pn = _unit(np.asarray(P, np.float64))
    n, d = Q.shape
    k = min(int(k), Pn.shape[0])
    out = {key: np.zeros((n, rounds, k), np.int32) for key in ("top_loss", "top_gain")}
    out.update({key: np.zeros((n, rounds, k), np.float32) for key in ("val_loss", "val_gain")})
    out.update({key: np.zeros((n, rounds), np.float64) for key in ("energy", "z", "zloc")})
    out["n_table"] = int(Pn.shape[0])
    if effect is not None:
        out["effect"] = np.zeros((n, rounds + 1), np.float64)
    qn2 = np.maximum((Q ** 2).sum(1), 1e-24)
    for s in range(0, n, chunk):
        R = Q[s:s + chunk].copy()
        bases = [np.zeros((d, 0)) for _ in range(len(R))]
        if effect is not None:
            out["effect"][s:s + chunk, 0] = effect(R)
        for r in range(rounds):
            rn = np.sqrt((R ** 2).sum(1))
            Cs = (R @ Pn.T) / np.clip(rn[:, None], 1e-12, None)          # cosine to the residual
            scale = (rn / np.sqrt(qn2[s:s + chunk]))[:, None]
            for tail, sign in (("loss", 1.0), ("gain", -1.0)):
                part = np.argpartition(-sign * Cs, k - 1, axis=1)[:, :k]
                v = np.take_along_axis(sign * Cs, part, 1)
                o = np.argsort(-v, axis=1)
                out[f"top_{tail}"][s:s + chunk, r] = np.take_along_axis(part, o, 1)
                out[f"val_{tail}"][s:s + chunk, r] = np.take_along_axis(v, o, 1) * scale
            for i in range(len(R)):
                qi = s + i
                out["z"][qi, r], out["zloc"][qi, r] = _outlier_z(Cs[i])
                U = _deflate_basis(Cs[i], Pn, mode, m, rank, programs, select, z_keep, band)
                B = bases[i]
                if B.shape[1]:
                    U = U - B @ (B.T @ U)
                U, sv, _ = np.linalg.svd(U, full_matrices=False)
                U = U[:, sv > 1e-6]
                proj = U @ (U.T @ R[i])
                out["energy"][qi, r] = float(proj @ proj) / qn2[qi]
                R[i] = R[i] - proj
                bases[i] = np.concatenate([B, U], 1)
            if effect is not None:
                out["effect"][s:s + chunk, r + 1] = effect(R)
    return out


def _blend_orders(a_ord: np.ndarray, b_ord: np.ndarray, alpha: np.ndarray, k: int) -> np.ndarray:
    """Per row, re-rank by (1 - alpha) * rank_in_a + alpha * rank_in_b over the union of both lists.

    Used to pull a query whose stop only just fired back toward the plain cosine order `a_ord`."""
    out = np.empty_like(b_ord[:, :k])
    big = k + 1
    for i in range(len(a_ord)):
        al = float(alpha[i])
        if al >= 1.0:
            out[i] = b_ord[i, :k]
            continue
        if al <= 0.0:
            out[i] = a_ord[i, :k]
            continue
        ra = {int(v): j for j, v in enumerate(a_ord[i, :k])}
        rb = {int(v): j for j, v in enumerate(b_ord[i, :k])}
        cand = list(dict.fromkeys(list(ra) + list(rb)))
        sc = [(1 - al) * ra.get(c, big) + al * rb.get(c, big) for c in cand]
        o = np.argsort(np.asarray(sc), kind="stable")[:k]
        pick = [cand[j] for j in o]
        out[i, :len(pick)] = pick
        if len(pick) < k:
            out[i, len(pick):] = pick[-1] if pick else 0
    return out


def stop_rounds(rr: dict, stop_t: float | None = 1.1, min_energy: float = 0.0) -> np.ndarray:
    """`(n,)` rounds to use: continue to round r+1 only while its residual's best prototype is an
    outlier (`nnomp`'s corr_gumbel rule at `stop_t`) and round r removed >= `min_energy` of the
    query. `stop_t=None` uses every computed round; `np.inf` always stops at round 1 (= cosine)."""
    n, R = rr["z"].shape
    if stop_t is None:
        return np.full(n, R)
    gap = _gumbel_gap(stop_t, 2 * int(rr["n_table"])) if np.isfinite(stop_t) else np.inf
    ok = (rr["z"][:, 1:] > rr["zloc"][:, 1:] + gap) & (rr["energy"][:, :-1] >= min_energy)
    return 1 + np.cumprod(ok, axis=1).sum(1)


def merge_rounds(rr: dict, n_used: np.ndarray, tail: str = "loss", method: str = "pm2",
                 k: int = 200, rrf_c: float = 60.0, head: int = 0) -> np.ndarray:
    """`(n, k)` merged prototype ranking over each query's first `n_used` rounds.

    pm2  proportional slot allocation (Dang & Croft 2012): each round is an "aspect" weighted by
         the share of the query it explained; the next slot goes to the round with the highest
         Sainte-Lague quotient w / (2 s + 1), which then contributes its best unplaced prototype.
    rrf  energy-weighted reciprocal rank fusion, sum_r w_r / (c + rank_r).
    max  each prototype's best residual-scaled projection <p_hat, residual_r> / ||q|| over rounds.
    With one round every method returns the round-1 list, i.e. exactly the cosine readout.

    `head` (pm2 only): the first `head` slots are round 1's own top `head` -- the cosine readout's top is
    never displaced -- and PM-2 allocates the rest, with round 1 already holding `head` seats.
    """
    if method not in MERGE_METHODS:
        raise ValueError(f"method must be one of {MERGE_METHODS}, got {method!r}")
    top, val, E = rr[f"top_{tail}"], rr.get(f"val_{tail}"), rr["energy"]     # val only needed by 'max'
    n, R, K = top.shape
    k = min(int(k), K)
    out = np.full((n, k), -1, np.int64)
    for i in range(n):
        u = int(n_used[i])
        if u <= 1:
            out[i] = top[i, 0, :k]
            continue
        # weight of round r = share of ||q||^2 its own deflation removed
        w = np.maximum(E[i, :u], 1e-6)
        if method == "pm2":
            seats, ptr, placed, res = np.zeros(u), np.zeros(u, int), set(), []
            h = min(int(head), k)
            if h:
                res = [int(j) for j in top[i, 0, :h]]
                placed.update(res)
                seats[0], ptr[0] = h, h
            while len(res) < k:
                r = int(np.argmax(w / (2 * seats + 1)))
                while ptr[r] < K and top[i, r, ptr[r]] in placed:
                    ptr[r] += 1
                if ptr[r] >= K:
                    w[r] = -1.0
                    if (w < 0).all():
                        break
                    continue
                j = int(top[i, r, ptr[r]])
                placed.add(j)
                res.append(j)
                seats[r] += 1
            out[i, :len(res)] = res
        else:
            score: dict[int, float] = {}
            for r in range(u):
                for rank, (j, v) in enumerate(zip(top[i, r], val[i, r])):
                    j = int(j)
                    if method == "rrf":
                        score[j] = score.get(j, 0.0) + w[r] / (rrf_c + rank)
                    else:
                        score[j] = max(score.get(j, -np.inf), float(v))
            best = sorted(score, key=score.get, reverse=True)[:k]
            out[i, :len(best)] = best
    return out


def span_basis(P: np.ndarray, span: int = 256) -> np.ndarray:
    """`(span, d)` top right-singular vectors of the prototype table: the directions the dictionary can explain."""
    _, _, Vt = np.linalg.svd(np.asarray(P, np.float64), full_matrices=False)
    return Vt[:int(span)]


def fit_noise_curve(C: np.ndarray, datasets, basis: np.ndarray | None = None, q: float = 0.95,
                    pool_cells: int = 10, ms=(1, 2, 3, 5, 10, 20), seed: int = 0) -> tuple[float, float]:
    """`(intercept, slope)` of log noise_q(n) vs log n_cells, from control deltas of `pool_cells` cells each.

    `C` are control-pool deltas (pool mean minus its dataset's control mean, as made by
    paper/scripts/fig5/fig5_hubness.py --stage ctrlref); pools of one dataset are averaged in disjoint groups of
    m to give 10m-cell control deltas. The statistic is ||c|| (basis=None) or ||basis c||. Measured slope
    -0.52 (EFFGATE3, v0.1.0): noise falls as 1/sqrt(n), and the fit extrapolates to single cells
    (||z|| 49.5 extrapolated vs 48.0 on single control cells)."""
    C = np.asarray(C, np.float64)
    ds = np.asarray(datasets).astype(str)
    rng = np.random.default_rng(seed)
    ns, qs = [], []
    for m in ms:
        vals = []
        for d in np.unique(ds):
            Pd = C[ds == d]
            g = len(Pd) // m
            if g < 1:
                continue
            X = Pd[rng.permutation(len(Pd))[:g * m]].reshape(g, m, -1).mean(1)
            vals.append(np.linalg.norm(X if basis is None else X @ basis.T, axis=1))
        if vals:
            ns.append(pool_cells * m), qs.append(np.quantile(np.concatenate(vals), q))
    slope, icpt = np.polyfit(np.log(ns), np.log(qs), 1)
    return float(icpt), float(slope)


def noise_level(n_cells, icpt: float, slope: float) -> np.ndarray:
    """noise_q at `n_cells` from a fitted curve."""
    return np.exp(icpt + slope * np.log(np.maximum(np.asarray(n_cells, np.float64), 1.0)))


def calibrated_readout(Q: np.ndarray, P: np.ndarray, n_cells, noise: np.ndarray | None = None,
                       rounds: int = 3, head: int = 10, k: int = 200, mode: str = "subspace",
                       span: int = 256, basis: np.ndarray | None = None, m: int = 25,
                       rank: int = 3, select: str = "fixed", z_keep: float = 3.0,
                       band: tuple = (10, 110), stop: str = "noise", ratio: float = 0.9,
                       min_cells: int = 0, damp: float = 0.0) -> dict:
    """The shipped readout: iterated cosine with deflation, a depth-matched noise stop, PM-2 with a protected head.

    Round 1 is the plain signed cosine readout; each further round deflates what the previous one explained
    and re-reads the residual against the whole table. A round is run only while the dictionary-EXPLAINABLE
    residual (its norm in the top-`span` prototype directions) is above the noise level of a control delta
    with the same number of cells. Rounds are merged by PM-2 with the first `head` slots held by round 1, so
    the top-`head` list IS the cosine readout. Selected on deep id_train, evaluated once (2026-10-15):
    deep OOD singles top-10 == cosine at every depth; GSE146194 doubles both@20 .200 -> .314; ALD healthy-null
    false-positive sets 10.1 -> 6.8 and replicated AH-gain sets 0 -> 5. OMP machinery stays in
    decode_combinations / nnomp for comparisons.

    `damp` > 0 makes the stop SOFT. A query that only just cleared the guard is the one most likely to have
    been deflated when it should not have been, so instead of an all-or-nothing decision its deflated list is
    blended back toward round 1 (plain cosine) by its margin: alpha = clip(margin / damp, 0, 1), with the
    final order taken on (1 - alpha) * rank_cosine + alpha * rank_deflate. alpha = 1 reproduces the hard
    rule exactly, alpha = 0 is cosine, so `damp` only ever pulls marginal queries back toward cosine.

    `min_cells` forces round 1 only (= cosine) for queries shallower than that, which a scale-free `stop`
    does not otherwise protect.

    `stop` picks the rule that decides how many rounds a query uses: 'noise' (the absolute depth-matched
    control-noise threshold), 'ratio' (keep a round only if it shrank the explainable residual to below
    `ratio` of the previous round -- scale free, no noise estimate), or 'both'.

    `m` / `rank` size the deflated subspace: each round projects out the span of its `m` leading prototypes,
    reduced to `rank` directions. They matter most when the table is large -- with a dictionary of only a few
    dozen prototypes `m` covers nearly all of it, while against thousands the leading set is mostly unrelated.

    `noise` = [icpt_norm, slope_norm, icpt_span, slope_span] (ExPert.add_readout_calibration). Without it the
    stop cannot be applied: every round is used (the head still protects the top) and `effect` is NaN.

    Returns `loss` / `gain` `(n, k)` merged prototype indices, `n_rounds`, `energy` `(n, rounds)`,
    `top_loss` / `top_gain` `(n, rounds, k)` per-round lists, `effect` = ||q|| / noise95(n) (effect size above
    a control delta of the same depth), `explainable` = span norm / its noise95(n), and `max_cos`.
    """
    Q = np.atleast_2d(np.asarray(Q, np.float64))
    n_cells = np.broadcast_to(np.asarray(n_cells, np.float64), (len(Q),))
    Vs = span_basis(P, span) if basis is None else basis
    rr = deflate_rounds(Q, P, rounds=rounds, mode=mode, k=k, m=m, rank=rank,
                        select=select, z_keep=z_keep, band=band,
                        effect=lambda R: np.linalg.norm(R @ Vs.T, axis=1))
    E = rr["effect"]
    if noise is not None:
        thr = noise_level(n_cells, noise[2], noise[3])
        # 'noise' is an ABSOLUTE threshold on the residual's explainable norm, so it needs the control
        # noise scale to transfer to the query. Where it does not (a screen with a different depth or
        # effect size) it runs every round on every query, which costs single-perturbation queries their
        # rank for nothing. 'ratio' is the scale-free alternative from the residual-ratio-thresholding
        # line of work (Kallummil & Kalyani, ICML 2018): a round is kept only if it shrank the
        # explainable residual by enough RELATIVE to the previous round, needing no noise estimate.
        keep = E[:, 1:rounds] > thr[:, None]
        if stop in ("ratio", "both"):
            rr_ratio = E[:, 1:rounds] / np.maximum(E[:, :rounds - 1], 1e-12)
            keep = (rr_ratio <= ratio) if stop == "ratio" else (keep & (rr_ratio <= ratio))
        used = 1 + np.cumprod(keep, axis=1).sum(1)
        rr["effect_size"] = np.linalg.norm(Q, axis=1) / noise_level(n_cells, noise[0], noise[1])
        rr["explainable"] = E[:, 0] / thr
    elif stop in ("ratio", "both"):
        # scale free: needs no control-noise calibration, which is the point of the ratio rule. 'both'
        # degrades to 'ratio' here rather than silently falling through to "use every round".
        rr_ratio = E[:, 1:rounds] / np.maximum(E[:, :rounds - 1], 1e-12)
        used = 1 + np.cumprod(rr_ratio <= ratio, axis=1).sum(1)
        rr["effect_size"] = rr["explainable"] = np.full(len(Q), np.nan)
    else:
        used = np.full(len(Q), rounds)
        rr["effect_size"] = rr["explainable"] = np.full(len(Q), np.nan)
    # Depth guard: below `min_cells` a query is read with round 1 only, i.e. the plain cosine readout.
    # The 'noise' rule gets this for free -- its threshold scales with n_cells, so a one-cell query clears
    # it and stops -- but a scale-free rule like 'ratio' has no such protection and would happily deflate
    # a single cell. Doubles are not decodable at one cell anyway, so there is nothing to lose here.
    if min_cells:
        used = np.where(np.asarray(n_cells) < min_cells, 1, used)
    # soft stop: blend each query's deflated list back toward cosine by the margin that let it through
    alpha = None
    if damp > 0:
        if noise is not None and stop in ("noise", "both"):
            marg = E[:, 1] / np.maximum(thr, 1e-12) - 1.0
        else:
            marg = (ratio - E[:, 1] / np.maximum(E[:, 0], 1e-12)) / max(ratio, 1e-12)
        alpha = np.clip(marg / damp, 0.0, 1.0)
        alpha[used <= 1] = 0.0
    rr["alpha"] = np.ones(len(Q)) if alpha is None else alpha
    rr["n_rounds"] = used
    # confidence = the largest POSITIVE cosine (the validated definition: per-cell AUROC .76 for truth-in-top-10);
    # round 1's residual is the query itself, so its scaled value is the plain cosine
    rr["max_cos"] = rr["val_loss"][:, 0, 0].astype(np.float64)
    for tail in TAILS:
        merged = merge_rounds(rr, used, tail, "pm2", k, head=head)
        if alpha is not None and (alpha < 1).any():
            cos_ord = rr[f"top_{tail}"][:, 0, :k]
            merged = _blend_orders(cos_ord, merged, alpha, k)
        rr[tail] = merged
    return rr


def transplant_scores(S1: np.ndarray, loss: np.ndarray, gain: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """`(n, N)` full score matrix in the cosine scale whose order follows the merged rankings.

    The prototype at merged position p (loss tail, from the top) gets the row's own p-th largest round-1
    score; the gain tail is mirrored from the bottom; every other prototype keeps its own score, clipped to
    lie strictly between the two merged blocks. Where the merge equals the round-1 order (the protected head,
    or a query the stop never let past round 1) the row is returned UNCHANGED."""
    S1 = np.asarray(S1)
    out = S1.copy()
    n, N = S1.shape
    K = loss.shape[1]
    desc = -np.sort(-S1, axis=1)
    rows = np.arange(n)[:, None]
    hi = desc[:, K - 1:K] - eps                       # everything outside the loss block sits below it
    lo = desc[:, N - K:N - K + 1] + eps if 2 * K <= N else None
    mid = np.minimum(out, hi)
    if lo is not None:
        mid = np.maximum(mid, lo)
    out = mid
    out[rows, gain[:, ::-1]] = desc[:, N - K:]        # gain block: the K smallest values, merged order from the bottom
    out[rows, loss] = desc[:, :K]                     # loss block: the K largest values, merged order from the top
    return out.astype(S1.dtype)


def deflate_rows(Z: np.ndarray, P: np.ndarray, n_cells, noise: np.ndarray, rounds: int = 3, head: int = 10,
                 k: int = 200, span: int = 256) -> dict:
    """Depth-matched deflated readout for EVERY row of a query -- cells, pseudobulks or bulk alike.

    Each row is judged at its own depth (`n_cells`, 1 for a single cell): it is decoded past round 1 only while
    its dictionary-explainable residual is above the noise95 of a control delta of that many cells
    (calibrated_readout). Rows whose explainable norm is already below that level cannot pass (the residual
    only shrinks), so they are skipped without any deflation: on single cells that is ~99.9% of rows (only 0.1%
    of OOD K562 cells exceed single-cell noise), i.e. the readout IS the cosine readout there at ~zero cost.

    Returns `rows` (indices decoded past round 1), their merged `loss` / `gain` prototype lists `(len(rows), k)`,
    `n_rounds` `(n,)`, and `effect_size` `(n,)` = ||z|| / noise95_norm(n_cells).
    """
    Z = np.atleast_2d(np.asarray(Z, np.float64))
    n_cells = np.broadcast_to(np.asarray(n_cells, np.float64), (len(Z),))
    Vs = span_basis(P, span)
    e0 = np.linalg.norm(Z @ Vs.T, axis=1)
    cand = np.flatnonzero(e0 > noise_level(n_cells, noise[2], noise[3]))
    out = dict(n_rounds=np.ones(len(Z), int), rows=np.zeros(0, int),
               loss=np.zeros((0, k), np.int64), gain=np.zeros((0, k), np.int64),
               effect_size=np.linalg.norm(Z, axis=1) / noise_level(n_cells, noise[0], noise[1]))
    if len(cand):
        rr = calibrated_readout(Z[cand], P, n_cells[cand], noise=noise, rounds=rounds, head=head, k=k, basis=Vs)
        used = rr["n_rounds"]
        out["n_rounds"][cand] = used
        keep = used >= 2
        out["rows"], out["loss"], out["gain"] = cand[keep], rr["loss"][keep], rr["gain"][keep]
    return out


def weight_profile(scores_row: np.ndarray, P: np.ndarray | None = None,
                   q: np.ndarray | None = None, method: str = "cosine_topk",
                   k: int = 200, tail: str = "loss", hub_pen=None) -> np.ndarray:
    """A dense weight vector over the prototypes, for one group and one tail.

    cosine_topk   top-k by cosine, weighted by positive cosine. THE DEFAULT: dense weights over a
                  correlated set are what module and text statistics need.
    cosine_nnls   the same set, coefficients refit by NNLS. Isolates weighting from selection;
                  useful as the control arm when comparing methods.
    omp           non-negative OMP to depth k. Best RECONSTRUCTION, worst readout -- its text
                  enrichment sits at the shuffled-description null. See point 3.

    `tail='gain'` reads the other end: the scores and the response are negated, so the weights
    describe what the group resembles the OPPOSITE of. Both tails are non-negative afterwards, so
    everything downstream (module null, term z) is identical in form.
    """
    if method not in PROFILE_METHODS:
        raise ValueError(f"method must be one of {PROFILE_METHODS}, got {method!r}")
    sgn = 1.0 if _check_tail(tail) == "loss" else -1.0
    cs = sgn * np.asarray(scores_row, np.float64)
    # hub_pen: SELECTION by the corrected score (cos - pen for this tail); weights stay the raw cosine
    off = _tail_offset(hub_pen, tail)
    sel = cs if off is None else cs - off
    v = np.zeros(cs.shape[0])
    if method == "cosine_topk":
        top = np.argsort(-sel)[:k]
        top = top[cs[top] > 0]
        v[top] = cs[top]
        return v
    if P is None or q is None:
        raise ValueError(f"method={method!r} needs P= (prototypes) and q= (the response vector)")
    qq = sgn * np.asarray(q, np.float64)
    if method == "cosine_nnls":
        top = np.argsort(-sel)[:k]
        top = top[cs[top] > 0]
        w, _ = explained(qq, P, top)
        v[top] = w
        return v
    idx, w = nnomp(qq, P, k=k, stop=None)
    if idx.size:
        v[idx] = w
    return v


def score_prototypes(
    model,
    adata,
    contrast: str = "delta",
    which: str = "auto",
    method: str = "cosine_topk",
    k: int = 200,
    tails=TAILS,
    stratify="auto",
    group_specific_profile: bool = False,
    uns_key: str = UNS_KEY,
    copy: bool = False,
    hub_correction: bool = True,
):
    """Score every group response against the perturbation prototypes.

    Both signed `tails` are profiled by default (module docstring, point 5); `stratify` picks the
    covariate whose deciles each score is z-scored within, and is a no-op unless
    `attach_prototype_covariates` was called. The stored cosines are always raw -- only the
    PROFILE is corrected -- so a stratified run and a raw one stay comparable.

    Writes into `adata.uns[uns_key]`
    -------------------------------
    prototypes     {'source', 'labels', 'n'}. The matrix itself is NOT stored -- it is hundreds of
                   MB and belongs to the checkpoint, not to the query.
    scores         {contrast: (n_groups, n_prototypes) cosine}
    profiles       {key: (n_groups, n_prototypes) weights}, key = contrast for the loss tail and
                   '<contrast>|gain' for the gain tail (`get_profiles(..., tail=)`)
    scored_groups  {contrast: group labels in row order}. `delta` drops controls, so row order is
                   contrast-specific and has to travel with the matrix.
    """
    adata = adata.copy() if copy else adata
    tails = (tails,) if isinstance(tails, str) else tuple(tails)
    for t in tails:
        _check_tail(t)
    src, labels, P = perturbation_prototypes(model, adata, which=which)
    keys, R = get_response(adata, contrast, uns_key)
    S = prototype_scores(R, P)

    cov_name, cov = _resolve_covariate(adata, stratify, uns_key)
    if cov is not None and len(cov) != S.shape[1]:
        raise ValueError(f"prototype covariate {cov_name!r} has {len(cov)} entries for "
                         f"{S.shape[1]} prototypes")
    if cov is not None and method == "omp":
        log.warning("stratify=%r ignored for method='omp': a pursuit selects against the response "
                    "itself, not against a score vector", cov_name)
        cov, cov_name = None, None
    V = S if cov is None else np.stack([stratified_z(S[i], cov) for i in range(len(keys))])

    ns = _ns(adata, uns_key, create=True)
    ns["prototypes"] = {"source": src, "labels": labels, "n": int(P.shape[0])}
    # stored hubness offset (add_hub_correction): applied to RANKING/SELECTION only, cosines stay raw
    hub_pen = hub_penalty(model, P.shape[0]) if (hub_correction and src == "atoms") else None
    if hub_pen is not None:
        ns["hub_pen"] = hub_pen.astype(np.float32)
        log.info("score_prototypes: hubness correction ON (stored per-prototype offset); "
                 "hub_correction=False for plain cosine")
    else:
        ns.pop("hub_pen", None)
    ns.setdefault("scores", {})[contrast] = S.astype(np.float32)
    ns.setdefault("scored_groups", {})[contrast] = np.array(keys)
    for tail in tails:
        prof = np.stack([
            weight_profile(V[i], P=P, q=R[i], method=method, k=k, tail=tail,
                           hub_pen=hub_pen if cov is None else None)
            for i in range(len(keys))
        ])
        if group_specific_profile:
            other = (prof.sum(0, keepdims=True) - prof) / max(len(keys) - 1, 1)
            prof = np.clip(prof - other, 0, None)
        ns.setdefault("profiles", {})[_pkey(contrast, tail)] = prof.astype(np.float32)
        ns.setdefault("profile_params", {})[_pkey(contrast, tail)] = {
            "method": method, "k": int(k), "group_specific": bool(group_specific_profile),
            "tail": tail, "stratify": cov_name,
        }
    log.info("score_prototypes: %d groups x %d prototypes (%s), contrast=%s, profile=%s(k=%d), "
             "tails=%s, stratify=%s", len(keys), P.shape[0], src, contrast, method, k,
             "+".join(tails), cov_name)
    return adata if copy else None


def get_scores(adata, contrast: str = "delta", uns_key: str = UNS_KEY) -> pd.DataFrame:
    """The stored cosine scores as a `groups x perturbations` DataFrame."""
    ns = _ns(adata, uns_key)
    if "scores" not in ns or contrast not in ns["scores"]:
        raise KeyError(f"no scores for contrast={contrast!r} -- run score_prototypes(...) first")
    return pd.DataFrame(np.asarray(ns["scores"][contrast], np.float64),
                        index=[str(k) for k in ns["scored_groups"][contrast]],
                        columns=[str(g) for g in ns["prototypes"]["labels"]])


def get_profiles(adata, contrast: str = "delta", tail: str = "loss",
                 uns_key: str = UNS_KEY) -> pd.DataFrame:
    """The stored weight profiles for one tail, as a `groups x perturbations` DataFrame.

    `tail` was inserted in this position when signed tails were added, so a legacy positional
    `get_profiles(q, "mean", key)` would silently mean "tail=key". Rather than raise on three
    existing harnesses, a third positional argument that is not a tail is read as `uns_key` with a
    warning -- passing both is an error, because then the intent is genuinely ambiguous.
    """
    if tail not in TAILS:
        if uns_key != UNS_KEY:
            raise ValueError(
                f"tail={tail!r} is not one of {TAILS}, and uns_key={uns_key!r} was also given; "
                f"call get_profiles(adata, contrast, tail=..., uns_key=...) explicitly")
        log.warning("get_profiles: third positional argument %r is not a tail %s -- reading it as "
                    "uns_key (legacy call). Pass uns_key=%r to silence this.", tail, TAILS, tail)
        uns_key, tail = tail, "loss"
    ns = _ns(adata, uns_key)
    key = _pkey(contrast, tail)
    if "profiles" not in ns or key not in ns["profiles"]:
        have = sorted(ns.get("profiles", {}))
        raise KeyError(f"no profiles for contrast={contrast!r} tail={tail!r} -- run "
                       f"score_prototypes(..., tails=...) first; stored: {have}")
    return pd.DataFrame(np.asarray(ns["profiles"][key], np.float64),
                        index=[str(k) for k in ns["scored_groups"][contrast]],
                        columns=[str(g) for g in ns["prototypes"]["labels"]])


def top_perturbations(adata, group: str | None = None, contrast: str = "delta", n: int = 20,
                      tail: str = "loss", stratify="auto", uns_key: str = UNS_KEY) -> pd.DataFrame:
    """Highest-scoring perturbations per group -- the plain "what does this resemble" answer.

    `tail='gain'` ranks the other end ("resembles the OPPOSITE of losing this gene"). Ranking is
    by `stratified_z` within the covariate's deciles whenever one is attached (module docstring,
    RANKING), and `raw_rank` is kept beside it so the effect of the correction is visible per gene
    rather than asserted.
    """
    S = get_scores(adata, contrast, uns_key)
    sgn = 1.0 if _check_tail(tail) == "loss" else -1.0
    cov_name, cov = _resolve_covariate(adata, stratify, uns_key)
    if cov is not None and len(cov) != S.shape[1]:
        raise ValueError(f"prototype covariate {cov_name!r} has {len(cov)} entries for "
                         f"{S.shape[1]} prototypes")
    labels = np.asarray(S.columns, str)
    pct = None
    if cov is not None:
        order = np.argsort(np.argsort(cov))
        pct = 100.0 * (order + 1) / len(cov)
    _ns_ = _ns(adata, uns_key, create=False) or {}
    off = _tail_offset(_ns_.get("hub_pen"), tail)
    rows = []
    for g in ([group] if group is not None else list(S.index)):
        s = sgn * S.loc[g].to_numpy(np.float64)
        # hubness-corrected score; skipped under stratification (the decile z already re-centres
        # each prototype), exactly as score_prototypes skips it for stratified profiles
        sc = s if (off is None or cov is not None or len(off) != len(s)) else s - off
        v = sc if cov is None else stratified_z(sc, cov)
        raw_rank = np.empty(len(s), int)
        raw_rank[np.argsort(-s)] = np.arange(1, len(s) + 1)
        for i, j in enumerate(np.argsort(-v)[:n]):
            r = dict(group=g, rank=i + 1, perturbation=str(labels[j]),
                     cosine=float(S.loc[g].to_numpy()[j]), tail=tail,
                     raw_rank=int(raw_rank[j]))
            if cov is not None:
                r.update(score=float(v[j]), covariate=cov_name,
                         covariate_pctile=round(float(pct[j]), 1))
            rows.append(r)
    return pd.DataFrame(rows)


# ======================================================================================
# modules
# ======================================================================================
def load_modules(path: str, vocab, min_size: int = 5) -> dict[str, list[str]]:
    """Parse a two-column `module<TAB>gene` file; a trailing `*` is a prefix wildcard.

    Only genes present in `vocab` are kept -- a member with no prototype contributes nothing but
    would still inflate the module's size, and size is what sets the null.
    """
    vocab = list(map(str, vocab))
    vset = set(vocab)
    out: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or "\t" not in line:
            continue
        mod, pat = (x.strip() for x in line.split("\t")[:2])
        if not mod or not pat:
            continue
        if pat.endswith("*"):
            hits = [g for g in vocab if g.startswith(pat[:-1])]
        else:
            hits = [pat] if pat in vset else []
            if not hits:
                missing.setdefault(mod, []).append(pat)
        out.setdefault(mod, []).extend(hits)
    # First-appearance order, NOT sorted. Member order changes the accumulation order of the
    # module's matrix products, which moves results in the last bits -- enough to flip a
    # Monte-Carlo null comparison that sits exactly on the threshold. Preserving file order keeps
    # a module's numbers reproducible against earlier runs of the same modules.tsv.
    kept = {m: list(dict.fromkeys(v)) for m, v in out.items() if len(set(v)) >= min_size}
    for m, v in out.items():
        if m not in kept:
            log.info("module %r dropped: only %d of its genes have a prototype (min_size=%d)",
                     m, len(set(v)), min_size)
    for m, miss in missing.items():
        if m in kept:
            log.info("module %r: %d gene(s) with no prototype (%s)", m, len(miss),
                     ", ".join(miss[:8]))
    return kept


def module_test(adata, modules, contrast: str = "delta", tail: str = "loss", n_null: int = 300,
                alpha: float = 0.05, seed: int = 0, uns_key: str = UNS_KEY,
                key_added: str | None = None, copy: bool = False):
    """Per (group, module): the module's best weight vs a SIZE-MATCHED random-set null.

    A module test takes the max over members, so a large module wins simply by having more draws.
    The null therefore resamples sets of the SAME SIZE from the same prototype table, which is
    the only way a 9-gene and a 130-gene module can be compared on one scale.

    `modules` is a `{name: [genes]}` mapping or a path to a `module<TAB>gene` file.

    `tail` is 'loss', 'gain', or 'both'. Each tail is tested against its OWN null -- the two are
    separate statements ("losing these" vs "these more active"), not two sides of one test -- and
    'both' adds a `tail` column and doubles the rows, so a hit rate over the frame counts twice as
    many tests. Results land in a per-tail key so they never overwrite each other.

    Writes `adata.uns[uns_key][key_added]` -- a DataFrame, one row per (group, module[, tail]).
    """
    adata = adata.copy() if copy else adata
    tails = TAILS if tail == "both" else (_check_tail(tail),)
    if key_added is None:
        key_added = {"loss": "module_test", "gain": "module_test_gain",
                     "both": "module_test_signed"}[tail]
    ns = _ns(adata, uns_key)
    labels = [str(g) for g in ns["prototypes"]["labels"]]
    if isinstance(modules, (str, bytes)):
        modules = load_modules(modules, labels)
    pos = {g: i for i, g in enumerate(labels)}
    n = len(labels)

    frames = []
    for t in tails:
        prof = get_profiles(adata, contrast, t, uns_key)
        rng = np.random.default_rng(seed)
        rows = []
        for key in prof.index:
            w = prof.loc[key].to_numpy(np.float64)
            for mod, members in modules.items():
                idx = np.array([pos[g] for g in members if g in pos], int)
                if idx.size == 0:
                    continue
                best = float(w[idx].max())
                null = np.array([w[rng.choice(n, idx.size, replace=False)].max()
                                 for _ in range(n_null)])
                p = float((null >= best).mean())
                rows.append(dict(group=key, module=mod, n_genes=int(idx.size), best=best,
                                 null_mean=float(null.mean()), excess=best - float(null.mean()),
                                 p=p, significant=bool(p < alpha),
                                 top_gene=labels[int(idx[np.argmax(w[idx])])]))
        f = pd.DataFrame(rows)
        if len(tails) > 1:
            f["tail"] = t
        frames.append(f)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    ns[key_added] = df
    if len(df):
        log.info("module_test[%s]: %d/%d (group, module) tests significant at p<%.2g (%.1f%%)",
                 "+".join(tails), int(df.significant.sum()), len(df), alpha,
                 100 * df.significant.mean())
    return adata if copy else None


# ======================================================================================
# panel-free text enrichment
# ======================================================================================
_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

# Words appearing in nearly every gene description, which therefore cannot distinguish one
# prototype from another ("protein", "cell", "binding", "disease", ...). Dropped BEFORE bigrams
# are formed, so "nuclear pore" survives while "binding protein" never forms.
_STOP = frozenset("""a an the and or of to in for with by on at as is are was were be been being
that this these those it its from into via than then thus which who whom whose can could may
might must shall should will would have has had do does did not no nor but if when where while
there here also such other others same different various several many few more most less least
both each either neither all any some one two three protein proteins gene genes encodes encoded
encoding human cell cells cellular expression expressed function functions functional role roles
involved associated related including include includes based using used activity level levels
type types form forms member family complex subunit isoform variant variants domain domains
region regions site sites terminal binding bind binds required necessary important known shown
suggests studies study analysis results data patients disease diseases""".split())


def terms_of(text, bigrams: bool = True) -> set[str]:
    """Free-text description -> set of terms (unigrams plus adjacent bigrams).

    Deliberately a SET, not a count: the test downstream is document frequency -- how many
    prototypes in a group mention a term -- not how often one verbose entry repeats it. Counting
    would rank whichever description happens to be wordiest.
    """
    w = [t.lower() for t in _TOKEN.findall(str(text))]
    w = [t for t in w if t not in _STOP and len(t) > 2]
    out = set(w)
    if bigrams:
        out |= {f"{a} {b}" for a, b in zip(w, w[1:])}
    return out


def term_incidence(descriptions: dict[str, str], labels, min_df: int = 5,
                   max_df_frac: float = 0.5, tokenizer=None):
    """`(terms, M)` prototype x term incidence from a gene-description database.

    Terms carried by almost every prototype say nothing and terms carried by almost none are
    noise, so the vocabulary is trimmed at both ends. This is the panel-free readout: it needs no
    curated gene set, so it cannot be tuned by choosing a favourable panel.
    """
    tokenizer = terms_of if tokenizer is None else tokenizer
    labels = list(map(str, labels))
    desc = {k.upper(): v for k, v in descriptions.items()}
    per = [tokenizer(desc[g.upper()]) if g.upper() in desc else set() for g in labels]
    df: dict[str, int] = {}
    for s in per:
        for t in s:
            df[t] = df.get(t, 0) + 1
    n = len(labels)
    terms = sorted(t for t, c in df.items() if min_df <= c <= max_df_frac * n)
    tpos = {t: i for i, t in enumerate(terms)}
    M = np.zeros((n, len(terms)), np.float32)
    for i, s in enumerate(per):
        for t in s:
            j = tpos.get(t)
            if j is not None:
                M[i, j] = 1.0
    log.info("term_incidence: %d/%d prototypes have a description, %d terms (df %d..%d)",
             sum(1 for s in per if s), n, len(terms), min_df, int(max_df_frac * n))
    return np.array(terms), M


def _term_z_core(W: np.ndarray, inc64: np.ndarray, dfj: np.ndarray) -> np.ndarray:
    """`(n_groups, n_terms)` z, with the incidence cast and its document frequencies hoisted.

    Batched because the incidence matrix is large (prototypes x terms, ~1 GB here): casting it or
    copying it per group -- let alone per shuffle -- dominates the runtime of the whole readout.
    """
    n = W.shape[1]
    S = W @ inc64
    mu = np.outer(W.mean(1), dfj)
    var = np.outer(W.var(1), dfj * (n - dfj) / max(n - 1, 1))
    return (S - mu) / np.sqrt(np.clip(var, 1e-12, None))


def text_enrichment(adata, descriptions, contrast: str = "delta", tail: str = "loss",
                    min_df: int = 5, max_df_frac: float = 0.5, n_shuffle: int = 20,
                    top_terms: int = 10, seed: int = 0, tokenizer=None, uns_key: str = UNS_KEY,
                    key_added: str | None = None, copy: bool = False):
    """Panel-free description enrichment per group, plus the three quality numbers.

    `descriptions` is a `{gene: free text}` mapping or a path to a JSON file of one.

    strength      median over groups of the top term's z. How loudly the model says anything.
    vs_null       strength divided by the strength obtained after SHUFFLING the
                  prototype->description assignment. THE metric that decides whether a readout
                  carries information: at ~1.0 the terms are no more concentrated on the weights
                  than random labelling would make them, whatever the raw z values look like.
    specificity   1 - median pairwise correlation of the per-group term-z profiles.

    `specificity` must be read conditional on `vs_null`. A sparse, near-random profile
    decorrelates across groups and so scores HIGH on specificity while carrying nothing; high
    specificity at vs_null ~ 1 is what noise looks like, not a sharper readout. This warns when
    that is the case rather than leaving it to the reader.

    Writes `adata.uns[uns_key][key_added]`              top terms per group, a DataFrame
           `adata.uns[uns_key][key_added + '_metrics']` strength / vs_null / specificity
    """
    adata = adata.copy() if copy else adata
    if isinstance(descriptions, (str, bytes)):
        import json
        descriptions = json.load(open(descriptions))
    if key_added is None:
        key_added = "text" if _check_tail(tail) == "loss" else f"text_{tail}"
    ns = _ns(adata, uns_key)
    labels = [str(g) for g in ns["prototypes"]["labels"]]
    prof = get_profiles(adata, contrast, tail, uns_key)
    terms, M = term_incidence(descriptions, labels, min_df=min_df, max_df_frac=max_df_frac,
                              tokenizer=tokenizer)
    rng = np.random.default_rng(seed)

    M64 = M.astype(np.float64)                     # cast ONCE; it is ~1 GB
    dfj = M64.sum(0)
    W = np.stack([prof.loc[k].to_numpy(np.float64) for k in prof.index])
    Z = _term_z_core(W, M64, dfj)
    rows = []
    for i, k in enumerate(prof.index):
        o = np.argsort(-Z[i])[:top_terms]
        rows += [dict(group=k, rank=r + 1, term=str(terms[j]), z=float(Z[i, j]),
                      df=int(M[:, j].sum())) for r, j in enumerate(o)]
    strength = float(np.median(Z.max(1)))
    if len(prof.index) > 1:
        C = np.corrcoef(Z)
        iu = np.triu_indices(len(prof.index), 1)
        specificity = 1.0 - float(np.median(C[iu]))
    else:
        specificity = float("nan")
    n = M.shape[0]
    # Shuffling the prototype->description map permutes the ROWS of M. Permuting the weights by
    # the inverse permutation gives the identical score (w @ M[p] == w[argsort(p)] @ M) without
    # materialising a fresh ~1 GB copy of M per group per shuffle. Permutations are drawn in the
    # same order as the row-shuffling formulation, one per (shuffle, group).
    nulls = []
    for _ in range(n_shuffle):
        per_group = []
        for gi in range(W.shape[0]):
            inv = np.argsort(rng.permutation(n))
            per_group.append(float(_term_z_core(W[gi:gi + 1, inv], M64, dfj).max()))
        nulls.append(float(np.median(per_group)))
    null_strength = float(np.mean(nulls))
    vs_null = strength / max(null_strength, 1e-9)

    ns[key_added] = pd.DataFrame(rows)
    ns[f"{key_added}_metrics"] = dict(strength=strength, null_strength=null_strength,
                                      vs_null=vs_null, specificity=specificity,
                                      n_groups=int(len(prof.index)), contrast=contrast, tail=tail)
    log.info("text_enrichment[%s]: strength %.2f  vs_null %.2fx  specificity %.3f",
             tail, strength, vs_null, specificity)
    if vs_null < 1.2:
        log.warning("vs_null=%.2f -- this readout is at (or below) the shuffled-description "
                    "null, so its terms carry no more signal than random labelling. Do NOT read "
                    "specificity=%.3f as a sharper answer; sparse noise scores high there.",
                    vs_null, specificity)
    return adata if copy else None


# ======================================================================================
# gene sets: what else are the named hits enriched for
# ======================================================================================
def load_gmt(path: str, vocab=None, min_size: int = 10, max_size: int = 500) -> dict[str, list]:
    """Parse a GMT (`name<TAB>description<TAB>gene...`) into `{name: [genes]}`.

    Sizes are counted AFTER restricting to `vocab` (the prototype labels), because that is the
    number the null is matched on. Sets outside `[min_size, max_size]` are dropped: tiny ones are
    unstable and genome-scale ones say nothing about a specific readout.
    """
    vset = None if vocab is None else set(map(str, vocab))
    out = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        genes = [g.strip() for g in f[2:] if g.strip()]
        if vset is not None:
            genes = [g for g in dict.fromkeys(genes) if g in vset]
        if min_size <= len(genes) <= max_size:
            out[f[0]] = genes
    log.info("load_gmt: %s -> %d sets (size %d..%d after vocabulary restriction)",
             os.path.basename(str(path)), len(out), min_size, max_size)
    return out


# ======================================================================================
# confound controls
# ======================================================================================
def hubness(P: np.ndarray, block: int = 1000) -> np.ndarray:
    """Mean |cosine| of each prototype to every other one.

    A prototype close to everything scores well on everything. Worth measuring because it is the
    other way a perturbation ranks highly without biology, and ruling it out is what let one of
    our results survive review.
    """
    Pn = _unit(np.asarray(P, np.float64)).astype(np.float32)
    n = Pn.shape[0]
    out = np.zeros(n)
    for s in range(0, n, block):
        e = min(s + block, n)
        C = np.abs(Pn[s:e] @ Pn.T)
        C[np.arange(e - s), np.arange(s, e)] = 0.0
        out[s:e] = C.sum(1) / max(n - 1, 1)
    return out


def training_support(h5ad_path: str, labels, label_key: str = "perturbation",
                     dataset_key: str = "dataset", index_file: str | None = None,
                     chunk: int = 2_000_000) -> pd.DataFrame:
    """Per-perturbation training support: cells, datasets, and (dataset, label) blocks.

    Read straight from the training store's `obs`, streamed, and optionally restricted to one
    fold's train indices (`index_file`, one 0-based row index per line, gzip fine) so the number
    reflects what the checkpoint actually saw.

    Use with `stratified_z`: a readout correlates weakly but non-zero with abundance, so a hit
    from a well-represented gene family has to be checked against equally-represented
    perturbations rather than against the whole table.
    """
    import gzip

    import h5py

    with h5py.File(h5ad_path, "r") as h:
        n = h[f"obs/{label_key}"].shape[0]
        mask = np.ones(n, bool)
        if index_file:
            opener = gzip.open if str(index_file).endswith(".gz") else open
            with opener(index_file, "rt") as f:
                idx = np.fromiter((int(x) for x in f), dtype=np.int64)
            mask = np.zeros(n, bool)
            mask[idx] = True
            log.info("training_support: restricted to %d of %d rows", int(mask.sum()), n)
        has_ds = f"obs/{dataset_key}" in h
        n_cells: dict[str, int] = {}
        blocks: set[tuple[str, str]] = set()
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            m = mask[s:e]
            if not m.any():
                continue
            lab = h[f"obs/{label_key}"][s:e].astype(str)[m]
            u, c = np.unique(lab, return_counts=True)
            for kk, vv in zip(u, c):
                n_cells[kk] = n_cells.get(kk, 0) + int(vv)
            if has_ds:
                dsn = h[f"obs/{dataset_key}"][s:e].astype(str)[m]
                blocks.update(map(tuple, np.unique(np.stack([lab, dsn], 1), axis=0)))
    n_ds: dict[str, int] = {}
    for lab, _ in blocks:
        n_ds[lab] = n_ds.get(lab, 0) + 1
    labels = list(map(str, labels))
    return pd.DataFrame({"perturbation": labels,
                         "n_train": [n_cells.get(g, 0) for g in labels],
                         "n_datasets": [n_ds.get(g, 0) for g in labels]})


def attach_prototype_covariates(adata, model=None, P: np.ndarray | None = None, support=None,
                                hubness_: bool = True, labels=None, uns_key: str = UNS_KEY,
                                copy: bool = False):
    """Attach the per-prototype covariates that a ranking is then corrected within.

    support   a `training_support()` DataFrame, a `{gene: n_cells}` mapping, or an array in label
              order. Prototypes with no training support get 0, which puts them in their own
              decile -- the only meaningful comparison for a never-perturbed (filled) row.
    hubness_  compute mean |cosine| to every other prototype from `P` (or from `model`).

    Once attached, `score_prototypes` and `top_perturbations` default to stratifying by support
    (falling back to hubness), which is what `stratify='auto'` resolves.
    """
    adata = adata.copy() if copy else adata
    ns = _ns(adata, uns_key, create=True)
    if labels is None:
        labels = [str(g) for g in ns["prototypes"]["labels"]] if "prototypes" in ns else None
    if labels is None:
        raise ValueError("labels unknown -- run score_prototypes(...) first or pass labels=")
    labels = [str(g) for g in labels]
    covs = dict(ns.get(COV_KEY) or {})
    if support is not None:
        if isinstance(support, pd.DataFrame):
            col = "n_train" if "n_train" in support else support.columns[-1]
            support = dict(zip(support["perturbation"].astype(str), support[col]))
        if isinstance(support, dict):
            support = [float(support.get(g, 0.0)) for g in labels]
        support = np.asarray(support, np.float64)
        if len(support) != len(labels):
            raise ValueError(f"support has {len(support)} entries for {len(labels)} prototypes")
        covs["support"] = support
        log.info("attach_prototype_covariates: support attached (%d of %d prototypes at 0)",
                 int((support == 0).sum()), len(labels))
    if hubness_:
        if P is None:
            if model is None:
                raise ValueError("hubness needs P= or model=")
            P = perturbation_prototypes(model, adata, which="auto")[2]
        h = hubness(np.asarray(P, np.float64))
        if len(h) != len(labels):
            raise ValueError(f"P has {len(h)} rows for {len(labels)} prototypes")
        covs["hubness"] = h
    ns[COV_KEY] = covs
    return adata if copy else None


def stratified_z(values, covariate, n_bins: int = 10) -> np.ndarray:
    """z-score `values` WITHIN bins of `covariate`.

    The question a raw rank cannot answer: is this perturbation's score high *for one of its
    abundance / hubness*, or just high because well-represented prototypes score high generally?
    """
    v = np.asarray(values, np.float64)
    c = np.asarray(covariate, np.float64)
    out = np.zeros_like(v)
    try:
        b = np.asarray(pd.qcut(c, n_bins, labels=False, duplicates="drop"))
    except ValueError:                       # pragma: no cover - degenerate covariate
        b = np.zeros(len(v), int)
    for kk in pd.unique(b[~pd.isna(b)]):
        m = b == kk
        out[m] = (v[m] - v[m].mean()) / max(v[m].std(), 1e-12)
    return out
