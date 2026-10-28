"""Selection: turn a config + the canonical store's sidecars into a recipe (Phase 4).

This is the layer that replaces everything the old pipeline baked into stored matrices.
It is PURE METADATA -- it never opens an X matrix -- so it runs in seconds on a login
node, and its output (``recipe.json``) is the reproducible artifact, not the h5ad.

    columns   per-dataset top-n_hvg from the stored stats -> union/intersection
              -> Ensembl->symbol dedup            -> col_idx into the vocabulary
    rows      pool membership (cross-dataset)
              & min_cells_per_perturbation (recounted AFTER the pool)
              | the STORED control barcodes       -> row mask per dataset

Two orderings are load-bearing and easy to get wrong:

  * ``min_cells_per_perturbation`` must recount the survivors of the pool filter, not
    the raw counts.
  * the control term is a UNION (``|=``), not an intersection -- controls bypass the
    pool test entirely, which is what the current pipeline does at
    ``src/preprocess.py:526-539``.

Efficiency is no longer a row filter (the absolute ``min_mixscale_score`` cut was
removed), but the SCALING still happens here, over the selected population, because the
stored value is a raw signed z.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# per-dataset sidecar loading
# ---------------------------------------------------------------------------

def load_sidecars(store: str | Path, dataset: str) -> dict:
    """Load the block-row-space obs, HVG stats and efficiency for one dataset."""
    store = Path(store)
    # Only the three columns selection actually uses.  Reading all nine (including the
    # object-dtype barcode/label_raw and four QC floats) held ~7.5 GB across 22 datasets
    # and OOM-killed build_recipe on a login node, contradicting this module's docstring.
    obs = pd.read_parquet(store / 'obs' / f'{dataset}.parquet',
                          columns=['barcode', 'label', 'is_ctrl', 'qc_pass'])
    if 'qc_pass' in obs.columns:
        # the block contains only qc_pass rows, so this IS the block's row space
        obs = obs[obs.qc_pass].reset_index(drop=True)
    hvg = pd.read_parquet(store / 'stats' / f'{dataset}.hvg.parquet')

    eff_p = store / 'stats' / f'{dataset}.eff.parquet'
    eff = pd.read_parquet(eff_p) if eff_p.exists() else None
    if eff is not None and len(eff) != len(obs):
        raise ValueError(f'{dataset}: efficiency has {len(eff)} rows, obs has {len(obs)}')

    # The control set comes from obs['is_ctrl'] on the qc_pass rows, which IS the stored
    # choice (canonicalize wrote both from the same mask).  ctrl/{ds}.txt is kept as an
    # auditable record, not read here.
    return {'obs': obs, 'hvg': hvg, 'eff': eff}


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------

def select_columns(
    sidecars: dict[str, dict],
    vocab: pd.DataFrame,
    n_hvg: int = 3000,
    var_merge: str = 'union',
) -> dict:
    """Pick the gene axis, then collapse Ensembl -> symbol.

    The store is Ensembl-keyed (unambiguous), but the training DB is symbol-keyed, so
    the collapse happens HERE, per config.  Reproduces ``src/preprocess.py:315``
    ``ens_to_symbol``: when several Ensembl IDs in the SELECTED set map to one symbol,
    keep the one with the highest harmonic mean of ``means`` and ``variances_norm``.
    Only 8 symbols corpus-wide carry >1 Ensembl, so this drops <=8 columns -- but it
    must be a SELECTION, not a sum, to match current behaviour.
    """
    per_ds = {}
    for ds, sc in sidecars.items():
        h = sc['hvg']
        order = np.argsort(-h['variances_norm'].to_numpy())
        per_ds[ds] = set(order[:n_hvg].tolist())

    sets = list(per_ds.values())
    if var_merge == 'intersection':
        cols = set.intersection(*sets) if sets else set()
    else:
        cols = set.union(*sets) if sets else set()
    col_idx = np.sort(np.fromiter(cols, dtype=np.int64))
    logging.info(f'columns: {len(col_idx):,} after {var_merge} of per-dataset top-{n_hvg}')

    # --- Ensembl -> symbol dedup ---
    symbols = vocab['symbol'].to_numpy().astype(str)[col_idx]
    # harmonic mean of mean-expression and normalised variance, averaged across datasets
    means = np.zeros(len(col_idx))
    vnorm = np.zeros(len(col_idx))
    for sc in sidecars.values():
        h = sc['hvg']
        means += h['means'].to_numpy()[col_idx]
        vnorm += h['variances_norm'].to_numpy()[col_idx]
    n = max(len(sidecars), 1)
    means /= n
    vnorm /= n
    hm = (2 * means * vnorm) / (means + vnorm + 1e-9)

    df = pd.DataFrame({'col': col_idx, 'symbol': symbols, 'hm': hm})
    keep = df.groupby('symbol', observed=True)['hm'].idxmax().to_numpy()
    keep_df = df.loc[np.sort(keep)].reset_index(drop=True)
    n_dropped = len(df) - len(keep_df)
    if n_dropped:
        logging.info(f'symbol dedup: dropped {n_dropped} column(s) whose symbol was '
                     f'claimed by a higher harmonic-mean Ensembl ID')
    return {'col_idx': keep_df['col'].to_numpy().astype(np.int64),
            'symbols': keep_df['symbol'].to_numpy().astype(str),
            'n_before_dedup': int(len(df)), 'n_dropped_dedup': int(n_dropped)}


def select_columns_fstat(
    store: str | Path,
    datasets: list[str],
    vocab: pd.DataFrame,
    n_genes_per_dataset: int = 500,
) -> dict:
    """Gene axis from per-dataset perturbation SEPARABILITY, with per-dataset membership.

    Differs from ``select_columns`` in two ways that matter downstream:

      * genes are ranked by how well they discriminate perturbations from each other
        (``fstat/{dataset}.parquet``), not by unsupervised variance;
      * the per-dataset membership is returned, because each dataset is ZERO-PADDED
        outside its own top-n.  Keeping the other datasets' genes measurably hurts --
        every one of 21 datasets scored worse, mean -10% relative (progress.md), so the
        union is the DB's column axis but not every dataset's content.

    Returns ``col_idx`` (ascending vocabulary positions), ``symbols``, and
    ``col_masks``: per dataset, a boolean over ``col_idx`` marking its own selection.
    """
    store = Path(store)
    datasets = sorted(datasets)
    sel = {}
    for ds in datasets:
        f = store / 'fstat' / f'{ds}.parquet'
        if not f.exists():
            raise FileNotFoundError(
                f'{f} missing -- gene_selection=fstat needs the gene_fstat rule to have '
                f'run for every dataset (it depends on pseudobulk/).')
        t = pd.read_parquet(f)
        score = np.where(t['measured'].to_numpy(),
                         t['f_stat'].to_numpy().astype(np.float64), -np.inf)
        sel[ds] = np.argsort(-score)[:n_genes_per_dataset]
        logging.info(f'  {ds}: top {len(sel[ds]):,} of '
                     f'{int(t["measured"].sum()):,} measured')

    col_idx = np.unique(np.concatenate([sel[d] for d in datasets]))
    symbols = vocab['symbol'].to_numpy().astype(str)[col_idx]

    # Ensembl -> symbol dedup, matching select_columns: the DB is symbol-keyed, and two
    # Ensembl IDs sharing a symbol would otherwise become two ~80%-empty columns.
    # Rank by breadth then mean F so the retained ID is the better-supported one.
    pos = {int(c): i for i, c in enumerate(col_idx)}
    breadth = np.zeros(len(col_idx), dtype=np.int64)
    fsum = np.zeros(len(col_idx))
    for ds in datasets:
        t = pd.read_parquet(store / 'fstat' / f'{ds}.parquet')
        fv = t['f_stat'].to_numpy().astype(np.float64)
        for g in sel[ds]:
            breadth[pos[int(g)]] += 1
            fsum[pos[int(g)]] += fv[g]
    df = pd.DataFrame({'col': col_idx, 'symbol': symbols,
                       'score': breadth + fsum / max(len(datasets), 1) * 1e-6})
    keep = df.groupby('symbol', observed=True)['score'].idxmax().to_numpy()
    keep_df = df.loc[np.sort(keep)].reset_index(drop=True)
    n_dropped = len(df) - len(keep_df)
    if n_dropped:
        logging.info(f'symbol dedup: dropped {n_dropped} column(s) whose symbol was '
                     f'claimed by a better-supported Ensembl ID')
    col_idx = keep_df['col'].to_numpy().astype(np.int64)

    # per-dataset membership over the FINAL col_idx -- this is what zero-pads
    pos = {int(c): i for i, c in enumerate(col_idx)}
    col_masks = {}
    for ds in datasets:
        m = np.zeros(len(col_idx), dtype=bool)
        m[[pos[int(g)] for g in sel[ds] if int(g) in pos]] = True
        col_masks[ds] = m
    logging.info(f'columns: {len(col_idx):,} (union of {len(datasets)} x top-'
                 f'{n_genes_per_dataset}); zero-padded density '
                 f'{np.mean([m.mean() for m in col_masks.values()]):.1%}')
    return {'col_idx': col_idx,
            'symbols': keep_df['symbol'].to_numpy().astype(str),
            'col_masks': col_masks,
            'n_before_dedup': int(len(df)), 'n_dropped_dedup': int(n_dropped)}


def build_feature_mask(
    datasets: list[str],
    col_idx: np.ndarray,
    vocab: pd.DataFrame,
    raw_dir: str | Path,
) -> tuple[np.ndarray, dict[str, int]]:
    """(n_genes x n_datasets) 0/1 mask of ACTIVE vs zero-padded genes, + the column map.

    A gene is active for a dataset when that dataset's raw ``var`` actually measured it.
    Zero-padding is free in CSR, so a padded gene is indistinguishable from a measured
    gene that happens to be all-zero -- downstream code needs this mask to tell them
    apart, e.g. to avoid training on structural zeros.

    Reproduces the old ``varm/feature_mask_per_dataset`` + ``uns/dataset_to_mask_idx``
    (``src/merge.py:40-52``): float64 0.0/1.0, datasets indexed in **sorted name order**,
    matching the reference DB.

    Derived from each raw file's /var (metadata only, seconds per dataset) using the same
    resolver canonicalize used, so the mask is exactly "was this gene in the source".
    """
    from src.vocab import _read_var, resolve_ensembl_mapped

    datasets = sorted(datasets)
    raw_dir = Path(raw_dir)
    vocab_pos = {e: i for i, e in enumerate(vocab['ensembl_id'].astype(str))}
    mask = np.zeros((len(col_idx), len(datasets)), dtype=np.float64)
    out_of = {int(c): i for i, c in enumerate(col_idx)}

    for j, ds in enumerate(datasets):
        var = _read_var(raw_dir / f'{ds}.h5ad')
        ens, _ = resolve_ensembl_mapped(var, ds)
        src_idx = var.index.to_numpy().astype(str)
        n_hit = 0
        for i, e in enumerate(ens):
            key = str(e) if e is not None and str(e) != 'None' else src_idx[i]
            c = vocab_pos.get(key)
            if c is not None and c in out_of:
                mask[out_of[c], j] = 1.0
                n_hit += 1
        logging.info(f'  feature mask {ds:38s} {n_hit:>6,}/{len(col_idx):,} active')

    return mask, {ds: i for i, ds in enumerate(datasets)}


# ---------------------------------------------------------------------------
# perturbation pool (the one genuinely cross-dataset step)
# ---------------------------------------------------------------------------

def build_pool(
    sidecars: dict[str, dict],
    broad_keys: dict[str, str],
    min_cells_per_class: int = 20,
    min_dataset_frac: float = 0.6,
) -> set:
    """Perturbations present in enough distinct dataset GROUPS.

    Mirrors ``src/info.py:70-94``: grouping is by BROAD_DATASET_KEY and the threshold is
    ``max(3, int(n_groups * frac))``.  Datasets whose broad key is NaN do not vote (a
    pre-existing quirk: 'Wong2026' + '_' + NaN == NaN, so groupby drops them) -- they
    are filtered BY the pool but contribute nothing TO it.
    """
    by_group: dict[str, set] = {}
    for ds, sc in sidecars.items():
        g = broad_keys.get(ds)
        if g is None or (isinstance(g, float) and np.isnan(g)) or str(g) == 'nan':
            continue
        obs = sc['obs']
        lab = obs.loc[~obs.is_ctrl.astype(bool), 'label']
        vc = lab.value_counts()
        keep = {str(p) for p, c in vc.items()
                if c >= min_cells_per_class and str(p) not in ('', 'nan', 'None')}
        by_group.setdefault(str(g), set()).update(keep)

    n_groups = len(by_group)
    if n_groups == 0:
        logging.warning('no voting dataset groups; pool is empty')
        return set()
    min_groups = max(3, int(n_groups * min_dataset_frac))
    counts: dict[str, int] = {}
    for s in by_group.values():
        for p in s:
            counts[p] = counts.get(p, 0) + 1
    pool = {p for p, c in counts.items() if c >= min_groups}
    logging.info(f'pool: {len(pool):,} perturbations present in >={min_groups} of '
                 f'{n_groups} groups (min_dataset_frac={min_dataset_frac})')
    return pool


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def select_rows(
    sidecars: dict[str, dict],
    pool: set,
    min_cells_per_perturbation: int = 50,
    zs_min_cells_per_perturbation: int = 0,
    zs_eval_cap: int = 0,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-dataset row masks over the block's row space: (selected, zero-shot).

    The zero-shot mask is the COMPLEMENT of the pool among perturbed cells -- every
    perturbation the shared pool excluded.  Those cells are otherwise discarded, so
    routing them to a third output costs one extra write on a pass we are making
    anyway, instead of a separate extraction job over the raw files.

    ``zs_eval_cap`` subsamples it to at most that many cells per
    (dataset x perturbation) group, the same rule val/test use.  The zero-shot set is an
    EVALUATION set, so breadth (all 17,305 perturbations represented) matters and depth
    does not: capping at 10 keeps 8.4% of the cells and turns a ~64 GB output into ~5 GB.
    Sampling is seeded, so the chosen cells are reproducible.
    """
    rng = np.random.default_rng(seed)
    masks, zs_masks = {}, {}
    for ds, sc in sidecars.items():
        obs = sc['obs']
        labels = obs['label'].to_numpy().astype(str)
        is_ctrl = obs['is_ctrl'].to_numpy().astype(bool)

        mask = np.isin(labels, list(pool)) & ~is_ctrl

        # recount AFTER the pool filter, not on raw counts
        if min_cells_per_perturbation and min_cells_per_perturbation > 0:
            vc = pd.Series(labels[mask]).value_counts()
            ok = {str(p) for p, c in vc.items() if c >= min_cells_per_perturbation}
            mask &= np.isin(labels, list(ok))

        # controls bypass the pool test entirely -- UNION, not intersection
        mask |= is_ctrl

        masks[ds] = mask

        # zero-shot: perturbed, valid label, NOT in the pool.  Disjoint from `mask`
        # by construction, so no cell can land in both outputs.
        valid = ~np.isin(labels, ['', 'nan', 'None'])
        zs = (~is_ctrl) & valid & ~np.isin(labels, list(pool))
        if zs_min_cells_per_perturbation > 0:
            vc = pd.Series(labels[zs]).value_counts()
            ok = {str(p) for p, c in vc.items() if c >= zs_min_cells_per_perturbation}
            zs &= np.isin(labels, list(ok))
        if zs_eval_cap > 0 and zs.any():
            keep = np.zeros(len(zs), dtype=bool)
            pos = np.where(zs)[0]
            lab_zs = labels[pos]
            order = np.argsort(lab_zs, kind='stable')
            ls = lab_zs[order]
            uniq, starts = np.unique(ls, return_index=True)
            ends = np.append(starts[1:], len(ls))
            for a, b in zip(starts, ends):
                grp = pos[order[a:b]]
                take = grp if len(grp) <= zs_eval_cap else rng.choice(
                    grp, zs_eval_cap, replace=False)
                keep[take] = True
            zs = keep
        assert not (zs & mask).any(), f'{ds}: zero-shot overlaps the selected set'
        zs_masks[ds] = zs

        logging.info(f'  {ds:38s} {int(mask.sum()):>9,}/{len(mask):>9,} rows '
                     f'({int((mask & ~is_ctrl).sum()):,} perturbed + '
                     f'{int((mask & is_ctrl).sum()):,} control)  '
                     f'zero-shot {int(zs.sum()):,}')
    return masks, zs_masks


def scale_efficiency(sidecars: dict[str, dict], masks: dict[str, np.ndarray],
                     q: float = 0.99) -> dict[str, np.ndarray]:
    """Per-dataset tanh saturation of the raw signed z, over the SELECTED population.

    This is ``_scale_scores`` moved to selection time.  Applying it here rather than at
    write time is the whole point: the quantile is a property of the population you
    chose, not of whatever happened to be in the file when scoring ran.  The sign is
    preserved (the original applied ``np.abs`` first, destroying it).
    """
    out = {}
    for ds, sc in sidecars.items():
        if sc['eff'] is None:
            out[ds] = None
            continue
        z = sc['eff']['efficiency_raw_z'].to_numpy(dtype=np.float64)
        sel = masks[ds]
        vals = z[sel]
        finite = vals[np.isfinite(vals)]
        M = float(np.quantile(np.abs(finite), q)) if finite.size else 0.0
        scaled = np.full(len(z), np.nan)
        if M > 0:
            scaled = np.sign(z) * M * np.tanh(np.abs(z) / M)
        out[ds] = scaled
    return out


# ---------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------

def build_recipe(
    store: str | Path,
    datasets: list[str],
    vocab: pd.DataFrame,
    broad_keys: dict[str, str],
    n_hvg: int = 3000,
    var_merge: str = 'union',
    gene_selection: str = 'hvg',
    n_genes_per_dataset: int = 500,
    zero_pad_selection: bool = False,
    min_cells_per_class: int = 20,
    min_dataset_frac: float = 0.6,
    min_cells_per_perturbation: int = 50,
    zs_min_cells_per_perturbation: int = 0,
    zs_eval_cap: int = 10,
    seed: int = 42,
    raw_dir: str | Path | None = None,
) -> dict:
    """Full selection. Returns a JSON-serialisable recipe plus the in-memory masks."""
    # Fail at second zero, not hours in.  DATASET_HASH folds in the vocabulary VERSION
    # string, not its contents, so a vocabulary that grew within v1 leaves older blocks
    # looking current while still declared at the old width.  Everything below indexes
    # per-dataset stats by vocabulary position, so a narrow block raises IndexError deep
    # in select_columns -- long after the caller has committed to the run.
    from src.canonical import check_store_width
    stale = check_store_width(store, list(datasets), vocab)
    if stale:
        raise ValueError(
            f'{len(stale)} block(s) are narrower than the {len(vocab):,}-gene '
            f'vocabulary: {stale}. The vocabulary grew after these were written. '
            f'Run `python scripts/widen_store.py` first -- it is a metadata-only '
            f'operation (X is never rewritten), because the vocabulary is append-only.')

    sidecars = {ds: load_sidecars(store, ds) for ds in datasets}

    if gene_selection == 'fstat':
        cols = select_columns_fstat(store, list(datasets), vocab,
                                    n_genes_per_dataset=n_genes_per_dataset)
    elif gene_selection == 'hvg':
        cols = select_columns(sidecars, vocab, n_hvg=n_hvg, var_merge=var_merge)
    else:
        raise ValueError(f"gene_selection must be 'hvg' or 'fstat', got "
                         f'{gene_selection!r}')
    # The panel membership is always CARRIED (training may want to mask with it), but it
    # is only baked into X when zero_pad_selection is on.  Baking it in is a supervised
    # transformation of the INPUT: the mask is derived from perturbation labels, and no
    # query dataset has an equivalent, so a model trained on hard-zeroed inputs sees a
    # pattern it cannot be given at inference.  Storing the full values keeps that a
    # reversible, train-time decision -- you can zero at load time, not un-zero.
    col_masks = cols.get('col_masks')
    pad_masks = col_masks if zero_pad_selection else None
    pool = build_pool(sidecars, broad_keys,
                      min_cells_per_class=min_cells_per_class,
                      min_dataset_frac=min_dataset_frac)
    masks, zs_masks = select_rows(
        sidecars, pool, min_cells_per_perturbation=min_cells_per_perturbation,
        zs_min_cells_per_perturbation=zs_min_cells_per_perturbation,
        zs_eval_cap=zs_eval_cap, seed=seed)
    eff_scaled = scale_efficiency(sidecars, masks)

    feature_mask, ds_to_idx = (None, None)
    if raw_dir is not None:
        feature_mask, ds_to_idx = build_feature_mask(
            list(datasets), cols['col_idx'], vocab, raw_dir)
        if pad_masks is not None:
            # Only when the selection is baked into X: a deselected gene is then an
            # unmeasured one as far as downstream can tell, so the mask must say so.
            # Without zero-padding, feature_mask keeps its original meaning (what the
            # dataset actually MEASURED), which is reproducible for a query dataset.
            for ds, j in ds_to_idx.items():
                feature_mask[:, j] *= pad_masks[ds].astype(feature_mask.dtype)

    total = int(sum(int(m.sum()) for m in masks.values()))
    recipe = {
        'datasets': list(datasets),
        'n_genes': int(len(cols['col_idx'])),
        'n_cells': total,
        'params': {
            'n_hvg': n_hvg, 'var_merge': var_merge,
            'gene_selection': gene_selection,
            'n_genes_per_dataset': n_genes_per_dataset,
            'zero_pad_selection': zero_pad_selection,
            'min_cells_per_class': min_cells_per_class,
            'min_dataset_frac': min_dataset_frac,
            'min_cells_per_perturbation': min_cells_per_perturbation,
            'zs_eval_cap': zs_eval_cap, 'seed': seed,
        },
        'columns': {'n_before_dedup': cols['n_before_dedup'],
                    'n_dropped_dedup': cols['n_dropped_dedup'],
                    **({'zero_padded_density':
                        float(np.mean([m.mean() for m in col_masks.values()]))}
                       if col_masks is not None else {})},
        'pool_size': len(pool),
        'n_cells_zero_shot': int(sum(int(m.sum()) for m in zs_masks.values())),
        'per_dataset': {ds: {'n_rows_block': int(len(m)),
                             'n_rows_selected': int(m.sum())}
                        for ds, m in masks.items()},
    }
    return {'recipe': recipe, 'col_idx': cols['col_idx'], 'symbols': cols['symbols'],
            'col_masks': col_masks, 'pad_masks': pad_masks,
            'masks': masks, 'pool': pool, 'sidecars': sidecars,
            'eff_scaled': eff_scaled, 'feature_mask': feature_mask,
            'dataset_to_mask_idx': ds_to_idx, 'zs_masks': zs_masks}


def save_recipe(sel: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sel['recipe'], indent=2, default=str))
    arrays = {
        'col_idx': sel['col_idx'],
        'symbols': np.asarray(sel['symbols'], dtype='U'),
        **{f'mask__{ds}': m for ds, m in sel['masks'].items()},
        **{f'zs__{ds}': m for ds, m in sel.get('zs_masks', {}).items()},
        # Panel membership per dataset.  ALWAYS stored: when zero_pad_selection is off
        # this is metadata for training-time masking, not something baked into X.
        **{f'colmask__{ds}': m
           for ds, m in (sel.get('col_masks') or {}).items()},
        'zero_padded': np.array([bool(sel.get('pad_masks') is not None)]),
    }
    # The feature mask is derived from the raw /var files, so persisting it here means
    # materialize does not have to re-read 22 raw headers.
    if sel.get('feature_mask') is not None:
        arrays['feature_mask'] = sel['feature_mask']
    np.savez_compressed(str(p).replace('.json', '.npz'), **arrays)
