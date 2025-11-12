# All relevant code to split full dataset into different training sets

import os
import logging
import anndata as ad

from itertools import product


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def filter_adata(
        adata: ad.AnnData,
        context_col: str = 'context',
        cls_col: str = 'cls_label',
        eff_col: str = 'mixscale_score',
        ctrl_key: str = 'control',
        min_mixscale_score: float = 0.0,
        min_cells: int = 100,
        min_contexts: int = 2,
        keep_ctrl: bool = True,
        inplace: bool = True,
    ) -> None:
    """Filter training adata"""
    # Mixscale filter
    ms_mask = (adata.obs[eff_col].abs() >= min_mixscale_score)
    filtered_obs = adata.obs[ms_mask]
    
    # Count number of perturbations across contexts
    cpp = (
        filtered_obs.groupby([cls_col, context_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    # Compute total counts per perturbation
    total_counts = cpp.groupby(cls_col, observed=True)["count"].sum()

    # Compute number of distinct contexts per perturbation
    context_counts = cpp.groupby(cls_col, observed=True)[context_col].nunique()

    # Filter for both conditions:
    # (1) enough total samples, (2) enough distinct contexts
    filtered_perturbations = total_counts[
        (total_counts >= min_cells) & (context_counts >= min_contexts)
    ].index.tolist()

    logging.info(
        f"Found {len(filtered_perturbations)}/{adata.obs[cls_col].nunique()} perturbations "
        f"with ≥{min_cells} cells total and present in ≥{min_contexts} {context_col}s."
    )
    # Create final filter mask
    p_mask = adata.obs[cls_col].isin(filtered_perturbations)
    mask = ms_mask & p_mask
    # Keep controls
    if keep_ctrl and ctrl_key in adata.obs[cls_col].unique():
        logging.info(f'Keeping control indices.')
        ctrl_mask = adata.obs[cls_col] == ctrl_key
        mask |= ctrl_mask
    # Subset adata
    if inplace:
        adata._inplace_subset_obs(mask)
    else:
        return adata[mask]
    
def save_splits(
        adata: ad.AnnData,
        output_dir: str,
        min_cells: list[int] = [100],
        min_contexts: list[int] = [2],
        min_mixscale: list[int] = [0, 1, 2],
        ctx_col: str = 'context',
        cls_col: str = 'cls_label',
    ) -> None:
    """Create a training dataset for each data filtering combination."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Loop over all combinations and save each filtered adata
    for mc, mctx, mms in product(min_cells, min_contexts, min_mixscale):
        # Filter data for combination
        filtered = filter_adata(
            adata,
            min_cells=mc,
            min_contexts=mctx,
            min_mixscale_score=mms,
            inplace=False
        ).copy()
        # Save dataset to disk
        nctx = filtered.obs[ctx_col].nunique()
        ncls = filtered.obs[cls_col].nunique()
        # Create unique file name
        fn = f'nctx:{nctx}_ncls:{ncls}_minms:{mms}_minc:{mc}_minctx:{mctx}.h5ad'
        f = os.path.join(output_dir, fn)
        logging.info(f'Saving filtered adata to: {f}')
        filtered.write_h5ad(f)


def load():
    pass