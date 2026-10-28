"""Per-dataset perturbation-separability ranking, over the full canonical gene space.

Keyed on DATASET_HASH like the other per-dataset sidecars: a dataset's F-stat is a pure
function of its pseudobulk, so adding dataset N+1 leaves the other N cached.

The FULL vector is stored, not a top-n, so the panel size is a selection-time decision --
changing `n_genes_per_dataset` re-runs `build_recipe` only, never this.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(pb, dataset, out, variant, log=None):
    import logging

    import numpy as np
    import pandas as pd

    from src.decode import (
        dataset_rows, delta_vs_control, f_stat_one, load_pseudobulk, normalize)

    # variant filtered at READ time -- the file holds all three groupings and loading
    # them all to keep one costs 3x the peak memory
    X, obs, var = load_pseudobulk(pb, variant=variant)
    measured = (X > 0).any(axis=0)
    P = delta_vs_control(normalize(X, obs), obs)
    del X

    sub = dataset_rows(obs, dataset, variant=variant)
    if sub.size < 4:
        raise ValueError(
            f'{dataset}: only {sub.size} replicated perturbed pseudobulk group(s); '
            f'separability is undefined. Lower min_cells_per_rep or drop the dataset.')

    # Unmoderated ratio: moderating the denominator by a quantile of the within-variances
    # was tested against decodability and degrades it monotonically (raw 0.234, q=.01
    # 0.220, q=.10 0.176, q=.50 0.124) -- see progress.md.
    f = f_stat_one(P, obs, sub, var_prior_quantile=None)
    logging.info(f'{dataset}: {sub.size:,} groups, {int(measured.sum()):,} genes '
                 f'measured of {len(var):,}')

    pd.DataFrame({'ensembl_id': np.asarray(var).astype(str),
                  'f_stat': f.astype(np.float32),
                  'measured': measured}).to_parquet(out, index=False)
    logging.info(f'wrote {out}')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            pb=snakemake.input.pb,
            dataset=snakemake.params.name,
            out=snakemake.output.fstat,
            variant=snakemake.params.variant,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
