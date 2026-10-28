# Analysis helpers

Scripts that the SLURM recipes in `paper/slurm/` or the figure notebooks call, kept at their original relative
layout:

| script | used by |
|---|---|
| `readout/cache_queries.py`, `readout/cache_zs_queries.py` | encode query cells for Figs 3, 5 and S6 (`fig3a_seen`, `fig3a_folds`, `fig5_hubness`, `figS6_ext`) |
| `readout/_tmp/make_cap_split.py` | the capped training-query split for `fig5_hubness` |
| `readout/prior_screen/` (`screen.py`, `plot.py`, `paths.sh`) | the prior screen behind Fig. 3 and Table S7; `plot.py` is also imported by the Fig. 3 notebook |
| `fingerprint/*.R` | the fingerprinting baseline (`fig4_sciplex_fp`, `fp_doubles_ourgwps`, `fp_singles_ourgwps`) |
| `tahoe_pseudobulk.py` | builds the Tahoe-100M pseudobulks deposited on Zenodo (ExPert - derived data) |
| `_fig4_*.py`, `_effgate3_train_coverage.py` | one-off checks behind Fig. 4 and Fig. S2 (`fig4_bankrefill`, `fig4_degron_proof`, `fig4_zs_offset`, `effgate3_train_coverage`) |

Comments elsewhere in `paper/` that name other `analysis/` scripts refer to exploratory code that is not published;
the tables those scripts produced are deposited on Zenodo.
