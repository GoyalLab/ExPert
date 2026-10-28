<!--
SPDX-License-Identifier: MIT

ExPert: a contrastive perturbation-prediction model for single-cell data.

Copyright (c) 2024-2026 Leon Schwartz
Northwestern University, Goyal Lab

This file is part of the ExPert package and is released under the MIT
License. See the LICENSE file in the project root for the full text.
-->

# Figures

Each figure folder (`fig1`…`fig6`, `si/figS1`…`si/figS14`) holds the canonical panel PNG/SVGs and a notebook `<fig>.ipynb`. The notebook redraws exactly the panels shown in the figure.

A notebook runs from its own folder and only uses relative paths:

| path | what |
|---|---|
| `data/` | the figure's plotting data (from the figure-data archive) |
| `inputs/` | the other files a panel reads: gene sets, another figure's data (from the figure-data archive) |
| `render/` | written by the notebook; `render/src/` holds the source tables, `render/compare.csv` the comparison with the canonical PNGs |

## Getting the plotting data

`data/` and `inputs/` are not in the repository. They are in the Zenodo record "ExPert - figure data",
[doi:10.5281/zenodo.23740198](https://doi.org/10.5281/zenodo.23740198): one zip per figure, 152 MB in total.
From the repository root:

```bash
python paper/figures/fetch_figure_data.py               # all figures
python paper/figures/fetch_figure_data.py --figs fig3   # one figure
```

It checks each zip's md5 against Zenodo and unpacks it into the figure's folder. The zips' members are
`figures/<fig>/...`; supplementary figures go to `figures/si/<fig>/` in this repository.

Then run a notebook from its folder, e.g. `cd paper/figures/fig1 && jupyter nbconvert --to notebook --execute fig1.ipynb` (needs `pip install -e ".[all]"`).

Plot scripts live in `paper/scripts/<fig1..fig6, si, tables>/`. Shared notebook helpers are in `paper/scripts/common/`; the style module is `paper/style/src/vizstyle.py`.

Execute every notebook and compare its panels with the canonical ones (SLURM):

    sbatch --array 1-$(grep -c . paper/figures/notebooks.txt) paper/slurm/notebooks.sbatch

List the files each notebook reads: `paper/slurm/notebooks_trace.sbatch` writes `analysis_out/paper_notebooks/trace/<fig>_reads.tsv`. The combined list is in `paper/reproducibility/plot_time_inputs.tsv`. Sources, caches and the Zenodo plan are in `paper/reproducibility/figure_sources.md`.
