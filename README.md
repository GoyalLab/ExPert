<!--
SPDX-License-Identifier: MIT

ExPert: a contrastive perturbation-prediction model for single-cell data.

Copyright (c) 2024-2026 Leon Schwartz
Northwestern University, Goyal Lab

This file is part of the ExPert package and is released under the MIT
License. See the LICENSE file in the project root for the full text.
-->

# ExPert

ExPert reads a transcriptional response and tells you which genetic perturbations it resembles. It was trained on
22 Perturb-seq datasets. Each perturbation has a prototype in a shared latent space: 8,964 learned from the screens,
and the rest of the 19,895-gene vocabulary predicted from prior knowledge. Every cell or pseudobulk you give it
(single-cell screens, drug screens, disease atlases) is scored against all of them.

Paper: _link to the preprint_ · Training data: [ExPert-data](https://github.com/GoyalLab/ExPert-data)

## Installation

```bash
pip install expert-model            # core: load a model, predict, train
pip install "expert-model[all]"     # + plotting/enrichment extras, paper figure scripts, tests
```

From source (needed for the paper's figure scripts):

```bash
git clone https://github.com/GoyalLab/ExPert.git && cd ExPert
pip install -e ".[all]"
```

Python ≥ 3.10. Install the PyTorch build that matches your GPU first if you need CUDA; CPU is enough for prediction.

`environment.yaml` pins the exact environment the paper was produced with. Use it if you need bit-identical
numbers:

```bash
mamba env create -f environment.yaml && mamba activate expert && pip install -e . --no-deps
```

## Usage

```python
import scanpy as sc
from huggingface_hub import snapshot_download
from expert import ExPert

# v0.1.0 from https://huggingface.co/GoyalLab/ExPert (2.3 GB, cached after the first call)
model = ExPert.load(snapshot_download("GoyalLab/ExPert", revision="v0.1.0"))

query = sc.read_h5ad("query.h5ad")
# X: raw counts; var_names: gene symbols (mouse symbols are mapped to human automatically)
# obs['perturbation']: 'control' for control cells, anything else (e.g. 'query') for the rest

out = model.predict(query)
out["soft_predictions"]    # cells x perturbations, cosine to each prototype (higher = more alike)
                           # obs_only=True keeps only the 8,964 perturbations seen in training
out["prediction"]          # top-scoring perturbation per cell
out["z_shared"]            # the latent the scores are computed in
```

`predict(query, inplace=True)` writes the same results onto the AnnData instead (`obsm['soft_predictions']`,
`obs['prediction']`, …).

The query's own control cells are the baseline the response is measured against: label them `'control'`
(without controls, predict() falls back to the query mean and warns). For heterogeneous data such as disease atlases,
`center_by='cell_type'` makes each cell type its own baseline. Pseudobulks work the same way; pass the number of
cells behind each row as `n_cells_key='n_cells'`.

Other useful calls:

- `model.get_prototypes()`: the perturbation prototypes as a DataFrame (perturbations × latent dims).
- `predict(..., readout='deflate')`: after the top hits, keeps reading what they leave unexplained. Useful for
  responses with more than one mechanism.
- `predict(..., mask_genes='MT-*')`: recommended for queries that are not CRISPR screens (drug screens, disease
  atlases).

See `help(ExPert.predict)` for all options.

Downloads are cached in `~/.cache/huggingface`; set `HF_HOME` to put them elsewhere. A local folder holding
`model.pt` and `gene_embeddings.npz` works the same way: `ExPert.load("path/to/folder")`.

## Reproducing the paper

`paper/` in the [GitHub repo](https://github.com/GoyalLab/ExPert) holds everything behind the figures and tables. It is
not part of the PyPI package; clone the repo and install it with `pip install -e ".[all]"`.

- `paper/figures/<fig>/<fig>.ipynb`: one notebook per figure (main 1–6, SI S1–S14). Each one redraws exactly the
  panels in the paper from that figure's plotting data. See [paper/figures/README.md](https://github.com/GoyalLab/ExPert/blob/main/paper/figures/README.md).
- Plotting data: [doi:10.5281/zenodo.23740198](https://doi.org/10.5281/zenodo.23740198).
  `python paper/figures/fetch_figure_data.py` downloads it and unpacks every figure's `data/` and `inputs/` in place.
- `paper/scripts/` and `paper/slurm/`: the scripts that compute that plotting data from the model and the screens,
  and the SLURM recipes with their exact arguments (set your paths in `paper/slurm/paths.sh`).
- Checkpoints: the production model is [GoyalLab/ExPert](https://huggingface.co/GoyalLab/ExPert); the 12
  cross-validation folds and the ablation arm are in
  [GoyalLab/ExPert-paper-models](https://huggingface.co/GoyalLab/ExPert-paper-models).
- `paper/reproducibility/`: which inputs each figure reads, and where every source comes from.
- Other data on Zenodo: [source data](https://doi.org/10.5281/zenodo.23740344) (new datasets from this study),
  [gene embeddings](https://doi.org/10.5281/zenodo.23741208) (text and prior banks) and
  [derived data](https://doi.org/10.5281/zenodo.23742338) (Tahoe-100M pseudobulks, baseline outputs, benchmark
  tables). Table S1A lists every dataset with its accession.

## Tests

```bash
pytest                   # all tests, ~2 min on CPU (trains a small toy model)
pytest -m "not e2e"      # unit tests only, seconds
```

## License

MIT, see [LICENSE](https://github.com/GoyalLab/ExPert/blob/main/LICENSE).
