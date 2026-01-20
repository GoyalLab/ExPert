#!/bin/bash

# Make script fail if any installation throws an error
set -euo pipefail
# Install SeuratDisk h5ad file support
R -e 'if (!requireNamespace("SeuratDisk", quietly=T)) remotes::install_github("mojaveazure/seurat-disk", dependencies = FALSE)'
# Pull mixscale from github
R -e 'if (!requireNamespace("Mixscale", quietly=T)) remotes::install_github("longmanz/Mixscale", dependencies = FALSE)'
# 7. (Optional) Install presto for faster wilcoxon test (used in DE)
R -e 'if (!requireNamespace("presto", quietly=T)) remotes::install_github("immunogenomics/presto", dependencies = FALSE)'
