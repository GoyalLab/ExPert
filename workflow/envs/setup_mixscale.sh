#!/bin/bash

# Install SeuratDisk h5ad file support
R -e 'if (!requireNamespace("SeuratDisk", quietly=T)) remotes::install_github("mojaveazure/seurat-disk", INSTALL_opts = "--no-lock")'
# Pull mixscale from github
R -e 'if (!requireNamespace("Mixscale", quietly=T)) remotes::install_github("longmanz/Mixscale", INSTALL_opts = "--no-lock")'
# 7. (Optional) Install presto for faster wilcoxon test (used in DE)
R -e 'if (!requireNamespace("presto", quietly=T)) remotes::install_github("immunogenomics/presto", INSTALL_opts = "--no-lock")'
