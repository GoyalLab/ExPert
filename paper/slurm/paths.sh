# Site paths for the paper's SLURM recipes. Every paper/slurm/*.sbatch sources this file first, so submit from the
# repository root:  sbatch [-A <account> -p <partition>] paper/slurm/<recipe>.sbatch
# Set the roots below for your system, or put your values in paper/slurm/paths.local.sh (git-ignored), which is
# sourced last and overrides anything here.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this repository

# Paper data root: trained models (models/ExPert_v0.1.0, models/folds/, models/v0.1.7/EFFGATE3), inference queries
# and their results (data/inference/<query>/), gene sets (resources/genesets) and benchmark tables (benchmarks/).
EXPERT_ROOT=${EXPERT_ROOT:-CHANGE_ME/expert}
# Where the cache steps write each figure's plotting data (<fig>/data). Point it at paper/figures to write into the
# repository; the published figure-data archive (Zenodo) has the same layout.
PAPER_FIGS=${PAPER_FIGS:-$EXPERT_ROOT/paper/figures}
# Training store built by ExPert-data: data/store_out/<hash>/ (training database, splits) and
# perturb-seq/store/<hash>/ (per-dataset obs, stats, pseudobulks), data/merge/ (merge inputs).
DL_ROOT=${DL_ROOT:-CHANGE_ME/dl}
# A checkout of https://github.com/GoyalLab/ExPert-data
EXPERT_DATA_REPO=${EXPERT_DATA_REPO:-CHANGE_ME/ExPert-data}
# A checkout of ExPert-benchmarks (metrics/, config/, and its snakemake conda envs) and its results (benchmarks/)
EXPERT_PROJECT=${EXPERT_PROJECT:-CHANGE_ME/expert_project}
# The fingerprinting benchmark inputs/results (Fig. 4 sci-Plex3 fingerprinting comparison)
COLLAB_DATA=${COLLAB_DATA:-CHANGE_ME/ExPert_real_data}
# Staging area for the Zenodo archives (zenodo_upload.sbatch)
GROUP_WORK=${GROUP_WORK:-CHANGE_ME/group_work}

# Conda environments: EXPERT_ENV has this package installed (pip install -e ".[all]"); AUX_ENV only needs R with
# Seurat (fig5_gse281574_fetch.sbatch reads the GEO Seurat object).
EXPERT_ENV=${EXPERT_ENV:-expert}
AUX_ENV=${AUX_ENV:-expert_aux}
activate_env() { source ~/.bashrc; conda activate "$1"; }

if [[ -f "$REPO/paper/slurm/paths.local.sh" ]]; then source "$REPO/paper/slurm/paths.local.sh"; fi
