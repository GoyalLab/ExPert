#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 12G --time 00:45:00
#SBATCH --job-name fig6bulk
#SBATCH --output paper/slurm/logs/fig6bulk_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6G/H sorted-cell bulk RNA-seq: PCA, PERMANOVA (stim vs PMD), effect sizes, fresh-PMD check
# (paper/scripts/fig6/fig6_bulk.py). Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
B=$PAPER_FIGS/fig6/data/mito_bulk
# STEP=data (default): the canonical Fig. 6G/H analysis, which reproduces the collaborators'
#   paper/figures/fig6/ext/261022_pca_minimal.R, into $B/analysis_repro_collab. Passing arguments replaces the
#   canonical flags (TAG names the output dir, analysis_<TAG>).
# STEP=plot: Fig. 6G/H (fig6_bulk_plot.py, panels f2 + g2) and 6I flow cytometry (fig6_flow_plot.py); light, login is fine.
STEP=${STEP:-data}
VIZ=$EXPERT_DATA_REPO
SRC=$PAPER_FIGS/fig6/data
if [[ $STEP == data ]]; then
  if [[ $# -eq 0 ]]; then
    TAG=${TAG:-repro_collab}
    set -- --norm tximport --lengths $B/salmon/salmon.merged.gene_lengths.tsv --gene-filter mean --n-hvg 1000 \
      --n-perm 999 --exclude-samples 'CD8:P43;Mono:P75,P83'
  fi
  python paper/scripts/fig6/fig6_bulk.py --counts $B/salmon/salmon.merged.gene_counts.tsv \
    --meta $B/salmon/261014_sample_metadata.csv --out-dir $B/analysis${TAG:+_$TAG} --n-cpus ${SLURM_CPUS_PER_TASK:-4} "$@"
fi
if [[ $STEP == plot ]]; then
  python paper/scripts/fig6/fig6_bulk_plot.py --analysis-dir $B/analysis_repro_collab --out-dir paper/figures/fig6 \
    --src-dir $SRC --vizstyle-root $VIZ --panels f2+g2 --pc-rule other
  python paper/scripts/fig6/fig6_flow_plot.py --xlsx $SRC/flow/mito_flow_combined_raw.xlsx --out-dir paper/figures/fig6 \
    --src-dir $SRC --vizstyle-root $VIZ
fi
