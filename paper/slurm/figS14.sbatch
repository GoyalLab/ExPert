#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 8G --time 00:15:00
#SBATCH --job-name figS14
#SBATCH --output paper/slurm/logs/figS14_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S14 (PBMC overview: cell types, donors, OXPHOS-loss score; related to Fig. 6). STEP=data|plot|all.
#   data -> $PAPER_FIGS/si/figS14/data/   (figS14_data.py; Fig. 6's per-cell table, fig6_pmd.sbatch STEP=cells, and
#           the query h5ad's SingleR labels, obs only)
#   plot -> paper/figures/si/figS14/        (render-only; fine on a login node)
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS14
export PYTHONPATH=.
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS14_data.py --cells $PAPER_FIGS/fig6/data/pmd_mtmt/score/cells.csv.gz \
    --h5ad $EXPERT_ROOT/data/inference/metabolic_weinberg/converted_data/s_pbmc_collab.h5ad \
    --min-cells 100 --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS14_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
