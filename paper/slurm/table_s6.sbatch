#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 2G --time 00:10:00
#SBATCH --job-name tableS6
#SBATCH --output paper/slurm/logs/table_s6_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S6 (zero-shot recovery per screen and fold model, related to Fig. 3B) -> paper/figures/si/tables/Table_S6.*
# Reads Fig. 3's per-fold zero-shot records (fig3.sbatch STEP=zs per fold, fig3a_seen.sbatch). Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
F3=$PAPER_FIGS/fig3/data
python paper/scripts/tables/table_s6.py --folds-dir $F3/folds --contexts $F3/fig3a_contexts.tsv --check $F3/fig3a_source.csv \
  --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
