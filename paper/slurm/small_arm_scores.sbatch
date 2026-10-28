#!/bin/bash
#SBATCH -A ACCOUNT
#SBATCH -p PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 16G --time 01:00:00
#SBATCH --job-name smallarm_scores
#SBATCH --output paper/slurm/logs/small_arm_scores_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S3 small-arm survey (S3C/S3D): per-epoch recall from every small arm's tfevents. Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
python paper/scripts/tables/small_arm_scores.py --out $PAPER_FIGS/si/tables/data/ablations_small_arm_scores_raw.csv \
  --roots $DL_ROOT/data/store_out/1e6bafc04836e06a/models/small \
          $DL_ROOT/data/store_out/a191ce7a37afc3f9/models/small \
          $DL_ROOT/data/merge/86ecffa180438d16/models/small_perc \
          $DL_ROOT/data/merge/86ecffa180438d16/models/small
