#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 16G --time 02:00:00
#SBATCH --job-name tableS2S3
#SBATCH --output paper/slurm/logs/table_s2_s3_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Tables S2 and S3 (ablations behind the model design, one part per development stage). Paths live here, never in
# the .py.
#   STEP=tables (default)  Table_S2.xlsx / Table_S3.xlsx + Table_S2A.csv ... -> paper/figures/si/tables/, from $PAPER_FIGS/si/tables/data/
#   STEP=curves            refresh ablations_training_curves.csv only (reads tfevents)
#   STEP=all               curves, then tables
# Other inputs in $PAPER_FIGS/si/tables/data/: ablations_inventory.csv (arm configs / baselines), ablations_scores_raw.csv
# (long-format scores), ablations_small_arm_survey.csv (small-scale arms, S3C/S3D; its per-epoch source
# ablations_small_arm_scores_raw.csv comes from small_arm_scores.sbatch).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
T=paper/figures/si/tables
D=$EXPERT_ROOT/$T/data
STEP=${STEP:-tables}
if [[ $STEP == curves || $STEP == all ]]; then
# in-training train/val/OOD fit summary (reads tfevents)
python paper/scripts/tables/table_s2_s3_training_curves.py --out "$D/ablations_training_curves.csv" \
  --roots $DL_ROOT/data/store_out/1e6bafc04836e06a/models/full \
          $EXPERT_ROOT/models/v0.1.0 $EXPERT_ROOT/models/v0.1.1
fi
[[ $STEP == curves ]] && exit 0
python paper/scripts/tables/table_s2_s3.py --inventory "$D/ablations_inventory.csv" \
  --scores "$D/ablations_scores_raw.csv" --curves "$D/ablations_training_curves.csv" \
  --small-log paper/slurm/logs/s3_ladder_readout.out \
  --legacy-scan-log analysis_out/logs/v1_scan.out \
  --legacy-models $DL_ROOT/data/merge/86ecffa180438d16/models \
  --small-survey "$D/ablations_small_arm_survey.csv" \
  --hparams-root $DL_ROOT/data/store_out/1e6bafc04836e06a/models/full \
  --out-dir $T
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
