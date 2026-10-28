#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:20:00
#SBATCH --job-name figS11
#SBATCH --output paper/slurm/logs/figS11_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S11 (per-cell effect size predicted by the effect-size head of the production model).
# STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS11/data/   (figS11_data.py; held-out id_val cells scored by
#           analysis/effect_size/effect_head_fit.sbatch --skip-extras -> analysis_out/effect_conf/conf_cells.parquet,
#           combined by effect_stack.py -> stack_val_cells.parquet; the shipped readout = predict()'s
#           obs['effect_size'], reproduced from model.pt by analysis/effect_size/apply_effect_head.py)
#   plot -> paper/figures/si/figS11/        (render-only; fine on a login node)
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS11
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS11_data.py --cells analysis_out/effect_conf/stack_val_cells.parquet \
    --pred-col ridge_head_global_main --abs-measured --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS11_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
