#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 36G --time 00:30:00
#SBATCH --job-name figS3
#SBATCH --output paper/slurm/logs/figS3_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S3 (benchmark, related to Fig. 2), from the benchmark pipeline's CACHED tables (nothing is
# rescored; inputs shared with fig2.sbatch via fig2_inputs.sh). STEP=data | plot | all (default all).
#   data -> $PAPER_FIGS/si/figS3/data/   (figS3_data.py; the per-cell cache makes this the slow step)
#   plot -> paper/figures/si/figS3/ panels (render-only; fine on a login node)
# Paths live here, never in the .py.
# Memory: data step peaked at 29.1 GB in 9:49 (sacct 4986752).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
source paper/slurm/fig2_inputs.sh
OUT=paper/figures/si/figS3
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$USER"; mkdir -p "$MPLCONFIGDIR"

if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS3_data.py --ctx7-dir "$CTX7" --small-dir "$SMALL" --costs "$COSTS" \
    --cost-overrides "$OVERRIDES" --pbulk-dir "$PB" --selective-dir "$SEL" \
    --mask-ablation "${MASK[@]}" --mask-query "$MASK_QUERY" \
    --expert-train-root "$TRAIN_ROOT" --expert-train-ram-gb "$TRAIN_RAM_GB" \
    --mrr "${MRR:-balanced_weighted_mrr}" --f1 "${F1:-balanced_weighted_macro_f1}" \
    --out-dir "$EXPERT_ROOT/$OUT/data" "${EXTRA_CTX7[@]}"
fi
if [[ $STEP == plot || $STEP == all ]]; then
  # S2D = Figure 2C on single cells: the same benchmark table fig2_data.py writes (effect-size-weighted metrics)
  cp $PAPER_FIGS/fig2/data/fig2b_shift.csv "$EXPERT_ROOT/$OUT/data/figS3d_ood_by_shift_cell.csv"
  python paper/scripts/si/figS3_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" \
    --vizstyle-root "$DATA"
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
