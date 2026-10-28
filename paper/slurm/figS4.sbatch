#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 24G --time 00:15:00
#SBATCH --job-name figS4
#SBATCH --output paper/slurm/logs/figS4_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S4 (Fingerprint's cell assignment, related to Fig. 2), from the benchmark's cached per-cell
# scores (score_selective.py; inputs shared with fig2.sbatch via fig2_inputs.sh). STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS4/data/   (figS4_data.py; loading the per-cell cache is the slow part)
#   plot -> paper/figures/si/figS4/        (render-only; fine on a login node)
# Paths live here and in fig2_inputs.sh, never in the .py. Data step: 18.8 GB peak, 3.4 min (sacct 4991612).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
source paper/slurm/fig2_inputs.sh
OUT=paper/figures/si/figS4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$USER"; mkdir -p "$MPLCONFIGDIR"

if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS4_data.py --ctx7-dir "$CTX7" --selective-dir "$SEL" --out-dir "$EXPERT_ROOT/$OUT/data"
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS4_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" --vizstyle-root "$DATA"
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
