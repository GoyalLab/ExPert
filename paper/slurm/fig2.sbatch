#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 40G --time 01:30:00
#SBATCH --job-name fig2
#SBATCH --output paper/slurm/logs/fig2_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 2 benchmark, from the benchmark pipeline's CACHED collect tables (nothing is rescored).
# STEP=data | plot | all (default all). Extra args pass through to fig2_plot.py (e.g. --panels a+b).
# Main panels -> fig2/. Supplementary Fig. S3 has its own driver (figS3.sbatch) reading
# the same inputs (fig2_inputs.sh).
# Needs ExPert-benchmarks/utils/score_extra.sbatch output per size (extra_metrics.v2.csv): the
# per-perturbation F1/MRR defaults read it. MRR=weighted_mrr F1=weighted_macro_f1 restore collect's columns.
# Paths live here and in fig2_inputs.sh, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-all}
source paper/slurm/fig2_inputs.sh
OUT=paper/figures/fig2
[[ -d $SEL ]] && ls $SEL/*.parquet >/dev/null 2>&1 && EXTRA+=(--selective-dir "$SEL")
[[ -d $PB ]] && ls $PB/*.parquet >/dev/null 2>&1 && EXTRA+=(--pbulk-dir "$PB")   # pseudobulk A-D
EXTRA+=(--expert-train-root "$TRAIN_ROOT" --expert-eval-logs "$EVAL_LOGS" --expert-eval-cpus "$EVAL_CPUS")
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$USER"; mkdir -p "$MPLCONFIGDIR"

if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/fig2/fig2_data.py --ctx7-dir "$CTX7" --small-dir "$SMALL" --costs "$COSTS" \
    --cost-overrides "$OVERRIDES" --out-dir "$EXPERT_ROOT/$OUT/data" --mrr "${MRR:-balanced_weighted_mrr}" --f1 "${F1:-balanced_weighted_macro_f1}" "${EXTRA[@]}"
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/fig2/fig2_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" \
    --vizstyle-root "$DATA" --pdf fig2b_radar_spokes "$@"
fi
