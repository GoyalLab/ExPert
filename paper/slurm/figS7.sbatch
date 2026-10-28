#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 2 --mem 8G -t 00:20:00 -J figS7
#SBATCH -o $REPO/paper/slurm/logs/figS7_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S7 (Figure 4B-E with EFFGATE3, no K562 in training; related to Fig. 4). STEP=data | plot | all.
# Upstream: fig4.sbatch STEP=sciplex_pb_{encode,score,panels} MTMASK=1 with
#   MDIR_OVERRIDE=<EFFGATE3>, SPB_OVERRIDE=<...>_effgate3, SPB_OUT_OVERRIDE=<...>_effgate3, and on panels
#   --resp-encode-dir <shipped v0.1.0 encode> so EFFGATE3 is scored on Figure 4's exact conditions;
#   fig4_flexplex.sbatch for panel D (FlexPlex). Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
F4D=$PAPER_FIGS/fig4/data/v0.1.0
OUT=paper/figures/si/figS7
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS7_data.py --eg3-panels $F4D/sciplex_pb_mtmask_fpcells_cc200_effgate3/panels \
    --v010-panels $F4D/sciplex_pb_mtmask_fpcells_cc200/panels --flexplex $F4D/flexplex/fig4e_ar42.csv \
    --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS7_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
