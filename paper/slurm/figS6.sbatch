#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 2G --time 00:10:00
#SBATCH --job-name figS6
#SBATCH --output paper/slurm/logs/figS6_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S6 (zero-shot prototype geometry and the fill map, related to Fig. 3). STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS6/data/   (figS6_data.py; Fig. 3's radial table, the fill sweeps of
#           analysis/readout/fill_fix/ -- 12-fold k and weighting tables (summarize.py) and the fold-4 arms --, the
#           fill-map scan analysis_out/mapcache/scores_nqxnds_<map>.tsv, and data/ext/ from figS6_ext.sbatch)
#   plot -> paper/figures/si/figS6/        (render-only; fine on a login node)
# Reads Fig. 3's data (fig3.sbatch STEP=data). Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
F3=$PAPER_FIGS/fig3/data
OUT=paper/figures/si/figS6
if [[ $STEP == data || $STEP == all ]]; then
  FF=analysis_out/fill_fix
  python paper/scripts/si/figS6_data.py --radial $F3/fig3b_radial.parquet \
    --wk12 $FF/wk12_summary.csv --w12 $FF/w12_summary.csv --w-fold4 $FF/w_fold4/k.json --wk-fold4 $FF/wk_fold4/k.json \
    --map-scores-dir analysis_out/mapcache --shipped-banks depmap_dstd+string+text \
    --extension $EXPERT_ROOT/$OUT/data/ext/fig3_extension.csv --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS6_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
