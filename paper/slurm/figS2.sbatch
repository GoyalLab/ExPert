#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:15:00
#SBATCH --job-name figS2
#SBATCH --output paper/slurm/logs/figS2_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S2 (the prior pull during training, related to Fig. 3): A pulled-set size (formerly S1F), B gated
# vs equal vs no pull (formerly Fig. 3D / S1G), C-E gain by training support and applied after vs during training
# (formerly S6). STEP=data | plot | pull | all.
#   data -> $PAPER_FIGS/si/figS2/data/   (figS2_data.py; A copies the effgate_decomp pull-set sweep; C-E fold-4
#           readout ranks, bootstrap over queries; E parses analysis/readout/effgate_decomp/decomp.py summary.txt)
#   plot -> paper/figures/si/figS2/        (A, C-E; render-only; fine on a login node)
#   pull -> B via figS2b_pull_plot.py (fig3_plot.py's panel code; render-only from the effgate_decomp records)
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS2
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS2_data.py --gates analysis_out/effgate_decomp/gates.npz \
    --ranks-dir analysis_out/v017_readout/ranks --base BASE --arms FULLOPEN27 EFFGATE3 \
    --summary analysis_out/effgate_decomp/summary.txt --trained EFFGATE3 \
    --pull-sweep analysis_out/effgate_decomp/dose_gate_sweep.tsv --fig1-data $PAPER_FIGS/fig1/data --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS2_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
if [[ $STEP == pull || $STEP == all ]]; then
  python paper/scripts/si/figS2b_pull_plot.py \
    --effgate-per-query analysis_out/effgate_decomp/per_query.tsv.gz \
    --ranks-dir analysis_out/v017_readout/ranks \
    --out-dir $OUT --src-dir $EXPERT_ROOT/$OUT/data --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
