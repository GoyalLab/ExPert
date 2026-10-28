#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 24G --time 01:00:00
#SBATCH --job-name figS10
#SBATCH --output paper/slurm/logs/figS10_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S10 (Proxy Anchor vs a softmax head: the gradient once the correct prototype wins). STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS10/data/   (figS10_data.py; controlled geometry at the shipped full-scale shape, the
#           shipped loss settings: v0.1.0 cosine head scale 30, proxy_anchor_alpha 32 / delta 0.1; CPU, torch)
#   plot -> paper/figures/si/figS10/        (render-only; fine on a login node)
# Paths and settings live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS10
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS10_data.py --batch 16384 --dim 512 --classes 8964 --ce-scale 30 --alpha 32 --delta 0.1 \
    --out-dir $EXPERT_ROOT/$OUT/data 2>&1 | grep -av --line-buffered -E "NumExpr|oneDNN|TensorFlow"
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS10_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
