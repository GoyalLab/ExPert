#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 2 --mem=8G -t 00:20:00 -J fig3dblpl
#SBATCH -o $REPO/paper/slurm/logs/fig3dblpl_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 3 H/I: panels (any@k, in-panel doubles) -> fig3/.
# Plot-only; run after fig3_doubles.sbatch. The matched candidate space is Supplementary Fig. S5 (figS5.sbatch).
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export CUDA_VISIBLE_DEVICES=""
F3=$PWD/paper/figures/fig3
python -u paper/scripts/fig3/fig3_doubles_plot.py \
  --data-dir $PAPER_FIGS/fig3/data/doubles --out-dir $F3 --source-dir $PAPER_FIGS/fig3/data \
  --vizstyle-root $EXPERT_DATA_REPO 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
