#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 8G --time 00:30:00
#SBATCH --job-name fig2head
#SBATCH --output paper/slurm/logs/fig2head_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 2 panel: PA vs MLP classifier head, same encoder and fold (K562 held out). Paths live here.
# Rank files come from expert.analysis.ood_readout_probe (analysis/readout/*readout.sbatch):
#   PA = the shipped recipe on fold 4; classifier = the classifier-only arm (head weight 200)
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
R=analysis_out/v017_readout/ranks
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/fig2
python paper/scripts/fig2/fig2_head.py \
  --pa-query "$R/EFFGATE3/EFFGATE3_ranks.csv" \
  --clf-query "$R/HEAD200_Hmz/HEAD200_Hmz_ranks.csv" \
  --clf-alt "$R/HEAD200_H/HEAD200_H_ranks.csv" \
  --pa-cell "$R/EFFGATE3_pc/EFFGATE3_pc_ranks.csv" \
  --clf-cell "$R/HEAD200_Hpc/HEAD200_Hpc_ranks.csv" \
  --vizstyle-root "$DATA" --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT"
