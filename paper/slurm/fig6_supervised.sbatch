#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 24G --time 00:45:00
#SBATCH --job-name fig6sup
#SBATCH --output paper/slurm/logs/fig6sup_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6 supervised PMD vs non-PMD on held-out donors, ExPert latent vs expression (paper/scripts/fig6/fig6_supervised.py).
# Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
W=$EXPERT_ROOT/data/inference/metabolic_weinberg
python paper/scripts/fig6/fig6_supervised.py --query $W/converted_data/s_pbmc_collab.h5ad \
  --z-dir $PAPER_FIGS/fig6/data/pmd_mtmt --out $PAPER_FIGS/fig6/data/supervised/supervised_auc.csv "$@"
