#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 32G --time 03:00:00
#SBATCH --job-name fig6rna
#SBATCH --output paper/slurm/logs/fig6rna_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6 RNA-only controls (paper/scripts/fig6/fig6_rna.py): expression-only versions of panels B (clustering) and C (donor-set
# contrast) + PyDESeq2 DEG counts with donor-label permutation. Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
W=$EXPERT_ROOT/data/inference/metabolic_weinberg
python paper/scripts/fig6/fig6_rna.py --query $W/converted_data/s_pbmc_collab.h5ad --out-dir $PAPER_FIGS/fig6/data/rna \
  --modules $PAPER_FIGS/fig4/data/leigh_pmd_repro_2026-10-15/leigh_v010/config/modules.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
