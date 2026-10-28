#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 24G --time 01:00:00
#SBATCH --job-name fig6cscan
#SBATCH --output paper/slurm/logs/fig6cscan_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6C scan over encodes x scorings (paper/scripts/fig6/fig6_cscan.py); one task per fig6_grid encode.
#   sbatch --array 0-11 paper/slurm/fig6_cscan.sbatch
# Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
W=$EXPERT_ROOT/data/inference/metabolic_weinberg
python paper/scripts/fig6/fig6_cscan.py --task ${SLURM_ARRAY_TASK_ID:-0} --grid-dir $PAPER_FIGS/fig6/data/grid \
  --out-dir $PAPER_FIGS/fig6/data/cscan \
  --banks $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz,$W/results/fullbank_EFFGATE3_v2/bank.npz \
  --modules $PAPER_FIGS/fig4/data/leigh_pmd_repro_2026-10-15/leigh_v010/config/modules.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
