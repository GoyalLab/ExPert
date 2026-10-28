#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 24G --time 01:00:00
#SBATCH --job-name fig5grid
#SBATCH --output paper/slurm/logs/fig5grid_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5 readout grid (paper/scripts/fig5/fig5_readout_grid.py): unit x bank tasks, then summarize.
#   G=$(sbatch --parsable --array 0-7 paper/slurm/fig5_readout_grid.sbatch)
#   sbatch -d afterok:$G --export=ALL,STEP=summarize --mem 16G paper/slurm/fig5_readout_grid.sbatch
# genhimem by default: genomics / long were backed up on 2026-10-17. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
F=$PAPER_FIGS/fig5/data
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
STEP=${STEP:-run}
python paper/scripts/fig5/fig5_readout_grid.py --stage $STEP --task ${SLURM_ARRAY_TASK_ID:-0} --out-dir $F/readout_grid \
  --predict-dir $F/predict/v0.1.0__mito_mt --variant-dir $F/variants/v0.1.0__mito_mt \
  --bank-npz $ALD/results/readout_v0.1.0/fullbank/bank.npz --modules $ALD/readout_panels.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
