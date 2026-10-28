#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 16G --time 01:00:00
#SBATCH --job-name fig5defl
#SBATCH --output paper/slurm/logs/fig5defl_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Deflated-readout settings on the ALD winner configuration (paper/scripts/fig5/fig5_deflate_grid.py): run, then summarize.
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
for S in run summarize; do
python paper/scripts/fig5/fig5_deflate_grid.py --stage $S --out-dir $PAPER_FIGS/fig5/data/deflate_grid \
  --predict-npz $PAPER_FIGS/fig5/data/predict/v0.1.0__mito_mt/predict_groups.npz --bank-npz $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz \
  --modules $EXPERT_ROOT/data/inference/alcohol_damage/readout_panels.tsv --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt
done
