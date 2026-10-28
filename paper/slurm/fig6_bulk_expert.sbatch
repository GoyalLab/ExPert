#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 16G --time 00:45:00
#SBATCH --job-name fig6bex
#SBATCH --output paper/slurm/logs/fig6bex_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6F ExPert on the sorted-cell bulk (paper/scripts/fig6/fig6_bulk_expert.py). Paths live here.
#   E=$(sbatch --parsable --export=ALL,STEP=encode paper/slurm/fig6_bulk_expert.sbatch)
#   sbatch -d afterok:$E --export=ALL,STEP=score paper/slurm/fig6_bulk_expert.sbatch
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
B=$PAPER_FIGS/fig6/data/mito_bulk
python paper/scripts/fig6/fig6_bulk_expert.py --stage $STEP --out-dir $B/expert \
  --counts $B/salmon/salmon.merged.gene_counts.tsv --meta $B/salmon/261014_sample_metadata.csv \
  --model-dir $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model \
  --bank-npz $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz \
  --modules $PAPER_FIGS/fig4/data/leigh_pmd_repro_2026-10-15/leigh_v010/config/modules.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
