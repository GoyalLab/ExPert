#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:10:00
#SBATCH --job-name tableS10
#SBATCH --output paper/slurm/logs/table_s10_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S10 (pathway enrichment in ALD hepatocytes, related to Fig. 5E/F) -> paper/figures/si/tables/Table_S10.*
# Reads Fig. 5's enrichment tables (fig5_final.sbatch STEP=data) and the pool-level DE. Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
F5=$PAPER_FIGS/fig5/data
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
PYTHONPATH=. python paper/scripts/tables/table_s10.py --fig5-data $F5/final --fig5-figdata $F5 \
  --deg $ALD/results/readout_deg_pbulk/deg.csv.gz --modules $ALD/readout_panels.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
  --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
