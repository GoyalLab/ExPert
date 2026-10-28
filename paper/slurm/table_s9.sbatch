#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 2G --time 00:10:00
#SBATCH --job-name tableS9
#SBATCH --output paper/slurm/logs/table_s9_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S9 (explained Tahoe-100M drug misses; Supplementary Note 4) -> paper/figures/si/tables/Table_S9.xlsx + .csv.
# The curated workbook, checked against Tahoe's target annotation. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
TP=$PAPER_FIGS/fig4/data/v0.1.0/tahoe_pb
python paper/scripts/tables/table_s9.py --source $PAPER_FIGS/si/tables/data/tahoe_misses_source.xlsx \
  --not-counted 'Elimusertib hydrochloride' \
  --annotations $TP/failing_drugs.csv --bridge-nodes $TP/atom_structure_bridge_graph_nodes.csv \
  --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
