#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 8G --time 00:30:00
#SBATCH --job-name tableS5
#SBATCH --output paper/slurm/logs/table_s5_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S5 (the 200 and ctx7 benchmark training databases and their 12 held-out-context folds) ->
# paper/figures/si/tables/Table_S5.xlsx + one CSV per part. Counts from the databases' obs and the folds' index files.
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
DB=$DL_ROOT/data/store_out/1e6bafc04836e06a
python paper/scripts/tables/table_s5.py \
  --db "200=$DB/database_perturbed_200.h5ad=$DB/splits/200=nested subset of the full database: every dataset contributes >= 30% of the 200 perturbations, greedy by cross-dataset breadth, random ties and fill (seed 42)" \
       "ctx7=$DB/database_perturbed_ctx7.h5ad=$DB/splits/ctx7=every perturbation measured in >= 7 cell contexts" \
  --ctx7-subset-json $DB/database_perturbed_ctx7.h5ad.subset.json \
  --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
