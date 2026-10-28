#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 16G --time 00:15:00
#SBATCH --job-name si_vocab
#SBATCH --output paper/slurm/logs/si_vocab_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# SI summary of the prediction vocabulary -> paper/figures/si/tables/vocabulary_summary.json: trained perturbations,
# rows predict() ranks (trained + zero-shot) in the production model, and the zero-shot perturbations scored in Fig. 3
# (>= 5 cells in a screen; the rule of the zero-shot query cache). Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
DB=$DL_ROOT/data/store_out/1e6bafc04836e06a
python paper/scripts/si/si_vocab_summary.py \
  --model $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model/model.pt \
  --zs-cache analysis_out/qcache/EFFGATE3_zs.npz --zs-adata $DB/database_perturbed_zs.h5ad --min-cells 5 \
  --table-s5 paper/figures/si/tables/Table_S6.csv --out paper/figures/si/tables/vocabulary_summary.json
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
