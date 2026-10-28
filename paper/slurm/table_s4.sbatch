#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 16G --time 00:30:00
#SBATCH --job-name tableS4
#SBATCH --output paper/slurm/logs/table_s4_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S4 (hyperparameters of the shipped ExPert v0.1.0, active settings only) ->
# paper/figures/si/tables/Table_S4.xlsx + .csv. Read from the run's hparams.yaml, model.pt and tfevents; the script
# asserts that every listed setting was live and every omitted loss was 0. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
R=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0
python paper/scripts/tables/table_s4.py --hparams $R/hparams.yaml --model $R/model/model.pt \
  --events $R/events.out.tfevents.1789581020.qgpu3009.2856941.0 --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
