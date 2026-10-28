#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 2 --mem=24G -t 00:25:00 -J fig4refill
#SBATCH -o $REPO/paper/slurm/logs/fig4refill_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# VERIFY=<freshly re-encoded bank.npz> checks the tail patch against a real predict() encode.
# Otherwise: refresh BANK's zero-shot tail from the refilled checkpoint in place (keeps a .oldfill copy).
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} CUDA_VISIBLE_DEVICES=""
M=${MODEL:-ExPert_v0.1.0}
VA=(); [[ -n ${VERIFY:-} ]] && VA=(--verify "$VERIFY")
python -u paper/analysis/_fig4_bank_refill.py \
  --model-pt $EXPERT_ROOT/models/$M/run_0/config/ExPert_v0/model/model.pt \
  --bank "$BANK" "${VA[@]}" 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
