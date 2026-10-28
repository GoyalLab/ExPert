#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 2 --mem=32G -t 00:25:00 -J fig4bank
#SBATCH -o $REPO/paper/slurm/logs/fig4bank_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Is the zero-shot tail of the bank predict() wrote the saved f(missing) fill, or still the random init?
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} CUDA_VISIBLE_DEVICES=""
M=${MODEL:-ExPert_v0.1.0}
SP=$PWD/analysis
python -u $SP/_fig4_newfill_audit.py \
  --model-pt $EXPERT_ROOT/models/$M/run_0/config/ExPert_v0/model/model.pt \
  --bank $PAPER_FIGS/fig4/data/pb/$M/carpoolseq/bank.npz 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
