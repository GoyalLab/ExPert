#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 4 --mem=24G -t 00:30:00 -J fig4dgpf
#SBATCH -o $REPO/paper/slurm/logs/fig4dgpf_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Replay the degron readout offline against the OLD and NEW banks to prove which fill the run used.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} CUDA_VISIBLE_DEVICES=""
M=v0.1.0
G=$PAPER_FIGS/fig4/data/$M/degron_pb
SCI=$EXPERT_ROOT/data/inference/sciplex/results/fig5_$M/pseudobulk_mtmask
PYTHONPATH=. python -u paper/analysis/_fig4_degron_fill_proof.py \
  --delta $G/degron_pb_delta_clone_nm1_did_mt.npy \
  --activity $G/degron_pb_activity_clone_nm1_did_mt.csv \
  --ranks-new $G/degron_pb_ranks_clone_nm1_did_mt.csv \
  --ranks-old ${G}_oldfill/degron_pb_ranks_clone_nm1_did_mt.csv \
  --banks new=$SCI/bank.npz old=$SCI/bank_oldfill.npz \
  --model-pt $EXPERT_ROOT/models/ExPert_$M/run_0/config/ExPert_v0/model/model.pt 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
