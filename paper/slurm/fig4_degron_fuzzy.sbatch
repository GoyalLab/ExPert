#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 8 --mem 16G -t 00:30:00 -J fig4dgfz
#SBATCH -o $REPO/paper/slurm/logs/fig4dgfz_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# The fuzzy barcode assignment of the degron pool, same arm as clone_nm1 _did_mt, so the
# barcode-stringency comparison in methods.md sits on ONE f(missing) fill.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} CUDA_VISIBLE_DEVICES=""
M=v0.1.0
MDIR=$EXPERT_ROOT/models/ExPert_$M/run_0/config/ExPert_v0/model
INF=$EXPERT_ROOT/data/inference
python paper/scripts/fig4/fig4_degron_pb.py \
  --model-dir "$MDIR" --query $INF/degron/data/degron_labeled.h5ad \
  --out-dir $PAPER_FIGS/fig4/data/$M/degron_pb --clone-col clone_fuzzy --tag _fuzzy_did_mt \
  --condition-col condition --control-label DMSO --did \
  --mask-genes "MT-*,MT1*,MT2A,MT3,MT4" --max-pct-mt 20 \
  --n-cores "${SLURM_CPUS_PER_TASK:-8}" 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
