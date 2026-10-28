#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 8 --mem 48G -t 01:00:00 -J fig4spold
#SBATCH -o $REPO/paper/slurm/logs/fig4spold_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Score sci-Plex3 against the PRE-REFILL bank (same latents, symlinked) so the zero-shot arm can be
# compared old fill vs new fill. Data-only; nothing here feeds a main panel.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} CUDA_VISIBLE_DEVICES=""
M=v0.1.0
SCI=$EXPERT_ROOT/data/inference/sciplex
ANN=$PAPER_FIGS/fig4/data/v0.1.0/sciplex_annot   # fig4.sbatch STEP=sciplex_annot
python paper/scripts/fig4/fig4_sciplex_pb.py --step score \
  --encode-dir $SCI/results/fig5_$M/pseudobulk_mtmask_oldfill \
  --out-dir $PAPER_FIGS/fig4/data/$M/sciplex_pb_mtmask_oldfill \
  --candidates $ANN/candidates.npz 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
