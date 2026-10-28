#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 4 --mem=12G -t 00:30:00 -J fig4zsoff
#SBATCH -o $REPO/paper/slurm/logs/fig4zsoff_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Does stripping the f(missing) tail's shared offset / top principal directions lift the true
# zero-shot gene above chance? Scored TAIL-ONLY so the trained block cannot free-ride.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} CUDA_VISIBLE_DEVICES=""
M=${MODEL:-ExPert_v0.1.0}
GW=$PAPER_FIGS/fig4/data/fingerprint_doubles_gwps
python -u paper/analysis/_fig4_zs_offset.py \
  --encode-root $PAPER_FIGS/fig4/data/pb/$M \
  --pair-ranks $GW/results/gwps_ranks.csv \
  --out $PAPER_FIGS/fig4/data/gh_ourgwps_zs/zs_offset_ranks.csv \
  --screens "gse146194:GSE146194_CRISPRi:loss:loss,carpoolseq:CaRPool_Cas13d:loss:loss,norman2019:Norman2019_CRISPRa:gain:gain,properseq:ProPerseq_CRISPRa:gain:gain" 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
