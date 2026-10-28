#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 4 --mem=6G -t 00:20:00 -J fig4zsdb
#SBATCH -o $REPO/paper/slurm/logs/fig4zsdb_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Cross candidate pool (dict 8347 / trained 8964 / full 19895 incl. zero-shot f(text) rows) with query
# set (shared = scorable under the dictionary / extra = dropped by it) on the four doubles screens.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} CUDA_VISIBLE_DEVICES=""
M=${MODEL:-ExPert_v0.1.0}
# PB=pb_p4 / OUTDIR=gh_ourgwps_zs_p4: banks with the cos^4 zero-shot fill (analysis/readout/fill_fix/swap_bank_fill.py)
PB=${PB:-pb}; OUTDIR=${OUTDIR:-gh_ourgwps_zs}
GW=$PAPER_FIGS/fig4/data/fingerprint_doubles_gwps
python -u paper/scripts/fig4/fig4_gh_ourgwps_zs.py \
  --encode-root $PAPER_FIGS/fig4/data/$PB/$M \
  --fp-root $PAPER_FIGS/fig4/data/fingerprint_doubles_ourgwps/results \
  --pair-ranks $GW/results/gwps_ranks.csv \
  --out-dir $PAPER_FIGS/fig4/data/$OUTDIR \
  --screens "gse146194:GSE146194_CRISPRi:loss:loss,carpoolseq:CaRPool_Cas13d:loss:loss,norman2019:Norman2019_CRISPRa:gain:gain,properseq:ProPerseq_CRISPRa:gain:gain" 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
