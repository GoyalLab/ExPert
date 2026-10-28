#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 8 --mem 48G -t 01:00:00 -J fig4flex
#SBATCH -o $REPO/paper/slurm/logs/fig4flex_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Figure 4E computed here from the FlexPlex cells, instead of re-plotting the collaborator's table.
# KEEP=1 reproduces their `_fpgenes` runs (ExPert inputs restricted to the fingerprinting gene set).
# Encoding is per CELL (readout.group_response / score_prototypes), as the published panel was;
# CHECK=1 diffs latents and scores against the collaborator's published run directories.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} CUDA_VISIBLE_DEVICES=""
FX=$EXPERT_ROOT/data/inference/flexplex
V010=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
EG3=$EXPERT_ROOT/models/v0.1.7/EFFGATE3/run_0/config/ExPert_v0/model
FPDIR=${FPDIR:-pip_matched}          # the arm Fig. 4E compares against
JR=$COLLAB_DATA/fingerprint_bench/results
CA=(); [[ ${CHECK:-1} == 1 ]] && CA=(--check-dirs "ExPert=$JR/fp_hdac_v010_fpgenes;ExPert, no K562 training=$JR/fp_hdac_EFFGATE3_fpgenes")
KA=(); [[ ${KEEP:-0} == 1 ]] && KA=(--keep-genes $FX/fingerprint_flexplex/$FPDIR/features_used.txt)
python -u paper/scripts/fig4/fig4_flexplex.py \
  --query $FX/fp_hdac_query.h5ad \
  --models "ExPert=$V010;ExPert, no K562 training=$EG3" \
  --fp-pip $FX/fingerprint_flexplex/$FPDIR/pip.csv \
  --fp-cor $FX/fingerprint_flexplex/$FPDIR/cor.csv \
  --out-dir ${OUT:-$PAPER_FIGS/fig4/data/v0.1.0/flexplex} \
  "${KA[@]}" "${CA[@]}" "$@" 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
