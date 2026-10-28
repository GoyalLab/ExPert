#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 4G --time 00:10:00
#SBATCH --job-name figS5
#SBATCH --output paper/slurm/logs/figS5_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S5 (extended doubles recovery, related to Fig. 3): Fig. 3 H/I's frozen doubles comparison with
# the candidates restricted to the genes in each screen's own doubles. STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS5/data/   (figS5_data.py; reuses fig3_doubles.py's scoring)
#   plot -> paper/figures/si/figS5/ panels (render-only; fine on a login node)
# Inputs (encodings, fingerprint fits) stay in $PAPER_FIGS/fig4/data, as for fig3_doubles.sbatch. Paths live here, never in the .py.
# Data step: 1.9 GB peak, 29 s (sacct 4991614).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} CUDA_VISIBLE_DEVICES=""
STEP=${STEP:-all}
M=${MODEL:-ExPert_v0.1.0}
D4=$PAPER_FIGS/fig4/data
GW=$D4/fingerprint_doubles_gwps
REF=$EXPERT_ROOT/data/inference/sciplex/results/fig5_v0.1.0/crispr_signature_ref.npz
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS5

if [[ $STEP == data || $STEP == all ]]; then
  python -u paper/scripts/si/figS5_data.py --encode-root $D4/pb/$M \
    --fp-root $D4/fingerprint_doubles_ourgwps/results \
    --fp-singles-root $D4/fingerprint_singles_ourgwps/results \
    --pair-ranks $GW/results/gwps_ranks.csv --centroid-ref $REF \
    --screens "gse146194:GSE146194_CRISPRi:loss:loss,carpoolseq:CaRPool_Cas13d:loss:loss,norman2019:Norman2019_CRISPRa:gain:gain,properseq:ProPerseq_CRISPRa:gain:gain" \
    --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS5_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
