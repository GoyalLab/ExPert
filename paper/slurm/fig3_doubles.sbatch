#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 4 --mem=24G -t 00:30:00 -J fig3dbl
#SBATCH -o $REPO/paper/slurm/logs/fig3dbl_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 3 H/I (formerly Fig. 4 L/M): score ExPert / our full-GWPS fingerprinting / centroids / random on the
# four doubles screens, same candidates and same doubles; per screen AND pooled by modality -> $PAPER_FIGS/fig3/data/doubles/.
# The same with candidates = genes in each screen's own doubles is Supplementary Fig. S5 (figS5.sbatch).
# Inputs (encodings, fingerprint fits) stay in fig4/data.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} CUDA_VISIBLE_DEVICES=""
M=${MODEL:-ExPert_v0.1.0}
D4=$PAPER_FIGS/fig4/data
GW=$D4/fingerprint_doubles_gwps
REF=$EXPERT_ROOT/data/inference/sciplex/results/fig5_v0.1.0/crispr_signature_ref.npz
python -u paper/scripts/fig3/fig3_doubles.py \
  --encode-root $D4/pb/$M \
  --fp-root $D4/fingerprint_doubles_ourgwps/results \
  --pair-ranks $GW/results/gwps_ranks.csv --centroid-ref $REF \
  --out-dir $PAPER_FIGS/fig3/data/doubles \
  --screens "gse146194:GSE146194_CRISPRi:loss:loss,carpoolseq:CaRPool_Cas13d:loss:loss,norman2019:Norman2019_CRISPRa:gain:gain,properseq:ProPerseq_CRISPRa:gain:gain" 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
