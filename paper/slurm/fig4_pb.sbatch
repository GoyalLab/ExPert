#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 4 --mem=14G -t 00:30:00 -J fig4pb
# genomics backs up (148 pending / 739 running, ~20 h waits); genhimem and normal (<project account>) are usually
# free for these small jobs -- override with: sbatch -A <project account> -p genhimem ...
# sizes above are for STEP=encode (measured MaxRSS 8.3-10.6 GB, ~2.5 min); STEP=score peaks ~1.5 GB / <2 min -> submit it with
#   sbatch --export=ALL,STEP=score --cpus-per-task 2 --mem 4G -t 00:20:00 paper/slurm/fig4_pb.sbatch
#SBATCH -a 0-3
#SBATCH -o $REPO/paper/slurm/logs/fig4pb_%x_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 4 through the shipped predict() path on pseudobulks: STEP=encode (one predict per input variant) then
# STEP=score (ExPert deflate/cosine, own-singles ceiling, training centroids). One array task per screen.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} CUDA_VISIBLE_DEVICES=""
STEP=${STEP:?set STEP=encode|score}
M=${MODEL:-ExPert_v0.1.0}
MDIR=$EXPERT_ROOT/models/$M/run_0/config/ExPert_v0/model
CTRL=$PAPER_FIGS/fig5/data/hubness/v0.1.0_ctrl_pools_ds.npz
REF=$EXPERT_ROOT/data/inference/sciplex/results/fig5_v0.1.0/crispr_signature_ref.npz
C=$EXPERT_ROOT/data/inference/combinatorics
OUT=$PAPER_FIGS/fig4/data/pb/$M
SCREENS=(gse146194 norman2019 properseq carpoolseq)
S=${SCREENS[${SLURM_ARRAY_TASK_ID:-0}]}
case $S in
  gse146194)  Q=$C/datasets/GSE146194/data/GSE146194_exp6.h5ad; TAIL=loss
              TRUTH=(--gene-cols gene_A,gene_B --group-col perturbation --strip-suffix '_[0-9]+$' --drop-features '^sg[A-Z0-9]') ;;
  norman2019) Q=$EXPERT_ROOT/data/inference/perturb-seq/data/NormanWeissman2019_filtered.h5ad; TAIL=gain
              TRUTH=(--pert-col perturbation --pert-sep '_') ;;
  properseq)  Q=$C/datasets/properseq/data/proper_filtered.h5ad; TAIL=gain
              TRUTH=(--pert-col perturbation --pert-sep '|') ;;
  # Wessels 2023 CaRPool-seq (GSE213957), THP1 AML: Cas13d knockdown, so loss-tail like GSE146194.
  carpoolseq) Q=$C/datasets/carpoolseq/data/carpoolseq_thp1.h5ad; TAIL=loss
              TRUTH=(--pert-col perturbation --pert-sep '_') ;;
esac
echo "[task] $STEP $M $S"
if [[ $STEP == encode ]]; then
  python paper/scripts/fig4/fig4_pb.py --step encode --screen $S --query $Q --model-dir $MDIR \
    --ctrl-pools $CTRL --out-dir $OUT/$S "${TRUTH[@]}" --alias-map $C/aliases.tsv --n-cores ${SLURM_CPUS_PER_TASK:-8} \
    2>&1 | grep -avE "Warn|warn|^ {10,}"
elif [[ $STEP == score ]]; then
  # Deflation settings default to the shipped predict() values; override per-run, e.g. the Fig. 4
  # combinatorial-optimum  DEFM=5 DEFSTOP=ratio DEFRATIO=0.96
  python -u paper/scripts/fig4/fig4_pb.py --step score --screen $S --encode-dir $OUT/$S \
    --out-dir $OUT/$S --centroid-ref $REF --tail $TAIL \
    --deflate-m ${DEFM:-25} --deflate-rank ${DEFRANK:-3} --deflate-select ${DEFSELECT:-fixed} \
    --deflate-stop ${DEFSTOP:-noise} --deflate-ratio ${DEFRATIO:-0.9} --deflate-damp ${DEFDAMP:-0.0} 2>&1
fi
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
