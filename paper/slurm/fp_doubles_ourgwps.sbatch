#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION --qos=normal -N 1 --ntasks 1 --cpus-per-task 8 --mem=220G -t 24:00:00 -J fpourgwps
#SBATCH -a 0-3
#SBATCH -o $REPO/paper/slurm/logs/fpourgwps_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Map the four combinatorial screens' doubles against OUR OWN full K562 GWPS dictionary (8,347 fingerprints).
# Identical to the collaborator's fp_doubles_gwps.R except the dictionary is read from disk. Their run used
# 8 cores / 180G for the 4,098-fingerprint published dictionary; ours doubles the fingerprint count, which
# adds ~3.5GB (newLambdas 2.8->5.7GB, store_cor 0.13->0.56GB), so 220G. CHECK sacct MaxRSS after.
set -o pipefail
RS=$EXPERT_PROJECT/ExPert-benchmarks/.conda-envs/a122f0948aa175884d623126da81d4fb_/bin/Rscript
cd $REPO
EXP=$COLLAB_DATA/fingerprint_bench/data/fp_doubles_export_expertgenes
DICT=$EXPERT_PROJECT/benchmarks/results/fingerprint/production_v010/fold_0/per_dataset_panel/ReplogleWeissman2022_K562_gwps/dict.rds
OUT=$PAPER_FIGS/fig4/data/fingerprint_doubles_ourgwps/results
SCREENS=(GSE146194_CRISPRi CaRPool_Cas13d Norman2019_CRISPRa ProPerseq_CRISPRa)
S=${SCREENS[${SLURM_ARRAY_TASK_ID:-0}]}
mkdir -p $OUT/$S
echo "[task] $S  dict=$DICT"
$RS $PWD/paper/analysis/fingerprint/fp_doubles_ourgwps.R $EXP/$S $OUT/$S $DICT ${SLURM_CPUS_PER_TASK:-8} none
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
