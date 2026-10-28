#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 8 --mem=64G -t 01:00:00 -J fpsgl_ogw
#SBATCH -a 0-3
#SBATCH -o $REPO/paper/slurm/logs/fpsgl_ogw_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Map the four combinatorial screens' SINGLE perturbations against our own full K562 GWPS dictionary (8,347
# fingerprints) -- Supplementary Fig. S4A (single CRISPRa recovery; CRISPRi/Cas13d singles as the aux reference).
# Same script and settings as fp_doubles_ourgwps.sbatch, query kind = single. Doubles peaked at 22-46 GB in 4-5 min
# (sacct 4564677); singles have up to 1.4x the cells, so 64G. CHECK sacct MaxRSS after.
set -o pipefail
RS=$EXPERT_PROJECT/ExPert-benchmarks/.conda-envs/a122f0948aa175884d623126da81d4fb_/bin/Rscript
cd $REPO
EXP=$COLLAB_DATA/fingerprint_bench/data/fp_doubles_export_expertgenes
DICT=$EXPERT_PROJECT/benchmarks/results/fingerprint/production_v010/fold_0/per_dataset_panel/ReplogleWeissman2022_K562_gwps/dict.rds
OUT=$PAPER_FIGS/fig4/data/fingerprint_singles_ourgwps/results
SCREENS=(GSE146194_CRISPRi CaRPool_Cas13d Norman2019_CRISPRa ProPerseq_CRISPRa)
S=${SCREENS[${SLURM_ARRAY_TASK_ID:-0}]}
mkdir -p $OUT/$S
echo "[task] $S  dict=$DICT"
$RS $PWD/paper/analysis/fingerprint/fp_doubles_ourgwps.R $EXP/$S $OUT/$S $DICT ${SLURM_CPUS_PER_TASK:-8} none single
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
