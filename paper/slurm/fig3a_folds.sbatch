#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 28G --time 01:30:00
#SBATCH --job-name fig3a_folds
#SBATCH --array 0-11
#SBATCH --output paper/slurm/logs/fig3a_folds_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 3a across folds: the zero-shot readout of every fold model (Fig. 2's ExPert fold models), so
# each context gets a distribution, and the fold that held its cell line / donors out is marked.
# Per task: (1) encode the zero-shot DB with the fold model, (2) score its SHIPPED fill
# (prior_missing_T) on the full vocabulary, per context, + 20 row-shuffled twins. Same scorer as
# STEP=zs in fig3.sbatch; the fill recipe rows are verified against the reference cache.
# Measured: ~15 min per fold, MaxRSS up to 23 GB (fold 4 cache alone 17 GB).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
F=${SLURM_ARRAY_TASK_ID:-4}
FOLDS=$EXPERT_ROOT/models/folds
FIG=paper/figures/fig3                        # not OUT: paths.sh sets its own OUT
CTX="$EXPERT_ROOT/$FIG/data/fig3a_contexts.tsv"            # written by fig3.sbatch STEP=zs
source paper/analysis/readout/prior_screen/paths.sh   # REPO PY S PATHS PANEL ZS_ADATA CAPTURE SPLIT_META DB
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
MODEL_F=$FOLDS/ExPert_fold$F/run_0/config/ExPert_v0/model/model.pt
CACHE=$REPO/analysis_out/qcache/folds/ExPert_fold${F}_zs.npz
mkdir -p "$(dirname "$CACHE")" "$EXPERT_ROOT/$FIG/data/folds"
[[ -f $CACHE ]] || $PY paper/analysis/readout/cache_zs_queries.py --model "$MODEL_F" \
  --adata $DB/database_perturbed_zs.h5ad --text-npz $PANEL --cap 25 --min-cells 5 --out "$CACHE"
$PY $S panels $PATHS --zs-cache "$CACHE" --zs-adata $ZS_ADATA --capture $CAPTURE \
  --capture-rows $REPO/analysis_out/qcache/EFFGATE3_zs.npz --split-meta $SPLIT_META --fold $F \
  --tag ExPert_fold$F --pred-tag ExPert_fold$F --shipped-fill --model-pt "$MODEL_F" \
  --zs-panel full --reps 20 --group-map "$CTX" --shipped-json "$EXPERT_ROOT/$FIG/data/folds/fig3a_fold${F}.json"
