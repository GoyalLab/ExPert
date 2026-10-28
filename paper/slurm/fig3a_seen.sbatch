#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 160G --time 03:00:00
#SBATCH --job-name fig3a_seen
#SBATCH --array 0-11
#SBATCH --output paper/slurm/logs/fig3a_seen_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 3a "all seen perturbations": per fold model, the TRAINED perturbations of the fold's ID test
# and OOD test splits (capped at 25 cells per dataset x perturbation, the zero-shot cap), scored on
# the same full-vocabulary table as the zero-shot rows (needs fig3a_folds.sbatch's zero-shot caches).
# Measured: 65-125 min per fold; MaxRSS ~100 GB typical, 145 GB on fold 10 (all 4 donors x 48 h held out).
# genomics queues 128G+ slowly: genhimem (account <project account>, qos normal) started all tasks at once.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
F=${SLURM_ARRAY_TASK_ID:-4}
FOLDS=$EXPERT_ROOT/models/folds
DB=$DL_ROOT/data/store_out/1e6bafc04836e06a
SPL=$DB/splits/full/fold_$F
CAP=analysis_out/capped_splits/seen/fold_$F
QC=analysis_out/qcache/folds
MODEL_F=$FOLDS/ExPert_fold$F/run_0/config/ExPert_v0/model/model.pt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$CAP" "$QC" $PAPER_FIGS/fig3/data/folds
[[ -f $CAP/id_test_cap25.txt.gz && -f $CAP/ood_test_cap25.txt.gz ]] || \
  python paper/scripts/fig3/fig3a_seen.py cap --adata $DB/database_perturbed.h5ad --fold-dir "$SPL" \
    --splits id_test ood_test --cap 25 --out-dir "$CAP"
for S in id_test ood_test; do
  [[ -f $QC/ExPert_fold${F}_${S}.npz ]] || python paper/analysis/readout/cache_queries.py --model "$MODEL_F" \
    --adata $DB/database_perturbed.h5ad --fold-dir "$CAP" --split ${S}_cap25.txt.gz --min-cells 5 \
    --no-atoms --out "$QC/ExPert_fold${F}_${S}.npz"
done
python paper/scripts/fig3/fig3a_seen.py score --model-pt "$MODEL_F" --zs-cache "$QC/ExPert_fold${F}_zs.npz" \
  --adata $DB/database_perturbed.h5ad --fold-dir "$CAP" \
  --id-cache "$QC/ExPert_fold${F}_id_test.npz" --ood-cache "$QC/ExPert_fold${F}_ood_test.npz" \
  --id-split id_test_cap25.txt.gz --ood-split ood_test_cap25.txt.gz --reps 20 \
  --out $PAPER_FIGS/fig3/data/folds/fig3a_seen_fold${F}.json
