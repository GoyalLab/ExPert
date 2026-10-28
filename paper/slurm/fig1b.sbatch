#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 40G --time 01:00:00
#SBATCH --job-name fig1b
#SBATCH --output paper/slurm/logs/fig1b_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 1b training-corpus panel. STEP=data|plot|all. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-all}
B=$DL_ROOT/data/store_out/1e6bafc04836e06a
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/fig1
if [[ $STEP == data || $STEP == all ]]; then
python paper/scripts/fig1/fig1b_data.py \
  --recipe "$B/recipe.json" \
  --datasheet "$DATA/resources/datasets/meta/training.csv" \
  --store-obs-dir $DL_ROOT/perturb-seq/store/27dab279946b4ea6/obs \
  --db "$B/database_perturbed.h5ad" \
  --train-idx "$B/splits/production_v010/fold_0/id_train.txt.gz" \
  --out-dir "$EXPERT_ROOT/$OUT/data" --pull-q 0.112 --gamma 16
fi
if [[ $STEP == plot || $STEP == all ]]; then
python paper/scripts/fig1/fig1b_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" \
  --vizstyle-root "$DATA" ${PLOT_ARGS}
fi
