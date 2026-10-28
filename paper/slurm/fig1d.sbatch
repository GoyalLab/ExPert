#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 48G --time 02:00:00
#SBATCH --job-name fig1d
#SBATCH --output paper/slurm/logs/fig1d_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 1d atom geometry + prior agreement. STEP=data|tree|plot|all. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-all}
M=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
P=resources/embeddings/priors
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/fig1
if [[ $STEP == data || $STEP == all ]]; then
python paper/scripts/fig1/fig1d_data.py \
  --model-pt "$M/model.pt" \
  --perts "$EXPERT_ROOT/$OUT/data/fig1c_perts.parquet" \
  --prior "text=$M/gene_embeddings.npz" \
  --prior "STRING=$P/string.npz" \
  --prior "STRING exp.=$P/string_exp.npz" \
  --prior "STRING coexp.=$P/string_coexp.npz" \
  --prior "DepMap=$P/depmap_dstd.npz" \
  --prior "L1000=$P/l1000.npz" \
  --prior "morphology=$P/morph_crispr.npz" \
  --prior "ProteinTalks=$P/proteintalks_global.npz" \
  --prior "genomic position=$P/gff_pos.npz" \
  --train-prior text --train-prior STRING --train-prior DepMap \
  --cache-dir "$EXPERT_ROOT/$OUT/data/cache" --out-dir "$EXPERT_ROOT/$OUT/data"
fi
if [[ $STEP == tree || $STEP == all ]]; then
python paper/scripts/fig1/fig1d_tree.py --model-pt "$M/model.pt" \
  --db $DL_ROOT/data/store_out/1e6bafc04836e06a/database_perturbed.h5ad \
  --cache-dir "$EXPERT_ROOT/$OUT/data/cache" --out-dir "$EXPERT_ROOT/$OUT/data"
fi
if [[ $STEP == annot || $STEP == all ]]; then
# per-prototype neighbourhood-enrichment annotation (replaces cluster-then-name for the colours)
python paper/scripts/fig1/fig1d_annotate.py --model-pt "$M/model.pt" \
  --cache-dir "$EXPERT_ROOT/$OUT/data/cache" --out-dir "$EXPERT_ROOT/$OUT/data"
fi
if [[ $STEP == plot || $STEP == all ]]; then
python paper/scripts/fig1/fig1d_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" \
  --vizstyle-root "$DATA" ${PLOT_ARGS}
fi
