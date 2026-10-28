#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 16G --time 01:00:00
#SBATCH --job-name fig5hub
#SBATCH --output paper/slurm/logs/fig5hub_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Hubness correction (inverted softmax) on the Fig. 5 ALD readout (paper/scripts/fig5/fig5_hubness.py).
# STEP=split    cap v0.1.0's production training split to <=10 cells per dataset x perturbation
# STEP=cache    encode those with v0.1.0 (paper/analysis/readout/cache_queries.py): the fixed training-query reference
#               (--mem 150G --time 04:00:00; the H8 id_train_cap10 cache took 1h44 at 117 GB)
# STEP=ctrlref  control-only reference pools from the training control database (--mem 40G)
# STEP=score    all corrections x betas vs plain cosine
# Chain:
#   S=$(sbatch --parsable --export=ALL,STEP=split paper/slurm/fig5_hubness.sbatch)
#   C=$(sbatch --parsable -d afterok:$S --mem 150G --time 04:00:00 --export=ALL,STEP=cache paper/slurm/fig5_hubness.sbatch)
#   R=$(sbatch --parsable --mem 40G --export=ALL,STEP=ctrlref paper/slurm/fig5_hubness.sbatch)
#   sbatch -d afterok:$C:$R --mem 24G --export=ALL,STEP=score paper/slurm/fig5_hubness.sbatch
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
DB=$DL_ROOT/data/store_out/1e6bafc04836e06a
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
MDIR=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
OUT=$PAPER_FIGS/fig5/data/hubness
SPLIT=analysis_out/capped_splits/production_v010
mkdir -p $OUT $SPLIT
if [[ $STEP == split ]]; then
  python paper/analysis/readout/_tmp/make_cap_split.py $DB/database_perturbed.h5ad $DB/splits/production_v010/fold_0/id_train.txt.gz \
    $SPLIT/id_train_cap10.txt.gz 10
fi
if [[ $STEP == cache ]]; then
  python paper/analysis/readout/cache_queries.py --model $MDIR --adata $DB/database_perturbed.h5ad --fold-dir $SPLIT \
    --split id_train_cap10.txt.gz --min-cells 5 --out $OUT/v0.1.0_prod_idtrain_cap10.npz
fi
if [[ $STEP == ctrlref ]]; then
  python paper/scripts/fig5/fig5_hubness.py --stage ctrlref --ctrl-h5ad $DB/database_ctrl.h5ad --model-dir $MDIR \
    --out $OUT/v0.1.0_ctrl_pools.npz --n-cores ${SLURM_CPUS_PER_TASK:-8}
fi
if [[ $STEP == score ]]; then
  python paper/scripts/fig5/fig5_hubness.py --stage score --variant-dir $PAPER_FIGS/fig5/data/variants/v0.1.0__mito_mt \
    --bank $ALD/results/readout_v0.1.0/fullbank/bank.npz --train-cache $OUT/v0.1.0_prod_idtrain_cap10.npz \
    --ctrl-cache $OUT/v0.1.0_ctrl_pools.npz --modules $ALD/readout_panels.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
    --deg $ALD/results/readout_deg/deg.csv.gz --out-dir $OUT "$@"
fi
