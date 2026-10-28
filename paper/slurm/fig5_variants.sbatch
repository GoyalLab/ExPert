#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 20G --time 01:00:00
#SBATCH --job-name fig5var
#SBATCH --output paper/slurm/logs/fig5var_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5 variant grid (fig5_variants.py): model x input mask (encode, --array 0-11), x bank (score, --array 0-23),
# then merge. Chain:
#   E=$(sbatch --parsable --array 0-8 --export=ALL,STEP=encode paper/slurm/fig5_variants.sbatch)
#   S=$(sbatch --parsable --array 0-17 -d afterok:$E --export=ALL,STEP=score --mem 12G paper/slurm/fig5_variants.sbatch)
#   sbatch -d afterok:$S --export=ALL,STEP=merge --mem 8G paper/slurm/fig5_variants.sbatch
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
M=$EXPERT_ROOT/models
G16=$DL_ROOT/data/store_out/963044872df9d6a8/models/full/G16/run_0/config/ExPert_v1/model
MODELS=v0.1.0,v0.1.1,v0.1.2,G16    # G16 = 16,092-gene input panel (v0.1.3 recipe, fold_4 split)
MDIRS=$M/ExPert_v0.1.0/run_0/config/ExPert_v0/model,$M/ExPert_v0.1.1/run_0/config/ExPert_v0/model,$M/ExPert_v0.1.2/run_0/config/ExPert_v0/model,$G16
BANKS=$ALD/results/readout_v0.1.0/fullbank/bank.npz,$ALD/results/readout_v0.1.1/fullbank/bank.npz,$ALD/results/readout_v0.1.2_mtfree/fullbank/bank.npz,$ALD/results/readout_G16/fullbank/bank.npz
OUT=$PAPER_FIGS/fig5/data/variants
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python paper/scripts/fig5/fig5_variants.py --stage $STEP --task ${SLURM_ARRAY_TASK_ID:-0} --out-dir $OUT \
  --models $MODELS --model-dirs $MDIRS --banks $BANKS --query $ALD/data/readout/alcohol_readout.h5ad \
  --modules $ALD/readout_panels.tsv --deg $ALD/results/readout_deg/deg.csv.gz \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
  --n-cores ${SLURM_CPUS_PER_TASK:-8} "$@"
