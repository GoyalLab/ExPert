#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 12G --time 00:45:00
#SBATCH --job-name fig5pred
#SBATCH --output paper/slurm/logs/fig5pred_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5 readout via the current predict() (fig5_predict.py): one pseudobulk per donor x cell type, deflate readout,
# no hubness. Chain:
#   E=$(sbatch --parsable --export=ALL,STEP=encode paper/slurm/fig5_predict.sbatch)       # ref run MaxRSS 7.6 GB
#   S=$(sbatch --parsable --array 0-1 -d afterok:$E --export=ALL,STEP=score --mem 8G paper/slurm/fig5_predict.sbatch)
#   sbatch --array 0-1 -d afterok:$S --export=ALL,STEP=ora --mem 8G paper/slurm/fig5_predict.sbatch   # top-k ORA
#   sbatch --array 0-1 --export=ALL,STEP=unbiased --mem 8G paper/slurm/fig5_predict.sbatch   # Fig. 1d libraries
#   soup: R=$(sbatch --parsable --mem 40G --time 02:00:00 --export=ALL,STEP=ctrlref,MODEL=v0.1.0_soup ...) then
#   encode / score / ora / unbiased with MODEL=v0.1.0_soup, encode -d afterok:$R
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
# MODEL=v0.1.0 (default) | v0.1.0_soup (uniform soup of 5 checkpoints, production/ExPert_v0.1.0_soup_u5)
MODEL=${MODEL:-v0.1.0}
HUB=$PAPER_FIGS/fig5/data/hubness
case $MODEL in
  v0.1.0)      MDIR=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
               BANK=$ALD/results/readout_v0.1.0/fullbank/bank.npz
               POOLS=(--pools-dir $PAPER_FIGS/fig5/data/variants/v0.1.0__mito_mt) ;;
  # the production model after the cos^4 zero-shot refit (2026-10-23): same weights, refit prior_missing_T; its bank
  # is the one STEP=encode writes from the model (the readout_v0.1.0 fullbank predates the refit). Figure S13.
  v0.1.0_cos4) MDIR=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
               BANK=$PAPER_FIGS/fig5/data/predict/v0.1.0_cos4__mito_mt/bank.npz   # written by STEP=encode
               POOLS=(--pools-dir $PAPER_FIGS/fig5/data/variants/v0.1.0__mito_mt) ;;
  v0.1.0_soup) MDIR=$EXPERT_ROOT/models/production/ExPert_v0.1.0_soup_u5/run_0/config/ExPert_v0/model
               BANK=$PAPER_FIGS/fig5/data/predict/v0.1.0_soup__mito_mt/bank.npz   # written by STEP=encode
               POOLS=() ;;
esac
OUT=$PAPER_FIGS/fig5/data/predict/${MODEL}__mito_mt
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
if [[ $STEP == ctrlref ]]; then      # readout calibration pools: the model's own 10-cell TRAINING control pools
  python paper/scripts/fig5/fig5_hubness.py --stage ctrlref \
    --ctrl-h5ad $DL_ROOT/data/store_out/1e6bafc04836e06a/database_ctrl.h5ad \
    --model-dir $MDIR --out $HUB/${MODEL}_ctrl_pools_ds.npz --n-cores ${SLURM_CPUS_PER_TASK:-8}
fi
if [[ $STEP == encode ]]; then
  python paper/scripts/fig5/fig5_predict.py --stage encode --out-dir $OUT --model-dir $MDIR \
    --ctrl-pools $HUB/${MODEL%_cos4}_ctrl_pools_ds.npz \
    --query $ALD/data/readout/alcohol_readout.h5ad --mask mito_mt "$@"
fi
if [[ $STEP == unbiased ]]; then      # --array 0-1; Fig. 1d libraries
  python paper/scripts/fig5/fig5_predict.py --stage unbiased --task ${SLURM_ARRAY_TASK_ID:-0} --out-dir $OUT \
    --lib-cache $PAPER_FIGS/fig1/data/cache "$@"
fi
if [[ $STEP == score || $STEP == ora ]]; then
  python paper/scripts/fig5/fig5_predict.py --stage $STEP --task ${SLURM_ARRAY_TASK_ID:-0} --out-dir $OUT \
    --bank-npz $BANK "${POOLS[@]}" \
    --modules $ALD/readout_panels.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
fi
