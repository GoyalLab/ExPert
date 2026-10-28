#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 24G --time 01:00:00
#SBATCH --job-name fig5chain
#SBATCH --output paper/slurm/logs/fig5chain_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# ALD readout chain via predict() (paper/scripts/fig5/fig5_readout_chain.py). Partition: whichever of long / genhimem /
# genomics has the shortest queue (<buy-in account> jobs need Account=<buy-in account> + QOS buyin; <project account> + QOS normal elsewhere).
#   E=$(sbatch --parsable --array 0-6 --export=ALL,STEP=encode paper/slurm/fig5_readout_chain.sbatch)
#   S=$(sbatch --parsable --array 0-15 -d afterok:$E --export=ALL,STEP=score --mem 32G paper/slurm/fig5_readout_chain.sbatch)
#   sbatch -d afterok:$S --export=ALL,STEP=summarize --mem 32G paper/slurm/fig5_readout_chain.sbatch
# (score task 14/15 = the pb_ct_normal arm, which needs no encode.) Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
F=$PAPER_FIGS/fig5/data
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python paper/scripts/fig5/fig5_readout_chain.py --stage $STEP --task ${SLURM_ARRAY_TASK_ID:-0} --out-dir $F/readout_chain \
  --model-dir $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model \
  --ctrl-pools $F/hubness/v0.1.0_ctrl_pools_ds.npz --query $ALD/data/readout/alcohol_readout.h5ad \
  --predict-npz $F/predict/v0.1.0__mito_mt/predict_groups.npz \
  --bank-npz $ALD/results/readout_v0.1.0/fullbank/bank.npz --modules $ALD/readout_panels.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
