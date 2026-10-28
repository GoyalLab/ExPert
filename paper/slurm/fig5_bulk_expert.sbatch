#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 12G --time 00:30:00
#SBATCH --job-name fig5bulkex
#SBATCH --output paper/slurm/logs/fig5bulkex_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig 5: score the bulk RNA-seq cohort with ExPert (fig5_bulk_expert.py --stage encode; score runs on login).
# Model / pools / bank as fig5_predict.sbatch (v0.1.0). Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
python paper/scripts/fig5/fig5_bulk_expert.py --stage encode --out-dir $PAPER_FIGS/fig5/data/bulk_expert \
  --bulk-dir $PAPER_FIGS/fig5/data/experiment_raw/BulkRNASeqAnalysis/tables \
  --model-dir $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model \
  --ctrl-pools $PAPER_FIGS/fig5/data/hubness/v0.1.0_ctrl_pools_ds.npz --mask mito_mt "$@"
