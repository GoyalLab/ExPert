#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 32G --time 01:00:00
#SBATCH --job-name fig6grid
#SBATCH --output paper/slurm/logs/fig6grid_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6 configuration grid (paper/scripts/fig6/fig6_grid.py). Paths live here.
#   E=$(sbatch --parsable --array 0-11 --export=ALL,STEP=encode paper/slurm/fig6_grid.sbatch)
#   sbatch -d afterok:$E --mem 48G --time 04:00:00 --export=ALL,STEP=score paper/slurm/fig6_grid.sbatch
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
W=$EXPERT_ROOT/data/inference/metabolic_weinberg
python paper/scripts/fig6/fig6_grid.py --stage $STEP --task ${SLURM_ARRAY_TASK_ID:-0} --out-dir $PAPER_FIGS/fig6/data/grid \
  --query $W/converted_data/s_pbmc_collab.h5ad --models v0.1.0,EFFGATE3 \
  --model-dirs $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model,$EXPERT_ROOT/models/v0.1.7/EFFGATE3/run_0/config/ExPert_v0/model \
  --banks $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz,$W/results/fullbank_EFFGATE3_v2/bank.npz \
  --ctrl-pools $PAPER_FIGS/fig5/data/hubness/v0.1.0_ctrl_pools_ds.npz,analysis_out/qcache/EFFGATE3_fold4_ctrlpools.npz \
  --modules $PAPER_FIGS/fig4/data/leigh_pmd_repro_2026-10-15/leigh_v010/config/modules.tsv \
  --n-cores ${SLURM_CPUS_PER_TASK:-8} "$@"
