#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 2G --time 00:10:00
#SBATCH --job-name figS9
#SBATCH --output paper/slurm/logs/figS9_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S9 (prototype bridges between drug classes; related to Fig. 4). STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS9/data/   (figS9_data.py; reads the bridge run of fig4.sbatch STEP=pb_bridges --
#           fig4_atom_bridges.py, kNN graph over the trained prototypes and over each prior, 200 matched permutations)
#   plot -> paper/figures/si/figS9/        (render-only; fine on a login node; draws with fig4_plot as a library)
# Supplementary Note: paper/figures/si/figS9/sn.md. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS9
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS9_data.py --bridges-dir $PAPER_FIGS/fig4/data/v0.1.0/tahoe_pb --prefix atom_structure \
    --min-pairs 50 --min-fold 2 --n-strip 24 --prior-fold 2 --main PDPK1 RALGDS --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS9_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
