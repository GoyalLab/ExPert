#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 34G --time 01:00:00
#SBATCH --job-name figS1
#SBATCH --output paper/slurm/logs/figS1_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S1 (training corpus, related to Fig. 1). Paths live here, never in the .py.
# Table S1 has its own driver (table_s1.sbatch -> paper/figures/si/tables/), reading this figure's data.
# Reads Fig. 1's corpus table ($PAPER_FIGS/fig1/data/fig1c_*), so run fig1.sbatch's data step first.
#
#   STEP=all | data | response | variance | plot
#     data      light panels A, B, E, F (+ the essentiality table Table S1 reads; figS1_data.py)
#     response  panel C, cross-dataset response similarity (figS1c_response_data.py)
#     variance  panel D, variance partition (figS1d_variance_data.py; 27 GB peak, sacct 4951369)
#     plot      panels from data/ only (fast; also fine on a login node)
#
# Outputs: panels -> paper/figures/si/figS1/, plotting data -> $PAPER_FIGS/si/figS1/data/.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
B=$DL_ROOT/data/store_out/1e6bafc04836e06a
STORE=$DL_ROOT/perturb-seq/store/27dab279946b4ea6
DATA=$EXPERT_DATA_REPO
F1=$PAPER_FIGS/fig1/data
OUT=paper/figures/si/figS1
want() { [[ $STEP == all || $STEP == "$1" ]]; }

if want data; then
  python paper/scripts/si/figS1_data.py --store $STORE --db $B --data-repo $DATA --fig1-data $F1 \
    --depmap resources/edges/depmap/CRISPRGeneEffect.csv \
    --gene-sweep $DATA/results/gene_sweep/sweep_holdout.csv --out-dir $EXPERT_ROOT/$OUT/data
fi
if want response; then
  python paper/scripts/si/figS1c_response_data.py --store $STORE --db $B/database_perturbed.h5ad \
    --data-repo $DATA --out-dir $EXPERT_ROOT/$OUT/data
fi
if want variance; then
  python paper/scripts/si/figS1d_variance_data.py --db $B/database_perturbed.h5ad \
    --ctrl-db $B/database_ctrl.h5ad --perts $F1/fig1c_perts.parquet --repo analysis --out-dir $EXPERT_ROOT/$OUT/data
fi
if want plot; then
  python paper/scripts/si/figS1_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
