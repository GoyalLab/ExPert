#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 2G --time 00:10:00
#SBATCH --job-name tableS1
#SBATCH --output paper/slurm/logs/table_s1_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S1 (training corpus) -> paper/figures/si/tables/Table_S1.xlsx + one CSV per sheet, plus
# store_summary.json / training_db_summary.json (condensed corpus summaries). Paths live here, never in the .py.
# Reads Fig. 1's corpus table ($PAPER_FIGS/fig1/data/fig1c_*) and Fig. S1's store counts ($PAPER_FIGS/si/figS1/data: effect-size summary,
# DepMap class per perturbation), so run fig1.sbatch's and figS1.sbatch's data steps first. Fast; fine on a login
# node (bash paper/slurm/table_s1.sbatch).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
B=$DL_ROOT/data/store_out/1e6bafc04836e06a
DATA=$EXPERT_DATA_REPO
F1=$PAPER_FIGS/fig1/data
S1=$PAPER_FIGS/si/figS1/data
python paper/scripts/tables/table_s1.py --fig1-data $F1 --data-dir $S1 \
  --datasheet $DATA/resources/datasets/meta/training.csv --cache-dir $F1/cache \
  --essentiality $S1/figS1aux_essential_cells.csv --ctrl-db $B/database_ctrl.h5ad --out-dir paper/figures/si/tables \
  --data-sources $PAPER_FIGS/si/tables/data/data_sources.csv
# at-a-glance JSON summaries of the store and the training DB, next to the tables
python paper/scripts/tables/si_corpus_summary.py --store $DL_ROOT/perturb-seq/store/27dab279946b4ea6 \
  --db $B --canon-logs $DATA/logs/phase2 --fig1-data $F1 --figS1-data $S1 --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
