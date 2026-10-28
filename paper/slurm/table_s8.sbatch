#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 2G --time 00:10:00
#SBATCH --job-name tableS8
#SBATCH --output paper/slurm/logs/table_s8_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S8 (HDAC machinery set of Fig. 4D and the class I corepressor subunits it was calibrated on) ->
# paper/figures/si/tables/Table_S8.xlsx + one CSV per part. Reads the two tables fig4_sciplex_panels.py builds the set
# from (fig4.sbatch --hdac-complexes / --partner-sets). Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
ANN=$PAPER_FIGS/fig4/data/v0.1.0/sciplex_annot   # fig4.sbatch STEP=sciplex_annot
CUR=$PAPER_FIGS/fig4/data/sciplex_annot_curated
python paper/scripts/tables/table_s8.py --hdac-complexes $CUR/hdac_complexes.csv \
  --partner-sets $ANN/corum_core_sets.csv \
  --candidates $ANN/candidates.npz \
  --vendor-targets $PAPER_FIGS/si/tables/data/sciplex_vendor_targets_source.xlsx --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
