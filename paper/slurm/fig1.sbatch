#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 96G --time 08:00:00
#SBATCH --job-name fig1
#SBATCH --output paper/slurm/logs/fig1_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 1 (main figure: panels C-E). Supplementary Figs. S1 / S2 and Table S1 have their own drivers
# (figS1.sbatch, figS3.sbatch); both read Fig. 1's corpus table, so run this data step first.
#
#   STEP=all|data|plot data = compute plotting data into $PAPER_FIGS/fig1/data/
#                      plot = render from data/ only (fast; also fine on a login node)
#
#   sbatch paper/slurm/fig1.sbatch                         # everything
#   STEP=plot sbatch paper/slurm/fig1.sbatch               # re-render all panels from data/
#
# Outputs: panels -> paper/figures/fig1/, plotting data (figure-named CSVs) -> $PAPER_FIGS/fig1/data/.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-all}
want() { [[ $STEP == all || $STEP == "$1" ]]; }

if want data; then
  STEP=data bash paper/slurm/fig1b.sbatch
  for s in data tree annot umap3d; do STEP=$s bash paper/slurm/fig1d.sbatch; done
fi
if want plot; then
  STEP=plot bash paper/slurm/fig1b.sbatch
  STEP=plot bash paper/slurm/fig1d.sbatch
fi
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
