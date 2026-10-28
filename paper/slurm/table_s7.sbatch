#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:10:00
#SBATCH --job-name tableS7
#SBATCH --output paper/slurm/logs/table_s7_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Table S7 (the prior screen, related to Fig. 3) -> paper/figures/si/tables/Table_S7.xlsx + one CSV per part.
# Reads the deployment-panel prior screen under the cos^4 fill (analysis_out/prior_screen/H4_full_p4), the trios scored
# separately on it (paper/analysis/readout/prior_screen/panels_p4.sbatch log), through its own loader and Fig. 3's
# single-prior table (fig3.sbatch STEP=plot writes fig3f_source.csv). Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
python paper/scripts/tables/table_s7.py --singles $PAPER_FIGS/fig3/data/fig3f_source.csv \
  --prior-screen-root paper/analysis/readout/prior_screen --screen-dir analysis_out/prior_screen/H4_full_p4 \
  --trio-log analysis_out/prior_screen/logs/panels_p4_5069896.out \
  --out-dir paper/figures/si/tables
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
