#!/bin/bash
#SBATCH -A ACCOUNT
#SBATCH -p PARTITION
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem 16G
#SBATCH -J paper_nb
#SBATCH -o analysis_out/paper_notebooks/%x_%A_%a.log
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Execute the per-figure notebooks in place (outputs kept) and compare their panels with the canonical ones.
#   sbatch --array 1-$(grep -c . paper/figures/notebooks.txt) paper/slurm/notebooks.sbatch
#   NB=paper/figures/si/figS14/figS14.ipynb sbatch paper/slurm/notebooks.sbatch        # one notebook
# notebooks.txt: one notebook per line (repo-relative). Each notebook runs from its own folder (data/, inputs/,
# render/ are relative to it). Elapsed / MaxRSS land at the end of the log.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
REPO=$PWD
NB=${NB:-$(sed -n "${SLURM_ARRAY_TASK_ID}p" paper/figures/notebooks.txt)}
DIR=$(dirname "$NB")
echo "notebook $NB"
cd "$DIR"
rm -rf render
/usr/bin/time -v jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 \
  "$(basename "$NB")"
python "$REPO/paper/scripts/common/compare_panels.py" --fig-dir . ${PANELS:+--panels $PANELS}
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
