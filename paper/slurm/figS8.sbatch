#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:20:00
#SBATCH --job-name figS8
#SBATCH --output paper/slurm/logs/figS8_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S8 (drug-target recovery with the zero-shot prototypes added to the candidate bank, as fold over
# random; related to Fig. 4). STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS8/data/   (figS8_data.py; sci-Plex3 ranks of the MT-masked pseudobulk arm under the
#           cos^4 and the earlier cos fill -- fig4.sbatch sciplex steps --, and the Tahoe-100M known-MOA panels on the
#           trained bank and on the full bank under both fills -- fig4.sbatch STEP=pb_panels --bank-rows full)
#   plot -> paper/figures/si/figS8/        (render-only; fine on a login node)
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS8
F4=$PAPER_FIGS/fig4/data/v0.1.0
SCI=$EXPERT_ROOT/data/inference/sciplex
ANN=$PAPER_FIGS/fig4/data/v0.1.0/sciplex_annot   # fig4.sbatch STEP=sciplex_annot
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS8_data.py \
    --sciplex-ranks "cos4=$F4/sciplex_pb_mtmask/ranks_arms.csv" "cos=$F4/sciplex_pb_mtmask_oldfill/ranks_arms.csv" \
    --sciplex-targets $PAPER_FIGS/fig4/data/sciplex3_drug_targets.tsv \
    --sciplex-bank $SCI/results/fig5_v0.1.0/pseudobulk_mtmask/bank.npz \
    --sciplex-matched $ANN/candidates.npz \
    --tahoe-panels "trained=cos4=$F4/tahoe_pb/panels_knownmoa" "padded=cos4=$F4/tahoe_pb/panels_knownmoa_full_new" \
                   "padded=cos=$F4/tahoe_pb/panels_knownmoa_full_old" \
    --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS8_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
