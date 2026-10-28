#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 1 --mem 4G --time 00:10:00
#SBATCH --job-name figS12
#SBATCH --output paper/slurm/logs/figS12_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S12 (ExPert scores of the bulk RNA-seq livers vs the snRNA-seq hepatocyte scores; related to
# Fig. 5; A-B), and Fig. 5 split by condition (C-H). STEP=data | plot | split | all.
#   data -> $PAPER_FIGS/si/figS12/data/   (figS12_data.py; reads fig5_bulk_expert.py's score output -- encode via
#           paper/slurm/fig5_bulk_expert.sbatch, score on the trained bank against $PAPER_FIGS/fig5/data/final/fig5_profiles.npz)
#   plot -> paper/figures/si/figS12/        (A-B; render-only; fine on a login node)
#   split -> C-H: figS12c_split_plot.py draws every panel with Fig. 5's own functions on Fig. 5's data (read only;
#           fig5_final.sbatch STEP=data, the donor-pseudobulk DEG table, the FISH frames, nucleus measurements)
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
DATA=$EXPERT_DATA_REPO
OUT=paper/figures/si/figS12
if [[ $STEP == data || $STEP == all ]]; then
  python paper/scripts/si/figS12_data.py --bulk-dir $PAPER_FIGS/fig5/data/bulk_expert --top 200 --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS12_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
if [[ $STEP == split || $STEP == all ]]; then
  ALD=$EXPERT_ROOT/data/inference/alcohol_damage
  F=paper/figures/fig5
  MIC=$EXPERT_ROOT/$F/data/experiment_raw/Microscopy
  export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
  python paper/scripts/si/figS12c_split_plot.py --fig5-data $EXPERT_ROOT/$F/data/final --deg $ALD/results/readout_deg_pbulk/deg.csv.gz \
    --modules $ALD/readout_panels.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
    --fish-frames $MIC/NuclearEnrichmentQuantification/nuc_puncta_frame_summary_master_raw.csv \
    --nuclei $MIC/NuclearMorphometry/NucleiMeasurements.csv --out-dir paper/figures/si/figS12 \
    --src-dir $PAPER_FIGS/si/figS12/data \
    --vizstyle-root $EXPERT_DATA_REPO
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
