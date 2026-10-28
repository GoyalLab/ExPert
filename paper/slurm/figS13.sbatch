#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 8G --time 00:20:00
#SBATCH --job-name figS13
#SBATCH --output paper/slurm/logs/figS13_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S13 (zero-shot predictions in ALD: ESRP2 and the other splicing regulators; related to Fig. 5).
# B: SRSF6 = the top predicted splicing hit in AH hepatocytes (rank 25 of 19,895); SRSF10 = the trained SR protein
# nearest the ESRP2 fill. STEP=data | plot | all.
#   data -> $PAPER_FIGS/si/figS13/data/   (figS13_data.py; needs the cos^4 full bank from
#           fig5_predict.sbatch MODEL=v0.1.0_cos4 STEP=encode, and the earlier cos-fill bank for comparison)
#   plot -> paper/figures/si/figS13/        (render-only; fine on a login node)
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
STEP=${STEP:-all}
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
DATA=$EXPERT_DATA_REPO
PR=$PAPER_FIGS/fig5/data/predict/v0.1.0_cos4__mito_mt
OUT=paper/figures/si/figS13
export PYTHONPATH=.
if [[ $STEP == data || $STEP == all ]]; then
  # A: the genes the cohort's paper (Nat Commun 2025, doi 10.1038/s41467-025-63251-2) implicates -- its splicing
  # regulators and the mis-spliced ESRP2 targets -- plus ESRP2's nearest prototypes. --genes: the splicing-regulator
  # family of fig5_predict.SPLICE (ESRP / RBFOX / PTBP / MBNL / CELF / QKI / RBM47), still used by B
  python paper/scripts/si/figS13_data.py --predict-dir $PR --bank $PR/bank.npz \
    --old-bank $ALD/results/readout_v0.1.0/fullbank/bank.npz \
    --genes ESRP2 ESRP1 RBM47 RBFOX1 RBFOX2 RBFOX3 PTBP1 PTBP2 PTBP3 MBNL1 MBNL2 MBNL3 CELF1 CELF2 QKI \
    --group "splicing regulators (cohort paper)=ESRP2,RBM47,A1CF,CELF2,SRSF1,PTBP2" \
            "ESRP2 targets (cohort paper)=SLK,TCF7L2,FGFR3,PDGFA,CD44,IL1R1" \
    --direct-of ESRP2 --n-direct 4 \
    --modules $ALD/readout_panels.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
    --top-ora 500 --n-top 15 --fig5e-source $PAPER_FIGS/fig5/data/fig5e_pathways_source.csv \
    --centres SRSF6 SRSF10 --graph-pool 50 --graph-n 10 --graph-k 4 --graph-condition AH --out-dir $EXPERT_ROOT/$OUT/data
fi
if [[ $STEP == plot || $STEP == all ]]; then
  python paper/scripts/si/figS13_plot.py --data-dir $EXPERT_ROOT/$OUT/data --out-dir $OUT --vizstyle-root $DATA
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
