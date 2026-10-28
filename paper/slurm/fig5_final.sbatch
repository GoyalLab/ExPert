#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 12G --time 00:45:00
#SBATCH --job-name fig5final
#SBATCH --output paper/slurm/logs/fig5final_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5 final (paper/scripts/fig5/fig5_final.py): ONE readout configuration (readout-chain winner) for every panel.
#   STEP=data (UMAP + tables; genomics / genhimem / long, whichever is free) then STEP=plot (light; login is fine),
#   then STEP=experiment (the experimental panels 5G/I/J/K/M; light)
# Paths live here, never in the .py. DEG = donor-level pseudobulk DESeq2 table (fig5_pbulk_de.sbatch); the old
# nucleus-level Wilcoxon table is $ALD/results/readout_deg/deg.csv.gz (DEG=... to use it).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
F=paper/figures/fig5
export PYTHONPATH=. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
if [[ ${STEP:-data} == experiment ]]; then
  # Fig. 5G/K (bulk RNA-seq vs ExPert; needs STEP=data + plot), 5I FISH, 5J nuclei, 5M RT-PCR splice assay. Light.
  RAW=$EXPERT_ROOT/$F/data/experiment_raw
  VIZ=$EXPERT_DATA_REPO
  python paper/scripts/fig5/fig5_bulk.py --bulk-dir $RAW/BulkRNASeqAnalysis/tables \
    --profiles $EXPERT_ROOT/$F/data/final/fig5_profiles.npz --fig-dir $F --src-dir $EXPERT_ROOT/$F/data \
    --modules $ALD/readout_panels.tsv --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
    --d-highlight $EXPERT_ROOT/$F/data/fig5d_volcano_highlight_genes.csv --vizstyle-root $VIZ
  python paper/scripts/fig5/fig5i_fish.py --frames $RAW/Microscopy/NuclearEnrichmentQuantification/nuc_puncta_frame_summary_master_raw.csv \
    --fig-dir $F --src-dir $EXPERT_ROOT/$F/data --vizstyle-root $VIZ
  python paper/scripts/fig5/fig5j_nuclei.py --measurements $RAW/Microscopy/NuclearMorphometry/NucleiMeasurements.csv \
    --fig-dir $F --src-dir $EXPERT_ROOT/$F/data --vizstyle-root $VIZ
  python paper/scripts/fig5/fig5m_psi.py --prism $RAW/Gel/Qty.prism --fig-dir $F --src-dir $EXPERT_ROOT/$F/data \
    --vizstyle-root $VIZ
  exit 0
fi
python paper/scripts/fig5/fig5_final.py --stage ${STEP:-data} --out-dir $EXPERT_ROOT/$F/data/final --fig-dir $F --src-dir $EXPERT_ROOT/$F/data \
  --predict-dir $EXPERT_ROOT/$F/data/predict/v0.1.0__mito_mt --cells-dir $EXPERT_ROOT/$F/data/readout_chain \
  --bank-npz $ALD/results/readout_v0.1.0/fullbank/bank.npz --deg ${DEG:-$ALD/results/readout_deg_pbulk/deg.csv.gz} \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
  --modules $ALD/readout_panels.tsv \
  --vizstyle-root $EXPERT_DATA_REPO "$@"
