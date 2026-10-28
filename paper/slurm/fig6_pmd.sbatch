#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 32G --time 02:00:00
#SBATCH --job-name fig6pmd
#SBATCH --output paper/slurm/logs/fig6pmd_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# DATA=<dir> overrides the output dir. Default data/pmd_mtmt = the Fig. 5/6 input mask (encode with
#   --mask MT-*,MT1*,MT2A,MT3,MT4); data/pmd = the collaborator MT-* only mask (same results).
# Fig. 6 v2: PMD PBMC readout on the collaborator protocol (paper/scripts/fig6/fig6_pmd.py). Paths live here.
#   E=$(sbatch --parsable --export=ALL,STEP=encode paper/slurm/fig6_pmd.sbatch)
#   sbatch -d afterok:$E --mem 24G --export=ALL,STEP=score paper/slurm/fig6_pmd.sbatch
#   sbatch -d afterok:$E --mem 24G --export=ALL,STEP=cells paper/slurm/fig6_pmd.sbatch
#   STEP=genes (needs fig6.sbatch deg), STEP=categories then STEP=panel (PMD gene sets), STEP=mag (E/F magnitude-aware
#   scores; needs score + categories + panel), STEP=plot
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
W=$EXPERT_ROOT/data/inference/metabolic_weinberg
REPRO=$PAPER_FIGS/fig4/data/leigh_pmd_repro_2026-10-15
MAG=$PAPER_FIGS/fig6/data/pmd_panel/donor_mag_trained.csv.gz
G=$EXPERT_ROOT/resources/genesets
SCORE=${DATA:-$PAPER_FIGS/fig6/data/pmd_mtmt}/score
BANK=$EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz
if [[ $STEP == categories ]]; then
  # MitoCarta3.0 PMD gene categories, set z per PBMC donor and sorted-bulk sample (read by STEP=mag); needs score
  python paper/scripts/fig6/fig6_pmd_categories.py --mitocarta $G/mitocarta3/human.mitocarta3.0.html \
    --score-dir $SCORE --bank-npz $BANK --out-dir ${CAT_OUT:-$PAPER_FIGS/fig6/data/pmd_categories} \
    --bulk-dir $PAPER_FIGS/fig6/data/mito_bulk/expert \
    --bulk-meta $PAPER_FIGS/fig6/data/mito_bulk/salmon/261014_sample_metadata.csv "$@"
  exit 0
fi
if [[ $STEP == panel ]]; then
  # PanelApp 'Mitochondrial disorders' (panel 112, green genes) set z per donor, trained bank (read by STEP=mag and
  # the 6E plot); needs score + categories
  python paper/scripts/fig6/fig6_pmd_panel.py --panel-json $G/panelapp/panel112.json \
    --mitocarta $G/mitocarta3/human.mitocarta3.0.html --score-dir $SCORE --bank-npz $BANK --bank trained \
    --out-dir ${PANEL_OUT:-$PAPER_FIGS/fig6/data/pmd_panel} "$@"
  exit 0
fi
if [[ $STEP == mag ]]; then
  python paper/scripts/fig6/fig6_donor_mag.py --score-dir ${DATA:-$PAPER_FIGS/fig6/data/pmd_mtmt}/score \
    --cells ${DATA:-$PAPER_FIGS/fig6/data/pmd_mtmt}/score/cells.csv.gz \
    --bank-npz $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz \
    --modules $REPRO/leigh_v010/config/modules.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
    --readout-py expert/utils/readout.py --categories-dir $PAPER_FIGS/fig6/data/pmd_categories \
    --panel-dir $PAPER_FIGS/fig6/data/pmd_panel --out $MAG
  exit 0
fi
if [[ $STEP == plot ]]; then
  python paper/scripts/fig6/fig6_pmd_panel_plot.py --donor-mag $MAG --panel-dir $PAPER_FIGS/fig6/data/pmd_panel \
    --complexes panel --name fig6e_pmd_genes_panelapp --out-dir paper/figures/fig6 --src-dir $PAPER_FIGS/fig6/data --vizstyle-root $EXPERT_DATA_REPO
  python paper/scripts/fig6/fig6_pmd_plot.py --data-dir ${DATA:-$PAPER_FIGS/fig6/data/pmd_mtmt}/score --out-dir paper/figures/fig6 \
    --src-dir $PAPER_FIGS/fig6/data \
    --modules $REPRO/leigh_v010/config/modules.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
    --rna-dir $PAPER_FIGS/fig6/data/rna --panel-dir $PAPER_FIGS/fig6/data/pmd_panel \
    --donor-mag $MAG --pmd-genes $PAPER_FIGS/fig6/data/PMD_gene_list.csv --vizstyle-root $EXPERT_DATA_REPO "$@"
  exit 0
fi
python paper/scripts/fig6/fig6_pmd.py --stage $STEP --out-dir ${DATA:-$PAPER_FIGS/fig6/data/pmd_mtmt} \
  --query $W/converted_data/s_pbmc_collab.h5ad \
  --model-dir $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model \
  --bank-npz $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz \
  --ctrl-pools $PAPER_FIGS/fig5/data/hubness/v0.1.0_ctrl_pools_ds.npz \
  --deg-dir $PAPER_FIGS/fig6/data/pbmc/deg \
  --modules $REPRO/leigh_v010/config/modules.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
  --n-cores ${SLURM_CPUS_PER_TASK:-8} "$@"
