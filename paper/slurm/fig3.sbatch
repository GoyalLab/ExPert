#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 8G --time 01:00:00
#SBATCH --job-name fig3
#SBATCH --output paper/slurm/logs/fig3_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Memory: measured MaxRSS 5.6 GB (data, 51 s).
# Fig. 3 zero-shot / prior augmentation. STEP=data|layout3|zs|plot|all. Paths live here, never in the .py.
#   data   prior-graph fill placement + radial angles (fig3_data.py; fig3b_radial.parquet feeds Supplementary Fig. S6A)
#   layout3  3C placement tree (fig3b_layout.py)
#   zs     3B contexts (fig3a_contexts.tsv) and EFFGATE3 shipped-fill zero-shot (the folds: fig3a_folds.sbatch,
#          fig3a_seen.sbatch)
#   (the EFFGATE3 pull ledger, formerly 3D, is Supplementary Fig. S2B: figS2.sbatch STEP=pull)
#   d,e,f  Fig. 3D-F, redrawn from the deployment-panel prior screen (analysis_out/prior_screen/H4_full_p4: cos^4
#          zero-shot fill, = shipped models since 2026-10-23; SCREEN_TAG=H4_full for the old cos^1 screen)
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-all}
M=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
DATA=$EXPERT_DATA_REPO
SCREEN=analysis_out/prior_screen
OUT=paper/figures/fig3
# zero-shot = the 10,931 untrained rows the shipped model fills (19,895 - 8,964 trained). Layout/radial
# (data, layout3) show all 10,931; scoring (zs) can only use the 9,874 measured in a screen
if [[ $STEP == data || $STEP == all ]]; then
# bank order = the checkpoint's: text, then class_emb_extra_paths (hparams.yaml)
python paper/scripts/fig3/fig3_data.py --model-pt "$M/model.pt" \
  --perts $PAPER_FIGS/fig1/data/fig1c_perts.parquet \
  --bank-names text,DepMap,STRING --k 25 --n-null 3 --out-dir "$EXPERT_ROOT/$OUT/data"
fi
if [[ $STEP == layout3 ]]; then
# main 3C: placement tree -- Ward over the trained prototypes, every zero-shot fill grafted beside its
# nearest trained prototype (joint trees group fills apart) -> data/b_place/
python paper/scripts/fig3/fig3b_layout.py --model-pt "$M/model.pt" \
  --perts $PAPER_FIGS/fig1/data/fig1c_perts.parquet --tree placement --no-umap \
  --categories $PAPER_FIGS/fig1/data/fig1d_layout.parquet --out-dir "$EXPERT_ROOT/$OUT/data/b_place"
fi
if [[ $STEP == zs ]]; then
# A: EFFGATE3's shipped fill, per query and per dataset, full 19,895-row vocabulary, + 100 row-shuffled twins.
# Same scorer and inputs as the prior screen (its paths.sh), so pooled numbers match panels_6502446.out.
# rows named as in Fig. 1 ("context - condition"; T-cell donors pooled per condition)
python paper/scripts/fig3/fig3a_contexts.py --cells $PAPER_FIGS/fig1/data/fig1c_cells.parquet \
  --donor-celltype "CD4+ T" --out "$EXPERT_ROOT/$OUT/data/fig3a_contexts.tsv"
FIG_OUT=$OUT; source paper/analysis/readout/prior_screen/paths.sh   # paths.sh sets its own OUT
python $S panels $PATHS --zs-cache $REPO/analysis_out/qcache/EFFGATE3_zs.npz --zs-adata $ZS_ADATA \
  --capture $CAPTURE --split-meta $SPLIT_META --fold $FOLD --tag EFFGATE3 --pred-tag EFFGATE3 \
  --shipped-fill --model-pt $MODEL --zs-panel full --reps 100 \
  --group-map "$EXPERT_ROOT/$FIG_OUT/data/fig3a_contexts.tsv" \
  --shipped-json "$EXPERT_ROOT/$FIG_OUT/data/fig3a_shipped_zs.json"
fi
if [[ $STEP == plot || $STEP == all ]]; then
python paper/scripts/fig3/fig3_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" \
  --vizstyle-root "$DATA" --screen-root "$SCREEN" --screen-tag "${SCREEN_TAG:-H4_full_p4}" \
  --prior-screen-root paper/analysis/readout/prior_screen ${PLOT_ARGS}
fi
