#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 96G --time 04:00:00
#SBATCH --job-name fig4
#SBATCH --array 0-1
#SBATCH --output paper/slurm/logs/fig4_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5 (drugs; no retraining -- drug queries are read through the CRISPR prototypes).
# DS=tahoe | sciplex selects the query.
# STEP=encode: one array task per input-centring arm (fig4_tahoe_encode.py); extra args pass
#   through (smoke: sbatch --array 1 paper/slurm/fig4.sbatch --max-cells 20000).
#   tahoe CTRL=subset  -> the subset's own 5,000 DMSO (98% plate1): arms global, line (--array 0-1)
#   tahoe CTRL=matched -> plate-matched DMSO replaces them: all four arms         (--array 0-3)
#   sciplex: one line per plate and vehicle on ~half the plates -> global, line   (--array 0-1)
# STEP=score: same array index -> arm, fig4_tahoe_score.py (no model). WORK_OVERRIDE / SCORE_OUT
#   redirect input / output (smoke).
# STEP=degron: fig4_degron.py, task 0 full input / task 1 MT-masked (--mem 24G is plenty: ~9 GB).
# STEP=degron_pb: degron through the shipped predict() path on pseudobulks, active-condition
#   filtered pre-model, best of loss/gain, trained + full bank (fig4_degron_pb.py).
#   array 0 = clone_nm1, 1 = clone_nm2.
# STEP=baseline_ref (once) / baseline (array 0 tahoe, 1 sciplex): fig4_baseline.py, the CMap-style
#   signature-correlation comparator, |Δx| per drug x dose, MoA kNN.
# STEP=degron_dropout: clone dropout under auxin vs DepMap (data only). STEP=plot: fig4_plot.py.
# STEP=degron_cov: data-only clone covariates (fig4_degron_covariates.py).
# STEP=summary: fig4_summary.py, fold over the permutation null per modality (Fig. 5b).
# STEP=attractor_probe: is the Tahoe GAPDH attractor a hub / own-transcript cue (fig4_attractor_probe.py).
# STEP=pb_encode -> pb_score -> pb_summary: full Tahoe from streamed pseudobulks (fig4_tahoe_pb.py).
# STEP=pb_sweep_encode (array 0-7) -> pb_sweep_score: full-Tahoe MT mask x input centre x latent ref x bank.
# STEP=pb_hub: best full-Tahoe arm with vs without the stored hubness offset (analytic k/C null).
# STEP=pb_panels: Fig. 5e/5f data (active responses: exact + pathway recall, worked examples).
# STEP=pb_atomstr: model-independent -- does the prototype geometry itself encode drug targets?
# STEP=pb_bridges: which prototypes sit BETWEEN drugs in the atom kNN graph (fig4_atom_bridges.py).
# STEP=pb_harmony: cross-line harmonization, drug-drug graph, hidden targets (fig4_tahoe_harmonized.py).
# STEP=pb_depmap: model gene-reading vs DepMap dependency vs cell-count depletion (fig4_tahoe_depmap.py).
#   MTMASK=1 on any sciplex_pb_* step = MT genes zeroed, into *_mtmask dirs (Tahoe: 1.7x at r@10).
# STEP=sciplex_pb_encode -> sciplex_pb_score: sci-Plex through the shipped predict() path on
#   pseudobulks (fig4_sciplex_pb.py); the score step sweeps bank x readout x loss/gain tail.
# STEP=mask_check: pairs the unmasked vs MT-masked sci-Plex runs (fig4_mask_check.py).
# STEP=sciplex_prep: sci-Plex 3 -> Tahoe-schema query (fig4_sciplex_targets + fig4_sciplex_prep).
# Memory (MaxRSS): tahoe encode global 204k cells 3-4 min; smoke 20k = 11 GB. Right-size from sacct.
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-encode}
DS=${DS:-tahoe}
CTRL=${CTRL:-subset}
M=v0.1.0
MDIR=${MDIR_OVERRIDE:-$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model}
INF=$EXPERT_ROOT/data/inference
T100=$INF/tahoe100m
SCI=$INF/sciplex
OUT=paper/figures/fig4
ARMS=(global line plate_line plate_line_alldrug)
GSETS=$T100/results/fig5_$M/gene_sets_corum_reactome_kegg.json
case $DS in
  tahoe)   Q=$T100/tahoe100m_ood_subset.h5ad
           WORK=$T100/results/fig5_$M/ctrl_$CTRL ;;     # per-cell latents: too big for the paper tree
  sciplex) Q=$SCI/fig5_sciplex3_query.h5ad
           WORK=$SCI/results/fig5_$M/ctrl_$CTRL ;;
esac

SPB=$SCI/results/fig5_$M/pseudobulk
# MTMASK=1 runs the sci-Plex path with the metallothionein/MT genes zeroed, into its OWN dirs so
# the unmasked tables stay comparable. On Tahoe this was worth 1.7x at recall@10.
MTGENES="MT-*,MT1*,MT2A,MT3,MT4"
SPB_OUT=$EXPERT_ROOT/$OUT/data/$M/sciplex_pb
if [[ ${MTMASK:-0} == 1 ]]; then
  SPB=$SCI/results/fig5_$M/pseudobulk_mtmask
  SPB_OUT=$EXPERT_ROOT/$OUT/data/$M/sciplex_pb_mtmask
  MTARGS=(--mask-genes "$MTGENES")
else
  MTARGS=()
fi
# SHIPPED Fig. 4B-D input (FPCELLS=1, default with MTMASK=1): ExPert reads EXACTLY the cells RNA
# fingerprinting reads -- its three per-line exports (<= 200 per condition + every vehicle cell) -- with the
# replicate plates pooled, so both methods have one input per line x drug x dose. Build the query with
# STEP=sciplex_prep_fpcells after fig4_sciplex_fp.sbatch. FPCELLS=0 restores the per-plate encode of the
# <= 150-cell query (the earlier configuration).
SCIQ=$SCI/fig5_sciplex3_query.h5ad
FPDIR=$EXPERT_ROOT/$OUT/data/$M/fingerprint_sciplex_perline
if [[ ${MTMASK:-0} == 1 && ${FPCELLS:-1} == 1 ]]; then
  # Conditions here hold ~200 cells (median and IQR both 200), so the vehicle cells are cut into
  # ~200-cell chunks: the responsiveness threshold is the 95th percentile of control-vs-control
  # distances AT THE DEPTH IT JUDGES. With the old 75-cell chunks it was 25-45% too strict
  # (406 vs 948 responsive conditions; 72 vs 110 drugs) -- responsive_conditions() warns on it.
  CTRL_CHUNK=${CTRL_CHUNK:-200}
  SCIQ=$SCI/fig5_sciplex3_query_fpcells.h5ad
  SPB=$SCI/results/fig5_$M/pseudobulk_mtmask_fpcells_cc200
  SPB_OUT=$EXPERT_ROOT/$OUT/data/$M/sciplex_pb_mtmask_fpcells_cc200
fi
# Side arms without touching the shipped dirs:
SPB=${SPB_OVERRIDE:-$SPB}; SPB_OUT=${SPB_OUT_OVERRIDE:-$SPB_OUT}; SCIQ=${SCIQ_OVERRIDE:-$SCIQ}
if [[ $STEP == sciplex_prep_fpcells ]]; then
  python paper/scripts/fig4/fig4_sciplex_prep.py --h5ad $SCI/SrivatsanTrapnell2020_sciplex3.h5ad \
    --targets-tsv $EXPERT_ROOT/$OUT/data/sciplex3_drug_targets.tsv \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --keep-barcodes $FPDIR/K562/export/barcodes.txt,$FPDIR/A549/export/barcodes.txt,$FPDIR/MCF7/export/barcodes.txt \
    --pool-plates --out $SCI/fig5_sciplex3_query_fpcells.h5ad "$@"
fi
# sci-Plex3 annotation inputs (candidate panel, drug -> family, CORUM-core partners, ChEMBL targets), built by
# STEP=sciplex_annot from our fingerprinting runs + curated tables; the ChEMBL API cache makes reruns offline.
CUR=$EXPERT_ROOT/$OUT/data/sciplex_annot_curated
ANN=$EXPERT_ROOT/$OUT/data/$M/sciplex_annot
if [[ $STEP == sciplex_annot ]]; then
  GMT=$EXPERT_ROOT/resources/genesets/Enrichr.CORUM.gmt
  [[ -f $GMT ]] || curl -sSf -o $GMT "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=CORUM"
  python paper/scripts/fig4/fig4_sciplex_annot.py --targets-tsv $EXPERT_ROOT/$OUT/data/sciplex3_drug_targets.tsv \
    --family-genes $CUR/sciplex_family_genes.csv --fp-pip $FPDIR/K562/pip.csv \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --corum-gmt $GMT \
    --chembl-cache $ANN/chembl_cache --out-dir $ANN "$@"
fi
if [[ $STEP == sciplex_pb_encode ]]; then
  python paper/scripts/fig4/fig4_sciplex_pb.py --step encode \
    --query $SCIQ --model-dir "$MDIR" --out-dir $SPB \
    "${MTARGS[@]}" --ctrl-chunk "${CTRL_CHUNK:-75}" \
    --n-cores "${SLURM_CPUS_PER_TASK:-8}" "$@"
fi
if [[ $STEP == sciplex_pb_score ]]; then
  python paper/scripts/fig4/fig4_sciplex_pb.py --step score \
    --encode-dir $SPB --out-dir $SPB_OUT \
    --candidates $ANN/candidates.npz "$@"
fi
if [[ $STEP == sciplex_pb_panels ]]; then
  # panel tables. Fig. 4B-D compare on MATCHED CELLS: fingerprinting is run per cell line
  # (fig4_sciplex_fp.sbatch, K562 + A549 + MCF7) and scored under the same annotation (STEP=sciplex_annot),
  # candidates and responsiveness mask as ExPert. Fig. 4E comes from the FlexPlex cells (fig4_flexplex.sbatch);
  # run both first. (A comment inside a \-continuation ends the command, so keep these lines above it.)
  python paper/scripts/fig4/fig4_sciplex_panels.py \
    --ranks $SPB_OUT/ranks_arms.csv \
    --drug-family $ANN/drug_family.csv \
    --hdac-complexes $CUR/hdac_complexes.csv \
    --flexplex $EXPERT_ROOT/$OUT/data/$M/flexplex/fig4e_ar42.csv \
    --fp-perline-dir $FPDIR \
    --ref-npz $INF/sciplex/results/fig5_$M/crispr_signature_ref.npz \
    --pseudobulk $SPB/pseudobulk.h5ad \
    --candidates $ANN/candidates.npz \
    --encode-dir $SPB --target-csv $ANN/chembl_targets.csv \
    --family-csv $CUR/sciplex_family_genes.csv \
    --partner-sets $ANN/corum_core_sets.csv \
    --out-dir $SPB_OUT/panels "$@"
fi
if [[ $STEP == mask_check ]]; then
  # pairs the unmasked and MT-masked sci-Plex runs condition by condition (needs both to have run)
  python paper/scripts/fig4/fig4_mask_check.py \
    --unmasked $EXPERT_ROOT/$OUT/data/$M/sciplex_pb/ranks_arms.csv \
    --masked $EXPERT_ROOT/$OUT/data/$M/sciplex_pb_mtmask/ranks_arms.csv \
    --out $EXPERT_ROOT/$OUT/data/$M/sciplex_pb_mtmask/mask_check.csv "$@"
fi
if [[ $STEP == sciplex_prep ]]; then
  python paper/scripts/fig4/fig4_sciplex_targets.py \
    --drugs-csv $EXPERT_ROOT/$OUT/data/sciplex3_drug_annotation_raw.csv --out $EXPERT_ROOT/$OUT/data/sciplex3_drug_targets.tsv
  python paper/scripts/fig4/fig4_sciplex_prep.py --h5ad $SCI/SrivatsanTrapnell2020_sciplex3.h5ad \
    --targets-tsv $EXPERT_ROOT/$OUT/data/sciplex3_drug_targets.tsv \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --out "$Q" "$@"
fi
if [[ $STEP == encode ]]; then
  T=${SLURM_ARRAY_TASK_ID:-0}
  EXTRA=()
  [[ $DS == tahoe && $CTRL == matched ]] && EXTRA+=(--controls "$T100/tahoe100m_dmso_plate_matched.h5ad")
  [[ $T == 0 ]] && EXTRA+=(--harvest-bank)
  echo "[task $T] ds=$DS arm=${ARMS[$T]} ctrl=$CTRL"
  python paper/scripts/fig4/fig4_tahoe_encode.py \
    --model-dir "$MDIR" --query "$Q" --out-dir "$WORK" --arm "${ARMS[$T]}" \
    --n-cores "${SLURM_CPUS_PER_TASK:-8}" "${EXTRA[@]}" "$@"
fi
if [[ $STEP == score ]]; then
  T=${SLURM_ARRAY_TASK_ID:-0}
  WORK=${WORK_OVERRIDE:-$WORK}
  python paper/scripts/fig4/fig4_tahoe_score.py \
    --encode-dir "$WORK" --bank "$WORK/bank.npz" --arm "${ARMS[$T]}" \
    --out-dir "${SCORE_OUT:-$EXPERT_ROOT/$OUT/data/$M/$DS/ctrl_$CTRL/${ARMS[$T]}}" \
    --gene-sets-json "$GSETS" "$@"
fi
if [[ $STEP == degron_pb ]]; then
  # The degron through the path the drug screens settled on: pseudobulks, active-condition
  # filtering BEFORE the model, deflate at shipped settings, best of the loss and gain tails,
  # trained AND full bank (the full bank is what scores the zero-shot clone genes).
  # 8 tasks = 2 barcode assignments x 4 arms.
  #   assignment: 0-3 = clone_nm1 (1 mismatch), 4-7 = clone_nm2 (2 mismatches)
  #   arm: did = remove the auxin shift common to all clones (difference-in-differences);
  #        _mt   = + metallothionein/MT genes masked and dead nuclei (pct_counts_mt > 20) dropped,
  #        nodid = the within-clone difference only (the common auxin effect left in).
  T=${SLURM_ARRAY_TASK_ID:-4}
  CCOL=(clone_nm1 clone_nm1 clone_nm1 clone_nm1 clone_nm2 clone_nm2 clone_nm2 clone_nm2)
  TAG=(_did _did_mt _nodid _nodid_mt _did _did_mt _nodid _nodid_mt)
  MT="MT-*,MT1*,MT2A,MT3,MT4"
  ARGS_did=(--did); ARGS_did_mt=(--did --mask-genes "$MT" --max-pct-mt 20)
  ARGS_nodid=(--no-did); ARGS_nodid_mt=(--no-did --mask-genes "$MT" --max-pct-mt 20)
  case ${TAG[$T]} in
    _did)       A=("${ARGS_did[@]}") ;;      _did_mt)   A=("${ARGS_did_mt[@]}") ;;
    _nodid)     A=("${ARGS_nodid[@]}") ;;    _nodid_mt) A=("${ARGS_nodid_mt[@]}") ;;
  esac
  python paper/scripts/fig4/fig4_degron_pb.py \
    --model-dir "$MDIR" --query $INF/degron/data/degron_labeled.h5ad \
    --out-dir "$EXPERT_ROOT/$OUT/data/$M/degron_pb" --clone-col "${CCOL[$T]}" --tag "${TAG[$T]}" \
    --condition-col condition --control-label DMSO "${A[@]}" \
    --n-cores "${SLURM_CPUS_PER_TASK:-8}" "$@"
fi
if [[ $STEP == degron ]]; then
  # task 0 = full input, task 1 = MT-encoded + metallothionein genes masked (same string as ALD/TA)
  # tasks 2-4 = MT-masked input + dead-nucleus filter (pct_counts_mt <= 20 / 17.5 / 15)
  T=${SLURM_ARRAY_TASK_ID:-0}
  VAR=(full mtfree mtfree_mt20 mtfree_mt17.5 mtfree_mt15)
  MT="MT-*,MT1*,MT2A,MT3,MT4"; MASK=("" "$MT" "$MT" "$MT" "$MT")
  MAXMT=("" "" 20 17.5 15)
  FILT=(); [[ -n "${MAXMT[$T]}" ]] && FILT=(--max-pct-mt "${MAXMT[$T]}")
  python paper/scripts/fig4/fig4_degron.py \
    --model-dir "$MDIR" --query $INF/degron/data/degron_labeled.h5ad \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --out-dir "$EXPERT_ROOT/$OUT/data/$M/degron/${VAR[$T]}" \
    --condition-col condition --control-condition DMSO --treated-condition IAA_24h \
    --clone-col clone_fuzzy --mask-genes "${MASK[$T]}" "${FILT[@]}" --n-cores "${SLURM_CPUS_PER_TASK:-8}" "$@"
fi
REF=$INF/sciplex/results/fig5_$M/crispr_signature_ref.npz
if [[ $STEP == baseline_ref ]]; then
  python paper/scripts/fig4/fig4_baseline.py --stage ref --ref-npz "$REF" \
    --model-genes-h5ad $DL_ROOT/data/store_out/1e6bafc04836e06a/database_perturbed.h5ad \
    --pbulk-dir $DL_ROOT/perturb-seq/store/27dab279946b4ea6/pseudobulk \
    --symbol-map-h5ad $SCI/SrivatsanTrapnell2020_sciplex3.h5ad "$@"
fi
if [[ $STEP == baseline ]]; then
  # task 0 = Tahoe (global arm, subset DMSO), task 1 = sci-Plex (line arm): the arms the figure shows
  T=${SLURM_ARRAY_TASK_ID:-0}
  if [[ $T == 0 ]]; then BQ=$T100/tahoe100m_ood_subset.h5ad; BD=$EXPERT_ROOT/$OUT/data/$M/tahoe/ctrl_subset/global; BS=""; MOA=()
  else BQ=$SCI/fig5_sciplex3_query.h5ad; BD=$EXPERT_ROOT/$OUT/data/$M/sciplex/ctrl_subset/line; BS=cell_name
       MOA=(--moa-tsv $EXPERT_ROOT/$OUT/data/sciplex3_drug_targets.tsv); fi
  python paper/scripts/fig4/fig4_baseline.py --stage query --ref-npz "$REF" \
    --query "$BQ" --expert-dir "$BD" --out-dir "$BD/baseline" --stratum-col "$BS" \
    --gene-sets-json "$GSETS" "${MOA[@]}" "$@"
fi
if [[ $STEP == degron_dropout ]]; then
  python paper/scripts/fig4/fig4_degron_dropout.py --query $INF/degron/data/degron_labeled.h5ad \
    --pool-tsv $INF/degron/pool.tsv --depmap-csv resources/edges/depmap/CRISPRGeneEffect.csv \
    --out "$EXPERT_ROOT/$OUT/data/$M/degron/degron_dropout.csv" --condition-col condition \
    --control-condition DMSO --treated-condition IAA_24h --clone-col clone_fuzzy "$@"
fi
if [[ $STEP == plot ]]; then
  python paper/scripts/fig4/fig4_plot.py --data-dir "$EXPERT_ROOT/$OUT/data" --out-dir "$OUT" \
    --vizstyle-root $EXPERT_DATA_REPO ${PLOT_ARGS}
fi
if [[ $STEP == degron_cov ]]; then
  # data-only clone covariates (response strength, target expression) on the nuclei Fig. 5g uses
  D=$EXPERT_ROOT/$OUT/data/$M/degron
  python paper/scripts/fig4/fig4_degron_covariates.py --query $INF/degron/data/degron_labeled.h5ad \
    --clones $D/mtfree_mt20/clones.csv --out $D/degron_covariates.csv --condition-col condition \
    --control-condition DMSO --treated-condition IAA_24h --clone-col clone_fuzzy --max-pct-mt 20 "$@"
fi
if [[ $STEP == summary ]]; then
  # Fig. 5b: fold over the permutation null with bootstrap CIs, one row per modality (the arms 5c-5h show)
  # main = trained targets ranked among trained prototypes; S = all targets over the full bank
  B=$EXPERT_ROOT/$OUT/data/$M
  DSETS=(--dataset sci-Plex $B/sciplex/ctrl_subset/line/scores_drug.npz $B/sciplex/ctrl_subset/line/ranks_drug.csv drug targets target_class
         --dataset Tahoe $B/tahoe/ctrl_subset/global/scores_drug.npz $B/tahoe/ctrl_subset/global/ranks_drug.csv drug targets target_class
         --dataset degron $B/degron/mtfree_mt20/scores_did.npz $B/degron/mtfree_mt20/clones.csv clone clone -)
  python paper/scripts/fig4/fig4_summary.py "${DSETS[@]}" --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --trained-only --out $EXPERT_ROOT/$OUT/data/fig4b_summary_source.csv --out-queries $EXPERT_ROOT/$OUT/data/fig4b_queries_source.csv "$@"
  python paper/scripts/fig4/fig4_summary.py "${DSETS[@]}" --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --out $EXPERT_ROOT/$OUT/data/fig4S_summary_alltargets_source.csv --out-queries $EXPERT_ROOT/$OUT/data/fig4S_queries_alltargets_source.csv "$@"
  # per (drug, target) pair instead of any-target-per-drug, trained targets: the S comparison for B
  python paper/scripts/fig4/fig4_summary.py "${DSETS[@]}" --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --trained-only --unit pair --out $EXPERT_ROOT/$OUT/data/fig4S_summary_pairs_source.csv --out-queries $EXPERT_ROOT/$OUT/data/fig4S_queries_pairs_source.csv "$@"
fi
if [[ $STEP == attractor_probe ]]; then
  # Fig. 5g follow-up: GAPDH attractor = biology, prototype hub, or own-transcript cue?
  W0=$T100/results/fig5_$M/ctrl_subset
  python paper/scripts/fig4/fig4_attractor_probe.py --encode-dir $W0 --arm global \
    --bank $W0/bank.npz --query $T100/tahoe100m_ood_subset.h5ad \
    --train-h5ad $DL_ROOT/data/store_out/1e6bafc04836e06a/database_perturbed.h5ad \
    --out-dir $EXPERT_ROOT/$OUT/data/$M/tahoe/attractor_probe "$@"
fi
# ---- full Tahoe-100M via streamed pseudobulks (paper/analysis/tahoe_pseudobulk.py) -----------------------
PB=$T100/pseudobulk/tahoe100m_pseudobulk_plate_line_drug_dose.h5ad
PBW=$T100/results/fig5_$M/pseudobulk
PBO=$EXPERT_ROOT/$OUT/data/$M/tahoe_pb
if [[ $STEP == pb_encode ]]; then
  python paper/scripts/fig4/fig4_tahoe_pb.py --stage encode --pseudobulk "$PB" \
    --model-dir "$MDIR" --out-dir "$PBW" --n-cores "${SLURM_CPUS_PER_TASK:-8}" "$@"
fi
if [[ $STEP == pb_score ]]; then
  python paper/scripts/fig4/fig4_tahoe_pb.py --stage score --encode-dir "$PBW" \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --out-dir "$PBO" --lof-only "$@"
fi
if [[ $STEP == pb_summary ]]; then
  for LEV in drug drug_dose; do
    python paper/scripts/fig4/fig4_summary.py \
      --dataset "Tahoe-full|$LEV" $PBO/scores_$LEV.npz $PBO/ranks_$LEV.csv key targets target_class \
      --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --trained-only \
      --out $PBO/summary_$LEV.csv --out-queries $PBO/queries_$LEV.csv "$@"
  done
fi
# ---- full-Tahoe readout sweep: MT mask x input centre (encode) -> latent ref x bank (score) ----
PBS=$T100/results/fig5_$M/pseudobulk_sweep
if [[ $STEP == pb_sweep_encode ]]; then
  # array 0-7: task = mt*4 + ic; mt 0 none / 1 masked, ic global / line / plate / plate_line
  T=${SLURM_ARRAY_TASK_ID:-0}
  MTS=(none mtmask); ICS=(global line plate plate_line)
  MTV=${MTS[$((T / 4))]}; ICV=${ICS[$((T % 4))]}
  MASK=""; [[ $MTV == mtmask ]] && MASK="MT-*,MT1*,MT2A,MT3,MT4"
  python paper/scripts/fig4/fig4_tahoe_pb.py --stage encode --pseudobulk "$PB" \
    --model-dir "$MDIR" --out-dir "$PBS/${MTV}_${ICV}" --mask-genes "$MASK" --input-center "$ICV" \
    --n-cores "${SLURM_CPUS_PER_TASK:-8}" "$@"
fi
if [[ $STEP == pb_sweep_score ]]; then
  python paper/scripts/fig4/fig4_tahoe_pb_sweep.py --encode-dirs $PBS/*_* \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --out $PBO/readout_sweep.csv "$@"
fi
if [[ $STEP == pb_hub ]]; then
  # best arm (MT mask, pooled-DMSO input, matched plate x line DMSO), with vs without the stored hub offset
  HUB=$T100/results/fig5_$M/hub_pen_v0.1.0.npz
  for H in off on; do
    HA=(); [[ $H == on ]] && HA=(--hub-pen "$HUB")
    python paper/scripts/fig4/fig4_tahoe_pb_sweep.py --encode-dirs $PBS/mtmask_global \
      --bank $T100/results/fig5_$M/ctrl_subset/bank.npz --refs pl --banks trained full "${HA[@]}" \
      --out $PBO/readout_best_hub_${H}.csv "$@"
  done
fi
if [[ $STEP == pb_panels ]]; then
  # Fig. 5e/5f data on the recommended arm (MT mask, per-line input centring), active responses only
  python paper/scripts/fig4/fig4_tahoe_panels.py \
    --encode-dir $PBS/mtmask_line --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --gene-sets-json "$GSETS" --out-dir $PBO/panels "$@"
fi
if [[ $STEP == pb_atomstr ]]; then
  # no drug responses: only the trained prototypes and the drug->target annotations (Fig. 5f)
  python paper/scripts/fig4/fig4_atom_structure.py \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --drug-meta $T100/tahoe100m_drug_target_map.csv --gene-sets-json "$GSETS" \
    --out-prefix $PBO/atom_structure "$@"
fi
if [[ $STEP == pb_bridges ]]; then
  # shortest paths between drug target sets through the bank's own kNN graph (Fig. 5i)
  # The priors MUST be passed here: the prior-fold table is written only under `if priors:`, so a
  # run without them leaves a STALE *_bridge_prior_folds.csv next to a freshly rewritten
  # *_bridges.csv -- the two then disagree about which bridges exist.
  PR=resources/embeddings/priors
  GS=$EXPERT_ROOT/resources/genesets
  python paper/scripts/fig4/fig4_atom_bridges.py \
    --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --drug-meta $T100/tahoe100m_drug_target_map.csv --out-prefix $PBO/atom_structure \
    --annotation-gmt "$GS/Enrichr.GO_Biological_Process_2023.gmt" \
    --prior "text=$MDIR/gene_embeddings.npz" \
    --prior "STRING=$PR/string.npz" \
    --prior "DepMap=$PR/depmap_dstd.npz" \
    --prior "L1000=$PR/l1000.npz" \
    --prior "ProteinTalks=$PR/proteintalks_global.npz" \
    --prior "morphology=$PR/morph_crispr.npz" \
    --prior "genomic position=$PR/gff_pos.npz" "$@"
fi
if [[ $STEP == pb_harmony ]]; then
  # beyond recovery: cross-line harmonization, drug-drug graph, hidden-target candidates
  python paper/scripts/fig4/fig4_tahoe_harmonized.py \
    --encode-dir $PBS/mtmask_line --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --pseudobulk "$PB" --out-dir $PBO/harmonized "$@"
fi
if [[ $STEP == pb_depmap ]]; then
  # does the model's gene-level reading predict WHICH LINES a drug depletes? (DepMap dependency)
  python paper/scripts/fig4/fig4_tahoe_depmap.py \
    --encode-dir $PBS/mtmask_line --bank $T100/results/fig5_$M/ctrl_subset/bank.npz \
    --depmap-csv resources/edges/depmap/CRISPRGeneEffect.csv \
    --depmap-models resources/edges/depmap/Model.csv --out-dir $PBO/depmap "$@"
fi
if [[ $STEP == tahoe_mech ]]; then
  # Fig. 4G/H (port of the collaborator's Tahoe mechanism scripts): expression units -> well positions ->
  # expression activity -> position-corrected unit z -> per-dose ranks (G) + glucocorticoid scatter (H).
  # Public Tahoe tables: huggingface.co/datasets/tahoebio/Tahoe-100M metadata/{sample,drug}_metadata.parquet.
  # Measured: all stages 18 min, MaxRSS 16.7 GB (expr is 15 min of it). STAGES=all by default.
  MD=$T100/metadata
  for f in sample_metadata.parquet drug_metadata.parquet; do
    [[ -f $MD/$f ]] || { mkdir -p $MD; curl -sSL -o $MD/$f \
      "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/$f"; }
  done
  MW=$T100/results/fig5_$M/tahoe_mech
  python paper/scripts/fig4/fig4_tahoe_mech.py --stage "${STAGES:-all}" --out-dir "$MW" --pseudobulk "$PB" \
    --sample-metadata $MD/sample_metadata.parquet --drug-metadata $MD/drug_metadata.parquet \
    --encode-dir $PBS/mtmask_line --bank $T100/results/fig5_$M/ctrl_subset/bank.npz "$@"
  mkdir -p $PBO/../tahoe_mech
  cp "$MW/dose_ranks.csv" "$MW/clobetasol_scatter.csv" $PBO/../tahoe_mech/
fi
