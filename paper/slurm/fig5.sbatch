#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 6G --time 01:00:00
#SBATCH --job-name fig5
#SBATCH --output paper/slurm/logs/fig5_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5 (production v0.1.0 on alcohol-associated liver disease nuclei; no retraining, no model load).
# Reads the per-cell-type ALD readout chain output (alcohol_damage/submit_readout_chain.sh, VARIANT=mtfree:
# mt-encoded genes + metallothioneins masked, the headline input).
# STEP=data: fig5_data.py (PANELS=a+b+c+d+top or all; SRC=predict default = fig5_predict.sbatch output, BANK=
#   trained|full; VAR=<model>__<mask>, default v0.1.0__mito_mt). (panels: fig5_final.sbatch).
# Memory: data MaxRSS 2.2 GB (4.6 min, UMAP of 26.8k nuclei dominates).
# STEP=audit_ald (--array 0-5, one input-baseline arm each) / audit_train (--array 0-7, one training
#   context each) / audit_merge: fig5_centring_audit.py (model load, CPU). audit_ald writes
#   input_genes.txt that audit_train restricts its input statistics to, so chain train after ald:
#   A=$(sbatch --parsable --array 0-5 --export=ALL,STEP=audit_ald --mem 30G paper/slurm/fig5.sbatch)   # MaxRSS 13-24 GiB, 3-4 min
#   T=$(sbatch --parsable --array 0-7 -d afterok:$A --export=ALL,STEP=audit_train --mem 13G paper/slurm/fig5.sbatch)  # MaxRSS 8-10 GiB, 3-4 min
#   sbatch -d afterok:$T --export=ALL,STEP=audit_merge --mem 8G paper/slurm/fig5.sbatch
# STEP=audit_score (--array 0-4, one scoring arm each; needs audit_ald run with --save-latents for arms
#   0,1,3 = pooled, self_ct, self_ctdonor) / audit_scoremerge: per-cell + pseudobulk readouts scored against
#   ALD modules, ctrl panels, established-ALD Reactome, a full Reactome scan, donor replication, collapse.
# STEP=audit_pairs (--array 0-2: input = Normal ctrl / cell-type mean / donor x cell-type mean) /
#   audit_pairsmerge: single-donor-vs-single-donor contrasts, null = same-condition pairs.
# Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
STEP=${STEP:-data}
PANELS=${PANELS:-all}
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
RO=$ALD/results/readout_v0.1.0_mtfree
OUT=paper/figures/fig5
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
if [[ $STEP == data ]]; then
  # SRC=predict (default): the current predict() readout (fig5_predict.sbatch: one pseudobulk per donor x cell type,
  #   deflate readout, no hubness); BANK=trained (main, DATA=$EXPERT_ROOT/$OUT/data) | full (extended data: zero-shot rows too,
  #   DATA=$OUT/ext_fullbank/data). SRC=variants: the older pbulk_x pools path (HUB=1 = stored hubness offset).
  VAR=${VAR:-v0.1.0__mito_mt}; SRC=${SRC:-predict}; BANK=${BANK:-trained}
  DATA=${DATA:-$EXPERT_ROOT/$OUT/data}
  COMMON=(--variant-dir $EXPERT_ROOT/$OUT/data/variants/$VAR --bank-npz $ALD/results/readout_${VAR%%__*}/fullbank/bank.npz
          --deg-dir $ALD/results/readout_deg --out-dir $DATA --panels "$PANELS" --modules $ALD/readout_panels.tsv
          --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt)
  if [[ $SRC == predict ]]; then
    PR=$EXPERT_ROOT/$OUT/data/predict/$VAR
    python paper/scripts/fig5/fig5_data.py "${COMMON[@]}" --bank $BANK --readout group_deflate \
      --tests $PR/tests__predict__${BANK}.csv.gz --profiles-npz $PR/profiles__predict__${BANK}.npz \
      --profiles-trained-npz $PR/profiles__predict__trained.npz --profiles-full-npz $PR/profiles__predict__full.npz "$@"
  else
    HUB=${HUB:-0}; HUBARGS=(); SFX=""
    [[ $HUB == 1 ]] && { HUBARGS=(--hub-pen $EXPERT_ROOT/$OUT/data/hubness/${VAR%%__*}_hub_pen.npz); SFX="__hub"; }
    python paper/scripts/fig5/fig5_data.py "${COMMON[@]}" "${HUBARGS[@]}" \
      --tests $EXPERT_ROOT/$OUT/data/variants/tests__${VAR}__trained${SFX}.csv.gz "$@"
  fi
fi
MDIR=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
DB=$DL_ROOT/data/store_out/1e6bafc04836e06a
AUD=$EXPERT_ROOT/$OUT/data/centring_audit
TASK=()
[[ -n ${SLURM_ARRAY_TASK_ID:-} ]] && TASK=(--task $SLURM_ARRAY_TASK_ID)
if [[ $STEP == audit_ald ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage ald --model-dir $MDIR --bank $RO/fullbank/bank.npz \
    --out-dir $AUD --query $ALD/data/readout/alcohol_readout_mtfree.h5ad \
    --saved-latents $RO/latents_true/z_raw_celltype.npy --n-cores ${SLURM_CPUS_PER_TASK:-8} "${TASK[@]}" "$@"
fi
if [[ $STEP == audit_train ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage train --model-dir $MDIR --bank $RO/fullbank/bank.npz \
    --out-dir $AUD --train-h5ad $DB/database_perturbed_200.h5ad --ctrl-h5ad $DB/database_ctrl.h5ad \
    --input-genes $AUD/input_genes.txt --n-cores ${SLURM_CPUS_PER_TASK:-8} "${TASK[@]}" "$@"
fi
if [[ $STEP == audit_merge ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage merge --out-dir $AUD "$@"
fi
if [[ $STEP == audit_score ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage score --bank $RO/fullbank/bank.npz --out-dir $AUD \
    --modules $ALD/readout_panels.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "${TASK[@]}" "$@"
fi
if [[ $STEP == audit_scoremerge ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage scoremerge --out-dir $AUD "$@"
fi
if [[ $STEP == audit_pairs ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage pairs --bank $RO/fullbank/bank.npz --out-dir $AUD \
    --modules $ALD/readout_panels.tsv \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "${TASK[@]}" "$@"
fi
if [[ $STEP == audit_pairsmerge ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage pairsmerge --out-dir $AUD "$@"
fi
if [[ $STEP == audit_zoom ]]; then
  python paper/scripts/fig5/fig5_centring_audit.py --stage zoom --bank $RO/fullbank/bank.npz --out-dir $AUD \
    --deg $ALD/results/readout_deg/deg.csv.gz --score-arm ${ENC:-self_ct} "$@"
fi
if [[ $STEP == audit_rest ]]; then
  ENCS=(self_ct pooled)
  python paper/scripts/fig5/fig5_centring_audit.py --stage rest --bank $RO/fullbank/bank.npz --out-dir $AUD \
    --modules $ALD/readout_panels.tsv --top-n 300 --score-arm ${ENCS[${SLURM_ARRAY_TASK_ID:-0}]} \
    --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
fi
