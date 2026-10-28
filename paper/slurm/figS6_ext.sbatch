#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 16 --mem 90G --time 01:30:00
#SBATCH --job-name figS6ext
#SBATCH --output paper/slurm/logs/figS6ext_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Supplementary Fig. S6D inputs: the cost of adding the zero-shot fills as candidates, on held-out queries of TRAINED
# perturbations (fold 4, K562 held out), for the post-hoc model (H4, never saw a filled row), ExPert's gated pull and the
# full pull (both trained with the fills as negatives). STEP=cache | ext | all (default all).
#   cache  FULLOPEN27's OOD query cache (cache_queries.py, verified against its v017 rank dump; ~71 GB, 8 min as for
#          EFFGATE3 / FULLOPEN3 in slurm/hub_caches.sbatch)
#   ext    fig3_extension.py over the three arms, exactly as fig3.sbatch STEP=ext -> $PAPER_FIGS/si/figS6/data/ext/
# Then figS6.sbatch STEP=data reads data/ext/fig3_extension.csv. Paths live here, never in the .py.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
STEP=${STEP:-all}
B=$DL_ROOT/data/store_out/1e6bafc04836e06a
FD=$B/models/full
MP=run_0/config/ExPert_v0/model/model.pt
OUT=$PAPER_FIGS/si/figS6/data/ext
if [[ $STEP == cache || $STEP == all ]]; then
  python paper/analysis/readout/cache_queries.py --model "$FD/FULLOPEN27/$MP" --adata $B/database_perturbed.h5ad \
    --fold-dir analysis_out/capped_splits/fold_4 --split ood_test_cap25.txt.gz --min-cells 5 \
    --out analysis_out/qcache/FULLOPEN27_fold4_ood.npz \
    --ranks-csv analysis_out/v017_readout/ranks/FULLOPEN27/FULLOPEN27_ranks.csv \
    2>&1 | grep -av --line-buffered -E "NumExpr|oneDNN|TensorFlow|SSE4|INFO "
fi
if [[ $STEP == ext || $STEP == all ]]; then
  python paper/scripts/fig3/fig3_extension.py \
    --arm "EFFGATE3=$FD/EFFGATE3/$MP=analysis_out/effgate_decomp/EFFGATE3_fold4_ood.npz" \
    --arm "FULLOPEN27=$FD/FULLOPEN27/$MP=analysis_out/qcache/FULLOPEN27_fold4_ood.npz" \
    --arm "H4=$FD/H4/$MP=analysis_out/qcache/H4_fold4_ood.npz" \
    --banks-from "$FD/EFFGATE3/$MP" --zs-cache analysis_out/qcache/EFFGATE3_zs.npz \
    --ref-ranks analysis_out/effgate_decomp/per_query.tsv.gz --k 25 --sim-pow 4 --out-dir $OUT
fi
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
