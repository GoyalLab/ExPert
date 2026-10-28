#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 4 --mem=180G -t 02:00:00 -J fig4spfp
#SBATCH -a 0-2
#SBATCH -o $REPO/paper/slurm/logs/fig4spfp_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# RNA fingerprinting on sci-Plex3 PER CELL LINE (task 0 K562, 1 A549, 2 MCF7), so Fig. 4B-D can compare
# ExPert and fingerprinting on the same cells. Same export rules, dictionary (precomputed GWPS, 4,098
# fingerprints), input-gene restriction and mapping as the collaborator's published K562 run
# (fp_sciplex_pip.R, copied unchanged). 180G as their run requested: sacct reported 52.5 GB, but pbmclapply forks 4
# workers and MaxRSS misses them -- 72G was OOM-killed at ~13 min on all three lines. Measured with
# 180G: K562 132 GB, A549 126 GB, MCF7 179 GB (MCF7 has 2x the vehicle cells) -- do NOT lower it.
# Task 0 doubles as the check: its export and pip/cor are diffed against the published K562 run.
set -o pipefail
activate_env "$EXPERT_ENV"
cd $REPO
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
RS=$EXPERT_PROJECT/ExPert-benchmarks/.conda-envs/a122f0948aa175884d623126da81d4fb_/bin/Rscript
# The PUBLISHED GWPS dictionary is its own R package (`gwps`); the benchmarks env never installed it (Fig. 2/3
# use our own dict.rds). Only that package is copied into a private lib, so Seurat / RNAFingerprinting still
# come from the benchmarks env and are not shadowed by the collaborator's library.
export R_LIBS=$EXPERT_PROJECT/Rlib_fpdicts
RAW=$EXPERT_ROOT/data/inference/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad
V010=$EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model
EG3=$EXPERT_ROOT/models/v0.1.7/EFFGATE3/run_0/config/ExPert_v0/model
JR=$COLLAB_DATA/fingerprint_bench
LINES=(K562 A549 MCF7)
L=${LINES[${SLURM_ARRAY_TASK_ID:-0}]}
OUT=$PAPER_FIGS/fig4/data/v0.1.0/fingerprint_sciplex_perline/$L
CK=(); [[ $L == K562 && -d $JR ]] && CK=(--check-barcodes $JR/data/fp_sciplex_export/barcodes.txt)
echo "[task] $L -> $OUT"
[ -f $OUT/export/export_info.txt ] && [ -f $OUT/export/input_genes_shared.txt ] || \
  python -u paper/scripts/fig4/fig4_sciplex_fp_export.py --raw $RAW --cell-line $L --out $OUT/export \
    --models "$V010,$EG3" "${CK[@]}" || exit 1
$RS paper/analysis/fingerprint/fp_sciplex_pip.R $OUT/export $OUT ${SLURM_CPUS_PER_TASK:-4} \
  $OUT/export/input_genes_shared.txt || exit 1
if [[ $L == K562 && -d $JR ]]; then   # optional cross-check against the collaborator's published K562 run
  python - "$OUT" "$JR/results/fingerprint_sciplex_pip_matched" <<'PY'
import sys, numpy as np, pandas as pd
a, b = sys.argv[1], sys.argv[2]
for f in ("pip.csv", "cor.csv"):
    x, y = pd.read_csv(f"{a}/{f}", index_col=0), pd.read_csv(f"{b}/{f}", index_col=0)
    r, c = x.index.intersection(y.index), x.columns.intersection(y.columns)
    d = (x.loc[r, c] - y.loc[r, c]).abs().to_numpy()
    print(f"[check] {f}: {len(r)}/{len(y)} groups, {len(c)} fingerprints, max|diff| {d.max():.3g}, mean {d.mean():.3g}")
PY
fi
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
