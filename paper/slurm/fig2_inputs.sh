#!/bin/bash
# SPDX-License-Identifier: MIT
#
# ExPert: a contrastive perturbation-prediction model for single-cell data.
#
# Copyright (c) 2024-2026 Leon Schwartz
# Northwestern University, Goyal Lab
#
# This file is part of the ExPert package and is released under the MIT
# License. See the LICENSE file in the project root for the full text.

# Benchmark inputs shared by fig2.sbatch (Fig. 2) and figS3.sbatch (Supplementary Fig. S3), so both read the
# same cached tables. Source it; it only sets variables. Paths live here and in the sbatch files, never in the .py.
BENCH=$EXPERT_PROJECT/ExPert-benchmarks
CTX7=$BENCH/metrics/ctx7
SMALL=$EXPERT_ROOT/benchmarks/metrics_small_rescored   # Sep 30 rescore: has the rank metrics
COSTS=$BENCH/metrics/slurm_costs.rebuilt.csv   # full sacct history; slurm_costs.csv lost ctx7 folds in an Oct 1 --merge
OVERRIDES=$BENCH/config/cost_overrides.yaml
DATA=$EXPERT_DATA_REPO
EXTRA=()                                      # newest score_extra output per size (--extra-ctx7 / --extra-small)
EXTRA_CTX7=()                                 # ... ctx7 only (Fig. S2)
for z in ctx7 small; do
  for v in extra_metrics.v2.csv extra_metrics.csv; do
    [[ -f $BENCH/metrics/$z/$v ]] && { EXTRA+=(--extra-$z "$BENCH/metrics/$z/$v"); break; }
  done
done
for v in extra_metrics.v2.csv extra_metrics.csv; do
  [[ -f $CTX7/$v ]] && { EXTRA_CTX7=(--extra-ctx7 "$CTX7/$v"); break; }
done
MASK=($EXPERT_ROOT/benchmarks/abl_mask_target_fold_*/metrics/mask_ablation.csv)   # target-mask ablation, every fold scored
MASK_QUERY=$DL_ROOT/data/store_out/1e6bafc04836e06a/database_perturbed_ctx7.h5ad
SEL=$CTX7/selective                           # score_selective.py per-cell cache
PB=$CTX7/pbulk_groups                         # score_pbulk_groups.py pooled ranks (pseudobulk)
# ExPert fold models (training logs + checkpoints): GPU inference estimate and training cost.
# 257 GB = measured MaxRSS of the fold_4 training job (sacct 6444867).
TRAIN_ROOT=$EXPERT_ROOT/models/folds
TRAIN_RAM_GB=257
EVAL_LOGS=$BENCH/logs/slurm/expert_logits
EVAL_CPUS=8                                   # expert_logits threads
