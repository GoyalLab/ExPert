#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 2 --mem 64G -t 00:40:00 -J eg3cov
#SBATCH -o $REPO/paper/slurm/logs/eg3cov_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Per-perturbation training coverage in EFFGATE3's fold 4 (K562 essential + GWPS held out).
activate_env "$EXPERT_ENV"
cd $REPO
S=$DL_ROOT/data/store_out/1e6bafc04836e06a
python -u paper/analysis/_effgate3_train_coverage.py --db $S/database_perturbed.h5ad \
  --train-idx $S/splits/full/fold_4/id_train.txt.gz \
  --out $PAPER_FIGS/fig4/data/v0.1.0/effgate3_fold4_train_coverage.csv 2>&1
sacct -j "${SLURM_JOB_ID}" --format=JobID,Elapsed,MaxRSS,State -P || true
