#!/bin/bash
#SBATCH -A ACCOUNT -p PARTITION -N 1 --ntasks 1 --cpus-per-task 2 --mem=8G -t 24:00:00 -J zenodo
#SBATCH -o $REPO/analysis_out/logs/zenodo_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Zenodo records for the paper, as PRIVATE DRAFTS (zenodo_upload.py has no publish call).
#   CMD=plan | stage | upload   (default plan)
#   RECORDS="figure_data source_data"  subset (default all)
#   SANDBOX=1                   upload to sandbox.zenodo.org with $ZENODO_SANDBOX_TOKEN
#   REBUILD=1                   stage: rebuild archives that already exist
#   OPT_IN=1                    also the opt_in records (code tarball; final publication only)
# Token: export ZENODO_TOKEN (or ZENODO_SANDBOX_TOKEN) in your shell before sbatch; --export=ALL passes it on.
# Staged archives (~25 GB) live on <buy-in account>, the draft ids + reserved DOIs in zenodo_state.json.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "${SLURM_SUBMIT_DIR:-.}"
CMD=${CMD:-plan}
SPEC=paper/reproducibility/zenodo_records.yaml
STAGING=$GROUP_WORK/zenodo_staging
STATE=paper/reproducibility/zenodo_state.json
ARGS=(--spec $SPEC --staging $STAGING --state $STATE)
[[ -n $RECORDS ]] && ARGS+=(--records $RECORDS)
[[ $REBUILD == 1 ]] && ARGS+=(--rebuild)
[[ $OPT_IN == 1 ]] && ARGS+=(--opt-in)
[[ $SANDBOX == 1 ]] && ARGS+=(--sandbox --token-env ZENODO_SANDBOX_TOKEN)
python paper/reproducibility/zenodo_upload.py $CMD "${ARGS[@]}"
[[ -n $SLURM_JOB_ID ]] && sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State -P || true
