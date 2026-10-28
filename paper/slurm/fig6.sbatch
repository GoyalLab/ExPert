#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 8 --mem 32G --time 02:00:00
#SBATCH --job-name fig6
#SBATCH --output paper/slurm/logs/fig6_%A_%a.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6 (production v0.1.0 on Weinberg paediatric mitochondrial-disease PBMCs; plain cosine (score stage keeps hub on/off arms), MT-* +
# metallothioneins masked). paper/scripts/fig6/fig6_data.py stages; paths live here, never in the .py.
# Chain:
#   E=$(sbatch --parsable --export=ALL,STEP=encode paper/slurm/fig6.sbatch)
#   D=$(sbatch --parsable --array 0-6 --mem 12G --export=ALL,STEP=deg paper/slurm/fig6.sbatch)
#   sbatch -d afterok:$E:$D --mem 24G --time 04:00:00 --export=ALL,STEP=score paper/slurm/fig6.sbatch
#   sbatch -d afterok:$E --mem 12G --export=ALL,STEP=umap paper/slurm/fig6.sbatch
#   sbatch -d afterok:$E:$D --mem 12G --export=ALL,STEP=genes paper/slurm/fig6.sbatch
#   (panels: fig6_pmd.sbatch STEP=plot)
# In-vitro (unit = cell type x stim): DSET=invitro, and pass  --celltype-col celltype_broad --split-col stim_condition
#   to every data step (deg --array 0-5).
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
W=$EXPERT_ROOT/data/inference/metabolic_weinberg
DSET=${DSET:-pbmc}
OUT=$PAPER_FIGS/fig6/data/$DSET
mkdir -p $OUT
python paper/scripts/fig6/fig6_data.py --stage $STEP --out-dir $OUT --task ${SLURM_ARRAY_TASK_ID:-0} \
  --query $W/converted_data/s_${DSET}_collab.h5ad \
  --model-dir $EXPERT_ROOT/models/ExPert_v0.1.0/run_0/config/ExPert_v0/model \
  --bank-npz $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz \
  --modules $W/modules.tsv --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt \
  --n-cores ${SLURM_CPUS_PER_TASK:-8} "$@"
