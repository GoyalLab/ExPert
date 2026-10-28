#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 32G --time 02:00:00
#SBATCH --job-name fig6dset
#SBATCH --output paper/slurm/logs/fig6dset_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 6 donor-set contrast (paper/scripts/fig6/fig6_donorset.py): unbiased pathway hits replicated across every balanced
# split of the donors into two sets. genhimem/<project account> while genomics is backed up. Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D=${DATA:-$PAPER_FIGS/fig6/data/pmd_mtmt}
python paper/scripts/fig6/fig6_donorset.py --data-dir $D --out-dir $D/donorset \
  --bank-npz $EXPERT_ROOT/data/inference/alcohol_damage/results/readout_v0.1.0/fullbank/bank.npz \
  --modules $PAPER_FIGS/fig4/data/leigh_pmd_repro_2026-10-15/leigh_v010/config/modules.tsv \
  --reactome-gmt $EXPERT_ROOT/resources/genesets/Enrichr.Reactome_Pathways_2024.gmt "$@"
