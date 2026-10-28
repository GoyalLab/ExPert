#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 4 --mem 8G --time 00:40:00
#SBATCH --job-name fig5pbde
#SBATCH --output paper/slurm/logs/fig5pbde_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fig. 5: donor-level pseudobulk DE (PyDESeq2) replacing the nucleus-level Wilcoxon table. Paths live here.
set -eo pipefail
activate_env "$EXPERT_ENV"
cd "$SLURM_SUBMIT_DIR"
ALD=$EXPERT_ROOT/data/inference/alcohol_damage
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python paper/scripts/fig5/fig5_pbulk_de.py --h5ad $ALD/data/readout/alcohol_readout.h5ad \
  --out-dir $ALD/results/readout_deg_pbulk --n-cpus ${SLURM_CPUS_PER_TASK:-4} "$@"
