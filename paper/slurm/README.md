# SLURM recipes for the paper

Each `<fig>.sbatch` runs the cache steps behind one figure or table: the exact `paper/scripts/` calls with their
arguments, selected by `STEP=` (see the comment block at the top of each file). They were written for Northwestern's
Quest cluster.

To run them elsewhere:

1. Set the data roots and conda environment names in `paths.sh`, or put your values in `paths.local.sh` next to it
   (git-ignored, sourced last). Every recipe sources `paths.sh` before doing anything else.
2. Submit from the repository root. The `#SBATCH` headers carry placeholder `ACCOUNT` / `PARTITION` values, which
   are not valid on any cluster: pass your own on the command line (command-line options override the header) or
   edit the header. Times, memory and CPUs are what the original runs used.

   ```bash
   mkdir -p paper/slurm/logs analysis_out
   sbatch -A <account> -p <partition> --export=ALL,STEP=data paper/slurm/fig1.sbatch
   ```

3. Log files go to `paper/slurm/logs/` (create it once: `mkdir -p paper/slurm/logs`); outputs to `analysis_out/` relative to the repository root.

Most figures do not need any of this: their notebooks only read the plotting data from the Zenodo archive (see
`paper/figures/README.md`).
