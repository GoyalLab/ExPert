#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --partition PARTITION
#SBATCH --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 40G --time 01:30:00
#SBATCH --job-name gse281574
#SBATCH --output paper/slurm/logs/gse281574_%j.out
source paper/slurm/paths.sh   # site paths + env (submit from the repo root)
# Fetch the cohort's GEO Seurat object (GSE281574) and list its cell metadata: are the 2 libraries / condition
# pools of individuals, and is there a per-individual (demultiplexed) column? Paths live here.
set -eo pipefail
D=$EXPERT_ROOT/data/inference/alcohol_damage/data/GSE281574
F=GSE281574_Liver_Multiome_Seurat_GEO.rds.gz
cd $D
[ -s $F ] || wget -q https://ftp.ncbi.nlm.nih.gov/geo/series/GSE281nnn/GSE281574/suppl/$F
ls -la $F
R=${F%.gz}
[ -s $R ] || gunzip -k $F                      # GEO wraps an (already gzipped) .rds in a second gzip
activate_env "$AUX_ENV"
Rscript -e '
o <- readRDS("'$D/$R'")
cat("class:", class(o), "\n")
md <- tryCatch(o@meta.data, error = function(e) o$meta.data)
cat("cells:", nrow(md), "\n"); print(colnames(md))
for (c in colnames(md)) { v <- md[[c]]; if (!is.numeric(v) || length(unique(v)) < 40) { cat("==", c, "\n"); print(table(v, useNA="ifany")) } }
write.csv(md, "'$D'/meta_data.csv")
'
