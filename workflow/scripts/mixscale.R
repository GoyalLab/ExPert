library(argparse)

parser <- ArgumentParser(description = "Parser for mixscale pipeline.")

# Add argument
parser$add_argument("-i", "--input", type = "character", required = TRUE, help = "Path to input h5ad file.")
parser$add_argument("-o", "--output", type = "character", required = TRUE, help = "Path to output h5ad file.")
parser$add_argument("-d", "--min_deg", type = "integer", required = FALSE, default = 5,
          help = "Minimum number of differentially expressed genes vs. control per perturbation, default is 5 (int)")
parser$add_argument("-t", "--threshold", type = "double", required = FALSE, default = 2,
          help = "Number of standart deviations from control group to filter cells for, default is 2 (float)")
parser$add_argument("-p", "--pcol", type = "character", required = FALSE, default = "perturbation",
          help = "Column that labels the perturbation in meta data, default is 'perturbation' (string)")
parser$add_argument("-c", "--ctrl", type = "character", required = FALSE, default = "control",
          help = "Control label, default is 'control' (string)")
parser$add_argument("-s", "--save_seurat", type = "logical", required = FALSE, default = FALSE,
          help = "Whether to save the seurat_obj to disk or not.")
parser$add_argument("-w", "--work_dir", type = "character", required = FALSE, default = "./",
          help = "Give path to current working directory of pipeline.")
parser$add_argument("-n", "--n_cores", type = "integer", required = FALSE, default = 1,
          help = "Number of cores to use for parallel calculations.")

# Parse args
args <- parser$parse_args()

# Set working directory
source("bin/mixscale_utils.R")

# Get input file path
adata_p <- args$input
out_p <- args$output
min_deg <- args$min_deg
ctrl_dev <- args$threshold
perturbation_col <- args$pcol
ctrl_key <- args$ctrl
save_seurat <- args$save_seurat
n_cores <- args$n_cores
ds_name <- file_path_sans_ext(basename(adata_p))
# Log setup
print(args)

# 1. Read h5ad to Seurat object (this approach still doubles the memory, but works)
seurat_obj <- read_h5ad_to_seurat(adata_p)
# 2. Calculate Mixscale scores --> most time intensive part
seurat_obj <- mixscale_pipeline(
  seurat_obj,
  condition_col = perturbation_col,
  assay = "originalexp",
  ctrl_col = ctrl_key,
  min.de.genes = min_deg,
  verbose = T,
  n_cores = n_cores,
)

# 3. Write filtered object to disk (optional)
if (save_seurat) {
  seurat_out_p <- paste0(file_path_sans_ext(out_p), '.all.h5seurat')
  
  message("Saving seurat object to: ", seurat_out_p)
  SaveH5Seurat(seurat_obj, filename = seurat_out_p)
}
# 3.2 Save meta-data
message("Saving meta-data.")
write.csv(seurat_obj@meta.data, file = paste0(file_path_sans_ext(out_p), '_obs.csv'), quote = F)
# 3.3 Save DEGs for each perturbation as
if (max(seurat_obj$mixscale_score)!=0) {
  message("Saving DEGs for each perturbation")
  DE.genes.result.list <- seurat_obj@tools$deg
  # Extract all features of object
  all_features <- rownames(seurat_obj@assays$originalexp@meta.features)
  # Create mask of degs per perturbation
  deg.per.perturbation <- as.data.frame(sapply(DE.genes.result.list$genes, function(x) {
    all_features %in% x
  }))
  # Label features
  rownames(deg.per.perturbation) <- all_features
  # Convert to binary mask
  deg.per.perturbation <- deg.per.perturbation * 1
  # Write to file
  deg.o = paste0(file_path_sans_ext(out_p), '_deg_mask.csv')
  write.csv(deg.per.perturbation, file = deg.o, quote = F)
}

# 4. Subset by mixscale score threshold and include control cells in filtering
message("Filtering dataset based on absolute mixscore > ", ctrl_dev)
# 4.1 Filter with combined mask
mask = (abs(seurat_obj$mixscale_score) > ctrl_dev) | (seurat_obj[[perturbation_col]]==ctrl_key)
seurat_obj$mixscale_mask <- mask
seurat_obj <- subset(seurat_obj, subset = mixscale_mask)

# 5. Save only raw counts and convert to h5ad
message("Saving filtered dataset to file.")
tmp_out_p <- paste0(file_path_sans_ext(out_p), ".h5seurat")

# Convert metadata factors safely
seurat_obj@meta.data <- convert_factors_to_characters(seurat_obj@meta.data)
seurat_obj[["originalexp"]]@meta.features <- convert_factors_to_characters(
  seurat_obj[["originalexp"]]@meta.features
)
# Save file as seurat
SaveH5Seurat(seurat_obj, filename = tmp_out_p, overwrite = T)
# Convert to .h5ad on disk
Convert(
  tmp_out_p,
  dest = "h5ad",
  assay = "originalexp",
  X.layer = "counts",
  overwrite = T
)
# Remove temporary seurat file
file.remove(tmp_out_p)
# Print session info
sessionInfo()
