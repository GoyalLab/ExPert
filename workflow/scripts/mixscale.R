library(argparse)

parser <- ArgumentParser(description = "Parser for mixscale pipeline.")

# Add argument
parser$add_argument("-i", "--input", required = T, help = "Path to input h5ad file.")
parser$add_argument("-o", "--output", required = T, help = "Path to output csv file.")
parser$add_argument("-d", "--min_deg", required = F, default = 5,
                    help = "Minimum number of differentially expressed genes vs. control per perturbation, default is 5 (int)")
parser$add_argument("-t", "--threshold", required = F, default = 2,
                    help = "Number of standart deviations from control group to filter cells for, default is 2 (float)")
parser$add_argument("-p", "--pcol", required = F, default = "perturbation",
                    help = "Column that labels the perturbation in meta data, default is 'perturbation' (string)")
parser$add_argument("-c", "--ctrl", required = F, default = "control",
                    help = "Control label, default is 'control' (string)")
parser$add_argument("-s", "--save_seurat", required = F, default = F,
                    help = "Whether to save the seurat_obj to disk or not.")
parser$add_argument("-w", "--work_dir", required = F, default = "./",
                    help = "Give path to current working directory of pipeline.")

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

# 1. Read h5ad to Seurat object (this approach still doubles the memory, but works)
seurat_obj <- read_h5ad_to_seurat(adata_p)
# 2. Calculate Mixscale scores --> most time intensive part
seurat_obj <- mixscale_pipeline(
  seurat_obj,
  condition_col = perturbation_col,
  ctrl_col = ctrl_key,
  min.de.genes = min_deg,
  verbose = T
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
# 4. Subset by mixscale score threshold and include control cells in filtering
message("Filtering dataset based on absolute mixscore > ", ctrl_dev)
# 4.1 Filter with combined mask
mask = (abs(seurat_obj$mixscale_score) > ctrl_dev) | (seurat_obj$is_ctrl)
seurat_obj$mixscale_mask <- mask
seurat_obj <- subset(seurat_obj, subset = mixscale_mask)

# 5. Save only raw counts and convert to h5ad
message("Saving filtered dataset to file.")
tmp_out_p <- paste0(file_path_sans_ext(out_p), '.h5seurat')
seurat_obj@assays$originalexp@scale.data <- matrix()
seurat_obj@assays$originalexp@data <- seurat_obj@assays$originalexp@counts
seurat_obj@assays$originalexp@counts <- matrix()
# 6. Convert meta data factors to characters to avoid numeric conversion
seurat_obj@meta.data <- convert_factors_to_characters(seurat_obj@meta.data)
seurat_obj[["originalexp"]]@meta.features <- convert_factors_to_characters(seurat_obj[["originalexp"]]@meta.features)
SaveH5Seurat(seurat_obj, filename = tmp_out_p)
Convert(tmp_out_p, dest = 'h5ad', assay = "originalexp", X.layer = "data", overwrite = T)
rm_tmp_seurat_file <- file.remove(tmp_out_p)
