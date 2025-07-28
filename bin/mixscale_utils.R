## Util functions
# Set Seurat object version
options(Seurat.object.assay.version = 'v3') 
# Load libraries
library(Seurat)
library(ggridges)
library(Mixscale)
library(SeuratDisk)
library(ggplot2)
library(tools)
library(zellkonverter)
library(SingleCellExperiment)
library(dplyr)


get_n_cores <- function(default = 1) {
  job_id <- Sys.getenv("SLURM_JOB_ID")
  if (job_id == "") {
    return(default)
  }
  cpu_check_cmd <- paste0("squeue -j ", job_id, " -o '%C' | tail -n 1")
  n.cores <- as.integer(system(cpu_check_cmd, intern = T))
  return(n.cores)
}

read_h5ad_to_seurat <- function(adata_p, layer="X") {
  # Use zellkonverter to read sce
  message("Reading h5ad adata", adata_p)
  sce <- readH5AD(adata_p, reader = "R")
  # Convert sce to Seurat object
  message("Converting to Seurat object")
  seurat_obj <- as.Seurat(sce, counts = "X", data = NULL)
  rm(sce)
  gc()
  return(seurat_obj)
}

subset_to_top_perturbations <- function(seurat_obj, perturbation = "perturbation", n_perts = 20) {
  target_perturbations <- seurat_obj@meta.data %>% 
    group_by(!!sym(perturbation)) %>%
    summarize(n=n()) %>% arrange(desc(n)) %>%
    head(n_perts) %>% pull(!!sym(perturbation)) %>% as.vector
  seurat_obj <- subset(seurat_obj, subset = !!sym(perturbation) %in% target_perturbations)
  return(seurat_obj)
}

get_mixscale_summary <- function(
    seurat_obj,
    perturbation_col = "perturbation"
) {
  if (!"mixscale_score" %in% colnames(seurat_obj@meta.data)) {
    stop("Run Mixscale before using this function.")
  }
  summary_stats <- seurat_obj@meta.data %>% 
    group_by(!!sym(perturbation_col)) %>% 
    summarise(
      fano=var(mixscale_score)**2/(mean(mixscale_score)+1e-9),
      mean=mean(mixscale_score),
      count=n(),
      max=max(mixscale_score),
      min=min(mixscale_score)
    )
  # Set mixscale scores of NP to 0 instead of 1
  np_mask <- (summary_stats$min==1) & (summary_stats$max==1) & (summary_stats$mean==1)
  summary_stats[np_mask,c("mean", "min", "max")] <- 0
  return(summary_stats)
}

adjust_low_deg_mixscale <- function(
    seurat_obj,
    summary_stats,
    plot = T
) {
  # Mark perturbations that have no differentially expressed genes compared to control (exactly 1 for all cells)
  valid_perturbations <- summary_stats %>% filter(min!=0 & max!=0) %>% pull(perturbation) %>% as.vector
  seurat_obj@meta.data$NP <- F
  # Center NP perturbations on 0 instead of 1
  seurat_obj@meta.data[!seurat_obj@meta.data$perturbation %in% valid_perturbations,"NP"] <- T
  seurat_obj@meta.data[seurat_obj$NP, "mixscale_score"] <- 0
  
  if (plot) {
    # Plot NP ratio
    if (!"is_ctrl" %in% colnames(seurat_obj@meta.data)) {
      seurat_obj <- add_ctrl_label(seurat_obj)
    }
    pert_table <- table(seurat_obj@meta.data[!seurat_obj$is_ctrl,"NP"])
    names(pert_table) <- ifelse(names(pert_table) == "TRUE", "NP", "P")
    percentages <- round(pert_table / sum(pert_table) * 100)
    labels <- paste0(names(pert_table), ": ", percentages, "%")
    pie(
      pert_table,
      labels = labels,
      main = "Ratio of perturbations with no DEGs"
    )
  }
  return(seurat_obj)
}

subset_ctrl <- function(seurat_obj, n_ctrl = 10000, col = 'perturbation', ctrl_key = 'control', seed = 42) {
  # Split data into control and perturbed
  ctrl_idc <- rownames(seurat_obj@meta.data[seurat_obj@meta.data[[col]]==ctrl_key,])
  p_idc <- rownames(seurat_obj@meta.data[seurat_obj@meta.data[[col]]!=ctrl_key,])
  # Check if n_ctrl is larger than actual number of control cells
  n_ctrl_ref <- length(ctrl_idc)
  if (n_ctrl >= n_ctrl_ref) {
    message('Number of control cells is not smaller than n_ctrl: ', n_ctrl, '; using all cells: ', n_ctrl_ref)
    return(seurat_obj)
  }
  # Set seed for reproducibility
  set.seed(seed)
  # Concatenate sampled control cells and all other perturbed cells to include
  subset_idc <- c(sample(ctrl_idc, size=n_ctrl), p_idc)
  # Save cell barcode as column to filter for
  seurat_obj$cell_idx <- rownames(seurat_obj@meta.data)
  # Filter for selected cells
  seurat_obj <- subset(seurat_obj, subset = cell_idx %in% subset_idc)
  return(seurat_obj)
}

pp_seurat <- function(seurat_obj) {
  # Standard pre-processing
  # Filter genes with non-zero expression across all cells
  seurat_obj <- subset(seurat_obj, features = rownames(seurat_obj)[Matrix::rowSums(seurat_obj[["originalexp"]]@counts) > 0])
  seurat_obj <- NormalizeData(seurat_obj)
  seurat_obj <- FindVariableFeatures(seurat_obj)
  seurat_obj <- ScaleData(seurat_obj)
  seurat_obj <- RunPCA(seurat_obj)
  return(seurat_obj)
}

plt_umap <- function(seurat_obj, col = "perturbation", n_dims = 50, nneighbors = 15) {
  seurat_obj <- RunUMAP(seurat_obj, reduction = 'pca', dims = 1:n_dims, n.neighbors = nneighbors)
  palette_c <- scales::hue_pal(l = n_dims)(n_perts)
  palette_c[which(levels(perts)=="control")] = "#919aa1"
  DimPlot(object = seurat_obj, reduction = 'umap', group.by = col, pt.size = 1, cols = palette_c) +
    labs(title = "Perturbation labels") +
    theme(text = element_text(size=24))
}

convert_factors_to_characters <- function(df) {
  factor_col_idc = which(sapply(df, class)=="factor")
  df[,factor_col_idc] <- lapply(df[,factor_col_idc], as.character)
  return(df)
}

# ---- Helper: Perform DEG for a single gene ----
calc_deg_for_gene <- function(object, gene, cells.s, labels, nt.class.name, 
                              assay, test.use, logfc.threshold, pval.cutoff) {
  # Set identities
  Idents(object) <- labels
  # Get perturbed & control cells
  orig.guide.cells <- intersect(WhichCells(object, idents = gene), cells.s)
  nt.cells <- intersect(WhichCells(object, idents = nt.class.name), cells.s)
  
  # Try to calculate DEGs
  deg.result <- data.frame()
  tryCatch({
    deg.result <- FindMarkers(
      object, ident.1 = orig.guide.cells, ident.2 = nt.cells,
      group.by = labels, assay = assay, 
      test.use = test.use, logfc.threshold = logfc.threshold, 
      verbose = FALSE, min.pct = 0.1
    )
    # Filter by p-value
    deg.result <- deg.result[deg.result$p_val_adj < pval.cutoff, ]
    # Add combined score
    deg.result$score <- abs(deg.result$avg_log2FC) * -log10(deg.result$p_val_adj)
  }, error = function(e) {
    message("Error processing gene ", gene, ": ", e$message)
  })
  
  # Return object
  return(list(genes = rownames(deg.result), result = deg.result))
}

# ---- Sequential version ----
calc_degs_iterative <- function(object, assay = "RNA", labels = "perturbation", 
                                nt.class.name = "control", logfc.threshold = 0.2, 
                                verbose = FALSE, pval.cutoff = 0.05, 
                                seed = 42, test.use = "wilcox") {
  message("Calculating DEGs iteratively.")
  if (is.null(labels)) stop("Please specify target gene class metadata name")
  
  # Prepare identities and gene list
  Idents(object) <- "con1"
  cells.s <- WhichCells(object, idents = "con1")
  genes <- setdiff(unique(object[[labels]][cells.s, 1]), nt.class.name)
  
  # Loop over genes
  results_list <- lapply(genes, function(gene) {
    if (verbose) message("Processing ", gene)
    calc_deg_for_gene(object, gene, cells.s, labels, nt.class.name, 
                      assay, test.use, logfc.threshold, pval.cutoff)
  })
  
  # Combine
  DE.genes <- setNames(lapply(results_list, `[[`, "genes"), genes)
  results <- setNames(lapply(results_list, `[[`, "result"), genes)
  return(list(genes = DE.genes, results = results))
}

# ---- Parallel version ----
calc_degs_parallel <- function(object, assay = "RNA", labels = "perturbation", 
                               nt.class.name = "control", logfc.threshold = 0.2, 
                               verbose = FALSE, pval.cutoff = 0.05, 
                               seed = 42, test.use = "wilcox",
                               n.cores = 10) {
  library(foreach)
  library(doParallel)
  message("Calculating DEGs in parallel with: ", n.cores, " cores.")
  if (is.null(labels)) stop("Please specify target gene class metadata name")
  
  # Prepare identities and gene list
  Idents(object) <- "con1"
  cells.s <- WhichCells(object, idents = "con1")
  genes <- setdiff(unique(object[[labels]][cells.s, 1]), nt.class.name)
  
  # Setup parallel backend
  registerDoParallel(cores = n.cores)
  
  # Main parallel execution
  results_list <- foreach(gene = genes, .packages = "Seurat", .export = c("calc_deg_for_gene")) %dopar% {
    calc_deg_for_gene(object, gene, cells.s, labels, nt.class.name, 
                      assay, test.use, logfc.threshold, pval.cutoff)
  }
  # Remove NA entries
  results_list <- results_list[!is.na(results_list)]
  # Combine results
  DE.genes <- setNames(lapply(results_list, `[[`, "genes"), genes)
  results <- setNames(lapply(results_list, `[[`, "results"), genes)
  return(list(genes = DE.genes, results = results))
}


calc_degs_parallel_no_log <- function(object, assay = "RNA", labels = "perturbation", 
                                      nt.class.name = "control", logfc.threshold = 0.2, 
                                      verbose = FALSE, pval.cutoff = 0.05, 
                                      seed = 42, test.use = "wilcox",
                                      n.cores = 10) {
  library(foreach)
  library(doParallel)
  
  message("Calculating DEGs in parallel with:.")
  if (is.null(labels)) stop("Please specify target gene class metadata name")
  
  # Prepare identities and gene list
  Idents(object) <- "con1"
  cells.s <- WhichCells(object, idents = "con1")
  genes <- setdiff(unique(object[[labels]][cells.s, 1]), nt.class.name)
  
  # Parallel backend
  cl <- makeCluster(n.cores)
  registerDoParallel(cl)
  handlers(global = T)
  
  # Parallel loop with progress
  with_progress({
    p <- progressor(along = genes)
    results_list <- foreach(gene = genes, .packages = "Seurat", .export = c("calc_deg_for_gene")) %dopar% {
      if (verbose) p(sprintf("Processing %s", gene))
      calc_deg_for_gene(object, gene, cells.s, labels, nt.class.name, 
                        assay, test.use, logfc.threshold, pval.cutoff)
    }
  })
  stopCluster(cl)
  # Remove NA entries
  results_list <- results_list[!is.na(results_list)]
  # Combine
  DE.genes <- setNames(lapply(results_list, `[[`, "genes"), genes)
  results <- setNames(lapply(results_list, `[[`, "result"), genes)
  return(list(genes = DE.genes, results = results))
}


mixscale_pipeline <- function(
    seurat_obj, 
    assay = "originalexp", slot = "data", 
    condition_col = "perturbation", ctrl_col = "control",
    split_by = NULL,
    ndims = 50, nneighbors = 15, max.de.genes = 100,
    min.de.genes = 2, logfc.threshold = 0.2,
    adjust_low_deg = T,
    verbose = F, cache = F,
    clear_mem = F,
    n_cores = 1
) {
  # Check number of perturbations
  if (length(unique(seurat_obj@meta.data[[condition_col]])) < 2) {
    message("No perturbations found in this dataset. Returning object unchanged.")
    seurat_obj$mixscale_score = 0
    return(seurat_obj)
  }
  if ((sum(dim(seurat_obj@assays[[assay]]@scale.data))==0) || (!cache)) {
    # Pre-process data
    message("Pre-processing")
    seurat_obj <- pp_seurat(seurat_obj)
  }
  # Add ctrl label to object
  seurat_obj <- add_ctrl_label(seurat_obj, perturbation_col = condition_col, ctrl_key = ctrl_col)
  # Calculate DEGs and save in obj
  parallel_available <- requireNamespace("doParallel", quietly = T) &&
    requireNamespace("foreach", quietly = T)
  if (n_cores > 1 && parallel_available) {
    DE.genes.result.list <- calc_degs_parallel(
      object = seurat_obj,
      assay = assay,
      labels = condition_col,
      nt.class.name = ctrl_col,
      logfc.threshold = logfc.threshold,
      verbose = verbose,
      n.cores = n_cores,
    )
  } else {
    DE.genes.result.list <- calc_degs_iterative(
      object = seurat_obj,
      assay = assay,
      labels = condition_col,
      nt.class.name = ctrl_col,
      logfc.threshold = logfc.threshold,
      verbose = verbose
    )
  }
  
  if ((!"PRTB" %in% names(seurat_obj@assays) || (!cache))) {
    # Calculate Perturbation signatures 
    seurat_obj <- CalcPerturbSig(
      object = seurat_obj, 
      assay = assay, 
      slot = "data", 
      gd.class = condition_col, 
      nt.cell.class = ctrl_col, 
      reduction = "pca", 
      ndims = ndims, 
      num.neighbors = nneighbors, 
      new.assay.name = "PRTB", 
      split.by = split_by,
      verbose = verbose
    )
  }
  # Remove count assay to clear up memory
  if ((clear_mem) & (!cache)) {
    message("Removing ", assay, " from object to free up memory")
    seurat_obj[[assay]] <- NULL
    gc()
  }
  # Mixscale
  seurat_obj <- RunMixscale(
    object = seurat_obj, 
    DE.gene = DE.genes.result.list[["genes"]],
    assay = "PRTB", 
    slot = "scale.data", 
    labels = condition_col, 
    nt.class.name = ctrl_col, 
    min.de.genes = min.de.genes, 
    logfc.threshold = logfc.threshold,
    de.assay = assay,
    max.de.genes = max.de.genes, 
    new.class.name = "mixscale_score", 
    fine.mode = F, 
    verbose = verbose, 
    split.by = split_by
  )
  summary_stats <- get_mixscale_summary(seurat_obj)
  seurat_obj@misc[['mixscale_summary']] <- summary_stats
  if (adjust_low_deg) {
    # Set perturbations with too little DEGs to 0 instead 1
    seurat_obj <- adjust_low_deg_mixscale(seurat_obj, summary_stats)
  }
  # Save DEGs to object
  seurat_obj@tools[["deg"]] <- DE.genes.result.list
  return(seurat_obj)
}


plot_continuous_umap <- function(
    seurat_obj,
    col = "mixscale_score",
    abs = F,
    log = F
) {
  umap_df <- as.data.frame(Embeddings(seurat_obj, "umap"))
  umap_df$score <- seurat_obj@meta.data[[col]]
  if (abs) {
    umap_df$score <- abs(umap_df$score)
  }
  if (log) {
    umap_df$score <- log1p(umap_df$score)
  }
  
  p <- ggplot(umap_df, aes(x = umap_1, y = umap_2, color = score)) +
    geom_point(size = 2) +
    scale_color_viridis_c() +
    theme_minimal() +
    labs(title = paste0("UMAP colored by ", col))
  return(p)
}

add_ctrl_label <- function(
    seurat_obj,
    perturbation_col = "perturbation",
    ctrl_key = "control"
) {
  seurat_obj$is_ctrl = F
  seurat_obj@meta.data[seurat_obj[[perturbation_col]]==ctrl_key,'is_ctrl'] = T
  return(seurat_obj)
}

mixscale_filter <- function(
    seurat_obj,
    min_dev = 0.5,
    abs = T,
    keep_ctrl = T,
    perturbation_col = "perturbation",
    ctrl_key = "control",
    relabel_col = "adj_perturbation",
    relabel_group = "control",
    subset = F
) {
  score <- seurat_obj$mixscale_score
  if (abs) {
    score <- abs(score)
  }
  # Build mask of cells to keep
  mask <- score > min_dev
  if (keep_ctrl) {
    ctrl_mask <- seurat_obj@meta.data[[perturbation_col]]==ctrl_key
    mask <- mask | ctrl_mask
  }
  # Set label to significant cells
  seurat_obj$is_sign_mixscale <- mask
  # Subset the obj to filtered cells only
  if (subset) {
    message("Subsetting based on filtered mixscale scores")
    filtered <- subset(seurat_obj, subset = is_sign_mixscale)
    return(filtered)
  }
  # Re-label low-response cells as control
  else {
    message("Re-labeling low-signal cells as", relabel_group)
    # Copy perturbation column for all cells
    seurat_obj[[relabel_col]] = seurat_obj[[perturbation_col]]
    # Set low-response cells to control
    seurat_obj@meta.data[!seurat_obj$is_sign_mixscale, relabel_col] = relabel_group
    return(seurat_obj)
  }
}

run_iterative <- function(
    original_obj,
    dataset_col = 'dataset',
    n_ctrl = 10000,
    ...
) {
  # Run pipeline seperately for each dataset in the object
  for (ds in unique(original_obj$dataset)) {
    seurat_obj <- subset(x = original_obj, subset = !!sym(dataset_col) == ds)
    seurat_obj <- subset_ctrl(seurat_obj, n_ctrl = n_ctrl)
    seurat_obj <- mixscale_pipeline(seurat_obj, ...)
  }
}
