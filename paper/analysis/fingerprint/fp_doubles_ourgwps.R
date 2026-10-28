# RNA fingerprinting on the four combinatorial screens with OUR OWN full K562 GWPS dictionary (8,347 fingerprints,
# fitted by ExPert-benchmarks on ExPert's 3,096-gene input panel). Identical to fp_doubles_gwps.R in every respect
# EXCEPT the dictionary: the published precomputed dictionary holds only 4,098 fingerprints, which cost ExPert ~half
# its doubles to coverage rather than to ranking. Nothing is refitted here -- the dictionary is frozen input.
#
# Usage: Rscript fp_doubles_ourgwps.R <export dir> <out dir> <dict.rds> [mc.cores] [input gene list | none] [kind]
# kind = which query perturbations to map: double (default; Fig. 3 H/I) or single (Supplementary Fig. S4A).
# Export = fp_doubles_export.py --keep-genes <ExPert input genes>: counts already restricted to ExPert's input panel.
# Query = every double + all control cells (singles are not scored), SCTransform, then exactly as fp_sciplex_pip.R:
#   features = GWPS dictionary features present in the data (and in the optional input list), control-PC residuals,
#   one PSEUDOBULK per double (sum of its residualised cells), SuSiE on the 100 fingerprints most |cor|-related
#   (package default), calibrated PIPs via the package's own internals.
#   pip.csv   doubles x 4,098 fingerprints (PIP 0 outside the pre-filter)
#   cor.csv   doubles x 4,098 fingerprints: |cor(pseudobulk residual, fingerprint)|
#   check.csv per double: top credible set (stage 1)
# Ranked list = PIP desc, ties by |cor|; candidates are restricted afterwards (95_fp_doubles_gwps_eval.py).
suppressMessages({ library(Seurat); library(Matrix); library(RNAFingerprinting); library(RNAFingerprintingData)
                   library(irlba); library(pbmcapply) })
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; out <- args[2]; dict_path <- args[3]
cores <- if (length(args) > 3) as.integer(args[4]) else 1L
keep_genes <- if (length(args) > 4 && args[5] != "none") readLines(args[5]) else NULL
kind <- if (length(args) > 5) args[6] else "double"
stopifnot(file.exists(dict_path))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
options(future.globals.maxSize = 40 * 1024^3)
t0 <- Sys.time(); say <- function(...) cat(sprintf("[%6.0fs] ", as.numeric(Sys.time() - t0, units = "secs")), ..., "\n")
susie_plus <- RNAFingerprinting:::susie_plus
calibrate <- RNAFingerprinting:::calibrate
get_cs <- RNAFingerprinting:::get_credible_sets

genes <- readLines(file.path(inp, "genes.txt")); cells <- readLines(file.path(inp, "barcodes.txt"))
rb <- function(f) readBin(file.path(inp, f), "integer", n = file.size(file.path(inp, f)) / 4, size = 4, endian = "little")
X <- sparseMatrix(i = rb("indices.bin") + 1L, p = rb("indptr.bin"), x = as.numeric(rb("data.bin")),
                  dims = c(length(genes), length(cells)), dimnames = list(genes, cells))
meta <- read.csv(file.path(inp, "meta.csv"), row.names = 1, stringsAsFactors = FALSE)
stopifnot(identical(rownames(meta), cells), "Control" %in% meta$condition)
use <- meta$kind %in% c(kind, "control")
X <- X[, use]; meta <- meta[use, ]
obj <- CreateSeuratObject(counts = X, meta.data = meta)
rm(X); gc()
say("query", ncol(obj), "cells x", nrow(obj), "genes;", length(unique(meta$condition[meta$kind == kind])), paste0(kind, "s,"),
    sum(meta$kind == "control"), "control cells")
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
obj <- SCTransform(obj, return.only.var.genes = FALSE, verbose = FALSE)
say("SCTransform done")
gwps <- readRDS(dict_path)   # our OWN full-GWPS fit (8,347 fingerprints)

set.seed(1448145)
data <- GetAssayData(obj, assay = "SCT", layer = "scale.data")
fp <- gwps$fingerprints
feat <- intersect(rownames(fp$Lambdas), rownames(data))
writeLines(feat, file.path(out, "features_all.txt"))
say("dictionary features present in the data:", length(feat), "of", nrow(fp$Lambdas))
if (!is.null(keep_genes)) {
  feat <- intersect(feat, keep_genes)
  say("restricted to the supplied input genes:", length(feat), "(list had", length(keep_genes), "genes)")
}
writeLines(feat, file.path(out, "features_used.txt"))
Lambdas <- fp$Lambdas[feat, ]
dists <- lapply(fp$dists, function(d) d[feat, ])
cond <- as.character(obj$condition); cond[cond == "Control"] <- "control"; names(cond) <- colnames(obj)
rm(obj); gc()

ctrlm <- data[feat, cond == "control"]
mus <- rowMeans(ctrlm)
PCs <- prcomp_irlba(t(ctrlm), n = 30)
w <- data[feat, cond != "control"]
w.res <- w - sweep(PCs$rotation %*% crossprod(PCs$rotation, sweep(w, 1, mus)), 1, mus, "+")
lab <- cond[colnames(w.res)]
idx <- split(seq_len(ncol(w.res)), lab)
w.res <- sapply(unique(lab), function(id) rowSums(w.res[, idx[[id]], drop = FALSE]))   # one pseudobulk per double
rm(data, ctrlm, w); gc()
newLambdas <- lapply(1:50, function(x) {
  nl <- sapply(1:ncol(Lambdas), function(k) rnorm(length(feat), dists[[colnames(Lambdas)[k]]][, 1],
                                                  dists[[colnames(Lambdas)[k]]][, 2]))
  colnames(nl) <- colnames(Lambdas); nl })
Sxx <- lapply(1:50, function(x) { s <- colSums(newLambdas[[x]]^2); names(s) <- colnames(Lambdas); s })
store_cor <- cor(Lambdas)
say("doubles", ncol(w.res), "; features", length(feat), "; mapping with", cores, "cores")

res <- pbmclapply(seq_len(ncol(w.res)), function(x) {
  crs <- c(abs(cor(w.res[, x], Lambdas))); names(crs) <- colnames(Lambdas)
  filter <- colnames(Lambdas)[order(crs, decreasing = TRUE)[1:100]]
  pip <- setNames(numeric(ncol(Lambdas)), colnames(Lambdas))
  top <- "unassigned"
  r <- try({
    mod <- susie_plus(X = Lambdas[, filter], y = w.res[, x], S = NULL, varS = NULL, standardize = FALSE, L = 10,
                      intercept = FALSE, max_iter = 5000, estimate_prior_method = "optim")
    rownames(mod$alpha) <- rownames(mod$mu) <- names(mod$lbf) <- names(mod$V) <- paste0("cs", 1:nrow(mod$alpha))
    keep <- names(mod$lbf)[which(mod$lbf >= 0.1 * max(mod$lbf) & mod$V > 0)]
    if (length(keep) > 0 && length(get_cs(mod$alpha[keep, , drop = FALSE], store_cor, 0.95)) > 0) {
      al <- calibrate(w.res[, x], mod, keep, Lambdas[, filter], filter, newLambdas, Sxx)
      pip[colnames(al)] <- 1 - apply(1 - al, 2, prod)
      cs <- get_cs(al, store_cor, 0.95)
      if (length(cs) > 0) top <- paste(cs[[which.max(mod$lbf[names(cs)])]], collapse = ",")
    }
  }, silent = TRUE)
  if (inherits(r, "try-error")) top <- paste("error:", conditionMessage(attr(r, "condition")))
  list(pip = pip, cor = crs, top = top)
}, mc.cores = cores)

g <- colnames(w.res)
P <- t(sapply(res, `[[`, "pip")); C <- t(sapply(res, `[[`, "cor"))
rownames(P) <- rownames(C) <- g
write.csv(signif(P, 5), file.path(out, "pip.csv"))
write.csv(signif(C, 5), file.path(out, "cor.csv"))
write.csv(data.frame(group = g, top_credible_set_stage1 = sapply(res, `[[`, "top")), file.path(out, "check.csv"),
          row.names = FALSE)
say("all done:", nrow(P), "doubles x", ncol(P), "fingerprints")
