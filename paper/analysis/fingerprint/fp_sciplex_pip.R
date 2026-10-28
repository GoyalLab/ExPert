# Copied unchanged from the collaborator's fingerprint_bench/code/fp_sciplex_pip.R (2026-10-25) so the
# per-cell-line sci-Plex3 runs use exactly the same mapping as the published K562 run.
#
# RNA fingerprinting on sci-Plex3 K562, re-run to KEEP its per-gene probabilities, so fingerprinting gets a full
# ranked list (a line over answer length) instead of only its credible sets.
#
# Usage: Rscript fp_sciplex_pip.R <export dir> <out dir> [mc.cores] [input gene list | none] [control, default Vehicle]
# Optional 4th arg: a text file of gene symbols (one per line); fingerprinting's features are then restricted to
# them -- used to give fingerprinting the SAME input genes as ExPert (its 3,096 model genes).
# Same input, preprocessing and dictionary as fp_sciplex.R (SCTransform, precomputed GWPS, group level, Vehicle
# control, seed 1448145). The package's internal classify() keeps only credible sets; this re-runs its first stage
# with the package's own internals (RNAFingerprinting:::susie_plus / :::calibrate / :::get_credible_sets) and saves:
#   pip.csv   groups x fingerprints: PIP = 1 - prod over the kept effects (1 - calibrated alpha). Only the 100
#             fingerprints most correlated with a group enter the regression (the package default, top = 100);
#             every other fingerprint has PIP 0.
#   cor.csv   groups x fingerprints: |cor(group residual, fingerprint)|, the package's own pre-filter score
#   check.csv per group: top credible set recomputed here (to compare with summary_gwps.csv)
# Ranked list for a group = PIP descending, ties (incl. all PIP 0) broken by |cor|.
# Not reproduced: the package's optional second stage that proposes JOINT fingerprints when credible sets correlate;
# it changes only those groups' sets (check.csv vs summary_gwps.csv shows how many).
suppressMessages({ library(Seurat); library(Matrix); library(RNAFingerprinting); library(RNAFingerprintingData)
                   library(irlba); library(pbmcapply) })
args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]; out <- args[2]; cores <- if (length(args) > 2) as.integer(args[3]) else 1L
keep_genes <- if (length(args) > 3 && args[4] != "none") readLines(args[4]) else NULL
ctrl <- if (length(args) > 4) args[5] else "Vehicle"   # FlexPlex HDAC screen: "NT"
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
meta <- read.csv(file.path(inp, "meta.csv"), row.names = 1)
stopifnot(identical(rownames(meta), cells), ctrl %in% meta$condition)
obj <- CreateSeuratObject(counts = X, meta.data = meta)
rm(X); gc()
say("object", ncol(obj), "cells x", nrow(obj), "genes")
obj <- SCTransform(obj, return.only.var.genes = FALSE, verbose = FALSE)
say("SCTransform done")
gwps <- LoadPrecomputedDictionary(name = "gwps", gene_key = "SYMBOL")

# ---- as FingerprintCells() ----
set.seed(1448145)
data <- GetAssayData(obj, assay = "SCT", layer = "scale.data")
fp <- gwps$fingerprints
feat <- intersect(rownames(fp$Lambdas), rownames(data))
say("dictionary features present in the data:", length(feat), "of", nrow(fp$Lambdas))
writeLines(feat, file.path(out, "features_all.txt"))
if (!is.null(keep_genes)) {
  feat <- intersect(feat, keep_genes)
  say("restricted to the supplied input genes:", length(feat), "(list had", length(keep_genes), "genes)")
}
writeLines(feat, file.path(out, "features_used.txt"))
Lambdas <- fp$Lambdas[feat, ]
dists <- lapply(fp$dists, function(d) d[feat, ])
cond <- as.character(obj$condition); cond[cond == ctrl] <- "control"; names(cond) <- colnames(obj)
rm(obj); gc()
say("features", length(feat), "of", nrow(fp$Lambdas))

# ---- as classify(group_level = TRUE, top = 100), first stage ----
ctrlm <- data[feat, cond == "control"]
mus <- rowMeans(ctrlm)
PCs <- prcomp_irlba(t(ctrlm), n = 30)
w <- data[feat, cond != "control"]
w.res <- w - sweep(PCs$rotation %*% crossprod(PCs$rotation, sweep(w, 1, mus)), 1, mus, "+")
lab <- cond[colnames(w.res)]
idx <- split(seq_len(ncol(w.res)), lab)
w.res <- sapply(unique(lab), function(id) rowSums(w.res[, idx[[id]], drop = FALSE]))
rm(data, ctrlm, w); gc()
newLambdas <- lapply(1:50, function(x) {
  nl <- sapply(1:ncol(Lambdas), function(k) rnorm(length(feat), dists[[colnames(Lambdas)[k]]][, 1],
                                                  dists[[colnames(Lambdas)[k]]][, 2]))
  colnames(nl) <- colnames(Lambdas); nl })
Sxx <- lapply(1:50, function(x) { s <- colSums(newLambdas[[x]]^2); names(s) <- colnames(Lambdas); s })
store_cor <- cor(Lambdas)
say("groups", ncol(w.res), "; mapping with", cores, "cores")

res <- pbmclapply(seq_len(ncol(w.res)), function(x) {
  crs <- c(abs(cor(w.res[, x], Lambdas))); names(crs) <- colnames(Lambdas)
  filter <- colnames(Lambdas)[order(crs, decreasing = TRUE)[1:100]]
  pip <- setNames(numeric(ncol(Lambdas)), colnames(Lambdas))
  mod <- susie_plus(X = Lambdas[, filter], y = w.res[, x], S = NULL, varS = NULL, standardize = FALSE, L = 10,
                    intercept = FALSE, max_iter = 5000, estimate_prior_method = "optim")
  rownames(mod$alpha) <- rownames(mod$mu) <- names(mod$lbf) <- names(mod$V) <- paste0("cs", 1:nrow(mod$alpha))
  keep <- names(mod$lbf)[which(mod$lbf >= 0.1 * max(mod$lbf) & mod$V > 0)]
  top <- "unassigned"
  if (length(keep) > 0 && length(get_cs(mod$alpha[keep, , drop = FALSE], store_cor, 0.95)) > 0) {
    al <- calibrate(w.res[, x], mod, keep, Lambdas[, filter], filter, newLambdas, Sxx)
    pip[colnames(al)] <- 1 - apply(1 - al, 2, prod)
    cs <- get_cs(al, store_cor, 0.95)
    if (length(cs) > 0) top <- paste(cs[[which.max(mod$lbf[names(cs)])]], collapse = ",")
  }
  list(pip = pip, cor = crs, top = top)
}, mc.cores = cores)

g <- colnames(w.res)
P <- t(sapply(res, `[[`, "pip")); C <- t(sapply(res, `[[`, "cor"))
rownames(P) <- rownames(C) <- g
write.csv(signif(P, 5), file.path(out, "pip.csv"))
write.csv(signif(C, 5), file.path(out, "cor.csv"))
write.csv(data.frame(group = g, top_credible_set_stage1 = sapply(res, `[[`, "top")), file.path(out, "check.csv"),
          row.names = FALSE)
say("all done:", nrow(P), "groups x", ncol(P), "fingerprints")
