library(package="GenomicFeatures")
library("biomaRt")
library(dplyr)

save_view <- function(view, name, directory) {
  if (!dir.exists(directory)){
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  }
  write.csv(x = view, file = paste(directory,name,".csv", sep = ""), row.names = TRUE)
}


single_to_multi_view <- function(whole,label_pos) {
  ids <- rownames(whole)
  labels <- whole[,label_pos]
  names(labels) <- ids
  features <- whole[, -label_pos]
  res <- list(features = features, labels = labels)
  return (res)
}


load_tumor <- function(tumor) {
  print(paste("loading from",tumor))
  mrna <- suppressMessages(curatedTCGAData::curatedTCGAData(diseaseCode = tumor, assays = "RNASeq2GeneNorm", dry.run = FALSE, version = '2.0.1'))
  TCGAutils::sampleTables(mrna)
  mrna <- TCGAutils::splitAssays(mrna, "01")
  mrna <- Reduce("cbind", lapply(experiments(mrna), function(x) assays(x)[[1]]))
  print(paste("mrna samples",ncol(mrna)))
  print(paste("genes",nrow(mrna)))
  return (mrna)
}


load_tumours <- function(tumours) {
  n <- length(tumours)
  mrnas <- vector("list", n)
  for (i in 1:n) {
    mrnas[[i]] <- load_tumor(tumor = tumours[i])
  }
  res <- do.call(cbind, mrnas)
  print(paste("total samples",ncol(res)))
  print(paste("total genes",nrow(res)))
  return (res)
}


reduce_to_common_patient_ids_and_save <- function(
    data, common_ids, directory, verbose_name = "single view data", chars_for_id = 12) {
  print(paste("Reducing",verbose_name))
  print(paste("Samples before reducing:",ncol(data)))
  res <- data[,match(common_ids, substr(colnames(data), start=1, stop=chars_for_id))]
  data <- NULL
  print(paste("Samples after reducing:",ncol(res)))
  print(paste("Number of features:",nrow(res)))
  colnames(res) <- common_ids
  res <- t(res)
  print("Saving to file.")
  save_view(view = res,name = verbose_name,directory = directory)
  return (res)
}


samples_uq <- function(data, include_all_zero_genes = TRUE, include_zero_genes = TRUE) {
  if (include_all_zero_genes) {
    filtered_data <- data
  } else {
    sumatot <- colSums(data)
    counts0 <- which(sumatot == 0)
    if (length(counts0) > 0) {
      filtered_data <- data[,-counts0]
    } else {
      filtered_data <- data
    }
  }
  if (include_zero_genes) {
    sample_uq <- apply(filtered_data, 1, function(x){quantile(x, 0.75)})
  } else {
    sample_uq <- apply(filtered_data, 1, function(x){quantile(x[x>0], 0.75)})
  }
  return (sample_uq)
}


# Data has samples on rows and genes on columns.
upper_quartile_normalization <- function(
  data, include_all_zero_genes = TRUE, include_zero_genes = TRUE, verbose = FALSE) {
  sample_uq <- samples_uq(
    data=data, include_all_zero_genes = include_all_zero_genes, include_zero_genes = include_zero_genes)
  if (verbose) {
    print("sample_uq before normalization:")
    print(summary(sample_uq))
  }
  uqn <- sweep(data, 1, sample_uq, FUN = '/')
  if (verbose) {
    sample_uq <- samples_uq(
    data=uqn, include_all_zero_genes = include_all_zero_genes, include_zero_genes = include_zero_genes)
    print("sample_uq after normalization:")
    print(summary(sample_uq))
  }
  return (uqn)
}


# Assumes that rpm columns are genes in the gene symbols notation.
# Uses gene lengths computed from gencode.v40.annotation
# Release 40 (GRCh38.p13) downloaded from https://www.gencodegenes.org/human/
rpm_to_tpm <- function(rpm, verbose = FALSE) {
  print("Computing gene lengths")
  txdb <- makeTxDbFromGFF("work/gse138042/raw/gencode.v40.annotation.gtf",format="gtf")
  # Release 40 (GRCh38.p13) downloaded from https://www.gencodegenes.org/human/
  exons.list.per.gene <- exonsBy(txdb,by="gene")
  exonic.gene.sizes <- as.data.frame(sum(width(reduce(exons.list.per.gene))))
  colnames(exonic.gene.sizes)[1] <- "kbases"
  exonic.gene.sizes$ensembl_gene_id <- sub("[.][0-9]*","",rownames(exonic.gene.sizes))

  mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
  gene_IDs <- getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id","hgnc_symbol"),
                values = exonic.gene.sizes$ensembl_gene_id, mart= mart)

  gene_sizes <- left_join(exonic.gene.sizes, gene_IDs, by = c("ensembl_gene_id"="ensembl_gene_id"))
  gene_sizes <- gene_sizes[!is.na(gene_sizes$hgnc_symbol),]
  gene_sizes <- gene_sizes[gene_sizes$hgnc_symbol != "",]

  common_genes <- intersect(gene_sizes$hgnc_symbol,colnames(rpm))
  gene_sizes <- gene_sizes[gene_sizes$hgnc_symbol %in% common_genes,]
  gene_sizes <- gene_sizes[with(gene_sizes, order(-kbases)), ]
  gene_sizes <- gene_sizes[!duplicated(gene_sizes$hgnc_symbol),]

  rpm <- rpm[, colnames(rpm) %in% common_genes]

  sorted_gene_sizes <- numeric(length = ncol(rpm))
  cols <- colnames(rpm)
  for (i in 1:ncol(rpm)) {
    sorted_gene_sizes[i] <- gene_sizes[gene_sizes$hgnc_symbol==cols[i],]$kbases[1]
  }

  print("Computing TPM using gene lengths.")
  genes_divided_by_length <- sweep(1000 * rpm, 2, sorted_gene_sizes, FUN = '/')
  sample_sums <- rowSums(genes_divided_by_length) / 1000000
  tpm <- sweep(genes_divided_by_length, 1, sample_sums, FUN = '/')
  if (verbose) {
    print("Sample sums after tpm:")
    print(summary(rowSums(tpm)))
  }
  return (tpm)
}


rpkm_to_tpm <- function(rpkm) {
  sample_sums <- rowSums(rpkm)
  tpm <- sweep(1000000 * rpkm, 1, sample_sums, FUN = '/')
  return (tpm)
}


rpm_to_tpm_uq <- function(rpm, include_all_zero_genes = FALSE, include_zero_genes = FALSE) {
  tpm <- rpm_to_tpm(rpm=rpm)
  print("Applying upper quartile normalization.")
  tpm_uq <- upper_quartile_normalization(
    data = tpm, include_all_zero_genes = include_all_zero_genes, include_zero_genes = include_zero_genes)
  return (tpm_uq)
}


plot_density <- function(gene_values, directory, name, to_sum = 1.0) {
  if (!dir.exists(directory)){
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  }
  png(file=paste(directory,name,".png", sep = ""), height = 1024, width = 1024)
  plot(density(log2(gene_values+to_sum)))
  dev.off()
}


drop_many_zeros_cols <- function(data, prevalence) {
  print("Removing columns with many zeros.")
  print("Columns before dropping")
  print(ncol(data))
  column_cut_off <- prevalence * nrow(data)
  i <- colSums(data < 0.5, na.rm=TRUE) < column_cut_off
  res <- data[, i, drop=FALSE]
  print("Columns after dropping")
  print(ncol(res))
  return (res)
}


drop_many_zeros_rows <- function(data, prevalence) {
  print("Removing rows with many zeros.")
  print("Rows before dropping")
  print(nrow(data))
  cut_off <- prevalence * ncol(data)
  i <- rowSums(data < 0.5, na.rm=TRUE) < cut_off
  res <- data[i, , drop=FALSE]
  print("Rows after dropping")
  print(nrow(res))
  return (res)
}


drop_by_min_nonzero_rows <- function(data, min_non_zero) {
  print("Removing rows with many zeros.")
  print("Rows before dropping")
  print(nrow(data))
  i <- rowSums(data > 0.5, na.rm=TRUE) > min_non_zero
  res <- data[i, , drop=FALSE]
  print("Rows after dropping")
  print(nrow(res))
  return (res)
}