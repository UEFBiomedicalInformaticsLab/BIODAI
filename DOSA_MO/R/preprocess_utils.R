library(package="GenomicFeatures")
library("biomaRt")
library(dplyr)


TUMOR_SAMPLE_CODE <- "01"


# Does not transpose
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


# If code not present returns NULL
load_tcga_data_one_assay <- function(tumor, sample_code, assays = "RNASeq2GeneNorm") {
  print(paste("loading from",tumor,"with sample code",sample_code))
  data <- suppressMessages(curatedTCGAData::curatedTCGAData(diseaseCode = tumor, assays = assays, dry.run = FALSE, version = '2.0.1'))
  tables <- TCGAutils::sampleTables(data)
  print(tables)
  if (length(tables) > 0 && sample_code %in% names(tables[[1]])) {
    data <- TCGAutils::splitAssays(data, sample_code)
    data <- Reduce("cbind", lapply(experiments(data), function(x) assays(x)[[1]]))
    print(paste("samples",ncol(data)))
    print(paste("features",nrow(data)))
    return (data)
  } else {
    print("Code not present!")
    return (NULL)
  }
}


# If code not present returns NULL
# Returns a list of views with only common ids.
load_tcga_data_multiple_assays <- function(tumor, sample_code, assays = "RNASeq2GeneNorm", chars_for_id = 12) {
  n_assays <- length(assays)
  assay_data <- NULL
  for (i in 1:n_assays) {
    assay <- assays[i]
    assay_data[[i]] <- load_tcga_data_one_assay(tumor = tumor, sample_code = sample_code, assays = assay)
  }
  return (reduce_to_common_patient_ids(assay_data, chars_for_id = chars_for_id))
}


# If code not present returns NULL
# Returns a list of views with only common ids.
load_tumor_multiple_assays <- function(tumor, assays = "RNASeq2GeneNorm", chars_for_id = 12) {
  return (load_tcga_data_multiple_assays(tumor = tumor, sample_code = TUMOR_SAMPLE_CODE, assays=assays,
                                         chars_for_id = chars_for_id))
}


# Merges by row names, keeping all rows and columns and filling missings with NA. No "row.names" column is added.
# Dataframes and matrices are accepted as elements of the list. NULL elements of the list are skipped.
clean_merge <- function(data_list) {
  require(tidyverse)
  datas <- compact(data_list)
  for (i in 1:(length(datas))) {
    datas[[i]] <- as.data.frame(datas[[i]])
  }
  res <- datas[[1]]
  n_data <- length(datas)
  if (n_data > 1) {
    for (i in 2:(length(datas))) {
      res <- merge(x=res, y=datas[[i]], by = "row.names", all = TRUE)
      rownames(res) <- res[[1]]
      res <- res[-1]
    }
  }
  return (res)
}


load_multiple_tcga_data <- function(tumours, sample_codes, assays = "RNASeq2GeneNorm") {
  n <- length(tumours)
  n_codes <- length(sample_codes)
  mrnas <- vector("list", n*n_codes)
  for (i in 1:n) {
    for (j in 1:n_codes) {
      mrnas[[i*n_codes+j]] <- load_tcga_data_one_assay(tumor = tumours[i], sample_code = sample_codes[j], assays = assays)
    }
  }
  res <- clean_merge(mrnas)
  print(paste("total samples",ncol(res)))
  print(paste("total features",nrow(res)))
  return (res)
}


load_tumor <- function(tumor, assays = "RNASeq2GeneNorm") {
  return (load_tcga_data_one_assay(tumor, TUMOR_SAMPLE_CODE, assays = assays))
}


load_normal <- function(tumor, assays = "RNASeq2GeneNorm") {
  return (load_tcga_data_one_assay(tumor, "11", assays = assays))
}


load_tumours_and_normal <- function(tissues, assays = "RNASeq2GeneNorm") {
  return (load_multiple_tcga_data(tumours = tissues, sample_codes = c(TUMOR_SAMPLE_CODE, "11"), assays = assays))
}


load_normals <- function(tumours, assays = "RNASeq2GeneNorm") {
  return (load_multiple_tcga_data(tumours = tumours, sample_codes = "11", assays = assays))
}


# Samples in columns and genes in rows.
load_tumours <- function(tumours, assays = "RNASeq2GeneNorm") {
  return (load_multiple_tcga_data(tumours = tumours, sample_codes = TUMOR_SAMPLE_CODE, assays = assays))
}


# Data has samples on columns and features on rows.
just_save <- function(
    data, directory, verbose_name = "single view data") {
  print(paste("Samples:",ncol(data)))
  print(paste("Number of features:",nrow(data)))
  data <- t(data)
  print("Saving to file.")
  save_view(view = data,name = verbose_name,directory = directory)
  return (data)
}


# NAs are ignored for computing the variance
remove_zero_variance_cols <- function(dat) {
    out <- lapply(dat, function(x) length(unique(x[!is.na(x)])) )
    want <- which(!out > 1)
    return (dat[,-unlist(want)])
}


# NAs are ignored for computing the variance
remove_zero_variance_rows <- function(dat) {
    out <- apply(dat, 1, function(x) length(unique(x[!is.na(x)])) )
    want <- which(!(out > 1))
    if (length(want > 0)) {
      return (dat[-unlist(want),])
    } else {
      return (dat)
    }
}


# Assumes ids are on columns.
# Returned ids are sorted alphabetically.
common_ids <- function(views) {
  n_views <- length(views)
  if (n_views == 0) {
    return (NULL)
  }
  res <- colnames(views[[1]])
  if (n_views > 1) {
    for (i in 2:n_views) {
      res <- intersect(res,colnames(views[[i]]))
    }
  }
  return (sort(res))
}


reduce_to_common_patient_ids <- function(views, chars_for_id = 12) {
  res <- NULL
  for (i in seq_along(views)) {
    v <- views[[i]]
    colnames(v) <- substr(colnames(v), 1, stop=chars_for_id)
    res[[i]] <- v
  }
  common_ids <- common_ids(res)
  for (i in seq_along(res)) {
    res[[i]] <- res[[i]][, common_ids]
  }
  return (res)
}


reduce_to_common_patient_ids_and_save <- function(
    data, common_ids, directory, verbose_name = "single view data", chars_for_id = 12, zero_variance_filter=TRUE) {
  print(paste("Reducing",verbose_name))
  print(paste("Samples before reducing:",ncol(data)))
  res <- data[,match(common_ids, substr(colnames(data), start=1, stop=chars_for_id))]
  data <- NULL
  print(paste("Samples after reducing:",ncol(res)))
  print(paste("Number of features:",nrow(res)))
  colnames(res) <- common_ids
  if (zero_variance_filter) {
    res <- remove_zero_variance_rows(res)
    print(paste("Number of features after zero variance filter:",nrow(res)))
  }
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


load_tumours_and_clean <- function(tumours, assays = "RNASeq2GeneNorm", chars_for_id = 12) {
  tumour_data <- load_tumours(tumours = tumours, assays = assays)
  colnames(tumour_data) <- strtrim(colnames(tumour_data), chars_for_id)
  colnames(tumour_data) <- paste(colnames(tumour_data),"T",sep = "")  # Add an T to avoid clashes with tumour samples.
  print(paste("Tumour samples for tumours ", toString(tumours), ", assays ", assays, ": ", ncol(tumour_data), sep=""))
  print(paste("Features:", nrow(tumour_data)))
  return (tumour_data)
}


load_normals_and_clean <- function(tumours, assays = "RNASeq2GeneNorm", chars_for_id = 12) {
  normal <- load_normals(tumours = tumours, assays = assays)
  colnames(normal) <- strtrim(colnames(normal), chars_for_id)
  colnames(normal) <- paste(colnames(normal),"N",sep = "")  # Add an N to avoid clashes with tumour samples.
  print(paste("Normal samples for tumours ", toString(tumours), ", assays ", assays, ": ", ncol(normal), sep=""))
  print(paste("Features:", nrow(normal)))
  return (normal)
}


select_samples <- function(x, samples_ids) {
  x <- x[, colnames(x) %in% samples_ids]
  return (x)
}


select_features <- function(x, feature_names) {
  x <- x[rownames(x) %in% feature_names, ]
  return (x)
}


# Samples on columns and features on rows.
# Reduces the features to the common ones.
concat_samples <- function(x, y) {
  common_features <- intersect(rownames(x), rownames(y))
  print(paste("Features in common:", length(common_features)))
  x <- select_features(x=x, feature_names = common_features)
  y <- select_features(x=y, feature_names = common_features)
  xy <- cbind(x,y)
  print(paste("Samples after concatenation:", ncol(xy)))
  return (xy)
}


intersect_all <- function(all) {
  return (Reduce(intersect, all))
}


analyze_views <- function(tumor, assays = "RNASeq2GeneNorm", chars_for_id = 12, labels = NULL) {
  data <- load_tumor_multiple_assays(tumor=tumor, assays = assays, chars_for_id = CHARS_FOR_ID)
  print(paste("Views:",as.character(assays)))
  print(paste("Number of samples:",ncol(data[[1]])))
}

# Data from supplementary of
# Liu et al.
# An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics,
# Cell, Volume 173, Issue 2, 2018, Pages 400-416.e11, ISSN 0092-8674,
# https://doi.org/10.1016/j.cell.2018.02.052.
# (https://www.sciencedirect.com/science/article/pii/S0092867418302290)
load_survival_data <- function(tumours, event = "OS") {
  sorvival_time_column <- paste(event, ".time", sep="")
  survival_data <- suppressWarnings(read_excel("work/TCGA survival/1-s2.0-S0092867418302290-mmc1.xlsx"))
  survival_data <- as.data.frame(survival_data)
  rownames(survival_data) <- survival_data$bcr_patient_barcode
  survival_data <- survival_data[survival_data$type %in% tumours, ]
  survival_data <- survival_data[ , names(survival_data) %in% c(event, sorvival_time_column)]
  # print(paste("Original column names:", colnames(survival_data)))
  colnames(survival_data) <- c("Event", "Time")
  survival_data <- na.omit(survival_data)
  return (survival_data)
}


# Includes all survival types and all additional columns.
# Data from supplementary of
# Liu et al.
# An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics,
# Cell, Volume 173, Issue 2, 2018, Pages 400-416.e11, ISSN 0092-8674,
# https://doi.org/10.1016/j.cell.2018.02.052.
# (https://www.sciencedirect.com/science/article/pii/S0092867418302290)
load_whole_survival_data<- function(tumours) {
  survival_data <- suppressWarnings(read_excel("work/TCGA survival/1-s2.0-S0092867418302290-mmc1.xlsx"))
  survival_data <- as.data.frame(survival_data)
  rownames(survival_data) <- survival_data$bcr_patient_barcode
  survival_data <- survival_data[survival_data$type %in% tumours, ]
  return (survival_data)
}


save_labels_and_data_common_ids <- function(
  directory, data, labels, labels_name = "outcome",
  data_name = "single view data",
  chars_for_id = 12, zero_variance_filter=TRUE) {
  common_ids <- intersect(
    substr(rownames(labels), 1, stop=chars_for_id),
    substr(colnames(data), 1, stop=chars_for_id))
  labels <- labels[match(common_ids, substr(rownames(labels), start=1, stop=chars_for_id)), ]
  rownames(labels) <- common_ids
  save_view(labels, name=labels_name, directory = directory)
  reduce_to_common_patient_ids_and_save(
  data, common_ids, directory = directory, verbose_name = data_name, chars_for_id = chars_for_id,
  zero_variance_filter = zero_variance_filter)
}