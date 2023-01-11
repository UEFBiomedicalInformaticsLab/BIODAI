# Author: Teemu Rintala

get_selected_multi_omics_data <- function(selected_cancers,
                                          data_path, 
                                          dissimilarities = TRUE, 
                                          gene_map = FALSE, 
                                          return_unmapped = FALSE,
                                          ...) {
  dat_list <- list()
  for (i in selected_cancers) {
    dat <- get_multi_omics_data(i, 
                                data_path = paste0(data_path, i, "/"), 
                                dissimilarities = dissimilarities, 
                                gene_map = gene_map, 
                                ...)
    dat_list[[i]] <- dat
  }
  f_comb <- function(x, y) {
    res <- list()
    if(!is.null(x$sample_info)) res$sample_info <- rbind(x$sample_info, y$sample_info)
    if(!is.null(x$survival)) res$survival <- rbind(x$survival, y$survival)
    if(!is.null(x$subtype)) res$subtype <- COPS::rbind_fill(x$subtype, y$subtype)
    if(!is.null(x$omics[["mrna"]])) res$omics[["mrna"]] <- COPS::cbind_fill(x$omics[["mrna"]], y$omics[["mrna"]])
    if(!is.null(x$omics[["mirna"]])) res$omics[["mirna"]] <- COPS::cbind_fill(x$omics[["mirna"]], y$omics[["mirna"]])
    if(!is.null(x$omics[["mirna_gene"]])) res$omics[["mirna_gene"]] <- COPS::cbind_fill(x$omics[["mirna_gene"]], y$omics[["mirna_gene"]])
    if(!is.null(x$omics[["meth"]])) res$omics[["meth"]] <- COPS::cbind_fill(x$omics[["meth"]], y$omics[["meth"]])
    if(!is.null(x$omics[["meth_gene"]])) res$omics[["meth_gene"]] <- COPS::cbind_fill(x$omics[["meth_gene"]], y$omics[["meth_gene"]])
    if(!is.null(x$omics[["cnv"]])) res$omics[["cnv"]] <- COPS::cbind_fill(x$omics[["cnv"]], y$omics[["cnv"]])
    if(!is.null(x$differential_features)) res$differential_features <- c(x$differential_features, y$differential_features)
    return(res)
  }
  
  out <- Reduce(f_comb, dat_list)
  
  if (dissimilarities) {
    out$diss <- list()
    if (!is.null(out$omics[["mrna"]])) {
      out$diss[["mrna"]] <- COPS::clustering_dissimilarity_from_data(t(out$omics[["mrna"]]))
    }
    if (!is.null(out$omics[["mirna"]])) {
      out$diss[["mirna"]] <- COPS::clustering_dissimilarity_from_data(t(out$omics[["mirna"]]))
      if (gene_map & !return_unmapped) out$omics[["mirna"]] <- NULL
    }
    if (!is.null(out$omics[["meth"]])) {
      out$diss[["meth"]] <- COPS::clustering_dissimilarity_from_data(t(out$omics[["meth"]]))
      if (gene_map & !return_unmapped) out$omics[["meth"]] <- NULL
    }
  }
  
  if (!is.null(out$omics[["meth"]])) {
    out$omics[["meth"]] <- out$omics[["meth"]][!apply(is.na(out$omics[["meth"]]), 1, any),]
  }
  if (!is.null(out$omics[["meth_gene"]])) {
    out$omics[["meth_gene"]] <- out$omics[["meth_gene"]][!apply(is.na(out$omics[["meth_gene"]]), 1, any),]
  }
  
  return(out)
}

read_csv_with_retry <- function(path, 
                                header = TRUE, 
                                row.names = 1, 
                                max_retry = 10, 
                                sleep_time = 1) {
  for (i in 1:max_retry) {
    out <- tryCatch(read.csv(path, header = TRUE, row.names = 1),
                    error = function(e) return(NULL))
    if (!is.null(out)) break else Sys.sleep(sleep_time)
  }
  if (is.null(out)) stop("Reading data failed!")
  return(out)
}

get_multi_omics_data <- function(cancer_name, 
                                 data_path, 
                                 sample_info = TRUE, 
                                 survival = TRUE,
                                 mrna = TRUE,
                                 mirna = TRUE,
                                 meth = TRUE, 
                                 cnv = FALSE,
                                 subtype = FALSE,
                                 gene_map = FALSE,
                                 dissimilarities = TRUE,
                                 impute = TRUE, 
                                 max_retry = 10, 
                                 intersect_samples = TRUE,
                                 tumour_only = TRUE,
                                 differential = FALSE,
                                 log_fold_change_threshold = 0.5,
                                 p_value_threshold = 0.01,
                                 p_adjust_method = "BH",
                                 apply_log = FALSE) {
  out <- list()
  out$omics <- list()
  if (mrna) {
    out$omics[["mrna"]] <- read_csv_with_retry(paste0(data_path, "mrna.csv.gz"), 
                                          header = TRUE, row.names = 1,
                                          max_retry = max_retry, 
                                          sleep_time = 1)
    if (apply_log) {
      out$omics[["mrna"]] <- log2(out$omics[["mrna"]] + 1)
    }
  }
  if (mirna) {
    if (!gene_map | dissimilarities) {
      out$omics[["mirna"]] <- read_csv_with_retry(paste0(data_path, "mirna.csv.gz"), 
                                             header = TRUE, row.names = 1,
                                             max_retry = max_retry, 
                                             sleep_time = 1)
      if (apply_log) {
        out$omics[["mirna"]] <- log2(out$omics[["mirna"]] + 1)
      }
    }
    if (gene_map) {
      out$omics[["mirna_gene"]] <- read_csv_with_retry(paste0(data_path, "mirna_gene_mean.csv.gz"), 
                                                  header = TRUE, row.names = 1,
                                                  max_retry = max_retry, 
                                                  sleep_time = 1)
    }
  }
  if (meth) {
    if (!gene_map | dissimilarities) {
      out$omics[["meth"]] <- read_csv_with_retry(paste0(data_path, "meth.csv.gz"), 
                                            header = TRUE, row.names = 1,
                                            max_retry = max_retry, 
                                            sleep_time = 1)
      out$omics[["meth"]] <- out$omics[["meth"]][apply(is.na(out$omics[["meth"]]), 1, mean) <= 0.5,]
      out$omics[["meth"]] <- out$omics[["meth"]][apply(out$omics[["meth"]] > 0, 1, mean, na.rm = TRUE) >= 0.1,]
      out$omics[["meth"]][out$omics[["meth"]] == 0] <- NA
      out$omics[["meth"]][out$omics[["meth"]] == 1] <- NA
      out$omics[["meth"]] <- as.matrix(-log(1/out$omics[["meth"]]-1))
      if (impute) {
        out$omics[["meth"]] <- impute::impute.knn(out$omics[["meth"]], 
                                             k = 10, 
                                             rowmax = 0.2, 
                                             colmax = 0.2)$data
      }
    }
    if (gene_map) {
      out$omics[["meth_gene"]] <- read_csv_with_retry(paste0(data_path, "meth_gene_promoter_mean.csv.gz"), 
                                                 header = TRUE, row.names = 1,
                                                 max_retry = max_retry, 
                                                 sleep_time = 1)
      out$omics[["meth_gene"]][out$omics[["meth_gene"]] == 0] <- NA
      out$omics[["meth_gene"]][out$omics[["meth_gene"]] == 1] <- NA
      out$omics[["meth_gene"]] <- as.matrix(-log(1/out$omics[["meth_gene"]]-1))
      if (impute) {
        # Remove features with too many missing values
        out$omics[["meth_gene"]] <- out$omics[["meth_gene"]][apply(is.na(out$omics[["meth_gene"]]), 1, mean) < 0.2,]
        out$omics[["meth_gene"]] <- impute::impute.knn(as.matrix(out$omics[["meth_gene"]]), 
                                                  k = 10, 
                                                  rowmax = 0.2, 
                                                  colmax = 0.2)$data
      }
    }
  }
  if (cnv) {
    out$omics[["cnv"]] <- read_csv_with_retry(paste0(data_path, "cnv.csv.gz"), 
                                               header = TRUE, row.names = 1,
                                               max_retry = max_retry, 
                                               sleep_time = 1)
  }
  
  out$omics <- lapply(out$omics, as.matrix)
  
  if (differential) {
    sample_types <- lapply(out$omics, colnames)
    sample_types <- lapply(sample_types, substr, start = 14, stop = 15)
    diff <- list()
    lfc <- list()
    for (omic in names(out$omics)) {
      if (all(c("01", "11") %in% sample_types[[omic]])) {
        diff[[omic]] <- c()
        lfc[[omic]] <- c()
        if (omic == "cnv") {
          for (row_i in rownames(out$omics[[omic]])) {
            # Compare to no variation (0)
            diff[[omic]][row_i] <- wilcox.test(out$omics[[omic]][row_i, sample_types[[omic]] == "01"])$p.value
          }
        } else {
          for (row_i in rownames(out$omics[[omic]])) {
            if (all(c("01", "11") %in% unique(sample_types[[omic]]))) {
              diff[[omic]][row_i] <- wilcox.test(out$omics[[omic]][row_i, sample_types[[omic]] == "01"],
                                                 out$omics[[omic]][row_i, sample_types[[omic]] == "11"])$p.value
              if (omic %in% c("mrna", "mirna", "mirna_gene")) {
                lfc[[omic]][row_i] <- log2(mean(out$omics[[omic]][row_i, sample_types[[omic]] == "01"])+1) - log2(mean(out$omics[[omic]][row_i, sample_types[[omic]] == "11"])+1)
              }
            }
          }
        }
      }
    }
    diff <- lapply(diff, p.adjust, method = p_adjust_method)
    diff_names <- list()
    for (omic in names(diff)) {
      diff_names[[omic]] <- names(which(diff[[omic]] < p_value_threshold))
      if (omic %in% c("mrna", "mirna", "mirna_gene")) {
        diff_names[[omic]] <- diff_names[[omic]][abs(lfc[[omic]][diff_names[[omic]]]) > log_fold_change_threshold]
      }
    }
    out$differential_features <- diff_names
  }
  
  if (tumour_only) {
    sample_types <- lapply(out$omics, colnames)
    sample_types <- lapply(sample_types, substr, start = 14, stop = 15)
    for (omic in names(out$omics)) {
      out$omics[[omic]] <- out$omics[[omic]][,sample_types[[omic]] == "01"]
    }
  }
  
  if (intersect_samples) {
    sample_ids <- lapply(out$omics, colnames)
    sample_ids <- lapply(sample_ids, substr, start = 1, stop = 16)
    sample_intersection <- Reduce(intersect, sample_ids)
    for (omic in names(out$omics)) {
      out$omics[[omic]] <- out$omics[[omic]][,sample_ids[[omic]] %in% sample_intersection]
    }
  }
  
  # Remove duplicated samples
  duplicated_samples <- c()
  for (omic in names(out$omics)) {
    sample_ids <- substr(colnames(out$omics[[omic]]), 1, 16)
    duplicated_samples <- c(duplicated_samples, 
                            sample_ids[which(duplicated(sample_ids))])
  }
  for (omic in names(out$omics)) {
    sample_ids <- substr(colnames(out$omics[[omic]]), 1, 16)
    out$omics[[omic]] <- out$omics[[omic]][,!sample_ids %in% duplicated_samples]
    colnames(out$omics[[omic]]) <- gsub("\\.", "-", sample_ids[!sample_ids %in% duplicated_samples])
  }
  # Add sample info
  if (sample_info) {
    out$sample_info <- read_csv_with_retry(paste0(data_path, "sample_info.csv.gz"), 
                                           header = TRUE, row.names = 1,
                                           max_retry = max_retry, 
                                           sleep_time = 1)
    out$sample_info <- unique(out$sample_info[,c("biopsy", "disease")])
    rownames(out$sample_info) <- out$sample_info$biopsy
  }
  # Add survival info
  if (survival) {
    sample_ids <- unique(Reduce(c, lapply(out$omics, colnames)))
    out$survival <- read_csv_with_retry(paste0(data_path, "survival.csv.gz"), 
                                        header = TRUE, row.names = 1,
                                        max_retry = max_retry, 
                                        sleep_time = 1)
    # In case there are multiple samples per patient
    out$survival <- out$survival[match(substr(sample_ids, 1, 12), out$survival$bcr_patient_barcode),]
    out$survival$sample_id <- sample_ids
  }
  # Add subtype info from TCGAbiolinks
  if (subtype) {
    out$subtype <- tryCatch({TCGAbiolinks::TCGAquery_subtype(cancer_name)}, 
                            error = function(e){warning("TCGA Subtypes not available.");return(NULL)})
    if (!is.null(out$subtype)) {
      out$subtype <- as.data.frame(out$subtype)
      na_strings <- c("", "NA", "N/A", "[Not Available]", "[Not Evaluated]", "[Unknown]")
      for (i in 1:ncol(out$subtype)) {
        out$subtype[[i]][out$subtype[[i]] %in% na_strings] <- NA
        numeric_i <- as.numeric(as.character(gsub(",", ".",out$subtype[[i]])))
        if (all(!is.na(numeric_i) | is.na(out$subtype[[i]]))) {
          out$subtype[[i]] <- numeric_i
        }
      }
      # Need to manually add irrelevant columns
      useless_columns <- c("SomaticRearrangment_story", 
                           "medical_history_thyroid",
                           "mutTumorPortalGene_Protein_Change",
                           "mutDriver_Protein_Change",
                           "mutClinDxTxGene_Protein_Change",
                           "Tissue.source.site")
      out$subtype <- out$subtype[,!colnames(out$subtype) %in% useless_columns]
      sample_ids <- unique(Reduce(c, lapply(out$omics, colnames)))
      matched_ids <- sample_ids[match(out$subtype$patient, substr(sample_ids, 1, 12))]
      out$subtype <- out$subtype[!is.na(matched_ids),]
      rownames(out$subtype) <- matched_ids[!is.na(matched_ids)]
    } else {
      if (cancer_name %in% c("OV")) {
        warning("Using Pan-Cancer Atlas subtypes")
        type_map <- "OVCA"
        names(type_map) <- ("OV")
        pca_types <- as.data.frame(TCGAbiolinks::PanCancerAtlas_subtypes())
        out$subtype <- pca_types[pca_types$cancer.type == type_map[cancer_name],]
        sample_ids <- unique(Reduce(c, lapply(out$omics, colnames)))
        matched_ids <- sample_ids[match(substr(out$subtype$pan.samplesID, 1, 12), 
                                        substr(sample_ids, 1, 12))]
        out$subtype <- out$subtype[!is.na(matched_ids),]
        rownames(out$subtype) <- matched_ids[!is.na(matched_ids)]
      }
    }
  }
  
  return(out)
}

rows_to_set_means <- function(x, map, id_col = "id", set_col = "symbol", parallel = 1) {
  map <- map[!is.na(map[[set_col]]),]
  map <- map[map[[id_col]] %in% rownames(x),]
  x_mapped <- as.data.frame(x[map[[id_col]],])
  x_mapped$set_column <- map[[set_col]]
  # Take setwise mean for each column (sample), set columns with only missing values to NA
  #f <- function(y) as.data.frame(lapply(y[colnames(y) != "set_column"], function(t) ifelse(all(is.na(t)), NA, mean(t, na.rm = TRUE))))
  # But maybe this is not better than imputation
  f <- function(y) as.data.frame(lapply(y[colnames(y) != "set_column"], mean))
  parallel_clust <- COPS:::setup_parallelization(parallel)
  x_mapped <- plyr::ddply(x_mapped, "set_column", f, .parallel = parallel > 1) 
  COPS:::close_parallel_cluster(parallel_clust)
  rownames(x_mapped) <- x_mapped$set_column
  x_mapped$set_column <- NULL
  return(x_mapped)
}

nonnegative_transform <- function(x) {
  if ("rppa" %in% names(x)) {
    # Rank based value (empirical cdf)
    x$rppa <- apply(x$rppa, 2, function(x) rank(x) / length(x))
  }
  if ("meth"  %in% names(x)) {
    # Transform back from logit by using the logistic function
    x[["meth"]] <- 1/(1 + exp(-x[["meth"]]))
  }
  if ("cnv"  %in% names(x)) {
    # Copy number loss greater than 1 is rare and 2 is max, although
    # technically there could be more than 2 copies of some genes in the germline, 
    # but I'm not sure if this would be something that current methods can quantify. 
    x$cnv <- x$cnv + 2
  }
  return(x)
}