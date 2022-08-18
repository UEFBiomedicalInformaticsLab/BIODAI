library(TCGAbiolinks)
library(curatedTCGAData)
library(TCGAutils)

source("R/preprocess_utils.R")

disease_code <- "BRCA"

directory <- "work/brca/input/"
classification_to_use <-
  "BRCA_Subtype_PAM50"
  # "Subtype_Integrative"
  # "cancer.type"
assays_to_use <- "RNASeq2GeneNorm"


transpose_and_save <- function(data, verbose_name = "single view data") {
  print(paste("Saving",verbose_name))
  print(paste("Samples:",ncol(data)))
  print(paste("Number of features:",nrow(data)))
  data <- t(data)
  save_view(data,verbose_name,directory)
  return (data)
}

# mRNA data (Upper quartile normalized RSEM TPM gene expression values)
mrna <- curatedTCGAData::curatedTCGAData(diseaseCode = disease_code, assays = assays_to_use, version="2.0.1", dry.run = FALSE, verbose=TRUE)
TCGAutils::sampleTables(mrna)
tumour_mrna <- Reduce("cbind", lapply(experiments(mrna), function(x) assays(x)[[1]]))
patient_ids <- substr(colnames(tumour_mrna), start = 1, stop = 12)

# Samples are on columns and the first 12 characters of the IDs correspond to patientID

# Courtesy of Teemu
brca_labels <- TCGAbiolinks::TCGAquery_subtype(tumor = "BRCA")
brca_labels$BRCA_Subtype_PAM50

outcome_df <- as.data.frame(brca_labels[,classification_to_use])
rownames(outcome_df) <- brca_labels$patient

common_patient_ids_outcome <- intersect(patient_ids, substr(rownames(outcome_df), 1, 12))
if (length(common_patient_ids_outcome) != length(patient_ids)) {
  print("Missing patients in outcome")
  print(paste("num patients:", length(patient_ids)))
  print(paste("num patients in outcome:", length(common_patient_ids_outcome)))
}

filtered_outcome <- outcome_df[match(common_patient_ids_outcome, substr(rownames(outcome_df), 1, 12)),,drop=FALSE]

if (anyNA(filtered_outcome)) {
  stop("NA in outcome")
}

save_view(filtered_outcome,name="outcome",directory=directory)
saved <- reduce_to_common_patient_ids_and_save(
  data=tumour_mrna,common_ids=common_patient_ids_outcome,directory = directory,verbose_name = "mrna")