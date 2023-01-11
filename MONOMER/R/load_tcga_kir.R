library("readxl")
source("R/preprocess_utils.R")

SAVE_DIR <- "work/tcga_kir/input/"
SAVE_DIR3 <- "work/tcga_kir3/input/"

TUMOURS <- c("KIRC", "KIRP", "KICH")
OUTCOME_COL <- "PanKidney Pathology"
CHARS_FOR_ID <- 12

tumour_mrna <- load_tumours(tumours = TUMOURS)

if (FALSE) {
  labels <- TCGAbiolinks::TCGAquery_subtype(tumor = TUMOR)
  rownames(labels) <- labels$patient
  print("microRNA_cluster")
  table(labels$microRNA_cluster)
  print("mRNA_cluster")
  table(labels$mRNA_cluster)
}

labels <- read_excel("work/tcga_kir/raw/NIHMS958988-supplement-2.xlsx")
labels <- labels[ , !names(labels) %in% "...1"]
print(paste("Loaded labels:", nrow(labels)))
labels <- labels[labels[,OUTCOME_COL] != "KIRP CIMP", ,drop=FALSE]
print(paste("Labels after dropping KIRP CIMP outcome:", nrow(labels)))
labels <- labels[labels[,OUTCOME_COL] != "PRCC Unc", ,drop=FALSE]
print(paste("Labels after dropping PRCC Unc outcome:", nrow(labels)))
labels <- as.data.frame(labels)
labels <- labels[!is.na(labels$"bcr_patient_barcode"),]
row.names(labels) <- labels[, "bcr_patient_barcode"]

common_ids <- intersect(rownames(labels), substr(colnames(tumour_mrna), 1, stop=CHARS_FOR_ID))
print(paste("Number of common ids:",length(common_ids)))
labels <- labels[labels$"bcr_patient_barcode" %in% common_ids,]

print(table(labels[,OUTCOME_COL]))

save_view(labels,"outcome",SAVE_DIR)
saved <- reduce_to_common_patient_ids_and_save(
  tumour_mrna,common_ids,directory = SAVE_DIR,verbose_name = "mrna", chars_for_id = CHARS_FOR_ID)

print("Creating dataset with 3 labels.")
labels3 <- labels
labels3[OUTCOME_COL][labels3[OUTCOME_COL]=="PRCC T1"] <- "PRCC"
labels3[OUTCOME_COL][labels3[OUTCOME_COL]=="PRCC T2"] <- "PRCC"

save_view(labels3,"outcome",SAVE_DIR3)
saved <- reduce_to_common_patient_ids_and_save(
  tumour_mrna,common_ids,directory = SAVE_DIR3,verbose_name = "mrna", chars_for_id = CHARS_FOR_ID)
