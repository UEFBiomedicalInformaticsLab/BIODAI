library("readxl")
source("R/preprocess_utils.R")

MIN_DETECTIONS <- 3
SAVE_DIR_IHC_DET <- "work/kidney_ihc_det/input/"

TUMOURS <- c("KIRC", "KIRP", "KICH")
OUTCOME_COL <- "PanKidney Pathology"
CHARS_FOR_ID <- 12

SURVIVAL_EVENTS <- c("OS")

pathology <- read.csv(file = "work/kidney_ihc_det/raw/pathology.tsv", sep = "\t")
pathology <- pathology[pathology$Cancer == "renal cancer", ]
pathology$detected_sum <- pathology$High + pathology$Medium + pathology$Low
pathology <- pathology[!is.na(pathology$detected_sum), ]
# Remove duplicate gene names.
pathology <- pathology[order(pathology[,'Gene.name'],-pathology[,'detected_sum']),]
pathology <- pathology[!duplicated(pathology$Gene.name),]

labels <- read_excel("work/kidney_ihc_det/raw/NIHMS958988-supplement-2.xlsx")
labels <- labels[ , !names(labels) %in% "...1"]
print(paste("Loaded labels:", nrow(labels)))
labels <- labels[labels[,OUTCOME_COL] != "KIRP CIMP", ,drop=FALSE]
print(paste("Labels after dropping KIRP CIMP outcome:", nrow(labels)))
labels <- labels[labels[,OUTCOME_COL] != "PRCC Unc", ,drop=FALSE]
print(paste("Labels after dropping PRCC Unc outcome:", nrow(labels)))
labels <- as.data.frame(labels)
labels <- labels[!is.na(labels$"bcr_patient_barcode"),]
row.names(labels) <- labels[, "bcr_patient_barcode"]
row.names(labels) <- strtrim(row.names(labels), CHARS_FOR_ID)
row.names(labels) <- paste(row.names(labels),"T", sep="")

tumour_mrna <- load_tumours_and_clean(tumours = TUMOURS, chars_for_id = CHARS_FOR_ID)

common_ids <- intersect(rownames(labels), substr(colnames(tumour_mrna), 1, stop=CHARS_FOR_ID+1))
print(paste("Number of common ids:",length(common_ids)))
labels <- labels[row.names(labels) %in% common_ids,]

genes_with_antibodies <- read_tsv(file = "work/kidney_ihc_det/raw/ihc_ab_validation_Enhanced_names.tsv", col_select = "Gene")
genes_with_antibodies <- genes_with_antibodies[[1]]
print(paste("Genes with antibodies:", length(genes_with_antibodies)))

tumour_mrna_with_antibodies <- tumour_mrna[rownames(tumour_mrna) %in% genes_with_antibodies, ]
colnames(tumour_mrna_with_antibodies) <- strtrim(colnames(tumour_mrna_with_antibodies), CHARS_FOR_ID+1)
print(paste("Genes with antibodies in tumour mrna:", nrow(tumour_mrna_with_antibodies)))
print(paste("Tumour samples:", ncol(tumour_mrna_with_antibodies)))

normal_mrna <- load_normals_and_clean(tumours = TUMOURS, chars_for_id = CHARS_FOR_ID)
common_genes <- intersect(rownames(normal_mrna), rownames(tumour_mrna_with_antibodies))
print(paste("Genes with antibodies in tumour and normal mrna:", length(common_genes)))
normal_mrna <- normal_mrna[rownames(normal_mrna) %in% common_genes, ]

tumour_and_normal_mrna_with_anti <- cbind(tumour_mrna_with_antibodies,normal_mrna)
print(paste("Tumour and normal samples:", ncol(tumour_and_normal_mrna_with_anti)))

labels_for_normal <- DataFrame(replicate(ncol(normal_mrna), "normal"))
rownames(labels_for_normal) <- colnames(normal_mrna)
labels_for_normal <- as.data.frame(labels_for_normal)
colnames(labels_for_normal) <- "PanKidney Pathology"
labels_with_normal <- labels[,OUTCOME_COL, drop=FALSE]
labels_with_normal <- rbind(labels_with_normal, labels_for_normal)  # Warning: creates new rowlabels if finds duplicates

common_ids_with_normal <- intersect(rownames(labels_with_normal), substr(colnames(tumour_and_normal_mrna_with_anti), 1, stop=CHARS_FOR_ID+1))

pathology <- pathology[pathology$Gene.name %in% rownames(tumour_and_normal_mrna_with_anti), ]

well_detected_pathology <- pathology[pathology$detected_sum >= MIN_DETECTIONS, ]
  # Genes with at least MIN_DETECTIONS in patients with kidney cancer.

tumour_and_normal_mrna_with_anti_det <- tumour_and_normal_mrna_with_anti[
  rownames(tumour_and_normal_mrna_with_anti) %in% well_detected_pathology$Gene.name, ]
save_view(labels_with_normal,"outcome",SAVE_DIR_IHC_DET)
saved <- reduce_to_common_patient_ids_and_save(
  tumour_and_normal_mrna_with_anti_det,
  common_ids_with_normal,
  directory = SAVE_DIR_IHC_DET,
  verbose_name = "mrna",
  chars_for_id = CHARS_FOR_ID+1)

for (survival_event in SURVIVAL_EVENTS) {
  print(paste("Processing survival type", survival_event))
  survival_data <- load_survival_data(tumours = TUMOURS, event = survival_event)
  row.names(survival_data) <- paste(row.names(survival_data),"T", sep="")
  print(paste("Patients with survival data",nrow(survival_data)))
  print(paste("Patients with subtype data",nrow(labels_with_normal)))
  ids_with_surv_and_subtype <- intersect(rownames(labels_with_normal), rownames(survival_data))
  print(paste("Patients with both survival and subtype data",length(ids_with_surv_and_subtype)))
  survival_data_in_common <- survival_data[rownames(survival_data) %in% ids_with_surv_and_subtype,]
  labels_in_common <- labels_with_normal[rownames(labels_with_normal) %in% ids_with_surv_and_subtype, ,drop=FALSE]
  subtype_and_survival <- merge(labels_in_common, survival_data_in_common, by=0, all=TRUE)  # merge by row names (by=0 or by="row.names")
  rownames(subtype_and_survival) <- subtype_and_survival$Row.names
  subtype_and_survival <- subtype_and_survival[, !(colnames(subtype_and_survival) %in% "Row.names")]
  print(table(subtype_and_survival[, !(colnames(subtype_and_survival) %in% "Time")]))
  save_dir_by_surv_type <- paste("work/kidney_ihc_det_", tolower(survival_event), "/input/", sep="")

  save_view(subtype_and_survival,"outcome",save_dir_by_surv_type)
  paste("Saving mrna with genes detected", MIN_DETECTIONS, "or more times.")
  saved <- reduce_to_common_patient_ids_and_save(
    tumour_and_normal_mrna_with_anti_det,
    rownames(subtype_and_survival),
    directory = save_dir_by_surv_type,
    verbose_name = "mrna",
    chars_for_id = CHARS_FOR_ID+1)
}
