library("readxl")
source("R/preprocess_utils.R")

SAVE_DIR <- "work/tcga_kir/input/"
SAVE_DIR3 <- "work/tcga_kir3/input/"
SAVE_DIR_IHC <- "work/kidney_ihc/input/"

MIN_DETECTIONS <- 3
SAVE_DIR_IHC_DET <- "work/kidney_ihc_det/input/"


SAVE_DIR_IHC_MRNA_MIRNA <- "work/kidney_ihc_mrna_mirna/input/"  # MRNA with IHC but no min detections.
SAVE_DIR_IHC_MRNA_MIRNA_CNVSNP <- "work/kidney_ihc_mrna_mirna_cnvsnp/input/"
SAVE_DIR_IHC_MRNA_MIRNA_METH <- "work/kidney_ihc_mrna_mirna_meth/input/"

SAVE_DIR_IHC_MRNA_MIRNA_OS <- "work/kidney_ihc_mrna_mirna_os/input/"  # MRNA with IHC but no min detections.

TUMOURS <- c("KIRC", "KIRP", "KICH")
OUTCOME_COL <- "PanKidney Pathology"
CHARS_FOR_ID <- 12

SURVIVAL_EVENTS <- c("OS", "DSS", "PFI")
SURVIVAL_EVENTS_MULTI_VIEW <- c("OS")

pathology <- read.csv(file = "work/kidney_ihc_det/raw/pathology.tsv", sep = "\t")
pathology <- pathology[pathology$Cancer == "renal cancer", ]
pathology$detected_sum <- pathology$High + pathology$Medium + pathology$Low
pathology <- pathology[!is.na(pathology$detected_sum), ]
# Remove duplicate gene names.
pathology <- pathology[order(pathology[,'Gene.name'],-pathology[,'detected_sum']),]
pathology <- pathology[!duplicated(pathology$Gene.name),]
print(paste("Tot genes in pathology.tsv:", nrow(pathology)))
print(paste("Genes detected 0 times:", sum(pathology$detected_sum == 0)))
for (i in 1:12) {
  print(paste("Genes detected", i, "or more times:", sum(pathology$detected_sum > i-1)))
}
jpeg("work/kidney_ihc_det/raw/detected_sum_all.png")
hist(pathology$detected_sum)
dev.off()


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
row.names(labels) <- strtrim(row.names(labels), CHARS_FOR_ID)
row.names(labels) <- paste(row.names(labels),"T", sep="")

# There is no survival data for KIRC
# biolinks_labels_kirc <- TCGAbiolinks::TCGAquery_subtype(tumor = "KIRC")
# print(colnames(biolinks_labels_kirc))

tumour_mrna <- load_tumours_and_clean(tumours = TUMOURS, chars_for_id = CHARS_FOR_ID)

common_ids <- intersect(rownames(labels), substr(colnames(tumour_mrna), 1, stop=CHARS_FOR_ID+1))
print(paste("Number of common ids:",length(common_ids)))
labels <- labels[row.names(labels) %in% common_ids,]

print(table(labels[,OUTCOME_COL]))

save_view(labels,"outcome",SAVE_DIR)
saved <- reduce_to_common_patient_ids_and_save(
  tumour_mrna,common_ids,directory = SAVE_DIR,verbose_name = "mrna", chars_for_id = CHARS_FOR_ID+1)

genes_with_antibodies <- read_tsv(file = "work/kidney_ihc/raw/ihc_ab_validation_Enhanced_names.tsv", col_select = "Gene")
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

save_view(labels_with_normal,"outcome",SAVE_DIR_IHC)
saved <- reduce_to_common_patient_ids_and_save(
  tumour_and_normal_mrna_with_anti,common_ids_with_normal,directory = SAVE_DIR_IHC,verbose_name = "mrna", chars_for_id = CHARS_FOR_ID+1)

print(paste("Genes included in",SAVE_DIR_IHC,":",nrow(tumour_and_normal_mrna_with_anti)))

pathology <- pathology[pathology$Gene.name %in% rownames(tumour_and_normal_mrna_with_anti), ]
print(paste("Genes included in",SAVE_DIR_IHC_DET,":",nrow(pathology)))
print(paste("Tot genes in pathology.tsv with existing data:", nrow(pathology)))
print(paste("Genes detected 0 times:", sum(pathology$detected_sum == 0)))
for (i in 1:12) {
  print(paste("Genes detected", i, "or more times:", sum(pathology$detected_sum > i-1)))
}
jpeg("work/kidney_ihc_det/raw/detected_sum_with_data.png")
hist(pathology$detected_sum)
dev.off()

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

print("Loading all survival types and all additional columns.")
whole_survival_data <- load_whole_survival_data(tumours=TUMOURS)
row.names(whole_survival_data) <- paste(row.names(whole_survival_data),"T", sep="")
save_view(
  view=whole_survival_data,
  name="whole_survival_data",
  directory="work/TCGA survival/")

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


# Adding views proteins and mirna to mrna with antibodies.
tumour_mirna <- load_tumours_and_clean(tumours = TUMOURS, assays = "miRNASeqGene", chars_for_id = CHARS_FOR_ID)
# tumour_rppaa <- load_tumours_and_clean(tumours = TUMOURS, assays = "RPPAArray", chars_for_id = CHARS_FOR_ID)
# tumour_cnvsnp <- load_tumours_and_clean(tumours = TUMOURS, assays = "CNVSNP", chars_for_id = CHARS_FOR_ID)
# tumour_meth27 <- load_tumours_and_clean(tumours = TUMOURS, assays = "Methylation_methyl27", chars_for_id = CHARS_FOR_ID)
# tumour_meth450 <- load_tumours_and_clean(tumours = TUMOURS, assays = "Methylation_methyl450", chars_for_id = CHARS_FOR_ID)
common_ids_mv_t_mrna_mirna <- intersect_all(list(common_ids_with_normal,colnames(tumour_mirna)))
print(paste("Tumour samples with mrna and mirna:", length(common_ids_mv_t_mrna_mirna)))
# common_ids_mv_t_mrna_cnvsnp <- intersect_all(list(common_ids_with_normal,colnames(tumour_cnvsnp)))
# print(paste("Tumour samples with mrna and cnvsnp:", length(common_ids_mv_t_mrna_cnvsnp)))
# common_ids_mv_t_mrna_mirna_rppaa <- intersect_all(list(common_ids_with_normal,colnames(tumour_mirna), colnames(tumour_rppaa)))
# print(paste("Tumour samples with mrna, mirna and rppaa:", length(common_ids_mv_t_mrna_mirna_rppaa)))
# common_ids_mv_t_mrna_mirna_cnvsnp <- intersect_all(list(common_ids_with_normal,colnames(tumour_mirna), colnames(tumour_cnvsnp)))
# print(paste("Tumour samples with mrna, mirna and cnvsnp:", length(common_ids_mv_t_mrna_mirna_cnvsnp)))
# common_ids_mv_t_mrna_meth27 <- intersect_all(list(common_ids_with_normal,colnames(tumour_meth27)))
# print(paste("Tumour samples with mrna and meth27:", length(common_ids_mv_t_mrna_meth27)))
# common_ids_mv_t_mrna_meth450 <- intersect_all(list(common_ids_with_normal,colnames(tumour_meth450)))
# print(paste("Tumour samples with mrna and meth450:", length(common_ids_mv_t_mrna_meth450)))
# common_ids_mv_t_mrna_mirna_meth450 <- intersect_all(list(common_ids_with_normal,colnames(tumour_mirna),colnames(tumour_meth450)))
# print(paste("Tumour samples with mrna, mirna and meth450:", length(common_ids_mv_t_mrna_mirna_meth450)))


normal_mirna <- load_normals_and_clean(tumours = TUMOURS, assays = "miRNASeqGene", chars_for_id = CHARS_FOR_ID)
# normal_rppaa <- load_normals_and_clean(tumours = TUMOURS, assays = "RPPAArray", chars_for_id = CHARS_FOR_ID)
# normal_meth <- load_normals_and_clean(tumours = TUMOURS, assays = "Methylation", chars_for_id = CHARS_FOR_ID)
# normal_meth27 <- load_normals_and_clean(tumours = TUMOURS, assays = "Methylation_methyl27", chars_for_id = CHARS_FOR_ID)
# We do not have any intersection of samples with meth27
# normal_meth450 <- load_normals_and_clean(tumours = TUMOURS, assays = "Methylation_methyl450", chars_for_id = CHARS_FOR_ID)
# We have normal samples for meth450.
# normal_cnvsnp <- load_normals_and_clean(tumours = TUMOURS, assays = "CNVSNP", chars_for_id = CHARS_FOR_ID)
# The features of cnvsnp do not pass ANOVA test.
# normal_mut <- load_normals_and_clean(tumours = TUMOURS, assays = "Mutation", chars_for_id = CHARS_FOR_ID)
# Mutations are not available.

common_ids_mv_n_mrna_mirna <- intersect_all(list(common_ids_with_normal,colnames(normal_mirna)))
print(paste("Normal samples with mrna and mirna:", length(common_ids_mv_n_mrna_mirna)))
# common_ids_mv_n_mrna_cnvsnp <- intersect_all(list(common_ids_with_normal,colnames(normal_cnvsnp)))
# print(paste("Normal samples with mrna and cnvsnp:", length(common_ids_mv_n_mrna_cnvsnp)))
# common_ids_mv_n_mrna_meth27 <- Reduce(intersect, list(common_ids_with_normal,colnames(normal_meth27)))
# print(paste("Normal samples with mrna and meth27:", length(common_ids_mv_n_mrna_meth27)))
# common_ids_mv_n_mrna_meth450 <- Reduce(intersect, list(common_ids_with_normal,colnames(normal_meth450)))
# print(paste("Normal samples with mrna and meth450:", length(common_ids_mv_n_mrna_meth450)))
# common_ids_mv_n_mrna_mirna_cnvsnp <- intersect_all(list(common_ids_with_normal,colnames(normal_mirna),colnames(normal_cnvsnp)))
# print(paste("Normal samples with mrna, mirna and cnvsnp:", length(common_ids_mv_n_mrna_mirna_cnvsnp)))
# common_ids_mv_n_mrna_mirna_meth450 <- Reduce(intersect, list(common_ids_with_normal,colnames(normal_mirna),colnames(normal_meth450)))
# print(paste("Normal samples with mrna, mirna and meth:", length(common_ids_mv_n_mrna_mirna_meth450)))
# common_ids_mv_n_mrna_cnvsnp_meth <- Reduce(intersect, list(common_ids_with_normal,colnames(normal_cnvsnp),colnames(normal_meth450)))
# print(paste("Normal samples with mrna, cnvsnp and meth:", length(common_ids_mv_n_mrna_cnvsnp_meth)))

common_ids_mrna_mirna <- c(common_ids_mv_t_mrna_mirna, common_ids_mv_n_mrna_mirna)
# common_ids_mrna_meth450 <- c(common_ids_mv_t_mrna_meth450, common_ids_mv_n_mrna_meth450)
# common_ids_mrna_meth27 <- c(common_ids_mv_t_mrna_meth27, common_ids_mv_n_mrna_meth27)
# common_ids_mrna_meth <- unique(union(common_ids_mrna_meth450,common_ids_mrna_meth27))
# common_ids_mrna_mirna_cnvsnp <- c(common_ids_mv_t_mrna_mirna_cnvsnp, common_ids_mv_n_mrna_mirna_cnvsnp)
# common_ids_mrna_mirna_meth450 <- c(common_ids_mv_t_mrna_mirna_meth450, common_ids_mv_n_mrna_mirna_meth450)
common_ids_mv <- common_ids_mrna_mirna
selected_mrna_mv <- select_samples(tumour_and_normal_mrna_with_anti,common_ids_mv)
selected_mirna_mv <- concat_samples(select_samples(tumour_mirna,common_ids_mv),
                                    select_samples(normal_mirna,common_ids_mv))
# selected_cnvsnp_mv <- concat_samples(select_samples(tumour_cnvsnp,common_ids_mrna_mirna_cnvsnp),
#                                      select_samples(normal_cnvsnp,common_ids_mrna_mirna_cnvsnp))
# selected_meth_mv <- concat_samples(select_samples(tumour_meth450,common_ids_mv_t_mrna_mirna_meth450),
#                                    select_samples(normal_meth450,common_ids_mv_n_mrna_mirna_meth450))
labels_mv <- labels_with_normal[rownames(labels_with_normal) %in% common_ids_mv,,drop=FALSE]
print("Multi-view labels")
print(table(labels_mv))

save_view(labels_mv,"outcome",SAVE_DIR_IHC_MRNA_MIRNA)
saved <- reduce_to_common_patient_ids_and_save(
  selected_mrna_mv,common_ids_mv,directory = SAVE_DIR_IHC_MRNA_MIRNA,verbose_name = "mrna", chars_for_id = CHARS_FOR_ID+1)
saved <- reduce_to_common_patient_ids_and_save(
  selected_mirna_mv,common_ids_mv,directory = SAVE_DIR_IHC_MRNA_MIRNA,verbose_name = "mirna", chars_for_id = CHARS_FOR_ID+1)
# saved <- reduce_to_common_patient_ids_and_save(
#   selected_cnvsnp_mv,common_ids_mv,directory = SAVE_DIR_IHC_MRNA_MIRNA_CNVSNP,verbose_name = "cnvsnp", chars_for_id = CHARS_FOR_ID+1)

print("Creating mrna and mirna dataset with subtypes and survival.")
for (survival_event in SURVIVAL_EVENTS_MULTI_VIEW) {
  print(paste("Processing survival type", survival_event))
  survival_data <- load_survival_data(tumours = TUMOURS, event = survival_event)
  row.names(survival_data) <- paste(row.names(survival_data),"T", sep="")
  print(paste("Patients with survival data",nrow(survival_data)))
  print(paste("Patients with subtype data",nrow(common_ids_mv)))
  ids_with_surv_and_subtype <- intersect(common_ids_mv, rownames(survival_data))
  print(paste("Patients with both survival and subtype data",length(ids_with_surv_and_subtype)))
  survival_data_in_common <- survival_data[rownames(survival_data) %in% ids_with_surv_and_subtype,]
  labels_in_common <- labels_mv[rownames(labels_mv) %in% ids_with_surv_and_subtype, ,drop=FALSE]
  subtype_and_survival <- merge(labels_in_common, survival_data_in_common, by=0, all=TRUE)  # merge by row names (by=0 or by="row.names")
  rownames(subtype_and_survival) <- subtype_and_survival$Row.names
  subtype_and_survival <- subtype_and_survival[, !(colnames(subtype_and_survival) %in% "Row.names")]
  print(table(subtype_and_survival[, !(colnames(subtype_and_survival) %in% "Time")]))
  save_dir_by_surv_type <- paste("work/kidney_ihc_mrna_mirna_", tolower(survival_event), "/input/", sep="")

  save_view(subtype_and_survival,"outcome",save_dir_by_surv_type)
  saved <- reduce_to_common_patient_ids_and_save(
    selected_mrna_mv,
    rownames(subtype_and_survival),
    directory = save_dir_by_surv_type,
    verbose_name = "mrna",
    chars_for_id = CHARS_FOR_ID+1)
  saved <- reduce_to_common_patient_ids_and_save(
    selected_mirna_mv,
    rownames(subtype_and_survival),
    directory = save_dir_by_surv_type,
    verbose_name = "mirna",
    chars_for_id = CHARS_FOR_ID+1)
}

print("Creating dataset with 3 labels.")
labels3 <- labels
labels3[OUTCOME_COL][labels3[OUTCOME_COL]=="PRCC T1"] <- "PRCC"
labels3[OUTCOME_COL][labels3[OUTCOME_COL]=="PRCC T2"] <- "PRCC"

save_labels_and_data_common_ids(
  directory=SAVE_DIR3, data=tumour_mrna, labels=labels3, labels_name = "outcome",
  data_name = "mrna",
  chars_for_id = CHARS_FOR_ID+1)
