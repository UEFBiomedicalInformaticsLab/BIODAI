source("R/preprocess_utils.R")

SAVE_DIR <- "work/tcga_ov/input/"

TUMOURS <- c("OV")

CHARS_FOR_ID <- 20

tumour_mrna <- load_tumours(tumours = TUMOURS)
subtypes <- PanCancerAtlas_subtypes()

common_ov_ids <- intersect(subtypes$pan.samplesID, substr(colnames(tumour_mrna), 1, stop=CHARS_FOR_ID))
ov.subtypes <- data.frame(subtypes[which(subtypes$pan.samplesID %in% substr(colnames(tumour_mrna), 1, stop=CHARS_FOR_ID)),], row.names = "pan.samplesID")
ov.subtypes <- ov.subtypes[common_ov_ids,]
print(colnames(ov.subtypes))
print(table(ov.subtypes$Subtype_Selected))

save_view(ov.subtypes,"outcome",SAVE_DIR)
saved <- reduce_to_common_patient_ids_and_save(
  data=tumour_mrna,common_ids=common_ov_ids,directory = SAVE_DIR,verbose_name = "mrna", chars_for_id = CHARS_FOR_ID)