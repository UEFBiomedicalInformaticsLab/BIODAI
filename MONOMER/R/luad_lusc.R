source("R/preprocess_utils.R")
directory <- "work/luad_lusc/input/"
chars_for_id <- 12

luad_mrna <- suppressMessages(curatedTCGAData::curatedTCGAData(diseaseCode = "LUAD", assays = "RNASeq2GeneNorm", dry.run = FALSE, version = "2.0.1"))
TCGAutils::sampleTables(luad_mrna)
luad_tumour_mrna <- TCGAutils::splitAssays(luad_mrna, "01")
luad_tumour_mrna <- Reduce("cbind", lapply(experiments(luad_tumour_mrna), function(x) assays(x)[[1]]))

lusc_mrna <- suppressMessages(curatedTCGAData::curatedTCGAData(diseaseCode = "LUSC", assays = "RNASeq2GeneNorm", dry.run = FALSE, version = "2.0.1"))
TCGAutils::sampleTables(lusc_mrna)
lusc_tumour_mrna <- TCGAutils::splitAssays(lusc_mrna, "01")
lusc_tumour_mrna <- Reduce("cbind", lapply(experiments(lusc_tumour_mrna), function(x) assays(x)[[1]]))

luad_labels <- TCGAbiolinks::TCGAquery_subtype(tumor = "LUAD")
rownames(luad_labels) <- luad_labels$patient
table(luad_labels$expression_subtype)

lusc_labels <- TCGAbiolinks::TCGAquery_subtype(tumor = "lusc")
rownames(lusc_labels) <- lusc_labels$patient
names(lusc_labels)[names(lusc_labels) == 'Expression.Subtype'] <- 'expression_subtype'
table(lusc_labels$expression_subtype)

common_luad_ids <- intersect(rownames(luad_labels), substr(colnames(luad_tumour_mrna), 1, stop=chars_for_id))
length(common_luad_ids)
table(luad_labels[common_luad_ids,"expression_subtype"])

common_lusc_ids <- intersect(rownames(lusc_labels), substr(colnames(lusc_tumour_mrna), 1, stop=chars_for_id))
length(common_lusc_ids)
table(lusc_labels[common_lusc_ids,"expression_subtype"])

print(paste("LUAD initial samples:", ncol(luad_tumour_mrna)))
print(paste("LUSC initial samples:", ncol(lusc_tumour_mrna)))
luad_tumour_mrna <- luad_tumour_mrna[,match(common_luad_ids, substr(colnames(luad_tumour_mrna), start=1, stop=chars_for_id))]
lusc_tumour_mrna <- lusc_tumour_mrna[,match(common_lusc_ids, substr(colnames(lusc_tumour_mrna), start=1, stop=chars_for_id))]
print(paste("LUAD labelled samples:", ncol(luad_tumour_mrna)))
print(paste("LUSC labelled samples:", ncol(lusc_tumour_mrna)))
print(paste("Number of LUAD features:",nrow(luad_tumour_mrna)))
print(paste("Number of LUSC features:",nrow(lusc_tumour_mrna)))
combined <- cbind(luad_tumour_mrna, lusc_tumour_mrna)
print(paste("Combined labelled samples:", ncol(combined)))

filtered_luad_labels <- luad_labels[common_luad_ids,"expression_subtype", drop=FALSE]
filtered_lusc_labels <- lusc_labels[common_lusc_ids,"expression_subtype", drop=FALSE]

filtered_outcome <- rbind(filtered_luad_labels, filtered_lusc_labels)

save_view(filtered_outcome,"outcome",directory)

combined <- t(combined)
save_view(combined,"mrna",directory)
