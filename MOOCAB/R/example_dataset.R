source("R/preprocess_utils.R")

SAVE_DIR <- "work/custom/input/"

MIN_DETECTIONS <- 3

TUMOURS <- c("KIRC", "KIRP", "KICH")
OUTCOME_COL <- "PanKidney Pathology"
CHARS_FOR_ID <- 12

survival_event <- "OS"

TYPE_COL_NAME <- "type"

outcome <- read.csv(file = "work/kidney_ihc_det_os/input/outcome.csv", row.names = 1, header= TRUE)
mrna <- read.csv(file = "work/kidney_ihc_det_os/input/mrna.csv", row.names = 1, header= TRUE)

colnames(outcome)[1] <- TYPE_COL_NAME

outcome <- outcome[seq(1, nrow(outcome), 5), ]
mrna <- mrna[seq(1, nrow(mrna), 5), ]

save_view(outcome,"outcome",SAVE_DIR)
save_view(mrna,"mrna",SAVE_DIR)
