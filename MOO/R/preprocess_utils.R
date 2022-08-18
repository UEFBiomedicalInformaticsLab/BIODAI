save_view <- function(view, name, directory) {
  if (!dir.exists(directory)){
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  }
  write.csv(x = view, file = paste(directory,name,".csv", sep = ""), row.names = TRUE)
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