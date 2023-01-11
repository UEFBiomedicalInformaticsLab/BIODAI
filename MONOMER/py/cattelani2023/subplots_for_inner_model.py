from cattelani2023.cattelani2023_utils import CATTELANI2023_COL_I, CATTELANI2023_COL_J, CATTELANI2023_SETUP, CATTELANI2023_TCGA_DATASETS, CATTELANI2023_MAIN_LABS
from cattelani2023.subplots_runner_alltogether import save_path_trade_external, SUBSCATTERPLOTS_EXTERNAL_FILE, \
    SUBTRADEPLOTS_EXTERNAL_FILE, save_path_trade_cv, SUBSCATTERPLOTS_CV_FILE, SUBTRADEPLOTS_CV_FILE
from plots.archives.automated_hofs_archive import nested_hofs_for_dataset_cv
from plots.archives.shallow_saved_hofs_archive_external import cattelani2023_external_validations
from plots.plot_labels import ALL_INNER_LABS
from plots.subplots_by_strategy import subscatterplots, subtradeplots

if __name__ == '__main__':
    inner_labs = ALL_INNER_LABS
    ncols = 2
    n_inner = len(inner_labs)
    for i in range(n_inner):
        inner = inner_labs[i]
        print("Processing inner model " + inner)
        print("Processing external validations")
        external_hofs = []
        for ext in cattelani2023_external_validations(inner_labs=[inner]):
            external_hofs.extend(ext.nested_hofs())
        print("Processing external scatterplots")
        subscatterplots(
            hofs=external_hofs,
            save_path=save_path_trade_external(
                plot_name=SUBSCATTERPLOTS_EXTERNAL_FILE + "_" + inner),
            ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=CATTELANI2023_SETUP)
        print("Processing external tradeplots")
        subtradeplots(
            hofs=external_hofs,
            save_path=save_path_trade_external(
                plot_name=SUBTRADEPLOTS_EXTERNAL_FILE + "_" + inner),
            ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=CATTELANI2023_SETUP)

        print("Processing cross-validations")
        hofs = []
        for dataset_label in CATTELANI2023_TCGA_DATASETS:
            hofs.extend(nested_hofs_for_dataset_cv(
                dataset_lab=dataset_label, main_labs=CATTELANI2023_MAIN_LABS, inner_labs=[inner]))
        print("Processing CV scatterplots")
        subscatterplots(
            hofs=hofs,
            save_path=save_path_trade_cv(plot_name=SUBSCATTERPLOTS_CV_FILE + "_" + inner),
            ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=CATTELANI2023_SETUP)
        print("Processing CV tradeplots")
        subtradeplots(
            hofs=hofs,
            save_path=save_path_trade_cv(plot_name=SUBTRADEPLOTS_CV_FILE + "_" + inner),
            ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=CATTELANI2023_SETUP)
