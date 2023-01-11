from cattelani2023.cattelani2023_utils import CATTELANI2023_DIR, CATTELANI2023_COL_I, CATTELANI2023_COL_J, CATTELANI2023_SETUP, CATTELANI2023_MAIN_LABS, CATTELANI2023_TCGA_DATASETS
from plots.archives.shallow_saved_hofs_archive_external import cattelani2023_external_validations
from plots.archives.automated_hofs_archive import nested_hofs_for_dataset_cv
from plots.plot_labels import ALL_INNER_LABS
from plots.saved_external_val import SavedExternalVal
from plots.subplots_by_strategy import subscatterplots, subtradeplots

SUBSCATTERPLOTS_CV_FILE = "subscatterplots_cv"
SUBSCATTERPLOTS_EXTERNAL_FILE = "subscatterplots_external"
SUBTRADEPLOTS_CV_FILE = "subtradeplots_cv"
SUBTRADEPLOTS_EXTERNAL_FILE = "subtradeplots_external"


def save_path_scatter_cv(plot_name: str) -> str:
    return CATTELANI2023_DIR + "/" + plot_name


def save_path_scatter_from_saved_ext_val(saved_ext_val: SavedExternalVal) -> str:
    return save_path_scatter_external_from_nicks(
        internal_nick=saved_ext_val.internal_label(), external_nick=saved_ext_val.external_nick())


def save_path_scatter_external_from_nicks(internal_nick: str, external_nick: str) -> str:
    return save_path_scatter_external(plot_name=internal_nick + "_" + external_nick)


def save_path_scatter_external(plot_name: str) -> str:
    return CATTELANI2023_DIR + "/" + plot_name


def save_path_trade_cv(plot_name: str) -> str:
    return CATTELANI2023_DIR + "/" + plot_name


def save_path_trade_external(plot_name: str) -> str:
    return CATTELANI2023_DIR + "/" + plot_name


if __name__ == '__main__':
    inner_labs = ALL_INNER_LABS
    n_inner = len(inner_labs)
    for i in range(n_inner):
        inner_1 = inner_labs[i]
        for j in range(i+1, n_inner):
            inner_2 = inner_labs[j]
            for k in range(j+1, n_inner):
                inner_3 = inner_labs[k]
                temp_inner_labs = [inner_1, inner_2, inner_3]
                print("Processing inner models " + inner_1 + ", " + inner_2 + " and " + inner_3)
                inner_nicks = inner_1 + "_" + inner_2 + "_" + inner_3
                ncols = len(temp_inner_labs)

                print("Processing external validations")
                external_hofs = []
                for ext in cattelani2023_external_validations(inner_labs=temp_inner_labs):
                    external_hofs.extend(ext.nested_hofs())
                print("Processing external scatterplots")
                subscatterplots(
                    hofs=external_hofs,
                    save_path=save_path_trade_external(
                        plot_name=SUBSCATTERPLOTS_EXTERNAL_FILE + "_" + inner_nicks),
                    ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
                    x_label="number of features", y_label="balanced accuracy",
                    setup=CATTELANI2023_SETUP)
                print("Processing external tradeplots")
                subtradeplots(
                    hofs=external_hofs,
                    save_path=save_path_trade_external(
                        plot_name=SUBTRADEPLOTS_EXTERNAL_FILE + "_" + inner_nicks),
                    ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
                    x_label="number of features", y_label="balanced accuracy",
                    setup=CATTELANI2023_SETUP)

                print("Processing cross-validations")

                hofs = []
                for dataset_label in CATTELANI2023_TCGA_DATASETS:
                    hofs.extend(nested_hofs_for_dataset_cv(
                        dataset_lab=dataset_label, main_labs=CATTELANI2023_MAIN_LABS, inner_labs=temp_inner_labs))
                print("Processing CV scatterplots")
                subscatterplots(
                    hofs=hofs,
                    save_path=save_path_trade_cv(plot_name=SUBSCATTERPLOTS_CV_FILE + "_" + inner_nicks),
                    ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
                    x_label="number of features", y_label="balanced accuracy",
                    setup=CATTELANI2023_SETUP)
                print("Processing CV tradeplots")
                subtradeplots(
                    hofs=hofs,
                    save_path=save_path_trade_cv(plot_name=SUBTRADEPLOTS_CV_FILE + "_" + inner_nicks),
                    ncols=ncols, col_x=CATTELANI2023_COL_I, col_y=CATTELANI2023_COL_J,
                    x_label="number of features", y_label="balanced accuracy",
                    setup=CATTELANI2023_SETUP)
