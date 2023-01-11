from consts import FONT_SIZE
from plots.archives.shallow_saved_hofs_archive_external import all_external_validations
from plots.default_labels_map import default_labels_map
from plots.archives.automated_hofs_archive import nested_hofs_for_dataset_cv
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.plot_labels import ALL_CV_DATASETS, ALL_MAIN_NO_NSGA3
from plots.saved_external_val import SavedExternalVal
from plots.subplots_by_strategy import subscatterplots, subtradeplots

SUBSCATTERPLOTS_CV_DIR = "subscatterplots_cv"
SUBSCATTERPLOTS_EXTERNAL_DIR = "subscatterplots_external"
SUBTRADEPLOTS_CV_DIR = "subtradeplots_cv"
SUBTRADEPLOTS_EXTERNAL_DIR = "subtradeplots_external"

MAIN_LABS = ALL_MAIN_NO_NSGA3

X_MIN = -1
X_MAX = 59
Y_MIN = 0.10
Y_MAX = 1.00

COL_I = 1
COL_J = 0
ALPHA = 1.0

SETUP = PlotSetup(
    x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, alpha=ALPHA,
    labels_map=default_labels_map, font_size=FONT_SIZE)


def save_path_scatter_cv(plot_name: str) -> str:
    return SUBSCATTERPLOTS_CV_DIR + "/" + plot_name


def save_path_scatter_from_saved_ext_val(saved_ext_val: SavedExternalVal) -> str:
    return save_path_scatter_external_from_nicks(
        internal_nick=saved_ext_val.internal_label(), external_nick=saved_ext_val.external_nick())


def save_path_scatter_external_from_nicks(internal_nick: str, external_nick: str) -> str:
    return save_path_scatter_external(plot_name=internal_nick + "_" + external_nick)


def save_path_scatter_external(plot_name: str) -> str:
    return SUBSCATTERPLOTS_EXTERNAL_DIR + "/" + plot_name


def save_path_trade_cv(plot_name: str) -> str:
    return SUBTRADEPLOTS_CV_DIR + "/" + plot_name


def save_path_trade_from_saved_ext_val(saved_ext_val: SavedExternalVal) -> str:
    return save_path_trade_external_from_nicks(
        internal_nick=saved_ext_val.internal_label(), external_nick=saved_ext_val.external_nick())


def save_path_trade_external_from_nicks(internal_nick: str, external_nick: str) -> str:
    return save_path_trade_external(plot_name=internal_nick + "_" + external_nick)


def save_path_trade_external(plot_name: str) -> str:
    return SUBTRADEPLOTS_EXTERNAL_DIR + "/" + plot_name


if __name__ == '__main__':
    for ext in all_external_validations(main_labs=MAIN_LABS):
        external_hofs = ext.nested_hofs()
        internal_label = ext.internal_label()
        print("Processing external validation " + str(ext.internal_label() + " - " + str(ext.external_nick())))
        subscatterplots(
            hofs=external_hofs,
            save_path=save_path_scatter_from_saved_ext_val(saved_ext_val=ext), ncols=2, col_x=COL_I, col_y=COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=SETUP)
        subtradeplots(
            hofs=external_hofs,
            save_path=save_path_trade_from_saved_ext_val(saved_ext_val=ext), ncols=2, col_x=COL_I, col_y=COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=SETUP)

    for dataset_label in ALL_CV_DATASETS:
        print("Processing dataset " + str(dataset_label))
        hofs = nested_hofs_for_dataset_cv(dataset_lab=dataset_label, main_labs=MAIN_LABS)
        subscatterplots(
            hofs=hofs,
            save_path=save_path_scatter_cv(plot_name=dataset_label), ncols=2, col_x=COL_I, col_y=COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=SETUP)
        subtradeplots(
            hofs=hofs,
            save_path=save_path_trade_cv(plot_name=dataset_label), ncols=2, col_x=COL_I, col_y=COL_J,
            x_label="number of features", y_label="balanced accuracy",
            setup=SETUP)
