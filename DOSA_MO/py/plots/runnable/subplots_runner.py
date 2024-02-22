from consts import FONT_SIZE
from plots.default_labels_map import DEFAULT_LABELS_TRANSFORMER
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.plot_labels import ALL_MAIN_NO_NSGA3
from plots.saved_external_val import SavedExternalVal

SUBTRADEPLOTS_STR = "subtradeplots"
SUBSCATTERPLOTS_STR = "subscatterplots"
SUBSCATTERPLOTS_CV_DIR = SUBSCATTERPLOTS_STR + "_cv"
SUBSCATTERPLOTS_EXTERNAL_DIR = SUBSCATTERPLOTS_STR + "_external"
SUBTRADEPLOTS_CV_DIR = SUBTRADEPLOTS_STR + "_cv"
SUBTRADEPLOTS_EXTERNAL_DIR = SUBTRADEPLOTS_STR + "_external"

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
    labels_map=DEFAULT_LABELS_TRANSFORMER, font_size=FONT_SIZE)


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
