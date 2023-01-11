from collections.abc import Sequence

from plots.archives.automated_hofs_archive import flatten_hofs_for_dataset_cv, flatten_hofs_for_dataset_external
from plots.archives.shallow_saved_hofs_archive_external import cattelani2023_external_validations
from plots.default_labels_map import default_labels_map
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.plot_labels import ALL_MAIN_NO_NSGA3, TCGA_BRCA_LAB, TCGA_KI_LAB, TCGA_LU_LAB, TCGA_OV_LAB
from plots.saved_hof import SavedHoF

CATTELANI2023_DIR = "Cattelani2023"

CATTELANI2023_MAIN_LABS = ALL_MAIN_NO_NSGA3

CATTELANI2023_X_MIN = -1
CATTELANI2023_X_MAX = 59
CATTELANI2023_Y_MIN = 0.10
CATTELANI2023_Y_MAX = 1.00

CATTELANI2023_COL_I = 1
CATTELANI2023_COL_J = 0
CATTELANI2023_ALPHA = 1.0

CATTELANI2023_SETUP = PlotSetup(
    x_min=CATTELANI2023_X_MIN, x_max=CATTELANI2023_X_MAX,
    y_min=CATTELANI2023_Y_MIN, y_max=CATTELANI2023_Y_MAX, alpha=CATTELANI2023_ALPHA,
    labels_map=default_labels_map, font_size=11)
CATTELANI2023_TCGA_DATASETS = [TCGA_BRCA_LAB, TCGA_KI_LAB, TCGA_LU_LAB, TCGA_OV_LAB]


def cattelani2023_internal_hofs() -> list[Sequence[SavedHoF]]:
    return [flatten_hofs_for_dataset_cv(dataset_lab=dataset_label, main_labs=CATTELANI2023_MAIN_LABS)
            for dataset_label in CATTELANI2023_TCGA_DATASETS]


def cattelani2023_external_hofs() -> list[Sequence[SavedHoF]]:
    external_hofs = []
    for ext in cattelani2023_external_validations():
        internal_label = ext.internal_label()
        external_nick = ext.external_nick()
        external_hofs.append(flatten_hofs_for_dataset_external(
            dataset_lab=internal_label, external_nick=external_nick, main_labs=CATTELANI2023_MAIN_LABS))
    return external_hofs
