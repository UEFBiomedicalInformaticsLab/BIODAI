from collections.abc import Sequence

from input_data.swedish_input_creator import SWEDISH_NICK
from plots.archives.automated_hofs_archive import flatten_hofs_for_dataset_external
from plots.archives.objectives_dir_from_label import BalAccRootLeannessCIndex
from plots.plot_labels import ALL_INNER_LABS, TCGA_BRCA_LAB, ALL_MAIN_LABS
from plots.saved_external_val import SavedExternalVal
from plots.saved_hof import SavedHoF


def survival_external_validations(main_labs: list[str] = ALL_MAIN_LABS,
                                  inner_labs: list[str] = ALL_INNER_LABS) -> list[SavedExternalVal]:
    return [
        SavedExternalVal(
            internal_label=TCGA_BRCA_LAB, external_nick=SWEDISH_NICK, main_labs=main_labs, inner_labs=inner_labs,
            dir_from_label=BalAccRootLeannessCIndex()),
    ]


def survival_external_hofs() -> list[Sequence[SavedHoF]]:
    external_hofs = []
    for ext in survival_external_validations():
        internal_label = ext.internal_label()
        external_nick = ext.external_nick()
        external_hofs.append(flatten_hofs_for_dataset_external(
            dataset_lab=internal_label, external_nick=external_nick, main_labs=ALL_MAIN_LABS,
            dir_from_label=BalAccRootLeannessCIndex()))
    return external_hofs
