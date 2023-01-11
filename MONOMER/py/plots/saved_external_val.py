from collections.abc import Sequence

from plots.archives.automated_hofs_archive import nested_hofs_for_dataset_external
from plots.plot_labels import ALL_MAIN, ALL_INNER_LABS
from plots.saved_hof import SavedHoF


class SavedExternalVal:
    __internal_lab: str
    __external_nick: str
    __main_labs: list[str]
    __inner_labs: list[str]

    def __init__(self, internal_label: str, external_nick: str,
                 main_labs: list[str] = ALL_MAIN, inner_labs: list[str] = ALL_INNER_LABS):
        self.__internal_lab = internal_label
        self.__external_nick = external_nick
        self.__main_labs = main_labs
        self.__inner_labs = inner_labs

    def nested_hofs(self) -> Sequence[Sequence[SavedHoF]]:
        return nested_hofs_for_dataset_external(
            dataset_lab=self.__internal_lab, external_nick=self.__external_nick,
            main_labs=self.__main_labs, inner_labs=self.__inner_labs)

    def internal_label(self) -> str:
        return self.__internal_lab

    def external_nick(self) -> str:
        return self.__external_nick
