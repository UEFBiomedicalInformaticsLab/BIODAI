from collections.abc import Sequence

from plots.archives.automated_hofs_archive import nested_hofs_for_dataset_external
from plots.saved_hof import SavedHoF


class SavedExternalVal:
    __internal_lab: str
    __external_nick: str

    def __init__(self, internal_label: str, external_nick: str):
        self.__internal_lab = internal_label
        self.__external_nick = external_nick

    def nested_hofs(self) -> Sequence[Sequence[SavedHoF]]:
        return nested_hofs_for_dataset_external(dataset_lab=self.__internal_lab, external_nick=self.__external_nick)

    def internal_label(self) -> str:
        return self.__internal_lab

    def external_nick(self) -> str:
        return self.__external_nick
