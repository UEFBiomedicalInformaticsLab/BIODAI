from collections.abc import Sequence
from typing import Union

from location_manager.location_manager_utils import COVARIATES_DEFAULT, COX_FI_STR
from plots.archives.automated_hofs_archive import all_hof_combinations_external, existing_hofs
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel, BalAccLeanness
from plots.plot_labels import ALL_MAIN_LABS, ALL_INNER_LABS, DEFAULT_ADJUSTER_REGRESSORS_LABS
from plots.saved_hof import SavedHoF


class SavedExternalVal:
    """Legacy class that might be completely substituted by TestBatteryExternal."""
    __internal_lab: str
    __external_nick: str
    __main_labs: list[str]
    __inner_labs: list[str]
    __dir_from_inner_lab: ObjectivesDirFromLabel
    __survival_fi: str
    __adjuster_regressors: Sequence[Union[None, str]]
    __covariates: set[str]


    def __init__(self, internal_label: str, external_nick: str,
                 main_labs: list[str] = ALL_MAIN_LABS, inner_labs: list[str] = ALL_INNER_LABS,
                 dir_from_label: ObjectivesDirFromLabel = BalAccLeanness(),
                 survival_fi: str = COX_FI_STR,
                 adjuster_regressors: Sequence[Union[None, str]] = DEFAULT_ADJUSTER_REGRESSORS_LABS,
                 covariates: set[str] = COVARIATES_DEFAULT):
        self.__internal_lab = internal_label
        self.__external_nick = external_nick
        self.__main_labs = main_labs
        self.__inner_labs = inner_labs
        self.__dir_from_inner_lab = dir_from_label
        self.__survival_fi = survival_fi
        self.__adjuster_regressors = adjuster_regressors
        self.__covariates = covariates

    def nested_hofs(self) -> Sequence[Sequence[SavedHoF]]:
        res = []
        for inner in self.__inner_labs:
            hofs = all_hof_combinations_external(
                main_labs=self.__main_labs,
                dir_from_label=self.__dir_from_inner_lab,
                dataset_lab=self.__internal_lab,
                external_nick=self.__external_nick,
                inner_labs=[inner],
                survival_fis=[self.__survival_fi],
                adjuster_regressors=self.__adjuster_regressors,
                covariate_sets=[self.__covariates])
            existing = existing_hofs(hofs)
            if len(existing) > 0:
                res.append(existing)
        return res

    def flat_hofs(self) -> Sequence[SavedHoF]:
        hofs = all_hof_combinations_external(
            main_labs=self.__main_labs,
            dir_from_label=self.__dir_from_inner_lab,
            dataset_lab=self.__internal_lab,
            external_nick=self.__external_nick,
            inner_labs=self.__inner_labs,
            survival_fis=[self.__survival_fi],
            adjuster_regressors=self.__adjuster_regressors,
            covariate_sets=[self.__covariates])
        return existing_hofs(hofs)

    def internal_label(self) -> str:
        return self.__internal_lab

    def external_nick(self) -> str:
        return self.__external_nick
