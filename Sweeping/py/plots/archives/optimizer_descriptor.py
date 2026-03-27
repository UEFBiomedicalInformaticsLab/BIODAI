from collections.abc import Sequence, Iterable
from typing import Optional

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from location_manager.location_manager_utils import default_pop_size, default_generations, COVARIATES_DEFAULT, \
    DEFAULT_CATEGORICAL_FI_STR, DEFAULT_SURVIVAL_FI_STR, VIEWS_DEFAULT
from plots.plot_labels import has_population, has_generations
from univariate_feature_selection.univariate_feature_selector_descriptor import (ManyFeatureSelectorClassDescriptor,
                                                                                 DEFAULT_CATEGORICAL_FS_DESCRIPTOR)
from util.printer.printer import Printer, NullPrinter
from views.adjusted_view_definition import AdjustedViewDef


class OptimizerDescriptor:
    __main_lab: str
    __inner_lab: Optional[str]  # Can be None if main algorithm does not have an inner model.
    __population: Optional[int]
    __generations: Optional[GenerationsStrategy]
    __categorical_fi_nick: str
    __survival_fi_nick: str
    __adjuster_regressor: Optional[str]
    views: AdjustedViewDef
    __covariate_set: set[str]
    __categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor

    def __init__(
            self,
            main_lab: str,
            inner_lab: Optional[str],
            population: Optional[int] = None,
            generations: Optional[GenerationsStrategy] = None,
            categorical_fi_nick: str = DEFAULT_CATEGORICAL_FI_STR,
            survival_fi_nick: str = DEFAULT_SURVIVAL_FI_STR,
            adjuster_regressor: Optional[str] = None,
            view_set: AdjustedViewDef = VIEWS_DEFAULT,
            covariate_set: Iterable[str] = COVARIATES_DEFAULT,
            categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR
    ):

        self.__main_lab = main_lab
        self.__inner_lab = inner_lab

        if has_population(main_lab=self.__main_lab):
            if population is None:
                population = default_pop_size(inner_lab=inner_lab)
        else:
            if population is not None:
                raise ValueError("This main algorithm does not use a population.")
        self.__population = population

        if has_generations(main_lab=self.__main_lab):
            if generations is None:
                generations = default_generations(main_lab=self.__main_lab)
        else:
            if generations is not None:
                raise ValueError("This main algorithm does not use a population.")
        self.__generations = generations

        self.__categorical_fi_nick = categorical_fi_nick
        self.__survival_fi_nick = survival_fi_nick
        self.__adjuster_regressor = adjuster_regressor
        self.views = view_set
        self.__covariate_set = set(covariate_set)
        self.__categorical_fs_descriptor = categorical_fs_descriptor

    def main_lab(self) -> str:
        return self.__main_lab

    def inner_lab(self) -> Optional[str]:
        return self.__inner_lab

    def generations(self) -> Optional[GenerationsStrategy]:
        return self.__generations

    def population(self) -> Optional[int]:
        return self.__population

    def adjuster_regressor(self) -> Optional[str]:
        return self.__adjuster_regressor

    def view_set(self) -> AdjustedViewDef:
        return self.views

    def covariate_set(self) -> set[str]:
        return self.__covariate_set

    def categorical_fs_descriptor(self) -> ManyFeatureSelectorClassDescriptor:
        return self.__categorical_fs_descriptor

    def is_included(
            self,
            main_labs: Sequence[str],
            inner_labs: Sequence[str],
            generations: Optional[Sequence[GenerationsStrategy]],
            population: Optional[Sequence[int]],
            categorical_fi_nicks: Sequence[str],
            survival_fi_nicks: Sequence[str],
            adjuster_regressors: Sequence[Optional[str]],
            view_defs: Sequence[AdjustedViewDef],
            covariate_sets: Sequence[set[str]],
            categorical_fs_descriptors: Iterable[
                ManyFeatureSelectorClassDescriptor],
            printer: Printer = NullPrinter()) -> bool:
        if not self.__main_lab in main_labs:
            printer.print("Main algorithm not included.")
            return False
        if self.__inner_lab is not None:
            if not self.__inner_lab in inner_labs:
                printer.print("Inner algorithm not included.")
                return False
        if generations is None:
            generations = [default_generations(main_lab=self.__main_lab)]
        if not self.generations() in generations:
            printer.print("Generations not included.")
            return False
        if self.__population is not None:
            if population is None:
                population = [default_pop_size(inner_lab=self.__inner_lab)]
            if not self.__population in population:
                printer.print("Population not included.")
                return False
        if not self.__adjuster_regressor in adjuster_regressors:
            printer.print("Adjuster regressor not included.")
            return False
        if not self.views in view_defs:
            printer.print("View set " + str(self.views) + " not included in " + str(view_defs) + ".")
            return False
        covariate_sets = [set(v) for v in covariate_sets]
        if not self.__covariate_set in covariate_sets:
            printer.print("Covariate set " + str(self.__covariate_set) + " not included in " + str(covariate_sets) + ".")
            return False
        if not self.__categorical_fi_nick in categorical_fi_nicks:
            return False
        if not self.__survival_fi_nick in survival_fi_nicks:
            return False
        if not self.__categorical_fs_descriptor in categorical_fs_descriptors:
            return False
        return True
