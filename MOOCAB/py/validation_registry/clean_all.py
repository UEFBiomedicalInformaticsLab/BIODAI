from collections.abc import Iterable, Sequence

from consts import VALIDATION_REGISTRY_FILE_NAME
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from hall_of_fame.hof_utils import hof_path
from validation_registry.validation_registry import FileValidationRegistry


def clean_all_registries_cv(optimizer_save_path: str, folds_hofs: Iterable[Sequence[MultiObjectiveOptimizerResult]]):
    for hof in folds_hofs:
        hof_registry = FileValidationRegistry(
            file_path=hof_path(
                optimizer_save_path=optimizer_save_path, hof_nick=hof[0].nick()) + VALIDATION_REGISTRY_FILE_NAME)
        hof_registry.clean()


def clean_all_registries_external(optimizer_save_path: str, hofs: Iterable[MultiObjectiveOptimizerResult]):
    for hof in hofs:
        hof_registry = FileValidationRegistry(
            file_path=hof_path(
                optimizer_save_path=optimizer_save_path, hof_nick=hof.nick()) + VALIDATION_REGISTRY_FILE_NAME)
        hof_registry.clean()
