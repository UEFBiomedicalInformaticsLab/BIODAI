from __future__ import annotations

from typing import Sequence

from saved_solutions.run_measure.run_best_dice import RunBestDice
from saved_solutions.run_measure.run_cross_hypervolume import RunCrossHypervolume
from saved_solutions.run_measure.run_dice import RunDice
from saved_solutions.run_measure.run_fold_jaccard import RunFoldJaccard
from saved_solutions.run_measure.run_inner_cv_hypervolume import RunInnerCVHypervolume
from saved_solutions.run_measure.run_pareto_delta import RunParetoDelta
from saved_solutions.run_measure.run_test_hypervolume import RunTestHypervolume
from saved_solutions.run_measure.run_weight_overlap import RunWeightOverlap
from util.cross_hypervolume.pareto_delta import PARETO_DELTA_NAME, PARETO_DELTA_NICK
from util.math.list_math import list_subtract, list_abs
from validation_registry.hof_property_computer import HofPropertyComputerFromFoldMeasure, \
    HofPropertyComputerOneMeasure, HofPropertyComputerWithFolds
from validation_registry.hof_property_computer_folds_mean import HofPropertyComputerFoldsMeanFromRegistry, \
    HofPropertyComputerFoldsMeanFromMeasure
from validation_registry.registry_property import RegistryProperty
from validation_registry.allowed_property_names import TEST_HV_NAME, CROSS_HV_NAME, FOLDS_TEST_HV_NAME, \
    FOLDS_CROSS_HV_NAME, MEAN_JACCARD_NAME, FOLDS_MEAN_JACCARD_NAME, STABILITY_BY_WEIGHTS_NAME, \
    STABILITY_BY_DICE_NAME, \
    TEST_HV_NICK, CROSS_HV_NICK, MEAN_JACCARD_NICK, STABILITY_BY_WEIGHTS_NICK, STABILITY_BY_DICE_NICK, \
    STABILITY_BY_BEST_DICE_NAME, STABILITY_BY_BEST_DICE_NICK, INNER_CV_HV_NAME, INNER_CV_HV_NICK, \
    FOLDS_INNER_CV_HV_NAME, PERFORMANCE_GAP_NAME, PERFORMANCE_GAP_NICK, FOLDS_PERFORMANCE_GAP_NAME, \
    PERFORMANCE_ERROR_NAME, PERFORMANCE_ERROR_NICK, FOLDS_PERFORMANCE_ERROR_NAME, FOLDS_PARETO_DELTA_NAME

INNER_CV_HV_PROPERTY = RegistryProperty(
    property_name=INNER_CV_HV_NAME, nick_for_humans=INNER_CV_HV_NICK, folds_property_name=FOLDS_INNER_CV_HV_NAME,
    computer=HofPropertyComputerFoldsMeanFromMeasure(measure=RunInnerCVHypervolume()),
    folds_computer=HofPropertyComputerFromFoldMeasure(measure=RunInnerCVHypervolume()))

TEST_HV_PROPERTY = RegistryProperty(
    property_name=TEST_HV_NAME, nick_for_humans=TEST_HV_NICK, folds_property_name=FOLDS_TEST_HV_NAME,
    computer=HofPropertyComputerFoldsMeanFromMeasure(measure=RunTestHypervolume()),
    folds_computer=HofPropertyComputerFromFoldMeasure(measure=RunTestHypervolume()))

CROSS_HV_PROPERTY = RegistryProperty(
    property_name=CROSS_HV_NAME, nick_for_humans=CROSS_HV_NICK, folds_property_name=FOLDS_CROSS_HV_NAME,
    computer=HofPropertyComputerFoldsMeanFromMeasure(measure=RunCrossHypervolume()),
    folds_computer=HofPropertyComputerFromFoldMeasure(measure=RunCrossHypervolume()))

MEAN_JACCARD_PROPERTY = RegistryProperty(
    property_name=MEAN_JACCARD_NAME, nick_for_humans=MEAN_JACCARD_NICK, folds_property_name=FOLDS_MEAN_JACCARD_NAME,
    computer=HofPropertyComputerFoldsMeanFromMeasure(measure=RunFoldJaccard()),
    folds_computer=HofPropertyComputerFromFoldMeasure(measure=RunFoldJaccard()))

STABILITY_BY_WEIGHTS_PROPERTY = RegistryProperty(
    property_name=STABILITY_BY_WEIGHTS_NAME, nick_for_humans=STABILITY_BY_WEIGHTS_NICK,
    computer=HofPropertyComputerOneMeasure(measure=RunWeightOverlap()))

STABILITY_BY_DICE_PROPERTY = RegistryProperty(
    property_name=STABILITY_BY_DICE_NAME, nick_for_humans=STABILITY_BY_DICE_NICK,
    computer=HofPropertyComputerOneMeasure(measure=RunDice()))

STABILITY_BY_BEST_DICE_PROPERTY = RegistryProperty(
    property_name=STABILITY_BY_BEST_DICE_NAME, nick_for_humans=STABILITY_BY_BEST_DICE_NICK,
    computer=HofPropertyComputerOneMeasure(measure=RunBestDice()))


class FoldsPerformanceGap(HofPropertyComputerWithFolds):

    def compute(self, hof_path: str) -> Sequence[float]:
        folds_inner_cv = INNER_CV_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        folds_chv = CROSS_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        return list_subtract(folds_inner_cv, folds_chv)

    def compute_fold(self, hof_path: str, fold: int) -> float:
        folds_inner_cv = INNER_CV_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        folds_chv = CROSS_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        return folds_inner_cv[fold] - folds_chv[fold]


PERFORMANCE_GAP_PROPERTY = RegistryProperty(
    property_name=PERFORMANCE_GAP_NAME, nick_for_humans=PERFORMANCE_GAP_NICK,
    folds_property_name=FOLDS_PERFORMANCE_GAP_NAME,
    computer=HofPropertyComputerFoldsMeanFromRegistry(
        folds_property_name=FOLDS_PERFORMANCE_GAP_NAME, folds_property_computer=FoldsPerformanceGap()),
    folds_computer=FoldsPerformanceGap())


class FoldsPerformanceError(HofPropertyComputerWithFolds):

    def compute(self, hof_path: str) -> Sequence[float]:
        folds_inner_cv = INNER_CV_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        folds_chv = CROSS_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        return list_abs(list_subtract(folds_inner_cv, folds_chv))

    def compute_fold(self, hof_path: str, fold: int) -> float:
        folds_inner_cv = INNER_CV_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        folds_chv = CROSS_HV_PROPERTY.smart_extract_folds(hof_path=hof_path)
        return abs(folds_inner_cv[fold] - folds_chv[fold])


PERFORMANCE_ERROR_PROPERTY = RegistryProperty(
    property_name=PERFORMANCE_ERROR_NAME, nick_for_humans=PERFORMANCE_ERROR_NICK,
    folds_property_name=FOLDS_PERFORMANCE_ERROR_NAME,
    computer=HofPropertyComputerFoldsMeanFromRegistry(
        folds_property_name=FOLDS_PERFORMANCE_ERROR_NAME, folds_property_computer=FoldsPerformanceError()),
    folds_computer=FoldsPerformanceError())


PARETO_DELTA_PROPERTY = RegistryProperty(
    property_name=PARETO_DELTA_NAME, nick_for_humans=PARETO_DELTA_NICK, folds_property_name=FOLDS_PARETO_DELTA_NAME,
    computer=HofPropertyComputerFoldsMeanFromMeasure(measure=RunParetoDelta()),
    folds_computer=HofPropertyComputerFromFoldMeasure(measure=RunParetoDelta()))
