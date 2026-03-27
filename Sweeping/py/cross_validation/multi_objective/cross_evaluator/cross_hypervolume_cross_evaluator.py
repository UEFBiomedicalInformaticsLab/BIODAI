from __future__ import annotations

import time
from collections.abc import Sequence
from typing import NamedTuple, Optional, Union

from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from individual.fitness.fitness import Fitness
from individual.individual_with_context import IndividualWithContext
from input_data.evaluation_ready_input_data import NoOutcomesInputData
from input_data.input_data import InputData
from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from objective.social_objective import PersonalObjective
from util.cross_hypervolume.cross_hypervolume import hypervolume, cross_hypervolume
from util.cross_hypervolume.hv_utils import DEFAULT_HV_PROCS
from util.hyperbox.hyperbox import ConcreteHyperbox0B, Hyperbox0B
from util.math.list_math import list_subtract, list_abs
from util.printer.printer import Printer
from util.math.summer import KahanSummer
from util.str_utils import pretty_duration
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry
from validation_registry.allowed_property_names import INNER_CV_HV_NAME, TRAIN_HV_NAME, TEST_HV_NAME, CROSS_HV_NAME, \
    FOLDS_INNER_CV_HV_NAME, FOLDS_TRAIN_HV_NAME, FOLDS_TEST_HV_NAME, FOLDS_CROSS_HV_NAME, FOLDS_PERFORMANCE_GAP_NAME, \
    PERFORMANCE_GAP_NAME, PERFORMANCE_ERROR_NAME, FOLDS_PERFORMANCE_ERROR_NAME


def hyperbox_from_fitness(fitness: Fitness) -> Hyperbox0B:
    return ConcreteHyperbox0B.create_by_b_vals(fitness.getValues())


def fold_inner_cv_hyperboxes(hyperparams: Sequence[IndividualWithContext]) -> Sequence[Hyperbox0B]:
    n_solutions = len(hyperparams)
    if n_solutions == 0:
        return []
    if hyperparams[0].has_fitness():
        return [hyperbox_from_fitness(h.fitness) for h in hyperparams]
    else:
        raise ValueError("Individuals missing fitness.")


def fold_inner_cv_hypervolume(hyperparams: Sequence[IndividualWithContext],
                              n_proc: Optional[int] = DEFAULT_HV_PROCS) -> Optional[float]:
    if len(hyperparams) == 0:
        return 0.0
    if hyperparams[0].has_fitness():
        return hypervolume(hyperboxes=fold_inner_cv_hyperboxes(hyperparams), n_proc=n_proc)
    else:
        return None


def evaluation_ready_for_objective(data: InputData, objective: PersonalObjective
                                   ) -> Union[ModelReadyInputData, NoOutcomesInputData]:
    if objective.has_outcome_label():
        return data.model_ready(outcome=objective.outcome_label())
    else:
        return NoOutcomesInputData(
            all_views=data.views(), adjusted_views=data.adjusted_view_def(), nick=data.nick())


def fold_hyperboxes(test_data: InputData,
                    hyperparams: Sequence[IndividualWithContext], predictors: Sequence[Sequence[MVPredictor]],
                    objectives: Sequence[PersonalObjective]) -> Sequence[Hyperbox0B]:
    for o in objectives:
        if o.requires_training_predictions():
            raise ValueError()  # The fitness cannot be computed using only the testing set.
    hp_manager = MaskMvHpManager.create_from_input_data(input_data=test_data)
    hyperboxes = []
    for p, h in zip(predictors, hyperparams):
        fit = []
        for p_o, o in zip(p, objectives):
            eval_test_data = evaluation_ready_for_objective(data=test_data, objective=o)
            if o.requires_predictions():
                fit.append(o.compute_from_predictor_and_test_mv(predictor=p_o, test_data=eval_test_data).fitness())
            else:
                fit.append(o.objective_computer().compute_from_structure_with_importance(
                    hyperparams=h, hp_manager=hp_manager,
                    data=hp_manager.filter_evaluation_ready_data(hyperparams=h,data=eval_test_data)).fitness())
        hyperboxes.append(ConcreteHyperbox0B.create_by_b_vals(fit))
    return hyperboxes


def fold_classic_hypervolume(test_data: InputData,
                             hyperparams: Sequence[IndividualWithContext],
                             predictors: Sequence[Sequence[MVPredictor]],
                             objectives: Sequence[PersonalObjective],
                             n_proc: Optional[int] = DEFAULT_HV_PROCS) -> Optional[float]:
    if len(hyperparams) == 0:
        return 0.0
    for o in objectives:
        if o.is_class_based() and o.requires_training_predictions():
            return None  # The fitness cannot be computed using only the training set.
    hyperboxes = fold_hyperboxes(test_data=test_data, hyperparams=hyperparams,
                                 predictors=predictors, objectives=objectives)
    return hypervolume(hyperboxes=hyperboxes, n_proc=n_proc)


class MVClassifier:
    pass


def fold_cross_hypervolume(test_data: InputData,
                           hyperparams: Sequence[IndividualWithContext], predictors: Sequence[Sequence[MVPredictor]],
                           objectives: Sequence[PersonalObjective],
                           n_proc: Optional[int] = DEFAULT_HV_PROCS) -> Optional[float]:
    if len(hyperparams) == 0:
        return 0.0
    for o in objectives:
        if o.requires_training_predictions():
            return None  # We cannot compute the train hyperboxes.
    if hyperparams[0].has_fitness():
        inner_cv_hyperboxes = fold_inner_cv_hyperboxes(hyperparams=hyperparams)
    else:
        if len(hyperparams) == 1:  # Since we have just one solution its actual fitnesses have no impact
            inner_cv_hyperboxes = [ConcreteHyperbox0B.create_by_b_vals([1.0]*len(objectives))]
        else:
            return None
    test_hyperboxes = []
    hp_manager = MaskMvHpManager.create_from_input_data(input_data=test_data)
    for p, h in zip(predictors, hyperparams):
        test_fit = []
        for p_o, o in zip(p, objectives):
            eval_test_data = evaluation_ready_for_objective(data=test_data, objective=o)
            if o.requires_predictions():
                test_fit.append(
                    o.compute_from_predictor_and_test_mv(predictor=p_o, test_data=eval_test_data).fitness())
            else:
                test_fit.append(o.objective_computer().compute_from_structure_with_importance(
                    hyperparams=h, hp_manager=hp_manager,
                    data=hp_manager.filter_evaluation_ready_data(hyperparams=h,data=eval_test_data)).fitness())
        test_hyperboxes.append(ConcreteHyperbox0B.create_by_b_vals(test_fit))
    return cross_hypervolume(train_hyperboxes=inner_cv_hyperboxes, test_hyperboxes=test_hyperboxes, n_proc=n_proc)


class Hypervolumes(NamedTuple):
    inner_cv_hypervolume: Optional[float]
    train_hypervolume: Optional[float]
    test_hypervolume: Optional[float]
    cross_hypervolume: Optional[float]

    def __str__(self) -> str:
        res = ""
        if self.inner_cv_hypervolume is not None:
            res += "Inner cross-validation hypervolume: " + str(self.inner_cv_hypervolume) + "\n"
        if self.train_hypervolume is not None:
            res += "Train hypervolume: " + str(self.train_hypervolume) + "\n"
        if self.test_hypervolume is not None:
            res += "Test hypervolume: " + str(self.test_hypervolume) + "\n"
        if self.cross_hypervolume is not None:
            res += "Cross hypervolume: " + str(self.cross_hypervolume) + "\n"
        return res


def hypervolumes_mean(hypervolumes_list: Sequence[Hypervolumes]) -> Hypervolumes:
    has_inner = hypervolumes_list[0].inner_cv_hypervolume is not None
    has_train = hypervolumes_list[0].train_hypervolume is not None
    has_test = hypervolumes_list[0].test_hypervolume is not None
    has_cross = hypervolumes_list[0].cross_hypervolume is not None
    inner_sum = KahanSummer()
    train_sum = KahanSummer()
    test_sum = KahanSummer()
    cross_sum = KahanSummer()
    for h in hypervolumes_list:
        if has_inner:
            inner_sum.add(h.inner_cv_hypervolume)
        if has_train:
            train_sum.add(h.train_hypervolume)
        if has_test:
            test_sum.add(h.test_hypervolume)
        if has_cross:
            cross_sum.add(h.cross_hypervolume)
    n_items = len(hypervolumes_list)
    if has_inner:
        res_inner = inner_sum.get_sum() / n_items
    else:
        res_inner = None
    if has_train:
        res_train = train_sum.get_sum() / n_items
    else:
        res_train = None
    if has_test:
        res_test = test_sum.get_sum() / n_items
    else:
        res_test = None
    if has_cross:
        res_cross = cross_sum.get_sum() / n_items
    else:
        res_cross = None
    return Hypervolumes(
        inner_cv_hypervolume=res_inner,
        train_hypervolume=res_train,
        test_hypervolume=res_test,
        cross_hypervolume=res_cross)


def fold_hypervolumes(train_data: InputData, test_data: InputData,
                      hyperparams: Sequence[IndividualWithContext], predictors: Sequence[Sequence[MVPredictor]],
                      objectives: Sequence[PersonalObjective],
                      n_proc: Optional[int] = DEFAULT_HV_PROCS) -> Hypervolumes:
    inner_cv_hypervolume = fold_inner_cv_hypervolume(hyperparams=hyperparams, n_proc=n_proc)
    train_hypervolume = fold_classic_hypervolume(test_data=train_data,
                                                 hyperparams=hyperparams, predictors=predictors,
                                                 objectives=objectives, n_proc=n_proc)
    test_hypervolume = fold_classic_hypervolume(test_data=test_data,
                                                hyperparams=hyperparams, predictors=predictors,
                                                objectives=objectives, n_proc=n_proc)
    c_hypervolume = fold_cross_hypervolume(test_data=test_data,
                                           hyperparams=hyperparams, predictors=predictors,
                                           objectives=objectives, n_proc=n_proc)
    return Hypervolumes(
        inner_cv_hypervolume=inner_cv_hypervolume,
        train_hypervolume=train_hypervolume,
        test_hypervolume=test_hypervolume,
        cross_hypervolume=c_hypervolume)


class CrossHypervolumeCrossEvaluator(MultiObjectiveCrossEvaluator):
    __objectives: Sequence[PersonalObjective]

    def __init__(self, objectives: list[PersonalObjective]):
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: Sequence[MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry(),
                 n_proc: Optional[int] = None):

        all_folds_hvols = []

        start_time = time.time()

        for i in range(folds.n_folds()):

            train_data = input_data.select_samples(row_indices=folds.train_indices(fold_number=i)).as_cached()
            test_data = input_data.select_samples(row_indices=folds.test_indices(fold_number=i)).as_cached()

            non_dominated_predictors_with_hyperparams_i = non_dominated_predictors_with_hyperparams[i]
            predictors = non_dominated_predictors_with_hyperparams_i.predictors()
            hyperparams = non_dominated_predictors_with_hyperparams_i.hyperparams()

            fold_hvols = fold_hypervolumes(train_data=train_data, test_data=test_data,
                                           hyperparams=hyperparams, predictors=predictors, objectives=self.__objectives, n_proc=n_proc)
            printer.print("Hypervolumes for fold " + str(i))
            printer.print(str(fold_hvols))
            all_folds_hvols.append(fold_hvols)

        printer.print("Computation of hypervolumes finished in " + pretty_duration(time.time() - start_time))

        mean_hvols = hypervolumes_mean(all_folds_hvols)
        printer.print("Mean hypervolumes")
        printer.print(str(mean_hvols))

        has_inner_cv_hv = mean_hvols.inner_cv_hypervolume is not None
        has_chv = mean_hvols.cross_hypervolume is not None
        if has_inner_cv_hv:
            hof_registry.set_property(name=INNER_CV_HV_NAME, value=mean_hvols.inner_cv_hypervolume)
            hof_registry.set_property(name=FOLDS_INNER_CV_HV_NAME,
                                      value=[h.inner_cv_hypervolume for h in all_folds_hvols])
        if mean_hvols.train_hypervolume is not None:
            hof_registry.set_property(name=TRAIN_HV_NAME, value=mean_hvols.train_hypervolume)
            hof_registry.set_property(name=FOLDS_TRAIN_HV_NAME,
                                      value=[h.train_hypervolume for h in all_folds_hvols])
        if mean_hvols.test_hypervolume is not None:
            hof_registry.set_property(name=TEST_HV_NAME, value=mean_hvols.test_hypervolume)
            hof_registry.set_property(name=FOLDS_TEST_HV_NAME,
                                      value=[h.test_hypervolume for h in all_folds_hvols])
        if has_chv:
            hof_registry.set_property(name=CROSS_HV_NAME, value=mean_hvols.cross_hypervolume)
            hof_registry.set_property(name=FOLDS_CROSS_HV_NAME,
                                      value=[h.cross_hypervolume for h in all_folds_hvols])
        if has_inner_cv_hv and has_chv:
            fold_gaps = list_subtract(
                [h.inner_cv_hypervolume for h in all_folds_hvols], [h.cross_hypervolume for h in all_folds_hvols])
            hof_registry.set_property(name=PERFORMANCE_GAP_NAME, value=KahanSummer.mean(fold_gaps))
            hof_registry.set_property(name=FOLDS_PERFORMANCE_GAP_NAME, value=fold_gaps)
            fold_errs = list_abs(fold_gaps)
            hof_registry.set_property(name=PERFORMANCE_ERROR_NAME, value=KahanSummer.mean(fold_errs))
            hof_registry.set_property(name=FOLDS_PERFORMANCE_ERROR_NAME, value=fold_errs)

        return mean_hvols

    def name(self) -> str:
        return "cross hypervolume"
