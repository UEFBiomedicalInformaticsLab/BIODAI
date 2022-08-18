from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.fitness.fitness import Fitness
from individual.individual_with_context import IndividualWithContext
from input_data.input_data import InputData
from model.model import Classifier, Predictor
from objective.social_objective import PersonalObjective
from util.hypervolume.hypervolume import hypervolume
from util.hyperbox.hyperbox import ConcreteHyperbox0B, Hyperbox0B
from util.printer.printer import Printer
from util.summer import KahanSummer
from views.views import Views


def hyperbox_from_fitness(fitness: Fitness) -> Hyperbox0B:
    return ConcreteHyperbox0B.create_by_b_vals(fitness.getValues())


def fold_inner_cv_hyperboxes(hyperparams: [IndividualWithContext]) -> [Hyperbox0B]:
    n_solutions = len(hyperparams)
    if n_solutions == 0:
        return []
    if hyperparams[0].has_fitness():
        return [hyperbox_from_fitness(h.fitness) for h in hyperparams]
    else:
        raise ValueError("Individuals missing fitness.")


def fold_inner_cv_hypervolume(hyperparams: [IndividualWithContext]) -> float | None:
    if len(hyperparams) == 0:
        return 0.0
    if hyperparams[0].has_fitness():
        return hypervolume(hyperboxes=fold_inner_cv_hyperboxes(hyperparams))
    else:
        return None


def fold_hyperboxes(x: Views, y: dict,
                    hyperparams: [IndividualWithContext], predictors: [[Predictor]],
                    objectives: [PersonalObjective]) -> Sequence[Hyperbox0B]:
    for o in objectives:
        if o.requires_training_predictions():
            raise ValueError()  # The fitness cannot be computed using only the testing set.
    hp_manager = DummyHpManager()
    hyperboxes = []
    for p, h in zip(predictors, hyperparams):
        fit = []
        for p_o, o in zip(p, objectives):
            if o.requires_predictions():
                true_for_o = y[o.outcome_label()]
                fit.append(o.compute_from_predictor_and_test(predictor=p_o, x_test=x, y_test=true_for_o))
            else:
                fit.append(o.compute_from_classes(hyperparams=h, hp_manager=hp_manager,
                                                  test_pred=None, test_true=None,
                                                  train_pred=None, train_true=None))
        hyperboxes.append(ConcreteHyperbox0B.create_by_b_vals(fit))
    return hyperboxes


def fold_classic_hypervolume(x: Views, y: dict,
                             hyperparams: [IndividualWithContext], predictors: [[Classifier]],
                             objectives: [PersonalObjective]) -> float | None:
    if len(hyperparams) == 0:
        return 0.0
    for o in objectives:
        if o.is_class_based() and o.requires_training_predictions():
            return None  # The fitness cannot be computed using only the training set.
    hyperboxes = fold_hyperboxes(x=x, y=y, hyperparams=hyperparams,
                                 predictors=predictors, objectives=objectives)
    return hypervolume(hyperboxes=hyperboxes)


class Hypervolumes(NamedTuple):
    inner_cv_hypervolume: float | None
    train_hypervolume: float | None
    test_hypervolume: float | None

    def __str__(self) -> str:
        res = ""
        if self.inner_cv_hypervolume is not None:
            res += "Inner cross-validation hypervolume: " + str(self.inner_cv_hypervolume) + "\n"
        if self.train_hypervolume is not None:
            res += "Train hypervolume: " + str(self.train_hypervolume) + "\n"
        if self.test_hypervolume is not None:
            res += "Test hypervolume: " + str(self.test_hypervolume) + "\n"
        return res


def hypervolumes_mean(hypervolumes_list: [Hypervolumes]) -> Hypervolumes:
    has_inner = hypervolumes_list[0].inner_cv_hypervolume is not None
    has_train = hypervolumes_list[0].train_hypervolume is not None
    has_test = hypervolumes_list[0].test_hypervolume is not None
    inner_sum = KahanSummer()
    train_sum = KahanSummer()
    test_sum = KahanSummer()
    for h in hypervolumes_list:
        if has_inner:
            inner_sum.add(h.inner_cv_hypervolume)
        if has_train:
            train_sum.add(h.train_hypervolume)
        if has_test:
            test_sum.add(h.test_hypervolume)
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
    return Hypervolumes(
        inner_cv_hypervolume=res_inner,
        train_hypervolume=res_train,
        test_hypervolume=res_test)


def fold_hypervolumes(x_train: Views, y_train: dict, x_test: Views, y_test: dict,
                      hyperparams: [IndividualWithContext], predictors: [[Classifier]],
                      objectives: [PersonalObjective]) -> tuple[float | None, float | None, float | None]:
    inner_cv_hypervolume = fold_inner_cv_hypervolume(hyperparams=hyperparams)
    train_hypervolume = fold_classic_hypervolume(x=x_train, y=y_train,
                                                 hyperparams=hyperparams, predictors=predictors,
                                                 objectives=objectives)
    test_hypervolume = fold_classic_hypervolume(x=x_test, y=y_test,
                                                hyperparams=hyperparams, predictors=predictors,
                                                objectives=objectives)
    return Hypervolumes(
        inner_cv_hypervolume=inner_cv_hypervolume,
        train_hypervolume=train_hypervolume,
        test_hypervolume=test_hypervolume)


class HypervolumeCrossEvaluator(MultiObjectiveCrossEvaluator):
    __objectives: [PersonalObjective]

    def __init__(self, objectives: list[PersonalObjective]):
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):

        all_folds_hvols = []

        for i in range(folds.n_folds()):

            x_train, y_train, x_test, y_test =\
                input_data.select_all_sets(
                    train_indices=folds.train_indices(fold_number=i),
                    test_indices=folds.test_indices(fold_number=i))
            x_train = x_train.as_cached()
            x_test = x_test.as_cached()
            non_dominated_predictors_with_hyperparams_i = non_dominated_predictors_with_hyperparams[i]
            predictors = non_dominated_predictors_with_hyperparams_i.predictors()
            hyperparams = non_dominated_predictors_with_hyperparams_i.hyperparams()

            fold_hvols = fold_hypervolumes(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                hyperparams=hyperparams, predictors=predictors, objectives=self.__objectives)
            printer.print("Hypervolumes for fold " + str(i))
            printer.print(str(fold_hvols))
            all_folds_hvols.append(fold_hvols)

        mean_hvols = hypervolumes_mean(all_folds_hvols)
        printer.print("Mean hypervolumes")
        printer.print(str(mean_hvols))

        return mean_hvols

    def name(self) -> str:
        return "hypervolume"
