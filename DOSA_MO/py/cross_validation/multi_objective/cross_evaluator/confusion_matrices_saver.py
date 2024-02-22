from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import confusion_matrix

from cross_validation.folds import Folds
from location_manager.path_utils import path_for_saves
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from input_data.outcome import OutcomeType, Outcome
from model.mv_predictor import MVPredictor
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry
from views.views import Views


CONFUSION_MATRIX_STR = "confusion_matrix"
LABEL_SEPARATOR = "_"


def confusion_matrix_from_predictor_and_test(predictor: MVPredictor, views: Views, outcome: Outcome) -> ndarray:
    """Confusion matrix has true labels in rows and predicted labels in columns.
    E.g.
    [[true0_pred0, true0_pred1, true0_pred2],
     [true1_pred0, true1_pred1, true1_pred2],
     [true2_pred0, true2_pred1, true2_pred2]]"""
    try:
        predictions = predictor.predict(views=views.as_dict())
    except TypeError as te:
        raise ValueError("TypeError " + str(te) + " with predictor " + str(predictor))
    y = outcome.fist_col()
    return confusion_matrix(y_true=y, y_pred=predictions, labels=outcome.class_labels())


def confusion_matrices_from_predictors_and_test(predictors: Sequence[MVPredictor], views: Views, outcome: Outcome
                                                ) -> list[ndarray]:
    """Confusion matrices have true labels in rows and predicted labels in columns.
    E.g.
    [[true0_pred0, true0_pred1, true0_pred2],
     [true1_pred0, true1_pred1, true1_pred2],
     [true2_pred0, true2_pred1, true2_pred2]]"""
    return [confusion_matrix_from_predictor_and_test(predictor=p, views=views, outcome=outcome) for p in predictors]


def cms_to_df(cms: Sequence[ndarray], labels: Sequence) -> DataFrame:
    """In df columns the first string is the true label and the second string is the prediction."""
    cols = [str(l1)+LABEL_SEPARATOR+str(l2) for l1 in labels for l2 in labels]
    return DataFrame(data=[cm.ravel() for cm in cms], columns=cols)


def save_hof_confusions(path_saves: str,
                        fold_index: Optional[int],
                        hof: MultiObjectiveOptimizerResult,
                        objectives: Sequence[PersonalObjective],
                        fold_test_data: InputData):
    Path(path_saves).mkdir(parents=True, exist_ok=True)
    views = fold_test_data.x().as_cached()
    for i, obj in enumerate(objectives):
        if obj.has_outcome_label():
            out_label = obj.outcome_label()
            outcome = fold_test_data.outcome(name=out_label)
            if outcome.type() == OutcomeType.categorical:
                predictors = hof.predictors_for_objective(objective_num=i)
                if len(predictors) > 0 and predictors[0] is not None:
                    cms = confusion_matrices_from_predictors_and_test(
                        predictors=predictors, views=views, outcome=outcome)
                    cm_df = cms_to_df(cms=cms, labels=outcome.class_labels())
                    if fold_index is None:
                        file = path_saves + out_label + "_test" + ".csv"
                    else:
                        file = path_saves + out_label + "_test_" + str(fold_index) + ".csv"
                    cm_df.to_csv(path_or_buf=file, index=False)


class ConfusionMatricesSaver(MultiObjectiveCrossEvaluator):
    __save_path: str
    __objectives: Sequence[PersonalObjective]

    def __init__(self, save_path: str, objectives: list[PersonalObjective]):
        self.__save_path = save_path
        self.__objectives = objectives

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        n_folds = len(non_dominated_predictors_with_hyperparams)
        if n_folds == 0:
            return None
        hof_nick = non_dominated_predictors_with_hyperparams[0].nick()
        path_saves = path_for_saves(
            base_path=self.__save_path, optimizer_nick=optimizer_nick, hof_nick=hof_nick) + CONFUSION_MATRIX_STR + "/"
        printer.print_variable(var_name="Path for hall of fames", var_value=path_saves)

        for fold in range(n_folds):
            hof = non_dominated_predictors_with_hyperparams[fold]
            fold_test_data = input_data.select_samples(row_indices=folds.test_indices(fold_number=fold))
            save_hof_confusions(path_saves=path_saves,
                                fold_index=fold,
                                hof=hof,
                                objectives=self.__objectives,
                                fold_test_data=fold_test_data)

    def name(self) -> str:
        return "Confusion matrices saver"
