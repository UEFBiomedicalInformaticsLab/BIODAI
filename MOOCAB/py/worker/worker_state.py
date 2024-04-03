import multiprocessing
import random
import sys

import pandas as pd
from numpy import ravel
from pandas import DataFrame

from consts import DEFAULT_RECURSION_LIMIT
from cross_validation.multi_objective.mo_cv_result import MOCVResult
from cross_validation.single_objective.cv_result import CVResult
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.randoms import set_all_seeds
from util.sequence_utils import str_in_lines
from worker.work_package import WorkPackage
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from hyperparam_manager.hyperparam_manager import HyperparamManager
from model.masked_mv_model import MaskedMVPredictor
from model.model import Classifier, ClassModel
from model.mv_predictor import MVPredictor
from multi_view_utils import filter_by_mask
from objective.social_objective import PersonalObjective
from util.dataframes import has_non_finite, has_non_finite_error
from util.pickability_check import picklability_check
from util.printer.printer import Printer, UnbufferedOutPrinter

WORKER_RECURSION_LIMIT = DEFAULT_RECURSION_LIMIT


class WorkerState:
    """ The random state is set at the same value, fixed at worker creation, each time an objective is evaluated.
    In addition, the random state present when starting the evaluation of a whole individual is saved and
    restored when the evaluation is finished."""
    __collapsed_views: pd.DataFrame
    __outcomes: dict[str, DataFrame]
    __objectives: [PersonalObjective]
    __hp_manager: HyperparamManager
    __compute_feature_importance: bool
    __compute_confidence: bool
    __printer: Printer
    __seed: int

    def __init__(self, collapsed_views: pd.DataFrame, outcomes: dict[str, DataFrame], folds_list,
                 hp_manager: HyperparamManager,
                 objectives: [PersonalObjective],
                 seed=874390,
                 printer: Printer = UnbufferedOutPrinter(),
                 compute_feature_importance: bool = False,
                 compute_confidence: bool = False):
        # As of Python 3.9, there is no significant memory reduction in using numpy in place of pandas for views.
        self.__collapsed_views = collapsed_views
        self.__outcomes = outcomes
        self.__folds_list = folds_list
        self.__hp_manager = hp_manager
        self.__objectives = objectives
        self.__seed = seed
        self.__printer = printer
        self.__compute_feature_importance = compute_feature_importance
        self.__compute_confidence = compute_confidence

    def collapsed_views(self) -> pd.DataFrame:
        return self.__collapsed_views

    def hp_manager(self) -> HyperparamManager:
        return self.__hp_manager

    def __get_outcome(self, outcome_key) -> DataFrame:
        return self.__outcomes[outcome_key]

    def __mask(self, individual: PeculiarIndividualByListlike):
        return self.__hp_manager.active_features_mask(hyperparams=individual)

    def __views_filtered(self, individual):
        mask = self.__mask(individual)
        return filter_by_mask(x=self.__collapsed_views, mask=mask)

    @staticmethod
    def __fit_inner_class_model(
            x_train_filtered, y_train, model: ClassModel, check_training_data: bool = False) -> Classifier:
        """Columns are assumed to be already filtered. Differently from calling model.fit directly, this
        method checks the input if check_training_data is True."""
        if check_training_data:  # Debug check.
            if has_non_finite(x_train_filtered):
                raise has_non_finite_error(x_train_filtered)
        fit_model = model.fit(x_train_filtered, y_train)
        return fit_model

    def __fold_data(self, individual, fold, y_key):
        all_x = self.__views_filtered(individual)
        all_y = self.__get_outcome(outcome_key=y_key)
        train_mask = fold[0]
        test_mask = fold[1]
        x_train = all_x.iloc[train_mask]
        y_train = all_y.iloc[train_mask]  # TODO This could be a generic Sequence
        x_test = all_x.iloc[test_mask]
        y_test = all_y.iloc[test_mask]
        return x_train, y_train, x_test, y_test

    def __results_one_fold(self, individual, fold, y_key, model: ClassModel):
        x_train, y_train, x_test, y_test = self.__fold_data(individual, fold, y_key)
        fit_model = self.__fit_inner_class_model(x_train, y_train, model=model)
        y_train_pred = fit_model.predict(x_train)
        y_test_pred = fit_model.predict(x_test)
        return y_train_pred, ravel(y_train), y_test_pred, ravel(y_test)

    def __fit_masked_model(self, individual, y_key, model: ClassModel) -> MVPredictor:
        """Fits on all samples using collapsed views."""
        inner_predictor = self.__fit_inner_class_model(
            x_train_filtered=self.__views_filtered(individual=individual),
            y_train=self.__get_outcome(outcome_key=y_key),
            model=model)
        predictor = MaskedMVPredictor(mask=self.__mask(individual), inner_predictor=inner_predictor)
        return predictor

    def __fit_model(self, individual: PeculiarIndividualByListlike, objective: PersonalObjectiveWithImportance
                    ) -> MVPredictor:
        if objective.has_model():
            y_key = objective.outcome_label()
            model = objective.model()
            return self.__fit_masked_model(individual=individual, y_key=y_key, model=model)
        else:
            raise ValueError("Objective does not have a model.")

    def __evaluate_objective(
            self, individual: PeculiarIndividualByListlike,
            objective: PersonalObjectiveWithImportance) -> CVResult:
        set_all_seeds(self.__seed)
        cv_result = None
        objective_computer = objective.objective_computer()
        all_x = self.__views_filtered(individual)
        if objective.has_outcome_label():
            all_y = self.__get_outcome(outcome_key=objective.outcome_label())
        else:
            all_y = None
        if objective.requires_predictions():
            if objective.has_model():
                model = objective.model()
                cv_result = objective_computer.compute_with_kfold_cv_with_importance(
                    model=model, x=all_x, y=all_y, folds_list=self.__folds_list,
                    compute_fi=self.__compute_feature_importance,
                    compute_confidence=self.__compute_confidence)
                cv_result.set_final_predictor(self.__fit_model(individual=individual, objective=objective))
            else:
                ValueError("Unexpected case.")
        else:
            cv_result = objective_computer.compute_from_structure_with_importance(
                hyperparams=individual,
                hp_manager=self.__hp_manager,
                x=all_x, y=all_y,
                compute_fi=self.__compute_feature_importance,
                compute_confidence=self.__compute_confidence
            )
        return cv_result

    def evaluate(self, work_package: WorkPackage, check_picklability=False, verbose=False) -> MOCVResult:
        """Returns the fitnesses for the defined objectives, and the related predictors.
        The predictors are fitted on all the samples passed to the worker constructor.
        Repeatability is guaranteed by always starting with the same seed. Random state is saved at the beginning
        and restored at the end to avoid affecting repeatability of the caller when the framework is run
        with a single process.
        This is the common entry point for evaluation for both sequential and parallel execution.
        TODO Would be more efficient to filter the columns of all samples at the beginning.
        """
        try:
            if verbose:
                current_process = multiprocessing.current_process()
                msg = "Process pid: " + str(current_process.pid) + " name: " + current_process.name + "\n"
                msg += "Processing work package:\n"
                msg += str(work_package)
                self.__printer.print(msg)
            rand_state = random.getstate()
            individual = work_package.individual
            objectives = self.__objectives
            so_results = [self.__evaluate_objective(individual=individual, objective=o) for o in objectives]
            random.setstate(rand_state)
            res = MOCVResult(so_results=so_results)
            if check_picklability:
                p_check_res = picklability_check(res)
                if not p_check_res.passed():
                    message = "Picklability check of result inside worker failed\n"
                    message += "Picklability check result:\n"
                    message += str(p_check_res) + "\n"
                    message += "MOCVResult:\n"
                    message += str(res) + "\n"
                    self.__printer.print(message)
                    raise Exception(message)
            if verbose:
                current_process = multiprocessing.current_process()
                msg = "Process pid: " + str(current_process.pid) + " name: " + current_process.name + "\n"
                msg += "Sending worker result:\n"
                msg += str(res)
                self.__printer.print(msg)
        except BaseException as e:
            current_process = multiprocessing.current_process()
            msg = "Exception during worker evaluation\n"
            msg += "Process pid: " + str(current_process.pid) + " name: " + current_process.name + "\n"
            msg += "Worker:\n" + str(self) + "\n"
            msg += "Processing work package:\n"
            msg += str(work_package)
            msg += "Exception content:\n"
            msg += str(e) + "\n"
            self.__printer.print(msg)
            raise Exception(msg)
        return res

    def __str__(self) -> str:
        ret_string = "WorkerState object with attributes:\n"
        ret_string += "Objectives:\n"
        ret_string += str_in_lines(self.__objectives) + "\n"
        ret_string += "Collapsed views:\n"
        ret_string += str(self.__collapsed_views) + "\n"
        # ret_string += "folds list:\n"
        # ret_string += str(self.__folds_list) + "\n"
        ret_string += "HP manager:\n"
        ret_string += str(self.__hp_manager) + "\n"
        ret_string += "Compute feature importance: " + str(self.__compute_feature_importance) + "\n"
        ret_string += "Seed: " + str(self.__seed) + "\n"
        ret_string += "Printer: " + str(self.__printer) + "\n"
        return ret_string


def evaluate_by_worker(worker_state: WorkerState, work_package: WorkPackage) -> MOCVResult:
    """Takes both cross_evaluator object and individual to evaluate so that the worker process has all that it needs."""
    return worker_state.evaluate(work_package=work_package)


def unpack_wp(work_package: WorkPackage):
    return work_package


def multiprocessing_friendly_evaluation_with_init(work_package: WorkPackage) -> MOCVResult:
    unpacked_wp = unpack_wp(work_package)
    return evaluate_by_worker(_state_for_process, unpacked_wp)


def parallel_init(worker_state: WorkerState):
    """When passed as a parameter to Pool(), what happens inside this function happens in a worker process."""
    global _state_for_process  # This global is inside a worker process.
    _state_for_process = worker_state
    sys.setrecursionlimit(WORKER_RECURSION_LIMIT)
