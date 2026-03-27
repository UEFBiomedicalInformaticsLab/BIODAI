import multiprocessing
import random
import sys
from collections.abc import Sequence

from consts import DEFAULT_RECURSION_LIMIT
from cross_validation.multi_objective.mo_cv_result import MOCVResult
from evaluator.evaluate_individual_adj import evaluate_individual_adj
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.str_utils import str_in_lines
from worker.work_package import WorkPackage
from util.pickability_check import picklability_check
from util.printer.printer import Printer, UnbufferedOutPrinter

WORKER_RECURSION_LIMIT = DEFAULT_RECURSION_LIMIT


class WorkerState:
    """ The random state is set at the same value, fixed at worker creation, each time an objective is evaluated.
    In addition, the random state present when starting the evaluation of a whole individual is saved and
    restored when the evaluation is finished."""
    __input_data: InputData
    __folds_list: list[tuple[IndexArray,IndexArray]]
    __objectives: Sequence[PersonalObjectiveWithImportance]
    __hp_manager: MvHyperparamManager
    __compute_feature_importance: bool
    __compute_confidence: bool
    __printer: Printer
    __seed: int

    def __init__(self,
                 input_data: InputData,
                 folds_list: list[tuple[IndexArray,IndexArray]],
                 hp_manager: MvHyperparamManager,
                 objectives: Sequence[PersonalObjectiveWithImportance],
                 seed: int = 874390,
                 printer: Printer = UnbufferedOutPrinter(),
                 compute_feature_importance: bool = False,
                 compute_confidence: bool = False):
        # As of Python 3.9, there is no significant memory reduction in using numpy in place of pandas for views.
        self.__input_data = input_data
        self.__folds_list = folds_list
        self.__hp_manager = hp_manager
        self.__objectives = objectives
        self.__seed = seed
        self.__printer = printer
        self.__compute_feature_importance = compute_feature_importance
        self.__compute_confidence = compute_confidence

    def compile(self):
        """This can be called as an optimization if the worker state will be used in this same process.
        It uses memory to gain speed."""
        self.__input_data = self.__input_data.as_cached().compile()

    def hp_manager(self) -> MvHyperparamManager:
        return self.__hp_manager

    def evaluate(self, work_package: WorkPackage, check_picklability=False, verbose=False) -> MOCVResult:
        """Returns the fitnesses for the defined objectives, and the related predictors.
        The predictors are fitted on all the samples passed to the worker constructor.
        Repeatability is guaranteed by always starting with the same seed. Random state is saved at the beginning
        and restored at the end to avoid affecting repeatability of the caller when the framework is run
        with a single process.
        This is the common entry point for evaluation for both sequential and parallel execution.
        """
        try:
            if verbose:
                current_process = multiprocessing.current_process()
                msg = "Process pid: " + str(current_process.pid) + " name: " + current_process.name + "\n"
                msg += "Processing work package:\n"
                msg += str(work_package)
                self.__printer.print(msg)
            rand_state = random.getstate()
            res = evaluate_individual_adj(
                input_data=self.__input_data,
                folds_list=self.__folds_list,
                hp_manager=self.__hp_manager,
                individual=work_package.individual,
                objectives=self.__objectives,
                seed=self.__seed,
                compute_feature_importance=self.__compute_feature_importance,
                compute_confidence=self.__compute_confidence)
            random.setstate(rand_state)
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
        ret_string += "Input data:\n"
        ret_string += str(self.__input_data) + "\n"
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
    worker_state.compile()
