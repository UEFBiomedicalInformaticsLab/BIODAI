import multiprocessing
import random
import traceback

from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.mv_predictor import MVPredictor
from objective.objective_with_importance.social_objective_with_importance import SocialObjectiveWithImportance
from util.randoms import set_all_seeds
from cross_validation.validate_one_predictor_res import ValidateOnePredictorRes


class ValidateOnePredictorWorkerState:
    """ The random state is set at the same value, fixed at worker creation, each time a predictor is evaluated.
    In addition, the random state present when starting the evaluation of a predictor is saved and
    restored when the evaluation is finished."""
    __train_data: ModelReadyInputData
    __test_data: ModelReadyInputData
    __objective: SocialObjectiveWithImportance
    __compute_confidence: bool
    __seed: int

    def __init__(
            self,
            train_data: ModelReadyInputData,
            test_data: ModelReadyInputData,
            objective: SocialObjectiveWithImportance,
            compute_confidence: bool,
            seed: int = 83053):
        self.__train_data = train_data.fast_cols()
        self.__test_data = test_data.fast_cols()
        self.__objective = objective
        self.__compute_confidence = compute_confidence
        self.__seed = seed

    def validate(self, predictor: MVPredictor, hyperparams) -> ValidateOnePredictorRes:
        """Repeatability is guaranteed by always starting with the same seed. Random state is saved at the beginning
        and restored at the end to avoid affecting repeatability of the caller when the framework is run
        with a single process.
        This is the common entry point for evaluation for both sequential and parallel execution.
        """
        from cross_validation.cross_validation import validate_one_predictor
        try:
            rand_state = random.getstate()
            set_all_seeds(seed=self.__seed)
            res = validate_one_predictor(
                train_data=self.__train_data, test_data=self.__test_data,
                predictor=predictor, hyperparams=hyperparams, objective=self.__objective,
                compute_confidence=self.__compute_confidence)
            random.setstate(rand_state)
        except BaseException as e:
            current_process = multiprocessing.current_process()
            msg = "Exception during worker validation\n"
            msg += "Process pid: " + str(current_process.pid) + " name: " + current_process.name + "\n"
            msg += "Worker:\n" + str(self) + "\n"
            msg += "Processing predictor: " + str(predictor) + "\n"
            msg += "Processing hyperparams: " + str(hyperparams) + "\n"
            msg += "Exception content:\n"
            msg += str(e) + "\n"
            print(msg)
            raise Exception(msg)
        return res

    def __str__(self) -> str:
        ret_string = "ValidateOnePredictorWorkerState object"
        ret_string += "Seed: " + str(self.__seed) + "\n"
        return ret_string


def evaluate_by_worker(worker_state: ValidateOnePredictorWorkerState,
                       predictor: MVPredictor, hyperparams) -> ValidateOnePredictorRes:
    """Takes both cross_evaluator object and feature to evaluate so that the worker process has all that it needs."""
    return worker_state.validate(predictor=predictor, hyperparams=hyperparams)


def predictor_multiprocessing_friendly_validation(predictor: MVPredictor, hyperparams) -> ValidateOnePredictorRes:
    """This is supposed to happen inside a worker process."""
    try:
        return evaluate_by_worker(worker_state=_state_for_process, predictor=predictor, hyperparams=hyperparams)
    except BaseException:
        return traceback.format_exc()


def predictor_validation_parallel_init(worker_state: ValidateOnePredictorWorkerState):
    """When passed as a parameter to Pool(), what happens inside this function happens in a worker process."""
    global _state_for_process  # This global is inside a worker process.
    _state_for_process = worker_state
