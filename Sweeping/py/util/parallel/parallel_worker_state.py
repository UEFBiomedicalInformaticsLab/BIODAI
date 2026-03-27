import multiprocessing
import sys
import traceback
from typing import Any

from consts import DEFAULT_RECURSION_LIMIT
from util.parallel.parallel_computer import ParallelComputer
from util.parallel.parallel_result import CorrectParallelResult, WrongParallelResult
from util.randoms import random_state_context
from util.printer.printer import Printer, UnbufferedOutPrinter


PARALLEL_WORKER_RECURSION_LIMIT = DEFAULT_RECURSION_LIMIT


class ParallelWorkerState:
    """ The random state is set at the same value, fixed at worker creation, each time a job is processed.
    In addition, the random state present when starting the evaluation of a job is saved and
    restored when the evaluation is finished."""
    __worker_data: Any
    __computer: ParallelComputer
    __printer: Printer
    __seed: int

    def __init__(self, worker_data: Any,
                 computer: ParallelComputer,
                 seed=27875,
                 printer: Printer = UnbufferedOutPrinter()):
        """Covariates must not contain missing values."""
        self.__worker_data = worker_data
        self.__computer = computer
        self.__seed = seed
        self.__printer = printer

    def evaluate(self, job: Any) -> Any:
        """Repeatability is guaranteed by always starting with the same seed. Random state is saved at the beginning
        and restored at the end to avoid affecting repeatability of the caller in case the framework is run
        with a single process.
        """
        computer = self.__computer
        printer = self.__printer
        try:
            with random_state_context(seed=self.__seed):
                res = computer.compute(common_data=self.__worker_data, job=job)
                return CorrectParallelResult(result=res)
        except BaseException as e:
            current_process = multiprocessing.current_process()
            msg = "Exception during worker evaluation\n"
            msg += "Process pid: " + str(current_process.pid) + " name: " + current_process.name + "\n"
            msg += "Worker:\n" + str(self) + "\n"
            msg += "Processing job: " + str(job) + "\n"
            computer_error_msg = computer.error_message(job=job)
            if computer_error_msg is not None:
                msg += str(computer_error_msg) + "\n"
            msg += "Exception content:\n"
            msg += str(e) + "\n"
            printer.print(msg)
            return WrongParallelResult(message=msg, traceback=traceback.format_exc())

    def __str__(self) -> str:
        ret_string = "WorkerState object with attributes:\n"
        ret_string += "Data:\n"
        ret_string += str(self.__worker_data) + "\n"
        ret_string += "Computer:\n"
        ret_string += str(self.__computer) + "\n"
        ret_string += "Seed: " + str(self.__seed) + "\n"
        ret_string += "Printer: " + str(self.__printer) + "\n"
        return ret_string


def multiprocessing_friendly_evaluation_with_init(job: Any) -> Any:
    """This is supposed to happen inside a worker process."""
    return _state_for_process.evaluate(job=job)



def parallel_worker_init(worker_state: ParallelWorkerState):
    """When passed as a parameter to Pool(), what happens inside this function happens in a worker process."""
    global _state_for_process  # This global is inside a worker process.
    _state_for_process = worker_state
    sys.setrecursionlimit(PARALLEL_WORKER_RECURSION_LIMIT)