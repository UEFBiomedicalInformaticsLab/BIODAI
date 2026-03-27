import sys
import socket
import datetime
import time
import cProfile
import pstats
import matplotlib
from abc import ABC, abstractmethod
from setup.evaluation_setup import EvaluationSetup
from util.printer.printer import Printer
from consts import DEFAULT_RECURSION_LIMIT, FULL_STACK_WARNINGS, PROFILE_FILE
from setup.setup_reader import read_all_setups_in_argv
from util.concurrent.exclusive_number import ExclusiveNumber
from util.printer.printer import LogAndOutPrinterUnbuffered
from util.str_utils import pretty_duration
import warnings
import traceback


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else None
    traceback.print_stack(file=log)
    print(f"{filename}:{lineno}: {category.__name__}: {message}", file=log)



class SetupRunner(ABC):

    def run_setups(self):
        if FULL_STACK_WARNINGS:
            warnings.filterwarnings("always")
            warnings.simplefilter("always")
            warnings.warn("DEBUG: stacktrace", stacklevel=2)
            warnings.showwarning = warn_with_traceback
        matplotlib.use('Agg')
        with ExclusiveNumber() as exclusive_number:
            log_file_name = "temp/log" + str(exclusive_number) + ".txt"
            print("Writing main log to file " + log_file_name)
            printer = LogAndOutPrinterUnbuffered(log_file=log_file_name)

            printer.title_print("Executing " + self.title())

            printer.print_variable("Date", datetime.date.today())
            printer.print_variable("Machine", socket.gethostname())

            sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)
            printer.print_variable("Recursion limit", sys.getrecursionlimit())

            setups = read_all_setups_in_argv(printer=printer)  # We parse all of them immediately to catch some errors.
            start_time = time.time()

            for i in range(len(setups)):
                self.run_one_setup(setup=setups[i], printer=printer, config_file=sys.argv[i + 1])

            printer.print("Program finished")
            printer.print_variable("Total execution time", pretty_duration(time.time() - start_time))

    def run_setups_with_profiler(self):
        with cProfile.Profile() as pr:
            self.run_setups()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=PROFILE_FILE)

    @abstractmethod
    def run_one_setup(self, setup: EvaluationSetup, printer: Printer, config_file: str = None):
        raise NotImplementedError()

    @abstractmethod
    def title(self) -> str:
        raise NotImplementedError()