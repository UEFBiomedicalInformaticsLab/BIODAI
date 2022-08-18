import multiprocessing
import sys
import socket
import datetime
import time

from consts import DEFAULT_RECURSION_LIMIT, MAX_WORKERS
from external_validation.run_one_external_validation import run_one_external_validation
from setup.setup_reader import read_all_setups_in_argv
from util.concurrent.exclusive_number import ExclusiveNumber
from util.printer.printer import LogAndOutPrinter
from util.utils import pretty_duration


if __name__ == '__main__':

    with ExclusiveNumber() as exclusive_number:

        log_file_name = "temp/log" + str(exclusive_number) + ".txt"
        print("Writing main log to file " + log_file_name)
        printer = LogAndOutPrinter(log_file=log_file_name)

        printer.print_variable("Date", datetime.date.today())
        printer.print_variable("Machine", socket.gethostname())

        cpu_count = multiprocessing.cpu_count()
        printer.print_variable("CPUs detected in the system", cpu_count)
        n_workers = min(cpu_count, MAX_WORKERS)

        sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)
        printer.print_variable("Recursion limit", sys.getrecursionlimit())

        setups = read_all_setups_in_argv(printer=printer)  # We parse all of them immediately to catch some errors.
        start_time = time.time()

        for i in range(len(setups)):
            run_one_external_validation(
                setup=setups[i], printer=printer, n_workers=n_workers, config_file=sys.argv[i+1])

        printer.print("Program finished")
        printer.print_variable("Total execution time", pretty_duration(time.time() - start_time))
