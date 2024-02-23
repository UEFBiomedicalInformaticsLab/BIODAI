import sys
import socket
import datetime
import time

from consts import DEFAULT_RECURSION_LIMIT
from external_validation.run_one_external_validation import run_one_external_validation
from setup.setup_reader import read_all_setups_in_argv
from util.concurrent.exclusive_number import ExclusiveNumber
from util.printer.printer import LogAndOutPrinterUnbuffered
from util.utils import pretty_duration


def external_validator():
    with ExclusiveNumber() as exclusive_number:

        log_file_name = "temp/log" + str(exclusive_number) + ".txt"
        print("Writing main log to file " + log_file_name)
        printer = LogAndOutPrinterUnbuffered(log_file=log_file_name)

        printer.title_print("Executing external validations")

        printer.print_variable("Date", datetime.date.today())
        printer.print_variable("Machine", socket.gethostname())

        sys.setrecursionlimit(DEFAULT_RECURSION_LIMIT)
        printer.print_variable("Recursion limit", sys.getrecursionlimit())

        setups = read_all_setups_in_argv(printer=printer)  # We parse all of them immediately to catch some errors.
        start_time = time.time()

        for i in range(len(setups)):
            run_one_external_validation(
                setup=setups[i], printer=printer, config_file=sys.argv[i+1])

        printer.print("Program finished")
        printer.print_variable("Total execution time", pretty_duration(time.time() - start_time))


if __name__ == '__main__':
    external_validator()
