import multiprocessing
import os

from util.printer.printer import Printer, OutPrinter


def cpus_to_use(max_cpus: int, printer: Printer = OutPrinter) -> int:
    cpu_count = multiprocessing.cpu_count()
    printer.print_variable("CPUs detected in the system", cpu_count)
    n_workers = min(cpu_count, max_cpus)
    printer.print_variable("CPUs to be used", n_workers)
    return n_workers


def subdirectories(main_directory: str) -> list[str]:
    if os.path.isdir(main_directory):
        return [f.path for f in os.scandir(main_directory) if f.is_dir()]
    else:
        return []
