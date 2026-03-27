import multiprocessing
from collections.abc import Sequence
from typing import Any

from util.parallel.parallel_computer import ParallelComputer
from util.parallel.parallel_result import ParallelResult
from util.parallel.parallel_worker_state import ParallelWorkerState, parallel_worker_init, \
    multiprocessing_friendly_evaluation_with_init
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.progress_observer import SmartProgressObserverFactory
from util.progress_observer_for_jobs import ProgressObserverForJobs


def signal_job_done(po: ProgressObserverForJobs, result: ParallelResult):
    if not result.is_correct():
        po.message(text=f"Exception occurred: {result}")
    po.job_done()


def signal_job_done_global(result: ParallelResult):
    signal_job_done(po=_po, result=result)


class ParallelRunner:
    __computer: ParallelComputer
    __task_name: str

    def __init__(self, computer: ParallelComputer, task_name: str = "parallel task"):
        self.__computer = computer
        self.__task_name = task_name

    def run(self, worker_data: Any, jobs: Sequence, n_proc: int = 1,
            printer: Printer = UNBUFFERED_OUT_PRINTER, minutes_of_quiet: int = 1) -> list[ParallelResult]:
        cpu_count = multiprocessing.cpu_count()
        n_jobs = len(jobs)
        proc_to_use = max(1, min(n_proc, cpu_count, n_jobs))
        task_name = self.__task_name
        po = ProgressObserverForJobs(
            num_jobs=n_jobs, task_name=task_name, n_proc=proc_to_use,
            po_factory=SmartProgressObserverFactory(printer=printer, minutes_of_quiet=minutes_of_quiet))
        worker_state = ParallelWorkerState(worker_data=worker_data, computer=self.__computer, printer=printer)
        results = []
        if proc_to_use == 1:
            for j in jobs:
                j_res = worker_state.evaluate(j)
                signal_job_done(po=po, result=j_res)
                results.append(j_res)
        else:
            global _po
            _po = po
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(
                    processes=proc_to_use, initializer=parallel_worker_init, initargs=(worker_state,)) as workers_pool:
                results = [
                    workers_pool.apply_async(multiprocessing_friendly_evaluation_with_init, args=(j,),
                                             callback=signal_job_done_global) for j in jobs]
                results = [res.get() for res in results]  # Wait for all tasks to complete
                _po = None
        num_errors = 0
        for r in results:
            if not r.is_correct():
                num_errors += 1
        if num_errors == 0:
            printer.print("All jobs completed without errors.")
        else:
            printer.print("Of " + str(len(results)) + " jobs, " + str(num_errors) + " have failed.")
        return results

    def run_correctly(self, worker_data: Any, jobs: Sequence, n_proc: int = 1,
            printer: Printer = UNBUFFERED_OUT_PRINTER, minutes_of_quiet: int = 1) -> list:
        """Throws an exception if there are failed jobs. Unpackages the results."""
        results = self.run(
            worker_data=worker_data, jobs=jobs, n_proc=n_proc,
            printer=printer, minutes_of_quiet=minutes_of_quiet)
        for r in results:
            if not r.is_correct():
                raise RuntimeError("Job failure while running a sequence of jobs in strict mode.")
        return [r.result() for r in results]
