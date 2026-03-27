from typing import Optional

from util.progress_observer import ProgressObserver, ProgressObserverFactory, DEFAULT_PROGRESS_OBSERVER_FACTORY


class ProgressObserverForJobs:
    __num_jobs: int
    __jobs_done: int
    __progress_observer: ProgressObserver

    def __init__(self, num_jobs: int, task_name: str = "Running multiple jobs",
                 po_factory: ProgressObserverFactory = DEFAULT_PROGRESS_OBSERVER_FACTORY,
                 n_proc: Optional[int] = None):
        self.__num_jobs = num_jobs
        self.__jobs_done = 0
        self.__progress_observer = po_factory.create_progress_observer(job_name=task_name)
        self.__progress_observer.notify_start()
        if n_proc is not None:
            self.__progress_observer.notify_message(text="Using " + str(n_proc) + " processes.")

    def job_done(self):
        self.__jobs_done += 1
        self.__progress_observer.notify_progress(
            proportion=self.__jobs_done/self.__num_jobs, text=str(self.__jobs_done)+"/"+str(self.__num_jobs))
        if self.__jobs_done == self.__num_jobs:
            self.__progress_observer.notify_end()

    def message(self, text: str):
        self.__progress_observer.notify_message(text=text)


