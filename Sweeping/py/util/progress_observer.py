import time
from abc import ABC, abstractmethod
from typing import Optional

from util.printer.printer import OutPrinter, Printer, UnbufferedOutPrinter
from util.str_utils import pretty_duration


class ProgressObserver(ABC):

    @abstractmethod
    def notify_start(self):
        raise NotImplementedError()

    @abstractmethod
    def notify_progress(self, proportion: Optional[float] = None, text: Optional[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def notify_message(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def notify_end(self, report: Optional[str] = None):
        raise NotImplementedError()


class ProgressObserverFactory(ABC):

    @abstractmethod
    def create_progress_observer(self, job_name: Optional[str] = None) -> ProgressObserver:
        raise NotImplementedError()


class NullProgressObserver(ProgressObserver):

    def notify_start(self):
        pass

    def notify_progress(self, proportion: Optional[float] = None, text: Optional[str] = None):
        pass

    def notify_message(self, text: str):
        pass

    def notify_end(self, report: Optional[str] = None):
        pass


class NullProgressObserverFactory(ProgressObserverFactory):

    def create_progress_observer(self, job_name: Optional[str] = None) -> ProgressObserver:
        return NULL_PROGRESS_OBSERVER


class SmartProgressObserver(ProgressObserver):
    __job_name: Optional[str]
    __start_time: Optional[float]
    __prev_time: Optional[float]
    __minutes_of_quiet: int
    __printer: Printer

    def __pretty_job_name(self) -> str:
        if self.__job_name is None:
            return "job"
        else:
            return self.__job_name

    def __init__(self, job_name: Optional[str] = None, printer: Printer = OutPrinter(), minutes_of_quiet: int = 30):
        """Minutes of quiet: if zero there is no quiet."""
        self.__job_name = job_name
        self.__minutes_of_quiet = minutes_of_quiet
        self.__printer = printer
        self.__start_time = None
        self.__prev_time = None

    def notify_start(self):
        self.__start_time = time.time()
        self.__prev_time = self.__start_time
        self.__printer.title_print(self.__pretty_job_name())

    def notify_progress(self, proportion: Optional[float] = None, text: Optional[str] = None):
        must_write = False
        if self.__minutes_of_quiet == 0:
            must_write = True
        current_time = time.time()
        if self.__prev_time is not None:
            elapsed_time = current_time - self.__prev_time  # In seconds.
            if (elapsed_time / 60) > self.__minutes_of_quiet:
                must_write = True
        else:
            self.__prev_time = current_time  # Let's start tracking from now.
        if must_write:
            to_print = self.__pretty_job_name() + ":"
            some_detail = False
            if proportion is not None:
                to_print += " " + "{:.2f}".format(proportion*100.0) + "%"
                some_detail = True
            if self.__prev_time is not None and self.__start_time is not None:
                to_print += " " + str(pretty_duration(time.time() - self.__start_time))
                some_detail = True
            if text is not None:
                to_print += " " + str(text)
                some_detail = True
            if not some_detail:
                to_print += " working"
            self.__printer.print(to_print)
            self.__prev_time = current_time

    def notify_message(self, text: str):
        to_print = self.__pretty_job_name()
        to_print += ": " + str(text)
        self.__printer.print(to_print)

    def notify_end(self, report: Optional[str] = None):
        to_print = self.__pretty_job_name() + " finished"
        if self.__start_time is not None:
            to_print += " in " + str(pretty_duration(time.time() - self.__start_time))
        self.__printer.print(to_print)
        if report is not None:
            self.__printer.print("Report:")
            self.__printer.print(str(report))


class SmartProgressObserverFactory(ProgressObserverFactory):
    __minutes_of_quiet: int
    __printer: Printer

    def __init__(self, printer: Printer = OutPrinter(), minutes_of_quiet: int = 30):
        self.__minutes_of_quiet = minutes_of_quiet
        self.__printer = printer

    def create_progress_observer(self, job_name: Optional[str] = None) -> ProgressObserver:
        return SmartProgressObserver(
            job_name=job_name, printer=self.__printer, minutes_of_quiet=self.__minutes_of_quiet)


NULL_PROGRESS_OBSERVER = NullProgressObserver()
NULL_PROGRESS_OBSERVER_FACTORY = NullProgressObserverFactory()

DEFAULT_PROGRESS_OBSERVER_FACTORY = SmartProgressObserverFactory(printer=UnbufferedOutPrinter(), minutes_of_quiet=1)