import os
import sys
from abc import ABC, abstractmethod

import pathlib
from collections.abc import Iterable
from typing import Sequence

from util.named import Named
from util.sequence_utils import names, sequence_to_string, str_in_lines
from util.utils import name_value


class Printer(ABC, Named):

    @abstractmethod
    def print(self, s):
        raise NotImplementedError()

    def print_variable(self, var_name: str, var_value):
        self.print(name_value(var_name, var_value))

    def title_print(self, s: str):
        self.print("\n" + s.upper())

    def print_in_lines(self, obj):
        if isinstance(obj, Iterable) or isinstance(obj, Sequence):
            self.print(str_in_lines(obj))
        else:
            self.print(str(obj))


class CompositePrinter(Printer):
    __printers: list[Printer]

    def __init__(self, printers: list[Printer]):
        self.__printers = printers

    def print(self, s):
        for p in self.__printers:
            p.print(s)

    def title_print(self, s: str):
        """Not using inherited implementation since composing printers may have specific title_print"""
        for p in self.__printers:
            p.title_print(s)

    def name(self) -> str:
        return str(names(self.__printers))

    def __str__(self) -> str:
        return sequence_to_string(self.__printers)


class OutPrinter(Printer):

    def print(self, s):
        print(str(s))

    def name(self) -> str:
        return "out printer"

    def __str__(self) -> str:
        return self.name()


class UnbufferedOutPrinter(Printer):

    def print(self, s):
        print(str(s))
        sys.stdout.flush()

    def name(self) -> str:
        return "unbuffered out printer"

    def __str__(self) -> str:
        return self.name()


class NullPrinter(Printer):

    def print(self, s):
        pass

    def name(self) -> str:
        return "null printer"

    def __str__(self) -> str:
        return self.name()


class LogPrinter(Printer):
    """Does not buffer between writes."""

    def __init__(self, log_file: str):
        self.__log_file = log_file
        pathlib.Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
        with open(self.__log_file, "w") as f:
            f.truncate(0)

    def print(self, s):
        with open(self.__log_file, "a") as f:
            f.write(str(s) + "\n")

    def name(self) -> str:
        return "log printer"

    def __str__(self) -> str:
        return "log printer to " + self.__log_file


class LogAndOutPrinter(Printer):
    __inner: Printer

    def __init__(self, log_file: str):
        self.__inner = CompositePrinter(printers=[OutPrinter(), LogPrinter(log_file=log_file)])

    def print(self, s):
        self.__inner.print(s)

    def name(self) -> str:
        return self.__inner.name()

    def __str__(self) -> str:
        return str(self.__inner)


class LogAndOutPrinterUnbuffered(Printer):
    __inner: Printer

    def __init__(self, log_file: str):
        self.__inner = CompositePrinter(printers=[UnbufferedOutPrinter(), LogPrinter(log_file=log_file)])

    def print(self, s):
        self.__inner.print(s)

    def name(self) -> str:
        return self.__inner.name()

    def __str__(self) -> str:
        return str(self.__inner)
