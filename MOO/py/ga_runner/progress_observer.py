import time
from abc import ABC, abstractmethod

from individual.peculiar_individual import PeculiarIndividual
from util.printer.printer import OutPrinter, Printer


class ProgressObserver(ABC):

    @abstractmethod
    def notify_initial_pop(self, pop: [PeculiarIndividual]):
        raise NotImplementedError()

    @abstractmethod
    def notify_tournament_offsprings(self, offsprings: [PeculiarIndividual]):
        raise NotImplementedError()

    @abstractmethod
    def notify_modified_offsprings(self, offsprings: [PeculiarIndividual]):
        raise NotImplementedError()

    @abstractmethod
    def notify_pop_before_select(self, pop: [PeculiarIndividual]):
        raise NotImplementedError()

    @abstractmethod
    def notify_pop_after_select(self, pop: [PeculiarIndividual]):
        raise NotImplementedError()

    @abstractmethod
    def notify_generation_end(self, gen: int):
        raise NotImplementedError()


class LazyProgressObserver(ProgressObserver):

    def notify_initial_pop(self, pop: [PeculiarIndividual]):
        pass

    def notify_tournament_offsprings(self, offsprings: [PeculiarIndividual]):
        pass

    def notify_modified_offsprings(self, offsprings: [PeculiarIndividual]):
        pass

    def notify_pop_before_select(self, pop: [PeculiarIndividual]):
        pass

    def notify_pop_after_select(self, pop: [PeculiarIndividual]):
        pass

    def notify_generation_end(self, gen: int):
        pass


class SmartProgressObserver(LazyProgressObserver):
    __prev_time: float
    __minutes_of_quiet: int
    __printer: Printer

    def __init__(self, printer: Printer = OutPrinter(), minutes_of_quiet: int = 30):
        self.__prev_time = time.time()
        self.__minutes_of_quiet = minutes_of_quiet
        self.__printer = printer

    def notify_generation_end(self, gen: int):
        elapsed_time = time.time() - self.__prev_time  # In seconds.
        if (elapsed_time / 60) > self.__minutes_of_quiet:
            self.__printer.print_variable("Generations completed", gen)
            self.__prev_time = time.time()
