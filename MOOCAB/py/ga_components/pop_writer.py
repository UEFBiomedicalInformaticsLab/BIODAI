from ga_runner.progress_observer import LazyProgressObserver
from individual.peculiar_individual import PeculiarIndividual
from util.printer.printer import Printer, OutPrinter
from util.sequence_utils import str_in_lines
from util.math.summer import KahanSummer
from util.utils import str_sorted_dict


def write_pop(individuals: [PeculiarIndividual]) -> str:
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    fitnesses = [i.fitness for i in individuals]
    fit_values = [f.getValues() for f in fitnesses]
    cd_values = [f.get_crowding_distance() for f in fitnesses]
    n_features = [sum(i) for i in individuals]
    ind_strings =\
        [str(fit) + ", cd " + str(cd) + ", features " + str(n)
         for (fit, cd, n) in zip(fit_values, cd_values, n_features)]
    return str_in_lines(ind_strings)


def average_fitnesses(individuals: [PeculiarIndividual]) -> [float]:
    n_ind = len(individuals)
    if n_ind == 0:
        return None
    fitnesses = [i.fitness for i in individuals]
    fit_values = [f.getValues() for f in fitnesses]
    non_zero_fit_values = []
    for f in fit_values:
        if len(f) > 0:
            non_zero_fit_values.append(f)
    if len(non_zero_fit_values) == 0:
        return None
    n_objectives = len(fit_values[0])
    res = []
    for o in range(n_objectives):
        res.append(KahanSummer.mean([f[o] for f in non_zero_fit_values]))
    return res


def num_features_counts(individuals: [PeculiarIndividual]) -> [float]:
    res = {}
    n_features = [sum(i) for i in individuals]
    for n in n_features:
        if n in res:
            res[n] = res[n] + 1
        else:
            res[n] = 1
    return res


class PopStatsObserver(LazyProgressObserver):
    __printer: Printer
    __show_modified: bool

    def __init__(self, printer: Printer = OutPrinter(), show_modified: bool = False):
        self.__printer = printer
        self.__show_modified = show_modified

    def notify_initial_pop(self, pop: [PeculiarIndividual]):
        self.__printer.print("Initial pop mean fitness: " + str(average_fitnesses(pop)))

    def notify_tournament_offsprings(self, offsprings: [PeculiarIndividual]):
        self.__printer.print("Tournament selected offsprings mean fitness: " + str(average_fitnesses(offsprings)))

    def notify_modified_offsprings(self, offsprings: [PeculiarIndividual]):
        if self.__show_modified:  # May be not interesting since modified ones do not have valid fitnesses/cd
            self.__printer.print("Tournament selected offspring after crossover and mutation mean fitness: "
                                 + str(average_fitnesses(offsprings)))

    def notify_pop_before_select(self, pop: [PeculiarIndividual]):
        self.__printer.print("Pop before select: "
                             + str(average_fitnesses(pop)))

    def notify_pop_after_select(self, pop: [PeculiarIndividual]):
        self.__printer.print("Pop after nsga2 select: "
                             + str(average_fitnesses(pop)))


class PopWriterObserver(LazyProgressObserver):

    def notify_initial_pop(self, pop: [PeculiarIndividual]):
        print("Initial pop")
        print(write_pop(pop))

    def notify_tournament_offsprings(self, offsprings: [PeculiarIndividual]):
        print("Tournament selected offsprings")
        print(write_pop(offsprings))

    def notify_modified_offsprings(self, offsprings: [PeculiarIndividual]):
        print("Tournament selected offspring after crossover and mutation")
        print(write_pop(offsprings))

    def notify_pop_before_select(self, pop: [PeculiarIndividual]):
        print("Pop before select" + ", tot " + str(len(pop)))
        print(write_pop(pop))

    def notify_pop_after_select(self, pop: [PeculiarIndividual]):
        print("Pop after nsga2 select" + ", tot " + str(len(pop)))
        print(write_pop(pop))


class PopNumFeaturesObserver(LazyProgressObserver):
    __printer: Printer

    def __init__(self, printer: Printer = OutPrinter()):
        self.__printer = printer

    def notify_initial_pop(self, pop: [PeculiarIndividual]):
        self.__printer.print("Initial feature counts: " + str_sorted_dict(num_features_counts(pop)))

    def notify_tournament_offsprings(self, offsprings: [PeculiarIndividual]):
        self.__printer.print(
            "Tournament selected offsprings feature counts: " + str_sorted_dict(num_features_counts(offsprings)))

    def notify_modified_offsprings(self, offsprings: [PeculiarIndividual]):
        self.__printer.print("Tournament selected offspring after crossover and mutation feature counts: "
                             + str_sorted_dict(num_features_counts(offsprings)))

    def notify_pop_before_select(self, pop: [PeculiarIndividual]):
        self.__printer.print("Pop before select feature counts: "
                             + str_sorted_dict(num_features_counts(pop)) + ", tot " + str(len(pop)))

    def notify_pop_after_select(self, pop: [PeculiarIndividual]):
        self.__printer.print("Pop after nsga2 select feature counts: "
                             + str_sorted_dict(num_features_counts(pop)) + ", tot " + str(len(pop)))
