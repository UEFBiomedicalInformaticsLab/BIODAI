from ga_components.sorter.pop_sorter import PopSorter
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual
from util.sequence_utils import flatten_iterable_of_iterable


class SorterWithCloneIndex(PopSorter):
    __inner_sorter: PopSorter

    def __init__(self, inner_sorter: PopSorter):
        self.__inner_sorter = inner_sorter

    @staticmethod
    def to_tuple(ind: PeculiarIndividual, hp_manager: HyperparamManager) -> ():
        return hp_manager.to_tuple(ind)

    def ci_fronts(self, pop: [PeculiarIndividual], hp_manager: HyperparamManager) -> [[PeculiarIndividual]]:
        ci_fronts = []
        tuple_to_ci = {}
        for i in pop:
            t = self.to_tuple(i, hp_manager)
            if t in tuple_to_ci:
                ci = tuple_to_ci[t] + 1
                tuple_to_ci[t] = ci
                if len(ci_fronts) > ci:
                    ci_fronts[ci].append(i)
                else:
                    ci_fronts.append([i])
            else:
                tuple_to_ci[t] = 0
                if len(ci_fronts) == 0:
                    ci_fronts.append([i])
                else:
                    ci_fronts[0].append(i)
        return ci_fronts

    def sort(self, pop: [PeculiarIndividual], hp_manager: HyperparamManager) -> [PeculiarIndividual]:
        fronts = self.ci_fronts(pop=pop, hp_manager=hp_manager)
        fronts = [self.__inner_sorter.sort(pop=f, hp_manager=hp_manager) for f in fronts]
        res = flatten_iterable_of_iterable(fronts)
        return res

    def nick(self) -> str:
        return self.__inner_sorter.nick() + "CI"

    def name(self) -> str:
        return self.__inner_sorter.name() + " with Clone Index"

    def __str__(self) -> str:
        return str(self.__inner_sorter) + " with Clone Index"

    def basic_algorithm_nick(self) -> str:
        return self.__inner_sorter.basic_algorithm_nick()
