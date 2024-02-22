from collections.abc import Sequence

import numpy
from deap import tools
from deap.tools import sortNondominated
from deap.tools.emo import find_extreme_points, find_intercepts, associate_to_niche, niching
from numpy import ndarray

from cross_validation.multi_objective.optimizer.nsga.nsga_types import NSGA3_TYPE
from ga_components.sorter.pop_sorter import PopSorter
from hyperparam_manager.hyperparam_manager import HyperparamManager
from individual.peculiar_individual import PeculiarIndividual


class Nsga3(PopSorter):
    """In this implementation the reference lines are always created adaptively at each generation and at each
       front taking into account the extreme values of that same generation and of the current and better fronts.
       Notice that from the point of view of the selection the same individuals of the DEAP implementation and of
       algorithm 1 and 2 of the original paper [Deb2014] are selected. Additionally, all fronts are sorted.

       This implementation is based on the DEAP implementation and much code is shared.

        [Deb2014] Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
            Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
            Part I: Solving Problems With Box Constraints. IEEE Transactions on
            Evolutionary Computation, 18(4), 577-601. doi:10.1109/TEVC.2013.2281535."""

    __reference_points: ndarray

    def __init__(self, num_objectives: int, max_reference_points: int):
        ref_points = numpy.zeros(num_objectives)
        best_points = ref_points
        p = 1  # Does not work with p = 0
        while len(ref_points) < max_reference_points:
            ref_points = tools.uniform_reference_points(nobj=num_objectives, p=p)
            if len(ref_points) <= max_reference_points:
                best_points = ref_points
            p = p+1
        self.__reference_points = best_points

    def sort_front(self, all_fitnesses: ndarray, front_fitnesses: ndarray, front: Sequence[PeculiarIndividual]
                   ) -> Sequence[PeculiarIndividual]:
        ref_points = self.__reference_points
        best_point = numpy.min(all_fitnesses, axis=0)
        worst_point = numpy.max(all_fitnesses, axis=0)

        extreme_points = find_extreme_points(all_fitnesses, best_point)
        intercepts = find_intercepts(extreme_points, best_point, worst_point, worst_point)
        niches, dist = associate_to_niche(all_fitnesses, ref_points, best_point, intercepts)
        # We have to associate to niche again for each front added since associations of previous fronts may
        # change when considering also solutions from current front that can modify reference lines.

        len_previous_fronts = all_fitnesses.shape[0] - front_fitnesses.shape[0]
        # Get counts per niche for individuals in all fronts but the last
        niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
        index, counts = numpy.unique(niches[:len_previous_fronts], return_counts=True)
        niche_counts[index] = counts

        return niching(front, len(front), niches[len_previous_fronts:], dist[len_previous_fronts:], niche_counts)
        # Niching behaves like the returned list was filled one by one so the returned list is properly sorted.

    def sort(self, pop: Sequence[PeculiarIndividual], hp_manager: HyperparamManager) -> Sequence[PeculiarIndividual]:
        pareto_fronts = sortNondominated(individuals=pop, k=len(pop))
        res = []
        fitnesses = None
        for front in pareto_fronts:
            # Extract fitnesses as a numpy array in the nd-sort order
            # Use wvalues * -1 to tackle always as a minimization problem
            front_fitnesses = numpy.array([ind.fitness.wvalues for ind in front])
            front_fitnesses *= -1
            if fitnesses is None:
                fitnesses = front_fitnesses
            else:
                fitnesses = numpy.concatenate((fitnesses, front_fitnesses), axis=0)
            sorted_front = self.sort_front(all_fitnesses=fitnesses, front_fitnesses=front_fitnesses, front=front)
            res.extend(sorted_front)
        return res

    def basic_algorithm_nick(self) -> str:
        return NSGA3_TYPE.nick()

    def nick(self) -> str:
        return self.basic_algorithm_nick()

    def name(self) -> str:
        return NSGA3_TYPE.name()

    def __str__(self) -> str:
        return str(NSGA3_TYPE)
