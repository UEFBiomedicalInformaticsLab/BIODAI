from typing import Sequence

from cross_validation.multi_objective.cross_evaluator.feature_stability_mo_cross_eval import \
    stability_by_unions_from_counts
from saved_solutions.run_measure.run_measure import RunMeasure
from saved_solutions.saved_solution import SavedSolution, union_of_features, sum_of_individuals
from util.sequence_utils import flatten_iterable_of_iterable
from validation_registry.allowed_property_names import STABILITY_BY_SET_OVERLAP_NAME, STABILITY_BY_SET_OVERLAP_NICK


class RunSetOverlap(RunMeasure):

    def compute_measure(self, solutions: Sequence[Sequence[SavedSolution]]) -> float:
        all_features = union_of_features(flatten_iterable_of_iterable(solutions))
        sum_individuals = [sum_of_individuals(f, all_features) for f in solutions]
        return stability_by_unions_from_counts(sum_individuals)

    def name(self) -> str:
        return STABILITY_BY_SET_OVERLAP_NAME

    def nick(self) -> str:
        return STABILITY_BY_SET_OVERLAP_NICK
