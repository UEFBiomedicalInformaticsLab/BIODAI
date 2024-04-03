from typing import Sequence

from cross_validation.multi_objective.cross_evaluator.feature_stability_mo_cross_eval import \
    stability_by_weights_from_counts
from saved_solutions.run_measure.run_measure import RunMeasure
from saved_solutions.saved_solution import SavedSolution, union_of_features, average_individual
from util.sequence_utils import flatten_iterable_of_iterable
from validation_registry.allowed_property_names import STABILITY_BY_WEIGHTS_NAME, STABILITY_BY_WEIGHTS_NICK


class RunWeightOverlap(RunMeasure):

    def compute_measure(self, solutions: Sequence[Sequence[SavedSolution]]) -> float:
        all_features = union_of_features(flatten_iterable_of_iterable(solutions))
        average_individuals = [average_individual(f, all_features) for f in solutions]
        return stability_by_weights_from_counts(average_individuals)

    def name(self) -> str:
        return STABILITY_BY_WEIGHTS_NAME

    def nick(self) -> str:
        return STABILITY_BY_WEIGHTS_NICK
