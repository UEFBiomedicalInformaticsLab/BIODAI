from typing import Sequence

from plots.hof_stability_plots import smaller_feature_set_size, mean_dice
from saved_solutions.run_measure import RunMeasure
from saved_solutions.saved_solution import SavedSolution, union_of_features, sum_of_individuals
from util.components_transform import HigherKComponents
from util.sequence_utils import flatten_iterable_of_iterable


class RunBestDice(RunMeasure):

    def compute_measure(self, solutions: Sequence[Sequence[SavedSolution]]) -> float:
        if len(solutions) > 1:
            all_features = union_of_features(flatten_iterable_of_iterable(solutions))
            fold_counts = [sum_of_individuals(f, all_features) for f in solutions]
            max_k = smaller_feature_set_size(fold_counts=fold_counts)
            dice_values = []
            for k in range(1, max_k + 1):
                components_transform = HigherKComponents(k)
                fold_counts_k = [components_transform.apply(c) for c in fold_counts]
                dice_values.append(mean_dice(fold_counts_k))
            return max(dice_values)
        else:
            return 1.0

    def name(self) -> str:
        return "best Dice"

    def nick(self) -> str:
        return "best_dice"
