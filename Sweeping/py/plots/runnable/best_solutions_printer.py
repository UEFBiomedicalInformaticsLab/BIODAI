from collections.abc import Sequence

from hall_of_fame.hofers import Hofers
from hall_of_fame.pareto_front import ParetoFront
from input_data.view_prefix import all_unprefixed
from objective.objective_computer import ObjectiveComputer
from objective.objective_with_importance.objective_computer_with_importance import BalancedAccuracy
from objective.objective_with_importance.leanness import RootLeanness
from plots.default_labels_map import DUMMY_LABELS_TRANSFORMER, LabelsTransformer
from plots.plot_labels import ALL_MAIN_NO_NSGA3
from plots.solution_utils import solutions_from_algorithms
from plots.saved_hof import SavedHoF
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solution_from_algorithm import SolutionFromAlgorithm
from util.printer.printer import LogAndOutPrinter

BEST_SOLUTIONS_STR = "best_solutions.txt"
BEST_GENES_STR = "best_genes.txt"
MAIN_LABS = ALL_MAIN_NO_NSGA3


def print_quality_metric(value: float) -> str:
    return "{:.3f}".format(value)


def print_single_classes(solution: SavedSolution) -> str:
    res = ""
    if solution.has_confusion_matrix():
        cm = solution.confusion_matrix()
        bal_accuracies = cm.balanced_accuracies()
        if len(bal_accuracies) > 0:
            class_names = cm.labels()
            dic = {}
            for c, a in zip(class_names, bal_accuracies):
                dic[c] = a
            dic = dict(sorted(dic.items()))
            res += " ("
            first = True
            for d in dic:
                if not first:
                    res += ", "
                res += d + ": " + print_quality_metric(dic[d])
                first = False
            res += ")"
    return res


def solution_str(s: SolutionFromAlgorithm,
                 objectives: Sequence[ObjectiveComputer] = (BalancedAccuracy(), RootLeanness()),
                 labels_transformer: LabelsTransformer = DUMMY_LABELS_TRANSFORMER) -> str:
    if s is None:
        return ""
    fit = s.get_test_fitness()
    res = ""
    res += labels_transformer.apply(s.algorithm_name())
    res += ", "
    res += str(s.num_features())
    res += ", "
    res += print_quality_metric(objectives[0].val_to_label(fit.values[0]))
    res += print_single_classes(s.solution())
    res += " ["
    features = s.solution().features()
    res += ', '.join(all_unprefixed(features))
    res += "]"
    return res


def best_solutions_for_dataset(hofs: Sequence[SavedHoF]) -> Hofers:
    solutions = solutions_from_algorithms(hofs=hofs)
    pareto = ParetoFront()
    pareto.update(new_elems=solutions)
    return pareto.hofers()


def best_solutions_for_dataset_str(hofs: Sequence[SavedHoF],
                                   labels_transformer: LabelsTransformer = DUMMY_LABELS_TRANSFORMER) -> str:
    hofers = best_solutions_for_dataset(hofs=hofs)
    non_empty = []
    for h in hofers:
        if h.num_features() > 0:
            non_empty.append(h)
    non_empty.sort(key=lambda e: e.get_test_fitness(), reverse=False)
    res = ""
    for h in non_empty:
        res += solution_str(h, labels_transformer=labels_transformer) + "\n"
    return res


def save_best_solutions_for_dataset(save_path: str, hofs: Sequence[SavedHoF],
                                    labels_transformer: LabelsTransformer = DUMMY_LABELS_TRANSFORMER):
    to_write = best_solutions_for_dataset_str(hofs=hofs, labels_transformer=labels_transformer)
    printer = LogAndOutPrinter(log_file=save_path)
    printer.print(to_write)
