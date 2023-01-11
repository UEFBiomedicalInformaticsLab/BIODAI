from collections.abc import Sequence

from hall_of_fame.hofers import Hofers
from hall_of_fame.pareto_front import ParetoFront
from input_data.view_prefix import all_unprefixed
from objective.objective_computer import Leanness
from plots.archives.automated_hofs_archive import flatten_hofs_for_dataset_external
from plots.archives.shallow_saved_hofs_archive_external import all_external_validations
from plots.plot_labels import ALL_MAIN_NO_NSGA3
from plots.solution_utils import solutions_from_algorithms
from plots.saved_hof import SavedHoF
from plots.summary_statistics_plotter import SUMMARY_STAT_DIR
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solution_from_algorithm import SolutionFromAlgorithm
from util.printer.printer import LogAndOutPrinter

BEST_SOLUTIONS_STR = "best_solutions.txt"
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


def solution_str(s: SolutionFromAlgorithm) -> str:
    if s is None:
        return ""
    fit = s.get_test_fitness()
    res = ""
    res += s.algorithm_name()
    res += ", "
    res += str(round(Leanness().val_to_label(fit.values[1])))
    res += ", "
    res += print_quality_metric(fit.values[0])
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


def best_solutions_for_dataset_str(hofs: Sequence[SavedHoF]) -> str:
    hofers = best_solutions_for_dataset(hofs=hofs)
    non_empty = []
    for h in hofers:
        if h.num_features() > 0:
            non_empty.append(h)
    non_empty.sort(key=lambda e: e.get_test_fitness(), reverse=False)
    res = ""
    for h in non_empty:
        res += solution_str(h) + "\n"
    return res


def save_best_solutions_for_dataset(save_path: str, hofs: Sequence[SavedHoF]):
    to_write = best_solutions_for_dataset_str(hofs=hofs)
    printer = LogAndOutPrinter(log_file=save_path)
    printer.print(to_write)


if __name__ == '__main__':
    for ext in all_external_validations(main_labs=MAIN_LABS):
        external_hofs = ext.nested_hofs()
        internal_label = ext.internal_label()
        external_nick = ext.external_nick()
        print("Processing external validation " + str(ext.internal_label() + " - " + external_nick))
        hofs = flatten_hofs_for_dataset_external(
            dataset_lab=internal_label, external_nick=external_nick, main_labs=MAIN_LABS)
        plot_path = SUMMARY_STAT_DIR + "/external/" + internal_label + "_" + external_nick + "/" + BEST_SOLUTIONS_STR
        save_best_solutions_for_dataset(save_path=plot_path, hofs=hofs)
