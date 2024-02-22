import os.path
from collections.abc import Sequence

from cross_validation.multi_objective.cross_evaluator.hof_saver import INNER_CV_PREFIX
from plots.counts_over_steps import counts_over_steps_plot_to_file
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solutions_from_files import final_solutions_from_files, objective_names
from util.printer.printer import OutPrinter
from util.system_utils import subdirectories
from util.utils import mean_of_dicts, sorted_dict


def view_contributions_one_objective(hof_dir: str, saved_solutions: Sequence[SavedSolution],
                                     objective_pos: int, objective_name: str):
    """Uses training performances (Performances seen by the optimizer, might be inner CV)."""
    all_counts = {}
    for s in saved_solutions:
        train_fitnesses = s.train_fitnesses()
        if objective_pos < len(train_fitnesses):
            s_perf = train_fitnesses[objective_pos]
        else:
            raise ValueError(
                "train_fitnesses: " + str(train_fitnesses) + "\n" +
                "objective name: " + objective_name + "\n" +
                "objective pos: " + str(objective_pos) + "\n")
        s_counts = s.num_features_by_view()
        if s_perf in all_counts:
            all_counts[s_perf].append(s_counts)
        else:
            all_counts[s_perf] = [s_counts]
    averages = {}
    for k in all_counts:
        averages[k] = mean_of_dicts(all_counts[k])
    sorted_averages = sorted_dict(averages)
    labels = set()
    for s in sorted_averages:
        for k in sorted_averages[s]:
            labels.add(k)
    labels = [l for l in labels]
    labels.sort()
    counts = []
    for l in labels:
        l_counts = []
        for d in sorted_averages:
            sa_d = sorted_averages[d]
            if l in sa_d:
                l_counts.append(sa_d[l])
            else:
                l_counts.append(0)
        counts.append(l_counts)
    x = list(sorted_averages.keys())
    labels = [str(l) for l in labels]
    path = os.path.join(hof_dir, "view_counts_" + objective_name)
    counts_over_steps_plot_to_file(file=path, counts=counts, labels=labels, x=x,
                                   x_label=objective_name, y_label="counts", printer=OutPrinter())


def view_contributions_one_hof(hof_dir: str):
    saved_solutions = final_solutions_from_files(hof_dir=hof_dir)
    solutions_num = len(saved_solutions)
    if solutions_num > 0:
        obj_names = objective_names(hof_dir=hof_dir)
        n_objectives = len(obj_names)
        for i in range(n_objectives):
            if INNER_CV_PREFIX in obj_names[i]:
                view_contributions_one_objective(
                    hof_dir=hof_dir,
                    saved_solutions=saved_solutions, objective_pos=i, objective_name=obj_names[i])


def view_contributions_every_hof(main_hofs_dir: str):
    for f in subdirectories(main_directory=main_hofs_dir):
        view_contributions_one_hof(f)
