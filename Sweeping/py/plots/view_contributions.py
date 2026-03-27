import os.path
from collections.abc import Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from cross_validation.multi_objective.cross_evaluator.hof_saver import INNER_CV_PREFIX
from plots.counts_over_steps import weights_over_steps_plot_ax
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.monotonic_front import sequence_vals_to_labels
from plots.plot_utils import smart_save_fig
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solutions_from_files import final_solutions_from_files, objective_names
from util.printer.printer import OutPrinter
from util.system_utils import subdirectories
from util.dict_utils import sorted_dict, mean_of_dicts


def view_contributions_one_objective_to_ax(
        ax: Axes, saved_solutions: Sequence[SavedSolution],
        objective_pos: int, objective_name: str, view_names: Sequence[str] = None,
        setup: PlotSetup = PlotSetup()):
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
    int_labels = set()
    for s in sorted_averages:
        for k in sorted_averages[s]:
            int_labels.add(k)
    int_labels = [i for i in int_labels]
    int_labels.sort()
    counts = []
    for i in int_labels:
        l_counts = []
        for d in sorted_averages:
            sa_d = sorted_averages[d]
            if i in sa_d:
                l_counts.append(sa_d[i])
            else:
                l_counts.append(0)
        counts.append(l_counts)
    x = list(sorted_averages.keys())
    x = sequence_vals_to_labels(s=x, label=objective_name)
    str_labels = [str(i) for i in int_labels]
    if view_names is not None:
        if len(view_names) > 0:
            if len(int_labels) > 0:
                if max(int_labels) < len(view_names):
                    str_labels = [setup.label_transform(view_names[i]) for i in int_labels]
                else:
                    print("Number of views in solutions is greater than the number of views in the setup file. Using just "
                          "numbers.")
                    print("View names: " + str(view_names))
                    print("View numbers: " + str(str_labels))
            else:
                print("Number of views in solutions is 0.")
                str_labels = []
        else:
            print("Number of views in the setup file is 0. Using just numbers.")
    else:
        print("No view names specified. Using just numbers.")
    weights_over_steps_plot_ax(
        ax=ax, counts=counts, labels=str_labels, x=x,
        x_label=setup.label_transform(objective_name), y_label=setup.label_transform("counts"))


def view_contributions_one_objective(hof_dir: str, saved_solutions: Sequence[SavedSolution],
                                     objective_pos: int, objective_name: str, view_names: Sequence[str] = None):
    """Uses training performances (Performances seen by the optimizer, might be inner CV)."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    view_contributions_one_objective_to_ax(
        ax=ax, saved_solutions=saved_solutions,
        objective_pos=objective_pos, objective_name=objective_name, view_names = view_names)
    path = os.path.join(hof_dir, "view_counts_" + objective_name)
    smart_save_fig(path=path, printer=OutPrinter())


def view_contributions_one_hof(hof_dir: str, view_names: Sequence[str] = None):
    saved_solutions = final_solutions_from_files(hof_dir=hof_dir)
    solutions_num = len(saved_solutions)
    if solutions_num > 0:
        obj_names = objective_names(hof_dir=hof_dir)
        n_objectives = len(obj_names)
        for i in range(n_objectives):
            if INNER_CV_PREFIX in obj_names[i]:
                view_contributions_one_objective(
                    hof_dir=hof_dir,
                    saved_solutions=saved_solutions, objective_pos=i, objective_name=obj_names[i],
                    view_names=view_names)


def view_contributions_every_hof(main_hofs_dir: str, view_names: Sequence[str] = None):
    for f in subdirectories(main_directory=main_hofs_dir):
        view_contributions_one_hof(f, view_names=view_names)
