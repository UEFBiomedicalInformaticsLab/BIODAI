from collections.abc import Sequence

from matplotlib.axes import Axes

from cross_validation.multi_objective.cross_evaluator.two_objectives_cross_plot import objective_computer_pairs_to_plot
from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.archives.test_battery_external import TestBatteryExternal
from plots.best_hof_for_dataset_cv import best_hof_for_property, DEFAULT_COMPARISON_PROPERTY
from plots.hofs_plotter.hofs_detailed_plotter import HofsDetailedPlotter
from plots.hofs_plotter.palette import BASELINE_PALETTE
from plots.hofs_plotter.view_contributions_plotter import ViewContributionsPlotter
from plots.objective_pairs_plots import one_objective_pair_plot_from_saved_hofs
from plots.plotter.plotter import Plotter
from plots.plotter.plotter_from_hofs_plotter import PlotterFromHofsPlotter
from plots.runnable.summary_statistics_plotter import SUMMARY_STAT_DIR
from plots.saved_hof import SavedHoF
from plots.subplots import subplots
from validation_registry.registry_property import RegistryProperty


BASELINE_BEST_COMPARISON_STR = "baseline_best_comparison.png"


def baseline_best_hofs(
        battery: TestBattery,
        registry_property: RegistryProperty = DEFAULT_COMPARISON_PROPERTY) -> tuple[SavedHoF, SavedHoF]:
    base = battery.baseline()
    if base is None:
        raise ValueError("No baseline defined")
    hofs = battery.existing_flat_hofs()
    if len(hofs) == 0:
        raise ValueError("No hofs found.")
    if len(hofs) > 1:
        raise NotImplementedError("Multiple datasets are not supported yet.")
    baseline_hofs = battery.baseline_hofs()
    if len(baseline_hofs) == 0 or len(baseline_hofs[0]) == 0:
        raise ValueError("No baseline hofs found.")
    best_baseline_hof = best_hof_for_property(hofs=baseline_hofs[0], registry_property=registry_property)
    best_hof = best_hof_for_property(hofs=hofs[0], registry_property=registry_property)
    return best_baseline_hof, best_hof


def baseline_best_comparison_ax(ax_base: Axes, ax_best: Axes, battery: TestBattery,
                                obj_x: int = 1, obj_y: int = 0,
                                registry_property: RegistryProperty = DEFAULT_COMPARISON_PROPERTY):
    """obj_x and obj_y are the columns in the saved fitnesses csv to use as x and y coordinates."""
    best_baseline_hof, best_hof = baseline_best_hofs(battery=battery, registry_property=registry_property)

    one_objective_pair_plot_from_saved_hofs(ax=ax_base, saved_hofs=[best_baseline_hof], i=obj_x, j=obj_y)
    one_objective_pair_plot_from_saved_hofs(ax=ax_best, saved_hofs=[best_hof], i=obj_x, j=obj_y)


def baseline_best_comparison_all_pairs_plotters(
        test_battery: TestBattery) -> Sequence[Plotter]:
    if test_battery.baseline() is None:
        raise ValueError("No baseline defined.")
    combinations = list(objective_computer_pairs_to_plot(objectives=test_battery.objective_computers()))
    n_combinations = len(combinations)
    if n_combinations == 0:
        return []
    else:
        best_baseline_hof, best_hof = baseline_best_hofs(battery=test_battery)
        best_plot_setup = test_battery.plot_setup()
        baseline_plot_setup = best_plot_setup.set_palette(palette=BASELINE_PALETTE)
        hofs_to_plot = []
        plotters = []
        for i, c in enumerate(combinations):
            obj_x = c[0]
            obj_y = c[1]
            print("Processing objectives " + str(obj_x) + "-" + str(obj_y))
            plotter_base = HofsDetailedPlotter(col_x=obj_x, col_y=obj_y, setup=baseline_plot_setup)
            plotter_best = HofsDetailedPlotter(col_x=obj_x, col_y=obj_y, setup=best_plot_setup)
            plotters.extend([plotter_base, plotter_best])
            hofs_to_plot.extend([[best_baseline_hof], [best_hof]])
            if len(best_baseline_hof.views_from_saved_setup()) > 1 or len(best_hof.views_from_saved_setup()) > 1:
                views_plotter_base = ViewContributionsPlotter(objective_pos=obj_y, setup=baseline_plot_setup)
                views_plotter_best = ViewContributionsPlotter(objective_pos=obj_y, setup=best_plot_setup)
                plotters.extend([views_plotter_base, views_plotter_best])
                hofs_to_plot.extend([[best_baseline_hof], [best_hof]])
        return [PlotterFromHofsPlotter(hofs_plotter=p, saved_hofs=h) for p, h in zip(plotters, hofs_to_plot)]


def baseline_best_comparison_all_pairs(
        test_battery: TestBattery):
    if test_battery.baseline() is None:
        print("No baseline defined.")
        return
    print("Plotting baseline-best comparisons")
    combinations = list(objective_computer_pairs_to_plot(objectives=test_battery.objective_computers()))
    n_combinations = len(combinations)
    if n_combinations == 0:
        print("No objective combinations to plot.")
    else:
        plotters = baseline_best_comparison_all_pairs_plotters(test_battery=test_battery)
        type_str = test_battery.type_str()
        dataset_part = ""
        if isinstance(test_battery, TestBatteryCV):
            dataset_part = test_battery.dataset_report_path_part(dataset_lab=test_battery.dataset_labels()[0])
        if isinstance(test_battery, TestBatteryExternal):
            dataset_part = test_battery.datasets_report_path_part(datasets=test_battery.datasets()[0])
        save_path = SUMMARY_STAT_DIR + "/" + type_str + "/" +\
                    dataset_part + "/" + BASELINE_BEST_COMPARISON_STR
        setup = test_battery.plot_setup()
        subplots(
            plotters=plotters, save_path=save_path, ncols=2, sharex="row", sharey="row",
            font_size=setup.font_size(), labels_transformer=setup.labels_map(),
            add_letters=False)


def baseline_best_comparison_all_pairs_multiple_batteries(
        test_batteries: Sequence[TestBattery], save_path: str):
    print("Plotting baseline-best comparisons")
    plotters = []
    row_names = []
    for battery in test_batteries:
        battery_plotters = baseline_best_comparison_all_pairs_plotters(test_battery=battery)
        plotters.extend(battery_plotters)
        row_names.extend([battery.dataset_names()[0] for _ in range(len(battery_plotters)//2)])
        # We use only the first name because only one dataset is supported at the moment.
    subplots(plotters=plotters, save_path=save_path, ncols=2, row_names=row_names)