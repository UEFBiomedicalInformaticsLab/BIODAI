from collections.abc import Sequence
from typing import Optional

from cross_validation.multi_objective.cross_evaluator.two_objectives_cross_plot import objective_computer_pairs_to_plot
from plots.archives.test_battery import TestBattery
from plots.archives.test_battery_cv import TestBatteryCV
from plots.runnable.subplots_runner import SUBTRADEPLOTS_STR, SUBSCATTERPLOTS_STR
from plots.saved_hof import SavedHoF
from plots.subplots_by_strategy import subtradeplots, subscatterplots
from util.str_utils import iterable_to_string


def grouped_hofs_obj_nicks(hofs: Sequence[Sequence[SavedHoF]]) -> Optional[Sequence[str]]:
    if len(hofs) > 0:
        for h in hofs:
            if len(h) > 0:
                if h[0].has_obj_nicks():
                    return h[0].obj_nicks()
    return None


def subplots_for_battery_one_pair(
        test_battery: TestBattery,
        tradeplot_path: Optional[str] = None,
        scatterplot_path: Optional[str] = None,
        col_x: int = 1,
        col_y: int = 0,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
        ):
    """col_x and col_y are the columns in the saved fitnesses csv to use as x and y coordinates."""
    battery_nick = test_battery.nick()
    hofs = test_battery.existing_hofs_grouped_by_dataset_and_inner()  # Works for both CV and external.
    obj_nicks = grouped_hofs_obj_nicks(hofs=hofs)
    type_str = test_battery.type_str()
    if obj_nicks is not None:
        objectives_str = iterable_to_string(obj_nicks, compact=True, separator="_", brackets=False)
        objectives_pair_str = iterable_to_string([obj_nicks[col_x], obj_nicks[col_y]],
                                                 compact=True, separator="_", brackets=False)
        own_path = battery_nick + "/" + objectives_str + "/" + objectives_pair_str + ".png"
        if tradeplot_path is None:
            tradeplot_path = SUBTRADEPLOTS_STR + "_" + type_str + "/" + own_path
        if scatterplot_path is None:
            scatterplot_path = SUBSCATTERPLOTS_STR + "_" + type_str + "/" + own_path
        num_inner = test_battery.n_inner_labs()
        if test_battery.is_external():
            n_datasets = 1
        else:
            if isinstance(test_battery, TestBatteryCV):
                n_datasets = test_battery.n_datasets()
            else:
                raise ValueError()
        n_cols = None
        if num_inner > 1 and n_datasets > 1:
            n_cols = num_inner  # Number of columns in plot.
        print("Processing " + type_str + " tradeplot")
        subtradeplots(
            hofs=hofs,
            save_path=tradeplot_path,
            ncols=n_cols,
            col_x=col_x, col_y=col_y,
            x_label=x_label, y_label=y_label,
            setup=test_battery.plot_setup())
        print("Processing " + type_str + " scatterplot")
        subscatterplots(
            hofs=hofs,
            save_path=scatterplot_path,
            ncols=n_cols,
            col_x=col_x, col_y=col_y,
            x_label=x_label, y_label=y_label,
            setup=test_battery.plot_setup())


def subplots_for_battery_all_pairs(
        test_battery: TestBattery):
    for i, j, interpolate in objective_computer_pairs_to_plot(objectives=test_battery.objective_computers()):
        print("Processing objectives " + str(i) + "-" + str(j))
        subplots_for_battery_one_pair(test_battery=test_battery,
                                      col_x=i,
                                      col_y=j)
