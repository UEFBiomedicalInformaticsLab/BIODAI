from collections.abc import Sequence
from typing import Optional, Union

from consts import FONT_SIZE
from plots.default_labels_map import LabelsTransformer, DEFAULT_LABELS_TRANSFORMER
from plots.hofs_plotter.hofs_best_trade_plotter import HofsBestTradePlotter
from plots.hofs_plotter.hofs_plotter import HofsPlotter
from plots.hofs_plotter.hofs_scatterplotter import HofsScatterplotter
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.plotter.plotter_from_hofs_plotter import PlotterFromHofsPlotter
from plots.saved_hof import SavedHoF
from plots.subplots import subplots


def subplots_by_strategy(
        hofs: Sequence[Sequence[SavedHoF]],
        plotter: Union[HofsPlotter, Sequence[HofsPlotter]],
        save_path: str, ncols: Optional[int] = None,
        x_label: Optional[str] = None, y_label: Optional[str] = None,
        color_by_row: bool = False,
        font_size: int = FONT_SIZE,
        row_names: Optional[Sequence[str]] = None,
        labels_transformer: LabelsTransformer = DEFAULT_LABELS_TRANSFORMER):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot.
    If more than one plotter is passed, they are used cyclically."""
    if isinstance(plotter, HofsPlotter):
        plotter = [plotter]
    n_plotters = len(plotter)
    plotters_with_hofs = [
        PlotterFromHofsPlotter(hofs_plotter=plotter[i % n_plotters], saved_hofs=plot_hofs)
        for i, plot_hofs in enumerate(hofs)]
    subplots(
        plotters=plotters_with_hofs,
        save_path=save_path,
        ncols=ncols,
        x_label=x_label, y_label=y_label,
        color_by_row=color_by_row,
        font_size=font_size,
        row_names=row_names,
        desat=0.75,
        labels_transformer=labels_transformer)


def subscatterplots(
        hofs: Sequence[Sequence[SavedHoF]],
        save_path: str, ncols: int = 2, col_x: int = 1, col_y: int = 0,
        x_label: Optional[str] = None, y_label: Optional[str] = None,
        setup: Optional[PlotSetup] = None,
        row_names: Optional[Sequence[str]] = None):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot.
    ncols is the number of columns in the plot.
    col_x is the column from which to extract the x values,
    col_y is the column from which to extract the y values."""
    if setup is None:
        setup = PlotSetup()
    plotter = HofsScatterplotter(col_x=col_x, col_y=col_y, setup=setup)
    if x_label is None or y_label is None:
        obj_nicks = hofs[0][0].obj_nicks()
        if x_label is None:
            x_label = obj_nicks[col_x]
        if y_label is None:
            y_label = obj_nicks[col_y]
    subplots_by_strategy(
        hofs=hofs,
        plotter=plotter,
        save_path=save_path,
        ncols=ncols,
        x_label=setup.label_transform(x_label), y_label=setup.label_transform(y_label),
        font_size=setup.font_size(),
        row_names=row_names)


def subtradeplots(
        hofs: Sequence[Sequence[SavedHoF]],
        save_path: str, ncols: Optional[int] = None, col_x: int = 1, col_y: int = 0,
        x_label: Optional[str] = None, y_label: Optional[str] = None,
        setup: Optional[PlotSetup] = None,
        row_names: Optional[Sequence[str]] = None):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot.
    col_x is the column (objective index) from which to extract the x values,
    col_y is the column (objective index) from which to extract the y values."""
    if setup is None:
        setup = PlotSetup()
    plotter = HofsBestTradePlotter(col_x=col_x, col_y=col_y, setup=setup)
    if x_label is None or y_label is None:
        obj_nicks = hofs[0][0].obj_nicks()
        if x_label is None:
            x_label = obj_nicks[col_x]
        if y_label is None:
            y_label = obj_nicks[col_y]
    subplots_by_strategy(
        hofs=hofs,
        plotter=plotter,
        save_path=save_path,
        ncols=ncols,
        x_label=setup.label_transform(x_label), y_label=setup.label_transform(y_label),
        font_size=setup.font_size(), row_names=row_names, labels_transformer=setup.labels_map())
