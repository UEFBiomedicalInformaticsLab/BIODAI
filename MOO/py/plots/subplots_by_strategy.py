from collections.abc import Sequence
from string import ascii_lowercase
from typing import Optional

from matplotlib import pyplot as plt

from plots.hofs_plotter.hofs_best_trade_plotter import HofsBestTradePlotter
from plots.hofs_plotter.hofs_plotter import HofsPlotter
from plots.hofs_plotter.hofs_scatterplotter import HofsScatterplotter
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.plot_utils import smart_save_fig
from plots.saved_hof import SavedHoF
from util.grid import row_col_by_index
from util.utils import ceil_division


def subplots_by_strategy(
        hofs: Sequence[Sequence[SavedHoF]],
        plotter: HofsPlotter,
        save_path: str, ncols: int = 2, x_label: str = "x", y_label: str = "y"):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot."""
    n_subplots = len(hofs)
    nrows = ceil_division(num=n_subplots, den=ncols)
    figsize_x = 4.0*ncols + 1.0
    figsize_y = 4.0*nrows + 1.0
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=600, figsize=(figsize_x, figsize_y), sharex=True, sharey=True)
    n_boxes = nrows * ncols
    for i in range(n_boxes):
        row, col = row_col_by_index(index=i, ncol=ncols)
        plot_axs = axs[row, col]
        if i < n_subplots:
            plot_hofs = hofs[i]
            letter = ascii_lowercase[i]
            plot_axs.set_title(letter)
            plotter.plot(ax=plot_axs, saved_hofs=plot_hofs)
            plot_axs.set(xlabel=None)
            plot_axs.set(ylabel=None)
    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.tight_layout()
    smart_save_fig(path=save_path)


def subscatterplots(
        hofs: Sequence[Sequence[SavedHoF]],
        save_path: str, ncols: int = 2, col_x: int = 1, col_y: int = 0, x_label: str = "x", y_label: str = "y",
        setup: Optional[PlotSetup] = None):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot.
    col_x is the column from which to extract the x values,
    col_y is the column from which to extract the y values."""
    if setup is None:
        setup = PlotSetup()
    plotter = HofsScatterplotter(col_x=col_x, col_y=col_y, setup=setup)
    subplots_by_strategy(
        hofs=hofs,
        plotter=plotter,
        save_path=save_path,
        ncols=ncols,
        x_label=x_label, y_label=y_label)


def subtradeplots(
        hofs: Sequence[Sequence[SavedHoF]],
        save_path: str, ncols: int = 2, col_x: int = 1, col_y: int = 0, x_label: str = "x", y_label: str = "y",
        setup: Optional[PlotSetup] = None):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot.
    col_x is the column from which to extract the x values,
    col_y is the column from which to extract the y values."""
    if setup is None:
        setup = PlotSetup()
    plotter = HofsBestTradePlotter(col_x=col_x, col_y=col_y, setup=setup)
    subplots_by_strategy(
        hofs=hofs,
        plotter=plotter,
        save_path=save_path,
        ncols=ncols,
        x_label=x_label, y_label=y_label)
