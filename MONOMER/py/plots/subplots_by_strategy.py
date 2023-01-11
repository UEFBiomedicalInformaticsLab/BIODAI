from collections.abc import Sequence
from string import ascii_uppercase
from typing import Optional, Union
from matplotlib import pyplot as plt

from consts import FONT_SIZE
from plots.hofs_plotter.hofs_best_trade_plotter import HofsBestTradePlotter
from plots.hofs_plotter.hofs_plotter import HofsPlotter
from plots.hofs_plotter.hofs_scatterplotter import HofsScatterplotter
from plots.hofs_plotter.plot_setup import PlotSetup
from plots.plot_utils import smart_save_fig, default_color_list
from plots.saved_hof import SavedHoF
from util.grid import row_col_by_index
from util.utils import ceil_division


def subplots_by_strategy(
        hofs: Sequence[Sequence[SavedHoF]],
        plotter: Union[HofsPlotter, Sequence[HofsPlotter]],
        save_path: str, ncols: int = 2, x_label: str = None, y_label: str = None, color_by_row: bool = False,
        font_size: int = FONT_SIZE):
    """hofs is a sequence of sequences of SavedHoFs. Each element of the outer sequence feeds a subplot.
    If more than one plotter is passed, they are used cyclically."""
    with plt.style.context({'font.size': font_size}):
        if isinstance(plotter,  HofsPlotter):
            plotter = [plotter]
        n_subplots = len(hofs)
        nrows = ceil_division(num=n_subplots, den=ncols)
        figsize_x = 4.0*ncols + 1.0
        figsize_y = 4.0*nrows + 1.0
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, dpi=600, figsize=(figsize_x, figsize_y), sharex=True, sharey=True)
        n_boxes = nrows * ncols
        if color_by_row:
            n_colors = nrows
        else:
            n_colors = n_boxes
        color_list = default_color_list(n_colors=n_colors, desat=0.75)
        for i in range(n_boxes):
            row, col = row_col_by_index(index=i, ncol=ncols)
            plot_axs = axs[row, col]
            if i < n_subplots:
                plot_hofs = hofs[i]
                letter = ascii_uppercase[i]
                plot_axs.set_title(letter)
                row = i // ncols
                if color_by_row:
                    color = color_list[row]
                else:
                    color = color_list[i]
                plotter[i % len(plotter)].plot(ax=plot_axs, saved_hofs=plot_hofs, color=color)
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
        x_label=x_label, y_label=y_label,
        font_size=setup.font_size())


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
        x_label=x_label, y_label=y_label,
        font_size=setup.font_size())
