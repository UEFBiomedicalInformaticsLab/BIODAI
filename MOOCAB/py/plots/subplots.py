from collections.abc import Sequence
from string import ascii_uppercase
from matplotlib import pyplot as plt

from consts import FONT_SIZE
from plots.plot_utils import smart_save_fig, default_color_list
from plots.plotter.plotter import Plotter
from util.grid import row_col_by_index
from util.utils import ceil_division


def subplots(
        plotters: Sequence[Plotter],
        save_path: str, ncols: int = 2, x_label: str = None, y_label: str = None, color_by_row: bool = False,
        font_size: int = FONT_SIZE, x_stretch: float = 1.0, sharex: bool = True):
    with plt.style.context({'font.size': font_size}):
        n_subplots = len(plotters)
        nrows = ceil_division(num=n_subplots, den=ncols)
        figsize_x = 4.0*ncols*x_stretch + 1.0
        figsize_y = 4.0*nrows + 1.0
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, dpi=600, figsize=(figsize_x, figsize_y), sharex=sharex, sharey=True)
        n_boxes = nrows * ncols
        if color_by_row:
            n_colors = nrows
        else:
            n_colors = n_boxes
        color_list = default_color_list(n_colors=n_colors)
        for i in range(n_boxes):
            row, col = row_col_by_index(index=i, ncol=ncols)
            if n_boxes == 1:
                plot_axs = axs  # Necessary because in this case axs is directly the AxesSubplot. :(
            elif ncols == 1:
                plot_axs = axs[row]  # Necessary because in this case axs is one-dimensional.
            else:
                plot_axs = axs[row, col]
            if i < n_subplots:
                letter = ascii_uppercase[i]
                plot_axs.set_title(letter)
                row = i // ncols
                if color_by_row:
                    color = color_list[row]
                else:
                    color = color_list[i]
                plotters[i].plot(ax=plot_axs, color=color)
                plot_axs.set(xlabel=None)
                plot_axs.set(ylabel=None)
        fig.supxlabel(x_label)
        fig.supylabel(y_label)
        fig.tight_layout()
        smart_save_fig(path=save_path)
