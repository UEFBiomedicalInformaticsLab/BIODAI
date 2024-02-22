from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from consts import FONT_SIZE
from plots.plot_utils import smart_save_fig, default_color_list
from util.printer.printer import Printer, OutPrinter


def weights_over_steps_plot_ax(ax: Axes, counts: Sequence[Sequence], labels: Sequence = (), x: Sequence = None,
                               x_label: Optional[str] = None, y_label: Optional[str] = None):
    """Pass x to give specific x coordinates."""
    font_size = FONT_SIZE
    if x is None:
        x = range(len(counts[0]))
    with plt.style.context({'font.size': font_size}):
        ax.stackplot(x, counts, labels=labels, colors=default_color_list(n_colors=len(counts)))
        ax.grid()
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        if len(labels) > 0:
            ax.legend(loc='best')


def weights_over_steps_plot(counts: Sequence[Sequence], labels: Sequence = (), x: Sequence = None,
                            x_label: Optional[str] = None, y_label: Optional[str] = None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    weights_over_steps_plot_ax(ax=ax, counts=counts, labels=labels, x=x, x_label=x_label, y_label=y_label)


def counts_over_steps_plot_to_file(file: str, counts: Sequence[Sequence], labels: Sequence = (), x: Sequence = None,
                                   x_label: Optional[str] = None, y_label: Optional[str] = None,
                                   printer: Printer = OutPrinter()):
    """ Counts are a series of series. The external is for each quantity counted. The internal is for the
    readings of that quantity at each step. Each step must have the same number of counts."""
    weights_over_steps_plot(counts=counts, labels=labels, x=x, x_label=x_label, y_label=y_label)
    smart_save_fig(path=file, printer=printer)
