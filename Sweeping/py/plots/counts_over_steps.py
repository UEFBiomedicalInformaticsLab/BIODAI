from plots.hofs_plotter.plot_setup import PlotSetup, DEFAULT_PLOT_SETUP
from plots.plot_utils import smart_save_fig, lighten_color
from util.printer.printer import Printer, OutPrinter
from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def weights_over_steps_plot_ax(
    ax: Axes,
    counts: Sequence[Sequence],
    labels: Sequence = (),
    x: Optional[Sequence] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    lighten: float = 0.4,
    setup: PlotSetup = DEFAULT_PLOT_SETUP
):
    font_size = setup.font_size()
    labels = setup.labels_map().apply_all(labels)
    x_label = setup.labels_map().apply(x_label)
    y_label = setup.labels_map().apply(y_label)

    colors = [lighten_color(color=c, amount=lighten) for c in setup.palette().colors(len(counts))]

    """Pass x to give specific x coordinates."""
    if x is None:
        x = range(len(counts[0]) if len(counts) > 0 else 0)

    # Style context to ensure consistency across all text elements
    style = {
        'font.size': font_size,        # base font size
        'axes.labelsize': font_size,   # x/y label sizes
        'axes.titlesize': font_size,   # title size
        'xtick.labelsize': font_size,  # tick label sizes
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,  # legend text size
    }

    with plt.style.context(style):
        if len(counts) > 0:
            ax.stackplot(
                x,
                counts,
                labels=labels,
                colors=colors,
            )
        ax.grid(True)

        # IMPORTANT: enforce label sizes manually
        if x_label:
            ax.set_xlabel(x_label, fontsize=font_size)
        if y_label:
            ax.set_ylabel(y_label, fontsize=font_size)

        ax.tick_params(axis='both', labelsize=font_size)

        if labels:
            ax.legend(loc='best', fontsize=font_size)


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
