from __future__ import annotations

import os
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Optional, Any, Union

import seaborn
from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from consts import FONT_SIZE
from plots.plot_consts import DEFAULT_PALETTE_NAME
from util.printer.printer import Printer, OutPrinter
import matplotlib.colors as mc


DEFAULT_DECIMALS = 2
MAX_EMBEDDED_LEGEND_LINES = 10


def default_color_list(n_colors: int, desat: float = None, invert: bool = False, lighten: float = 0.0):
    colors = seaborn.color_palette(palette=DEFAULT_PALETTE_NAME, n_colors=n_colors, desat=desat)
    if len(colors) < n_colors or n_colors > 10:
        colors = seaborn.color_palette('Spectral', n_colors)  # seaborn.color_palette(n_colors=n_colors, desat=desat)
    if invert:
        colors = colors[::-1]
    colors = [lighten_color(c, amount=lighten) for c in colors]
    return colors


def lighten_color(color, amount: float = 0.4):
    """
    Lightens the given color by mixing it with white.
    amount=0 → original color
    amount=1 → white
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    r, g, b = mc.to_rgb(c)
    return (1 - amount) * r + amount, (1 - amount) * g + amount, (1 - amount) * b + amount


def smart_save_fig(path: str, printer: Printer = OutPrinter(), bbox_inches_tight: bool = True):
    """If no extension is specified, .png is added automatically.
    Potentially we could ensure to stay under 40 megapixels by using
    fig = plt.gcf()
    fig_width, fig_height = fig.get_size_inches()."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    dpi = 600  # 1200 can be too big for some Journals.
    try:
        if bbox_inches_tight:
            plt.savefig(path, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(path, dpi=dpi)
    except BaseException as e:
        printer.print("Exception while saving figure:\n" + str(e))
    plt.close()


def name_color_map(names: Sequence[str]) -> dict[str, Any]:
    """Uses one color for each different name. Colors are applied keeping the order of the names."""
    needed_colors = len(set(names))
    if needed_colors == 0:
        return {}
    colors = default_color_list(n_colors=needed_colors, invert=True)
    next_col_i = 0
    res = {}
    for n in names:
        if n not in res:
            res[n] = colors[next_col_i]
            next_col_i = next_col_i+1
    return res


def line_colors(names: Sequence[str], colors_by_label: bool = True) -> Sequence[Any]:
    """If colors_by_label, uses one color for each different name. Colors are applied keeping the order of the names."""
    if colors_by_label:
        name_color = name_color_map(names=names)
        return [name_color[n] for n in names]
    else:
        n_lines = len(names)
        return default_color_list(n_colors=n_lines, invert=True)


def font_size_by_lines(n_lines: int, initial_size: int = FONT_SIZE) -> int:
    return min(initial_size, int(120.0 / n_lines))  # Was 90

def add_external_legend(ax: Axes, lines: Sequence[Line2D], labs: Sequence[str]):
    """Can cause the plot to get squeezed."""
    ax.legend(lines, labs, loc='center left', bbox_to_anchor=(1, 0.5))


def multi_line_plot_ax(
        ax: Axes, y: Sequence[Sequence[float]], x: Optional[Sequence[Sequence[float]]],
        line_labels: Optional[Sequence[str]] = None,
        x_label: str = "x", y_label: str = "y",
        x_min: Optional[float] = None, y_min: Optional[float] = None,
        x_max: Optional[float] = None, y_max: Optional[float] = None,
        font_size: int = FONT_SIZE, colors_by_label: bool = True,
        decimals: Optional[int] = DEFAULT_DECIMALS,
        x_tick_labels: Optional[Sequence[str]] = None,
        legend_loc: str = "best"):
    """Decimals None for Mathplotlib default.
    x_tick_labels are used only if no x is specified."""
    n_lines = len(y)
    if line_labels is None:
        used_line_labels = [str(i) for i in range(n_lines)]
    else:
        used_line_labels = line_labels
    font_size = font_size_by_lines(n_lines=n_lines, initial_size=font_size)
    with plt.style.context({'font.size': font_size}):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        lines = []
        colors = line_colors(names=used_line_labels, colors_by_label=colors_by_label)
        for i in range(n_lines):
            color = colors[i % len(colors)]
            if x is None:
                line = ax.plot(range(len(y[i])), y[i], "-", color=color, label=str(used_line_labels[i]))
            else:
                line = ax.plot(x[i], y[i], "-", color=color, label=str(used_line_labels[i]))
            lines = lines + line
        if x_min is not None:
            plt.xlim(left=x_min)
        if x_max is not None:
            plt.xlim(right=x_max)
        if y_min is not None:
            plt.ylim(bottom=y_min)
        if y_max is not None:
            plt.ylim(top=y_max)
        ax.grid(visible=True)
        labs = [str(line.get_label()) for line in lines]
        if line_labels is not None or len(used_line_labels) > 1:
            ax.legend(lines, labs, loc=legend_loc)
        format_tick_labels(ax=ax, decimals=decimals)
        if x is None and x_tick_labels is not None:
            xticks_loc = ax.get_xticks()
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks_loc))
            x_lab = []
            for x in xticks_loc:
                if 0 <= x < len(x_tick_labels):
                    x_lab.append(x_tick_labels[round(x)])
                else:
                    x_lab.append(x)
            ax.set_xticklabels(x_lab)


def multi_line_plot(y: Sequence[Sequence[float]], x: Sequence[Sequence[float]],
                    line_labels: Optional[Sequence[str]] = None,
                    x_label: str = "x", y_label: str = "y",
                    x_min: float = None, y_min: float = None,
                    x_max: float = None, y_max: float = None, font_size: int = FONT_SIZE,
                    x_tick_labels: Optional[Sequence[str]] = None,
                    legend_loc: str = "best"):
    """x_tick_labels are used only if no x is specified."""
    fig, ax = plt.subplots()
    multi_line_plot_ax(ax=ax, y=y, x=x, line_labels=line_labels, x_label=x_label, y_label=y_label,
                       x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, font_size=font_size,
                       x_tick_labels=x_tick_labels, legend_loc=legend_loc)


def multi_line_plot_to_path(
        y: Sequence[Sequence[float]], x: Union[Sequence[Sequence[float]], None],
        path: str, line_labels: Sequence[str] = None, x_label: str = "x", y_label: str = "y",
        x_min: float = None, y_min: float = None,
        x_max: float = None, y_max: float = None,
        x_tick_labels: Optional[Sequence[str]] = None,
        legend_loc: str = "best"):
    """x_tick_labels are used only if no x is specified."""
    multi_line_plot(x=x, y=y, line_labels=line_labels, x_label=x_label, y_label=y_label,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, x_tick_labels=x_tick_labels,
                    legend_loc=legend_loc)
    smart_save_fig(path=path)


def single_line_plot(
        y: Sequence[float], x: Optional[Sequence[float]] = None, x_label: str = "x", y_label: str = "y"):
    if x is None:
        x = range(1, len(y)+1)
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    lines = []
    lines = lines + ax.plot(
        x, y, "b-", label=y_label)
    plt.grid()
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc="center right")


def single_line_plot_to_path(
        y: Sequence[float], plot_path: str, x: Optional[Sequence[float]] = None, x_label: str = "x", y_label: str = "y"):
    single_line_plot(y=y, x=x, x_label=x_label, y_label=y_label)
    smart_save_fig(path=plot_path)


def format_tick(x, decimals: int) -> str:
    if isinstance(x, int) or x.is_integer():
        return str(int(x))
    else:
        format_str = "{:." + str(decimals) + "f}"
        return format_str.format(x)


def format_tick_labels(ax: Axes, decimals: Optional[int]):
    if decimals is not None:
        xticks_loc = ax.get_xticks()
        yticks_loc = ax.get_yticks()
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks_loc))
        ax.yaxis.set_major_locator(ticker.FixedLocator(yticks_loc))
        ax.set_xticklabels([format_tick(x=x, decimals=decimals) for x in xticks_loc])
        ax.set_yticklabels([format_tick(x=x, decimals=decimals) for x in yticks_loc])


def add_row_names(fig: Figure, axs,
                  ncols: int,
                  row_names: Optional[Sequence[str]],
                  subplots_adjust_left: Optional[float] = None):
    if row_names is not None:
        pad = 0  # in points
        to_annotate = []
        if isinstance(axs[0], Axes):
            axs = [axs]
        for ax in list(chain.from_iterable(axs)):
            sbs = ax.get_subplotspec()
            if sbs.is_first_col():
                to_annotate.append(ax)
        for ax, row in zip(to_annotate, row_names):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90)
        if subplots_adjust_left is None:
            if ncols == 1:
                subplots_adjust_left = 0.22
            else:
                subplots_adjust_left = 0.19 - ncols * 0.03
        fig.subplots_adjust(left=subplots_adjust_left)
