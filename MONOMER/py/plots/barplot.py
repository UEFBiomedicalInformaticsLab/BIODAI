from collections.abc import Sequence
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import subplots

from consts import FONT_SIZE
from plots.plot_utils import smart_save_fig, default_color_list
from util.summer import KahanSummer


DEFAULT_BAR_COLOR = (0.75, 0.75, 0.75, 1.0)


def barplot_ax(
        ax: Axes, bar_lengths: Sequence[float], bar_names: Sequence[str], label_y: str = None,
        bar_color=DEFAULT_BAR_COLOR, font_size: int = FONT_SIZE, classes: Optional[Sequence] = None):
    barplot_with_std_ax(
        ax=ax, measures=[[b] for b in bar_lengths], bar_names=bar_names, label_y=label_y, bar_color=bar_color,
        font_size=font_size,
        classes=classes
    )


def barplot_with_std_ax(
        ax: Axes, measures: Sequence[Sequence[float]], bar_names: Sequence[str], label_y: str = None,
        bar_color=DEFAULT_BAR_COLOR, font_size: int = FONT_SIZE, classes: Optional[Sequence] = None):
    """bar_lengths is a sequence with an inner sequence for each bar. The inner sequences are used to compute mean
    and variance."""
    if bar_color is None:
        bar_color = DEFAULT_BAR_COLOR
    if classes is None:
        classes = [0]*len(measures)
    n_classes = len(classes)
    if n_classes > 1:
        unique_classes = sorted(set(classes), key=classes.index)  # Removes duplicates keeping order.
        class_colors = default_color_list(n_colors=len(unique_classes), invert=True, desat=0.75)
        color_map = {}
        for i, uc in enumerate(unique_classes):
            color_map[uc] = class_colors[i]
        colors = [color_map[c] for c in classes]
    else:
        colors = [bar_color]*len(measures)
    with plt.style.context({'font.size': font_size}):
        x_pos = np.arange(len(bar_names))
        ax.grid(zorder=0, axis='y')
        bar_lengths = [KahanSummer.mean(bar) for bar in measures]
        bar_std = [np.std(bar) for bar in measures]
        ax.bar(x_pos, bar_lengths, yerr=bar_std, zorder=2, color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_names, rotation='vertical')
        if label_y is not None:
            ax.set_ylabel(label_y)


def barplot_to_file(path: str, bar_lengths: Sequence[float], bar_names: Sequence[str], label_y: str = None,
                    classes: Optional[Sequence] = None):
    fig, ax = subplots()
    barplot_ax(ax=ax, bar_lengths=bar_lengths, bar_names=bar_names, label_y=label_y, classes=classes)
    smart_save_fig(path=path)


def barplot_with_std_to_file(
        path: str, measures: Sequence[Sequence[float]], bar_names: Sequence[str],
        label_y: str = None, classes: Optional[Sequence] = None):
    fig, ax = subplots()
    barplot_with_std_ax(ax=ax, measures=measures, bar_names=bar_names, label_y=label_y, classes=classes)
    smart_save_fig(path=path)
