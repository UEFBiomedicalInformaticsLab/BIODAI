from __future__ import annotations

from collections.abc import Sequence
from math import isnan
from typing import Optional, Union

import scipy
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.interpolate import PchipInterpolator
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors

from consts import FONT_SIZE
from plots.plot_utils import smart_save_fig, default_color_list
from util.hyperbox.hyperbox import Interval
from util.math.mean_builder import KahanMeanBuilder
from util.sequence_utils import sort_both_by_first


def plot_accuracy_and_confusion(y_pred, y_true):
    confusion = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    len_y = len(y_true)
    accuracy = sum(np.diag(confusion) / len_y)
    plt.figure(figsize=(9, 9))
    sns.heatmap(confusion, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)


def plot_frequencies_histogram(frequencies, labels=None):
    if labels is None:
        labels = range(len(frequencies))

    pos = np.arange(len(labels))
    width = 1.0  # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(labels)

    plt.bar(pos, frequencies, width, color='r')


def plot_scatter(x, y, save_file: str, x_label=None, y_label=None):
    plt.scatter(x=x, y=y)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    smart_save_fig(path=save_file)


def alpha_by_n_points(n_points: int) -> float:
    if n_points == 0:
        return 1.0
    else:
        return min(2000.0 / n_points, 1.0)


def mid(element: float | Interval) -> float:
    if isinstance(element, Interval):
        return element.mid_pos()
    else:
        return element  # Assuming float


def plot_ellipse_to_ax(ax: Axes, x_interval: Union[Interval, float], y_interval: Union[Interval, float], color, alpha,
                       label: Optional[str], min_width: Optional[float] = None, min_height: Optional[float] = None):
    if isinstance(x_interval, Interval):
        width = x_interval.length()
        mid_x = x_interval.mid_pos()
    else:
        width = 0.0
        mid_x = x_interval  # Assuming float
    new_alpha = alpha
    if min_width is not None and min_width > 0.0:
        width = max(width, min_width)
        new_alpha = new_alpha * (min_width / width)
    if isinstance(y_interval, Interval):
        height = y_interval.length()
        mid_y = y_interval.mid_pos()
    else:
        height = 0.0
        mid_y = y_interval  # Assuming float
    if min_height is not None and min_height > 0.0:
        height = max(height, min_height)
        new_alpha = new_alpha * (min_height / height)
    new_alpha = min(new_alpha*20, alpha)  # We increase it a bit to give more color.
    ellipse = Ellipse(xy=(mid_x, mid_y),
                      width=width, height=height,
                      facecolor=color, alpha=new_alpha, label=label)
    ax.add_patch(ellipse)


def plot_ellipses_to_ax(
        ax: Axes, x_intervals: Sequence[Interval], y_intervals: Sequence[Interval], color, alpha, label: str,
        min_width: Optional[float] = None, min_height: Optional[float] = None):
    for x, y in zip(x_intervals, y_intervals):
        plot_ellipse_to_ax(ax=ax, x_interval=x, y_interval=y, color=color, alpha=alpha,
                           label=label, min_width=min_width, min_height=min_height)
        label = None  # Sets the label only once for pretty legend.


def min_coord(coords: Sequence[Sequence[float] | Sequence[Interval]]) -> float:
    res = None
    for s in coords:
        for c in s:
            if isinstance(c, Interval):
                temp = c.a()
            else:
                temp = c
            if res is None or res > temp:
                res = temp
    return res


def max_coord(coords: Sequence[Sequence[float] | Sequence[Interval]]) -> float:
    res = None
    for s in coords:
        for c in s:
            if isinstance(c, Interval):
                temp = c.b()
            else:
                temp = c
            if res is None or res < temp:
                res = temp
    return res


def has_intervals(x: Sequence, y: Sequence) -> bool:
    for e in x:
        if isinstance(e, Interval):
            return True
    for e in y:
        if isinstance(e, Interval):
            return True
    return False


def average_same_x(x: Sequence[float], y: Sequence[float]) -> tuple[Sequence[float], Sequence[float]]:
    last_x = None
    mean_builder = None
    res_x = []
    res_y = []
    for xi, yi in zip(x, y):
        if xi != last_x:
            if last_x is not None:
                res_x.append(last_x)
                res_y.append(mean_builder.mean())
            last_x = xi
            mean_builder = KahanMeanBuilder()
        mean_builder.add(yi)
    if last_x is not None:
        res_x.append(last_x)
        res_y.append(mean_builder.mean())
    return res_x, res_y


def remove_nan(x: Sequence[float], y: Sequence[float]) -> tuple[Sequence[float], Sequence[float]]:
    res_x = []
    res_y = []
    for xi, yi in zip(x, y):
        if not (isnan(xi) or isnan(yi)):
            res_x.append(xi)
            res_y.append(yi)
    return res_x, res_y


def plot_interpolation_to_ax(ax: Axes, x: Sequence[float | Interval], y: Sequence[float | Interval], color):
    xf = [mid(xi) for xi in x]
    yf = [mid(yi) for yi in y]
    xf, yf = remove_nan(x=xf, y=yf)
    xf, yf = sort_both_by_first(seq1=xf, seq2=yf)
    xf, yf = average_same_x(x=xf, y=yf)  # Interpolation wants x to be unique.
    if len(xf) > 5:  # Interpolation algorithm wants at least 5 points.
        x_min = xf[0]
        x_max = xf[-1]
        cs = scipy.interpolate.make_smoothing_spline(np.array(xf), np.array(yf), lam=10.0)
        xs = np.arange(x_min, x_max, float(x_max-x_min)/200.0)
        ax.plot(xs, cs(xs), color=color)


def generic_multiclass_scatter_to_ax(ax: Axes,
                                     x: Sequence[Sequence[float] | Sequence[Interval]],
                                     y: Sequence[Sequence[float] | Sequence[Interval]],
                                     x_label=None, y_label=None, class_labels=None, colors=None,
                                     alpha: Optional[float] = None, legend_loc: str = 'best',
                                     x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
                                     font_size: int = FONT_SIZE,
                                     min_width: Optional[float] = None,
                                     vertical_lines: Sequence[float] = (),
                                     interpolate: bool = True):
    """x and y are one Sequence for each class."""
    n_classes = len(x)
    if len(y) != n_classes:
        raise ValueError("x and y must have the same number of classes.")
    for xi, yi in zip(x, y):
        if len(xi) != len(yi):
            raise ValueError("Each class must have the same number of x and y elements.")
    font_size = min(font_size, int(160.0 / n_classes))
    with plt.style.context({'font.size': font_size}):
        if class_labels is None:
            class_labels = []
            for i in range(n_classes):
                class_labels.append(str(i))
        if colors is None:
            colors = default_color_list(n_colors=n_classes, invert=True)
        if len(colors) < n_classes:
            colors = sns.color_palette(n_colors=n_classes)
        if alpha is None:
            tot_points = sum([len(x_i) for x_i in x])
            alpha = alpha_by_n_points(tot_points)
        for xc in vertical_lines:
            ax.axvline(x=xc)
        if x_min is not None:
            x_min_for_resize = x_min
        else:
            x_min_for_resize = min_coord(x)
        if x_max is not None:
            x_max_for_resize = x_max
        else:
            x_max_for_resize = max_coord(x)
        if y_min is not None:
            y_min_for_resize = y_min
        else:
            y_min_for_resize = min_coord(y)
        if y_max is not None:
            y_max_for_resize = y_max
        else:
            y_max_for_resize = max_coord(y)
        if min_width is None:
            min_width = 0.01
        min_width_for_resize = min_width * (x_max_for_resize - x_min_for_resize)
        min_height_for_resize = min_width * (y_max_for_resize - y_min_for_resize)
        for i in range(n_classes):
            x_i = x[i]
            y_i = y[i]
            if len(x_i) > 0 and has_intervals(x=x_i, y=y_i):
                plot_ellipses_to_ax(
                    ax=ax, x_intervals=x_i, y_intervals=y_i, color=colors[i], alpha=alpha,
                    label=class_labels[i], min_width=min_width_for_resize, min_height=min_height_for_resize)
            else:
                ax.scatter(x=x_i, y=y_i, c=[colors[i]], label=class_labels[i], alpha=alpha, edgecolors='none')
            if interpolate:
                plot_interpolation_to_ax(ax=ax, x=x_i, y=y_i, color=colors[i])
        border_size = 0.03
        if x_min is not None:
            ax.set_xlim(left=x_min)
        else:
            ax.set_xlim(left=x_min_for_resize - (border_size * (x_max_for_resize - x_min_for_resize)))
        if x_max is not None:
            ax.set_xlim(right=x_max)
        else:
            ax.set_xlim(right=x_max_for_resize + (border_size * (x_max_for_resize - x_min_for_resize)))
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        else:
            ax.set_ylim(bottom=y_min_for_resize - (border_size * (y_max_for_resize - y_min_for_resize)))
        if y_max is not None:
            ax.set_ylim(top=y_max)
        else:
            ax.set_ylim(top=y_max_for_resize + (border_size * (y_max_for_resize - y_min_for_resize)))
        ax.grid()
        bbox_to_anchor = None
        legend_cols = 1
        if legend_loc == 'upper center':
            bbox_to_anchor = (0.5, 1.1)
            legend_cols = n_classes
        leg = ax.legend(
            loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
            fancybox=False, shadow=False, ncol=legend_cols)
        for lh in leg.legendHandles:
            lh.set_alpha(1)  # Make legend opaque.
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=font_size)
        if y_label is not None:
            ax.set_ylabel(y_label, fontsize=font_size)


def multiclass_scatter_to_ax(ax, x: Sequence[Sequence[float]], y: Sequence[Sequence[float]],
                             x_label=None, y_label=None, class_labels=None, colors=None,
                             x_label_transform=None, alpha: Optional[float] = None, legend_loc: str = 'best',
                             x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
                             font_size: int = FONT_SIZE, vertical_lines: Sequence[float] = (),
                             interpolate: bool = True):
    """x and y are one Sequence for each class."""
    generic_multiclass_scatter_to_ax(ax=ax, x=x, y=y, x_label=x_label, y_label=y_label, class_labels=class_labels,
                                     colors=colors, alpha=alpha, legend_loc=legend_loc,
                                     x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                     font_size=font_size, vertical_lines=vertical_lines,
                                     interpolate=interpolate)


def multiclass_intervals_scatter_to_ax(
        ax, x: Sequence[Sequence[Interval]], y: Sequence[Sequence[Interval]],
        x_label=None, y_label=None, class_labels=None, colors=None,
        alpha: Optional[float] = None, legend_loc: str = 'best',
        x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
        font_size: int = FONT_SIZE,
        min_width: Optional[float] = None,
        vertical_lines: Sequence[float] = (),
        interpolate: bool = True):
    """x and y are one Sequence for each class."""
    generic_multiclass_scatter_to_ax(ax=ax, x=x, y=y, x_label=x_label, y_label=y_label, class_labels=class_labels,
                                     colors=colors, alpha=alpha, legend_loc=legend_loc,
                                     x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                     font_size=font_size, min_width=min_width, vertical_lines=vertical_lines,
                                     interpolate=interpolate)


def plot_multiclass_scatter(x: Sequence[Sequence[float] | Sequence[Interval]],
                            y: Sequence[Sequence[float] | Sequence[Interval]],
                            x_label=None, y_label=None, class_labels=None, colors=None,
                            alpha=None, legend_loc: str = 'best',
                            x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
                            min_width: Optional[float] = None,
                            interpolate: bool = True):
    """x and y are one Sequence for each class."""
    fig, ax = plt.subplots()
    generic_multiclass_scatter_to_ax(ax=ax, x=x, y=y, x_label=x_label, y_label=y_label,
                                     class_labels=class_labels, colors=colors,
                                     alpha=alpha, legend_loc=legend_loc,
                                     x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, min_width=min_width,
                                     interpolate=interpolate)


def save_multiclass_scatter(x: Sequence[Sequence[float] | Sequence[Interval]],
                            y: Sequence[Sequence[float] | Sequence[Interval]],
                            save_file: str, x_label=None, y_label=None, class_labels=None, colors=None,
                            alpha=None, legend_loc: str = 'best',
                            x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
                            min_width: Optional[float] = None,
                            interpolate: bool = True):
    """x and y are one Sequence for each class."""
    plot_multiclass_scatter(x=x, y=y, x_label=x_label, y_label=y_label,
                            class_labels=class_labels, colors=colors,
                            alpha=alpha, legend_loc=legend_loc,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, min_width=min_width,
                            interpolate=interpolate)
    smart_save_fig(path=save_file)


def grouped_barplot_by_dictionaries(groups, bar_width=0.25) -> matplotlib.pyplot:
    """ Receives a dictionary of dictionaries, the external is for groups and the internal for variables. """
    group_labels = list(groups.keys())
    var_labels = None
    groups_list = []
    for k in group_labels:
        v = groups[k]
        if var_labels is None:
            var_labels = list(v.keys())
        var_vals = []
        for vk in var_labels:
            var_vals.append(v[vk])
        groups_list.append(var_vals)
    return grouped_barplot_by_groups(
        groups=groups_list, group_labels=group_labels, var_labels=var_labels, bar_width=bar_width)


def grouped_barplot_by_groups(groups, group_labels=None, var_labels=None, bar_width=0.25) -> matplotlib.pyplot:
    """ Receives a list of lists, the external is for groups and the internals for variables. """
    print(groups)
    variables_list = []
    for g_i in range(len(groups)):
        g = groups[g_i]
        if g_i == 0:
            for v_i in range(len(g)):
                variables_list.append([g[v_i]])
        else:
            for v_i in range(len(g)):
                variables_list[v_i].append(g[v_i])
    return grouped_barplot_by_variables(
        variables=variables_list, group_labels=group_labels, var_labels=var_labels, bar_width=bar_width)


def grouped_barplot_by_variables(variables, group_labels=None, var_labels=None, bar_width=0.25) -> matplotlib.pyplot:
    """ Receives a list of lists, the external is for variables and the internals for groups. """

    # Set position of bar on X axis
    r = [np.arange(len(variables[0]))]
    for i in range(1, len(variables)):
        r.append([x + bar_width for x in r[i-1]])

    plt.figure(figsize=(15, 10))
    # Make the plot
    for i in range(len(variables)):
        if var_labels is None:
            var_lab = "var" + str(i)
        else:
            var_lab = var_labels[i]
        plt.bar(r[i], variables[i], color=cm.jet(1. * i / len(variables)),
                width=bar_width, edgecolor='white', label=var_lab)

    # Add xticks in the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    n_groups = len(variables[0])
    if group_labels is None:
        group_labels = []
        for i in range(n_groups):
            group_labels.append("g" + str(i))
    plt.xticks([r + bar_width for r in range(n_groups)], group_labels)

    # Create legend & Show graphic
    plt.legend()
    return plt
