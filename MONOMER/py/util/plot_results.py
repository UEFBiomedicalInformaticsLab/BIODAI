from collections.abc import Sequence
from typing import Optional

from matplotlib import cm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors

from consts import FONT_SIZE
from plots.plot_utils import smart_save_fig, default_color_list


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
        return min(500.0 / n_points, 1.0)


def multiclass_scatter_to_ax(ax, x: Sequence[Sequence[float]], y: Sequence[Sequence[float]],
                             x_label=None, y_label=None, class_labels=None, colors=None,
                             x_label_transform=None, alpha: Optional[float] = None, legend_loc: str = 'best',
                             x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None,
                             font_size: int = FONT_SIZE, vertical_lines: Sequence[float] = ()):
    """x and y are one Sequence for each class."""
    with plt.style.context({'font.size': font_size}):
        n_classes = len(x)
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
        for i in range(n_classes):
            ax.scatter(x=x[i], y=y[i], c=[colors[i]], label=class_labels[i], alpha=alpha, edgecolors='none')
        if x_min is not None:
            ax.set_xlim(left=x_min)
        if x_max is not None:
            ax.set_xlim(right=x_max)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)
        ax.grid()
        bbox_to_anchor = None
        legend_cols = 1
        if legend_loc == 'upper center':
            bbox_to_anchor = (0.5, 1.1)
            legend_cols = n_classes
        ax.legend(loc=legend_loc,
                  bbox_to_anchor=bbox_to_anchor,
                  fancybox=False, shadow=False, ncol=legend_cols)
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=font_size)
        if y_label is not None:
            ax.set_ylabel(y_label, fontsize=font_size)


def plot_multiclass_scatter(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]],
                            x_label=None, y_label=None, class_labels=None, colors=None,
                            x_label_transform=None, alpha=None, legend_loc: str = 'best',
                            x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None):
    """x and y are one Sequence for each class."""
    fig, ax = plt.subplots()
    multiclass_scatter_to_ax(ax=ax, x=x, y=y, x_label=x_label, y_label=y_label,
                             class_labels=class_labels, colors=colors, x_label_transform=x_label_transform,
                             alpha=alpha, legend_loc=legend_loc,
                             x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def save_multiclass_scatter(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]],
                            save_file: str, x_label=None, y_label=None, class_labels=None, colors=None,
                            x_label_transform=None, alpha=None, legend_loc: str = 'best',
                            x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None):
    """x and y are one Sequence for each class."""
    plot_multiclass_scatter(x=x, y=y, x_label=x_label, y_label=y_label,
                            class_labels=class_labels, colors=colors, x_label_transform=x_label_transform,
                            alpha=alpha, legend_loc=legend_loc,
                            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
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
        vars=variables_list, group_labels=group_labels, var_labels=var_labels, bar_width=bar_width)


def grouped_barplot_by_variables(vars, group_labels=None, var_labels=None, bar_width=0.25) -> matplotlib.pyplot:
    """ Receives a list of lists, the external is for variables and the internals for groups. """

    # Set position of bar on X axis
    r = [np.arange(len(vars[0]))]
    for i in range(1, len(vars)):
        r.append([x + bar_width for x in r[i-1]])

    plt.figure(figsize=(15, 10))
    # Make the plot
    for i in range(len(vars)):
        if var_labels is None:
            var_lab = "var" + str(i)
        else:
            var_lab = var_labels[i]
        plt.bar(r[i], vars[i], color=cm.jet(1.*i/len(vars)), width=bar_width, edgecolor='white', label=var_lab)

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    n_groups = len(vars[0])
    if group_labels is None:
        group_labels = []
        for i in range(n_groups):
            group_labels.append("g" + str(i))
    plt.xticks([r + bar_width for r in range(n_groups)], group_labels)

    # Create legend & Show graphic
    plt.legend()
    return plt
