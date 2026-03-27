from collections.abc import Sequence
from math import ceil, sqrt
from string import ascii_uppercase
from typing import Optional, Literal, Union, Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from consts import FONT_SIZE
from plots.default_labels_map import LabelsTransformer, DEFAULT_LABELS_TRANSFORMER
from plots.plot_utils import smart_save_fig, default_color_list, add_row_names
from plots.plotter.plotter import Plotter
from util.grid import row_col_by_index
from util.iterable_utils import copy_and_append
from util.math.utils import ceil_division
from util.name_parts import names_by_differences
from util.str_utils import has_duplicates_case_insensitive
from matplotlib import ticker as mticker


def initialize_subplots(
        n_subplots: int,
        ncols: int,
        font_size: int = FONT_SIZE,
        sharex: Union[bool, str] = True,
        sharey: Union[bool, str] = True,
        x_stretch: float = 1.0) -> tuple[Figure, Any]:
    """Returned axes might be more than n_subplots if n_subplots cannot be divided by ncols.
    sharex and sharey control sharing of properties among x (sharex) or y (sharey) axes:
    True: x- or y-axis will be shared among all subplots.
    False: each subplot x- or y-axis will be independent.
    'row': each subplot row will share an x- or y-axis.
    'col': each subplot column will share an x- or y-axis."""
    with plt.style.context({'font.size': font_size}):
        nrows = max(ceil_division(num=n_subplots, den=ncols), 1)
        figsize_x = 4.0*ncols*x_stretch + 1.0
        figsize_y = 4.0*nrows + 1.0
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, dpi=600, figsize=(figsize_x, figsize_y), sharex=sharex, sharey=sharey,
            constrained_layout = True)
        return fig, axs


def adjust_n_cols(ncols: Optional[int], n_subplots: int) -> int:
    if ncols is None:
        if n_subplots == 3:
            return 1  # We prefer a vertical layout that fits in a page column.
        nrows = ceil(sqrt(n_subplots))
        ncols = ceil_division(n_subplots,nrows)
    if ncols > n_subplots:
        ncols = n_subplots
    if n_subplots < 2:
        ncols = 1
    return ncols


def remove_redundant_xy_labels(fig: Figure, ncols: int):
    """If all x or y labels are equal we use the common label for the whole plot."""
    axs = fig.get_axes()
    naxes = len(axs)
    if naxes == 0:
        return
    nrows = ceil_division(num=naxes, den=ncols)
    row_sets = [set() for _ in range(nrows)]
    col_sets = [set() for _ in range(ncols)]
    all_x = set()
    all_y = set()
    for i, a in enumerate(axs):
        row, col = row_col_by_index(index=i, ncol=ncols)
        ylab = a.get_ylabel()
        xlab = a.get_xlabel()
        row_sets[row].add(ylab)
        col_sets[col].add(xlab)
        all_x.add(xlab)
        all_y.add(ylab)
    all_x_redundant = len(all_x) < 2
    all_y_redundant = len(all_y) < 2
    for i, a in enumerate(axs):
        row, col = row_col_by_index(index=i, ncol=ncols)
        sbs = a.get_subplotspec()
        if all_y_redundant or (len(row_sets[row]) == 1 and not sbs.is_first_col()):
            a.set(ylabel=None)
        if all_x_redundant or (len(col_sets[col]) == 1 and not sbs.is_last_row()):
            a.set(xlabel=None)
    if all_x_redundant and len(all_x) == 1:
        fig.supxlabel(list(all_x)[0])
    if all_y_redundant and len(all_y) == 1:
        fig.supylabel(list(all_y)[0])


def needs_letters(box_names: Sequence[str]) -> bool:
    if "" in box_names:
        return True
    return has_duplicates_case_insensitive(strings=box_names)



def update_box_names(fig: Figure, boxes_name_parts: Sequence[Sequence[str]],
                     labels_transformer: LabelsTransformer = DEFAULT_LABELS_TRANSFORMER,
                     add_letters: bool = True):
    """If all x or y labels are equal we use the common label for the whole plot."""
    axs = fig.get_axes()
    n_boxes = len(boxes_name_parts)

    if n_boxes > 1:
        name_parts_with_old_titles = [
            copy_and_append(boxes_name_parts[i], axs[i].get_title()) for i, names in enumerate(boxes_name_parts)]
        box_names = labels_transformer.apply_all(names_by_differences(object_features=name_parts_with_old_titles))
        if add_letters:
            add_letters = needs_letters(box_names=box_names)
        for i, box_name in enumerate(box_names):
            letter = ascii_uppercase[i]
            if box_name != "":
                if add_letters:
                    new_title = letter + " - " + box_name
                else:
                    new_title = box_name
            else:
                if add_letters:
                    new_title = letter
                else:
                    new_title = ""
            axs[i].set_title(new_title)


def legends_are_identical(axes):
    """Return True if all axes have identical legend labels (case-sensitive)."""

    # Collect legend items for each axes
    legends = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legends.append(labels)

    # If any subplot has no legend, they cannot be identical
    if any(len(lbls) == 0 for lbls in legends):
        return False

    # Compare all to the first
    first = legends[0]
    return all(lbls == first for lbls in legends)


def subplots(
        plotters: Sequence[Plotter],
        save_path: str, ncols: Optional[int] = 2,
        x_label: Optional[str] = None, y_label: Optional[str] = None,
        color_by_row: bool = False,
        font_size: int = FONT_SIZE, x_stretch: float = 1.0,
        sharex: Union[bool,  Literal["row", "col"]] = True,
        sharey: Union[bool,  Literal["row", "col"]] = True,
        row_names: Optional[Sequence[str]] = None,
        subplots_adjust_left: Optional[float] = None,
        desat: Optional[float] = None,
        labels_transformer: LabelsTransformer = DEFAULT_LABELS_TRANSFORMER,
        add_letters: bool = True
        ):
    """color_by_row and desat have an effect only if the plotters use a color passed from outside.
    sharex and sharey control sharing of properties among x (sharex) or y (sharey) axes:
    True: x- or y-axis will be shared among all subplots.
    False: each subplot x- or y-axis will be independent.
    'row': each subplot row will share an x- or y-axis.
    'col': each subplot column will share an x- or y-axis."""
    n_subplots = len(plotters)
    if n_subplots > 0:
        ncols = adjust_n_cols(ncols=ncols, n_subplots=n_subplots)
        fig, axs = initialize_subplots(
            n_subplots=n_subplots,
            ncols=ncols,
            font_size=font_size, x_stretch=x_stretch,
            sharex=sharex, sharey=sharey)
        nrows = max(ceil_division(num=n_subplots, den=ncols), 1)
        n_boxes = nrows * ncols
        if color_by_row:
            n_colors = nrows
        else:
            n_colors = n_boxes
        color_list = default_color_list(n_colors=n_colors, desat=desat)
        boxes_name_parts = [[] for _ in range(n_subplots)]
        with plt.style.context({'font.size': font_size}):
            axes = fig.get_axes()
            for i, plot_axs in enumerate(axes):
                if i < n_subplots:
                    row = i // ncols
                    if color_by_row:
                        color = color_list[row]
                    else:
                        color = color_list[i]
                    plotters[i].plot(ax=plot_axs, color=color)
                    if n_subplots > 1:
                        boxes_name_parts[i].extend(plotters[i].name_parts())
                else:
                    plot_axs.axis('off')
            update_box_names(
                fig=fig, boxes_name_parts=boxes_name_parts, labels_transformer=labels_transformer,
                add_letters=add_letters)
            remove_redundant_xy_labels(fig=fig, ncols=ncols)
            if x_label is not None:
                fig.supxlabel(labels_transformer.apply(x_label))
            if y_label is not None:
                fig.supylabel(labels_transformer.apply(y_label))
            add_row_names(fig=fig, axs=axs, ncols=ncols, row_names=row_names,
                          subplots_adjust_left=subplots_adjust_left)

            # Only consider the real axes (ignore the extra "off" ones if any)
            axes = [ax for ax in fig.get_axes()][:n_subplots]

            if legends_are_identical(axes):
                # Use only ONE legend: in the top–right subplot
                # Determine which axes is “upper right”
                # → It’s row 0, last column:
                top_right_ax = axes[min(n_subplots - 1, ncols - 1)]

                # Remove legends from all other axes
                for ax in axes:
                    if ax is not top_right_ax:
                        lgd = ax.get_legend()
                        if lgd:
                            lgd.remove()

            # Tell the locator not to prune end labels
            for ax in axes:
                ax.xaxis.set_major_locator(mticker.AutoLocator())
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

            smart_save_fig(path=save_path, bbox_inches_tight=False)
