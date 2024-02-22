import os

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from util.printer.printer import Printer, OutPrinter


def show_boxplot_first_cols(view, max_box = 10):
    to_use = view.iloc[0:,0:max_box]
    fig = plt.figure()
    to_use.boxplot()
    plt.show()
    plt.close(fig)


def boxplot_first_cols(view, name, dir, printer: Printer = OutPrinter(), max_box=10):
    to_use = view.iloc[0:, 0:max_box]
    fig = plt.figure()
    to_use.boxplot()
    try:
        plt.savefig(os.path.join(dir, name + "_boxplot_cols" + ".png"), dpi=600, bbox_inches='tight')
    except BaseException as e:
        printer.print("Exception while saving figure:\n" + str(e))
    plt.close(fig)


def boxplot_first_rows(view, name, dir, printer: Printer = OutPrinter(), max_box = 10):
    to_use = view.transpose().iloc[0:,0:max_box]
    fig = plt.figure()
    to_use.boxplot()
    try:
        plt.savefig(os.path.join(dir, name + "_boxplot_rows" + ".png"), dpi=600, bbox_inches='tight')
    except BaseException as e:
        printer.print("Exception while saving figure:\n" + str(e))
    plt.close(fig)


def freedman_diaconis_bin_number(x, max_bins=100):
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2*(q75 - q25)*len(x)**(-1/3)
    if bin_width == 0.0:
        return max_bins
    bins = min(round((x.max() - x.min())/bin_width), max_bins)
    return bins


def plot_view(view, name, directory, printer: Printer = OutPrinter()):
    l = view.to_numpy().flatten()
    l = l[~np.isnan(l)]
    bins = freedman_diaconis_bin_number(l)
    fig = plt.figure()
    plt.hist(l, density=True, bins=bins)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(os.path.join(directory, name + "_density.png"), dpi=600, bbox_inches='tight')
    except BaseException as e:
        printer.print("Exception while saving figure:\n" + str(e))
    plt.close(fig)
    boxplot_first_cols(view=view, name=name, dir=directory, printer=printer)
    boxplot_first_rows(view=view, name=name, dir=directory, printer=printer)


def plot_views(views, directory, printer: Printer):
    for v in views:
        printer.print("Plotting view " + v)
        plot_view(views[v], v, directory, printer=printer)
