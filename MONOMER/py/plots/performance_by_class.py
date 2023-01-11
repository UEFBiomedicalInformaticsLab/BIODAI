import math
import os
from collections.abc import Sequence

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame

from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import CONFUSION_MATRIX_STR, \
    LABEL_SEPARATOR
from external_validation.mo_external_evaluator.hof_saver import SOLUTION_FEATURES_EXTERNAL
from plots.plot_utils import smart_save_fig
from plots.hof_utils import feature_counts_from_df, hof_folds_feature_counts
from prediction_stats.confusion_matrix import ConfusionMatrix, PerformanceMeasure
from util.plot_results import multiclass_scatter_to_ax
from util.sequence_utils import flatten_iterable_of_iterable


def df_to_cms(df: DataFrame) -> Sequence[ConfusionMatrix]:
    col_names = df.columns
    num_classes = math.isqrt(len(col_names))
    labels = []
    for i in range(num_classes):
        labels.append(col_names[i*num_classes].partition(LABEL_SEPARATOR)[0])
    cms = []
    for row in df.values.tolist():
        cms.append(ConfusionMatrix.create_from_seq(seq=row, labels=labels))
    return cms


def read_one_fold_cms(cm_file: str) -> Sequence[ConfusionMatrix]:
    return df_to_cms(df=pd.read_csv(filepath_or_buffer=cm_file))


def read_all_cms(cm_dir: str) -> list[ConfusionMatrix]:
    """Fold files are read in alphabetical order. If passed path is not a directory returns an empty list."""
    cms = []
    if os.path.isdir(cm_dir):
        files = os.listdir(cm_dir)
        files.sort()
        for file in files:
            cms.extend(read_one_fold_cms(cm_file=os.path.join(cm_dir, file)))
    return cms


def read_cms_by_fold(cm_dir: str) -> Sequence[Sequence[ConfusionMatrix]]:
    """Fold files are read in alphabetical order."""
    files = os.listdir(cm_dir)
    files.sort()
    cms = []
    for file in files:
        cms.append(read_one_fold_cms(cm_file=os.path.join(cm_dir, file)))
    return cms


def read_cms_external(cm_dir: str) -> Sequence[ConfusionMatrix]:
    """Returns none if there is nothing to read."""
    files = os.listdir(cm_dir)
    cms = []
    if len(files) == 1:
        cms = read_one_fold_cms(cm_file=os.path.join(cm_dir, files[0]))
    return cms


def read_all_num_features(hof_dir: str) -> Sequence[int]:
    nested_counts = hof_folds_feature_counts(hof_dir=hof_dir)
    return flatten_iterable_of_iterable(x=nested_counts)


def read_all_num_features_external(hof_dir: str) -> Sequence[int]:
    for file in os.listdir(hof_dir):
        if file == SOLUTION_FEATURES_EXTERNAL:
            df = pd.read_csv(filepath_or_buffer=os.path.join(hof_dir, file))
            return feature_counts_from_df(df=df)
    return []


def cms_to_bal_accs(cms: Sequence[ConfusionMatrix]) -> Sequence[ndarray]:
    return [cm.balanced_accuracies() for cm in cms]


def one_performance_by_class_plot_to_ax(
        ax: Axes,
        cms: Sequence[ConfusionMatrix], num_features: Sequence[int], performance: PerformanceMeasure,
        performance_name: str, legend_loc: str = 'best', vertical_lines: Sequence[float] = ()):
    n_solutions = len(cms)
    if n_solutions > 0:
        perf = [cm.performance_measure(measure=performance) for cm in cms]
        n_classes = len(perf[0])
        labels = cms[0].labels()
        x = []
        y = []
        for c in range(n_classes):
            x.append(num_features)
            y_c = []
            for s in range(n_solutions):
                y_c.append(perf[s][c])
            y.append(y_c)
        multiclass_scatter_to_ax(ax=ax, x=x, y=y, x_label="number of features", y_label=performance_name,
                                 class_labels=labels, y_max=1.02, legend_loc=legend_loc,
                                 vertical_lines=vertical_lines)


def one_performance_by_class_plot(
        cms: Sequence[ConfusionMatrix], num_features: Sequence[int], hof_dir: str, performance: PerformanceMeasure,
        performance_name: str):
    n_solutions = len(cms)
    if n_solutions > 0:
        fig, ax = plt.subplots()
        one_performance_by_class_plot_to_ax(
            ax=ax, cms=cms, num_features=num_features, performance=performance,
            performance_name=performance_name)
        name_for_path = performance_name.replace(' ', '_')
        smart_save_fig(path=os.path.join(hof_dir, name_for_path + "_by_class.png"))


def confusion_matrices_from_hof_dir(hof_dir: str) -> list[ConfusionMatrix]:
    """If nothing to read returns empty list.
    TODO Does not handle cases with more than one categorical objective."""
    num_features = read_all_num_features(hof_dir=hof_dir)
    n_solutions = len(num_features)
    if n_solutions > 0:
        cm_dir = os.path.join(hof_dir, CONFUSION_MATRIX_STR)
        return read_all_cms(cm_dir=cm_dir)
    else:
        return []


def confusion_matrices_from_hof_dir_external(hof_dir: str) -> Sequence[ConfusionMatrix]:
    """If not external or nothing to read returns empty sequence.
    TODO Does not handle cases with more than one categorical objective."""
    num_features = read_all_num_features(hof_dir=hof_dir)
    n_solutions = len(num_features)
    if n_solutions > 0:
        cm_dir = os.path.join(hof_dir, CONFUSION_MATRIX_STR)
        if os.path.isdir(cm_dir):
            return read_cms_external(cm_dir=cm_dir)
        else:
            return []
    else:
        return []


def confusion_matrices_from_hof_dir_by_fold(hof_dir: str) -> Sequence[Sequence[ConfusionMatrix]]:
    """If nothing to read returns empty list.
    TODO Does not handle cases with more than one categorical objective."""
    num_features = read_all_num_features(hof_dir=hof_dir)
    n_solutions = len(num_features)
    if n_solutions > 0:
        cm_dir = os.path.join(hof_dir, CONFUSION_MATRIX_STR)
        if os.path.isdir(cm_dir):
            return read_cms_by_fold(cm_dir=cm_dir)
        else:
            return []
    else:
        return []


def performance_by_class_plots_one_hof(hof_dir: str):
    num_features = read_all_num_features(hof_dir=hof_dir)
    if len(num_features) > 0:
        cms = confusion_matrices_from_hof_dir(hof_dir=hof_dir)
        one_performance_by_class_plot(
            cms=cms, num_features=num_features, hof_dir=hof_dir,
            performance=PerformanceMeasure.balanced_accuracy, performance_name="balanced accuracy")
        one_performance_by_class_plot(
            cms=cms, num_features=num_features, hof_dir=hof_dir,
            performance=PerformanceMeasure.precision, performance_name="precision")
        one_performance_by_class_plot(
            cms=cms, num_features=num_features, hof_dir=hof_dir,
            performance=PerformanceMeasure.recall, performance_name="recall")


def performance_by_class_external_plots_one_hof(hof_dir: str):
    num_features = read_all_num_features_external(hof_dir=hof_dir)
    n_solutions = len(num_features)
    if n_solutions > 0:
        cm_dir = os.path.join(hof_dir, CONFUSION_MATRIX_STR)
        cms = read_all_cms(cm_dir=cm_dir)  # TODO Does not handle cases with more than one categorical objective.
        one_performance_by_class_plot(
            cms=cms, num_features=num_features, hof_dir=hof_dir,
            performance=PerformanceMeasure.balanced_accuracy, performance_name="balanced accuracy")
        one_performance_by_class_plot(
            cms=cms, num_features=num_features, hof_dir=hof_dir,
            performance=PerformanceMeasure.precision, performance_name="precision")
        one_performance_by_class_plot(
            cms=cms, num_features=num_features, hof_dir=hof_dir,
            performance=PerformanceMeasure.recall, performance_name="recall")


def performance_by_class_plots_every_hof(main_hofs_dir: str):
    if os.path.isdir(main_hofs_dir):
        subfolders = [f.path for f in os.scandir(main_hofs_dir) if f.is_dir()]
        for f in subfolders:
            performance_by_class_plots_one_hof(hof_dir=f)


def performance_by_class_external_plots_every_hof(main_hofs_dir: str):
    if os.path.isdir(main_hofs_dir):
        subfolders = [f.path for f in os.scandir(main_hofs_dir) if f.is_dir()]
        for f in subfolders:
            performance_by_class_external_plots_one_hof(hof_dir=f)
