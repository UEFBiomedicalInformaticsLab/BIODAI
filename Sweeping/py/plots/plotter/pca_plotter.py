from collections.abc import Sequence

from matplotlib.axes import Axes
from pandas import DataFrame

from plots.pca import pca2d_view_ax, pca2d_from_components, principal_components, pc_str
from plots.plotter.plotter import Plotter
from util.dataframe.dataframes import col_as_list
from util.printer.printer import NULL_PRINTER, Printer
from util.table.table import Table
from util.table.table_utils import n_col

PCA_PLOTTER_SHOW_COUNTS = True
PCA_PLOTTER_ORDER_BY_COUNTS = False
PCA_PLOTTER_POINT_SIZE = 10
PCA_PLOTTER_LEGEND_LOC = 'upper right'


class PcaPlotter(Plotter):
    __view_data: Table
    __outcome_data: DataFrame
    __principal_component_a: int
    __principal_component_b: int

    def __init__(self, view_data: Table, outcome_data: DataFrame,
                 principal_component_a: int = 0,
                 principal_component_b: int = 1):
        self.__view_data = view_data
        self.__outcome_data = outcome_data
        self.__principal_component_a = principal_component_a
        self.__principal_component_b = principal_component_b

    def plot(self, ax: Axes, color=None):
        pca2d_view_ax(ax=ax, view=self.__view_data, outcome=self.__outcome_data,
                      show_counts=PCA_PLOTTER_SHOW_COUNTS, order_by_counts=PCA_PLOTTER_ORDER_BY_COUNTS,
                      point_size=PCA_PLOTTER_POINT_SIZE, legend_loc=PCA_PLOTTER_LEGEND_LOC,
                      principal_component_a=self.__principal_component_a,
                      principal_component_b=self.__principal_component_b)


class PcaPlotterFromComponents(Plotter):
    __principal_component_a: Sequence[float]
    __principal_component_b: Sequence[float]
    __pc_a_name: str
    __pc_b_name: str
    __outcome: Sequence
    __show_counts: bool

    def __init__(self,
                 principal_component_a: Sequence[float],
                 principal_component_b: Sequence[float],
                 pc_a_name: str,
                 pc_b_name: str,
                 outcome: Sequence,
                 show_counts: bool = PCA_PLOTTER_SHOW_COUNTS
                 ):
        self.__principal_component_a = principal_component_a
        self.__principal_component_b = principal_component_b
        self.__pc_a_name = pc_a_name
        self.__pc_b_name = pc_b_name
        self.__show_counts =show_counts
        if isinstance(outcome,Sequence):
            self.__outcome = outcome
        else:
            raise ValueError("Outcome is not a Sequence.")

    def plot(self, ax: Axes, color=None):
        pca2d_from_components(
            ax=ax,
            principal_component_a = self.__principal_component_a,
            principal_component_b = self.__principal_component_b,
            pc_a_name=self.__pc_a_name,
            pc_b_name=self.__pc_b_name,
            outcome=self.__outcome,
            show_counts=self.__show_counts,
            order_by_counts=PCA_PLOTTER_ORDER_BY_COUNTS,
            point_size=PCA_PLOTTER_POINT_SIZE,
            legend_loc=PCA_PLOTTER_LEGEND_LOC)


def pca_component_combinations_from_components(
        principal_df: DataFrame, outcome_seq: Sequence, n_components: int,
        show_counts: bool = PCA_PLOTTER_SHOW_COUNTS) -> Sequence[Plotter]:
    res = []
    n_components = min(n_components, n_col(principal_df))
    for i in range(n_components):
        for j in range(i+1, n_components):
            res.append(PcaPlotterFromComponents(
                principal_component_a=col_as_list(df=principal_df, col=i),
                principal_component_b=col_as_list(df=principal_df, col=j),
                pc_a_name=pc_str(pc_index=i),
                pc_b_name=pc_str(pc_index=j),
                outcome=outcome_seq,
                show_counts=show_counts))
    return res


def pca_component_combinations(
        view_data: Table, outcome_data: DataFrame, n_components: int,
        printer: Printer = NULL_PRINTER) -> Sequence[Plotter]:
    principal_df, outcome = principal_components(
        view=view_data,
        outcome=outcome_data,
        n_components=n_components,
        printer=printer)
    outcome_list = outcome.iloc[:, 0].astype("category").cat.codes.tolist()
    return pca_component_combinations_from_components(
        principal_df=principal_df, outcome_seq=outcome_list, n_components=n_components)
