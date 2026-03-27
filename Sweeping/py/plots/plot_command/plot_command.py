from collections.abc import Iterable, Sequence
from typing import Optional

from plots.archives.test_battery import TestBattery
from plots.runnable.summary_feature_table_writer import MAX_TABLE_CELLS_REASONABLE
from plots.runnable.summary_statistics_plotter_from_registries import DEFAULT_REGISTRY_PROPERTIES
from validation_registry.registry_property import RegistryProperty


class PlotCommand:
    __batteries: Iterable[TestBattery]
    __run_postprocessing: bool
    __run_detailed_plots: bool
    __create_summary_feature_tables: bool
    __max_table_cells: Optional[int]
    __properties: Sequence[RegistryProperty]
    __show_gene_counts: bool

    def __init__(self,
                 batteries: Iterable[TestBattery],
                 run_postprocessing: bool = False,
                 run_detailed_plots: bool = True,
                 create_summary_feature_tables: bool = False,
                 max_table_cells: Optional[int] = MAX_TABLE_CELLS_REASONABLE,
                 properties: Sequence[RegistryProperty] = DEFAULT_REGISTRY_PROPERTIES,
                 show_gene_counts: bool = False):
        """max_table_cells: None to have no maximum."""
        self.__batteries = batteries
        self.__run_postprocessing = run_postprocessing
        self.__run_detailed_plots = run_detailed_plots
        self.__create_summary_feature_tables = create_summary_feature_tables
        self.__max_table_cells = max_table_cells
        self.__properties = properties
        self.__show_gene_counts = show_gene_counts

    def batteries(self) -> Iterable[TestBattery]:
        return self.__batteries

    def run_postprocessing(self) -> bool:
        return self.__run_postprocessing

    def run_detailed_plots(self) -> bool:
        return self.__run_detailed_plots

    def create_summary_feature_tables(self) -> bool:
        return self.__create_summary_feature_tables

    def max_table_cells(self) -> Optional[int]:
        return self.__max_table_cells

    def properties(self) -> Sequence[RegistryProperty]:
        return self.__properties

    def show_gene_counts(self) -> bool:
        return self.__show_gene_counts
