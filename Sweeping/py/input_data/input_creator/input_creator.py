from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

from input_data.fallback_locations import FallbackLocations, EMPTY_FALLBACK_LOCATIONS
from input_data.input_data import InputData
from util.named import NickNamed
from util.printer.printer import Printer, NullPrinter, UNBUFFERED_OUT_PRINTER
from util.table.table_representation_strategy import TableRepresentationStrategy, TableRepresentationStrategyDisk
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews

INPUT_DIR_NAME = "input"
DEFAULT_TABLE_REPRESENTATION_STRATEGY = TableRepresentationStrategyDisk()


class InputCreator(NickNamed, ABC):
    __nick: str
    __name: str

    def __init__(self, nick: str, fallback_locations: FallbackLocations = EMPTY_FALLBACK_LOCATIONS,
                 name: Optional[str] = None):
        self.__nick = nick
        if name is None:
            name = nick
        self.__name = name
        self.__fallback_locations = fallback_locations

    def _fallback_locations(self) -> FallbackLocations:
        return self.__fallback_locations

    def create(self,
               views_to_load: AdjustedViewDef,
               printer: Printer = UNBUFFERED_OUT_PRINTER,
               table_representation: TableRepresentationStrategy = DEFAULT_TABLE_REPRESENTATION_STRATEGY,
               skip_plotting_huge_views: bool = True,
               covariate_views: Optional[Iterable[str]] = None
               ) -> InputData:
        """TODO Can change here to have alternative matching by sample ID, so that samples may be filtered
        according to presence in all the actually used views. The plots dir could be of the form
        brca/mirna_mrna/plots"""
        from plots.plot_input_data import plot_input_data
        input_data = self.inner_create(views_to_load=views_to_load, printer=printer)
        printer.title_print("Applying representation strategy " + table_representation.nick())
        res_views = {}
        for v in input_data.view_names_seq():
            input_data.view_names_seq()
            res_views[v] = table_representation.represent(
                table=input_data.view(view_name=v), directory=self.cache_dir(), table_name=v)
        input_data = input_data.set_views(views=JustViews(views_dict=res_views))
        input_data = input_data.set_covariate_views(covariate_views=covariate_views)
        plot_input_data(
            input_data=input_data, plots_dir=self.plots_dir(), printer=printer,
            skip_huge_views = skip_plotting_huge_views)
        return input_data

    @abstractmethod
    def inner_create(self,
                     views_to_load: AdjustedViewDef,
                     printer: Printer) -> InputData:
        raise NotImplementedError()

    def input_dir(self) -> str:
        return "./" + self.nick() + "/" + INPUT_DIR_NAME + "/"

    def cache_dir(self) -> str:
        return "./" + self.nick() + "/" + "cache" + "/"

    def plots_dir(self) -> str:
        return self.input_dir() + "plots"

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    def _common_preprocessing(
            self, views: Views, printer: Printer = NullPrinter()) -> Views:
        from univariate_feature_selection.univariate_feature_selection import filter_views_pre_cv
        views = views.fast_cols()
        printer.title_print("Global feature selection")
        views = filter_views_pre_cv(views, printer=printer)
        return views
