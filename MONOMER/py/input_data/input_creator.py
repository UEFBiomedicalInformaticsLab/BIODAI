from abc import ABC, abstractmethod
import pandas as pd

import preprocess_views
from input_data.input_data import InputData
from plots import show_views
from plots.plot_utils import pca2d
from univariate_feature_selection.univariate_feature_selection import filter_views_pre_cv
from util.named import NickNamed
from util.printer.printer import Printer, NullPrinter


class InputCreator(NickNamed, ABC):
    __nick: str

    def __init__(self, nick: str):
        self.__nick = nick

    def create(self,
               views_to_load: [str],
               printer: Printer
               ) -> InputData:
        input_data = self.inner_create(views_to_load=views_to_load, printer=printer)
        pca2d(input_data, directory=self.plots_dir(), printer=printer)
        return input_data

    @abstractmethod
    def inner_create(self,
                     views_to_load: [str],
                     printer: Printer) -> InputData:
        raise NotImplementedError()

    def input_dir(self) -> str:
        return "./" + self.nick() + "/input/"

    def plots_dir(self) -> str:
        return self.input_dir() + "plots"

    def nick(self) -> str:
        return self.__nick

    def _common_preprocessing(
            self, views: dict[str, pd.DataFrame], printer: Printer = NullPrinter()) -> dict[str, pd.DataFrame]:
        printer.title_print("Global preprocessing of views")
        views = preprocess_views.preprocess_views(views, printer=printer)

        printer.title_print("Global feature selection")
        views = filter_views_pre_cv(views, printer=printer)

        printer.title_print("Plotting views")
        input_plots_dir = self.plots_dir()
        printer.print_variable("Directory for plots", input_plots_dir)
        show_views.plot_views(views, input_plots_dir, printer=printer)
        return views
