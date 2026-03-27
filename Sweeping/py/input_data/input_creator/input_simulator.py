from collections.abc import Sequence

import numpy as np
from pandas import DataFrame

from input_data.input_creator.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome
from util.printer.printer import Printer
from util.randoms import set_all_seeds
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from views.adjusted_view_definition import AdjustedViewDef
from views.views import Views, JustViews


SIMULATED_NICK = "simulated"
CLASS_OUTCOME_NAME = "type"

DEFAULT_N_CLASSES = 3
DEFAULT_N_ROWS = 1000
DEFAULT_N_COLS = 200000


class InputSimulator(InputCreator):
    __n_row: int
    __n_col: int
    __n_classes: int

    def __init__(self, n_row: int, n_col: int, n_classes: int):
        InputCreator.__init__(self=self, nick=SIMULATED_NICK + str(n_row) + "x" + str(n_col) + "x" + str(n_classes))
        self.__n_row = n_row
        self.__n_col = n_col
        self.__n_classes = n_classes

    def simulate_view(self) -> Table:
        set_all_seeds(893454)
        from util.table.backed_table import BackedTable
        return BackedTable(NpTable(data=np.random.rand(self.__n_row, self.__n_col)))

    def simulate_all_views(self, views_to_load: Sequence[str], printer: Printer) -> Views:
        printer.title_print(
            "Simulating views with " + str(self.__n_row) + " rows and " + str(self.__n_col) + " columns")
        res = {}
        for v in views_to_load:
            printer.print("Creating simulated view " + v)
            res[v] = self.simulate_view()
        return JustViews(views_dict=res)

    def inner_create(self, views_to_load: AdjustedViewDef, printer: Printer) -> InputData:
        input_dir = self.input_dir()
        printer.print_variable("Simulated input directory", input_dir)
        views = self.simulate_all_views(views_to_load=views_to_load.all_views_seq(), printer=printer)

        printer.title_print("Simulating " + str(self.__n_row) + " outcomes with " + str(self.__n_classes) + " classes")
        outcomes_df = DataFrame(data=np.random.randint(self.__n_classes, size=(self.__n_row, 1))).astype('category')
        outcomes_df.columns = [CLASS_OUTCOME_NAME]
        outcome_categorical = CategoricalOutcome(data=outcomes_df, name=CLASS_OUTCOME_NAME)
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData.smart_create(
            all_views=views, outcomes=[outcome_categorical], nick=self.nick(), stratify_outcome=CLASS_OUTCOME_NAME,
            adjusted_views=views_to_load)
