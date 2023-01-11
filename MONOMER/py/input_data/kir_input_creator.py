import os

import pandas as pd
import load_omics_views
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome
from util.printer.printer import Printer


OUTCOME_INPUT_COL = "PanKidney Pathology"
OUTCOME_NAME = "PanKidney"


class KirInputCreator(InputCreator):

    def __init__(self):
        InputCreator.__init__(self, nick="tcga_kir")

    def inner_create(self, views_to_load: [str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load, printer=printer)
        printer.title_print("Loading outcomes")
        to_load_path = os.path.join(input_dir, "outcome.csv")
        outcome_df = pd.read_csv(filepath_or_buffer=to_load_path, index_col=0, keep_default_na=False, na_values="NA")
        outcome_df = outcome_df.rename(columns={OUTCOME_INPUT_COL: OUTCOME_NAME})
        outcome = CategoricalOutcome(
            data=outcome_df[[OUTCOME_NAME]], name=OUTCOME_NAME)
        outcomes = [outcome]
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData(views=views, outcomes=outcomes, nick=self.nick(), stratify_outcome=OUTCOME_NAME)
