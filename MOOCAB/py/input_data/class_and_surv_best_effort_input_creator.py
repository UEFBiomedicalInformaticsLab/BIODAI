import os
from collections.abc import Sequence

import pandas as pd

import load_omics_views
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import SurvivalOutcome, CategoricalOutcome
from util.dataframes import has_column
from util.survival.survival_utils import SURVIVAL_DURATION_STR, SURVIVAL_EVENT_STR
from util.printer.printer import Printer


OUTCOME_SURVIVAL_NAME = "survival"


class ClassAndSurvBestEffortInputCreator(InputCreator):
    __class_outcome_col: str  # The column in the csv
    __class_outcome_name: str  # The name in the InputData
    __surv_event_col: str
    __surv_time_col: str

    def __init__(self, nick: str, class_outcome_col: str, class_outcome_name: str = "type",
                 surv_event_col: str = "Event", surv_time_col: str = "Time"):
        InputCreator.__init__(self, nick=nick)
        self.__class_outcome_col = class_outcome_col
        self.__class_outcome_name = class_outcome_name
        self.__surv_event_col = surv_event_col
        self.__surv_time_col = surv_time_col

    def inner_create(self, views_to_load: Sequence[str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load, printer=printer)
        printer.title_print("Loading outcomes")
        to_load_path = os.path.join(input_dir, "outcome.csv")

        pheno = pd.read_csv(filepath_or_buffer=to_load_path, index_col=0, keep_default_na=False, na_values="NA")

        outcomes = []

        if has_column(df=pheno, col_name=self.__class_outcome_col):
            outcome_categorical = pheno[[self.__class_outcome_col]].astype('category')
            outcome_categorical =(
                outcome_categorical.rename(columns={self.__class_outcome_col: self.__class_outcome_name}))
            outcome_categorical = CategoricalOutcome(data=outcome_categorical, name=self.__class_outcome_name)
            outcomes.append(outcome_categorical)

        if has_column(df=pheno, col_name=self.__surv_event_col) and has_column(df=pheno, col_name=self.__surv_time_col):
            outcome_survival = pheno[[self.__surv_event_col, self.__surv_time_col]]
            outcome_survival = outcome_survival.rename(columns={self.__surv_event_col: SURVIVAL_EVENT_STR})
            outcome_survival[SURVIVAL_EVENT_STR] = outcome_survival[SURVIVAL_EVENT_STR].astype(int)
            outcome_survival = outcome_survival.rename(columns={self.__surv_time_col: SURVIVAL_DURATION_STR})
            outcome_survival[SURVIVAL_DURATION_STR] = outcome_survival[SURVIVAL_DURATION_STR].astype(int)
            outcome_survival = SurvivalOutcome(data=outcome_survival, name=OUTCOME_SURVIVAL_NAME)
            outcomes.append(outcome_survival)

        views = self._common_preprocessing(views=views, printer=printer)
        return InputData(views=views, outcomes=outcomes, nick=self.nick(), stratify_outcome=self.__class_outcome_name)
