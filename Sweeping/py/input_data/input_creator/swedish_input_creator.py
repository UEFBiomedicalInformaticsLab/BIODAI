import os
from typing import Union

import pandas as pd
from pandas import Series, DataFrame

from input_data.input_creator.brca_input_creator import OUTCOME_PAM50_NAME, OUTCOME_SURVIVAL_NAME
from input_data.input_creator.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome, SurvivalOutcome
from util.survival.survival_utils import SURVIVAL_DURATION_STR, SURVIVAL_EVENT_STR
from util.printer.printer import Printer
from util.table.table import Table
from util.table.table_backend.np_table import NpTable
from views.adjusted_view_definition import AdjustedViewDef
from views.views import JustViews

SWEDISH_NICK = "swedish"


class SwedishInputCreator(InputCreator):
    """This legacy input creator ignores fallback locations."""
    __drop_many_zeroes: bool

    def __init__(self, drop_many_zeroes: bool = True):
        InputCreator.__init__(self, nick=SWEDISH_NICK)
        self.__drop_many_zeroes = drop_many_zeroes

    def inner_create(self, views_to_load: AdjustedViewDef, printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views: dict[str,Union[DataFrame,Table]] = {"mrna": pd.read_csv(
            filepath_or_buffer=input_dir + "mrna.csv", header=0, index_col=0, keep_default_na=True)}
        # TODO There is a spurious column "X" in the file.
        views_to_load = views_to_load.select_views(view_names=views.keys())
        df = views["mrna"]
        df.reset_index(drop=True, inplace=True)
        printer.print("Setting negative values to 0.0")
        for c in df.columns:  # Cut below zero
            df.loc[df[c] < 0.0, c] = 0.0

        if self.__drop_many_zeroes:
            printer.print("Dropping columns with 70% or more zeros.")
            printer.print_variable("Columns before dropping", len(df.columns))
            column_cut_off = int(70 / 100 * len(df))
            b = (df == 0).sum(axis='rows')
            df = df[b[b <= column_cut_off].index.values]
            printer.print_variable("Columns after dropping", len(df.columns))

        views["mrna"] = df

        printer.title_print("Loading outcomes")
        to_load_pheno_path = os.path.join(input_dir, "pheno.csv")
        pheno = pd.read_csv(filepath_or_buffer=to_load_pheno_path, index_col=0, keep_default_na=True)
        pheno.reset_index(drop=True, inplace=True)
        outcome_pam50 = pheno.Pam50.astype('category')
        outcome_pam50 = Series.to_frame(outcome_pam50, name=OUTCOME_PAM50_NAME)
        outcome_pam50 = CategoricalOutcome(data=outcome_pam50, name=OUTCOME_PAM50_NAME)
        outcome_survival = pheno[['OverallSurv', 'SurvDays']]
        outcome_survival = outcome_survival.rename(columns={'OverallSurv': SURVIVAL_EVENT_STR})
        outcome_survival = outcome_survival.rename(columns={'SurvDays': SURVIVAL_DURATION_STR})
        outcome_survival = SurvivalOutcome(data=outcome_survival, name=OUTCOME_SURVIVAL_NAME)
        outcomes = [outcome_pam50, outcome_survival]
        for v in views:
            from util.table.backed_table import BackedTable
            views[v] = BackedTable(NpTable(data=views[v]))
        just_views = JustViews(views_dict=views)
        just_views = self._common_preprocessing(views=just_views, printer=printer)
        return InputData.smart_create(
            all_views=just_views, outcomes=outcomes, nick=self.nick(), stratify_outcome=OUTCOME_PAM50_NAME,
            adjusted_views=views_to_load)
