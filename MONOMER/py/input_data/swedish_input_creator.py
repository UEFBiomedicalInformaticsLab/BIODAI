import os

import pandas as pd
from pandas import Series

from input_data.brca_input_creator import OUTCOME_PAM50_NAME, OUTCOME_SURVIVAL_NAME
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome, SurvivalOutcome
from model.survival_model import SURVIVAL_EVENT_STR, SURVIVAL_DURATION_STR
from util.printer.printer import Printer


SWEDISH_NICK = "swedish"


class SwedishInputCreator(InputCreator):

    def __init__(self):
        InputCreator.__init__(self, nick=SWEDISH_NICK)

    def inner_create(self, views_to_load: [str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = {"mrna": pd.read_csv(
            filepath_or_buffer=input_dir + "mrna.csv", header=0, index_col=0, keep_default_na=True)}
            # TODO There is a spurious column "X" in the file.
        df = views["mrna"]
        df.reset_index(drop=True, inplace=True)
        printer.print("Setting negative values to 0.0")
        for c in df.columns:  # Cut below zero
            df[c].loc[df[c] < 0.0] = 0.0

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
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData(views=views, outcomes=outcomes, nick=self.nick(), stratify_outcome=OUTCOME_PAM50_NAME)
