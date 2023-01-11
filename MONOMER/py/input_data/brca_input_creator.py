import os

import pandas as pd
from pandas import Series

import load_omics_views
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome, SurvivalOutcome
from model.survival_model import SURVIVAL_DURATION_STR, SURVIVAL_EVENT_STR
from util.printer.printer import Printer

OUTCOME_SURVIVAL_NAME = "survival"
OUTCOME_PAM50_NAME = "Pam50"
PAM50_CLASSES = {'Normal', 'LumA', 'LumB', 'Basal', 'Her2'}


class BrcaInputCreator(InputCreator):

    def __init__(self):
        InputCreator.__init__(self, nick="brca")

    def inner_create(self, views_to_load: [str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load, printer=printer)
        printer.title_print("Loading outcomes")
        to_load_path = os.path.join(input_dir, "pheno.csv")
        pheno = pd.read_csv(filepath_or_buffer=to_load_path, index_col=0, keep_default_na=False, na_values="NA")
        outcome_pam50 = pheno.Pam50.astype('category')
        outcome_pam50 = Series.to_frame(outcome_pam50, name=OUTCOME_PAM50_NAME)
        outcome_pam50 = CategoricalOutcome(data=outcome_pam50, name=OUTCOME_PAM50_NAME)
        outcome_survival = pheno[['OverallSurv', 'days_to_death', 'days_to_last_followup']]
        outcome_survival = outcome_survival.rename(columns={'OverallSurv': SURVIVAL_EVENT_STR})
        days = outcome_survival[['days_to_death', 'days_to_last_followup']].abs().max(axis=1)
        outcome_survival = outcome_survival.drop(labels='days_to_death', axis=1)
        outcome_survival = outcome_survival.drop(labels='days_to_last_followup', axis=1)
        outcome_survival = outcome_survival.assign(**{SURVIVAL_DURATION_STR: days})
        outcome_survival = SurvivalOutcome(data=outcome_survival, name=OUTCOME_SURVIVAL_NAME)
        outcomes = [outcome_pam50, outcome_survival]
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData(views=views, outcomes=outcomes, nick=self.nick(), stratify_outcome=OUTCOME_PAM50_NAME)
