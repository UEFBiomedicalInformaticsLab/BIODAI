import os
from collections.abc import Sequence

import pandas as pd

import load_omics_views
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome, SurvivalOutcome
from input_data.outcome_type import OutcomeType
from input_data.outcome_descriptor import OutcomeDescriptorWithColumns
from util.dataframes import has_column, common_row_names, n_row
from util.printer.printer import Printer
from util.survival.survival_utils import SURVIVAL_EVENT_STR, SURVIVAL_DURATION_STR


class SmartInputCreator(InputCreator):
    __outcome_descriptors: Sequence[OutcomeDescriptorWithColumns]

    def __init__(self, nick: str, outcome_descriptors: Sequence[OutcomeDescriptorWithColumns]):
        InputCreator.__init__(self=self, nick=nick)
        self.__outcome_descriptors = outcome_descriptors

    def inner_create(self, views_to_load: Sequence[str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load, printer=printer)
        # Views not checked for consistency yet.
        printer.title_print("Loading outcomes")
        to_load_path = os.path.join(input_dir, "outcome.csv")
        outcomes_df = pd.read_csv(filepath_or_buffer=to_load_path, index_col=0, keep_default_na=False, na_values="NA")
        # outcomes_df includes row names.
        printer.print_variable(var_name="Outcome rows", var_value=n_row(outcomes_df))
        common_rows = sorted(common_row_names(dfs=list(views.values()) + [outcomes_df]))
        for v in views:
            views[v] = views[v].loc[common_rows]
        outcomes_df = outcomes_df.loc[common_rows]

        outcomes = []
        for o in self.__outcome_descriptors:
            outcome_type = o.outcome_type()
            outcome_name = o.name()
            if outcome_type == OutcomeType.categorical:
                col_name = o.categories_col()
                if has_column(df=outcomes_df, col_name=col_name):
                    outcome_categorical = outcomes_df[[col_name]].astype('category')
                    outcome_categorical = (
                        outcome_categorical.rename(columns={col_name: outcome_name}))
                    outcome_categorical = CategoricalOutcome(data=outcome_categorical, name=outcome_name)
                    outcomes.append(outcome_categorical)
                else:
                    raise ValueError("Missing outcome column for outcome " + str(outcome_name))
            elif outcome_type == OutcomeType.survival:
                event_col = o.event_col()
                time_col = o.time_col()
                if has_column(df=outcomes_df, col_name=event_col) and has_column(df=outcomes_df, col_name=time_col):
                    outcome_survival = outcomes_df[[event_col, time_col]]
                    outcome_survival = outcome_survival.rename(columns={event_col: SURVIVAL_EVENT_STR})
                    outcome_survival[SURVIVAL_EVENT_STR] = outcome_survival[SURVIVAL_EVENT_STR].astype(int)
                    outcome_survival = outcome_survival.rename(columns={time_col: SURVIVAL_DURATION_STR})
                    outcome_survival[SURVIVAL_DURATION_STR] = outcome_survival[SURVIVAL_DURATION_STR].astype(int)
                    outcome_survival = SurvivalOutcome(data=outcome_survival, name=outcome_name)
                    outcomes.append(outcome_survival)
                else:
                    raise ValueError("Missing outcome columns for outcome " + str(outcome_name))
            else:
                raise ValueError("Unsupported outcome type.")

        views = self._common_preprocessing(views=views, printer=printer)
        stratify_outcome_name = None
        if len(outcomes) > 0:
            stratify_outcome_name = outcomes[0].name()
        return InputData(views=views, outcomes=outcomes, nick=self.nick(), stratify_outcome=stratify_outcome_name)