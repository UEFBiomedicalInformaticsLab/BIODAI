import load_omics_views
from input_data.fallback_locations import FallbackLocations, EMPTY_FALLBACK_LOCATIONS
from input_data.input_creator.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome
from util.printer.printer import Printer
from views.adjusted_view_definition import AdjustedViewDef


class ClassInputCreator(InputCreator):
    """Input creator with a single outcome for a classification problem."""
    __outcome_col: str
    __outcome_name: str

    def __init__(self, nick: str, outcome_col: str, outcome_name: str,
                 fallback_locations: FallbackLocations = EMPTY_FALLBACK_LOCATIONS):
        InputCreator.__init__(self, nick=nick, fallback_locations=fallback_locations)
        self.__outcome_col = outcome_col
        self.__outcome_name = outcome_name

    def inner_create(self, views_to_load: AdjustedViewDef, printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load.all_views_seq(), printer=printer,
                                                fallback_locations=self._fallback_locations())
        views_to_load = views_to_load.select_views(view_names=views.keys())
        printer.title_print("Loading outcome")
        outcome_df = load_omics_views.load_outcome(input_dir)
        outcome_df = outcome_df.rename(columns={self.__outcome_col: self.__outcome_name})
        outcome_df = outcome_df[[self.__outcome_name]]
        outcome = CategoricalOutcome(data=outcome_df, name=self.__outcome_name)
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData.create_one_outcome(views=views, outcome=outcome, nick=self.nick(),
                                            adjusted_views=views_to_load)
