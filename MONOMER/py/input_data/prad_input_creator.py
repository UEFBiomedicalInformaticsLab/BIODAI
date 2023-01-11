import load_omics_views
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome
from util.printer.printer import Printer


class PradInputCreator(InputCreator):

    def __init__(self):
        InputCreator.__init__(self, nick="prad")

    def inner_create(self, views_to_load: [str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load, printer=printer)
        printer.title_print("Loading outcome")
        outcome_df = load_omics_views.load_outcome(input_dir)
        outcome = CategoricalOutcome(data=outcome_df, name="subtype")
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData.create_one_outcome(views=views, outcome=outcome, nick=self.nick())
