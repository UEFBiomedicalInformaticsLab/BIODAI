import load_omics_views
from input_data.input_creator import InputCreator
from input_data.input_data import InputData
from util.printer.printer import Printer


class NoOutcomeInputCreator(InputCreator):
    """Input creator with no outcomes."""

    def __init__(self, nick: str):
        InputCreator.__init__(self, nick=nick)

    def inner_create(self, views_to_load: [str], printer: Printer) -> InputData:
        printer.title_print("Loading views")
        input_dir = self.input_dir()
        printer.print_variable("Input directory", input_dir)
        views = load_omics_views.load_all_views(input_dir, views_to_load, printer=printer)
        views = self._common_preprocessing(views=views, printer=printer)
        return InputData.create_no_outcome(views=views, nick=self.nick())
