from input_data.input_data import InputData
from setup.allowed_names import DEFAULT_VIEWS_MV
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from views.adjusted_view_definition import AdjustedViewDef


def plot_input_data(
        input_data: InputData, plots_dir: str, printer: Printer = UNBUFFERED_OUT_PRINTER, skip_huge_views: bool = True):
    from plots import show_views
    from plots.pca_subplots import pca2d
    from plots.plotter.survival_plotter import plot_all_survival_outcomes
    printer.title_print("Plotting views")
    printer.print_variable("Directory for plots", plots_dir)
    plot_all_survival_outcomes(input_data=input_data, directory=plots_dir, printer=printer)
    show_views.plot_views(data=input_data, directory=plots_dir, printer=printer, skip_huge_views=skip_huge_views)
    pca2d(input_data, directory=plots_dir, printer=printer, skip_huge_views=skip_huge_views)


def plot_from_dataset_name(
        dataset_name: str, views_to_use: AdjustedViewDef = DEFAULT_VIEWS_MV, printer: Printer = UNBUFFERED_OUT_PRINTER):
    """Creates every plot from input data, whatever the size."""
    from setup.setup_utils import load_input_data
    load_input_data(
        dataset_name=dataset_name, printer=printer, views_to_use=views_to_use, skip_plotting_huge_views=False)
