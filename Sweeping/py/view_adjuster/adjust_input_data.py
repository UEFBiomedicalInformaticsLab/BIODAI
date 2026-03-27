from input_data.input_data import InputData
from util.printer.printer import NULL_PRINTER, Printer
from util.str_utils import str_paste
from view_adjuster.view_adjuster_model import DEFAULT_VIEW_ADJUSTER_MODEL, ViewAdjusterModel
from views.views import JustViews


def adjust_input_data(
        input_data: InputData,
        view_adjuster_model: ViewAdjusterModel = DEFAULT_VIEW_ADJUSTER_MODEL,
        printer: Printer = NULL_PRINTER) -> InputData:
    """May return the same object if no adjustment is needed."""
    printer.title_print("Adjusting input data")
    if not input_data.needs_adjustment():
        printer.print("Data does not need any adjustment.")
        return input_data
    else:
        views_to_adjusters = input_data.adjusted_view_def()
        adjusted_views_dict = {}
        for view_name in views_to_adjusters.predictive_view_names_seq():
            table = input_data.view(view_name=view_name)
            adjusting_view_names = views_to_adjusters.adjusters_for_view(view=view_name)
            adjusting_views = JustViews(
                views_dict={adj_name: input_data.view(view_name=adj_name) for adj_name in adjusting_view_names})
            printer.print("Adjusting " + view_name + " with " + str_paste(adjusting_view_names))
            adjusted_views_dict[view_name] = view_adjuster_model.fit_adjust(
                view_to_adjust=table, adjusting_views=adjusting_views)
        stratify_outcome = None
        if input_data.has_stratify_outcome():
            stratify_outcome = input_data.stratify_outcome()
        return InputData.smart_create(
            all_views=adjusted_views_dict,
            outcomes=input_data.outcomes(),
            nick=input_data.nick(),
            stratify_outcome=stratify_outcome,
            covariate_views=input_data.covariate_view_names())
