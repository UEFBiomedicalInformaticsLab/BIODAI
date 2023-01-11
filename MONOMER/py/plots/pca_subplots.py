from collections.abc import Sequence

from pandas import DataFrame

from input_data.input_data import InputData
from load_omics_views import MRNA_NAME
from plots.plot_consts import PRINCIPAL_COMPONENT1_STR, PRINCIPAL_COMPONENT2_STR
from plots.plotter.pca_plotter import PcaPlotter
from plots.subplots import subplots
from util.utils import IllegalStateError


def outcome_data(input_data: InputData) -> DataFrame:
    for outcome in input_data.outcomes():
        outcome_d = outcome.data()
        if len(outcome_d.columns) == 1:
            if len(set(outcome_d.iloc[:, 0])) <= 20:
                return outcome_d
    raise IllegalStateError()


def pca_subplots(
        input_data: Sequence[InputData],
        save_path: str,
        view_name: str = MRNA_NAME):
    plotters = []
    for i in input_data:
        plotters.append(PcaPlotter(view_data=i.view(view_name=view_name), outcome_data=outcome_data(input_data=i)))
    subplots(plotters=plotters, save_path=save_path, ncols=2,
             x_label=PRINCIPAL_COMPONENT1_STR, y_label=PRINCIPAL_COMPONENT2_STR)
