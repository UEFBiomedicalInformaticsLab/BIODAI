from collections.abc import Sequence
from typing import Optional

from pandas import DataFrame

from input_data.input_data import InputData
from plots.plotter.survival_plotter import SurvivalPlotter
from plots.subplots import subplots


def survival_subplots_multiple_datasets(
        outcome_dfs: Sequence[DataFrame],
        save_path: str,
        n_cols: Optional[int] = None,
        dataset_names: Optional[Sequence[str]] = None):
    plotters = []
    if dataset_names is None:
        plot_parts = [[] for _ in range(len(outcome_dfs))]
    else:
        plot_parts = [[n] for n in dataset_names]
    for outcome_df, parts in zip(outcome_dfs, plot_parts):
        plotters.append(
            SurvivalPlotter(outcome_data=outcome_df, name_parts=parts))
    subplots(plotters=plotters, save_path=save_path, ncols=n_cols, sharex=False, sharey=False, x_stretch=1.25)


def extract_survival_outcome_data(input_data: InputData) -> DataFrame:
    """Extract the first survival outcome data that is found."""
    for outcome in input_data.outcomes():
        if outcome.is_survival():
            return outcome.data()
    raise ValueError("No survival outcome data found")


def survival_subplots(
        input_data: Sequence[InputData],
        save_path: str,
        n_cols: Optional[int] = None):
    outcome_dfs = []
    dataset_names = []
    for i in input_data:
        outcome_dfs.append(extract_survival_outcome_data(input_data=i))
        dataset_names.append(i.name())
    survival_subplots_multiple_datasets(
        outcome_dfs=outcome_dfs, save_path=save_path, n_cols=n_cols, dataset_names=dataset_names)