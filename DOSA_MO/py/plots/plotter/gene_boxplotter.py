from collections.abc import Sequence

import pandas
import seaborn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from consts import FONT_SIZE
from input_data.input_data import InputData
from input_data.outcome import CategoricalOutcome
from load_omics_views import MRNA_NAME
from plots.plot_consts import DEFAULT_PALETTE
from plots.plotter.plotter import Plotter


class GeneBoxplotter(Plotter):
    __df: DataFrame

    def __init__(self, input_data: InputData, feature_names: Sequence[str], standardize: bool = True):
        mrna_view = input_data.view(MRNA_NAME)
        self.__df = mrna_view[mrna_view.columns.intersection(feature_names).sort_values()]
        if standardize:
            self.__df[self.__df.columns] = StandardScaler().fit_transform(self.__df[self.__df.columns])
        if input_data.n_outcomes() > 0:
            outcome = input_data.outcomes()[0].data()  # Assuming first outcome is the one we need.
        else:
            outcome = DataFrame({"subtype": ["Normal tissue"] * input_data.n_samples()})
            # Assuming if there is no outcome is normal tissue
        self.__df["subtype"] = outcome

    def plot(self, ax: Axes, color=None):
        df = self.__df
        n_genes = len(df.columns)-1
        melted_df = pandas.melt(df,
                                id_vars="subtype",
                                value_vars=df.columns[0:n_genes],
                                var_name="genes",
                                )
        with plt.style.context({'font.size': FONT_SIZE}):
            seaborn.boxplot(x="subtype", y="value", data=melted_df, hue="genes", ax=ax, palette=DEFAULT_PALETTE)
            ax.legend(loc="lower center", ncol=min(n_genes, 8))


def create_gene_box_plotter_with_normal(input_data_normal: InputData, input_data_tumor: InputData,
                                        feature_names: Sequence[str]) -> GeneBoxplotter:
    mrna_view_tumor = input_data_tumor.view(MRNA_NAME)
    mrna_view_normal = input_data_normal.view(MRNA_NAME)
    df_normal = mrna_view_normal[mrna_view_normal.columns.intersection(feature_names).sort_values()]
    df_tumor = mrna_view_tumor[mrna_view_tumor.columns.intersection(feature_names).sort_values()]
    df = df_normal.append(df_tumor)
    outcome_normal = ["Normal tissue"] * input_data_normal.n_samples()
    outcome_tumor = list(input_data_tumor.outcomes()[0].data().iloc[:, 0])
    outcome_df = DataFrame({"subtype": outcome_normal + outcome_tumor})
    outcome = CategoricalOutcome(data=outcome_df, name="subtype")
    input_data = InputData.create_one_outcome(views={MRNA_NAME: df}, outcome=outcome, nick=input_data_tumor.nick())
    return GeneBoxplotter(input_data=input_data, feature_names=feature_names, standardize=False)
