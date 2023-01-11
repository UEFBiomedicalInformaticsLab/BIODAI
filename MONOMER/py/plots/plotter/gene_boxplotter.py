from collections.abc import Sequence

import pandas
import seaborn
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from consts import FONT_SIZE
from input_data.input_data import InputData
from load_omics_views import MRNA_NAME
from plots.plot_consts import DEFAULT_PALETTE
from plots.plotter.plotter import Plotter


class GeneBoxplotter(Plotter):
    __df: DataFrame

    def __init__(self, input_data: InputData, feature_names: Sequence[str]):
        mrna_view = input_data.view(MRNA_NAME)
        self.__df = mrna_view[mrna_view.columns.intersection(feature_names).sort_values()]
        self.__df[self.__df.columns] = StandardScaler().fit_transform(self.__df[self.__df.columns])
        outcome = input_data.outcomes()[0].data()  # Assuming first outcome is the one we need.
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
