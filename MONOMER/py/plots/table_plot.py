from collections.abc import Sequence
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from pandas import DataFrame

from plots.plot_utils import smart_save_fig


def plot_table_ax(ax: Axes, df: DataFrame,
                  inner_cells_colour=Optional[Sequence[Sequence]]):
    ax.axis('off')
    ax.axis('tight')
    ax.set_facecolor(color="w")
    tab = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    for (row, col), cell in tab.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        if (row > 0) and col > -1:
            if inner_cells_colour is not None:
                cell.set_facecolor(inner_cells_colour[row-1][col])
    tab.auto_set_column_width(col=list(range(len(df.columns))))


def plot_table_to_file(path: str, df: DataFrame, inner_cells_colour=Optional[Sequence[Sequence]]):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)  # hide axes
    plot_table_ax(ax=ax, df=df, inner_cells_colour=inner_cells_colour)
    fig.tight_layout()
    smart_save_fig(path=path)
