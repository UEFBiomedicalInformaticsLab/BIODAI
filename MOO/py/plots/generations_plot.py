import os

import pandas as pd
from matplotlib import pyplot as plt

from consts import GEN_STR, FOLD_RES_DATA_PREFIX, FOLD_RES_DATA_EXTENSION
from util.utils import change_extension, mean_of_dataframes

RECOGNIZED_ENDS = ["min", "avg", "max"]


def csv_to_generations_plot(csv_file: str, plot_file: str):
    df_to_generations_plot(df=pd.read_csv(filepath_or_buffer=csv_file), plot_file=plot_file)


def df_to_generations_plot(df, plot_file: str):
    if GEN_STR in df.columns:
        gen = df[GEN_STR]
        groups = {}
        for col_name in df.columns:
            for end in RECOGNIZED_ENDS:
                if col_name.endswith("_" + end):
                    group_name = col_name[0:len(col_name)-(len(end)+1)]
                    if group_name not in groups:
                        groups[group_name] = {}
                    groups[group_name][end] = df[col_name]
        generations_plot(gen=gen, lines_data=groups, file=plot_file)


def generations_plot(gen: [int], lines_data: dict[str, dict[str, [float]]], file: str):
    group_names = list(lines_data.keys())
    if len(group_names) > 0:
        """lines_data is a dict of dict with values that are the y coordinates of the lines."""
        COLOR_CHARS = ['b', 'r', 'g', 'y', 'm', 'c', 'k']

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Generation")

        lines = []

        for group_i in range(len(group_names)):
            color = COLOR_CHARS[group_i % len(COLOR_CHARS)]
            group_name = group_names[group_i]
            group = lines_data[group_name]
            for line_name in group.keys():
                line_values = group[line_name]
                line = ax1.plot(gen, line_values, str(color)+"-", label=str(line_name)+" "+str(group_name))
                lines = lines + line

        labs = [line.get_label() for line in lines]
        # ax1.legend(lines, labs, bbox_to_anchor=(1.04, 1), loc="upper center")
        ax1.legend(lines, labs, loc="lower center")

        plt.savefig(file, bbox_inches='tight', dpi=600)
        plt.close()


def generations_plots_for_directory(optimizer_dir: str):
    files = []
    for file in os.listdir(optimizer_dir):
        if file.endswith("." + FOLD_RES_DATA_EXTENSION):
            if file.startswith(FOLD_RES_DATA_PREFIX):
                files.append(os.path.join(optimizer_dir, file))
    for f in files:
        plot_file = change_extension(file=f, new_ext="png")
        csv_to_generations_plot(f, plot_file=plot_file)
    dats = []
    if len(files) > 0:
        for f in files:
            dats.append(pd.read_csv(filepath_or_buffer=f))
        dat_tot = mean_of_dataframes(dats)
        df_to_generations_plot(df=dat_tot, plot_file=os.path.join(optimizer_dir, "log.png"))


