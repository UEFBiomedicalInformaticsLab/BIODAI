import os

import pandas as pd
from matplotlib import pyplot as plt

from consts import GEN_STR, FOLD_RES_DATA_EXTENSION, FOLD_RES_DATA_PREFIX, GENERATION_FEATURES_STRING, \
    EXPLORED_FEATURES_STRING
from util.utils import mean_of_dataframes


def csv_to_explored_features_plot(csv_file: str, plot_file: str):
    df_to_explored_features_plot(df=pd.read_csv(filepath_or_buffer=csv_file), plot_file=plot_file)


def df_to_explored_features_plot(df, plot_file: str):
    if GEN_STR in df.columns:
        gen = df[GEN_STR]
        if GENERATION_FEATURES_STRING in df.columns:
            gen_feats = df[GENERATION_FEATURES_STRING]
        else:
            gen_feats = None
        if EXPLORED_FEATURES_STRING in df.columns:
            explored = df[EXPLORED_FEATURES_STRING]
        else:
            explored = None
        explored_features_plot(gen=gen, gen_feats=gen_feats, explored=explored, file=plot_file)


def explored_features_plot(gen, file: str, gen_feats=None, explored=None):
    n_gf = 0
    n_exp = 0
    if gen_feats is not None:
        n_gf = len(gen_feats)
    if explored is not None:
        n_exp = len(explored)
    if n_gf > 0 or n_exp > 0:
        fig, ax = plt.subplots()
        ax.set_xlabel("Generation")
        lines = []
        if n_gf > 0:
            lines = lines + ax.plot(gen, gen_feats, "b-", label="generation features")
        if n_exp > 0:
            lines = lines + ax.plot(gen, explored, "r-", label="total features")

        labs = [line.get_label() for line in lines]
        ax.legend(lines, labs, loc="center right")

        plt.savefig(file, bbox_inches='tight', dpi=600)
        plt.close()


def explored_features_plots_for_directory(optimizer_dir: str):
    files = []
    for file in os.listdir(optimizer_dir):
        if file.endswith("." + FOLD_RES_DATA_EXTENSION):
            if file.startswith(FOLD_RES_DATA_PREFIX):
                files.append(os.path.join(optimizer_dir, file))
    for f in files:
        base = os.path.splitext(f)
        plot_file = base[0] + "_features" + ".png"
        csv_to_explored_features_plot(f, plot_file=plot_file)
    dats = []
    if len(files) > 0:
        for f in files:
            dats.append(pd.read_csv(filepath_or_buffer=f))
        dat_tot = mean_of_dataframes(dats)
        df_to_explored_features_plot(df=dat_tot, plot_file=os.path.join(optimizer_dir, "log_features.png"))
