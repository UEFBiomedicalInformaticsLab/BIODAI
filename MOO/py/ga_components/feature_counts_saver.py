from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def feature_counts_to_df(feature_counts: [[int]], feature_names: [str], save_interval: int = 1):
    df = pd.DataFrame(feature_counts, columns=feature_names)
    n_rows = len(df)
    df.insert(0, "gen", np.arange(n_rows))
    df = df[[a or b for a, b in zip(df["gen"] % save_interval == 0, df["gen"] == (n_rows-1))]]
    # We always save the last gen.
    df = df.loc[:, (df != 0).any(axis=0)]  # Remove features that are never used.
    return pd.DataFrame(df)


def feature_counts_to_csv(feature_counts: [[int]], feature_names: [str], file: str, save_interval: int = 1):
    df_counts = feature_counts_to_df(feature_counts=feature_counts, feature_names=feature_names,
                                     save_interval=save_interval)
    df_counts.to_csv(file, index=False)


class FeatureCountsSaver(ABC):

    @abstractmethod
    def save(self, feature_counts: [[int]], feature_names: [str]):
        raise NotImplementedError()


class DummyFeatureCountsSaver(FeatureCountsSaver):

    def save(self, feature_counts: [[int]], feature_names: [str]):
        pass


class CsvFeatureCountsSaver(FeatureCountsSaver):
    __file: str
    __save_interval: int

    def __init__(self, file: str, save_interval: int = 1):
        self.__file = file
        self.__save_interval = save_interval

    def save(self, feature_counts: [[int]], feature_names: [str]):
        feature_counts_to_csv(feature_counts=feature_counts, feature_names=feature_names,
                              save_interval=self.__save_interval, file=self.__file)
