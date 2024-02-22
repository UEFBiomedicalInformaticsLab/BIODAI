from collections.abc import Sequence
from typing import Optional

import pandas as pd
from pandas import DataFrame

from fitness_adjuster.fitness_adjuster import FitnessAdjuster
from fitness_adjuster.fitness_adjuster_by_regressor import FitnessAdjusterByRegressor
from fitness_adjuster.fitness_adjuster_input import FitnessAdjusterInput, BOOTSTRAP_MEAN_STR, ORIGINAL_FITNESS_STR, \
    NUM_FEATURES_STR, STANDARD_DEVIATION_STR
from model.model import InputTransformer
from model.regression.regressor import RegressorModel
from util.math.list_math import list_abs
from util.named import NickNamed
from util.sequence_utils import sequence_to_string

BOOSTRAP_MEAN_DIFF_STR = "bootstrap_mean_diff"


class FitnessAdjusterInputTransformer(InputTransformer):
    __use_bootstrap_mean: bool
    __use_n_features: bool
    __use_standard_deviation: bool
    __use_original_fitness: bool

    def __init__(self,
                 use_bootstrap_mean: bool = False,
                 use_n_features: bool = True,
                 use_standard_deviation: bool = True,
                 use_original_fitness: bool = True):
        self.__use_bootstrap_mean = use_bootstrap_mean
        self.__use_n_features = use_n_features
        self.__use_standard_deviation = use_standard_deviation
        self.__use_original_fitness = use_original_fitness

    def apply(self, x: DataFrame) -> DataFrame:
        if self.__use_bootstrap_mean:
            x[BOOSTRAP_MEAN_DIFF_STR] = list_abs(x[ORIGINAL_FITNESS_STR] - x[BOOTSTRAP_MEAN_STR])
        x = x.drop(columns=[BOOTSTRAP_MEAN_STR])
        if not self.__use_n_features:
            x = x.drop(columns=[NUM_FEATURES_STR])
        if not self.__use_standard_deviation:
            x = x.drop(columns=[STANDARD_DEVIATION_STR])
        if not self.__use_original_fitness:
            x = x.drop(columns=[ORIGINAL_FITNESS_STR])
        return x

    def __str__(self) -> str:
        keeping = []
        dropping = []
        if self.__use_original_fitness:
            keeping.append("original fitness")
        else:
            dropping.append("original fitness")
        if self.__use_standard_deviation:
            keeping.append("standard deviation")
        else:
            dropping.append("standard deviation")
        if self.__use_n_features:
            keeping.append("number of features")
        else:
            dropping.append("number of features")
        if self.__use_bootstrap_mean:
            keeping.append("bootstrap mean")
        else:
            dropping.append("bootstrap mean")
        res = []
        if len(keeping) > 0:
            res.append("keeping " + sequence_to_string(li=keeping, brackets=False))
        if len(dropping) > 0:
            res.append("dropping " + sequence_to_string(li=dropping, brackets=False))
        return sequence_to_string(li=res, brackets=False, separator=" and")


class FitnessAdjusterLearner(NickNamed):
    __model: RegressorModel
    __input_transformer: InputTransformer

    def __init__(self, model: RegressorModel, use_bootstrap_mean: bool = False):
        self.__model = model
        self.__input_transformer = FitnessAdjusterInputTransformer(use_bootstrap_mean=use_bootstrap_mean)

    def fit(self, inputs: Sequence[FitnessAdjusterInput], test_fitness: Sequence[float],
            sample_weight: Optional[Sequence[float]] = None) -> FitnessAdjuster:
        """The learned regressor predicts the overfitting, predicts higher when there is more overfitting.
        The returned fitness adjuster uses the predicted overfitting to adjust the fitness,
        so the fitness adjuster predicts fitness values."""
        x = self.__create_x(inputs=inputs)
        y = [i.original_fitness() - tf for i, tf in zip(inputs, test_fitness)]
        regressor = self.__model.fit(x=x, y=y, sample_weight=sample_weight)
        return FitnessAdjusterByRegressor(regressor=regressor, input_transformer=self.__input_transformer)

    def __create_x(self, inputs: Sequence[FitnessAdjusterInput]) -> DataFrame:
        df = pd.concat([i.to_df() for i in inputs])
        return self.__input_transformer.apply(df)

    def nick(self) -> str:
        return "adj_" + self.__model.nick()

    def name(self) -> str:
        return ("adjuster learner with " + self.__model.name() + " and " +
                "input transformer " + self.__input_transformer.name())

    def __str__(self) -> str:
        return ("fitness adjuster learner with " + str(self.__model) + " model and\n" +
                "input transformer " + str(self.__input_transformer) + "\n")
