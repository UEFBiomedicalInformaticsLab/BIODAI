from fitness_adjuster.fitness_adjuster import FitnessAdjuster
from fitness_adjuster.fitness_adjuster_input import FitnessAdjusterInput
from model.model import InputTransformer
from model.regression.regressor import Regressor


class FitnessAdjusterByRegressor(FitnessAdjuster):
    __regressor: Regressor
    __input_transformer: InputTransformer

    def __init__(self, regressor: Regressor, input_transformer: InputTransformer):
        self.__regressor = regressor
        self.__input_transformer = input_transformer

    def adjust_fitness(self, input_data: FitnessAdjusterInput) -> float:
        regressor_x = self.__input_transformer.apply(input_data.to_df())
        inner_res = input_data.original_fitness() - (self.__regressor.predict(regressor_x)[0])
        return max(min(inner_res, 1.0), 0.0)  # Fitnesses are in [0, 1]

    def __str__(self) -> str:
        return "fitness adjuster with regressor\n" + str(self.__regressor) + "\n"
