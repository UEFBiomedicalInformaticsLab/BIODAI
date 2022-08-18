from cross_validation.folds import Folds
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from util.named import Named
from util.printer.printer import Printer


class MultiObjectiveCrossEvaluator(Named):

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):
        """
        Can return an object representing the result of the evaluation.
        :param non_dominated_predictors_with_hyperparams: A MultiObjectiveOptimizerResult for each fold.
        """
        raise NotImplementedError()


class DummyMOCrossEvaluator(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):
        return None


class PrinterMOCrossEvaluator(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):
        print("Non dominated predictors and hyperparameters for evaluation:")
        print(non_dominated_predictors_with_hyperparams)
        return None
