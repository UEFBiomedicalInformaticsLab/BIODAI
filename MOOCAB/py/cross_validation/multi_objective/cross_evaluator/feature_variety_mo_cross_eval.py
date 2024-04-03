import statistics

from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from input_data.input_data import InputData

from util.printer.printer import Printer
from util.sparse_bool_list_by_set import jaccard_score
from util.math.summer import KahanSummer
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry
from validation_registry.allowed_property_names import MEAN_JACCARD_NAME, FOLDS_MEAN_JACCARD_NAME


class FeatureVarietyMOCrossEval(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: list[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        mean_jacs = []
        for i in range(folds.n_folds()):
            non_dominated_predictors_with_hyperparams_i = non_dominated_predictors_with_hyperparams[i]
            hyperparams = non_dominated_predictors_with_hyperparams_i.hyperparams()
            n_hp = len(hyperparams)
            mean_jac = 1.0  # If there are zero or one individuals then the concordance of the features is total
            if n_hp > 1:
                summation = KahanSummer()
                for j in range(n_hp):
                    for k in range(j+1, n_hp):
                        summation.add(jaccard_score(y_true=hyperparams[j].active_features_mask(),
                                                    y_pred=hyperparams[k].active_features_mask(),
                                                    zero_division=1.0))
                        # We consider to have perfect concordance when both individuals have no features.
                        # Converting to numpy is necessary to avoid sklearn complaining
                        # "TypeError: 'ListLikeIterator' object is not iterable"
                denominator = ((n_hp*n_hp)-n_hp)/2
                mean_jac = summation.get_sum()/denominator
            mean_jacs.append(mean_jac)
        printer.print("Mean Jaccard all vs others for each fold")
        printer.print(str(mean_jacs))
        mean_of_folds = statistics.mean(mean_jacs)
        printer.print_variable("Mean of folds", mean_of_folds)
        hof_registry.set_property(name=MEAN_JACCARD_NAME, value=mean_of_folds)
        hof_registry.set_property(name=FOLDS_MEAN_JACCARD_NAME, value=mean_jacs)
        return mean_jacs

    def name(self) -> str:
        return "multi-objective feature variety cross-evaluator"
