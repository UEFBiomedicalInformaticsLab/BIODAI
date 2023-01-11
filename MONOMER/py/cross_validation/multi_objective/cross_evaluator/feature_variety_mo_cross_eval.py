import statistics

from cross_validation.folds import Folds
from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from input_data.input_data import InputData

from util.printer.printer import Printer
from util.sparse_bool_list_by_set import jaccard_score
from util.summer import KahanSummer


class FeatureVarietyMOCrossEval(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds: Folds,
                 non_dominated_predictors_with_hyperparams: list[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer"):
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
        printer.print_variable("Mean of folds", statistics.mean(mean_jacs))
        return mean_jacs

    def name(self) -> str:
        return "multi-objective feature variety cross-cross_evaluator"
