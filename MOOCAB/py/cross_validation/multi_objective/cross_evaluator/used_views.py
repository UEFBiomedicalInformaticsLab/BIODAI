from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from util.distribution.distribution import ConcreteDistribution, Distribution
from util.math.list_math import vector_mean
from util.printer.printer import Printer
from util.sequence_utils import sequence_to_string
from util.stability_of_distributions import stability_of_distributions
from validation_registry.validation_registry import ValidationRegistry, MemoryValidationRegistry
from validation_registry.allowed_property_names import STABILITY_OF_VIEWS_NAME, DISTRIBUTION_OF_VIEWS_NAME


class UsedViewsCrossEval(MultiObjectiveCrossEvaluator):

    @staticmethod
    def __distribution_one_fold(n_views: int, view_belonging: list[int],
                                mo_res: MultiObjectiveOptimizerResult) -> Distribution:
        n_usages_per_view = [0] * n_views
        for s in mo_res.hyperparams():
            mask = s.active_features_mask()
            for i in range(len(mask)):
                if mask[i]:
                    n_usages_per_view[view_belonging[i]] += 1
        res = ConcreteDistribution(probs=n_usages_per_view)
        return res

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: list[MultiObjectiveOptimizerResult], printer: Printer,
                 optimizer_nick="unknown_optimizer", hof_registry: ValidationRegistry = MemoryValidationRegistry()):
        n_views = input_data.n_views()
        view_names = input_data.view_names()
        view_belonging = []
        for v in range(n_views):
            view_belonging.extend([v]*len(input_data.view(view_names[v]).columns))
        distributions = []
        for mo_res in non_dominated_predictors_with_hyperparams:
            distributions.append(self.__distribution_one_fold(
                n_views=n_views, view_belonging=view_belonging, mo_res=mo_res))
        mean = vector_mean(distributions)
        mean_dict = {}
        printer.print("Prevalence of features selected in each view")
        for i in range(n_views):
            printer.print_variable(view_names[i], mean[i])
            mean_dict[view_names[i]] = mean[i]
        printer.print("Distribution of views in each fold")
        printer.print(sequence_to_string(distributions))
        views_stability = stability_of_distributions(distributions)
        printer.print_variable("Stability of views", views_stability)

        hof_registry.set_property(name=STABILITY_OF_VIEWS_NAME, value=views_stability)
        hof_registry.set_property(name=DISTRIBUTION_OF_VIEWS_NAME, value=mean_dict)

    def name(self) -> str:
        return "Usage of views"
