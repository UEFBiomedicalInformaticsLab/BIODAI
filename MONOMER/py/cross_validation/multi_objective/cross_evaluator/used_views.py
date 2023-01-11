from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from util.list_math import list_div, vector_mean
from util.printer.printer import Printer
from util.stability_of_distributions import stability_of_distributions


class UsedViewsCrossEval(MultiObjectiveCrossEvaluator):

    @staticmethod
    def __distribution_one_fold(n_views: int, view_belonging: list[int],
                                mo_res: MultiObjectiveOptimizerResult) -> list[float]:
        n_usages_per_view = [0] * n_views
        for s in mo_res.hyperparams():
            mask = s.active_features_mask()
            for i in range(len(mask)):
                if mask[i]:
                    n_usages_per_view[view_belonging[i]] += 1
        tot_usages = sum(n_usages_per_view)
        res = list_div(li=n_usages_per_view, d=tot_usages)
        return res

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: list[MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):
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
        printer.print("Prevalence of features selected in each view")
        for i in range(n_views):
            printer.print_variable(view_names[i], mean[i])
        printer.print("Distribution of views in each fold")
        printer.print(distributions)
        views_stability = stability_of_distributions(distributions)
        printer.print_variable("Stability of views", views_stability)

    def name(self) -> str:
        return "Usage of views"
