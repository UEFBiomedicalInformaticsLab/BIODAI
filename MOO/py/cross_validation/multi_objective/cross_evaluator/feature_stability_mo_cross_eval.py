from __future__ import annotations

from typing import NamedTuple, Sequence
from scipy.stats import spearmanr

from cross_validation.multi_objective.cross_evaluator.multi_objective_cross_evaluator import \
    MultiObjectiveCrossEvaluator
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from input_data.input_data import InputData
from util.components_transform import ComponentsTransform, IdentityComponentsTransform
from util.list_math import list_div, mean_all_vs_others
from crowding_dist import average_individual
from util.printer.printer import Printer
from util.sparse_bool_list_by_set import union_sparse, top_k_mask
from util.stability_of_distributions import stability_of_distributions
from util.summer import KahanSummer
from util.utils import same_len


def stability_by_weights_from_counts(
        counts: list[Sequence[int | float]],
        components_transform: ComponentsTransform = IdentityComponentsTransform()) -> float:
    """Works as long as the input list contains weighted lists, also of floats."""
    if not same_len(counts):
        raise ValueError()
    distributions = []
    for c in counts:
        c_transformed = components_transform.apply(c)
        c_sum = KahanSummer.sum(c_transformed)

        # Weights must sum to 1.
        if c_sum <= 0.0:
            le = len(c_transformed)
            weights = [1.0 / le] * le
        else:
            weights = list_div(li=c_transformed, d=c_sum)
        distributions.append(weights)
    return stability_of_distributions(distributions)


def stability_by_unions_from_counts(counts: list[Sequence[int]],
                                    components_transform=IdentityComponentsTransform()) -> float:
    if not same_len(counts):
        raise ValueError()
    distributions = []
    for c in counts:
        c_transformed = components_transform.apply(c)
        c_transformed = [bool(x) for x in c_transformed]
        c_sum = sum(c_transformed)

        # Weights must sum to 1.
        if c_sum <= 0.0:
            le = len(c_transformed)
            weights = [1.0 / le] * le
        else:
            weights = list_div(li=c_transformed, d=c_sum)
        distributions.append(weights)
    return stability_of_distributions(distributions)


def stability_by_weights(non_dominated_solutions: list[MultiObjectiveOptimizerResult]) -> float:
    counts = []
    for optimizer_result in non_dominated_solutions:  # Loop on folds
        counts.append(average_individual(optimizer_result.hyperparams()))
    return stability_by_weights_from_counts(counts)


def stability_by_unions_from_individuals(non_dominated_solutions: list[MultiObjectiveOptimizerResult]) -> float:
    """Stability of the union of all the solutions. For each fold a distribution is computed with a weight w for
    each feature that is present and 0 for each feature that is not present. w is 1 / num features present in fold."""
    fold_feature_weights = []

    for optimizer_result in non_dominated_solutions:  # Loop on folds
        fold_union = union_sparse(optimizer_result.hyperparams())
        fold_union_sum = sum(fold_union)

        # Weights must sum to 1.
        if fold_union_sum <= 0.0:
            le = len(fold_union)
            weights = [1.0 / le] * le
        else:
            weights = list_div(li=fold_union, d=fold_union_sum)
        fold_feature_weights.append(weights)

    return stability_of_distributions(fold_feature_weights)


def stability_by_top_k(non_dominated_solutions: list[MultiObjectiveOptimizerResult], k: int) -> float:

    if k <= 0:
        return 1

    fold_feature_weights = []

    for optimizer_result in non_dominated_solutions:  # Loop on folds
        avg_ind = average_individual(optimizer_result.hyperparams())
        top_mask = top_k_mask(avg_ind, k)
        weights = list_div(li=top_mask, d=k)
        fold_feature_weights.append(weights)

    return stability_of_distributions(fold_feature_weights)


def stability_by_spearman(non_dominated_solutions: list[MultiObjectiveOptimizerResult]) -> float:
    fold_feature_weights = []

    for optimizer_result in non_dominated_solutions:  # Loop on folds
        avg_ind = average_individual(optimizer_result.hyperparams())
        avg_ind_sum = KahanSummer.sum(avg_ind)

        # Weights must sum to 1.
        if avg_ind_sum <= 0.0:
            le = len(avg_ind)
            weights = [1.0 / le] * le
        else:
            weights = list_div(li=avg_ind, d=avg_ind_sum)
        fold_feature_weights.append(weights)

    if len(fold_feature_weights) < 2:
        stability = 1.0
    else:
        stability = mean_all_vs_others(elems=fold_feature_weights, measure_function=lambda a, b: spearmanr(a, b)[0])
    return stability


class FeatureStabilityRes(NamedTuple):
    stab_by_weights: float
    stab_by_unions: float
    stab_by_spearman: float
    stab_by_top: float


class FeatureStabilityMOCrossEval(MultiObjectiveCrossEvaluator):

    def evaluate(self, input_data: InputData, folds,
                 non_dominated_predictors_with_hyperparams: [MultiObjectiveOptimizerResult],
                 printer: Printer,
                 optimizer_nick="unknown_optimizer"):

        fold_representatives = []
        fold_feature_weights = []

        for optimizer_result in non_dominated_predictors_with_hyperparams:  # Loop on folds
            avg_ind = average_individual(optimizer_result.hyperparams())
            fold_representatives.append(avg_ind)
            avg_ind_sum = KahanSummer.sum(avg_ind)

            # Weights must sum to 1.
            if avg_ind_sum <= 0.0:
                le = len(avg_ind)
                weights = [1.0/le] * le
            else:
                weights = list_div(li=avg_ind, d=avg_ind_sum)
            fold_feature_weights.append(weights)

        stab_by_weights = stability_by_weights(non_dominated_predictors_with_hyperparams)
        printer.print("Stability of features by weights: " + str(stab_by_weights))

        stab_by_unions = stability_by_unions_from_individuals(non_dominated_predictors_with_hyperparams)
        printer.print("Stability of features by unions: " + str(stab_by_unions))

        stab_by_spearman = stability_by_spearman(non_dominated_predictors_with_hyperparams)
        printer.print("Stability of features by spearman: " + str(stab_by_spearman))

        stab_by_top = stability_by_top_k(non_dominated_predictors_with_hyperparams, k=50)
        printer.print("Stability of features by top 50: " + str(stab_by_top))

        return FeatureStabilityRes(
            stab_by_weights=stab_by_weights, stab_by_unions=stab_by_unions, stab_by_spearman=stab_by_spearman,
            stab_by_top=stab_by_top)

    def name(self) -> str:
        return "multi-objective feature stability cross-cross_evaluator"
