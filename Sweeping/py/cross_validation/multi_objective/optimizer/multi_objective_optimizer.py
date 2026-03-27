from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence, Iterable
from typing import Union

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from individual.individual_with_context import IndividualWithContext
from individual.fit_individual import FitIndividual
from input_data.input_data import InputData
from util.list_like import ListLike
from util.named import NickNamed
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.str_utils import iterable_to_string
from util.select_from_sequence import select_by_indices, select_by_mask


class MultiObjectiveOptimizer(NickNamed):

    @abstractmethod
    def optimize(self, input_data: InputData, printer: Printer,
                 n_proc=1,
                 workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> Sequence[MultiObjectiveOptimizerResult]:
        """Returns a sequence of results, one for each HoF."""
        raise NotImplementedError()

    @abstractmethod
    def optimizer_type(self) -> MOOptimizerType:
        raise NotImplementedError()

    def uses_inner_models(self) -> bool:
        return self.optimizer_type().uses_inner_models()

    def nick(self) -> str:
        return self.optimizer_type().nick()

    def name(self) -> str:
        return self.optimizer_type().name()

    def __str__(self) -> str:
        return str(self.optimizer_type())
    

def features_string(features: Sequence[str], max_len: int = 100) -> str:
    res = ""
    n = len(features)
    if n > 3:
        res += str(n) + " features: "
    res += iterable_to_string(features, max_len=max_len)
    return res


def mo_result_feature_string_by_mask(mask: Union[Sequence[int], ListLike], feature_names: Sequence[str]) -> str:
    try:
        str_list = select_by_mask(data=feature_names, mask=mask)
    except TypeError as e:
        raise TypeError("Original error:\n" + str(e) +
                        "mask:\n" + str(mask) + "\n")
    return features_string(features=str_list)


def mo_result_feature_string_by_indices(indices: Sequence[int], feature_names: Sequence[str]) -> str:
    try:
        str_list = select_by_indices(data=feature_names, indices=indices)
    except TypeError as e:
        raise TypeError("Original error:\n" + str(e) +
                        "indices:\n" + str(indices) + "\n")

    return features_string(features=str_list)


def mo_result_feature_strings(hps: list[IndividualWithContext], feature_names: list[str]) -> list[str]:
    res = []
    for h in hps:
        res.append(mo_result_feature_string_by_mask(mask=h.collapsed_used_features_mask(), feature_names=feature_names))
    return res


def individual_to_line(hp: IndividualWithContext, feature_names: Sequence[str]) -> str:
    res = ""
    if isinstance(hp, FitIndividual):
        if hp.has_fitness():
            res += str(hp.fitness) + " "
    try:
        res += mo_result_feature_string_by_mask(mask=hp.collapsed_used_features_mask(), feature_names=feature_names)
    except ValueError as e:
        raise ValueError("Original error:\n" + str(e) +
                         "Individual:\n" + str(hp) + "\n")
    return res


def hofs_to_results(hofs: Iterable[HallOfFame]) -> Sequence[MultiObjectiveOptimizerResult]:
    return hofers_to_results([h.hofers() for h in hofs])


def hofers_to_results(hofers: Iterable[Hofers]) -> Sequence[MultiObjectiveOptimizerResult]:
    """Takes in input one Hofers object for each hall of fame.
    Returns a MultiObjectiveOptimizerResult for each hall of fame."""
    res = []
    for h in hofers:
        res.append(MultiObjectiveOptimizerResult(
            name=h.name(),
            nick=h.nick(),
            predictors=[h_i.get_predictors() for h_i in h],
            hyperparams=h))
    return res
