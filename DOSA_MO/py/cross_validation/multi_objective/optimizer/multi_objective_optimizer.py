from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence, Iterable

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from individual.individual_with_context import IndividualWithContext
from individual.fit_individual import FitIndividual
from input_data.input_data import InputData
from util.named import NickNamed
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.sequence_utils import sequence_to_string


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


def mo_result_feature_string(mask, feature_names: list[str]) -> str:
    """mask can also be an Iterable of positions"""
    str_list = feature_names[list(mask)]
    res = ""
    n = len(str_list)
    if n > 3:
        res += str(n) + " features: "
    res += sequence_to_string(str_list)
    return res


def mo_result_feature_strings(hps: list[IndividualWithContext], feature_names: list[str]) -> list[str]:
    res = []
    for h in hps:
        res.append(mo_result_feature_string(mask=h, feature_names=feature_names))
    return res


def individual_to_line(hp: IndividualWithContext, feature_names: list[str]) -> str:
    res = ""
    if isinstance(hp, FitIndividual):
        if hp.has_fitness():
            res += str(hp.fitness) + " "
    res += mo_result_feature_string(mask=hp, feature_names=feature_names)
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
