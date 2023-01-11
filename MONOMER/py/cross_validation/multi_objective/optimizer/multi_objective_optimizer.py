from abc import abstractmethod
from collections.abc import Sequence, Iterable

from pandas import DataFrame

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.hall_of_fame import HallOfFame
from hall_of_fame.hofers import Hofers
from individual.individual_with_context import IndividualWithContext
from individual.fit_individual import FitIndividual
from input_data.input_data import InputData
from model.multi_view_model import MVPredictor
from util.named import NickNamed
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.sequence_utils import sequence_to_string
from util.utils import name_value


class MultiObjectiveOptimizerResult(NickNamed):
    """One hall of fame on one fold.
    For each individual there is a list of predictors, one for each objective."""
    __name: str
    __nick: str
    __predictors: Sequence[Sequence[MVPredictor]]
    __hyperparams: Sequence[IndividualWithContext]

    def __init__(self, name: str, nick: str, predictors: Sequence[Sequence[MVPredictor]],
                 hyperparams: Sequence[IndividualWithContext]):
        if len(predictors) != len(hyperparams):
            raise ValueError()
        self.__name = name
        self.__nick = nick
        self.__predictors = predictors
        self.__hyperparams = hyperparams

    def predictors(self) -> Sequence[Sequence[MVPredictor]]:
        return self.__predictors

    def predictors_for_objective(self, objective_num: int) -> list[MVPredictor]:
        return [i[objective_num] for i in self.__predictors]

    def hyperparams(self) -> Sequence[IndividualWithContext]:
        return self.__hyperparams

    def __str__(self):
        res = ""
        res += name_value("Name", self.name()) + "\n"
        res += name_value("Nick", self.nick()) + "\n"
        n_solutions = len(self.__predictors)
        res += "Number of solutions: " + str(n_solutions) + "\n"
        if n_solutions <= 10:
            for i in range(n_solutions):
                res += "hyperparams: " + self.__hyperparams[i].brief_str() + "\n"
                res += "predictor: " + str(self.__predictors[i]) + "\n"
        elif n_solutions <= 20:
            for i in range(n_solutions):
                res += self.__hyperparams[i].brief_str() + "\n"
        return res

    def name(self) -> str:
        return self.__name

    def nick(self) -> str:
        return self.__nick

    def individuals_to_df(self) -> DataFrame:
        """A df with a row for each individual (keeping order) and a column for each feature.
        0 or 1 for absence/presence. All columns are kept."""
        if len(self.__hyperparams) == 0:
            return DataFrame()
        return DataFrame(data=[h.active_features_mask() for h in self.__hyperparams])

    def has_fitnesses(self) -> bool:
        for h in self.__hyperparams:
            if not h.has_fitness():
                return False
        return True

    def fitnesses_to_df(self) -> DataFrame:
        if len(self.__hyperparams) == 0:
            return DataFrame()
        if self.has_fitnesses():
            return DataFrame(data=[h.fitness.as_list() for h in self.__hyperparams])
        else:
            return DataFrame()


class MultiObjectiveOptimizer(NickNamed):

    @abstractmethod
    def optimize(self, input_data: InputData, printer: Printer,
                 n_proc=1,
                 workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> Sequence[MultiObjectiveOptimizerResult]:
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
