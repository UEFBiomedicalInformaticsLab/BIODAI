import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from numpy import number
from pandas import DataFrame

from feature_importance.feature_importance import FeatureImportance
from input_data.outcome import Outcome
from input_data.outcome_type import OutcomeType
from model.survival.survival_model import LifelinesModel, cross_validate
from univariate_property_computer.parallel_univariate_property_computer import compute_univariate_property_with_workers
from univariate_property_computer.univariate_property_computer import UnivariatePropertyComputer
from util.printer.printer import Printer, UNBUFFERED_OUT_PRINTER
from util.table.table import Table
from util.table.table_utils import n_col
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.randoms import random_seed


class CoxComputer(UnivariatePropertyComputer):

    def outcome_types(self) -> Sequence[OutcomeType]:
        return OutcomeType.survival,

    def inner_compute_property(
            self, feature: Sequence[number], outcome: Outcome, covariates: Optional[Table] = None) -> float:
        score = cross_validate(
            x= DataFrame(feature), y=outcome.data(), model=LifelinesModel(step_size=1), n_folds=2, seed=random_seed())
        score = max(0.0, ((score - 0.5) * 2))
        return score

    def nick(self) -> str:
        return "cox"


class FeatureImportanceUnivariateCox(FeatureImportance):
    __verbose: bool

    def __init__(self, verbose: bool = False):
        self.__verbose = verbose

    def compute(self, x: Table, y: Outcome, n_proc: int = 1, printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        scores = compute_univariate_property_with_workers(
            single_feature_computer=CoxComputer(), data=x, outcome=y,
            n_proc=n_proc, task_name="ANOVA", printer=printer)
        if self.__verbose:
            printer.print("Num scores: " + str(len(scores)))
            printer.print("Scores sum: " + str(sum(scores)))
            printer.print("Nonzero scores: " + str(sum([s > 0.0 for s in scores])))
        return ConcreteDistribution(probs=scores)

    @staticmethod
    def fold_specific_execution(fold_input) -> float:  # Cannot be private otherwise multiprocessing does not work.
        score = cross_validate(
            x=fold_input[0], y=fold_input[1], model=LifelinesModel(step_size=1), n_folds=2, seed=random_seed())
        score = max(0.0, ((score - 0.5) * 2))
        return score

    def compute_df(self, x: DataFrame, y: DataFrame, n_proc: int = 1,
                   printer: Printer = UNBUFFERED_OUT_PRINTER) -> Distribution:
        n_features = n_col(x)
        fold_inputs = [(x.loc[:, [column]], y) for column in x]
        cpu_count = multiprocessing.cpu_count()
        n_workers = min(cpu_count, n_features, n_proc)
        if n_workers <= 1:
            scores = [self.fold_specific_execution(fold_input=fi) for fi in fold_inputs]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as workers_pool:
                scores = workers_pool.map(
                    self.fold_specific_execution, fold_inputs, chunksize=1)
            scores = list(scores)
        if self.__verbose:
            printer.print("Num scores: " + str(len(scores)))
            printer.print("Scores sum: " + str(sum(scores)))
            printer.print("Nonzero scores: " + str(sum([s > 0.0 for s in scores])))
        return ConcreteDistribution(probs=scores)

    def nick(self) -> str:
        return "uniCoxFI"

    def name(self) -> str:
        return "univariate Cox FI"

    def __str__(self) -> str:
        return "univariate Cox feature importance"
