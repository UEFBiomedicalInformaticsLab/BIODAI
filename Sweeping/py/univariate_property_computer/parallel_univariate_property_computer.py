from typing import Optional, Any

from input_data.outcome import Outcome
from univariate_property_computer.univariate_property_computer import UnivariatePropertyComputer
from util.parallel.parallel_computer import ParallelComputer
from util.parallel.parallel_runner import ParallelRunner
from util.printer.printer import UNBUFFERED_OUT_PRINTER, Printer
from util.table.table import Table


class UnivariatePropertyCommonData:
    table: Table
    outcome: Outcome
    covariates: Optional[Table]

    def __init__(self, table: Table, outcome: Outcome, covariates: Optional[Table]):
        self.table = table
        self.outcome = outcome
        self.covariates = covariates


class UnivariatePropertyParallelComputer(ParallelComputer):
    __inner: UnivariatePropertyComputer

    def __init__(self, inner: UnivariatePropertyComputer):
        self.__inner = inner

    def compute(self, common_data: Any, job: Any) -> Any:
        assert isinstance(common_data, UnivariatePropertyCommonData)
        assert isinstance(job, int)
        feature = common_data.table.select_cols(
            selected=[job]).impute().standardize().to_numpy().flatten().tolist()
        return self.__inner.compute_property(
            feature=feature, outcome=common_data.outcome, covariates=common_data.covariates)

    def __str__(self) -> str:
        return "Univariate property parallel computer with inner computer " + str(self.__inner)


def compute_univariate_property_with_workers(
        single_feature_computer: UnivariatePropertyComputer,
        data: Table, outcome: Outcome, n_proc: int = 1, task_name: str = "Computing univariate property",
        covariates: Optional[Table] = None,
        printer: Printer = UNBUFFERED_OUT_PRINTER,
        minutes_of_quiet: int = 1) -> list:
    """Covariates and data are imputed and standardized."""
    if covariates is not None:
        covariates = covariates.impute().standardize().serialize()
    computer = UnivariatePropertyParallelComputer(inner=single_feature_computer)
    runner = ParallelRunner(computer=computer, task_name = task_name)
    worker_data = UnivariatePropertyCommonData(table=data.serialize(), outcome=outcome, covariates=covariates)
    return runner.run_correctly(
        worker_data=worker_data, jobs=range(data.n_col()), n_proc=n_proc,
        printer=printer, minutes_of_quiet=minutes_of_quiet)
