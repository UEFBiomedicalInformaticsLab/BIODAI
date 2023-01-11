import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from feature_importance.feature_importance import FeatureImportance
from model.survival_model import CoxModel, cross_validate
from util.dataframes import n_col
from util.distribution.distribution import Distribution, ConcreteDistribution


class FeatureImportanceUnivariateCox(FeatureImportance):
    __verbose: bool

    def __init__(self, verbose: bool = False):
        self.__verbose = verbose

    @staticmethod
    def fold_specific_execution(fold_input) -> float:  # Cannot be private otherwise multiprocessing does not work.
        score = cross_validate(x=fold_input[0], y=fold_input[1], model=CoxModel(step_size=1), n_folds=2)
        score = max(0.0, ((score - 0.5) * 2))
        return score

    def compute(self, x, y, n_proc: int = 1) -> Distribution:
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
            print("Num scores: " + str(len(scores)))
            print("Scores sum: " + str(sum(scores)))
            print("Nonzero scores: " + str(sum([s > 0.0 for s in scores])))
        return ConcreteDistribution(probs=scores)

    def nick(self) -> str:
        return "uniCoxFI"

    def name(self) -> str:
        return "univariate Cox FI"

    def __str__(self) -> str:
        return "univariate Cox feature importance"
