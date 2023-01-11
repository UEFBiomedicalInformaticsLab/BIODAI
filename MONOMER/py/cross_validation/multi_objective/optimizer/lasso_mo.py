from numpy import ravel
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from ga_components.feature_counts_saver import DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.individual_with_context import IndividualWithContext
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer, \
    MultiObjectiveOptimizerResult
from cross_validation.single_objective.optimizer.single_objective_optimizer import SingleObjectiveOptimizerResult
from individual.sparse_individual import SparseIndividual
from input_data.input_data import InputData
from model.model import SklearnPredictorWrapper
from model.multi_view_model import SVtoMVPredictorWrapper
from util.printer.printer import Printer, NullPrinter


LASSO_STR = "LASSO"


class LassoMO(MultiObjectiveOptimizer):
    __n_objectives: int
    __shrink_factor: float

    __optimizer_type = ConcreteMOOptimizerType(
        uses_inner_models=False, nick="LASSO_MO", name="LASSO multi-view multi-objective optimizer")

    def __init__(self, n_objectives: int, shrink_factor: float = 0.8):
        self.__n_objectives = n_objectives
        self.__shrink_factor = shrink_factor

    @staticmethod
    def optimize_with_c(x: DataFrame, y, c: float, n_jobs=None) -> SingleObjectiveOptimizerResult:
        """ The result cannot have an inner cv fitness. """
        imputer = SimpleImputer()
        logistic_reg = LogisticRegression(
            penalty='l1', solver='liblinear', max_iter=1000, C=c)
        # Not setting n_jobs since it has no effect with liblinear.
        pipe = make_pipeline(imputer, StandardScaler(), logistic_reg)
        pipe.fit(x, y)
        coefs = logistic_reg.coef_
        n_features = len(coefs[0])
        n_classes = len(coefs)
        active_features = []
        for i in range(n_features):
            active = False
            for j in range(n_classes):
                if abs(coefs[j][i]) > 0.0:
                    active = True
            active_features.append(active)
        predictor = SVtoMVPredictorWrapper(SklearnPredictorWrapper(sklearn_predictor=pipe))
        hyperparams = SparseIndividual(seq=active_features)
        hp_manager = DummyHpManager()
        return SingleObjectiveOptimizerResult(predictor, hyperparams, hp_manager)

    def optimize(self, input_data: InputData, printer, n_proc=1,
                 workers_printer: Printer = NullPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver=DummyFeatureCountsSaver()) -> [MultiObjectiveOptimizerResult]:
        """ TODO stratify outcome used as default, allow for other outcomes."""

        collapsed_views = input_data.collapsed_views()
        y = input_data.stratify_outcome_data()
        y = ravel(y)
        res_predictors = []
        res_hyperparams = []
        c = 1.0
        shrink_factor = self.__shrink_factor
        onward = True
        smaller = None
        while onward:
            c_res = self.optimize_with_c(x=collapsed_views, y=y, c=c, n_jobs=n_proc)
            n_features = c_res.hp_manager.n_active_features(c_res.hyperparams)
            if smaller is None or smaller > n_features:
                smaller = n_features
                res_predictors.append([c_res.predictor]*self.__n_objectives)
                res_hyperparams.append(IndividualWithContext(individual=c_res.hyperparams, hp_manager=c_res.hp_manager))
            c = c * shrink_factor
            if n_features < 1:
                onward = False
        return [MultiObjectiveOptimizerResult(
            name="LASSO HoF",
            nick=LASSO_STR,
            predictors=res_predictors,
            hyperparams=res_hyperparams)]

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type
