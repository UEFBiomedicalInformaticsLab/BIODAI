from numpy import ravel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizerResult
from cross_validation.single_objective.optimizer.single_objective_optimizer import SingleObjectiveOptimizerResult, \
    SOOptimizer
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.sparse_individual import SparseIndividual
from model.model import SklearnPredictorWrapper
from model.multi_view_model import SVtoMVPredictorWrapper
from multi_view_utils import collapse_views


class LassoSingleObjectiveOptimizer(SOOptimizer):

    def __init__(self):
        pass

    def optimize(self, views, y) -> MultiObjectiveOptimizerResult:
        # Could add pre-filtering
        y = ravel(y)
        collapsed_views = collapse_views(views=views)
        imputer = SimpleImputer()
        logistic_reg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        pipe = make_pipeline(imputer, StandardScaler(), logistic_reg)
        pipe.fit(collapsed_views, y)
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

    def name(self) -> str:
        return "Lasso multi-view single-objective optimizer"

    def nick(self) -> str:
        return "lasso"
