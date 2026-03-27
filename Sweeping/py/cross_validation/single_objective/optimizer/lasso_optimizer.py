from numpy import ravel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_validation.single_objective.optimizer.single_objective_optimizer import SOOptimizer, \
    SingleObjectiveOptimizerResultMV
from hyperparam_manager.mv_hyperparam_manager.mask_mv_hp_manager import MaskMvHpManager
from individual.confident_predictive_individual import ConfidentPredictiveIndividualSparseMV
from individual.individual_with_context_mv import IndividualWithContextMV
from input_data.evaluation_ready_input_data import EvaluationReadyInputData
from model.class_crisp.sv_classifier import SklearnSVClassifierWrapper
from model.multi_view.mv_predictor import SVtoMVPredictorWrapper


class LassoSingleObjectiveOptimizer(SOOptimizer):

    def __init__(self):
        pass

    def optimize(self, input_data: EvaluationReadyInputData) -> SingleObjectiveOptimizerResultMV:
        """Assumes that input data has a single classification outcome."""
        # Could add pre-filtering
        y = ravel(input_data.the_outcome().data())
        collapsed_views = input_data.collapsed_views().to_dataframe()
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
        predictor = SVtoMVPredictorWrapper(SklearnSVClassifierWrapper(sklearn_predictor=pipe))
        hp_manager = MaskMvHpManager.create_from_input_data(input_data=input_data)
        inner_individual = ConfidentPredictiveIndividualSparseMV(fitness=None, seq=active_features)
        hyperparams = IndividualWithContextMV(individual=inner_individual, hp_manager=hp_manager)
        return SingleObjectiveOptimizerResultMV(predictor=predictor, hyperparams=hyperparams, hp_manager=hp_manager)

    def name(self) -> str:
        return "Lasso multi-view single-objective optimizer"

    def nick(self) -> str:
        return "lasso"
