from collections.abc import Sequence

from numpy import ravel
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from consts import DEFAULT_SURVIVAL_MODEL
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from ga_components.feature_counts_saver import DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hyperparam_manager.mv_hyperparam_manager.sv_to_mv_hp_manager_wrapper import SvToMvHpManagerWrapper
from hyperparam_manager.sv_hyperparam_manager.dummy_hp_manager import DUMMY_HP_MANAGER
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.single_objective.optimizer.single_objective_optimizer import SingleObjectiveOptimizerResultMV
from individual.confident_predictive_individual import ConfidentPredictiveIndividualSparseMV
from individual.individual_with_context_mv import IndividualWithContextMV
from input_data.input_data import InputData
from model.multi_view.masked_mv_model import MaskedMVPredictor
from model.class_crisp.sv_classifier import SklearnSVClassifierWrapper
from model.multi_view.mv_predictor import SVtoMVPredictorWrapper
from multi_view_utils import filter_by_mask
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer, NullPrinter
from util.utils import PlannedUnreachableCodeError

LASSO_STR = "LASSO"


class LassoMO(MultiObjectiveOptimizer):
    """Does not support feature adjustment yet."""
    __objectives: list[PersonalObjective]
    __shrink_factor: float

    __optimizer_type = ConcreteMOOptimizerType(
        uses_inner_models=False, nick="LASSO_MO", name="LASSO multi-view multi-objective optimizer")

    def __init__(self, objectives: list[PersonalObjective], shrink_factor: float = 0.8):
        self.__objectives = objectives
        self.__shrink_factor = shrink_factor

    @staticmethod
    def optimize_with_c(x: DataFrame, y, c: float, n_jobs=None) -> SingleObjectiveOptimizerResultMV:
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
        predictor = SVtoMVPredictorWrapper(SklearnSVClassifierWrapper(sklearn_predictor=pipe))
        hyperparams = ConfidentPredictiveIndividualSparseMV(fitness=None, seq=active_features)
        hp_manager = SvToMvHpManagerWrapper(
            sv_hp_manager=DUMMY_HP_MANAGER,
            view_name="collapsed",
            predictive_features_num=len(hyperparams))  # Every Individual implements Sized.
        return SingleObjectiveOptimizerResultMV(predictor=predictor, hyperparams=hyperparams, hp_manager=hp_manager)

    def optimize(self, input_data: InputData, printer, n_proc=1,
                 workers_printer: Printer = NullPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver=DummyFeatureCountsSaver()) -> Sequence[MultiObjectiveOptimizerResult]:
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
        survival_model = DEFAULT_SURVIVAL_MODEL
        x = collapsed_views.to_dataframe()
        while onward:
            c_res = self.optimize_with_c(x=x, y=y, c=c, n_jobs=n_proc)
            n_features = c_res.hp_manager.n_predictive_features(c_res.hyperparams)
            if smaller is None or smaller > n_features:
                smaller = n_features
                predictors = []
                for o in self.__objectives:
                    if o.requires_predictions():
                        if o.is_class_based():
                            predictors.append(c_res.predictor)
                        elif o.is_survival():
                            mask = c_res.hp_manager.collapsed_used_features_mask(c_res.hyperparams)
                            x_selected = filter_by_mask(
                                x=x, mask=mask)
                            inner_predictor = survival_model.fit(
                                x=x_selected, y=input_data.outcome(o.outcome_label()).data())
                            predictors.append(MaskedMVPredictor(mask=mask, inner_predictor=inner_predictor))
                        else:
                            raise PlannedUnreachableCodeError()
                    else:
                        predictors.append(None)
                res_predictors.append(predictors)
                res_hyperparams.append(IndividualWithContextMV(
                    individual=c_res.hyperparams, hp_manager=c_res.hp_manager))
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
