from typing import Optional

from feature_importance.feature_importance_by_fold import FeatureImportanceByFold, DummyFeatureImportanceByFold
from feature_importance.multi_outcome_feature_importance import MultiOutcomeFeatureImportance
from feature_importance.multi_view_feature_importance import MVFeatureImportanceUniform, MultiViewFeatureImportance, \
    MVFeatureImportanceLasso, MVFeatureImportanceSoftLasso, MVFeatureImportanceAnova, MVFeatureImportanceNone, \
    MVFeatureImportanceUniCox, MVFeatureImportanceCox
from setup.allowed_names import UNIVARIATE_COX_NAME
from setup.allowed_names import UNIFORM_NAME, LASSO_NAME, SOFT_LASSO_NAME, ANOVA_NAME, NONE_NAME, COX_NAME
from util.printer.printer import Printer, OutPrinter


def parse_categorical_fi(feature_importance_str: str = NONE_NAME) -> MultiViewFeatureImportance:
    if feature_importance_str == NONE_NAME:
        feature_importance = MVFeatureImportanceNone()
    elif feature_importance_str == UNIFORM_NAME:
        feature_importance = MVFeatureImportanceUniform()
    elif feature_importance_str == LASSO_NAME:
        feature_importance = MVFeatureImportanceLasso()
    elif feature_importance_str == SOFT_LASSO_NAME:
        feature_importance = MVFeatureImportanceSoftLasso()
    elif feature_importance_str == ANOVA_NAME:
        feature_importance = MVFeatureImportanceAnova()
    else:
        raise ValueError("Unknown feature importance.")
    return feature_importance


def parse_survival_fi(feature_importance_str: str = NONE_NAME) -> MultiViewFeatureImportance:
    if feature_importance_str == NONE_NAME:
        feature_importance = MVFeatureImportanceNone()
    elif feature_importance_str == UNIFORM_NAME:
        feature_importance = MVFeatureImportanceUniform()
    elif feature_importance_str == UNIVARIATE_COX_NAME:
        feature_importance = MVFeatureImportanceUniCox()
    elif feature_importance_str == COX_NAME:
        feature_importance = MVFeatureImportanceCox()
    else:
        raise ValueError("Unknown feature importance.")
    return feature_importance


def parse_feature_importance_mo(
        categorical_fi_str: str = NONE_NAME,
        survival_fi_str: str = NONE_NAME) -> MultiOutcomeFeatureImportance:
    categorical_fi = parse_categorical_fi(categorical_fi_str)
    survival_fi = parse_survival_fi(survival_fi_str)
    return MultiOutcomeFeatureImportance(class_fi=categorical_fi, survival_fi=survival_fi)


def parse_feature_importance_by_fold(
        categorical_fi_str: str = NONE_NAME,
        survival_fi_str: str = NONE_NAME,
        base_dir: Optional[str] = None,
        printer: Printer = OutPrinter()) -> FeatureImportanceByFold:
    if isinstance(base_dir, str):
        raise NotImplementedError(
            "Loading previous hall of fame for feature importance is not supported in this branch.")
    else:
        categorical_fi = parse_categorical_fi(categorical_fi_str)
        survival_fi = parse_survival_fi(survival_fi_str)
        multi_outcome = MultiOutcomeFeatureImportance(class_fi=categorical_fi, survival_fi=survival_fi)
        return DummyFeatureImportanceByFold(fi=multi_outcome)
