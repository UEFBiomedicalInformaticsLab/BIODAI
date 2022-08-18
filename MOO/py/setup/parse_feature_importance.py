from feature_importance.feature_importance_by_fold import FeatureImportanceByFold, DummyFeatureImportanceByFold
from feature_importance.multi_outcome_feature_importance import MultiOutcomeFeatureImportance
from feature_importance.multi_view_feature_importance import MVFeatureImportanceUniform, MultiViewFeatureImportance, \
    MVFeatureImportanceLasso, MVFeatureImportanceNone
from setup.allowed_names import UNIFORM_NAME, LASSO_NAME, NONE_NAME


def parse_categorical_fi(feature_importance_str: str = NONE_NAME) -> MultiViewFeatureImportance:
    if feature_importance_str == NONE_NAME:
        feature_importance = MVFeatureImportanceNone()
    elif feature_importance_str == UNIFORM_NAME:
        feature_importance = MVFeatureImportanceUniform()
    elif feature_importance_str == LASSO_NAME:
        feature_importance = MVFeatureImportanceLasso()
    else:
        raise ValueError("Unknown feature importance.")
    return feature_importance


def parse_feature_importance_mo(
        categorical_fi_str: str = NONE_NAME) -> MultiOutcomeFeatureImportance:
    categorical_fi = parse_categorical_fi(categorical_fi_str)
    return MultiOutcomeFeatureImportance(class_fi=categorical_fi)


def parse_feature_importance_by_fold(
        categorical_fi_str: str = NONE_NAME) -> FeatureImportanceByFold:
    categorical_fi = parse_categorical_fi(categorical_fi_str)
    multi_outcome = MultiOutcomeFeatureImportance(class_fi=categorical_fi)
    return DummyFeatureImportanceByFold(fi=multi_outcome)
