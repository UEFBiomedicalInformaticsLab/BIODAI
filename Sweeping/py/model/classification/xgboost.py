from xgboost import XGBClassifier

from model.coef_extractor import SklearnCoefExtractor, OffCoefExtractor
from model.model_label_wrapper import ModelLabelWrapper
from model.model_with_coef import SKLearnModelFactoryWithExtractor
from model.pipe_wrapper import PipeWrapper

XGBOOST_NAME = "XGBoost"
XGBOOST_NICK = "xgb"


class XGBoostFactory(SKLearnModelFactoryWithExtractor):
    __impute: bool
    __l2: float
    """L2 regularization."""
    __n_estimators: int
    __min_child_weight: float

    def __init__(self, impute: bool = True, l2: float = 1.0, n_estimators: int = 20, min_child_weight: float = 8):
        """XGBoost handles missing values. Still, it can be useful to impute in some cases.
        In the xgboost library, the default l2 is 1.0, min_child_weight is 1, and the default n_estimators is 100."""
        self.__impute = impute
        self.__l2 = l2
        self.__n_estimators = n_estimators
        self.__min_child_weight = min_child_weight

    def create(self):
        classifier = ModelLabelWrapper(
            sklearn_model=XGBClassifier(
                reg_lambda=self.__l2, n_estimators=self.__n_estimators, min_child_weight=self.__min_child_weight,
                n_jobs=1))
        if self.__impute:
            return PipeWrapper(
                sklearn_model=classifier,
                model_name=XGBOOST_NAME,
                supports_weights=True,
                scale=False)  # Scaling is not needed with xgboost.
        else:
            return classifier

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return XGBOOST_NICK

    def name(self) -> str:
        return XGBOOST_NAME
