from abc import ABC, abstractmethod

from model.model import Classifier
from multi_view_utils import collapse_views
from util.utils import IllegalStateError


class MVPredictor(ABC):

    # x is a list of views, each being a matrix with samples on rows.
    # Returns a list of anything, each element being an expected output.
    @abstractmethod
    def predict(self, views):
        raise NotImplementedError()

    @abstractmethod
    def score_concordance_index(self, views, y) -> float:
        raise NotImplementedError()


# Abstract class for models able to learn creating a predictor.
class MVModel:

    # x is a list of views, each being a matrix with samples on rows.
    # y is a list of anything, each element being an expected output.
    # Returns a Predictor
    def fit(self, views, y) -> MVPredictor:
        raise NotImplementedError()

    def fit_and_predict(self, views_train, y_train, views_test):
        predictor = self.fit(views=views_train, y=y_train)
        predictions_on_train = predictor.predict(views=views_train)
        predictions_on_test = predictor.predict(views=views_test)
        return predictions_on_train, predictions_on_test


# A model that has tunable hyperparameters.
class MVTunableModel:

    def tune(self, hyperparameters) -> MVModel:
        raise NotImplementedError()


class SVtoMVPredictorWrapper(MVPredictor):

    __inner: Classifier

    def __init__(self, sv_predictor: Classifier):
        self.__inner = sv_predictor

    # x is a list of views, each being a matrix with samples on rows.
    # Returns a list of anything, each element being an expected output.
    def predict(self, views):
        collapsed_views = collapse_views(views=views)
        return self.__inner.predict(x=collapsed_views)

    def score_concordance_index(self, views, y) -> float:
        raise IllegalStateError()

    def __str__(self):
        return "Single view to multi view predictor wrapper with inner predictor " + str(self.__inner)
