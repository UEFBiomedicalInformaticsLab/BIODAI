from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pandas import DataFrame

from input_data.model_ready_input_data import ModelReadyInputData
from model.sv_model import SVPredictor, Predictor
from util.feature_space_lifter import FeatureSpaceLifterMV
from views.views import Views


class MVPredictor(Predictor, ABC):

    @abstractmethod
    def predict(self, views: Views) -> Sequence:
        """Returns a list of anything, each element being an expected output."""
        raise NotImplementedError()

    @abstractmethod
    def predict_crisp(self, views: Views) -> Sequence:
        """Returns a list of class labels, each element being an expected output.
        Throws an exception if the predictor is not of the correct type."""
        raise NotImplementedError()

    @abstractmethod
    def score_concordance_index(self, data: ModelReadyInputData) -> float:
        raise NotImplementedError()

    def score_concordance_index_raw(self, views: Views, y) -> float:
        return self.score_concordance_index(data=ModelReadyInputData.create_raw(x=views, y=y))

    @abstractmethod
    def predict_survival_probabilities(self, views: Views, times: Sequence[float]) -> DataFrame:
        """Return probabilities that event has not happened up to the passed times.
        It returns times on the rows and individuals on the columns."""
        raise NotImplementedError()

    def downlift(self, lifter: FeatureSpaceLifterMV) -> MVPredictor:
        return DownliftedMVPredictor(inner_predictor=self, lifter=lifter)


class SVtoMVPredictorWrapper(MVPredictor):
    """Collapses the views and uses the inner single view predictor."""
    __inner: SVPredictor

    def __init__(self, sv_predictor: SVPredictor):
        self.__inner = sv_predictor

    def predict(self, views: Views) -> Sequence:
        return self.__inner.predict(x=views.to_dataframe())

    def predict_crisp(self, views: Views) -> Sequence:
        return self.__inner.predict_crisp(x=views.to_dataframe())

    def score_concordance_index_raw(self, views: Views, y) -> float:
        return self.__inner.score_concordance_index(x_test=views.to_dataframe(), y_test=y)

    def score_concordance_index(self, data: ModelReadyInputData) -> float:
        return self.score_concordance_index_raw(views=data.views(), y=data.outcome_data())

    def predict_survival_probabilities(self, views: Views, times: Sequence[float]) -> DataFrame:
        return self.__inner.predict_survival_probabilities(x=views.to_dataframe(), times=times)

    def __str__(self) -> str:
        return "Single view to multi view predictor wrapper with inner predictor " + str(self.__inner)


class DownliftedMVPredictor(MVPredictor):
    """Wrapper for a predictor for a smaller feature space making it able to operate in a bigger feature space."""
    __inner: MVPredictor
    __lifter: FeatureSpaceLifterMV

    def __init__(self, inner_predictor: MVPredictor, lifter: FeatureSpaceLifterMV):
        self.__inner = inner_predictor
        self.__lifter = lifter

    def predict(self, views: Views) -> Sequence:
        if not isinstance(views, Views):
            raise ValueError("Passed object is not of type Views.\n" + "Passed object:\n" + str(views) + "\n")
        return self.__inner.predict(views=self.__lifter.uplift_views(views))

    def predict_crisp(self, views: Views) -> Sequence:
        if not isinstance(views, Views):
            raise ValueError("Passed object is not of type Views.\n" + "Passed object:\n" + str(views) + "\n")
        return self.__inner.predict_crisp(views=self.__lifter.uplift_views(views))

    def score_concordance_index_raw(self, views: Views, y) -> float:
        return self.__inner.score_concordance_index_raw(views=self.__lifter.uplift_views(views), y=y)

    def score_concordance_index(self, data: ModelReadyInputData) -> float:
        return self.score_concordance_index_raw(views=data.views(), y=data.outcome_data())

    def predict_survival_probabilities(self, views: Views, times: Sequence[float]) -> DataFrame:
        return self.__inner.predict_survival_probabilities(views=self.__lifter.uplift_views(views), times=times)

    def __str__(self) -> str:
        return "Downlifted multi view predictor with inner predictor " + str(self.__inner)
