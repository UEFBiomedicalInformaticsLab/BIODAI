from abc import ABC, abstractmethod
from typing import Optional

from input_data.model_ready_input_data import ModelReadyInputData
from model.multi_view.multi_view_model import MVModel
from model.multi_view.mv_predictor import MVPredictor

from model.sv_model import SampleWeight
from util.utils import IllegalStateError


class CrispMVPredictor(MVPredictor, ABC):

    def score_concordance_index(self, test_data: ModelReadyInputData) -> float:
        raise IllegalStateError("Called object is of class " + str(self.__class__))


class CrispMVModel(MVModel):

    @abstractmethod
    def fit(self, data: ModelReadyInputData, sample_weight: Optional[SampleWeight] = None) -> CrispMVPredictor:
        raise NotImplementedError()
