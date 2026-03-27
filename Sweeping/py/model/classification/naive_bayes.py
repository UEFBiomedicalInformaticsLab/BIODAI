from sklearn.naive_bayes import GaussianNB

from model.coef_extractor import SklearnCoefExtractor, OffCoefExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor
from model.pipe_wrapper import PipeWrapper

NB_NICK = "NB"
NB_NAME = "naive Bayes"


class NBFactory(SKLearnModelFactoryWithExtractor):

    def create(self):
        # We do not need to standardize since gaussian nb does it internally.
        return PipeWrapper(sklearn_model=GaussianNB(), model_name=NB_NICK, scale=False, supports_weights=True)

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def supports_weights(self) -> bool:
        return True

    def nick(self) -> str:
        return NB_NICK

    def name(self) -> str:
        return NB_NAME
