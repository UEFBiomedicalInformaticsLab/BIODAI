from abc import abstractmethod, ABC

from sklearn import svm

from model.class_proba.probability_wrapper import ProbabilityWrapper
from model.coef_extractor import SklearnCoefExtractor, OffCoefExtractor
from model.model_with_coef import SKLearnModelFactoryWithExtractor
from model.pipe_wrapper import PipeWrapper

RBF_SVM_NICK = "svm"
LINEAR_SVM_NICK = "lsvm"


class SVMFactory(SKLearnModelFactoryWithExtractor, ABC):
    __probability: bool

    def __init__(self, probability: bool = False):
        """Supporting predict_proba is more expensive, and it is not activated by default."""
        self.__probability = probability

    def create(self):
        model = svm.SVC(kernel=self.kernel(), class_weight='balanced')
        if self.__probability:
            model = ProbabilityWrapper(base_model=model)
            # Using ProbabilityWrapper instead of setting probabilities to True in svm because this second option is
            # extremely slow.
        return PipeWrapper(sklearn_model=model,
                           model_name=self.nick(),
                           scale=True,
                           supports_weights=False)
        # In fact svm.SVC supports weights, but setting class_weight='balanced' is a bit faster.

    def coef_extractor(self) -> SklearnCoefExtractor:
        return OffCoefExtractor()

    def supports_weights(self) -> bool:
        """Not supported at the moment. Support might be provided in the future."""
        return False

    @abstractmethod
    def kernel(self) -> str:
        raise NotImplementedError()


class LSVMFactory(SVMFactory):

    def kernel(self) -> str:
        return "linear"

    def nick(self) -> str:
        return LINEAR_SVM_NICK


class RSVMFactory(SVMFactory):

    def kernel(self) -> str:
        return "rbf"

    def nick(self) -> str:
        return RBF_SVM_NICK

