from abc import ABC, abstractmethod

from util.list_like import ListLike, BoolListLike


class HyperparamManager(ABC):

    @abstractmethod
    def n_predictive_features(self, hyperparams: ListLike) -> int:
        """Including predictive but not adjusting.
        Refers to the predictive features used by these specific hyperparams."""
        raise NotImplementedError()

    @abstractmethod
    def n_used_features(self, hyperparams: ListLike) -> int:
        """Including both predictive and adjusting.
        Refers to the predictive features used by these specific hyperparams."""
        raise NotImplementedError()

    @abstractmethod
    def collapsed_used_features_mask(self, hyperparams: ListLike) -> BoolListLike:
        """Obtained by concatenating all the views together (both predictive and adjusting views) in alphabetical order.
        Returned mask is in the form of a Boolean list.
        This is used also to create masked predictors, where input is collapsed and then masked."""
        raise NotImplementedError()

    def to_tuple(self, hyperparams: ListLike) -> tuple:
        """Returns a tuple of the true positions that can be used e.g. for hashmaps.
        Two object generating the same tuple represent the same hyperparameters
        as long as the feature space (set of views) is the same.
        It is based on both predictive and adjusting features."""
        return tuple(self.collapsed_used_features_mask(hyperparams=hyperparams).true_positions())

    @abstractmethod
    def predictive_features_mask_len(self, hyperparams: ListLike) -> int:
        """Size of the mask: total number of existing positions in a predictive feature mask.
        A hyperparam instance is passed because certain HP managers do not have the size of the mask
        in the internal state."""
        raise NotImplementedError()
