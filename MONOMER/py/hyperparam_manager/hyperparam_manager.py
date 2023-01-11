from abc import ABC, abstractmethod

from util.list_like import ListLike


class HyperparamManager(ABC):

    @abstractmethod
    def n_active_features(self, hyperparams):
        raise NotImplementedError()

    @abstractmethod
    def active_features_mask(self, hyperparams: ListLike) -> ListLike:
        raise NotImplementedError()

    @abstractmethod
    def active_features_mask_len(self, hyperparams) -> int:
        raise NotImplementedError()

    @abstractmethod
    def active_features_mask_numpy(self, hyperparams):
        raise NotImplementedError()

    def to_tuple(self, hyperparams: ListLike) -> ():
        """Returns a tuple of the true positions that can be used e.g. for hashmaps."""
        return tuple(self.active_features_mask(hyperparams=hyperparams).true_positions())
