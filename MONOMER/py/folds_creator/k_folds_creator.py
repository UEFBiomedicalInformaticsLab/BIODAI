import sklearn.model_selection

from folds_creator.folds_creator import FoldsCreator


class KFoldsCreator(FoldsCreator):
    __n_folds: int

    def __init__(self, n_folds):
        self.__n_folds = n_folds

    def create_folds(self, x, y, seed: int = 365):
        skf = sklearn.model_selection.StratifiedKFold(
            n_splits=self.__n_folds, shuffle=True, random_state=seed)
        res = []
        for train_index, test_index in skf.split(X=x, y=y):
            res.append([train_index, test_index])
        return res
