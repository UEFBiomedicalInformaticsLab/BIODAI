import sklearn.model_selection

from folds_creator.folds_creator import FoldsCreator


class SplitFoldsCreator(FoldsCreator):

    def create_folds_categorical(self, x, y, seed=36534):
        ss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        res = next(ss.split(x, y))
        return [res, ]
