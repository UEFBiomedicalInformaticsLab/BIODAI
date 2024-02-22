from folds_creator.input_data_k_folds_creator import InputDataKFoldsCreator
from folds_creator.mo_class_assigner.mo_class_assigner import MOClassAssigner
from folds_creator.mo_class_assigner.simple_class_assigner import SimpleClassAssigner


class MonoInputDataKFoldsCreator(InputDataKFoldsCreator):

    def __init__(self, n_folds, n_repeats: int = 1):
        InputDataKFoldsCreator.__init__(self=self, n_folds=n_folds, n_repeats=n_repeats)

    def name(self) -> str:
        return str(self.n_folds()) + "-folds creator"

    def nick(self) -> str:
        return "k" + str(self.n_folds())

    def __str__(self) -> str:
        return "input data " + str(self.n_folds()) + "-folds creator"

    def _class_assigner(self) -> MOClassAssigner:
        return SimpleClassAssigner()
