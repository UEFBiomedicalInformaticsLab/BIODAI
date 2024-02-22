from folds_creator.input_data_k_folds_creator import InputDataKFoldsCreator
from folds_creator.mo_class_assigner.full_class_assigner import FullClassAssigner
from folds_creator.mo_class_assigner.mo_class_assigner import MOClassAssigner


class MOInputDataKFoldsCreator(InputDataKFoldsCreator):

    def __init__(self, n_folds: int, n_repeats: int = 1):
        InputDataKFoldsCreator.__init__(self=self, n_folds=n_folds, n_repeats=n_repeats)

    def name(self) -> str:
        return "MO-" + str(self.n_folds()) + "-folds creator"

    def nick(self) -> str:
        return "k" + str(self.n_folds())  # We do not add MO for backward compatibility with plotting routines.

    def __str__(self) -> str:
        return "multi-outcome input data " + str(self.n_folds()) + "-folds creator"

    def _class_assigner(self) -> MOClassAssigner:
        return FullClassAssigner()
