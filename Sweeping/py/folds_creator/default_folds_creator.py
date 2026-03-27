from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from folds_creator.mo_input_data_kfolds_creator import MOInputDataKFoldsCreator


def default_folds_creator(n_folds: int, n_repeats: int = 1) -> InputDataFoldsCreator:
    return MOInputDataKFoldsCreator(n_folds=n_folds, n_repeats=n_repeats)
