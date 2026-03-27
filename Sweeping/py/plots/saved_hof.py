import os
from typing import Optional, Sequence

import pandas as pd
from pandas import DataFrame
from scipy.stats import pearsonr

from cross_validation.multi_objective.cross_evaluator.confusion_matrices_saver import CONFUSION_MATRIX_STR
from cross_validation.multi_objective.multi_objective_cross_validation import VALIDATION_REGISTRY_FILE_NAME
from location_manager.location_manager_utils import views_name, COVARIATES_DEFAULT, VIEWS_DEFAULT
from plots.hof_utils import is_external_dir
from plots.performance_by_class import read_all_cms
from prediction_stats.confusion_matrix import ConfusionMatrix
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solution_attribute import SolutionAttribute
from saved_solutions.solution_attributes_archive import STD_DEV, FITNESS
from saved_solutions.solutions_from_files import final_solutions_from_files
from setup.views_from_saved_setup import views_from_saved_setup
from util.str_utils import fdr_str
from univariate_feature_selection.univariate_feature_selector_descriptor import (ManyFeatureSelectorClassDescriptor,
                                                                                 DEFAULT_CATEGORICAL_FS_DESCRIPTOR)
from util.dataframe.dataframes import columnwise_correlations, columnwise_correlations_p_val, columnwise_measures
from util.math.sequences_to_float import SequencesToFloat
from util.named import Named
from util.utils import IllegalStateError
from util.name_parts import parts_in_common, names_by_differences
from validation_registry.validation_registry import ValidationRegistry, FileValidationRegistry
from views.adjusted_view_definition import AdjustedViewDef


def validation_registry_path(hof_path: str) -> str:
    return os.path.join(hof_path, VALIDATION_REGISTRY_FILE_NAME)


def validation_registry_from_hof_path(hof_path: str) -> ValidationRegistry:
    return FileValidationRegistry(file_path=validation_registry_path(hof_path=hof_path))


class SavedHoF(Named):
    __path: str
    __main_algorithm_label: str
    __inner_lab: Optional[str] = None
    __adjuster_regressor: Optional[str]
    __views: AdjustedViewDef
    __covariates: set[str]
    __categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor
    __dataset_lab: Optional[str]

    def __init__(self, path: str,
                 main_algorithm_label: str = "",
                 inner_lab: Optional[str] = None,
                 adjuster_regressor: Optional[str] = None,
                 views: AdjustedViewDef = VIEWS_DEFAULT,
                 categorical_fs_descriptor: ManyFeatureSelectorClassDescriptor = DEFAULT_CATEGORICAL_FS_DESCRIPTOR,
                 covariates: set[str] = COVARIATES_DEFAULT,
                 dataset_lab: Optional[str] = None
                 ):
        """Ignores passed fdr threshold and sets it to None if the categorical fs does not use it."""
        self.__path = path
        self.__main_algorithm_label = main_algorithm_label
        self.__inner_lab = inner_lab
        self.__adjuster_regressor = adjuster_regressor
        self.__views = views
        self.__categorical_fs_descriptor = categorical_fs_descriptor
        self.__dataset_lab = dataset_lab

        if self.__categorical_fs_descriptor.uses_covariates():
            self.__covariates = covariates
        else:
            self.__covariates = set()

    def path(self) -> str:
        return self.__path

    def name(self) -> str:
        """For example: NSGA2 NB ANOVA."""
        res = ""
        if self.__adjuster_regressor is not None:
            res += self.__adjuster_regressor + " "
        res += self.__main_algorithm_label
        if self.__inner_lab is not None:
            res += " " + self.__inner_lab
        res += " " + self.__categorical_fs_descriptor.name()
        return res

    def main_algorithm_label(self) -> str:
        return self.__main_algorithm_label

    def views_name(self) -> str:
        return self.__views.name()

    def covariates_name(self) -> str:
        return views_name(view_names=self.__covariates)

    def fdr_threshold_name(self) -> str:
        if self.__categorical_fs_descriptor.has_fdr_threshold():
            return fdr_str(fdr_threshold=self.__categorical_fs_descriptor.fdr_threshold())
        else:
            return ""

    def dataset_lab(self) -> str:
        if self.__dataset_lab is None:
            return ""
        else:
            return self.__dataset_lab

    def name_parts(self) -> Sequence[str]:
        return (self.dataset_lab(), self.views_name(), self.main_algorithm_label(), self.inner_str(),
                self.adjuster_str(),
                self.categorical_fs_algorithm_nick(), self.covariates_name(), self.fdr_threshold_name())

    def inner_str(self) -> str:
        if self.__inner_lab is None:
            return ""
        else:
            return self.__inner_lab

    def adjuster_str(self) -> str:
        if self.__adjuster_regressor is None:
            return ""
        else:
            return self.__adjuster_regressor

    def path_exists(self) -> bool:
        return os.path.isdir(self.path())

    def to_df(self, verbose: bool = False) -> Optional[DataFrame]:
        """If it is not possible to create a df returns None. It is a df of external fitness or a
        df with all test performance together (all folds) if the folder does not contain an external validation."""
        f = self.path()
        df = None
        if self.path_exists():
            if is_external_dir(hof_dir=f):
                df = FITNESS.external_df(hof_dir=f)
            else:
                df = FITNESS.test_df(hof_dir=f)
            if df is None:
                print("Unable to create dataframe from directory " + str(f))
        else:
            if verbose:
                print("path is not a directory: " + str(f))
        return df

    def test_df(self, attribute: SolutionAttribute) -> Optional[DataFrame]:
        """Puts all folds in one df. One column for each objective."""
        return attribute.test_df(hof_dir=self.path())

    def test_fitness_df(self) -> Optional[DataFrame]:
        """Puts all folds in one df. One column for each objective."""
        return self.test_df(attribute=FITNESS)

    def test_objective(self, attribute: SolutionAttribute, obj: int) -> Optional[Sequence[float]]:
        """Works for both k-fold CV and external validation. Puts all folds in one sequence if it is k-fold CV."""
        if self.is_external():
            df = attribute.external_df(hof_dir=self.path())
        else:
            df = self.test_df(attribute=attribute)
        if df is None:
            return None
        else:
            return df.iloc[:, obj]

    def test_fitness_objective(self, obj: int) -> Optional[Sequence[float]]:
        """Works for both k-fold CV and external validation. Puts all folds in one sequence if it is k-fold CV."""
        return self.test_objective(attribute=FITNESS, obj=obj)

    def test_fitness_dfs(self) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        return FITNESS.test_data(hof_dir=self.path())

    def train_dfs(self, attribute: SolutionAttribute) -> Optional[Sequence[DataFrame]]:
        """Data from inner cv if available, from training on whole folds otherwise.
        One df for each fold."""
        return attribute.train_data(hof_dir=self.path())

    def test_dfs(self, attribute: SolutionAttribute) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        return attribute.test_data(hof_dir=self.path())

    def train_fitness_dfs(self) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        return self.train_dfs(attribute=FITNESS)

    def train_objective_folds(self, attribute: SolutionAttribute, obj: int) -> Optional[Sequence[Sequence[float]]]:
        """Data from inner cv if available, from training on whole folds otherwise.
        One sequence for each fold."""
        dfs = self.train_dfs(attribute=attribute)
        if dfs is None:
            return None
        else:
            return [df.iloc[:, obj] for df in dfs]

    def test_objective_folds(self, attribute: SolutionAttribute, obj: int) -> Optional[Sequence[Sequence[float]]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        dfs = self.test_dfs(attribute=attribute)
        if dfs is None:
            return None
        else:
            return [df.iloc[:, obj] for df in dfs]

    def train_fitness_objective_folds(self, obj: int) -> Optional[Sequence[Sequence[float]]]:
        """Data from inner cv if available, from training on whole folds otherwise.
        One sequence for each fold."""
        return self.train_objective_folds(attribute=FITNESS, obj=obj)

    def test_fitness_objective_folds(self, obj: int) -> Optional[Sequence[Sequence[float]]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        return self.test_objective_folds(attribute=FITNESS, obj=obj)

    def train_std_dev_dfs(self) -> Optional[Sequence[DataFrame]]:
        """Data from inner cv if available, from training on whole folds otherwise.
        One df for each fold."""
        return self.train_dfs(attribute=STD_DEV)

    def test_std_dev_dfs(self) -> Optional[Sequence[DataFrame]]:
        """One df for each fold, or just a sequence of one df if it is from external validation."""
        return STD_DEV.test_data(hof_dir=self.path())

    def performance_gap_dfs(self) -> Optional[Sequence[DataFrame]]:
        train_dfs = self.train_fitness_dfs()
        test_dfs = self.test_fitness_dfs()
        if train_dfs is None or test_dfs is None:
            return None
        else:
            return [tr.subtract(te) for tr, te in zip(train_dfs, test_dfs)]

    def sd_gap_correlations(self, corr_function=pearsonr) -> list[float]:
        deviations = pd.concat(self.train_std_dev_dfs())
        gaps = pd.concat(self.performance_gap_dfs())
        return columnwise_correlations(df1=deviations, df2=gaps, corr_function=corr_function)

    def sd_gap_correlations_by_fold(self, corr_function=pearsonr) -> list[list[float]]:
        deviations = self.train_std_dev_dfs()
        gaps = self.performance_gap_dfs()
        return [columnwise_correlations(df1=d, df2=g, corr_function=corr_function) for d, g in zip(deviations, gaps)]

    def sd_gap_measures_by_fold(self, measure: SequencesToFloat) -> Optional[list[list[float]]]:
        deviations = self.train_std_dev_dfs()
        gaps = self.performance_gap_dfs()
        if deviations is None or gaps is None:
            return None
        else:
            try:
                return [columnwise_measures(df1=d, df2=g, measure=measure) for d, g in zip(deviations, gaps)]
            except ValueError:
                return None  # There might be NaN if deviations are not available.

    def sd_gap_correlations_p_val(self, corr_function=pearsonr) -> list[float]:
        deviations = pd.concat(self.train_std_dev_dfs())
        gaps = pd.concat(self.performance_gap_dfs())
        return columnwise_correlations_p_val(df1=deviations, df2=gaps, corr_function=corr_function)

    def sd_gap_correlations_p_val_by_fold(self, corr_function=pearsonr) -> list[list[float]]:
        deviations = self.train_std_dev_dfs()
        gaps = self.performance_gap_dfs()
        return\
            [columnwise_correlations_p_val(df1=d, df2=g, corr_function=corr_function) for d, g in zip(deviations, gaps)]

    def confusion_matrices(self) -> list[ConfusionMatrix]:
        """Fold files are read in alphabetical order. If passed path is not a directory returns an empty list.
        Does not handle cases with more than one categorical objective."""
        cm_dir = os.path.join(self.path(), CONFUSION_MATRIX_STR)
        return read_all_cms(cm_dir=cm_dir)

    def validation_registry_path(self) -> str:
        return validation_registry_path(hof_path=self.path())

    def has_validation_registry(self) -> bool:
        return os.path.exists(self.validation_registry_path())

    def validation_registry(self) -> ValidationRegistry:
        """Validation registry is created if not existing."""
        return FileValidationRegistry(file_path=self.validation_registry_path())

    def categorical_fs_algorithm_name(self) -> str:
        return self.__categorical_fs_descriptor.algorithm_name()

    def categorical_fs_algorithm_nick(self) -> str:
        return self.__categorical_fs_descriptor.algorithm_nick()

    def __str__(self) -> str:
        res = ""
        res += "views: " + self.views_name() + "\n"
        res += "main algorithm: " + self.__main_algorithm_label + "\n"
        if self.__inner_lab is not None:
            res += "inner model: " + self.__inner_lab + "\n"
        if self.__adjuster_regressor is not None:
            res += "adjuster regressor: " + self.__adjuster_regressor + "\n"
        res += "categorical fs: " + str(self.__categorical_fs_descriptor) + "\n"
        if self.__covariates is not None:
            res += "covariate views: " + self.covariates_name() + "\n"
        res += "path: " + self.__path + "\n"
        return res

    def __eq__(self, other) -> bool:
        if isinstance(other, SavedHoF):
            return self.path() == other.path()
        else:
            return False

    def __hash__(self):
        return hash(self.path())

    def optimizer_dir(self) -> str:
        p = self.path()
        p, _ = os.path.split(p)
        p, _ = os.path.split(p)
        return p

    def has_obj_nicks(self) -> bool:
        return self.test_fitness_dfs() is not None

    def obj_nicks(self) -> Sequence[str]:
        """Nicknames of the objectives, including inner models.
        They are returned in the same order used in the output CSVs."""
        if self.has_obj_nicks():
            return self.test_fitness_dfs()[0].columns
        else:
            raise IllegalStateError()

    def obj_pos(self, nick:str) -> int:
        """Objective position in output CSVs (starting from 0).
        Gets the objective nick in input (including inner model if relevant).
        Throws exception if not preset or mapping not available."""
        nicks = self.obj_nicks()
        nicks_list = list(nicks)
        try:
            return nicks_list.index(nick)
        except ValueError as e:
            raise ValueError(str(e) + "\nlist: " + str(nicks_list))

    def is_external(self) -> bool:
        return is_external_dir(hof_dir=self.path())

    def final_solutions(self) -> Sequence[SavedSolution]:
        """Works for both internal final training and for external validation.
            Do not call this for results of k-fold cross-validation.
            Returns empty sequence if nothing to read."""
        return final_solutions_from_files(hof_dir=self.path())

    def views_from_saved_setup(self) -> Sequence[str]:
        return views_from_saved_setup(optimizer_dir=self.optimizer_dir())


def no_hof_exists(hofs: Sequence[SavedHoF]) -> bool:
    for h in hofs:
        if h.path_exists():
            return False
    return True


def battery_names(hofs: Sequence[SavedHoF]) -> Sequence[str]:
    return names_by_differences(object_features=[h.name_parts() for h in hofs])


def battery_common_name_parts(hofs: Sequence[SavedHoF]) -> Sequence[str]:
    return parts_in_common(object_features=[h.name_parts() for h in hofs])


def existing_hofs(hofs: Sequence[SavedHoF]) -> Sequence[SavedHoF]:
    res = []
    for hof in hofs:
        if hof.path_exists():
            res.append(hof)
    return res
