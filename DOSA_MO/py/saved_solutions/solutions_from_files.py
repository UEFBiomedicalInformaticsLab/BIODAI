from collections.abc import Sequence
from itertools import compress

from cross_validation.multi_objective.cross_evaluator.hof_saver import INNER_CV_PREFIX
from plots.performance_by_class import confusion_matrices_from_hof_dir_by_fold, confusion_matrices_from_hof_dir_external
from plots.hof_utils import hof_used_features_final_df, hof_used_features_fold_dfs, hof_final_fitness_df
from saved_solutions.saved_solution import SavedSolution
from saved_solutions.solution_attributes_archive import STD_DEV, FITNESS, CI_MIN, CI_MAX
from util.dataframes import n_row, row_as_list
from util.hyperbox.hyperbox import ConcreteInterval


def n_folds_from_files(hof_dir: str) -> int:
    """Returns 1 if it is external validation. Returns 0 if nothing is found."""
    test_fits = FITNESS.to_test_dfs(hof_dir=hof_dir)
    if test_fits is None:
        return 0
    else:
        return len(test_fits)


def solutions_from_files(hof_dir: str) -> Sequence[Sequence[SavedSolution]]:
    """External sequence is for folds. If it is external validation, it is returned as sequence of a single fold."""
    test_fits = FITNESS.to_test_dfs(hof_dir=hof_dir)
    if test_fits is None:
        return []
    train_cv_fits = FITNESS.train_data(hof_dir=hof_dir)
    if train_cv_fits is None:
        raise ValueError("Test fitnesses are readable but inner cv fitnesses are not.")
    train_sds = STD_DEV.train_data(hof_dir=hof_dir)
    test_sds = STD_DEV.to_test_dfs(hof_dir=hof_dir)
    train_ci_mins = CI_MIN.train_data(hof_dir=hof_dir)
    test_ci_mins = CI_MIN.to_test_dfs(hof_dir=hof_dir)
    train_ci_maxs = CI_MAX.train_data(hof_dir=hof_dir)
    test_ci_maxs = CI_MAX.to_test_dfs(hof_dir=hof_dir)
    cms = confusion_matrices_from_hof_dir_by_fold(hof_dir=hof_dir)
    features = hof_used_features_fold_dfs(hof_dir=hof_dir)
    n_folds = len(test_fits)
    res = []
    read_cms = len(cms) == n_folds
    read_feats = len(features) == n_folds
    for fold_i, test_fold_fits in enumerate(test_fits):
        inner_cv_fold_fits = train_cv_fits[fold_i]
        if read_cms:
            fold_cms = cms[fold_i]
        else:
            fold_cms = None
        if read_feats:
            fold_features = features[fold_i]
            fold_feature_names = fold_features.columns
        else:
            fold_features = None
            fold_feature_names = None
        fold_solutions = []
        for i in range(n_row(test_fold_fits)):
            test_fit = row_as_list(df=test_fold_fits, row=i)
            inner_cv_fit = row_as_list(df=inner_cv_fold_fits, row=i)
            if read_feats:
                features_mask = row_as_list(df=fold_features, row=i)
                features_i = list(compress(fold_feature_names, features_mask))
            else:
                features_i = None
            if read_cms:
                cm = fold_cms[i]
            else:
                cm = None
            if train_sds is not None:
                train_sd = row_as_list(df=train_sds[fold_i], row=i)
            else:
                train_sd = None
            if test_sds is not None:
                test_sd = row_as_list(df=test_sds[fold_i], row=i)
            else:
                test_sd = None
            if train_ci_mins is not None and train_ci_maxs is not None:
                train_ci = [ConcreteInterval(a=a, b=b) for a, b in
                            zip(row_as_list(df=train_ci_mins[fold_i], row=i),
                                row_as_list(df=train_ci_maxs[fold_i], row=i))]
            else:
                train_ci = None
            if test_ci_mins is not None and test_ci_maxs is not None:
                test_ci = [ConcreteInterval(a=a, b=b) for a, b in
                           zip(row_as_list(df=test_ci_mins[fold_i], row=i),
                               row_as_list(df=test_ci_maxs[fold_i], row=i))]
            else:
                test_ci = None
            fold_solutions.append(
                SavedSolution(
                    train_fitnesses=inner_cv_fit, test_fitnesses=test_fit, confusion_matrix=cm, features=features_i,
                    train_std_devs=train_sd, test_std_devs=test_sd, train_ci=train_ci, test_ci=test_ci))
        res.append(fold_solutions)
    return res


def objective_names(hof_dir: str) -> Sequence[str]:
    """Names extracted from table of final solution performances."""
    train_fits = FITNESS.to_final_internal_cv_df(hof_dir=hof_dir)
    if train_fits is None:
        train_fits = hof_final_fitness_df(hof_dir=hof_dir)
    if train_fits is None:
        return []
    else:
        return train_fits.columns


def final_solutions_from_files(hof_dir: str) -> Sequence[SavedSolution]:
    """Works for both internal final training and for external validation.
    Do not call this for results of internal-external k-fold cross-validation.
    Returns empty sequence if nothing to read.
    TODO Read also standard deviations and CIs."""
    train_fits = FITNESS.to_final_internal_cv_df(hof_dir=hof_dir)
    if train_fits is None:
        train_fits = hof_final_fitness_df(hof_dir=hof_dir)
    test_fits = FITNESS.to_final_external_df(hof_dir=hof_dir)
    test_cms = confusion_matrices_from_hof_dir_external(hof_dir=hof_dir)
    features = hof_used_features_final_df(hof_dir=hof_dir)
    has_train_fits = train_fits is not None
    has_test_fits = test_fits is not None
    n_cms = len(test_cms)
    has_features = features is not None
    if has_features:
        feature_names = features.columns
    else:
        feature_names = None
    solutions = []
    n_solutions = None
    if has_test_fits:
        n_solutions = n_row(test_fits)
    if n_cms > 0:
        if n_solutions is None:
            n_solutions = n_cms
        else:
            if n_cms != n_solutions:
                raise ValueError("Number of fitnesses differs from number of confusion matrices.")
    if has_features:
        if n_solutions is None:
            n_solutions = n_row(features)
        else:
            if n_row(features) != n_solutions:
                raise ValueError(
                    "Number of lines in features file differs from previously inferred number of solutions.")
    if n_solutions is None:
        return solutions
    else:
        for i in range(n_solutions):
            if has_train_fits:
                cols_to_keep = [c.startswith(INNER_CV_PREFIX) for c in train_fits.columns]
                train_fit = train_fits.iloc[i, cols_to_keep].values.flatten().tolist()
            else:
                train_fit = None
            if has_test_fits:
                test_fit = test_fits.iloc[i, :].values.flatten().tolist()
            else:
                test_fit = None
            if has_features:
                features_mask = features.iloc[i, :].values.flatten().tolist()
                features_i = list(compress(feature_names, features_mask))
            else:
                features_i = None
            if n_cms > 0:
                test_cm = test_cms[i]
            else:
                test_cm = None
            solutions.append(SavedSolution(
                train_fitnesses=train_fit, test_fitnesses=test_fit, confusion_matrix=test_cm, features=features_i))
        return solutions
