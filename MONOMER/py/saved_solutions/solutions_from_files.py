from collections.abc import Sequence
from itertools import compress

from plots.performance_by_class import confusion_matrices_from_hof_dir_by_fold, confusion_matrices_from_hof_dir_external
from plots.hof_utils import hof_used_features_final_df, hof_used_features_fold_dfs, to_train_dfs, to_test_dfs, \
    to_final_internal_cv_df, to_final_external_df
from saved_solutions.saved_solution import SavedSolution
from util.dataframes import n_row


def solutions_from_files(hof_dir: str) -> Sequence[Sequence[SavedSolution]]:
    """External sequence is for folds. If it is external validation, it is returned as a single fold."""
    test_fits = to_test_dfs(hof_dir=hof_dir)
    if test_fits is None:
        return []
    train_cv_fits = to_train_dfs(hof_dir=hof_dir)
    if train_cv_fits is None:
        raise ValueError("Test fitnesses are readable but inner cv fitnesses are not.")
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
            test_fit = test_fold_fits.iloc[i, :].values.flatten().tolist()
            inner_cv_fit = inner_cv_fold_fits.iloc[i, :].values.flatten().tolist()
            if read_feats:
                features_mask = fold_features.iloc[i, :].values.flatten().tolist()
                features_i = list(compress(fold_feature_names, features_mask))
            else:
                features_i = None
            if read_cms:
                cm = fold_cms[i]
            else:
                cm = None
            fold_solutions.append(
                SavedSolution(
                    train_fitnesses=inner_cv_fit, test_fitnesses=test_fit, confusion_matrix=cm, features=features_i))
        res.append(fold_solutions)
    return res


def final_solutions_from_files(hof_dir: str) -> Sequence[SavedSolution]:
    """Works for both internal final training and for external validation. Returns empty sequence if nothing to read."""
    train_fits = to_final_internal_cv_df(hof_dir=hof_dir)
    test_fits = to_final_external_df(hof_dir=hof_dir)
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
                train_fit = train_fits.iloc[i, :].values.flatten().tolist()
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
