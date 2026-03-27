import ast
import os
import sys
from collections.abc import Sequence
from configparser import ConfigParser
from typing import Optional

from deprecation import deprecated

from consts import DEFAULT_FOLD_PARALLELISM, DEFAULT_MAX_WORKERS
from cross_validation.multi_objective.optimizer.adjusted_optimizer import DEFAULT_ADJUSTER_REGRESSOR
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from ga_components.selection import DEFAULT_N_PARTICIPANTS, DEFAULT_SELECTION_NAME
from model.classification.logistic import DEFAULT_LOGISTIC_PENALTY, DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER
from model.regression.regressors_list import NICK_TO_REGRESSOR
from objective.balanced_accuracy_with_deviation import DEFAULT_MAX_DEVIATION
from setup.allowed_names import \
    DEFAULT_VIEWS_MV, SOCIAL_SPACE_NAME, NSGA_STAR_NAME, CROWDING_DISTANCE_NAME, RESAMPLED_SWEEPING_NAME, \
    DEFAULT_OBJECTIVE_NAMES, NONE_NAME, DEFAULT_DATASET_NAME, DEFAULT_INITIAL_FEATURES_STRATEGY_NAME, \
    DEFAULT_ALGORITHM_NAME, DEFAULT_OUTER_FOLDS_NAME, FAT_CONCATENATED_SWEEPING_NAME, LEAN_CONCATENATED_SWEEPING_NAME, \
    ADJUSTED_NAME
from setup.evaluation_setup import EvaluationSetup, DEFAULT_SEED, DEFAULT_CV_REPEATS
from setup.ga_mo_optimizer_setup import DEFAULT_MATING_PROB, DEFAULT_MUTATING_FREQUENCY, POP_SMALL, \
    DEFAULT_SWEEPING_STRATEGY_SMALL, DEFAULT_SWEEPING_STRATEGY_BIG, POP_BIG, CLASSIC_GENERATIONS_BIG, \
    CLASSIC_GENERATIONS_SMALL, DEFAULT_INNER_N_FOLDS, DEFAULT_MUTATION_OPERATOR, DEFAULT_USE_CLONE_REPURPOSING, \
    OUTER_N_FOLDS_BIG, OUTER_N_FOLDS_SMALL
from individual.num_features import DEFAULT_INITIAL_FEATURES_MIN, DEFAULT_INITIAL_FEATURES_MAX
from univariate_feature_selection.univariate_feature_selector_descriptor import DEFAULT_FDR_THRESHOLD
from util.printer.printer import Printer, OutPrinter
from util.str_utils import iterable_to_string, parse_json_dict_property
from views.adjusted_view_definition import AdjustedViewDef


def read_all_setups_in_argv(printer: Printer = OutPrinter()) -> Sequence[EvaluationSetup]:
    setups = []
    for i in range(1, len(sys.argv)):  # We parse all of them immediately to catch some errors.
        arg = sys.argv[i]
        if os.path.isfile(arg):
            printer.print("Parsing setup file " + arg)
            setups.append(read_setup(arg))
        else:
            printer.print("File not found " + arg)
    if len(setups) > 0:
        printer.title_print("Running optimizers sequentially according to setups")
    else:
        printer.print("Missing setups.")
    return setups

@deprecated("Old version not used anymore.")
def parse_view_list(views_str: Optional[str]) -> Optional[Sequence[str]]:
    """None triggers the use of default values."""
    if views_str is None:
        return None
    views_list = ast.literal_eval(views_str)
    return [n.strip() for n in views_list]


def parse_view_dict(views_str: Optional[str]) -> AdjustedViewDef:
    """None triggers the use of default views."""
    if views_str is None:
        return DEFAULT_VIEWS_MV
    return AdjustedViewDef(view_to_adjusters=parse_json_dict_property(value=views_str, allow_list_as_dict=True))


def parse_categorical_fs(fs_str: Optional[str]) -> Sequence[str]:
    if fs_str is None:
        return []
    try:
        evaluated = ast.literal_eval(fs_str)
        return [n.strip() for n in evaluated]
    except ValueError:
        return [fs_str.strip()]  # Assuming a single feature selection nick.


def read_setup(file: str) -> EvaluationSetup:
    config = ConfigParser()
    config.read(file)
    section = config["MVMOO_SETUP"]

    dataset = section.get("dataset", DEFAULT_DATASET_NAME)
    mvmo_algorithm = section.get("mvmo_algorithm", DEFAULT_ALGORITHM_NAME)

    use_big_defaults = section.getboolean("use_big_defaults", False)
    if use_big_defaults:
        generations_default = iterable_to_string([CLASSIC_GENERATIONS_BIG])
        pop_default = POP_BIG
        outer_n_folds_default = OUTER_N_FOLDS_BIG
        if (mvmo_algorithm == RESAMPLED_SWEEPING_NAME or mvmo_algorithm == FAT_CONCATENATED_SWEEPING_NAME or
                mvmo_algorithm == LEAN_CONCATENATED_SWEEPING_NAME):
            generations_default = iterable_to_string(DEFAULT_SWEEPING_STRATEGY_BIG.sweeping_list())
    else:
        generations_default = iterable_to_string([CLASSIC_GENERATIONS_SMALL])
        pop_default = POP_SMALL
        outer_n_folds_default = OUTER_N_FOLDS_SMALL
        if (mvmo_algorithm == RESAMPLED_SWEEPING_NAME or mvmo_algorithm == FAT_CONCATENATED_SWEEPING_NAME or
                mvmo_algorithm == LEAN_CONCATENATED_SWEEPING_NAME):
            generations_default = iterable_to_string(DEFAULT_SWEEPING_STRATEGY_SMALL.sweeping_list())

    sorting_strategy_default = SOCIAL_SPACE_NAME
    if mvmo_algorithm == NSGA_STAR_NAME or mvmo_algorithm == ADJUSTED_NAME:
        sorting_strategy_default = CROWDING_DISTANCE_NAME

    generations_str = section.get("generations", generations_default)
    # Property "generations" kept for backward compatibility.
    generations_list = ast.literal_eval(generations_str)
    if (mvmo_algorithm == RESAMPLED_SWEEPING_NAME or mvmo_algorithm == FAT_CONCATENATED_SWEEPING_NAME or mvmo_algorithm
            == LEAN_CONCATENATED_SWEEPING_NAME):
        concatenated_generations = 0
        sweeping_generations = generations_list
    else:
        concatenated_generations = sum(generations_list)
        sweeping_generations = []
    generations_str = section.get("sweeping_generations")
    if generations_str is not None:
        sweeping_generations = ast.literal_eval(generations_str)
    concatenated_generations = section.getint("concatenated_generations", concatenated_generations)
    generations = GenerationsStrategy(sweeps=sweeping_generations, concatenated=concatenated_generations)

    objectives_default = str(DEFAULT_OBJECTIVE_NAMES)
    objectives_str = section.get("objectives", objectives_default)
    objectives = ast.literal_eval(objectives_str)

    feature_importance = section.get("feature_importance", NONE_NAME)
    # Kept for backward compatibility.

    sorting_strategy_default = section.get("secondary_sorting_strategy", sorting_strategy_default)
    # Kept for backward compatibility.

    adjuster_regressor_nick = section.get("adjuster_regressor", DEFAULT_ADJUSTER_REGRESSOR.nick())
    if adjuster_regressor_nick in NICK_TO_REGRESSOR:
        adjuster_regressor = NICK_TO_REGRESSOR[adjuster_regressor_nick]
    else:
        raise ValueError("Unsupported regressor: " + str(adjuster_regressor_nick) + "\n" +
                         "Valid regressors: " + str(NICK_TO_REGRESSOR.keys()) + "\n")

    return EvaluationSetup(
        dataset=dataset,
        mvmo_algorithm=mvmo_algorithm,
        mating_prob=section.getfloat("mating_prob", DEFAULT_MATING_PROB),
        mutation_frequency=section.getfloat("mutation_frequency", DEFAULT_MUTATING_FREQUENCY),
        sorting_strategy=section.get(option="sorting_strategy", fallback=sorting_strategy_default),
        feature_importance_categorical=section.get(
            option="feature_importance_categorical", fallback=feature_importance),
        feature_importance_survival=section.get(option="feature_importance_survival", fallback=NONE_NAME),
        views_to_use=parse_view_dict(views_str=section.get("views_to_use")),
        pop=section.getint("pop", pop_default),
        generations=generations,
        objectives=objectives,
        inner_n_folds=section.getint("inner_n_folds", DEFAULT_INNER_N_FOLDS),
        outer_n_folds=section.getint("outer_n_folds", outer_n_folds_default),
        use_big_defaults=use_big_defaults,
        cross_validation=section.getboolean("cross_validation", True),
        final_optimization=section.getboolean("final_optimization", False),
        bitlist_mutation_operator=section.get(
            option="bitlist_mutation_operator", fallback=DEFAULT_MUTATION_OPERATOR.nick()),
        initial_features_strategy=section.get(
            option="initial_features_strategy", fallback=DEFAULT_INITIAL_FEATURES_STRATEGY_NAME),
        initial_features_min=section.getint("initial_features_min", DEFAULT_INITIAL_FEATURES_MIN),
        initial_features_max=section.getint("initial_features_max", DEFAULT_INITIAL_FEATURES_MAX),
        max_deviation=section.getfloat("max_deviation", DEFAULT_MAX_DEVIATION),
        use_clone_repurposing=section.getboolean("use_clone_repurposing", DEFAULT_USE_CLONE_REPURPOSING),
        selection=section.get(option="selection", fallback=DEFAULT_SELECTION_NAME),
        selection_tournament_size=section.getint("selection_tournament_size", DEFAULT_N_PARTICIPANTS),
        external_dataset=section.get(option="external_dataset", fallback=DEFAULT_DATASET_NAME),
        fold_parallelism=section.getboolean("fold_parallelism", DEFAULT_FOLD_PARALLELISM),
        logistic_max_iter=section.getint("logistic_max_iter", DEFAULT_LOGISTIC_INNER_MODEL_MAX_ITER),
        outer_folds=section.get(option="outer_folds", fallback=DEFAULT_OUTER_FOLDS_NAME),
        load_base_dir=section.get(option="load_base_dir", fallback=None),
        penalty=section.get(option="penalty", fallback=DEFAULT_LOGISTIC_PENALTY),
        max_workers=section.getint("max_workers", DEFAULT_MAX_WORKERS),
        seed=section.getint("seed", DEFAULT_SEED),
        cv_repeats=section.getint("cv_repeats", DEFAULT_CV_REPEATS),
        adjuster_regressor=adjuster_regressor,
        univariate_fs_categorical=parse_categorical_fs(
            fs_str=section.get(option="univariate_fs_categorical", fallback=None)),
        univariate_fs_covariates=parse_view_list(views_str=section.get("univariate_fs_covariates", fallback="[]")),
        univariate_fs_fdr = section.getfloat("univariate_fs_fdr", DEFAULT_FDR_THRESHOLD),
        draw_huge_views=section.getboolean("draw_huge_views", False)
    )
