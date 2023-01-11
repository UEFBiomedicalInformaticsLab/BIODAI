import ast
import os
import sys
from configparser import ConfigParser

from consts import DEFAULT_FOLD_PARALLELISM
from ga_components.selection import DEFAULT_N_PARTICIPANTS, DEFAULT_SELECTION_NAME
from model.model import DEFAULT_LOGISTIC_MAX_ITER
from objective.balanced_accuracy_with_deviation import DEFAULT_MAX_DEVIATION
from setup.allowed_names import \
    DEFAULT_VIEWS_MV, SOCIAL_SPACE_NAME, NSGA_STAR_NAME, CROWDING_DISTANCE_NAME, SWEEPING_NAME, \
    DEFAULT_OBJECTIVE_NAMES, NONE_NAME, DEFAULT_DATASET_NAME, DEFAULT_INITIAL_FEATURES_STRATEGY_NAME, \
    DEFAULT_ALGORITHM_NAME, DEFAULT_OUTER_FOLDS_NAME
from setup.evaluation_setup import EvaluationSetup
from setup.ga_mo_optimizer_setup import DEFAULT_MATING_PROB, DEFAULT_MUTATING_FREQUENCY, POP_SMALL, \
    DEFAULT_SWEEPING_STRATEGY_SMALL, DEFAULT_SWEEPING_STRATEGY_BIG, POP_BIG, CLASSIC_GENERATIONS_BIG, \
    CLASSIC_GENERATIONS_SMALL, DEFAULT_INNER_N_FOLDS, DEFAULT_MUTATION_OPERATOR, DEFAULT_USE_CLONE_REPURPOSING
from individual.num_features import DEFAULT_INITIAL_FEATURES_MIN, DEFAULT_INITIAL_FEATURES_MAX
from util.printer.printer import Printer, OutPrinter
from util.sequence_utils import sequence_to_string


def read_all_setups_in_argv(printer: Printer = OutPrinter()) -> [EvaluationSetup]:
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


def read_setup(file: str) -> EvaluationSetup:
    config = ConfigParser()
    config.read(file)
    section = config["MVMOO_SETUP"]

    dataset = section.get("dataset", DEFAULT_DATASET_NAME)
    mvmo_algorithm = section.get("mvmo_algorithm", DEFAULT_ALGORITHM_NAME)

    use_big_defaults = section.getboolean("use_big_defaults", False)
    if use_big_defaults:
        generations_default = sequence_to_string([CLASSIC_GENERATIONS_BIG])
        pop_default = POP_BIG
        if mvmo_algorithm == SWEEPING_NAME:
            generations_default = sequence_to_string(DEFAULT_SWEEPING_STRATEGY_BIG)
    else:
        generations_default = sequence_to_string([CLASSIC_GENERATIONS_SMALL])
        pop_default = POP_SMALL
        if mvmo_algorithm == SWEEPING_NAME:
            generations_default = sequence_to_string(DEFAULT_SWEEPING_STRATEGY_SMALL)

    sorting_strategy_default = SOCIAL_SPACE_NAME
    if mvmo_algorithm == NSGA_STAR_NAME:
        sorting_strategy_default = CROWDING_DISTANCE_NAME

    generations_str = section.get("generations", generations_default)
    generations_list = ast.literal_eval(generations_str)
    default_views = DEFAULT_VIEWS_MV
    views_str = section.get("views_to_use", str(default_views))
    views_list = ast.literal_eval(views_str)
    views_list = [n.strip() for n in views_list]

    objectives_default = str(DEFAULT_OBJECTIVE_NAMES)
    objectives_str = section.get("objectives", objectives_default)
    objectives = ast.literal_eval(objectives_str)

    feature_importance = section.get("feature_importance", NONE_NAME)
    # Kept for backward compatibility.

    sorting_strategy_default = section.get("secondary_sorting_strategy", sorting_strategy_default)
    # Kept for backward compatibility.

    return EvaluationSetup(
        dataset=dataset,
        mvmo_algorithm=mvmo_algorithm,
        mating_prob=section.getfloat("mating_prob", DEFAULT_MATING_PROB),
        mutation_frequency=section.getfloat("mutation_frequency", DEFAULT_MUTATING_FREQUENCY),
        sorting_strategy=section.get("sorting_strategy", sorting_strategy_default),
        feature_importance_categorical=section.get("feature_importance_categorical", feature_importance),
        feature_importance_survival=section.get("feature_importance_survival", NONE_NAME),
        views_to_use=views_list,
        pop=section.getint("pop", pop_default),
        generations=generations_list,
        objectives=objectives,
        inner_n_folds=section.getint("inner_n_folds", DEFAULT_INNER_N_FOLDS),
        use_big_defaults=use_big_defaults,
        cross_validation=section.getboolean("cross_validation", True),
        final_optimization=section.getboolean("final_optimization", False),
        bitlist_mutation_operator=section.get("bitlist_mutation_operator", DEFAULT_MUTATION_OPERATOR.nick()),
        initial_features_strategy=section.get("initial_features_strategy", DEFAULT_INITIAL_FEATURES_STRATEGY_NAME),
        initial_features_min=section.getint("initial_features_min", DEFAULT_INITIAL_FEATURES_MIN),
        initial_features_max=section.getint("initial_features_max", DEFAULT_INITIAL_FEATURES_MAX),
        max_deviation=section.getfloat("max_deviation", DEFAULT_MAX_DEVIATION),
        use_clone_repurposing=section.getboolean("use_clone_repurposing", DEFAULT_USE_CLONE_REPURPOSING),
        selection=section.get("selection", DEFAULT_SELECTION_NAME),
        selection_tournament_size=section.getint("selection_tournament_size", DEFAULT_N_PARTICIPANTS),
        external_dataset=section.get("external_dataset", DEFAULT_DATASET_NAME),
        fold_parallelism=section.getboolean("fold_parallelism", DEFAULT_FOLD_PARALLELISM),
        logistic_max_iter=section.getint("logistic_max_iter", DEFAULT_LOGISTIC_MAX_ITER),
        outer_folds=section.get("outer_folds", DEFAULT_OUTER_FOLDS_NAME),
        load_base_dir=section.get("load_base_dir", None)
    )
