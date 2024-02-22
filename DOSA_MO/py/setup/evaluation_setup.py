from collections.abc import Sequence
from typing import Optional

from consts import DEFAULT_FOLD_PARALLELISM, DEFAULT_MAX_WORKERS
from cross_validation.multi_objective.optimizer.adjusted_optimizer import DEFAULT_ADJUSTER_REGRESSOR
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from ga_components.selection import DEFAULT_N_PARTICIPANTS, DEFAULT_SELECTION_NAME
from model.logistic import DEFAULT_LOGISTIC_PENALTY
from model.model import DEFAULT_LOGISTIC_MAX_ITER
from model.regression.regressor import RegressorModel
from setup.allowed_names import \
    DEFAULT_VIEWS_MV, DEFAULT_OBJECTIVE_NAMES, NONE_NAME, DEFAULT_DATASET_NAME, \
    DEFAULT_INITIAL_FEATURES_STRATEGY_NAME, SORTING_STRATEGY_DEFAULT, DEFAULT_ALGORITHM_NAME, DEFAULT_OUTER_FOLDS_NAME
from setup.ga_mo_optimizer_setup import DEFAULT_MATING_PROB, DEFAULT_MUTATING_FREQUENCY, POP_SMALL, \
    DEFAULT_SWEEPING_STRATEGY_SMALL, DEFAULT_INNER_N_FOLDS, DEFAULT_MUTATION_OPERATOR, DEFAULT_USE_CLONE_REPURPOSING
from individual.num_features import DEFAULT_INITIAL_FEATURES_MIN, DEFAULT_INITIAL_FEATURES_MAX


DEFAULT_SEED = 48723
DEFAULT_CV_REPEATS = 1


class EvaluationSetup:
    __dataset: str
    __external_dataset: str
    __mvmo_algorithm: str
    __mating_prob: float
    __mutation_frequency: float
    __sorting_strategy: str
    __feature_importance_categorical: str
    __feature_importance_survival: str
    __views_to_use: list[str]
    __pop: int
    __generations: GenerationsStrategy
    __objectives: Sequence
    __inner_n_folds: int
    __outer_n_folds: int
    __use_big_defaults: bool
    __cross_validation: bool
    __final_optimization: bool
    __bitlist_mutation_operator: str
    __initial_features_strategy: str
    __initial_features_min: int
    __initial_features_max: int
    __use_clone_repurposing: bool
    __selection: str
    __selection_tournament_size: int
    __fold_parallelism: bool
    __logistic_max_iter: int
    __outer_folds: str
    __load_base_dir: Optional[str]
    __penalty: str
    __max_workers: int
    __seed: int
    __cv_repeats: int
    __adjuster_regressor: RegressorModel

    def __init__(self,
                 dataset: str = DEFAULT_DATASET_NAME,
                 mvmo_algorithm: str = DEFAULT_ALGORITHM_NAME,
                 mating_prob: float = DEFAULT_MATING_PROB,
                 mutation_frequency: float = DEFAULT_MUTATING_FREQUENCY,
                 sorting_strategy: str = SORTING_STRATEGY_DEFAULT,
                 feature_importance_categorical: str = NONE_NAME,
                 feature_importance_survival: str = NONE_NAME,
                 views_to_use: Sequence[str] = DEFAULT_VIEWS_MV,
                 pop: int = POP_SMALL,
                 generations: GenerationsStrategy = DEFAULT_SWEEPING_STRATEGY_SMALL,
                 objectives: Sequence = DEFAULT_OBJECTIVE_NAMES,
                 inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
                 outer_n_folds: int = None,
                 use_big_defaults: bool = False,
                 cross_validation: bool = True,
                 final_optimization: bool = False,
                 bitlist_mutation_operator=DEFAULT_MUTATION_OPERATOR.nick(),
                 initial_features_strategy: str = DEFAULT_INITIAL_FEATURES_STRATEGY_NAME,
                 initial_features_min: int = DEFAULT_INITIAL_FEATURES_MIN,
                 initial_features_max: int = DEFAULT_INITIAL_FEATURES_MAX,
                 use_clone_repurposing: bool = DEFAULT_USE_CLONE_REPURPOSING,
                 selection: str = DEFAULT_SELECTION_NAME,
                 selection_tournament_size: int = DEFAULT_N_PARTICIPANTS,
                 external_dataset: str = DEFAULT_DATASET_NAME,
                 fold_parallelism: bool = DEFAULT_FOLD_PARALLELISM,
                 logistic_max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
                 outer_folds: str = DEFAULT_OUTER_FOLDS_NAME,
                 load_base_dir: Optional[str] = None,
                 penalty: str = DEFAULT_LOGISTIC_PENALTY,
                 max_workers: int = DEFAULT_MAX_WORKERS,
                 seed: int = DEFAULT_SEED,
                 cv_repeats: int = DEFAULT_CV_REPEATS,
                 adjuster_regressor: RegressorModel = DEFAULT_ADJUSTER_REGRESSOR
                 ):
        self.__dataset = dataset
        self.__mvmo_algorithm = mvmo_algorithm
        self.__mating_prob = mating_prob
        self.__mutation_frequency = mutation_frequency
        self.__sorting_strategy = sorting_strategy
        self.__feature_importance_categorical = feature_importance_categorical
        self.__feature_importance_survival = feature_importance_survival
        self.__views_to_use = list(views_to_use)
        self.__views_to_use.sort()
        self.__pop = pop
        self.__generations = generations
        self.__objectives = objectives
        self.__inner_n_folds = inner_n_folds
        self.__outer_n_folds = outer_n_folds
        self.__use_big_defaults = use_big_defaults
        self.__final_optimization = final_optimization
        self.__cross_validation = cross_validation
        self.__bitlist_mutation_operator = bitlist_mutation_operator
        self.__initial_features_strategy = initial_features_strategy
        self.__initial_features_min = initial_features_min
        self.__initial_features_max = initial_features_max
        self.__use_clone_repurposing = use_clone_repurposing
        self.__selection = selection
        self.__selection_tournament_size = selection_tournament_size
        self.__external_dataset = external_dataset
        self.__fold_parallelism = fold_parallelism
        self.__logistic_max_iter = logistic_max_iter
        self.__outer_folds = outer_folds
        self.__load_base_dir = load_base_dir
        self.__penalty = penalty
        self.__max_workers = max_workers
        self.__seed = seed
        self.__cv_repeats = cv_repeats
        self.__adjuster_regressor = adjuster_regressor

    def dataset(self) -> str:
        return self.__dataset

    def external_dataset(self) -> str:
        return self.__external_dataset

    def mvmo_algorithm(self) -> str:
        return self.__mvmo_algorithm

    def mating_prob(self) -> float:
        return self.__mating_prob

    def mutation_frequency(self) -> float:
        return self.__mutation_frequency

    def sorting_strategy(self) -> str:
        return self.__sorting_strategy

    def feature_importance_categorical(self) -> str:
        return self.__feature_importance_categorical

    def feature_importance_survival(self) -> str:
        return self.__feature_importance_survival

    def views_to_use(self) -> list[str]:
        """Sorted by name"""
        return self.__views_to_use

    def pop(self) -> int:
        return self.__pop

    def generations(self) -> GenerationsStrategy:
        return self.__generations

    def objectives(self) -> Sequence:
        return self.__objectives

    def inner_n_folds(self) -> int:
        return self.__inner_n_folds

    def outer_n_folds(self) -> int:
        return self.__outer_n_folds

    def use_big_defaults(self) -> bool:
        return self.__use_big_defaults

    def final_optimization(self) -> bool:
        return self.__final_optimization

    def bitlist_mutation_operator(self) -> str:
        return self.__bitlist_mutation_operator

    def initial_features_strategy(self) -> str:
        return self.__initial_features_strategy

    def initial_features_min(self) -> int:
        return self.__initial_features_min

    def initial_features_max(self) -> int:
        return self.__initial_features_max

    def use_clone_repurposing(self) -> bool:
        return self.__use_clone_repurposing

    def selection(self) -> str:
        return self.__selection

    def selection_tournament_size(self) -> int:
        return self.__selection_tournament_size

    def fold_parallelism(self) -> bool:
        return self.__fold_parallelism

    def logistic_max_iter(self) -> int:
        return self.__logistic_max_iter

    def outer_folds(self) -> str:
        return self.__outer_folds

    def load_base_dir(self) -> Optional[str]:
        return self.__load_base_dir

    def cross_validation(self) -> bool:
        return self.__cross_validation

    def penalty(self) -> str:
        return self.__penalty

    def max_workers(self) -> int:
        return self.__max_workers

    def seed(self) -> int:
        return self.__seed

    def cv_repeats(self) -> int:
        return self.__cv_repeats

    def adjuster_regressor(self) -> RegressorModel:
        return self.__adjuster_regressor
