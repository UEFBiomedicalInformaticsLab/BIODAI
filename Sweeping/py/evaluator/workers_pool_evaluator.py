import multiprocessing
from collections.abc import Sequence, Iterator, Iterable

from cross_validation.multi_objective.mo_cv_result import MOCVResult
from folds_creator.index_array import IndexArray
from hyperparam_manager.mv_hyperparam_manager.mv_hyperparam_manager import MvHyperparamManager
from input_data.input_data import InputData
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from worker.worker_state import WorkerState
from worker.work_package import WorkPackage
from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from worker.worker_state import parallel_init, multiprocessing_friendly_evaluation_with_init
from util.printer.printer import Printer, UnbufferedOutPrinter
from itertools import islice


class WorkersPoolEvaluator:
    __workers_pool: multiprocessing.Pool
    __worker_state: WorkerState
    __n_views: int
    __n_predictive_features: int
    """Does not include features used for adjustment."""
    __n_workers: int
    __folds_list: list[tuple[IndexArray,IndexArray]]

    def __init__(
            self, input_data: InputData, folds_list: list[tuple[IndexArray,IndexArray]], hp_manager: MvHyperparamManager,
            objectives: Sequence[PersonalObjectiveWithImportance],
            n_workers=1, seed=35634, workers_printer: Printer = UnbufferedOutPrinter(),
            compute_feature_importance: bool = False,
            compute_confidence: bool = False
            ):
        self.__n_views = input_data.n_views()
        self.__n_predictive_features = input_data.n_predictive_features()
        self.__n_workers = n_workers
        self.__workers_pool = None
        self.__folds_list = folds_list
        self.__worker_state = WorkerState(
            input_data=input_data.serialize(),
            folds_list=self.__folds_list,
            hp_manager=hp_manager,
            objectives=objectives,
            seed=seed,
            printer=workers_printer,
            compute_feature_importance=compute_feature_importance,
            compute_confidence=compute_confidence
        )

    def individual_size(self):
        """Actual number of attributes of an individual. May be different from the number of selected features.
        Used when creating a new individual to generate its attributes."""
        raise NotImplementedError()

    def n_predictive_features(self):
        return self.__n_predictive_features

    def n_views(self):
        return self.__n_views

    def hp_manager(self) -> MvHyperparamManager:
        return self.__worker_state.hp_manager()

    def evaluate_batch(
            self, individuals,
            verbose=False, force_sequential=False) -> Sequence[MOCVResult]:
        if verbose:
            print("Evaluating batch with num workers: " + str(self.__n_workers))
        if self.__n_workers > 1 and not force_sequential:
            return self.evaluate_batch_parallel_with_init(individuals)
        else:
            return self.evaluate_batch_sequential(individuals)

    def evaluate_batch_sequential(self, individuals) -> Sequence[MOCVResult]:
        self.__worker_state.compile()
        res = []
        for i in individuals:
            res.append(self.__worker_state.evaluate(self.pack_individual(i)))
        return res

    def evaluate_batch_parallel_with_init(self, individuals: Sequence) -> Sequence[MOCVResult]:
        self.init_workers_pool_if_needed()
        pop_size = len(individuals)
        packed_pop = self.pack_pop(individuals)
        individuals = None  # Help GC
        func = multiprocessing_friendly_evaluation_with_init
        try:
            res = self.__workers_pool.map(
                func=func, iterable=packed_pop, chunksize=1)
        except BaseException as e:
            msg = ""
            msg += "Original exception: " + str(e) + "\n"
            msg += "while map-reducing from a packed population of size " + str(pop_size) + "\n"
            msg += "Showing first elements:\n"
            fe = islice(packed_pop, min(3, pop_size))
            for f in fe:
                msg += str(f) + "\n"
            self.cleanup()
            raise Exception(msg)
        return res

    def init_workers_pool_if_needed(self, verbose=False):
        if self.__workers_pool is None:
            if verbose:
                print("Initializing workers")
                print("Workers state:")
                print(self.__worker_state)
            ctx = multiprocessing.get_context('spawn')
            self.__workers_pool = ctx.Pool(
                processes=self.__n_workers, initializer=parallel_init, initargs=(self.__worker_state,))
            if verbose:
                print("Initialized workers")

    # Call this at the end of a GA to terminate the workers.
    def cleanup(self):
        if self.__workers_pool is not None:
            self.__workers_pool.close()
            self.__workers_pool.join()
        self.__workers_pool = None

    def __del__(self):
        self.cleanup()

    # Override to provide a custom packaging for interprocess message passing.
    @staticmethod
    def pack_individual(
            individual: PeculiarIndividualByListlike) -> WorkPackage:
        return WorkPackage(individual=individual)

    def pack_pop(self, pop: Iterable) -> Iterator[WorkPackage]:
        return (self.pack_individual(i) for i in pop)
