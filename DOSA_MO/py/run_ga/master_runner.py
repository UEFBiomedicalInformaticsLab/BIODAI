import random
from abc import ABC, abstractmethod
from collections.abc import Sequence, Iterable

from joblib.numpy_pickle_utils import xrange

from ga_components.bitlist_mutation import BitlistMutation, FlipMutation
from ga_components.sorter.sorting_strategy import SortingStrategy
from ga_runner.flip_ga_runner import FlipGARunner
from ga_runner.ga_runner import GAResult
from ga_runner.progress_observer import SmartProgressObserver
from ga_runner.select_ga_runner import SelectGARunner
from hall_of_fame.hall_of_fame import HallOfFame
from hyperparam_manager.select_hp_manager import SelectHpManager
from individual.num_features import NumFeatures
from individual.peculiar_individual import PeculiarIndividual
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.distribution.random_distribution import random_distribution
from util.named import Named
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed
from util.sparse_bool_list_by_set import SparseBoolListBySet


RESAMPLED_SWEEPING_TYPE_NICK = "Sweeping"
CONCATENATED_SWEEPING_TYPE_NICK = "CSweeping"
LEAN_CONCATENATED_SWEEPING_TYPE_NICK = "LCSweeping"


def view_pops_from_resampling(
        master_pop: Sequence[PeculiarIndividual],
        view_pops: Sequence[Sequence[PeculiarIndividual]]) -> Sequence[Sequence[PeculiarIndividual]]:
    n_views = len(view_pops)
    res_pops = [[] for _ in xrange(n_views)]
    for m_ind in master_pop:
        for view_i in range(n_views):
            m_ind_pointer = m_ind[view_i]
            res_pops[view_i].append(view_pops[view_i][m_ind_pointer])
    return res_pops


def single_views_to_concatenated_solutions(
        view_pops: Sequence[Sequence[PeculiarIndividual]]) -> list[PeculiarIndividual]:
    """Solutions of each view are permuted randomly in a new list,
    then the ones at the same position are concatenated. Sparse Bool lists are used for performance.
    It is assumed that the view_pops have the views in the same order as the input data.
    Calls module random."""
    n_views = len(view_pops)
    n_solutions = len(view_pops[0])
    n_objectives = view_pops[0][0].n_objectives()
    permuted_sv = [random.sample(sv, n_solutions) for sv in view_pops]
    res = []
    for i in range(n_solutions):
        mask_i = SparseBoolListBySet()
        for j in range(n_views):
            mask_i.extend(permuted_sv[j][i])
        res.append(PeculiarIndividualSparse(seq=mask_i, n_objectives=n_objectives))
    return res


def single_views_to_lean_concatenated_solutions(
        view_pops: Sequence[Sequence[PeculiarIndividual]]) -> list[PeculiarIndividual]:
    """Solutions of each view are permuted randomly in a new list,
    then the ones at the same position are concatenated after filtering them randomly.
    Sparse Bool lists are used for performance.
    It is assumed that the view_pops have the views in the same order as the input data.
    Calls module random."""
    n_views = len(view_pops)
    tot_features = [len(v[0]) for v in view_pops]
    n_solutions = len(view_pops[0])
    n_objectives = view_pops[0][0].n_objectives()
    permuted_sv = [random.sample(sv, n_solutions) for sv in view_pops]
    res = []
    for i in range(n_solutions):
        mask_i = SparseBoolListBySet()
        view_dist = random_distribution(n_values=n_views)
        for j in range(n_views):
            mask_ij = SparseBoolListBySet(min_size=tot_features[j])
            true_positions_ji = permuted_sv[j][i].true_positions()
            for k in range(len(true_positions_ji)):
                if random.random() < view_dist[j]:
                    mask_ij.set_true(true_positions_ji[k])
            mask_i.extend(mask_ij)
        res.append(PeculiarIndividualSparse(seq=mask_i, n_objectives=n_objectives))
    return res


def view_pops_from_concatenated(
        master_pop: Sequence[PeculiarIndividual], features_per_view: Sequence[int]) -> list[list[PeculiarIndividual]]:
    n_objectives = master_pop[0].n_objectives()
    n_views = len(features_per_view)
    res_lists = [[] for _ in range(n_views)]
    for s in master_pop:
        start = 0
        for i in range(n_views):
            end = start + features_per_view[i]
            current_list = res_lists[i]
            current_list.append(PeculiarIndividualSparse(seq=s[start:end], n_objectives=n_objectives))
            start = end
    return res_lists


class MasterRunner(Named, ABC):

    @abstractmethod
    def run_master(
            self,
            input_data: InputData,
            view_pops: Sequence[Sequence[PeculiarIndividual]],
            pop_size: int,
            mutating_prob: float,
            mating_prob: float,
            objectives: Iterable[PersonalObjective],
            sorting_strategy: SortingStrategy,
            folds_list,
            n_gen: int,
            result_hofs: Sequence[HallOfFame],
            initial_features: NumFeatures,
            printer: Printer,
            bitlist_mutation: BitlistMutation = FlipMutation(),
            use_clone_repurposing: bool = False,
            workers_printer: Printer = UnbufferedOutPrinter(),
            return_history: bool = False,
            n_workers: int = 1) -> tuple[GAResult, Sequence[Sequence[PeculiarIndividual]]]:
        raise NotImplementedError()

    @abstractmethod
    def sweeping_type_nick(self) -> str:
        raise NotImplementedError()


class ResamplingMaster(MasterRunner):

    def run_master(self, input_data: InputData, view_pops: Sequence[Sequence[PeculiarIndividual]], pop_size: int,
                   mutating_prob: float, mating_prob: float, objectives: Iterable[PersonalObjective],
                   sorting_strategy: SortingStrategy, folds_list, n_gen: int, result_hofs: Sequence[HallOfFame],
                   initial_features: NumFeatures, printer: Printer, bitlist_mutation: BitlistMutation = FlipMutation(),
                   use_clone_repurposing: bool = False,
                   workers_printer: Printer = UnbufferedOutPrinter(), return_history: bool = False,
                   n_workers: int = 1) -> tuple[GAResult, Sequence[Sequence[PeculiarIndividual]]]:
        """Uses module random."""
        select_hp_manager = SelectHpManager(view_pops=view_pops)
        master_runner = SelectGARunner(
            pop_size=pop_size, mating_prob=mating_prob, mutation_frequency=mutating_prob,
            hp_manager=select_hp_manager, objectives=objectives,
            sorting_strategy=sorting_strategy, use_clone_repurposing=use_clone_repurposing)
        master_result = master_runner.run(
            input_data=input_data, folds_list=folds_list, n_gen=n_gen, seed=random_seed(), n_workers=n_workers,
            hofs=result_hofs,
            return_history=return_history, workers_printer=workers_printer,
            progress_observers=[SmartProgressObserver(printer=printer)])
        view_pops = view_pops_from_resampling(master_pop=master_result.pop, view_pops=view_pops)
        # TODO: perhaps we can use the HOF instead of the final population.
        return master_result, view_pops

    def sweeping_type_nick(self) -> str:
        return RESAMPLED_SWEEPING_TYPE_NICK

    def name(self) -> str:
        return "resampling"


class MasterWithConcatenation(MasterRunner, ABC):
    __lean: bool

    def __init__(self, lean: bool):
        self.__lean = lean

    def run_master(self, input_data: InputData, view_pops: Sequence[Sequence[PeculiarIndividual]], pop_size: int,
                   mutating_prob: float, mating_prob: float, objectives: Iterable[PersonalObjective],
                   sorting_strategy: SortingStrategy, folds_list, n_gen: int, result_hofs: Sequence[HallOfFame],
                   initial_features: NumFeatures, printer: Printer, bitlist_mutation: BitlistMutation = FlipMutation(),
                   use_clone_repurposing: bool = False,
                   workers_printer: Printer = UnbufferedOutPrinter(), return_history: bool = False,
                   n_workers: int = 1) -> tuple[GAResult, Sequence[Sequence[PeculiarIndividual]]]:
        """Uses module random."""
        printer.title_print(self.name() + " with " + bitlist_mutation.name())
        master_runner = FlipGARunner(
            pop_size=pop_size,
            mating_prob=mating_prob,
            mutation_frequency=mutating_prob,
            initial_features=initial_features,
            objectives=objectives,
            sorting_strategy=sorting_strategy,
            mutation=bitlist_mutation,
            use_clone_repurposing=use_clone_repurposing)
        if self.__lean:
            master_pop = single_views_to_lean_concatenated_solutions(view_pops=view_pops)
        else:
            master_pop = single_views_to_concatenated_solutions(view_pops=view_pops)
        master_result = master_runner.run(
            input_data=input_data, folds_list=folds_list, n_gen=n_gen, seed=random_seed(), n_workers=n_workers,
            return_history=return_history,
            initial_pop=master_pop,
            workers_printer=workers_printer,
            hofs=result_hofs,
            progress_observers=[SmartProgressObserver(printer=printer)])
        view_pops = view_pops_from_concatenated(
            master_pop=master_result.pop, features_per_view=input_data.n_features_per_view())
        return master_result, view_pops


class FatConcatenatedMaster(MasterWithConcatenation):

    def __init__(self):
        MasterWithConcatenation.__init__(self=self, lean=False)

    def sweeping_type_nick(self) -> str:
        return CONCATENATED_SWEEPING_TYPE_NICK

    def name(self) -> str:
        return "concatenated master"


class LeanConcatenatedMaster(MasterWithConcatenation):

    def __init__(self):
        MasterWithConcatenation.__init__(self=self, lean=True)

    def sweeping_type_nick(self) -> str:
        return LEAN_CONCATENATED_SWEEPING_TYPE_NICK

    def name(self) -> str:
        return "lean concatenated master"