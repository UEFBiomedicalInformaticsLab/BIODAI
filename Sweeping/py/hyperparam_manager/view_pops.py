from collections.abc import Sequence

from individual.fit_individual import FitIndividual
from util import sparse_bool_list_by_set
from util.list_like import ListLike
from util.sparse_bool_list_by_set import SparseBoolListBySet


class ViewPops:
    __view_pops: Sequence[Sequence[FitIndividual]]
    """Each ListLike is a hyperparam sequence."""
    __n_view_individuals: list[int]
    """Number of individuals for each view."""
    __tot_hyperparams: int
    """Sum of the number of existing hyperparams of every view."""

    def __init__(self, view_pops: Sequence[Sequence[FitIndividual]]):
        """The external sequence is for the view. Each view has a sequence of individuals."""
        self.__view_pops = view_pops
        self.__n_view_individuals = []
        for p in view_pops:
            self.__n_view_individuals.append(len(p))
        le = 0
        for i in range(0, len(view_pops)):
            view_individual = self.__view_pops[i][0]
            le += len(view_individual)
        self.__tot_hyperparams = le

    def get_individual(self, view: int, pos: int) -> FitIndividual:
        return self.__view_pops[view][pos]

    def n_predictive_features(self, hyperparams: ListLike) -> int:
        res = 0
        for i in range(0, len(hyperparams)):
            try:
                res += self.__view_pops[i][hyperparams[i]].sum()
            except IndexError as e:
                print("IndexError exception caught inside n_predictive_features")
                print("i: " + str(i) + "\n" + "pop: " + str(self.__view_pops) + "\n" + "hyperparams: " + str(hyperparams) + "\n")
                raise e
        return res

    def predictive_features_mask(self, hyperparams: ListLike, verbose=False) -> SparseBoolListBySet:
        if verbose:
            print("Executing predictive_features_mask")
            print("master individual: " + str(hyperparams))
        view_individuals = []
        for i in range(0, len(hyperparams)):
            view_individual = self.__view_pops[i][hyperparams[i]]
            if verbose:
                print("view individual: " + str(view_individual.true_positions()))
            view_individuals.append(view_individual)
        mask = sparse_bool_list_by_set.chain(view_individuals)
        if verbose:
            print("resulting mask: " + str(mask.true_positions()))
        return mask

    def tot_hyperparams(self) -> int:
        return self.__tot_hyperparams

    def view_hyperparams(self, view_pos: int) -> int:
        """Choosable hyperparams for the selected view."""
        return len(self.__view_pops[view_pos][0])

    def max_view_individual_index(self, view_pos: int):
        return self.view_size(view_pos=view_pos)-1

    def __str__(self):
        return "n_view_individuals: " + str(self.__n_view_individuals) + "\n"

    def view_individuals(self, hyperparams: ListLike) -> Sequence[FitIndividual]:
        return [self.__view_pops[i][hyperparams[i]] for i in range(0, len(hyperparams))]

    def all_individuals_for_view(self, view_pos: int) -> Sequence[FitIndividual]:
        return self.__view_pops[view_pos]

    def num_views(self) -> int:
        return len(self.__view_pops)

    def view_size(self, view_pos: int) -> int:
        return self.__n_view_individuals[view_pos]
