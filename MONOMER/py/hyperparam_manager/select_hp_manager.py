from individual.peculiar_individual_by_listlike import PeculiarIndividualByListlike
from hyperparam_manager.hyperparam_manager import HyperparamManager

from util import sparse_bool_list_by_set
from util.list_like import ListLike
from util.list_math import indices_of_true
from util.sparse_bool_list_by_set import SparseBoolListBySet


class SelectHpManager(HyperparamManager):
    __active_features_mask_len: int

    def __init__(self, view_pops):
        self.__view_pops = view_pops
        # TODO We could have HpManagers also for the single view individuals, one manager per view.
        self.__n_view_individuals = []
        for p in view_pops:
            self.__n_view_individuals.append(len(p))

        le = 0
        for i in range(0, len(view_pops)):
            view_individual = self.__view_pops[i][0]
            le += len(view_individual)
        self.__active_features_mask_len = le

    def n_active_features(self, hyperparams):
        res = 0
        for i in range(0, len(hyperparams)):
            try:
                res += self.__view_pops[i][hyperparams[i]].sum()
            except IndexError as e:
                print("IndexError exception caught inside n_active_features")
                print("i: " + str(i) + "\n" + "pop: " + str(self.__view_pops) + "\n" + "hyperparams: " + str(hyperparams) + "\n")
                raise e
        return res

    def active_features_mask(self, hyperparams: ListLike, verbose=False) -> SparseBoolListBySet:
        if verbose:
            print("Executing active_features_mask")
            print("master individual: " + str(hyperparams))
        view_individuals = []
        for i in range(0, len(hyperparams)):
            view_individual = self.__view_pops[i][hyperparams[i]]
            if verbose:
                print("view individual: " + str(indices_of_true(view_individual)))
            view_individuals.append(view_individual)
        mask = sparse_bool_list_by_set.chain(view_individuals)
        if verbose:
            print("resulting mask: " + str(indices_of_true(mask)))
        return mask

    def active_features_mask_len(self, hyperparams: PeculiarIndividualByListlike) -> int:
        return self.__active_features_mask_len

    def active_features_mask_numpy(self, hyperparams: PeculiarIndividualByListlike):
        return self.active_features_mask(hyperparams).to_numpy()

    def max_view_individual_index(self, pos):
        return self.__n_view_individuals[pos]-1

    def __str__(self):
        ret_string = "SelectHpManager object with attributes:\n"
        ret_string += "__n_view_individuals: "
        ret_string += str(self.__n_view_individuals) + "\n"
        return ret_string
