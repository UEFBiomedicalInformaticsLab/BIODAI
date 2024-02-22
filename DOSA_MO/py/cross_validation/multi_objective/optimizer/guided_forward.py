from __future__ import annotations

import os
from collections.abc import Iterable
from copy import copy
from typing import Sequence

import pandas as pd

from cross_validation.multi_objective.cross_evaluator.hof_saver import solution_features_file_name
from location_manager.location_consts import HOFS_STR
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import MultiObjectiveOptimizer, \
    hofs_to_results
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from evaluator.individual_updater import IndividualUpdater
from evaluator.mask_evaluator import MaskEvaluator
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.fronts import PARETO_NICK
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.peculiar_individual_sparse import PeculiarIndividualSparse
from individual.peculiar_individual_with_context import contextualize_all
from input_data.input_data import InputData
from multi_view_utils import collapse_feature_importance
from objective.social_objective import PersonalObjective
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed
from util.sequence_utils import sort_permutation
from util.sparse_bool_list_by_set import SparseBoolListBySet
from util.utils import name_value
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import MultiObjectiveOptimizerByFold

GUIDED_FORWARD_OPTIMIZER_TYPE = ConcreteMOOptimizerType(
        uses_inner_models=True, nick="guided_forward", name="Guided Forward")


class GuidedForward(MultiObjectiveOptimizer):
    __optimizer_type = GUIDED_FORWARD_OPTIMIZER_TYPE

    __feature_queue: Sequence[str]
    __feature_queue_nick: str
    __hof_factories: Iterable[HallOfFameFactory]
    __folds_creator: InputDataFoldsCreator
    __objectives: Sequence[PersonalObjective]

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

    def __init__(self,
                 feature_queue: Sequence[str],
                 folds_creator: InputDataFoldsCreator,
                 objectives: Sequence[PersonalObjective],
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
                 feature_queue_nick: str = "unnamed_feature_queue"):
        self.__feature_queue = feature_queue
        self.__feature_queue_nick = feature_queue_nick
        self.__hof_factories = hof_factories
        self.__folds_creator = folds_creator
        self.__objectives = objectives

    @staticmethod
    def feature_queue_nick(previous_optimizer_dir: str, hof_nick: str = PARETO_NICK):
        if not isinstance(previous_optimizer_dir, str):
            raise ValueError("previous_optimizer_dir is not str: " + str(previous_optimizer_dir))
        if not isinstance(hof_nick, str):
            raise ValueError("hof_nick is not str: " + str(hof_nick))
        last_dir_name = os.path.basename(os.path.normpath(previous_optimizer_dir))
        return last_dir_name + "_" + hof_nick

    @staticmethod
    def create_individual(features: Sequence[str],
                          all_feature_names: Sequence[str],
                          individual_updater: IndividualUpdater) -> PeculiarIndividualSparse:
        """Starting from a sequence of feature names we create an individual to be evaluated."""
        res_list = SparseBoolListBySet(min_size=individual_updater.n_features())
        all_feature_names = list(all_feature_names)
        for f in features:
            index = all_feature_names.index(f)
            res_list[index] = True
        return PeculiarIndividualSparse(seq=res_list, n_objectives=individual_updater.n_objectives())

    def optimize(self, input_data: InputData, printer: Printer, n_proc=1, workers_printer=UnbufferedOutPrinter(),
                 logbook_saver: LogbookSaver = DummyLogbookSaver(),
                 feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver()
                 ) -> Sequence[MultiObjectiveOptimizerResult]:
        all_feature_names = input_data.collapsed_feature_names()
        inner_folds_list = self.__folds_creator.create_folds_from_input_data(
            input_data=input_data, seed=random_seed(), printer=printer)
        individual_updater = IndividualUpdater(
            evaluator=MaskEvaluator(
                input_data=input_data,
                folds_list=inner_folds_list,
                objectives=self.__objectives,
                n_workers=n_proc,
                workers_printer=workers_printer,
                seed=random_seed(),
                compute_confidence=True),
            objectives=self.__objectives)
        growing_features = []
        pop = []
        # We start by creating the zero-features individual.
        ind = self.create_individual(
            features=growing_features, all_feature_names=all_feature_names, individual_updater=individual_updater)
        pop.append(ind)
        # Then all the others adding a feature at a time.
        for feature_name in self.__feature_queue:
            growing_features.append(feature_name)
            ind = self.create_individual(
                features=growing_features, all_feature_names=all_feature_names, individual_updater=individual_updater)
            pop.append(ind)
        individual_updater.eval_invalid(pop=pop)
        pop = contextualize_all(hps=pop, hp_manager=DummyHpManager())
        hofs = [h.create_population_observer() for h in self.__hof_factories]
        for h in hofs:
            h.update(new_elems=pop)
        return hofs_to_results(hofs)

    def nick(self) -> str:
        return self.__feature_queue_nick + "_" + self.optimizer_type().nick()

    @staticmethod
    def feature_queue_name(previous_optimizer_dir: str, hof_nick: str = PARETO_NICK):
        last_dir_name = os.path.basename(os.path.normpath(previous_optimizer_dir))
        return last_dir_name + " " + hof_nick

    @staticmethod
    def create_from_previous_optimizer_dir(
            previous_optimizer_dir: str,
            objectives: Sequence[PersonalObjective],
            outer_fold_index: int,
            folds_creator: InputDataFoldsCreator,
            hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
            hof_nick: str = PARETO_NICK,
            ) -> GuidedForward:
        if os.path.isdir(previous_optimizer_dir):
            hofs_dir = os.path.join(previous_optimizer_dir, HOFS_STR, hof_nick)
            if os.path.isdir(hofs_dir):
                feature_queue_nick = GuidedForward.feature_queue_nick(
                    previous_optimizer_dir=previous_optimizer_dir, hof_nick=hof_nick)
                fold_hof_file_name = solution_features_file_name(fold_index=outer_fold_index)
                hof_df = pd.read_csv(filepath_or_buffer=os.path.join(hofs_dir, fold_hof_file_name))
                s = hof_df.sum()
                hof_df = hof_df[s.sort_values(ascending=False).index]
                feature_queue = hof_df.columns
                return GuidedForward(
                    feature_queue=feature_queue,
                    objectives=objectives,
                    folds_creator=folds_creator,
                    hof_factories=hof_factories,
                    feature_queue_nick=feature_queue_nick)
            else:
                raise ValueError("hof nicks are not directories: " + str(hof_nick))
        else:
            raise ValueError("previous_optimizer_dir is not a directory: " + str(previous_optimizer_dir))

    @staticmethod
    def create_from_weights(
            weights: Sequence[float],
            feature_names: Sequence[str],
            objectives: Sequence[PersonalObjective],
            folds_creator: InputDataFoldsCreator,
            hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
            feature_queue_nick: str = "feature_queue_from_weights",
    ) -> GuidedForward:
        """Zero weights are not considered."""
        if len(weights) != len(feature_names):
            raise ValueError()
        weights = copy(weights)
        sorting = sort_permutation(weights, reverse=True)
        non_zero_sorting = []
        for i in sorting:
            if weights[i] > 0.0:
                non_zero_sorting.append(i)
        feature_queue = [feature_names[i] for i in non_zero_sorting]
        return GuidedForward(
            feature_queue=feature_queue,
            objectives=objectives,
            folds_creator=folds_creator,
            hof_factories=hof_factories,
            feature_queue_nick=feature_queue_nick)

    @staticmethod
    def create_from_fi(
            feature_importance: Sequence[Distribution],
            input_data: InputData,
            objectives: Sequence[PersonalObjective],
            folds_creator: InputDataFoldsCreator,
            hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
            feature_queue_nick: str = "feature_queue_from_fi"
    ) -> GuidedForward:
        feature_names = input_data.collapsed_feature_names()
        weights = collapse_feature_importance(distributions=feature_importance)
        return GuidedForward.create_from_weights(
            weights=weights,
            feature_names=feature_names,
            objectives=objectives,
            folds_creator=folds_creator,
            hof_factories=hof_factories,
            feature_queue_nick=feature_queue_nick)


class GuidedForwardByFold(MultiObjectiveOptimizerByFold):
    __objectives: Sequence[PersonalObjective]
    __inner_folds_creator: InputDataFoldsCreator
    __previous_optimizer_dir: str
    __previous_hof_nick: str
    __hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),)

    def __init__(self,
                 objectives: Sequence[PersonalObjective],
                 inner_folds_creator: InputDataFoldsCreator,
                 previous_optimizer_dir: str,
                 previous_hof_nick: str = PARETO_NICK,
                 hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),)
                 ):
        self.__objectives = objectives
        self.__inner_folds_creator = inner_folds_creator
        self.__previous_optimizer_dir = previous_optimizer_dir
        self.__previous_hof_nick = previous_hof_nick
        self.__hof_factories = hof_factories

    def optimizer_for_fold(self, fold_index: int) -> MultiObjectiveOptimizer:
        return GuidedForward.create_from_previous_optimizer_dir(
            previous_optimizer_dir=self.__previous_optimizer_dir,
            objectives=self.__objectives,
            outer_fold_index=fold_index,
            folds_creator=self.__inner_folds_creator,
            hof_nick=self.__previous_hof_nick,
            hof_factories=self.__hof_factories
        )

    def nick(self) -> str:
        feature_queue_nick = GuidedForward.feature_queue_nick(
            previous_optimizer_dir=self.__previous_optimizer_dir, hof_nick=self.__previous_hof_nick)
        return feature_queue_nick + "_" + GUIDED_FORWARD_OPTIMIZER_TYPE.nick()

    def name(self) -> str:
        feature_queue_name = GuidedForward.feature_queue_name(
            previous_optimizer_dir=self.__previous_optimizer_dir, hof_nick=self.__previous_hof_nick)
        return feature_queue_name + " " + GUIDED_FORWARD_OPTIMIZER_TYPE.name()

    def __str__(self) -> str:
        res = ""
        res += name_value("Directory of previous optimizer", self.__previous_optimizer_dir) + "\n"
        res += name_value("Previous hall of fame nick", self.__previous_hof_nick) + "\n"
        res += name_value("Objectives", self.__objectives) + "\n"
        res += name_value("Inner folds creator", self.__inner_folds_creator) + "\n"
        res += name_value("Hall of fame factories", self.__hof_factories) + "\n"
        return res

    def optimizer_for_all_data(self) -> MultiObjectiveOptimizer:
        raise NotImplementedError("Not yet")

    def uses_inner_models(self):
        return True
