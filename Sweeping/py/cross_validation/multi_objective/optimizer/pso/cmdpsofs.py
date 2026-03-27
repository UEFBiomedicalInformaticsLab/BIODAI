from collections.abc import Iterable
from typing import Sequence

from cross_validation.multi_objective.optimizer.ga_str_utils import name_paste, pop_name, gen_name, nick_paste, \
    pop_nick, gen_nick
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer import hofs_to_results
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult
from cross_validation.multi_objective.optimizer.pso.leader_set import LeaderSet
from cross_validation.multi_objective.optimizer.pso.swarm_utils import init_swarm
from evaluator.individual_updater import IndividualUpdater
from evaluator.theta_evaluator import ThetaEvaluator
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from ga_runner.ga_progress_observer import SmartGAProgressObserver
from hall_of_fame.hof_consts import DEFAULT_HOFS
from hall_of_fame.population_observer_factory import HallOfFameFactory
from individual.individual_with_context import IndividualWithContext
from individual.num_features import NumFeatures
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.distribution.distribution import Distribution
from util.hyperbox.hyperbox import Interval, ConcreteInterval
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed
from util.str_utils import name_value

CMDPSOFS_NICK = "CMDPSOFS"

CMDPSOFS_NAME = "CMDPSOFS multi-view multi-objective optimizer"

CMDPSOFS_TYPE = ConcreteMOOptimizerType(
        uses_inner_models=True, nick=CMDPSOFS_NICK, name=CMDPSOFS_NAME)

DEFAULT_THETA = 0.6
# From "Particle Swarm Optimization for Feature Selection in Classification: A Multi-Objective Approach"

DEFAULT_V_MAX = 0.6
# From "Particle Swarm Optimization for Feature Selection in Classification: A Multi-Objective Approach"

DEFAULT_W = ConcreteInterval(a=0.1, b=0.5)
# From "Particle Swarm Optimization for Feature Selection in Classification: A Multi-Objective Approach"

DEFAULT_C1 = ConcreteInterval(a=1.5, b=2.0)
# From "Particle Swarm Optimization for Feature Selection in Classification: A Multi-Objective Approach"

DEFAULT_C2 = DEFAULT_C1
# From "Particle Swarm Optimization for Feature Selection in Classification: A Multi-Objective Approach"


class CMDPSOFS(MultiObjectiveOptimizerAcceptingFeatureImportance):
    __pop_size: int
    __n_gen: int
    __theta: float
    __v_max: float
    __w: Interval
    __c1: Interval
    __c2: Interval
    __objectives: Sequence[PersonalObjectiveWithImportance]
    __hof_factories: Iterable[HallOfFameFactory]
    __nick: str
    __name: str
    __folds_creator: InputDataFoldsCreator
    __initial_features: NumFeatures

    def __init__(self, pop_size, n_gen,
                 initial_features: NumFeatures,
                 folds_creator: InputDataFoldsCreator,
                 objectives: Iterable[PersonalObjectiveWithImportance],
                 theta: float = DEFAULT_THETA,
                 v_max: float = DEFAULT_V_MAX,
                 w: Interval = DEFAULT_W,
                 c1: Interval = DEFAULT_C1,
                 c2: Interval = DEFAULT_C2,
                 hof_factories: Iterable[HallOfFameFactory] = DEFAULT_HOFS):
        self.__pop_size = pop_size
        self.__n_gen = n_gen
        self.__folds_creator = folds_creator
        self.__objectives = list(objectives)
        self.__theta = theta
        self.__hof_factories = hof_factories
        self.__v_max = v_max
        self.__w = w
        self.__c1 = c1
        self.__c2 = c2
        self.__initial_features = initial_features
        self.__nick = nick_paste(parts=[CMDPSOFS_NICK,
                                 folds_creator.nick(),
                                 pop_nick(pop_size=pop_size),
                                 initial_features.nick(),
                                 gen_nick(n_gen=n_gen),
                                 "theta" + str(self.__theta),
                                 "v_max" + str(self.__v_max),
                                 "w" + str(self.__w),
                                 "c1" + str(self.__c1),
                                 "c2" + str(self.__c2)]) + ")"
        self.__name = CMDPSOFS_NAME + " (" + name_paste(parts=[
                      folds_creator.name(),
                      pop_name(pop_size=pop_size),
                      initial_features.name(),
                      gen_name(n_gen=n_gen),
                      "theta " + str(self.__theta),
                      "vel max " + str(self.__v_max),
                      "w " + str(self.__w),
                      "c1 " + str(self.__c1),
                      "c2 " + str(self.__c2)]) + ")"

    def optimizer_type(self) -> MOOptimizerType:
        return CMDPSOFS_TYPE

    def optimize_with_feature_importance(self, input_data: InputData, printer: Printer,
                                         feature_importance: dict[str,Distribution],
                                         n_proc: int = 1,
                                         workers_printer=UnbufferedOutPrinter(),
                                         logbook_saver: LogbookSaver = DummyLogbookSaver(),
                                         feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
                                         known_solutions: Sequence[IndividualWithContext] = ()
                                         ) -> Sequence[MultiObjectiveOptimizerResult]:

        # Make sure we do not include outcomes (potentially affecting stratification) that are not in objectives.
        input_data = select_outcomes_in_objectives(input_data=input_data, objectives=self.__objectives)

        collapsed_views = input_data.collapsed_views()

        printer.title_print("Creating inner folds")
        folds_list = self.__folds_creator.create_folds_from_input_data(
            input_data=input_data, seed=random_seed(), printer=printer)

        progress_observers = [SmartGAProgressObserver(printer=printer)]

        printer.title_print("Initializing swarm")
        swarm = init_swarm(
            n_objectives=len(self.__objectives), n_features=collapsed_views.n_col(),
            initial_features=self.__initial_features, pop_size=self.__pop_size,
            max_velocity=self.__v_max, w=self.__w, c1=self.__c1, c2=self.__c2, theta=self.__theta)

        evaluator = ThetaEvaluator(
            input_data=input_data, folds_list=folds_list,
            objectives=self.__objectives,
            theta=self.__theta,
            n_workers=n_proc,
            workers_printer=workers_printer,
            compute_feature_importance=False,
            seed=random_seed(),
            compute_confidence=True)
        hp_manager = evaluator.hp_manager()
        individual_updater = IndividualUpdater(evaluator=evaluator, objectives=self.__objectives)

        individual_updater.eval_invalid(pop=swarm)

        for o in progress_observers:
            o.notify_initial_pop(swarm)

        printer.title_print("Initializing leader set")
        leader_set = LeaderSet(self.__pop_size)
        leader_set.update(swarm)

        hofs = [h.create_population_observer() for h in self.__hof_factories]

        for h in hofs:
            h.update(new_elems=hp_manager.contextualize_all(pop=swarm))

        printer.title_print("Running iterations")
        for i in range(self.__n_gen):
            for p in swarm:
                g_best = leader_set.tournament_select()
                p.update_cinematic(g_best=g_best, completion=(i/self.__n_gen))
                # Perturbation with distribution N(0, 0.1) as in
                # "A multi-objective algorithm based upon particle swarm optimisation, an efficient data structure and
                # turbulence"
            individual_updater.eval_invalid(pop=swarm)
            leader_set.update(new_elems=swarm)
            for h in hofs:
                h.update(new_elems=hp_manager.contextualize_all(pop=swarm))
            for po in progress_observers:
                po.notify_generation_end(gen=i)

        for h in hofs:
            h.signal_final(final_elems=hp_manager.contextualize_all(pop=swarm))

        printer.title_print("Packing swarm optimization results")
        return hofs_to_results(hofs)

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    def __str__(self) -> str:
        res = ""
        res += name_value("Name", self.name()) + "\n"
        res += name_value("Nick", self.nick()) + "\n"
        res += name_value("Population size", self.__pop_size) + "\n"
        res += name_value("Number of features in initial individuals", self.__initial_features) + "\n"
        res += name_value("Number of generations", self.__n_gen) + "\n"
        res += name_value("Theta", self.__theta) + "\n"
        res += name_value("Max velocity", self.__v_max) + "\n"
        res += name_value("Inertia", self.__w) + "\n"
        res += name_value("C1", self.__c1) + "\n"
        res += name_value("C2", self.__c2) + "\n"
        res += name_value("Objectives", self.__objectives) + "\n"
        res += name_value("Hall of fame factories", self.__hof_factories) + "\n"
        res += name_value("Folds creator", self.__folds_creator) + "\n"
        return res
