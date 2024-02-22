import time
import random
from collections.abc import Iterable
from math import sqrt
from typing import Sequence, NamedTuple

from cross_validation.cross_validation import evaluate_objective_for_fold_with_inner_cv, \
    evaluate_all_objectives_for_fold_with_inner_cv
from cross_validation.folds import Folds
from cross_validation.multi_objective.multi_objective_cross_validation import MultiObjectiveOptimizeOnFoldsSerial
from cross_validation.multi_objective.optimizer.ga_str_utils import nick_paste, name_paste
from cross_validation.multi_objective.optimizer.mo_optimizer_factory import MOOptimizerFactory
from cross_validation.multi_objective.optimizer.mo_optimizer_type import MOOptimizerType, ConcreteMOOptimizerType
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_result import MultiObjectiveOptimizerResult, \
    merge_mo_optimizer_results
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_by_fold import \
    DummyMultiObjectiveOptimizerByFold
from fitness_adjuster.fitness_adjuster import FitnessAdjuster
from fitness_adjuster.fitness_adjuster_input import FitnessAdjusterInput
from fitness_adjuster.fitness_adjuster_learner import FitnessAdjusterLearner
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from ga_components.feature_counts_saver import FeatureCountsSaver, DummyFeatureCountsSaver
from ga_components.logbook_saver import LogbookSaver, DummyLogbookSaver
from hall_of_fame.pareto_front import ParetoFront
from hall_of_fame.population_observer_factory import ParetoFrontFactory
from individual.fit_wrapper import FitWrapper
from individual.fitness.high_best_fitness import HighBestFitness
from individual.individual_with_context import IndividualWithContext
from input_data.input_data import InputData
from input_data.input_data_utils import select_outcomes_in_objectives
from model.regression.regressors_library import SVRegressor
from objective.objective_with_importance.adjusted_objective_computer import (ClassificationAdjustedObjectiveComputer,
                                                                             SurvivalAdjustedObjectiveComputer,
                                                                             StructuralAdjustedObjectiveComputer)
from objective.objective_with_importance.personal_objective_with_importance import PersonalObjectiveWithImportance
from util.cross_hypervolume.solution_weights_strategy import SolutionWeightsStrategy, SolutionDerivatives
from util.distribution.distribution import Distribution
from util.printer.printer import Printer, UnbufferedOutPrinter, NullPrinter
from util.randoms import random_seed, set_all_seeds
from util.sequence_utils import sequence_to_string, transpose
from util.utils import name_value, pretty_duration, PlannedUnreachableCodeError

TUNING_MULTIPLIER = 1.0  # Computational power to put in the tuning phase with respect to the main optimization phase.

TUNING_HOF = ParetoFrontFactory()
# HofUnionFactory(inner_factories=(ParetoFrontFactory(), LastPopFactory()))
# ParetoFrontFactory() LastPopFactory() ParticipantsFactory()

TUNING_HOFS = (TUNING_HOF,)
# It is more efficient to have just one hof here. Still the optimizer setup accepts a sequence. So a singleton.

TUNING_NICK = TUNING_HOF.create_population_observer().nick()
DEFAULT_ADJUSTER_REGRESSOR = SVRegressor()

BASE_NICK = "adj"


class ScaleParameters(NamedTuple):
    pop_size: int
    n_gen: int
    n_folds: int
    inner_n_folds: int


def tuning_parameters(main_parameters: ScaleParameters) -> ScaleParameters:
    main_n_folds = main_parameters.n_folds
    mult = TUNING_MULTIPLIER / float(main_n_folds-1)
    sqrt_mult = sqrt(mult)
    tuning_pop_size = int(main_parameters.pop_size * sqrt_mult)
    tuning_n_gen = int(main_parameters.n_gen * sqrt_mult)
    return ScaleParameters(
        pop_size=tuning_pop_size,
        n_gen=tuning_n_gen,
        n_folds=main_parameters.n_folds,
        inner_n_folds=main_parameters.inner_n_folds)


class AdjustedOptimizer(MultiObjectiveOptimizerAcceptingFeatureImportance):
    __tuning_optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance
    __main_optimizer_factory: MOOptimizerFactory
    __adjuster_learner: FitnessAdjusterLearner
    __tuning_folds_creator: InputDataFoldsCreator
    __objectives: Sequence[PersonalObjectiveWithImportance]
    __optimizer_type: MOOptimizerType
    __precog: bool
    __solution_weights_strategy: SolutionWeightsStrategy

    def __init__(self,
                 tuning_folds_creator: InputDataFoldsCreator,
                 objectives: Iterable[PersonalObjectiveWithImportance],
                 tuning_optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
                 adjuster_learner: FitnessAdjusterLearner,
                 main_optimizer_factory: MOOptimizerFactory,
                 precog: bool = False,
                 solution_weights_strategy: SolutionWeightsStrategy = SolutionDerivatives()):
        self.__tuning_folds_creator = tuning_folds_creator
        self.__objectives = list(objectives)
        self.__tuning_optimizer = tuning_optimizer
        self.__adjuster_learner = adjuster_learner
        self.__main_optimizer_factory = main_optimizer_factory
        self.__precog = precog
        self.__solution_weights_strategy = solution_weights_strategy
        self.__nick = nick_paste(parts=[
            BASE_NICK,
            self.__tuning_folds_creator.nick(),
            self.__tuning_optimizer.nick(),
            self.__adjuster_learner.nick(),
            self.__main_optimizer_factory.nick()])
        self.__name = "adjusted (" + name_paste(parts=[
            self.__tuning_folds_creator.name(),
            self.__tuning_optimizer.name(),
            self.__adjuster_learner.name(),
            self.__main_optimizer_factory.name()])
        uses_inner = tuning_optimizer.uses_inner_models() or main_optimizer_factory.uses_inner_models()
        self.__optimizer_type = ConcreteMOOptimizerType(
            uses_inner_models=uses_inner, nick=self.__nick, name=self.__name)

    def _train_adjuster_one_objective(
            self,
            tuning_hof: list[MultiObjectiveOptimizerResult],
            objective_index: int,
            input_data: InputData,
            folds_list: Sequence[Sequence[Sequence[int]]],
            solution_weights: Sequence[Sequence[float]],
            printer: Printer = NullPrinter()) -> FitnessAdjuster:
        """tuning_hof contains one MultiObjectiveOptimizerResult for each fold.
        solution_weights: one sequence of weights for each fold."""
        objective = self.__objectives[objective_index]
        original_fitnesses = []
        std_devs = []
        bootstrap_means = []
        n_features = []
        test_fitnesses = []
        sample_weights = []
        n_folds = len(folds_list)
        for fold_index in range(n_folds):
            fold_hof = tuning_hof[fold_index]
            fold = folds_list[fold_index]
            x_train, y_train, x_test, y_test = input_data.select_all_sets(train_indices=fold[0], test_indices=fold[1])
            # fold_hof is the tuning hof for one fold.
            all_fitnesses = evaluate_objective_for_fold_with_inner_cv(
                    fold_predictors_with_hyperparams=fold_hof,
                    objective=objective,
                    objective_index=objective_index,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    compute_ci=False)
            original_fitnesses.extend(all_fitnesses.inner_cv())
            std_devs.extend(all_fitnesses.inner_cv_sd())
            bootstrap_means.extend(all_fitnesses.inner_cv_bootstrap_mean())
            test_fitnesses.extend(all_fitnesses.test())
            n_features.extend([h.sum() for h in fold_hof.hyperparams()])
            sample_weights.extend(solution_weights[fold_index])
        inputs = [FitnessAdjusterInput(original_fitness=orig, std_dev=sd, num_features=n_feat, bootstrap_mean=bm)
                  for orig, sd, n_feat, bm in zip(original_fitnesses, std_devs, n_features, bootstrap_means)]
        printer.print("Training fitness adjuster regressor on " + str(len(test_fitnesses)) + " samples.")
        adjuster = self.__adjuster_learner.fit(inputs=inputs, test_fitness=test_fitnesses, sample_weight=sample_weights)
        printer.print(str(adjuster))
        return adjuster

    def optimize_with_feature_importance(self, input_data: InputData, printer: Printer,
                                         feature_importance: Sequence[Distribution], n_proc=1,
                                         workers_printer=UnbufferedOutPrinter(),
                                         logbook_saver: LogbookSaver = DummyLogbookSaver(),
                                         feature_counts_saver: FeatureCountsSaver = DummyFeatureCountsSaver(),
                                         known_solutions: Sequence[IndividualWithContext] = ()
                                         ) -> Sequence[MultiObjectiveOptimizerResult]:

        rand_state = random.getstate()
        # We want for the adjusted optimizer to run with the initial random state,
        # so it will be the same random state used by an unadjusted optimizer, and also will not depend on the
        # kind of adjusting regressor that is trained.
        # We are not doing the same with numpy.random because it is not clear if it is thread-safe.

        set_all_seeds(seed=random_seed())
        # To set a different random state than the one that will be used by the adjusted optimizer.

        # Make sure we do not include outcomes (potentially affecting stratification) that are not in objectives.
        input_data = select_outcomes_in_objectives(input_data=input_data, objectives=self.__objectives)

        printer.title_print("Creating folds for learning the fitness adjustment regression")
        folds_list = self.__tuning_folds_creator.create_folds_from_input_data(
            input_data=input_data, seed=random_seed(), printer=printer)
        folds = Folds(test_sets=[f[1] for f in folds_list])

        tuning_optimizer_by_fold = DummyMultiObjectiveOptimizerByFold(optimizer=self.__tuning_optimizer)

        start_time = time.time()
        printer.title_print(
            "Starting multi-objective optimization on different folds for learning the fitness adjustment regression")
        folds_optimizer = MultiObjectiveOptimizeOnFoldsSerial()
        folds_hofs = folds_optimizer.optimize(
            input_data=input_data, folds=folds, mo_optimizer_by_fold=tuning_optimizer_by_fold,
            save_path=None,
            n_proc=n_proc)
        printer.print(
            "Computation on different folds for learning the fitness adjustment regression finished in " +
            pretty_duration(time.time() - start_time))

        printer.print("Extracting hall of fames for tuning.")
        tuning_hof = None
        for hof in folds_hofs:
            hof_nick = hof[0].nick()
            if hof_nick == TUNING_NICK:
                tuning_hof = hof
        if tuning_hof is None:
            raise ValueError(
                "Missing tuning hall of fame " + TUNING_NICK + " in hall of fames " +
                sequence_to_string(li=[hof[0].nick() for hof in folds_hofs], brackets=False))
        # tuning_hof contains one element for each fold.
        if self.__precog:
            printer.print("Collecting for each fold the solutions of every considered hall of fame.")
            all_results = []  # List with a position for each fold.
            for hof in folds_hofs:
                for i, hof_fold in enumerate(hof):
                    if len(all_results) <= i:
                        all_results.append(hof_fold)
                    else:
                        all_results[i] = merge_mo_optimizer_results(results=(all_results[i], hof_fold))
            printer.print("Evaluating each solution on the left out set for its fold.")
            n_objectives = len(self.__objectives)
            precog_results = []
            for all_fold_results, fold in zip(all_results, folds_list):
                x_train, y_train, x_test, y_test = input_data.select_all_sets(train_indices=fold[0],
                                                                              test_indices=fold[1])
                fit_by_objective = evaluate_all_objectives_for_fold_with_inner_cv(
                    fold_predictors_with_hyperparams=all_fold_results,
                    objectives=self.__objectives,
                    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                    compute_ci=False)
                test_fold_results = []
                fold_hyperparams = all_fold_results.hyperparams()
                for i in range(len(fold_hyperparams)):
                    inner_solution = fold_hyperparams[i]
                    test_fitness = HighBestFitness(
                        n_objectives=n_objectives, values=[f.test()[i] for f in fit_by_objective])
                    wrapped_solution = FitWrapper(inner=inner_solution, fitness=test_fitness)
                    test_fold_results.append(wrapped_solution)
                precog_pareto = ParetoFront()
                precog_pareto.update(new_elems=test_fold_results)
                precog_fold_results = all_fold_results.select_individuals(
                    individuals=[h.unwrap() for h in precog_pareto.hofers()])
                printer.print("Fold precog results")
                printer.print(precog_fold_results)
                precog_results.append(precog_fold_results)
            for i in range(len(tuning_hof)):
                tuning_hof[i] = merge_mo_optimizer_results(results=[tuning_hof[i], precog_results[i]])

        printer.title_print("Assigning weights to the solutions in the tuning front results.")
        all_derivatives_all_folds = []
        for i in range(len(tuning_hof)):
            all_derivatives_all_folds.append(transpose(
                self.__solution_weights_strategy.assign_weights(hyperboxes=tuning_hof[i].fitness_hyperboxes())))
        # all_derivatives_all_folds is a sequence of folds of sequences of objectives of sequences of solutions.

        printer.title_print("Learning fitness adjusters using tuning front results, and creating adjusted objectives.")
        adjusters = []
        adjusted_objectives = []
        for obj_i in range(len(self.__objectives)):
            obj = self.__objectives[obj_i]
            weights_by_fold = [f[obj_i] for f in all_derivatives_all_folds]
            printer.print("Learning adjustment for " + obj.name())
            adjuster = self._train_adjuster_one_objective(
                tuning_hof=tuning_hof,
                objective_index=obj_i,
                input_data=input_data,
                folds_list=folds_list,
                solution_weights=weights_by_fold,
                printer=printer)
            adjusters.append(adjuster)
            if obj.is_class_based():
                adjusted_objective_computer = ClassificationAdjustedObjectiveComputer(
                    inner=obj.objective_computer(),
                    adjuster=adjuster)
            elif obj.is_survival():
                adjusted_objective_computer = SurvivalAdjustedObjectiveComputer(
                    inner=obj.objective_computer(),
                    adjuster=adjuster)
            elif obj.is_structural():
                adjusted_objective_computer = StructuralAdjustedObjectiveComputer(
                    inner=obj.objective_computer(),
                    adjuster=adjuster)
            else:
                raise PlannedUnreachableCodeError()
            adjusted_objective = obj.change_computer(objective_computer=adjusted_objective_computer)
            adjusted_objectives.append(adjusted_objective)

        printer.title_print("Running the main optimizer with the adjusted objectives.")
        main_optimizer = self.__main_optimizer_factory.create_optimizer(objectives=adjusted_objectives)
        random.setstate(rand_state)  # Setting the random state as it was at the beginning.
        return main_optimizer.optimize_with_feature_importance(
            input_data=input_data,
            printer=printer,
            feature_importance=feature_importance,
            n_proc=n_proc,
            workers_printer=workers_printer,
            logbook_saver=logbook_saver,
            feature_counts_saver=feature_counts_saver,
            known_solutions=known_solutions
        )

    def optimizer_type(self) -> MOOptimizerType:
        return self.__optimizer_type

    def name(self) -> str:
        return self.__optimizer_type.name()

    def nick(self) -> str:
        return self.__optimizer_type.nick()

    def __str__(self) -> str:
        res = "Adjusted optimizer" + "\n"
        res += name_value("Folds creator", self.__tuning_folds_creator) + "\n"
        res += "Optimizer for tuning:" + "\n"
        res += str(self.__tuning_optimizer) + "\n"
        res += name_value("Adjuster learner", self.__adjuster_learner) + "\n"
        res += "Main optimizer:" + "\n"
        res += str(self.__main_optimizer_factory) + "\n"
        return res
