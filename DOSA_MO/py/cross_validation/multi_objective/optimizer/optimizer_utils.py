from collections.abc import Sequence, Iterable

from cross_validation.multi_objective.optimizer.multi_objective_optimizer import hofs_to_results
from evaluator.individual_updater import IndividualUpdater
from evaluator.mask_evaluator import MaskEvaluator
from folds_creator.input_data_folds_creator import InputDataFoldsCreator
from hall_of_fame.population_observer_factory import HallOfFameFactory, ParetoFrontFactory
from hyperparam_manager.dummy_hp_manager import DummyHpManager
from individual.peculiar_individual import PeculiarIndividual
from individual.peculiar_individual_with_context import contextualize_all
from input_data.input_data import InputData
from objective.social_objective import PersonalObjective
from util.printer.printer import Printer, UnbufferedOutPrinter
from util.randoms import random_seed


def individuals_to_hofs(
        input_data: InputData,
        objectives: Sequence[PersonalObjective],
        folds_creator: InputDataFoldsCreator,
        pop: Iterable[PeculiarIndividual],
        hof_factories: Iterable[HallOfFameFactory] = (ParetoFrontFactory(),),
        n_workers: int = 1,
        workers_printer: Printer = UnbufferedOutPrinter()
        ):
    """Views are collapsed before use. Calls random_seed()."""
    inner_folds_list = folds_creator.create_folds_from_input_data(
        input_data=input_data, seed=random_seed())

    evaluator = MaskEvaluator(
        input_data=input_data,
        folds_list=inner_folds_list,
        objectives=objectives,
        n_workers=n_workers,
        workers_printer=workers_printer,
        seed=random_seed(),
        compute_confidence=True)
    individual_updater = IndividualUpdater(evaluator=evaluator, objectives=objectives)

    individual_updater.eval_invalid(pop=pop)
    pop = contextualize_all(hps=pop, hp_manager=DummyHpManager())
    hofs = [h.create_population_observer() for h in hof_factories]
    for h in hofs:
        h.update(new_elems=pop)
    return hofs_to_results(hofs)
