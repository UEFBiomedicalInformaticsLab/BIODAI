from cross_validation.multi_objective.optimizer.mo_optimizer_type import ConcreteMOOptimizerType

NSGA2_TYPE = ConcreteMOOptimizerType(
        uses_inner_models=True, nick="NSGA2", name="NSGA-II multi-view multi-objective optimizer")
