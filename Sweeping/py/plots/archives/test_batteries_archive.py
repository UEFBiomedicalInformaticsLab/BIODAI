from collections.abc import Sequence

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from load_omics_views import CLINIC_NAME
from location_manager.location_manager_utils import COX_FI_STR
from plots.archives.optimizer_descriptor import OptimizerDescriptor
from plots.plot_labels import NSGA3_CHS_LAB

from util.math.list_math import powerset
from views.adjusted_view_definition import AdjustedViewDef


def all_view_combinations(included_views: Sequence[str]) -> Sequence[AdjustedViewDef]:
    """Sequence because order is important."""
    return [AdjustedViewDef.create_unadjusted(view_names=c)
            for c in powerset(iterable=included_views, include_empty=False)]


MV_BASELINE = OptimizerDescriptor(
        main_lab=NSGA3_CHS_LAB,
        inner_lab=None,
        population=500,
        generations=GenerationsStrategy(concatenated=300),
        view_set=AdjustedViewDef.create_unadjusted(view_names=[CLINIC_NAME]),
        survival_fi_nick=COX_FI_STR)

MV_GENERATIONS = [GenerationsStrategy(concatenated=300), GenerationsStrategy(sweeps=[150]),
                  GenerationsStrategy(sweeps=[50, 50, 50]), GenerationsStrategy(sweeps=[50, 50], concatenated=100)]
