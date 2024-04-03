from collections.abc import Iterable
from typing import Optional

from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from hall_of_fame.fronts import PARETO_NICK
from load_omics_views import MRNA_NAME
from model.survival_model import COX_NICK
from objective.social_objective import PersonalObjective
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel, BalAccLeanness
from plots.plot_labels import LR_LAB, NSGA_LAB, NSGA2_LAB, NSGA3_LAB, \
    NSGA2_CHS_LAB, NSGA2_CHP_LAB, NSGA2_CH_LAB, NSGA3_CHS_LAB, NSGA3_CHP_LAB, NSGA3_CH_LAB
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG
from setup.setup_utils import combine_objective_strings

OBJECTIVES_DIR_FROM_LABEL_DEFAULT = BalAccLeanness()
N_OUTER_FOLDS_DEFAULT = OUTER_N_FOLDS_BIG
CV_REPEATS_DEFAULT = 1
VIEWS_DEFAULT = {MRNA_NAME}
HOF_DEFAULT = PARETO_NICK
DEFAULT_INNER_N_FOLDS = 3


def objectives_dir_from_label(
        inner_lab: Optional[str], dir_from_label: ObjectivesDirFromLabel = BalAccLeanness()) -> str:
    """Returns the string composed by the objectives. Left to support legacy code."""
    return dir_from_label.objectives_dir_from_label(classification_inner_lab=inner_lab, survival_inner_lab=COX_NICK)


def default_pop_size(inner_lab: Optional[str]) -> int:
    if inner_lab == LR_LAB:
        return 200
    else:
        return 500


def default_generations(main_lab: str) -> Optional[GenerationsStrategy]:
    if NSGA_LAB in main_lab:
        if NSGA2_LAB in main_lab or NSGA3_LAB in main_lab:
            return GenerationsStrategy(concatenated=1000)
        else:
            raise ValueError("Unexpected main algorithm: " + str(main_lab))
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))


def optimizer_dir_from_labels(main_lab: str, inner_lab: Optional[str],
                              generations: Optional[GenerationsStrategy] = None,
                              pop_size: Optional[int] = None,
                              inner_n_folds: int = DEFAULT_INNER_N_FOLDS) -> str:
    if pop_size is None:
        pop_size = default_pop_size(inner_lab=inner_lab)
    if NSGA_LAB in main_lab:
        if NSGA2_LAB in main_lab:
            first_part = "NSGA2"
        elif NSGA3_LAB in main_lab:
            first_part = "NSGA3"
        else:
            raise ValueError("Unexpected main algorithm: " + str(main_lab))
        final_part = "_uninitialized_final_part"
        if NSGA2_CHS_LAB in main_lab:
            final_part = "_CrowdCICR_c0.33_m1.0symm"
        elif NSGA2_CHP_LAB in main_lab:
            final_part = "_CrowdCICR_c0.33_m1.0pers"
        elif NSGA2_CH_LAB in main_lab:
            final_part = "_CrowdCI_c0.33_m1.0flip"
        elif NSGA2_LAB in main_lab:
            final_part = "_CrowdFull_c0.33_m1.0flip"
        elif NSGA3_CHS_LAB in main_lab:
            final_part = "_NSGA3CICR_c0.33_m1.0symm"
        elif NSGA3_CHP_LAB in main_lab:
            final_part = "_NSGA3CICR_c0.33_m1.0pers"
        elif NSGA3_CH_LAB in main_lab:
            final_part = "_NSGA3CI_c0.33_m1.0flip"
        elif NSGA3_LAB in main_lab:
            final_part = "_NSGA3_c0.33_m1.0flip"
        if generations is None:
            generations = default_generations(main_lab=main_lab)
        gen_part = generations.nick()  # generations should not be None at this point since this is a GA
        return first_part + "_k" + str(inner_n_folds) + "_pop" + str(pop_size) + "_uni0-50_gen" + gen_part + final_part
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))


def optimizer_dir_from_labels_with_fi(main_lab: str, inner_lab: Optional[str], cox_fi: bool = True,
                                      generations: Optional[GenerationsStrategy] = None,
                                      inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
                                      pop_size: Optional[int] = None) -> str:
    first_part = optimizer_dir_from_labels(
        main_lab=main_lab, inner_lab=inner_lab, generations=generations, inner_n_folds=inner_n_folds, pop_size=pop_size)
    if cox_fi:
        survival_fi_str = "MV_CoxFI"
    else:
        survival_fi_str = "none"
    if (NSGA2_CHS_LAB in main_lab or NSGA2_CHP_LAB in main_lab or NSGA2_CH_LAB in main_lab or
       NSGA3_CHS_LAB in main_lab or NSGA3_CHP_LAB in main_lab or NSGA3_CH_LAB in main_lab):
        final_part = "_(MV_lassoFI," + survival_fi_str + ")"
    elif NSGA2_LAB in main_lab or NSGA3_LAB in main_lab:
        final_part = "_(none,none)"
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))
    return first_part + final_part


def optimizer_dir_from_labels_with_adjuster(main_lab: str, inner_lab: Optional[str], cox_fi: bool = True,
                                            generations: Optional[GenerationsStrategy] = None,
                                            population: Optional[int] = None,
                                            adjuster_regressor: Optional[str] = None,
                                            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
                                            inner_n_folds: int = DEFAULT_INNER_N_FOLDS) -> str:
    """Only adjusted optimizers with GA as tuning algorithm and no sweep generations are supported."""
    if population is None:
        population = default_pop_size(inner_lab=inner_lab)
    if generations is None:
        generations = default_generations(main_lab=main_lab)
    base_optimizer_str = optimizer_dir_from_labels_with_fi(main_lab=main_lab, inner_lab=inner_lab, cox_fi=cox_fi,
                                                           generations=generations, inner_n_folds=inner_n_folds,
                                                           pop_size=population)
    return base_optimizer_str


def hof_dir_from_label(main_lab: str, hof_nick: str = PARETO_NICK) -> str:
    """hof_nick is used if it is not possible to infer the hof from the optimizer alone."""
    return hof_nick


def objectives_string(objectives: [PersonalObjective], uses_inner_models: bool) -> str:
    if uses_inner_models:
        objectives_str_list = [o.nick() for o in objectives]
    else:
        objectives_str_list = [o.computer_nick() for o in objectives]
    return combine_objective_strings(objective_strings=objectives_str_list)


def views_nick(view_names: Iterable[str]) -> str:
    """Views are ordered alphabetically."""
    res = ""
    for n in sorted(view_names):
        if res != "":
            res += "_"
        res += str(n)
    return res


def views_name(view_names: Iterable[str]) -> str:
    """Views are ordered alphabetically."""
    res = ""
    for n in sorted(view_names):
        if res != "":
            res += " "
        res += str(n)
    return res


def save_path_folds_str(outer_n_folds: int, cv_repeats: int = 1) -> str:
    res = str(outer_n_folds) + "_folds"
    if cv_repeats > 1:
        res += "_x" + str(cv_repeats)
    return res
