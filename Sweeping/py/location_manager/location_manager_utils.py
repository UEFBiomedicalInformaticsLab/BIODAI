from collections.abc import Iterable, Sequence
from typing import Optional, Union

from cross_validation.multi_objective.optimizer.adjusted_optimizer import BASE_NICK, ScaleParameters, tuning_parameters
from cross_validation.multi_objective.optimizer.generations_strategy import GenerationsStrategy
from cross_validation.multi_objective.optimizer.pso.cmdpsofs import CMDPSOFS_NICK
from feature_importance.feature_importance_by_cox import FeatureImportanceByCox
from feature_importance.feature_importance_by_lasso import FeatureImportanceByLasso
from feature_importance.feature_importance_by_univariate_cox import FeatureImportanceUnivariateCox
from feature_importance.feature_importance_by_univariate_nb import FeatureImportanceUnivariateNB
from feature_importance.multi_view_feature_importance import ConcreteMultiViewFeatureImportance
from hall_of_fame.hof_names import PARETO_NICK
from load_omics_views import MRNA_NAME
from model.survival.survival_model import COX_NICK
from objective.social_objective import PersonalObjective
from plots.archives.objectives_dir_from_label import ObjectivesDirFromLabel, BalAccLeanness
from plots.plot_labels import LR_LAB, NSGA_LAB, LCSW_PREF, CSW_PREF, SW_PREF, SWT_PREF, NSGA2_LAB, NSGA3_LAB, \
    NSGA2_CHS_LAB, NSGA2_CHP_LAB, NSGA2_CH_LAB, NSGA3_CHS_LAB, NSGA3_CHP_LAB, NSGA3_CH_LAB, GF_LAB, RFE_LAB, \
    LASSO_MO_LAB, CMDPSOFS_LAB
from setup.ga_mo_optimizer_setup import OUTER_N_FOLDS_BIG
from setup.setup_utils import combine_objective_strings
from views.adjusted_view_definition import AdjustedViewDef

OBJECTIVES_DIR_FROM_LABEL_DEFAULT = BalAccLeanness()
N_OUTER_FOLDS_DEFAULT = OUTER_N_FOLDS_BIG
CV_REPEATS_DEFAULT = 1
__VIEWS_DEFAULT_SET = {MRNA_NAME}
VIEWS_DEFAULT = AdjustedViewDef.create_unadjusted(view_names=__VIEWS_DEFAULT_SET)
HOF_DEFAULT = PARETO_NICK
DEFAULT_INNER_N_FOLDS = 3
COVARIATES_DEFAULT = {}
NONE_FI_STR = "none"
COX_FI_STR = ConcreteMultiViewFeatureImportance(FeatureImportanceByCox()).nick()
UNI_COX_FI_STR = ConcreteMultiViewFeatureImportance(FeatureImportanceUnivariateCox()).nick()
MV_LASSO_FI_STR = ConcreteMultiViewFeatureImportance(FeatureImportanceByLasso()).nick()
MV_UNI_NB_FI_STR = ConcreteMultiViewFeatureImportance(FeatureImportanceUnivariateNB()).nick()
DEFAULT_CATEGORICAL_FI_STR = MV_LASSO_FI_STR
DEFAULT_CATEGORICAL_FIS = (MV_LASSO_FI_STR,MV_UNI_NB_FI_STR)
DEFAULT_SURVIVAL_FI_STR = UNI_COX_FI_STR
DEFAULT_SURVIVAL_FIS = (NONE_FI_STR,COX_FI_STR,UNI_COX_FI_STR)


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
        if LCSW_PREF in main_lab or CSW_PREF in main_lab:
            return GenerationsStrategy(sweeps=[500])
        elif SWT_PREF in main_lab:
            return GenerationsStrategy(sweeps=[250, 150], concatenated=200)
        elif SW_PREF in main_lab:
            return GenerationsStrategy(sweeps=[250, 150, 100])
        elif NSGA2_LAB in main_lab or NSGA3_LAB in main_lab:
            return GenerationsStrategy(concatenated=1000)
        else:
            raise ValueError("Unexpected main algorithm: " + str(main_lab))
    elif main_lab == GF_LAB or main_lab == RFE_LAB or main_lab == LASSO_MO_LAB:
        return None
    elif main_lab == CMDPSOFS_LAB:
        return GenerationsStrategy(concatenated=500)
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))


def optimizer_dir_middle_part(
        main_lab: str,
        inner_lab: Optional[str],
        generations: Optional[GenerationsStrategy],
        pop_size: Optional[int],
        inner_n_folds: int) -> str:
    if pop_size is None:
        pop_size = default_pop_size(inner_lab=inner_lab)
    if generations is None:
        generations = default_generations(main_lab=main_lab)
    gen_part = generations.nick()
    return "_k" + str(inner_n_folds) + "_pop" + str(pop_size) + "_uni0-50_gen" + gen_part


def optimizer_dir_from_labels(main_lab: str, inner_lab: Optional[str],
                              generations: Optional[GenerationsStrategy] = None,
                              pop_size: Optional[int] = None,
                              inner_n_folds: int = DEFAULT_INNER_N_FOLDS) -> str:
    if NSGA_LAB in main_lab:
        if LCSW_PREF in main_lab:
            first_part = "LCSweeping"
        elif CSW_PREF in main_lab:
            first_part = "CSweeping"
        elif (SW_PREF in main_lab) or (SWT_PREF in main_lab):
            first_part = "Sweeping"
        elif NSGA2_LAB in main_lab:
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
        middle_part = optimizer_dir_middle_part(
            main_lab=main_lab,inner_lab=inner_lab,generations=generations,pop_size=pop_size,inner_n_folds=inner_n_folds)
        return first_part + middle_part + final_part
    elif main_lab == GF_LAB:
        return "guided_forward"
    elif main_lab == RFE_LAB:
        return "RFE"
    elif main_lab == LASSO_MO_LAB:
        return "LASSO_MO"
    elif main_lab == CMDPSOFS_LAB:
        middle_part = optimizer_dir_middle_part(
            main_lab=main_lab, inner_lab=inner_lab, generations=generations, pop_size=pop_size,
            inner_n_folds=inner_n_folds)
        return CMDPSOFS_NICK + middle_part + "_theta0.6_v_max0.6_w[0.1, 0.5]_c1[1.5, 2.0]_c2[1.5, 2.0])"
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))


def optimizer_dir_from_labels_with_fi(main_lab: str, inner_lab: Optional[str],
                                      categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
                                      survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
                                      generations: Optional[GenerationsStrategy] = None,
                                      inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
                                      pop_size: Optional[int] = None) -> str:
    first_part = optimizer_dir_from_labels(
        main_lab=main_lab, inner_lab=inner_lab, generations=generations, inner_n_folds=inner_n_folds, pop_size=pop_size)
    if (NSGA2_CHS_LAB in main_lab or NSGA2_CHP_LAB in main_lab or NSGA2_CH_LAB in main_lab or
       NSGA3_CHS_LAB in main_lab or NSGA3_CHP_LAB in main_lab or NSGA3_CH_LAB in main_lab or
       main_lab == GF_LAB or main_lab == RFE_LAB):
        final_part = "_(" + categorical_fi + "," + survival_fi + ")"
    elif NSGA2_LAB in main_lab or NSGA3_LAB in main_lab:
        final_part = "_(none,none)"
    elif main_lab == LASSO_MO_LAB:
        final_part = ""
    elif main_lab == CMDPSOFS_LAB:
        final_part = "_(none,none)"
    else:
        raise ValueError("Unexpected main algorithm: " + str(main_lab))
    return first_part + final_part


def optimizer_dir_from_labels_with_adjuster(main_lab: str, inner_lab: Optional[str],
                                            categorical_fi: str = DEFAULT_CATEGORICAL_FI_STR,
                                            survival_fi: str = DEFAULT_SURVIVAL_FI_STR,
                                            generations: Optional[GenerationsStrategy] = None,
                                            population: Optional[int] = None,
                                            adjuster_regressor: Optional[str] = None,
                                            n_outer_folds: int = N_OUTER_FOLDS_DEFAULT,
                                            inner_n_folds: int = DEFAULT_INNER_N_FOLDS
                                            ) -> str:
    """Only adjusted optimizers with GA as tuning algorithm and no sweep generations are supported.
    Does not include univariate feature selection part."""
    if population is None:
        population = default_pop_size(inner_lab=inner_lab)
    if generations is None:
        generations = default_generations(main_lab=main_lab)
    base_optimizer_str = optimizer_dir_from_labels_with_fi(main_lab=main_lab, inner_lab=inner_lab,
                                                           categorical_fi=categorical_fi,
                                                           survival_fi=survival_fi,
                                                           generations=generations, inner_n_folds=inner_n_folds,
                                                           pop_size=population)
    if adjuster_regressor is None or generations is None:  # Generations can be none if not GA.
        return base_optimizer_str
    else:
        main_parameters = ScaleParameters(
            pop_size=population, n_gen=generations.concatenated_generations(),
            n_folds=n_outer_folds, inner_n_folds=inner_n_folds)
        tuning_scale_parameters = tuning_parameters(main_parameters=main_parameters)
        adj_folds = tuning_scale_parameters.n_folds
        adj_pop = tuning_scale_parameters.pop_size
        adj_gen = GenerationsStrategy(concatenated=tuning_scale_parameters.n_gen)
        adj_optimizer_str = optimizer_dir_from_labels(
            main_lab=main_lab, inner_lab=inner_lab, generations=adj_gen, pop_size=adj_pop,
            inner_n_folds=tuning_scale_parameters.inner_n_folds)
        res = BASE_NICK + "_k" + str(adj_folds) + "_" + adj_optimizer_str
        res += "_" + BASE_NICK + "_" + adjuster_regressor + "_" + base_optimizer_str
        return res


def hof_dir_from_label(main_lab: str, hof_nick: str = PARETO_NICK) -> str:
    """hof_nick is used if it is not possible to infer the hof from the optimizer alone."""
    if main_lab == LASSO_MO_LAB:
        return "LASSO"
    else:
        return hof_nick


def objectives_string(objectives: Sequence[PersonalObjective], uses_inner_models: bool) -> str:
    if uses_inner_models:
        objectives_str_list = [o.nick() for o in objectives]
    else:
        objectives_str_list = [o.computer_nick() for o in objectives]
    return combine_objective_strings(objective_strings=objectives_str_list)


def views_nick(view_to_adjusters: Union[AdjustedViewDef,dict[str,set[str]],Iterable[str]]) -> str:
    """Views are ordered alphabetically.
    Supporting also legacy representations as dict or string Iterable.
    Views were represented as an Iterable of strings in previous versions of the program."""
    if isinstance(view_to_adjusters, AdjustedViewDef):
        adjusted_view_def = view_to_adjusters
    elif isinstance(view_to_adjusters, dict):
        adjusted_view_def = AdjustedViewDef(view_to_adjusters=view_to_adjusters)
    else:
        # Assuming it is an iterable.
        adjusted_view_def = AdjustedViewDef.create_unadjusted(view_names=view_to_adjusters)
    return adjusted_view_def.nick()


def covariates_dir(covariates_names: Iterable[str]) -> str:
    """Views are ordered alphabetically."""
    return "cov_" + views_nick(view_to_adjusters=covariates_names)

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
