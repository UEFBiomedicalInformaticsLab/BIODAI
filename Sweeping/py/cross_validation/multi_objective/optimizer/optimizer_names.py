from cross_validation.multi_objective.optimizer.mo_optimizer_including_feature_importance import \
    nick_from_optimizer_and_fi_nicks, name_from_optimizer_and_fi
from cross_validation.multi_objective.optimizer.multi_objective_optimizer_accepting_feature_importance import \
    MultiObjectiveOptimizerAcceptingFeatureImportance
from util.named import NickNamed


def nick_from_optimizer_and_fi_and_fs_nicks(
        optimizer: str,
        fi: str,
        fs: str) -> str:
    return nick_from_optimizer_and_fi_nicks(optimizer=optimizer, fi=fi) + "_" + fs


def nick_from_optimizer_and_fi_and_fs(
        optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
        fi: NickNamed,
        fs: NickNamed) -> str:
    return nick_from_optimizer_and_fi_and_fs_nicks(optimizer=optimizer.nick(), fi=fi.nick(), fs=fs.nick())


def name_from_optimizer_and_fi_and_fs(
        optimizer: MultiObjectiveOptimizerAcceptingFeatureImportance,
        fi: NickNamed,
        fs: NickNamed) -> str:
    return name_from_optimizer_and_fi(optimizer=optimizer, fi=fi) + " and " + fs.name()
