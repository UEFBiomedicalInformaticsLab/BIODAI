import os
from abc import abstractmethod

import pandas as pd
from pandas import DataFrame

from cross_validation.multi_objective.cross_evaluator.hof_saver import solution_features_file_name
from location_manager.location_consts import HOFS_STR
from cross_validation.multi_objective.optimizer.guided_forward import GuidedForward
from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance
from feature_importance.mv_feature_importance_by_previous import MVFeatureImportanceByPrevious
from hall_of_fame.fronts import PARETO_NICK
from input_data.view_prefix import remove_view_prefix
from util.named import NickNamed
from util.utils import name_value


class FeatureImportanceByFold(NickNamed):

    @abstractmethod
    def fi_for_fold(self, fold_index: int) -> MultiViewFeatureImportance:
        raise NotImplementedError()

    @abstractmethod
    def fi_for_all_data(self) -> MultiViewFeatureImportance:
        raise NotImplementedError()


class DummyFeatureImportanceByFold(FeatureImportanceByFold):
    __fi: MultiViewFeatureImportance

    def __init__(self, fi: MultiViewFeatureImportance):
        self.__fi = fi

    def fi_for_fold(self, fold_index: int) -> MultiViewFeatureImportance:
        return self.__fi

    def fi_for_all_data(self) -> MultiViewFeatureImportance:
        return self.__fi

    def nick(self) -> str:
        return self.__fi.nick()

    def name(self) -> str:
        return self.__fi.name()

    def __str__(self) -> str:
        return str(self.__fi)


class FeatureImportanceByPrevious(FeatureImportanceByFold):
    __previous_optimizer_dir: str
    __previous_hof_nick: str
    __nick: str
    __name: str

    def __init__(self,
                 previous_optimizer_dir: str,
                 previous_hof_nick: str = PARETO_NICK
                 ):
        if not isinstance(previous_optimizer_dir, str):
            raise ValueError("previous_optimizer_dir is not str: " + str(previous_optimizer_dir))
        if not isinstance(previous_hof_nick, str):
            raise ValueError("hof_nick is not str: " + str(previous_hof_nick))
        self.__previous_optimizer_dir = previous_optimizer_dir
        self.__previous_hof_nick = previous_hof_nick
        last_dir_name = os.path.basename(os.path.normpath(previous_optimizer_dir))
        self.__nick = "(" + last_dir_name + "_" + previous_hof_nick + ")"
        self.__name = "FI by previous" + " " + last_dir_name + " " + previous_hof_nick

    @staticmethod
    def feature_queue_nick(previous_optimizer_dir: str, hof_nick: str = PARETO_NICK):
        if not isinstance(previous_optimizer_dir, str):
            raise ValueError("previous_optimizer_dir is not str: " + str(previous_optimizer_dir))
        if not isinstance(hof_nick, str):
            raise ValueError("hof_nick is not str: " + str(hof_nick))
        last_dir_name = os.path.basename(os.path.normpath(previous_optimizer_dir))
        return last_dir_name + "_" + hof_nick

    @staticmethod
    def df_to_view_counts(df: DataFrame) -> dict:
        counts = df.sum()
        columns = df.columns
        res = {}
        for i in range(len(columns)):
            count = counts[i]
            col_name = columns[i]
            feature_name, view_num = remove_view_prefix(col_name)
            if view_num not in res:
                res[view_num] = {}
            res[view_num][feature_name] = count
        return res

    def fi_for_fold(self, fold_index: int) -> MultiViewFeatureImportance:
        previous_optimizer_dir = self.__previous_optimizer_dir
        hof_nick = self.__previous_hof_nick
        if os.path.isdir(previous_optimizer_dir):
            hofs_dir = os.path.join(previous_optimizer_dir, HOFS_STR, hof_nick)
            if os.path.isdir(hofs_dir):
                fold_hof_file_name = solution_features_file_name(fold_index=fold_index)
                hof_df = pd.read_csv(filepath_or_buffer=os.path.join(hofs_dir, fold_hof_file_name))
                view_counts = self.df_to_view_counts(df=hof_df)
                return MVFeatureImportanceByPrevious(
                    counts=view_counts,
                    nick=GuidedForward.feature_queue_nick(previous_optimizer_dir=previous_optimizer_dir,
                                                          hof_nick=hof_nick))
            else:
                raise ValueError("hof nicks are not directories: " + str(hof_nick))
        else:
            raise ValueError("previous_optimizer_dir is not a directory: " + str(previous_optimizer_dir))

    def fi_for_all_data(self) -> MultiViewFeatureImportance:
        raise NotImplementedError("Not yet")

    def nick(self) -> str:
        return self.__nick

    def name(self) -> str:
        return self.__name

    def __str__(self) -> str:
        res = "Feature importance by previous hall of fame\n"
        res += name_value("Directory of previous optimizer", self.__previous_optimizer_dir) + "\n"
        res += name_value("Previous hall of fame nick", self.__previous_hof_nick) + "\n"
        return res
