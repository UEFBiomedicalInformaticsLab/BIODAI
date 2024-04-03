from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance
from input_data.input_data import InputData
from util.distribution.distribution import Distribution, ConcreteDistribution


class MVFeatureImportanceByPrevious(MultiViewFeatureImportance):
    """The features are matched by name."""
    __counts: dict
    __nick: str

    def __init__(self, counts: dict, nick: str = "fi_prev"):
        self.__counts = counts
        self.__nick = nick

    def compute(self, input_data: InputData, n_proc: int = 1) -> list[Distribution]:
        res = []
        view_names = input_data.view_names()
        for i in range(input_data.n_views()):
            view_df = input_data.view(view_name=view_names[i])
            view_counts = self.__counts[i]
            res_i = [view_counts.get(c, 0) for c in view_df.columns]
            res.append(ConcreteDistribution(res_i))
        return res

    def is_none(self) -> bool:
        return False

    def nick(self) -> str:
        return self.__nick
