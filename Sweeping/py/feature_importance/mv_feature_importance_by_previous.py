from feature_importance.multi_view_feature_importance import MultiViewFeatureImportance
from input_data.input_data import InputData
from util.distribution.distribution import Distribution, ConcreteDistribution
from util.printer.printer import Printer, NULL_PRINTER


class MVFeatureImportanceByPrevious(MultiViewFeatureImportance):
    """The features are matched by name.
    This method does not need to do view adjustment.
    TODO counts is indexed by position at the moment. This might be error prone in presence of adjusting views."""
    __counts: dict
    __nick: str

    def __init__(self, counts: dict, nick: str = "fi_prev"):
        self.__counts = counts
        self.__nick = nick

    def compute(
            self, input_data: InputData, n_proc: int = 1, printer: Printer = NULL_PRINTER) -> dict[str,Distribution]:
        if input_data.needs_adjustment():
            raise ValueError(
                "Input data that needs adjustment is not supported at the moment, " +
                "because the views are indexed by position.")
        res = {}
        view_names = input_data.view_names_seq()
        for i in range(input_data.n_views()):
            name = view_names[i]
            view_df = input_data.view(view_name=name)
            view_counts = self.__counts[i]
            res_i = [view_counts.get(c, 0) for c in view_df.colnames()]
            res[name]=ConcreteDistribution(res_i)
        return res

    def is_none(self) -> bool:
        return False

    def nick(self) -> str:
        return self.__nick
